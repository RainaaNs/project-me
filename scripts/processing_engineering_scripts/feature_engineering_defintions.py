"""
EstablishedFeatureEngineer -- Non-Cold-Start Feature Engineering
================================================================
Encoding and scaling rules per dataset:

BANK
  - Drop:          Satisfaction Score
  - Keep as-is:    NumOfProducts (no clipping)
  - No derived:    has_zero_balance NOT created
  - No log1p:      all features encoded/scaled raw
  - Ordinal:       Card Type  (Silver=1, Gold=2, Platinum=3, Diamond=4)
  - Binary:        Gender (Male=0, Female=1)
  - One-hot:       Geography (all dummies, drop_first=False)
  - StandardScaler: CreditScore, Age, Tenure, Balance, EstimatedSalary, Point Earned
  - (Tenure is StandardScaler, NOT MinMaxScaler)

TELCO 1
  - Drop:          Satisfaction Score
  - No log1p:      all features scaled raw
  - Binary:        Gender, Married, Dependents, Referred a Friend, Phone Service,
                   Multiple Lines, Internet Service, Online Security, Online Backup,
                   Device Protection Plan, Premium Tech Support, Streaming TV,
                   Streaming Movies, Streaming Music, Unlimited Data, Paperless Billing
                   (flat binary -- Yes=1/No=0, no ternary expansion)
  - fillna:        Internet Type -> "No Internet", Offer -> "No Offer"
  - One-hot:       Offer, Internet Type, Contract, Payment Method (all dummies, drop_first=False)
  - StandardScaler: Age, Number of Dependents, Number of Referrals, Tenure in Months,
                    Avg Monthly Long Distance Charges, Avg Monthly GB Download,
                    Monthly Charge, Total Charges, Total Refunds,
                    Total Extra Data Charges, Total Long Distance Charges
  - (Tenure in Months is StandardScaler, NOT MinMaxScaler)

TELCO 2
  - No log1p:      TotalCharges scaled raw
  - Binary:        gender, Partner, Dependents, PaperlessBilling, PhoneService
                   (flat binary -- no ternary expansion)
  - One-hot:       InternetService, Contract, PaymentMethod, OnlineSecurity,
                   TechSupport, MultipleLines, OnlineBackup, DeviceProtection,
                   StreamingTV, StreamingMovies, SeniorCitizen
                   (all dummies, drop_first=False -- including SeniorCitizen as one-hot)
  - StandardScaler: tenure, MonthlyCharges, TotalCharges
  - (tenure is StandardScaler, NOT MinMaxScaler)
"""

import os
import numpy as np
import pandas as pd
import joblib
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────────────────────────────────────


def _safe_log1p(series: pd.Series) -> pd.Series:
    """log1p transform, safe against negatives (clips to 0 first)."""
    return np.log1p(series.clip(lower=0))


def _encode_binary(series: pd.Series) -> pd.Series:
    """
    Maps Yes/No, Male/Female, True/False strings and existing 0/1 ints
    to clean 0/1 integers. Unknown values become NaN (caught downstream).
    """
    mapping = {
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "male": 1,
        "female": 0,
        "1": 1,
        "0": 0,
        1: 1,
        0: 0,
    }
    return series.map(lambda x: mapping.get(str(x).lower().strip(), np.nan))


def _encode_contract(series: pd.Series) -> pd.Series:
    """Ordinal: Month-to-Month=0, One Year=1, Two Year=2."""
    mapping = {
        "month-to-month": 0,
        "one year": 1,
        "two year": 2,
        # Telco 2 variants
        "month to month": 0,
    }
    return series.map(lambda x: mapping.get(str(x).lower().strip(), 0))


def _encode_ternary_service(series: pd.Series) -> pd.DataFrame:
    """
    Converts Yes / No / No <Service> ternary columns to two binary columns:
      - has_<col>    : 1 if Yes
      - no_service_<col>: 1 if 'No <Service>' (i.e., can't get it, not just doesn't want it)
    Dropping the plain 'No' case as the reference category.
    """
    col_name = series.name if hasattr(series, "name") else "feature"
    has_col = f"has_{col_name}".replace(" ", "_").lower()
    no_svc_col = f"no_svc_{col_name}".replace(" ", "_").lower()

    has_vals = series.map(lambda x: 1 if str(x).lower().strip() == "yes" else 0)
    no_svc_vals = series.map(
        lambda x: (
            1 if ("no " in str(x).lower() and str(x).lower().strip() != "no") else 0
        )
    )
    return pd.DataFrame({has_col: has_vals, no_svc_col: no_svc_vals})


# ─────────────────────────────────────────────────────────────────────────────
# 1. COLD-START ENGINEER  (MPMN / Few-Shot Path)
# ─────────────────────────────────────────────────────────────────────────────


class ColdStartFeatureEngineer:
    """
    Phase 2a: Cold-Start Feature Engineering for the MPMN (Prototypical Network).

    Strategy:
      1. Explicit, rule-based encoding per feature type (not target encoding
         for low/medium cardinality features — see explanation below).
      2. log1p transform for right-skewed continuous features.
      3. StandardScaler for most continuous; MinMaxScaler for bounded ranges.
      4. Fit scaler on NON-COLD data (transfer learning) — transform cold users
         using those learned parameters.
      5. Correlation filter (>0.95) to remove redundant numeric features.

    Why NOT target encoding for binary/low-cardinality features:
      - MPMN constructs class prototypes by averaging embeddings. Target-encoded
        features are pre-told the answer, which collapses the metric space.
      - On small support sets (5–10 samples per class during episodic training),
        target encoding produces extremely noisy / overfit encodings.
      - Explicit 0/1 and one-hot encodings give the network a stable geometric
        structure to learn meaningful distances over.

    Target encoding is ONLY used for genuinely high-cardinality nominals (7+
    categories) where one-hot would explode the feature count.
    """

    def __init__(self, dataset_type: str = "telco"):
        self.dataset_type = dataset_type.lower()
        self.fitted = False

        # Scalers: one standard for most, one minmax for bounded features
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler(feature_range=(0, 1))
        self.standard_cols: list = []
        self.minmax_cols: list = []

        # Correlation filter
        self.drop_corr_cols: list = []

        # One-hot categories (fit-time learned categories)
        self.ohe_categories: dict = {}  # col → list of known categories

        # Target encoding fallback for high-cardinality only
        self.target_maps: dict = {}

        self.feature_names_out: list = []

        # ── FEATURE CONFIGS ───────────────────────────────────────────────────
        # Only features legitimately available at customer acquisition time.
        # Total Charges / accumulated billing features are excluded for cold path
        # because cold users have < 2 months of history (essentially 0).

        self.bank_config = {
            "standard": [
                "CreditScore",
                "Age",
                "Balance",
                "EstimatedSalary",
                "Point Earned",
            ],
            "minmax": ["Tenure"],
            "binary": ["HasCrCard", "IsActiveMember", "Gender"],
            "ordinal_bin": ["NumOfProducts"],  # bin 3+ → 3
            "ohe": ["Geography", "Card Type"],
            "engineered": ["has_zero_balance"],  # derived in _engineer()
            "target_enc": [],  # none needed for bank
        }

        self.telco1_config = {
            "standard": [
                "Age",
                "Number of Dependents",
                "Number of Referrals",  # log1p applied first
                "Avg Monthly Long Distance Charges",
                "Avg Monthly GB Download",  # log1p applied first
                "Monthly Charge",
            ],
            "minmax": ["Tenure in Months"],
            "binary": ["Gender", "Married", "Phone Service", "Paperless Billing"],
            "contract": ["Contract"],  # ordinal 0/1/2
            "ohe": ["Offer", "Internet Type", "Payment Method", "Internet Service"],
            "ternary": [  # → 2 binary cols each
                "Multiple Lines",
                "Online Security",
                "Online Backup",
                "Device Protection Plan",
                "Premium Tech Support",
                "Streaming TV",
                "Streaming Movies",
                "Streaming Music",
                "Unlimited Data",
            ],
            "target_enc": [],
        }

        self.telco2_config = {
            "standard": ["MonthlyCharges"],
            "minmax": ["tenure"],
            "binary": [
                "gender",
                "SeniorCitizen",
                "Partner",
                "Dependents",
                "PhoneService",
                "PaperlessBilling",
            ],
            "contract": ["Contract"],
            "ohe": ["InternetService", "PaymentMethod"],
            "ternary": [
                "MultipleLines",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
            ],
            "target_enc": [],
        }

    def _get_config(self) -> dict:
        if "bank" in self.dataset_type:
            return self.bank_config
        elif "telco_2" in self.dataset_type or "telco2" in self.dataset_type:
            return self.telco2_config
        else:
            return self.telco1_config

    # ── Engineering helpers ───────────────────────────────────────────────────

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features before encoding."""
        df = df.copy()

        # Bank: flag for structural zero balance (36% of customers)
        if "Balance" in df.columns:
            df["has_zero_balance"] = (df["Balance"] == 0).astype(int)

        # Bank: bin NumOfProducts 3 & 4 → 3 (marginal categories)
        if "NumOfProducts" in df.columns:
            df["NumOfProducts"] = df["NumOfProducts"].clip(upper=3)

        # Telco: log1p skewed features before scaling
        for col in [
            "Number of Referrals",
            "Avg Monthly GB Download",
            "Total Charges",
            "TotalCharges",
        ]:
            if col in df.columns:
                df[col] = _safe_log1p(df[col].fillna(0))

        return df

    def _apply_encoding(self, df: pd.DataFrame, config: dict, fit: bool) -> tuple:
        """
        Applies all encoding steps and returns a numpy array.
        If fit=True, learns categories/parameters. If False, transforms only.
        """
        parts = []
        col_names = []

        # 1. Standard-scaled continuous
        std_cols = [c for c in config.get("standard", []) if c in df.columns]
        if std_cols:
            X_std = df[std_cols].fillna(0).values.astype(float)
            if fit:
                self.standard_cols = std_cols
                X_std = self.standard_scaler.fit_transform(X_std)
            else:
                X_std = self.standard_scaler.transform(X_std)
            parts.append(X_std)
            col_names.extend(std_cols)

        # 2. MinMax-scaled bounded continuous
        mm_cols = [c for c in config.get("minmax", []) if c in df.columns]
        if mm_cols:
            X_mm = df[mm_cols].fillna(0).values.astype(float)
            if fit:
                self.minmax_cols = mm_cols
                X_mm = self.minmax_scaler.fit_transform(X_mm)
            else:
                X_mm = self.minmax_scaler.transform(X_mm)
            parts.append(X_mm)
            col_names.extend(mm_cols)

        # 3. Binary encoding (Yes/No strings → 0/1)
        for col in config.get("binary", []):
            if col in df.columns:
                encoded = _encode_binary(df[col]).fillna(0).values.reshape(-1, 1)
                parts.append(encoded)
                col_names.append(col)

        # 4. Contract ordinal encoding
        for col in config.get("contract", []):
            if col in df.columns:
                encoded = _encode_contract(df[col]).values.reshape(-1, 1)
                parts.append(encoded)
                col_names.append(col)

        # 5. One-Hot encoding for nominal categoricals
        for col in config.get("ohe", []):
            if col not in df.columns:
                continue
            # Apply dataset-specific fillna before OHE (e.g. Offer, Internet Type for telco1)
            if col == "Offer" and "telco1" in self.dataset_type:
                series = df[col].fillna("No Offer").astype(str)
            elif col == "Internet Type" and "telco1" in self.dataset_type:
                series = df[col].fillna("No Internet").astype(str)
            else:
                series = df[col].fillna("Unknown").astype(str)
            if fit:
                cats = sorted(series.unique().tolist())
                self.ohe_categories[col] = cats
            cats = self.ohe_categories.get(col, [])
            # Encode: one column per category except the first (reference)
            for cat in cats[1:]:
                parts.append((series == cat).astype(int).values.reshape(-1, 1))
                col_names.append(f"{col}_{cat}")

        # 6. Ternary service features → 2 binary columns each
        for col in config.get("ternary", []):
            if col not in df.columns:
                continue
            df_tern = _encode_ternary_service(df[col].fillna("No"))
            parts.append(df_tern.values)
            col_names.extend(df_tern.columns.tolist())

        # 7. Engineered features (already added by _engineer_features)
        for col in config.get("engineered", []):
            if col in df.columns:
                parts.append(df[col].fillna(0).values.reshape(-1, 1))
                col_names.append(col)

        # 8. Ordinal bin features
        for col in config.get("ordinal_bin", []):
            if col in df.columns:
                parts.append(df[col].fillna(0).values.reshape(-1, 1))
                col_names.append(col)

        X = np.hstack(parts) if parts else np.empty((len(df), 0))
        return X, col_names

    # ── Correlation filter ────────────────────────────────────────────────────

    def _fit_corr_filter(self, X: np.ndarray, col_names: list) -> list:
        """Identifies columns with pairwise correlation > 0.95."""
        df_X = pd.DataFrame(X, columns=col_names)
        corr = df_X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop = [c for c in upper.columns if any(upper[c] > 0.95)]
        return drop

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, df_non_cold: pd.DataFrame) -> "ColdStartFeatureEngineer":
        """
        Fit on non-cold (established) users.
        Learns scaler parameters, OHE categories, and correlation filter.
        """
        print(f"   Fitting Cold-Start Engineer ({self.dataset_type})...")
        config = self._get_config()

        df_eng = self._engineer_features(df_non_cold)
        X, col_names = self._apply_encoding(df_eng, config, fit=True)

        # Correlation filter
        self.drop_corr_cols = self._fit_corr_filter(X, col_names)
        if self.drop_corr_cols:
            print(f"      - Dropping highly correlated cols: {self.drop_corr_cols}")

        self.fitted = True
        print(f"      - Fit complete on {len(df_non_cold)} non-cold samples.")
        return self

    def transform(self, df: pd.DataFrame) -> tuple:
        """
        Transform a DataFrame (cold or non-cold) using fitted parameters.
        Returns (X, y, feature_names).
        """
        if not self.fitted:
            raise ValueError("Call fit() on non-cold data before transform().")

        config = self._get_config()
        df_eng = self._engineer_features(df)
        X, col_names = self._apply_encoding(df_eng, config, fit=False)

        # Apply correlation filter
        keep_idx = [i for i, c in enumerate(col_names) if c not in self.drop_corr_cols]
        X = X[:, keep_idx]
        col_names = [col_names[i] for i in keep_idx]

        # Extract target
        target_col = next(
            (t for t in ["Churn", "Exited", "Churn Label"] if t in df.columns), None
        )
        y = None
        if target_col:
            y = (
                df[target_col]
                .map({"Yes": 1, "No": 0, "yes": 1, "no": 0, 1: 1, 0: 0})
                .fillna(0)
                .values
            )

        self.feature_names_out = col_names
        _log_transform_warnings(df, col_names)

        return X, y, col_names


# ─────────────────────────────────────────────────────────────────────────────
# 2. ESTABLISHED FEATURE ENGINEER  (GATEFuse Path)
# ─────────────────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# Dataset-specific configuration
# ---------------------------------------------------------------------------

DATASET_CONFIG = {
    "bank": {
        "label_col": "Churn",
        "id_cols": ["RowNumber", "CustomerId", "Surname"],
        "drop_cols": ["Satisfaction Score", "is_cold_start"],
        "ordinal_cols": {
            "Card Type": {"SILVER": 1, "GOLD": 2, "PLATINUM": 3, "DIAMOND": 4}
        },
        "binary_cols": {"Gender": {"Male": 0, "Female": 1}},
        "fillna": {},
        "categorical_cols": ["Geography"],
        "dummy_regex": "^Geography_",
        "std_numerical_cols": [
            "CreditScore",
            "Age",
            "Tenure",
            "Balance",
            "EstimatedSalary",
            "Point Earned",
        ],
        "scalers_subdir": "bank_scalers",
    },
    "telco1": {
        "label_col": "Churn",
        "id_cols": [
            "Customer ID",
            "Under 30",
            "Senior Citizen",
            "Country",
            "State",
            "City",
            "Zip Code",
            "Latitude",
            "Longitude",
            "Population",
            "Quarter",
            "Total Revenue",
            "Customer Status",
            "Churn Score",
            "CLTV",
            "Churn Category",
            "Churn Reason",
        ],
        "drop_cols": ["Satisfaction Score", "is_cold_start"],
        "ordinal_cols": {},
        "binary_map": {"Yes": 1, "No": 0, "Male": 0, "Female": 1},
        "binary_cols": [
            "Gender",
            "Married",
            "Dependents",
            "Referred a Friend",
            "Phone Service",
            "Multiple Lines",
            "Internet Service",
            "Online Security",
            "Online Backup",
            "Device Protection Plan",
            "Premium Tech Support",
            "Streaming TV",
            "Streaming Movies",
            "Streaming Music",
            "Unlimited Data",
            "Paperless Billing",
        ],
        "fillna": {"Internet Type": "No Internet", "Offer": "No Offer"},
        "categorical_cols": ["Offer", "Internet Type", "Contract", "Payment Method"],
        "dummy_regex": "^(Offer_|Internet Type_|Contract_|Payment Method_)",
        "std_numerical_cols": [
            "Age",
            "Number of Dependents",
            "Number of Referrals",
            "Tenure in Months",
            "Avg Monthly Long Distance Charges",
            "Avg Monthly GB Download",
            "Monthly Charge",
            "Total Charges",
            "Total Refunds",
            "Total Extra Data Charges",
            "Total Long Distance Charges",
        ],
        "scalers_subdir": "telco1_scalers",
    },
    "telco2": {
        "label_col": "Churn",
        "id_cols": ["customerID"],
        "drop_cols": ["is_cold_start"],
        "ordinal_cols": {},
        "binary_map": {"Yes": 1, "No": 0, "Male": 0, "Female": 1},
        "binary_cols": [
            "gender",
            "Partner",
            "Dependents",
            "PaperlessBilling",
            "PhoneService",
        ],
        "fillna": {},
        "categorical_cols": [
            "SeniorCitizen",
            "InternetService",
            "Contract",
            "PaymentMethod",
            "OnlineSecurity",
            "TechSupport",
            "MultipleLines",
            "OnlineBackup",
            "DeviceProtection",
            "StreamingTV",
            "StreamingMovies",
        ],
        "dummy_regex": (
            "^(SeniorCitizen_|MultipleLines_|InternetService_|Contract_"
            "|PaymentMethod_|OnlineSecurity_|TechSupport_|OnlineBackup_"
            "|DeviceProtection_|StreamingTV_|StreamingMovies_)"
        ),
        "std_numerical_cols": ["tenure", "MonthlyCharges", "TotalCharges"],
        "scalers_subdir": "telco2_scalers",
    },
}


# ---------------------------------------------------------------------------
# Feature group configs (for GATE / attention model)
# ---------------------------------------------------------------------------

FEATURE_GROUPS = {
    "bank": {
        "Profile": [
            "CreditScore",
            "Age",
            "Geography_France",
            "Geography_Germany",
            "Geography_Spain",
            "Gender",
            "EstimatedSalary",
        ],
        "Contract": [
            "Tenure",
            "Card Type",
        ],
        "Billing": [
            "Balance",
            "Point Earned",
        ],
        "Usage": [
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
        ],
    },
    "telco1": {
        "Profile": [
            "Gender",
            "Age",
            "Married",
            "Number of Dependents",
        ],
        "Contract": [
            "Tenure in Months",
            "Offer_No Offer",
            "Offer_Offer A",
            "Offer_Offer B",
            "Offer_Offer C",
            "Offer_Offer D",
            "Offer_Offer E",
            "Contract_Month-to-Month",
            "Contract_One Year",
            "Contract_Two Year",
        ],
        "Billing": [
            "Monthly Charge",
            "Total Charges",
            "Total Refunds",
            "Total Extra Data Charges",
            "Total Long Distance Charges",
            "Paperless Billing",
            "Payment Method_Bank transfer (automatic)",
            "Payment Method_Credit card (automatic)",
            "Payment Method_Electronic check",
            "Payment Method_Mailed check",
        ],
        "Usage": [
            "Phone Service",
            "Multiple Lines",
            "Internet Service",
            "Internet Type_Cable",
            "Internet Type_DSL",
            "Internet Type_Fiber Optic",
            "Internet Type_No Internet Service",
            "Avg Monthly GB Download",
            "Online Security",
            "Online Backup",
            "Device Protection Plan",
            "Premium Tech Support",
            "Streaming TV",
            "Streaming Movies",
            "Streaming Music",
            "Unlimited Data",
            "Referred a Friend",
            "Dependents",
            "Number of Referrals",
            "Avg Monthly Long Distance Charges",
        ],
    },
    "telco2": {
        "Profile": [
            "gender",
            "SeniorCitizen_0",
            "SeniorCitizen_1",
            "Partner",
            "Dependents",
        ],
        "Contract": [
            "tenure",
            "Contract_Month-to-month",
            "Contract_One year",
            "Contract_Two year",
            "InternetService_DSL",
            "InternetService_Fiber optic",
            "InternetService_No",
        ],
        "Billing": [
            "MonthlyCharges",
            "TotalCharges",
            "PaperlessBilling",
            "PaymentMethod_Bank transfer (automatic)",
            "PaymentMethod_Credit card (automatic)",
            "PaymentMethod_Electronic check",
            "PaymentMethod_Mailed check",
        ],
        "Usage": [
            "PhoneService",
            "MultipleLines_No",
            "MultipleLines_No phone service",
            "MultipleLines_Yes",
            "OnlineSecurity_No",
            "OnlineSecurity_No internet service",
            "OnlineSecurity_Yes",
            "OnlineBackup_No",
            "OnlineBackup_No internet service",
            "OnlineBackup_Yes",
            "DeviceProtection_No",
            "DeviceProtection_No internet service",
            "DeviceProtection_Yes",
            "TechSupport_No",
            "TechSupport_No internet service",
            "TechSupport_Yes",
            "StreamingTV_No",
            "StreamingTV_No internet service",
            "StreamingTV_Yes",
            "StreamingMovies_No",
            "StreamingMovies_No internet service",
            "StreamingMovies_Yes",
        ],
    },
}


class EstablishedFeatureEngineer:
    """
    Feature engineering for established (non-cold-start) users.

    Parameters
    ----------
    dataset_type : str
        One of "bank", "telco1", "telco2".
    scalers_dir : str
        Root directory under which scaler .pkl files are saved/loaded.
        Artefacts land in  <scalers_dir>/<dataset>_scalers/.
    """

    def __init__(self, dataset_type: str, scalers_dir: str):
        self.dataset_type = dataset_type.lower()
        self.scalers_dir = scalers_dir
        self.fitted = False

        self._cfg = DATASET_CONFIG[self.dataset_type]
        self._groups_cfg = FEATURE_GROUPS[self.dataset_type]

        self.std_scaler = StandardScaler()
        self.ohe_columns: list = []  # column list after get_dummies, saved on fit

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scalers_subdir(self) -> str:
        return os.path.join(self.scalers_dir, self._cfg["scalers_subdir"])

    def _save(self, obj, filename: str):
        path = os.path.join(self._scalers_subdir(), filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(obj, path)

    def _load(self, filename: str):
        return joblib.load(os.path.join(self._scalers_subdir(), filename))

    def _extract_Xy(self, df: pd.DataFrame):
        """Separate features from label and id/dropped columns."""
        cfg = self._cfg
        label_col = cfg["label_col"]
        cols_to_drop = cfg["id_cols"] + cfg.get("drop_cols", []) + [label_col]
        cols_to_drop = [c for c in cols_to_drop if c in df.columns]

        y = df[label_col].map({"Yes": 1, "No": 0, 1: 1, 0: 0}).values
        X = df.drop(columns=cols_to_drop).copy()
        return X, y

    def _encode(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Apply all encoding steps in order."""
        cfg = self._cfg

        # 1. fillna
        for col, val in cfg.get("fillna", {}).items():
            if col in X.columns:
                X[col] = X[col].fillna(val)

        # 2. ordinal encoding (bank: Card Type)
        for col, mapping in cfg.get("ordinal_cols", {}).items():
            if col in X.columns:
                X[col] = X[col].map(mapping)
                if fit:
                    self._save(mapping, f"{col.replace(' ', '')}_mapping.pkl")

        # 3. binary encoding
        if self.dataset_type == "bank":
            # per-column maps
            for col, mapping in cfg["binary_cols"].items():
                if col in X.columns:
                    X[col] = X[col].map(mapping)
                    if fit:
                        self._save(mapping, f"{col.replace(' ', '')}_mapping.pkl")
        else:
            binary_map = cfg["binary_map"]
            for col in cfg["binary_cols"]:
                if col in X.columns:
                    X[col] = X[col].map(binary_map)
            if fit:
                self._save(binary_map, "binary_map.pkl")

        # 4. one-hot encoding
        categorical_cols = [c for c in cfg["categorical_cols"] if c in X.columns]
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

        dummy_cols = X.filter(regex=cfg["dummy_regex"]).columns
        X[dummy_cols] = X[dummy_cols].astype(int)

        if fit:
            self.ohe_columns = X.columns.tolist()
            self._save(self.ohe_columns, "ohe_columns.pkl")
        else:
            # Align to training columns
            ohe_columns = self._load("ohe_columns.pkl")
            for col in ohe_columns:
                if col not in X.columns:
                    X[col] = 0
            X = X[ohe_columns]

        return X

    def _scale(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Apply StandardScaler to numerical columns."""
        std_cols = [c for c in self._cfg["std_numerical_cols"] if c in X.columns]

        if fit:
            X[std_cols] = self.std_scaler.fit_transform(X[std_cols])
            self._save(self.std_scaler, "std_scaler.pkl")
        else:
            std_scaler = self._load("std_scaler.pkl")
            X[std_cols] = std_scaler.transform(X[std_cols])

        return X

    def _build_feature_groups(self, col_names: list) -> dict:
        """Map group names to the indices of their features in col_names."""
        feature_groups = {}
        for group_name, group_cols in self._groups_cfg.items():
            indices = [col_names.index(c) for c in group_cols if c in col_names]
            feature_groups[group_name] = indices
        return feature_groups

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> tuple:
        """
        Fit on and transform the training DataFrame.

        Returns
        -------
        X : np.ndarray,  shape (n_samples, n_features)
        y : np.ndarray,  shape (n_samples,)
        feature_names : list[str]
        feature_groups : dict[str, list[int]]
        """
        X, y = self._extract_Xy(df)
        X = self._encode(X, fit=True)
        X = self._scale(X, fit=True)

        feature_names = X.columns.tolist()
        feature_groups = self._build_feature_groups(feature_names)

        self.fitted = True
        print(f"[{self.dataset_type}] fit_transform complete.")
        print(f"   Shape: {X.shape}")
        print(f"   Groups: { {k: len(v) for k, v in feature_groups.items()} }")

        return X.values, y, feature_names, feature_groups

    def transform(self, df: pd.DataFrame) -> tuple:
        """
        Transform val/test data using fitted artefacts (no fitting).

        Returns
        -------
        X : np.ndarray,  shape (n_samples, n_features)
        y : np.ndarray,  shape (n_samples,)
        feature_names : list[str]
        feature_groups : dict[str, list[int]]
        """
        if not self.fitted:
            raise ValueError(
                "Call fit_transform() before transform(), "
                "or load a fitted instance from disk."
            )
        X, y = self._extract_Xy(df)
        X = self._encode(X, fit=False)
        X = self._scale(X, fit=False)

        feature_names = X.columns.tolist()
        feature_groups = self._build_feature_groups(feature_names)

        return X.values, y, feature_names, feature_groups


# class EstablishedFeatureEngineer:
#     """
#     Feature engineering for established (non-cold-start) users.

#     Parameters
#     ----------
#     dataset_type : str
#         One of "bank", "telco1", "telco2".
#     scalers_dir : str
#         Root directory under which scaler .pkl files are saved/loaded.
#         Artefacts land in  <scalers_dir>/<dataset>_scalers/.
#     """

#     def __init__(self, dataset_type: str, scalers_dir: str):
#         self.dataset_type = dataset_type.lower()
#         self.scalers_dir = scalers_dir
#         self.fitted = False

#         self._cfg = DATASET_CONFIG[self.dataset_type]
#         self._groups_cfg = FEATURE_GROUPS[self.dataset_type]

#         self.std_scaler = StandardScaler()
#         self.ohe_columns: list = []

#     # ------------------------------------------------------------------
#     # Internal helpers
#     # ------------------------------------------------------------------

#     def _scalers_subdir(self) -> str:
#         return os.path.join(self.scalers_dir, self._cfg["scalers_subdir"])

#     def _save(self, obj, filename: str):
#         path = os.path.join(self._scalers_subdir(), filename)
#         os.makedirs(os.path.dirname(path), exist_ok=True)
#         joblib.dump(obj, path)

#     def _load(self, filename: str):
#         return joblib.load(os.path.join(self._scalers_subdir(), filename))

#     def _extract_Xy(self, df: pd.DataFrame):
#         """Separate features from label and id/dropped columns."""
#         cfg = self._cfg
#         label_col = cfg["label_col"]
#         cols_to_drop = cfg["id_cols"] + cfg.get("drop_cols", []) + [label_col]
#         cols_to_drop = [c for c in cols_to_drop if c in df.columns]

#         y = df[label_col].map({"Yes": 1, "No": 0, 1: 1, 0: 0}).values
#         X = df.drop(columns=cols_to_drop).copy()
#         return X, y

#     def _encode(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
#         """Apply all encoding steps in order."""
#         cfg = self._cfg

#         # 1. fillna
#         for col, val in cfg.get("fillna", {}).items():
#             if col in X.columns:
#                 X[col] = X[col].fillna(val)

#         # 2. ordinal encoding (bank: Card Type)
#         for col, mapping in cfg.get("ordinal_cols", {}).items():
#             if col in X.columns:
#                 X[col] = X[col].map(mapping)
#                 if fit:
#                     self._save(mapping, f"{col.replace(' ', '')}_mapping.pkl")

#         # 3. binary encoding
#         if self.dataset_type == "bank":
#             for col, mapping in cfg["binary_cols"].items():
#                 if col in X.columns:
#                     X[col] = X[col].map(mapping)
#                     if fit:
#                         self._save(mapping, f"{col.replace(' ', '')}_mapping.pkl")
#         else:
#             binary_map = cfg["binary_map"]
#             for col in cfg["binary_cols"]:
#                 if col in X.columns:
#                     X[col] = X[col].map(binary_map)
#             if fit:
#                 self._save(binary_map, "binary_map.pkl")

#         # 4. one-hot encoding
#         categorical_cols = [c for c in cfg["categorical_cols"] if c in X.columns]
#         X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

#         dummy_cols = X.filter(regex=cfg["dummy_regex"]).columns
#         X[dummy_cols] = X[dummy_cols].astype(int)

#         if fit:
#             self.ohe_columns = X.columns.tolist()
#             self._save(self.ohe_columns, "ohe_columns.pkl")
#         else:
#             ohe_columns = self._load("ohe_columns.pkl")
#             for col in ohe_columns:
#                 if col not in X.columns:
#                     X[col] = 0
#             X = X[ohe_columns]

#         return X

#     def _scale(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
#         """Apply StandardScaler to numerical columns."""
#         std_cols = [c for c in self._cfg["std_numerical_cols"] if c in X.columns]

#         if fit:
#             X[std_cols] = self.std_scaler.fit_transform(X[std_cols])
#             self._save(self.std_scaler, "std_scaler.pkl")
#         else:
#             std_scaler = self._load("std_scaler.pkl")
#             X[std_cols] = std_scaler.transform(X[std_cols])

#         return X

#     def _build_feature_groups(self, col_names: list) -> dict:
#         """Map group names to the indices of their features in col_names."""
#         feature_groups = {}
#         for group_name, group_cols in self._groups_cfg.items():
#             indices = [col_names.index(c) for c in group_cols if c in col_names]
#             feature_groups[group_name] = indices
#         return feature_groups

#     # ------------------------------------------------------------------
#     # Public API
#     # ------------------------------------------------------------------

#     def fit_transform(self, df: pd.DataFrame) -> tuple:
#         """
#         Fit on and transform the training DataFrame.

#         Returns
#         -------
#         X : np.ndarray,  shape (n_samples, n_features)
#         y : np.ndarray,  shape (n_samples,)
#         feature_names : list[str]
#         feature_groups : dict[str, list[int]]
#         """
#         X, y = self._extract_Xy(df)
#         X = self._encode(X, fit=True)
#         X = self._scale(X, fit=True)

#         feature_names = X.columns.tolist()
#         feature_groups = self._build_feature_groups(feature_names)

#         self.fitted = True
#         print(f"[{self.dataset_type}] fit_transform complete.")
#         print(f"   Shape: {X.shape}")
#         print(f"   Groups: { {k: len(v) for k, v in feature_groups.items()} }")

#         return X.values, y, feature_names, feature_groups

#     def transform(self, df: pd.DataFrame) -> tuple:
#         """
#         Transform val/test data using fitted artefacts (no fitting).

#         Returns
#         -------
#         X : np.ndarray,  shape (n_samples, n_features)
#         y : np.ndarray,  shape (n_samples,)
#         feature_names : list[str]
#         feature_groups : dict[str, list[int]]
#         """
#         if not self.fitted:
#             raise ValueError(
#                 "Call fit_transform() before transform(), "
#                 "or load a fitted instance from disk."
#             )
#         X, y = self._extract_Xy(df)
#         X = self._encode(X, fit=False)
#         X = self._scale(X, fit=False)

#         feature_names = X.columns.tolist()
#         feature_groups = self._build_feature_groups(feature_names)

#         return X.values, y, feature_names, feature_groups


# # ─────────────────────────────────────────────────────────────────────────────
# # UTILITIES
# # ─────────────────────────────────────────────────────────────────────────────


def _log_transform_warnings(df: pd.DataFrame, col_names: list) -> None:
    """
    Logs which categorical columns triggered the OHE unknown-category fallback
    during transform. Helps detect distribution shift between non-cold fit data
    and cold-start transform data.
    """
    pass


# # import pandas as pd
# # import numpy as np
# # from sklearn.preprocessing import StandardScaler, MinMaxScaler
# # import warnings

# # warnings.filterwarnings("ignore")


# # # ─────────────────────────────────────────────────────────────────────────────
# # # SHARED UTILITIES
# # # ─────────────────────────────────────────────────────────────────────────────


# def _safe_log1p(series: pd.Series) -> pd.Series:
#     """log1p transform, safe against negatives (clips to 0 first)."""
#     return np.log1p(series.clip(lower=0))


# def _encode_binary(series: pd.Series) -> pd.Series:
#     """
#     Maps Yes/No, Male/Female, True/False strings and existing 0/1 ints
#     to clean 0/1 integers. Unknown values become NaN (caught downstream).
#     """
#     mapping = {
#         "yes": 1,
#         "no": 0,
#         "true": 1,
#         "false": 0,
#         "male": 1,
#         "female": 0,
#         "1": 1,
#         "0": 0,
#         1: 1,
#         0: 0,
#     }
#     return series.map(lambda x: mapping.get(str(x).lower().strip(), np.nan))


# def _encode_contract(series: pd.Series) -> pd.Series:
#     """Ordinal: Month-to-Month=0, One Year=1, Two Year=2."""
#     mapping = {
#         "month-to-month": 0,
#         "one year": 1,
#         "two year": 2,
#         # Telco 2 variants
#         "month to month": 0,
#     }
#     return series.map(lambda x: mapping.get(str(x).lower().strip(), 0))


# def _encode_ternary_service(series: pd.Series) -> pd.DataFrame:
#     """
#     Converts Yes / No / No <Service> ternary columns to two binary columns:
#       - has_<col>    : 1 if Yes
#       - no_service_<col>: 1 if 'No <Service>' (i.e., can't get it, not just doesn't want it)
#     Dropping the plain 'No' case as the reference category.
#     """
#     col_name = series.name if hasattr(series, "name") else "feature"
#     has_col = f"has_{col_name}".replace(" ", "_").lower()
#     no_svc_col = f"no_svc_{col_name}".replace(" ", "_").lower()

#     has_vals = series.map(lambda x: 1 if str(x).lower().strip() == "yes" else 0)
#     no_svc_vals = series.map(
#         lambda x: (
#             1 if ("no " in str(x).lower() and str(x).lower().strip() != "no") else 0
#         )
#     )
#     return pd.DataFrame({has_col: has_vals, no_svc_col: no_svc_vals})


# # ─────────────────────────────────────────────────────────────────────────────
# # 1. COLD-START ENGINEER  (MPMN / Few-Shot Path)
# # ─────────────────────────────────────────────────────────────────────────────


# class ColdStartFeatureEngineer:
#     """
#     Phase 2a: Cold-Start Feature Engineering for the MPMN (Prototypical Network).

#     Strategy:
#       1. Explicit, rule-based encoding per feature type (not target encoding
#          for low/medium cardinality features — see explanation below).
#       2. log1p transform for right-skewed continuous features.
#       3. StandardScaler for most continuous; MinMaxScaler for bounded ranges.
#       4. Fit scaler on NON-COLD data (transfer learning) — transform cold users
#          using those learned parameters.
#       5. Correlation filter (>0.95) to remove redundant numeric features.

#     Why NOT target encoding for binary/low-cardinality features:
#       - MPMN constructs class prototypes by averaging embeddings. Target-encoded
#         features are pre-told the answer, which collapses the metric space.
#       - On small support sets (5–10 samples per class during episodic training),
#         target encoding produces extremely noisy / overfit encodings.
#       - Explicit 0/1 and one-hot encodings give the network a stable geometric
#         structure to learn meaningful distances over.

#     Target encoding is ONLY used for genuinely high-cardinality nominals (7+
#     categories) where one-hot would explode the feature count.
#     """

#     def __init__(self, dataset_type: str = "telco"):
#         self.dataset_type = dataset_type.lower()
#         self.fitted = False

#         # Scalers: one standard for most, one minmax for bounded features
#         self.standard_scaler = StandardScaler()
#         self.minmax_scaler = MinMaxScaler(feature_range=(0, 1))
#         self.standard_cols: list = []
#         self.minmax_cols: list = []

#         # Correlation filter
#         self.drop_corr_cols: list = []

#         # One-hot categories (fit-time learned categories)
#         self.ohe_categories: dict = {}  # col → list of known categories

#         # Target encoding fallback for high-cardinality only
#         self.target_maps: dict = {}

#         self.feature_names_out: list = []

#         # ── FEATURE CONFIGS ───────────────────────────────────────────────────
#         # Only features legitimately available at customer acquisition time.
#         # Total Charges / accumulated billing features are excluded for cold path
#         # because cold users have < 2 months of history (essentially 0).

#         self.bank_config = {
#             "standard": [
#                 "CreditScore",
#                 "Age",
#                 "Balance",
#                 "EstimatedSalary",
#                 "Point Earned",
#             ],
#             "minmax": ["Tenure"],
#             "binary": ["HasCrCard", "IsActiveMember", "Gender"],
#             "ordinal_bin": ["NumOfProducts"],  # bin 3+ → 3
#             "ohe": ["Geography", "Card Type"],
#             "engineered": ["has_zero_balance"],  # derived in _engineer()
#             "target_enc": [],  # none needed for bank
#         }

#         self.telco1_config = {
#             "standard": [
#                 "Age",
#                 "Number of Dependents",
#                 "Number of Referrals",  # log1p applied first
#                 "Avg Monthly Long Distance Charges",
#                 "Avg Monthly GB Download",  # log1p applied first
#                 "Monthly Charge",
#             ],
#             "minmax": ["Tenure in Months"],
#             "binary": ["Gender", "Married", "Phone Service", "Paperless Billing"],
#             "contract": ["Contract"],  # ordinal 0/1/2
#             "ohe": ["Offer", "Internet Type", "Payment Method", "Internet Service"],
#             "ternary": [  # → 2 binary cols each
#                 "Multiple Lines",
#                 "Online Security",
#                 "Online Backup",
#                 "Device Protection Plan",
#                 "Premium Tech Support",
#                 "Streaming TV",
#                 "Streaming Movies",
#                 "Streaming Music",
#                 "Unlimited Data",
#             ],
#             "target_enc": [],
#         }

#         self.telco2_config = {
#             "standard": ["MonthlyCharges"],
#             "minmax": ["tenure"],
#             "binary": [
#                 "gender",
#                 "SeniorCitizen",
#                 "Partner",
#                 "Dependents",
#                 "PhoneService",
#                 "PaperlessBilling",
#             ],
#             "contract": ["Contract"],
#             "ohe": ["InternetService", "PaymentMethod"],
#             "ternary": [
#                 "MultipleLines",
#                 "OnlineSecurity",
#                 "OnlineBackup",
#                 "DeviceProtection",
#                 "TechSupport",
#                 "StreamingTV",
#                 "StreamingMovies",
#             ],
#             "target_enc": [],
#         }

#     def _get_config(self) -> dict:
#         if "bank" in self.dataset_type:
#             return self.bank_config
#         elif "telco_2" in self.dataset_type or "telco2" in self.dataset_type:
#             return self.telco2_config
#         else:
#             return self.telco1_config

#     # ── Engineering helpers ───────────────────────────────────────────────────

#     def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Add derived features before encoding."""
#         df = df.copy()

#         # Bank: flag for structural zero balance (36% of customers)
#         if "Balance" in df.columns:
#             df["has_zero_balance"] = (df["Balance"] == 0).astype(int)

#         # Bank: bin NumOfProducts 3 & 4 → 3 (marginal categories)
#         if "NumOfProducts" in df.columns:
#             df["NumOfProducts"] = df["NumOfProducts"].clip(upper=3)

#         # Telco: log1p skewed features before scaling
#         for col in [
#             "Number of Referrals",
#             "Avg Monthly GB Download",
#             "Total Charges",
#             "TotalCharges",
#         ]:
#             if col in df.columns:
#                 df[col] = _safe_log1p(df[col].fillna(0))

#         return df

#     def _apply_encoding(self, df: pd.DataFrame, config: dict, fit: bool) -> tuple:
#         """
#         Applies all encoding steps and returns a numpy array.
#         If fit=True, learns categories/parameters. If False, transforms only.
#         """
#         parts = []
#         col_names = []

#         # 1. Standard-scaled continuous
#         std_cols = [c for c in config.get("standard", []) if c in df.columns]
#         if std_cols:
#             X_std = df[std_cols].fillna(0).values.astype(float)
#             if fit:
#                 self.standard_cols = std_cols
#                 X_std = self.standard_scaler.fit_transform(X_std)
#             else:
#                 X_std = self.standard_scaler.transform(X_std)
#             parts.append(X_std)
#             col_names.extend(std_cols)

#         # 2. MinMax-scaled bounded continuous
#         mm_cols = [c for c in config.get("minmax", []) if c in df.columns]
#         if mm_cols:
#             X_mm = df[mm_cols].fillna(0).values.astype(float)
#             if fit:
#                 self.minmax_cols = mm_cols
#                 X_mm = self.minmax_scaler.fit_transform(X_mm)
#             else:
#                 X_mm = self.minmax_scaler.transform(X_mm)
#             parts.append(X_mm)
#             col_names.extend(mm_cols)

#         # 3. Binary encoding (Yes/No strings → 0/1)
#         for col in config.get("binary", []):
#             if col in df.columns:
#                 encoded = _encode_binary(df[col]).fillna(0).values.reshape(-1, 1)
#                 parts.append(encoded)
#                 col_names.append(col)

#         # 4. Contract ordinal encoding
#         for col in config.get("contract", []):
#             if col in df.columns:
#                 encoded = _encode_contract(df[col]).values.reshape(-1, 1)
#                 parts.append(encoded)
#                 col_names.append(col)

#         # 5. One-Hot encoding for nominal categoricals
#         # Geography has 3 categories: France, Germany, Spain
#         # Sorted alphabetically → France is reference (dropped)
#         # Output columns: Geography_Germany, Geography_Spain
#         for col in config.get("ohe", []):
#             if col not in df.columns:
#                 continue
#             series = df[col].fillna("Unknown").astype(str)
#             if fit:
#                 cats = sorted(series.unique().tolist())
#                 self.ohe_categories[col] = cats
#             cats = self.ohe_categories.get(col, [])
#             # Encode: one column per category except the first (reference)
#             for cat in cats[1:]:
#                 parts.append((series == cat).astype(int).values.reshape(-1, 1))
#                 col_names.append(f"{col}_{cat}")

#         # 6. Ternary service features → 2 binary columns each
#         for col in config.get("ternary", []):
#             if col not in df.columns:
#                 continue
#             df_tern = _encode_ternary_service(df[col].fillna("No"))
#             parts.append(df_tern.values)
#             col_names.extend(df_tern.columns.tolist())

#         # 7. Engineered features (already added by _engineer_features)
#         for col in config.get("engineered", []):
#             if col in df.columns:
#                 parts.append(df[col].fillna(0).values.reshape(-1, 1))
#                 col_names.append(col)

#         # 8. Ordinal bin features
#         for col in config.get("ordinal_bin", []):
#             if col in df.columns:
#                 parts.append(df[col].fillna(0).values.reshape(-1, 1))
#                 col_names.append(col)

#         X = np.hstack(parts) if parts else np.empty((len(df), 0))
#         return X, col_names

#     # ── Correlation filter ────────────────────────────────────────────────────

#     def _fit_corr_filter(self, X: np.ndarray, col_names: list) -> list:
#         """Identifies columns with pairwise correlation > 0.95."""
#         df_X = pd.DataFrame(X, columns=col_names)
#         corr = df_X.corr().abs()
#         upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
#         drop = [c for c in upper.columns if any(upper[c] > 0.95)]
#         return drop

#     # ── Public API ────────────────────────────────────────────────────────────

#     def fit(self, df_non_cold: pd.DataFrame) -> "ColdStartFeatureEngineer":
#         """
#         Fit on non-cold (established) users.
#         Learns scaler parameters, OHE categories, and correlation filter.
#         """
#         print(f"   ❄️  Fitting Cold-Start Engineer ({self.dataset_type})...")
#         config = self._get_config()

#         df_eng = self._engineer_features(df_non_cold)
#         X, col_names = self._apply_encoding(df_eng, config, fit=True)

#         # Correlation filter
#         self.drop_corr_cols = self._fit_corr_filter(X, col_names)
#         if self.drop_corr_cols:
#             print(f"      - Dropping highly correlated cols: {self.drop_corr_cols}")

#         self.fitted = True
#         print(f"      - Fit complete on {len(df_non_cold)} non-cold samples.")
#         return self

#     def transform(self, df: pd.DataFrame) -> tuple:
#         """
#         Transform a DataFrame (cold or non-cold) using fitted parameters.
#         Returns (X, y, feature_names).
#         """
#         if not self.fitted:
#             raise ValueError("Call fit() on non-cold data before transform().")

#         config = self._get_config()
#         df_eng = self._engineer_features(df)
#         X, col_names = self._apply_encoding(df_eng, config, fit=False)

#         # Apply correlation filter
#         keep_idx = [i for i, c in enumerate(col_names) if c not in self.drop_corr_cols]
#         X = X[:, keep_idx]
#         col_names = [col_names[i] for i in keep_idx]

#         # Extract target
#         target_col = next(
#             (t for t in ["Churn", "Exited", "Churn Label"] if t in df.columns), None
#         )
#         y = None
#         if target_col:
#             y = (
#                 df[target_col]
#                 .map({"Yes": 1, "No": 0, "yes": 1, "no": 0, 1: 1, 0: 0})
#                 .fillna(0)
#                 .values
#             )

#         self.feature_names_out = col_names
#         _log_transform_warnings(df, col_names)

#         return X, y, col_names


# # ─────────────────────────────────────────────────────────────────────────────
# # 2. ESTABLISHED FEATURE ENGINEER  (GATEFuse Path)
# # ─────────────────────────────────────────────────────────────────────────────
# #
# # Uses an explicit declarative config per dataset (binary list, ohe list,
# # standard/minmax lists, passthrough list) plus a group_layout dict that maps
# # each semantic group to an ordered list of the ACTUAL post-encoding column
# # names. The group_layout is the single source of truth for column order —
# # columns declared in the layout but absent from the data are zero-filled so
# # group indices stay perfectly aligned across train / val / test.
# #
# # Key design decisions:
# #   • Card Type    → ordinal (Silver=1, Gold=2, Platinum=3, Diamond=4),
# #                    single column, NOT one-hot.
# #   • Service cols → explicit binary (Yes=1, everything else=0).
# #                    Eligibility is captured by Internet/Phone Service cols.
# #   • Nominal OHE  → drop_first is controlled per-column via ohe_drop_first.
# #   • Continuous   → StandardScaler. No log1p (established users have full
# #                    history so distributions are stable).
# #   • Bounded      → MinMaxScaler (Satisfaction Score only).
# #   • Tenure       → StandardScaler for all three datasets.
# #   • Unmapped binary values raise a ValueError immediately rather than
# #     silently filling with 0 — prevents silent data quality issues.


# class EstablishedFeatureEngineer:
#     """Phase 2b: Established User Feature Engineering for GATEFuse."""

#     _BINARY_MAP = {
#         "Yes": 1,
#         "No": 0,
#         "No internet service": 0,
#         "No phone service": 0,
#         "Male": 0,
#         "Female": 1,
#     }

#     _CARD_TYPE_MAP = {
#         "SILVER": 1,
#         "GOLD": 2,
#         "PLATINUM": 3,
#         "DIAMOND": 4,
#     }

#     def __init__(self, dataset_type: str = "telco"):
#         self.dataset_type = dataset_type.lower()
#         self.fitted = False

#         self.standard_scaler = StandardScaler()
#         self.minmax_scaler = MinMaxScaler(feature_range=(0, 1))
#         self.standard_cols: list = []
#         self.minmax_cols: list = []
#         self.ohe_categories: dict = {}

#         self.feature_names_out: list = []
#         self.feature_groups: dict = {}

#         # ── PER-DATASET CONFIGS ───────────────────────────────────────────────
#         # group_layout column names must match what the encoders below produce.
#         # OHE columns appear as "<col>_<category>".
#         # Columns in the layout that are absent from a split are zero-filled
#         # so group indices stay aligned across train / val / test.

#         self.bank_config = {
#             "binary": ["Gender"],
#             "ordinal": ["Card Type"],
#             "ohe": ["Geography"],
#             "ohe_drop_first": {"Geography": False},  # keep all three dummies
#             "standard": [
#                 "CreditScore",
#                 "Age",
#                 "Tenure",
#                 "Balance",
#                 "EstimatedSalary",
#                 "Point Earned",
#             ],
#             "minmax": ["Satisfaction Score"],
#             "passthrough_numeric": [
#                 "NumOfProducts",
#                 "HasCrCard",
#                 "IsActiveMember",
#                 "Complain",
#             ],
#             "group_layout": {
#                 "Profile": [
#                     "Geography_Germany",
#                     "Geography_Spain",
#                     "Geography_France",
#                     "Gender",
#                     "Age",
#                     "Satisfaction Score",
#                 ],
#                 "Contract": ["Tenure", "Card Type"],
#                 "Billing": ["Balance", "EstimatedSalary", "CreditScore"],
#                 "Usage": [
#                     "NumOfProducts",
#                     "HasCrCard",
#                     "IsActiveMember",
#                     "Complain",
#                     "Point Earned",
#                 ],
#             },
#         }

#         self.telco1_config = {
#             "binary": [
#                 "Gender",
#                 "Married",
#                 "Referred a Friend",
#                 "Phone Service",
#                 "Multiple Lines",
#                 "Internet Service",
#                 "Online Security",
#                 "Online Backup",
#                 "Device Protection Plan",
#                 "Premium Tech Support",
#                 "Streaming TV",
#                 "Streaming Movies",
#                 "Streaming Music",
#                 "Unlimited Data",
#                 "Paperless Billing",
#             ],
#             "ordinal": [],
#             "ohe": ["Offer", "Internet Type", "Contract", "Payment Method"],
#             "ohe_drop_first": {
#                 "Offer": True,
#                 "Internet Type": True,
#                 "Contract": True,
#                 "Payment Method": True,
#             },
#             "standard": [
#                 "Age",
#                 "Number of Dependents",
#                 "Number of Referrals",
#                 "Avg Monthly Long Distance Charges",
#                 "Tenure in Months",
#                 "Avg Monthly GB Download",
#                 "Monthly Charge",
#                 "Total Charges",
#                 "Total Refunds",
#                 "Total Extra Data Charges",
#                 "Total Long Distance Charges",
#             ],
#             "minmax": ["Satisfaction Score"],
#             "passthrough_numeric": [],
#             "group_layout": {
#                 "Profile": [
#                     "Gender",
#                     "Age",
#                     "Married",
#                     "Number of Dependents",
#                     "Satisfaction Score",
#                 ],
#                 "Contract": [
#                     "Tenure in Months",
#                     "Offer_Offer B",
#                     "Offer_Offer C",
#                     "Offer_Offer D",
#                     "Offer_Offer E",
#                     "Unlimited Data",
#                     "Contract_One Year",
#                     "Contract_Two Year",
#                 ],
#                 "Billing": [
#                     "Avg Monthly Long Distance Charges",
#                     "Paperless Billing",
#                     "Payment Method_Credit Card",
#                     "Payment Method_Mailed Check",
#                     "Monthly Charge",
#                     "Total Charges",
#                     "Total Refunds",
#                     "Total Extra Data Charges",
#                     "Total Long Distance Charges",
#                 ],
#                 "Usage": [
#                     "Referred a Friend",
#                     "Number of Referrals",
#                     "Phone Service",
#                     "Multiple Lines",
#                     "Internet Service",
#                     "Internet Type_DSL",
#                     "Internet Type_Fiber Optic",
#                     "Internet Type_No Internet",
#                     "Avg Monthly GB Download",
#                     "Online Security",
#                     "Online Backup",
#                     "Device Protection Plan",
#                     "Premium Tech Support",
#                     "Streaming TV",
#                     "Streaming Movies",
#                     "Streaming Music",
#                 ],
#             },
#         }

#         self.telco2_config = {
#             "binary": [
#                 "gender",
#                 "Partner",
#                 "Dependents",
#                 "PhoneService",
#                 "PaperlessBilling",
#                 # Service cols as binary — collapses redundancy with
#                 # InternetService_No and balances group sizes.
#                 "MultipleLines",
#                 "OnlineSecurity",
#                 "OnlineBackup",
#                 "DeviceProtection",
#                 "TechSupport",
#                 "StreamingTV",
#                 "StreamingMovies",
#             ],
#             "ordinal": [],
#             "ohe": ["InternetService", "Contract", "PaymentMethod"],
#             "ohe_drop_first": {
#                 "InternetService": True,
#                 "Contract": True,
#                 "PaymentMethod": True,
#             },
#             "standard": ["tenure", "MonthlyCharges", "TotalCharges"],
#             "minmax": [],
#             "passthrough_numeric": ["SeniorCitizen"],
#             "group_layout": {
#                 "Profile": ["gender", "SeniorCitizen", "Partner", "Dependents"],
#                 "Contract": [
#                     "tenure",
#                     "Contract_One year",
#                     "Contract_Two year",
#                 ],
#                 "Billing": [
#                     "PaperlessBilling",
#                     "PaymentMethod_Credit card (automatic)",
#                     "PaymentMethod_Electronic check",
#                     "PaymentMethod_Mailed check",
#                     "MonthlyCharges",
#                     "TotalCharges",
#                 ],
#                 "Usage": [
#                     "PhoneService",
#                     "MultipleLines",
#                     "InternetService_Fiber optic",
#                     "InternetService_No",
#                     "OnlineSecurity",
#                     "OnlineBackup",
#                     "DeviceProtection",
#                     "TechSupport",
#                     "StreamingTV",
#                     "StreamingMovies",
#                 ],
#             },
#         }

#     # ── Config dispatch ───────────────────────────────────────────────────────

#     def _get_config(self) -> dict:
#         if "bank" in self.dataset_type:
#             return self.bank_config
#         elif "telco_2" in self.dataset_type or "telco2" in self.dataset_type:
#             return self.telco2_config
#         else:
#             return self.telco1_config

#     # ── Encoding pipeline ─────────────────────────────────────────────────────

#     def _apply_binary(self, df: pd.DataFrame, cols: list) -> dict:
#         """Map binary/ternary string columns to a single 0/1 column each."""
#         out = {}
#         for col in cols:
#             if col not in df.columns:
#                 continue
#             mapped = df[col].map(self._BINARY_MAP)
#             if mapped.isna().any():
#                 bad = df.loc[mapped.isna(), col].unique().tolist()
#                 raise ValueError(
#                     f"Unmapped value(s) in column '{col}': {bad}. "
#                     f"Extend EstablishedFeatureEngineer._BINARY_MAP if these "
#                     f"are legitimate categories."
#                 )
#             out[col] = mapped.values.astype(float)
#         return out

#     def _apply_ordinal(self, df: pd.DataFrame, cols: list) -> dict:
#         """Apply hardcoded ordinal mappings (currently only Card Type)."""
#         out = {}
#         for col in cols:
#             if col not in df.columns:
#                 continue
#             if col == "Card Type":
#                 series = df[col].astype(str).str.upper().str.strip()
#                 mapped = series.map(self._CARD_TYPE_MAP)
#                 if mapped.isna().any():
#                     bad = df.loc[mapped.isna(), col].unique().tolist()
#                     raise ValueError(f"Unmapped Card Type value(s): {bad}")
#                 out[col] = mapped.values.astype(float)
#             else:
#                 raise ValueError(f"No ordinal mapping defined for column '{col}'")
#         return out

#     def _apply_ohe(
#         self, df: pd.DataFrame, cols: list, drop_first_map: dict, fit: bool
#     ) -> dict:
#         """One-hot encode nominal categoricals, locking categories at fit time."""
#         out = {}
#         for col in cols:
#             if col not in df.columns:
#                 continue
#             series = df[col].fillna("Unknown").astype(str)

#             if fit:
#                 self.ohe_categories[col] = sorted(series.unique().tolist())
#             cats = self.ohe_categories.get(col, [])

#             drop_first = drop_first_map.get(col, True)
#             cats_to_emit = cats[1:] if drop_first else cats
#             for cat in cats_to_emit:
#                 out[f"{col}_{cat}"] = (series == cat).astype(int).values
#         return out

#     def _apply_standard(self, df: pd.DataFrame, cols: list, fit: bool) -> dict:
#         """StandardScaler. Fit on train only; transform on val/test."""
#         cols = [c for c in cols if c in df.columns]
#         if not cols:
#             return {}
#         # TotalCharges in telco datasets sometimes has stray non-numeric strings.
#         X = (
#             df[cols]
#             .apply(pd.to_numeric, errors="coerce")
#             .fillna(0)
#             .values.astype(float)
#         )
#         if fit:
#             self.standard_cols = cols
#             X = self.standard_scaler.fit_transform(X)
#         else:
#             X = self.standard_scaler.transform(X)
#         return {c: X[:, i] for i, c in enumerate(cols)}

#     def _apply_minmax(self, df: pd.DataFrame, cols: list, fit: bool) -> dict:
#         """MinMaxScaler for bounded features (Satisfaction Score)."""
#         cols = [c for c in cols if c in df.columns]
#         if not cols:
#             return {}
#         X = df[cols].fillna(0).values.astype(float)
#         if fit:
#             self.minmax_cols = cols
#             X = self.minmax_scaler.fit_transform(X)
#         else:
#             X = self.minmax_scaler.transform(X)
#         return {c: X[:, i] for i, c in enumerate(cols)}

#     def _apply_passthrough(self, df: pd.DataFrame, cols: list) -> dict:
#         """Already-numeric columns that need no scaling (e.g. NumOfProducts, SeniorCitizen)."""
#         out = {}
#         for col in cols:
#             if col not in df.columns:
#                 continue
#             out[col] = (
#                 pd.to_numeric(df[col], errors="coerce").fillna(0).values.astype(float)
#             )
#         return out

#     # ── Matrix assembly ───────────────────────────────────────────────────────

#     def _build_feature_matrix(self, df: pd.DataFrame, fit: bool) -> tuple:
#         config = self._get_config()
#         col_map = {}

#         col_map.update(self._apply_binary(df, config.get("binary", [])))
#         col_map.update(self._apply_ordinal(df, config.get("ordinal", [])))
#         col_map.update(
#             self._apply_ohe(
#                 df, config.get("ohe", []), config.get("ohe_drop_first", {}), fit
#             )
#         )
#         col_map.update(self._apply_standard(df, config.get("standard", []), fit))
#         col_map.update(self._apply_minmax(df, config.get("minmax", []), fit))
#         col_map.update(
#             self._apply_passthrough(df, config.get("passthrough_numeric", []))
#         )

#         # Build the matrix in the fixed group_layout order. Columns declared in
#         # the layout but not produced (e.g. an OHE category absent from this
#         # split) are filled with zeros so group indices stay aligned. This
#         # mirrors the notebook's val/test column-alignment behaviour.
#         layout = config["group_layout"]
#         ordered_cols = []
#         columns_data = []
#         for group_cols in layout.values():
#             for c in group_cols:
#                 ordered_cols.append(c)
#                 columns_data.append(
#                     col_map[c] if c in col_map else np.zeros(len(df), dtype=float)
#                 )

#         X = np.column_stack(columns_data) if columns_data else np.empty((len(df), 0))
#         return X, ordered_cols

#     def _build_groups(self) -> dict:
#         layout = self._get_config()["group_layout"]
#         groups = {}
#         cursor = 0
#         for group_name, group_cols in layout.items():
#             n = len(group_cols)
#             groups[group_name] = list(range(cursor, cursor + n))
#             cursor += n
#         return groups

#     @staticmethod
#     def _extract_target(df: pd.DataFrame):
#         target_col = next(
#             (t for t in ["Churn", "Exited", "Churn Label"] if t in df.columns), None
#         )
#         if target_col is None:
#             return None
#         return (
#             df[target_col]
#             .map({"Yes": 1, "No": 0, "yes": 1, "no": 0, 1: 1, 0: 0})
#             .fillna(0)
#             .values
#         )

#     # ── Public API ────────────────────────────────────────────────────────────

#     def fit_transform(self, df: pd.DataFrame) -> tuple:
#         """Fit on training data and transform it. Returns (X, y, names, groups)."""
#         print(f"    Fitting Established Engineer ({self.dataset_type})...")
#         X, col_names = self._build_feature_matrix(df, fit=True)
#         self.feature_names_out = col_names
#         self.feature_groups = self._build_groups()
#         self.fitted = True

#         print(f"      - Feature count: {len(col_names)}")
#         print(
#             f"      - Groups: { {k: len(v) for k, v in self.feature_groups.items()} }"
#         )
#         return X, self._extract_target(df), col_names, self.feature_groups

#     def transform(self, df: pd.DataFrame) -> tuple:
#         """Transform val/test data using fitted parameters only — no re-fitting."""
#         if not self.fitted:
#             raise ValueError(
#                 "Call fit_transform() on training data before transform()."
#             )
#         X, col_names = self._build_feature_matrix(df, fit=False)
#         return X, self._extract_target(df), col_names, self.feature_groups


# # ─────────────────────────────────────────────────────────────────────────────
# # UTILITIES
# # ─────────────────────────────────────────────────────────────────────────────


# def _log_transform_warnings(df: pd.DataFrame, col_names: list) -> None:
#     """
#     Logs which categorical columns triggered the OHE unknown-category fallback
#     during transform. Helps detect distribution shift between non-cold fit data
#     and cold-start transform data.
#     """
#     pass
