import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

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
        elif "telco_2" in self.dataset_type:
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
        # Geography has 3 categories: France, Germany, Spain
        # Sorted alphabetically → France is reference (dropped)
        # Output columns: Geography_Germany, Geography_Spain
        for col in config.get("ohe", []):
            if col not in df.columns:
                continue
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
        print(f"   ❄️  Fitting Cold-Start Engineer ({self.dataset_type})...")
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
# 2. ESTABLISHED ENGINEER  (GATEFuse Path)
# ─────────────────────────────────────────────────────────────────────────────


class EstablishedFeatureEngineer:
    """
    Phase 2b: Established User Feature Engineering for the GATEFuse model.

    Strategy:
      1. Feature Grouping (Profile, Contract, Billing, Usage) — preserved for
         attention masking. Group indices are returned alongside the feature matrix.
      2. Explicit encoding: binary map, ordinal for contract, one-hot for nominals,
         ternary expansion for service columns.
      3. StandardScaler for continuous, MinMaxScaler for bounded features.
      4. Variance Threshold deliberately SKIPPED to preserve group index alignment.
         (The GATEFuse model relies on stable column indices.)
    """

    def __init__(self, dataset_type: str = "telco"):
        self.dataset_type = dataset_type.lower()
        self.fitted = False

        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler(feature_range=(0, 1))
        self.standard_cols: list = []
        self.minmax_cols: list = []
        self.ohe_categories: dict = {}

        self.feature_names_out: list = []
        self.feature_groups: dict = {}

        # ── FEATURE GROUP CONFIGS ─────────────────────────────────────────────
        # Geography OHE: France (reference, dropped), Germany, Spain
        # Output columns: Geography_Germany, Geography_Spain
        # Card Type OHE: Diamond (reference, dropped), Gold, Platinum, Silver
        # Output columns: Card Type_Gold, Card Type_Platinum, Card Type_Silver

        self.bank_groups = {
            "Profile": [
                "CreditScore",
                "Age",
                "Geography_Germany",  # fixed: was Geography_DE
                "Geography_Spain",  # fixed: was Geography_FR
                "Geography_France",  # fixed: was Geography_ES
                "Gender",
                "EstimatedSalary",
                "Satisfaction Score",  # added
            ],
            "Contract": [
                "Tenure",
                "Card Type_Gold",
                "Card Type_Platinum",
                "Card Type_Silver",
            ],
            "Billing": ["Balance", "has_zero_balance", "Point Earned"],
            "Usage": ["NumOfProducts", "HasCrCard", "IsActiveMember"],
        }

        self.telco1_groups = {
            "Profile": [
                "Gender",
                "Age",
                "Married",
                "Number of Dependents",
                "Satisfaction Score",
            ],
            "Contract": [
                "Tenure in Months",
                "Contract",
                "Offer_Offer A",
                "Offer_Offer B",
                "Offer_Offer C",
                "Offer_Offer D",
                "Offer_Offer E",
            ],
            "Billing": [
                "Monthly Charge",
                "Total Charges",
                "Total Refunds",
                "Total Extra Data Charges",
                "Total Long Distance Charges",
                "Paperless Billing",
                "Payment Method_Credit Card (automatic)",
                "Payment Method_Electronic check",
                "Payment Method_Mailed check",
            ],
            "Usage": [
                "Phone Service",
                "has_Multiple Lines",
                "no_svc_Multiple Lines",
                "Internet Service_Fiber optic",
                "Internet Service_No",
                "Internet Type_Cable",
                "Internet Type_DSL",
                "Internet Type_Fiber Optic",
                "Internet Type_No Internet Service",
                "Avg Monthly GB Download",
                "has_Online Security",
                "has_Online Backup",
                "has_Device Protection Plan",
                "has_Premium Tech Support",
                "has_Streaming TV",
                "has_Streaming Movies",
                "has_Streaming Music",
                "has_Unlimited Data",
            ],
        }

        self.telco2_groups = {
            "Profile": ["gender", "SeniorCitizen", "Partner", "Dependents"],
            "Contract": [
                "tenure",
                "Contract",
                "InternetService_Fiber optic",
                "InternetService_No",
            ],
            "Billing": [
                "MonthlyCharges",
                "TotalCharges",
                "PaperlessBilling",
                "PaymentMethod_Credit card (automatic)",
                "PaymentMethod_Electronic check",
                "PaymentMethod_Mailed check",
            ],
            "Usage": [
                "PhoneService",
                "has_MultipleLines",
                "no_svc_MultipleLines",
                "has_OnlineSecurity",
                "has_OnlineBackup",
                "has_DeviceProtection",
                "has_TechSupport",
                "has_StreamingTV",
                "has_StreamingMovies",
            ],
        }

    def _get_group_config(self) -> dict:
        if "bank" in self.dataset_type:
            return self.bank_groups
        elif "telco_2" in self.dataset_type or "telco2" in self.dataset_type:
            return self.telco2_groups
        else:
            return self.telco1_groups

    # ── Engineering helpers ───────────────────────────────────────────────────

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "Balance" in df.columns:
            df["has_zero_balance"] = (df["Balance"] == 0).astype(int)

        if "NumOfProducts" in df.columns:
            df["NumOfProducts"] = df["NumOfProducts"].clip(upper=3)

        for col in [
            "Number of Referrals",
            "Avg Monthly GB Download",
            "Total Charges",
            "TotalCharges",
        ]:
            if col in df.columns:
                df[col] = _safe_log1p(df[col].fillna(0))

        return df

    def _build_feature_matrix(self, df: pd.DataFrame, fit: bool) -> tuple:
        """
        Build the full feature matrix in a consistent column order
        that matches the group config indices.
        """
        group_config = self._get_group_config()

        # Flatten all columns referenced in groups (preserves order)
        all_cols_ordered = []
        seen = set()
        for cols in group_config.values():
            for c in cols:
                if c not in seen:
                    all_cols_ordered.append(c)
                    seen.add(c)

        col_map = {}

        # Binary columns
        for col in df.columns:
            unique_vals = set(str(v).lower().strip() for v in df[col].dropna().unique())
            if unique_vals <= {
                "yes",
                "no",
                "1",
                "0",
                "true",
                "false",
                "male",
                "female",
            }:
                col_map[col] = _encode_binary(df[col]).fillna(0).values

        # Numeric columns
        for col in df.select_dtypes(include="number").columns:
            if col not in col_map:
                col_map[col] = df[col].fillna(0).values.astype(float)

        # Contract ordinal
        if "Contract" in df.columns:
            col_map["Contract"] = _encode_contract(df["Contract"]).values

        # One-hot for nominal categoricals
        for col in df.select_dtypes(include="object").columns:
            unique_vals = set(str(v).lower().strip() for v in df[col].dropna().unique())
            is_binary = unique_vals <= {"yes", "no", "true", "false", "male", "female"}
            is_contract = col == "Contract"
            if is_binary or is_contract:
                continue
            series = df[col].fillna("Unknown").astype(str)
            if fit:
                cats = sorted(series.unique().tolist())
                self.ohe_categories[col] = cats
            cats = self.ohe_categories.get(col, [])
            for cat in cats[1:]:
                col_map[f"{col}_{cat}"] = (series == cat).astype(int).values

        # Ternary service columns
        ternary_candidates = [
            c
            for c in df.columns
            if df[c].dtype == object
            and any("no " in str(v).lower() for v in df[c].dropna().unique())
        ]
        for col in ternary_candidates:
            df_tern = _encode_ternary_service(df[col].fillna("No"))
            for c in df_tern.columns:
                col_map[c] = df_tern[c].values

        # Scale continuous cols
        num_cols = [
            c
            for c in df.select_dtypes(include="number").columns
            if c in col_map
            and c
            not in [
                "SeniorCitizen",
                "HasCrCard",
                "IsActiveMember",
                "has_zero_balance",
                "NumOfProducts",
                "Contract",
            ]
        ]
        minmax_cols = [
            c for c in num_cols if c in ["Tenure", "tenure", "Tenure in Months"]
        ]
        std_cols = [c for c in num_cols if c not in minmax_cols]

        if std_cols:
            X_std = np.column_stack([col_map[c] for c in std_cols]).astype(float)
            if fit:
                self.standard_cols = std_cols
                X_std = self.standard_scaler.fit_transform(X_std)
            else:
                X_std = self.standard_scaler.transform(X_std)
            for i, c in enumerate(std_cols):
                col_map[c] = X_std[:, i]

        if minmax_cols:
            X_mm = np.column_stack([col_map[c] for c in minmax_cols]).astype(float)
            if fit:
                self.minmax_cols = minmax_cols
                X_mm = self.minmax_scaler.fit_transform(X_mm)
            else:
                X_mm = self.minmax_scaler.transform(X_mm)
            for i, c in enumerate(minmax_cols):
                col_map[c] = X_mm[:, i]

        # Build matrix in fixed group order
        available = [c for c in all_cols_ordered if c in col_map]
        X = np.column_stack([col_map[c] for c in available]).astype(float)

        return X, available

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> tuple:
        """
        Fit and transform established user data.
        Returns (X, y, feature_names, feature_groups).
        """
        print(f"   🔥 Fitting Established Engineer ({self.dataset_type})...")

        df_eng = self._engineer_features(df)
        X, col_names = self._build_feature_matrix(df_eng, fit=True)
        self.feature_names_out = col_names

        # Build group → index mapping
        group_config = self._get_group_config()
        self.feature_groups = {}
        for group_name, group_cols in group_config.items():
            indices = [col_names.index(c) for c in group_cols if c in col_names]
            self.feature_groups[group_name] = indices

        print(
            f"      - Groups: { {k: len(v) for k, v in self.feature_groups.items()} }"
        )

        # Target
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

        self.fitted = True
        return X, y, col_names, self.feature_groups

    def transform(self, df: pd.DataFrame) -> tuple:
        """Transform new data using fitted parameters."""
        if not self.fitted:
            raise ValueError("Call fit_transform() before transform().")
        df_eng = self._engineer_features(df)
        X, col_names = self._build_feature_matrix(df_eng, fit=False)
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
        return X, y, col_names, self.feature_groups


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────


def _log_transform_warnings(df: pd.DataFrame, col_names: list) -> None:
    """
    Logs which categorical columns triggered the OHE unknown-category fallback
    during transform. Helps detect distribution shift between non-cold fit data
    and cold-start transform data.
    """
    pass
