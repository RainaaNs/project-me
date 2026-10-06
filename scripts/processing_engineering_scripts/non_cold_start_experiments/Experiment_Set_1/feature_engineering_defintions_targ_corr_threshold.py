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
        "yes": 1, "no": 0,
        "true": 1, "false": 0,
        "male": 1, "female": 0,
        "1": 1, "0": 0,
        1: 1, 0: 0,
    }
    return series.map(lambda x: mapping.get(str(x).lower().strip(), np.nan))


def _encode_contract(series: pd.Series) -> pd.Series:
    """Ordinal: Month-to-Month=0, One Year=1, Two Year=2."""
    mapping = {
        "month-to-month": 0, "one year": 1, "two year": 2,
        "month to month": 0,
    }
    return series.map(lambda x: mapping.get(str(x).lower().strip(), 0))


def _encode_ternary_service(series: pd.Series) -> pd.DataFrame:
    """
    Converts Yes / No / No <Service> ternary columns to two binary columns:
      - has_<col>    : 1 if Yes
      - no_svc_<col> : 1 if 'No <Service>' (can't get it, not just doesn't want it)
    Dropping the plain 'No' case as the reference category.
    """
    col_name   = series.name if hasattr(series, "name") else "feature"
    has_col    = f"has_{col_name}".replace(" ", "_").lower()
    no_svc_col = f"no_svc_{col_name}".replace(" ", "_").lower()

    has_vals    = series.map(lambda x: 1 if str(x).lower().strip() == "yes" else 0)
    no_svc_vals = series.map(
        lambda x: 1 if ("no " in str(x).lower() and str(x).lower().strip() != "no") else 0
    )
    return pd.DataFrame({has_col: has_vals, no_svc_col: no_svc_vals})


# ─────────────────────────────────────────────────────────────────────────────
# 1. COLD-START ENGINEER  (MPMN / Few-Shot Path)
# ─────────────────────────────────────────────────────────────────────────────


class ColdStartFeatureEngineer:
    """
    Phase 2a: Cold-Start Feature Engineering for the MPMN (Prototypical Network).
    *** UNCHANGED — do not modify this class. ***
    """

    def __init__(self, dataset_type: str = "telco"):
        self.dataset_type = dataset_type.lower()
        self.fitted = False

        self.standard_scaler     = StandardScaler()
        self.minmax_scaler       = MinMaxScaler(feature_range=(0, 1))
        self.standard_cols: list = []
        self.minmax_cols: list   = []
        self.drop_corr_cols: list = []
        self.ohe_categories: dict = {}
        self.target_maps: dict    = {}
        self.feature_names_out: list = []

        self.bank_config = {
            "standard":    ["CreditScore", "Age", "Balance", "EstimatedSalary", "Point Earned"],
            "minmax":      ["Tenure"],
            "binary":      ["HasCrCard", "IsActiveMember", "Gender"],
            "ordinal_bin": ["NumOfProducts"],
            "ohe":         ["Geography", "Card Type"],
            "engineered":  ["has_zero_balance"],
            "target_enc":  [],
        }

        self.telco1_config = {
            "standard": [
                "Age", "Number of Dependents", "Number of Referrals",
                "Avg Monthly Long Distance Charges", "Avg Monthly GB Download",
                "Monthly Charge",
            ],
            "minmax":   ["Tenure in Months"],
            "binary":   ["Gender", "Married", "Phone Service", "Paperless Billing"],
            "contract": ["Contract"],
            "ohe":      ["Offer", "Internet Type", "Payment Method", "Internet Service"],
            "ternary":  [
                "Multiple Lines", "Online Security", "Online Backup",
                "Device Protection Plan", "Premium Tech Support",
                "Streaming TV", "Streaming Movies", "Streaming Music", "Unlimited Data",
            ],
            "target_enc": [],
        }

        self.telco2_config = {
            "standard": ["MonthlyCharges"],
            "minmax":   ["tenure"],
            "binary":   [
                "gender", "SeniorCitizen", "Partner", "Dependents",
                "PhoneService", "PaperlessBilling",
            ],
            "contract": ["Contract"],
            "ohe":      ["InternetService", "PaymentMethod"],
            "ternary":  [
                "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies",
            ],
            "target_enc": [],
        }

    def _get_config(self) -> dict:
        if "bank"   in self.dataset_type: return self.bank_config
        if "telco2" in self.dataset_type: return self.telco2_config
        return self.telco1_config

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "Balance" in df.columns:
            df["has_zero_balance"] = (df["Balance"] == 0).astype(int)
        if "NumOfProducts" in df.columns:
            df["NumOfProducts"] = df["NumOfProducts"].clip(upper=3)
        for col in ["Number of Referrals", "Avg Monthly GB Download",
                    "Total Charges", "TotalCharges"]:
            if col in df.columns:
                df[col] = _safe_log1p(df[col].fillna(0))
        return df

    def _apply_encoding(self, df: pd.DataFrame, config: dict, fit: bool) -> tuple:
        parts, col_names = [], []

        std_cols = [c for c in config.get("standard", []) if c in df.columns]
        if std_cols:
            X_std = df[std_cols].fillna(0).values.astype(float)
            if fit:
                self.standard_cols = std_cols
                X_std = self.standard_scaler.fit_transform(X_std)
            else:
                X_std = self.standard_scaler.transform(X_std)
            parts.append(X_std); col_names.extend(std_cols)

        mm_cols = [c for c in config.get("minmax", []) if c in df.columns]
        if mm_cols:
            X_mm = df[mm_cols].fillna(0).values.astype(float)
            if fit:
                self.minmax_cols = mm_cols
                X_mm = self.minmax_scaler.fit_transform(X_mm)
            else:
                X_mm = self.minmax_scaler.transform(X_mm)
            parts.append(X_mm); col_names.extend(mm_cols)

        for col in config.get("binary", []):
            if col in df.columns:
                parts.append(_encode_binary(df[col]).fillna(0).values.reshape(-1, 1))
                col_names.append(col)

        for col in config.get("contract", []):
            if col in df.columns:
                parts.append(_encode_contract(df[col]).values.reshape(-1, 1))
                col_names.append(col)

        for col in config.get("ohe", []):
            if col not in df.columns: continue
            if col == "Offer" and "telco1" in self.dataset_type:
                series = df[col].fillna("No Offer").astype(str)
            elif col == "Internet Type" and "telco1" in self.dataset_type:
                series = df[col].fillna("No Internet").astype(str)
            else:
                series = df[col].fillna("Unknown").astype(str)
            if fit:
                self.ohe_categories[col] = sorted(series.unique().tolist())
            for cat in self.ohe_categories.get(col, [])[1:]:
                parts.append((series == cat).astype(int).values.reshape(-1, 1))
                col_names.append(f"{col}_{cat}")

        for col in config.get("ternary", []):
            if col not in df.columns: continue
            df_tern = _encode_ternary_service(df[col].fillna("No"))
            parts.append(df_tern.values); col_names.extend(df_tern.columns.tolist())

        for col in config.get("engineered", []):
            if col in df.columns:
                parts.append(df[col].fillna(0).values.reshape(-1, 1))
                col_names.append(col)

        for col in config.get("ordinal_bin", []):
            if col in df.columns:
                parts.append(df[col].fillna(0).values.reshape(-1, 1))
                col_names.append(col)

        X = np.hstack(parts) if parts else np.empty((len(df), 0))
        return X, col_names

    def _fit_corr_filter(self, X: np.ndarray, col_names: list) -> list:
        df_X  = pd.DataFrame(X, columns=col_names)
        corr  = df_X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        return [c for c in upper.columns if any(upper[c] > 0.95)]

    def fit(self, df_non_cold: pd.DataFrame) -> "ColdStartFeatureEngineer":
        print(f"   Fitting Cold-Start Engineer ({self.dataset_type})...")
        config = self._get_config()
        df_eng = self._engineer_features(df_non_cold)
        X, col_names = self._apply_encoding(df_eng, config, fit=True)
        self.drop_corr_cols = self._fit_corr_filter(X, col_names)
        if self.drop_corr_cols:
            print(f"      - Dropping highly correlated cols: {self.drop_corr_cols}")
        self.fitted = True
        print(f"      - Fit complete on {len(df_non_cold)} non-cold samples.")
        return self

    def transform(self, df: pd.DataFrame) -> tuple:
        if not self.fitted:
            raise ValueError("Call fit() on non-cold data before transform().")
        config = self._get_config()
        df_eng = self._engineer_features(df)
        X, col_names = self._apply_encoding(df_eng, config, fit=False)
        keep_idx  = [i for i, c in enumerate(col_names) if c not in self.drop_corr_cols]
        X         = X[:, keep_idx]
        col_names = [col_names[i] for i in keep_idx]
        target_col = next(
            (t for t in ["Churn", "Exited", "Churn Label"] if t in df.columns), None
        )
        y = None
        if target_col:
            y = (
                df[target_col]
                .map({"Yes": 1, "No": 0, "yes": 1, "no": 0, 1: 1, 0: 0})
                .fillna(0).values
            )
        self.feature_names_out = col_names
        return X, y, col_names


# ─────────────────────────────────────────────────────────────────────────────
# 2. ESTABLISHED FEATURE ENGINEER  (GATEFuse Path)
# ─────────────────────────────────────────────────────────────────────────────
#
# Columns removed from X per dataset:
#
#   bank   — RowNumber, CustomerId, Surname (identifiers), is_cold_start
#             (routing flag), Churn (target).
#
#   telco1 — Customer ID (identifier), is_cold_start (routing flag), Churn
#             (target), Customer Status / Churn Score / CLTV / Churn Category /
#             Churn Reason / Quarter (post-hoc leakage), City / Zip Code /
#             Country / State (high-cardinality strings).
#
#   telco2 — customerID (identifier), is_cold_start (routing flag), Churn
#             (target).
#
# Target correlation filtering:
#   If corr_threshold is set (not None), features whose absolute Pearson
#   correlation with the Churn label falls below the threshold are dropped
#   after encoding and scaling. Threshold is fit on training data only and
#   applied consistently to val/test via the saved dropped_cols list.
#   Set corr_threshold=None (default) to disable filtering (baseline run).


DATASET_CONFIG = {
    "bank": {
        "label_col":   "Churn",
        "id_cols":     ["RowNumber", "CustomerId", "Surname"],
        "drop_cols":   ["is_cold_start"],
        "ordinal_cols": {
            "Card Type": {"SILVER": 1, "GOLD": 2, "PLATINUM": 3, "DIAMOND": 4}
        },
        "binary_cols": {
            "Gender":         {"Male": 0, "Female": 1},
            "HasCrCard":      {1: 1, 0: 0, "1": 1, "0": 0},
            "IsActiveMember": {1: 1, 0: 0, "1": 1, "0": 0},
            "Complain":       {1: 1, 0: 0, "1": 1, "0": 0},
        },
        "int_cols":    ["NumOfProducts"],
        "fillna":      {},
        "categorical_cols": ["Geography"],
        "dummy_regex": "^Geography_",
        "std_numerical_cols": [
            "CreditScore", "Age", "Tenure", "Balance",
            "EstimatedSalary", "Point Earned", "Satisfaction Score",
        ],
        "scalers_subdir": "bank_scalers",
    },

    "telco1": {
        "label_col": "Churn",
        "id_cols": [
            "Customer ID", "Customer Status", "Churn Score", "CLTV",
            "Churn Category", "Churn Reason", "Quarter",
            "City", "Zip Code", "Country", "State",
        ],
        "drop_cols":   ["is_cold_start"],
        "ordinal_cols": {},
        "binary_map":  {"Yes": 1, "No": 0, "Male": 0, "Female": 1},
        "binary_cols": [
            "Gender", "Married", "Under 30", "Senior Citizen",
            "Dependents", "Referred a Friend",
            "Phone Service", "Multiple Lines", "Internet Service",
            "Online Security", "Online Backup", "Device Protection Plan",
            "Premium Tech Support", "Streaming TV", "Streaming Movies",
            "Streaming Music", "Unlimited Data", "Paperless Billing",
        ],
        "fillna":      {"Internet Type": "No Internet", "Offer": "No Offer"},
        "categorical_cols": ["Offer", "Internet Type", "Contract", "Payment Method"],
        "dummy_regex": "^(Offer_|Internet Type_|Contract_|Payment Method_)",
        "std_numerical_cols": [
            "Age", "Number of Dependents", "Number of Referrals",
            "Tenure in Months", "Avg Monthly Long Distance Charges",
            "Avg Monthly GB Download", "Monthly Charge", "Total Charges",
            "Total Refunds", "Total Extra Data Charges",
            "Total Long Distance Charges", "Total Revenue",
            "Satisfaction Score", "Latitude", "Longitude", "Population",
        ],
        "scalers_subdir": "telco1_scalers",
    },

    "telco2": {
        "label_col":   "Churn",
        "id_cols":     ["customerID"],
        "drop_cols":   ["is_cold_start"],
        "ordinal_cols": {},
        "binary_map":  {"Yes": 1, "No": 0, "Male": 0, "Female": 1},
        "binary_cols": ["gender", "Partner", "Dependents", "PaperlessBilling", "PhoneService"],
        "fillna":      {},
        "categorical_cols": [
            "SeniorCitizen", "InternetService", "Contract", "PaymentMethod",
            "OnlineSecurity", "TechSupport", "MultipleLines",
            "OnlineBackup", "DeviceProtection", "StreamingTV", "StreamingMovies",
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


class EstablishedFeatureEngineer:
    """
    Phase 2b: Established User Feature Engineering for the GATEFuse model.

    Parameters
    ----------
    dataset_type : str
        One of "bank", "telco1", "telco2".
    scalers_dir : str
        Root directory under which scaler .pkl files are saved/loaded.
        Artefacts land in <scalers_dir>/<dataset>_scalers/.
    corr_threshold : float or None
        If set, features whose absolute Pearson correlation with the Churn
        label falls below this threshold are dropped after encoding and
        scaling. Fit on training data only; applied to val/test via saved
        dropped_cols list.
        None (default) = no filtering, all features kept (baseline run).
    """

    def __init__(
        self,
        dataset_type:    str,
        scalers_dir:     str,
        corr_threshold:  float = None,
    ):
        self.dataset_type    = dataset_type.lower()
        self.scalers_dir     = scalers_dir
        self.corr_threshold  = corr_threshold
        self.fitted          = False

        self._cfg        = DATASET_CONFIG[self.dataset_type]
        self.std_scaler  = StandardScaler()
        self.ohe_columns: list  = []
        self.dropped_cols: list = []   # columns dropped by correlation filter

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
        cfg          = self._cfg
        label_col    = cfg["label_col"]
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
                X[col] = X[col].str.upper().map(mapping)
                if fit:
                    self._save(mapping, f"{col.replace(' ', '')}_mapping.pkl")

        # 3. binary encoding
        if self.dataset_type == "bank":
            for col, mapping in cfg["binary_cols"].items():
                if col in X.columns:
                    X[col] = X[col].map(mapping)
                    if fit:
                        self._save(mapping, f"{col.replace(' ', '')}_mapping.pkl")
            for col in cfg.get("int_cols", []):
                if col in X.columns:
                    X[col] = X[col].fillna(0).astype(int)
        else:
            binary_map = cfg["binary_map"]
            for col in cfg["binary_cols"]:
                if col in X.columns:
                    X[col] = X[col].map(binary_map)
            if fit:
                self._save(binary_map, "binary_map.pkl")

        # 4. one-hot encoding (drop_first=False, all dummies kept)
        categorical_cols = [c for c in cfg["categorical_cols"] if c in X.columns]
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

        dummy_cols      = X.filter(regex=cfg["dummy_regex"]).columns
        X[dummy_cols]   = X[dummy_cols].astype(int)

        if fit:
            self.ohe_columns = X.columns.tolist()
            self._save(self.ohe_columns, "ohe_columns.pkl")
        else:
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
            std_scaler  = self._load("std_scaler.pkl")
            X[std_cols] = std_scaler.transform(X[std_cols])
        return X

    def _apply_corr_filter(self, X: pd.DataFrame, y: np.ndarray, fit: bool) -> pd.DataFrame:
        """
        Target correlation filter.
        Drops features whose |Pearson r| with Churn < corr_threshold.
        Only runs when corr_threshold is not None.
        Fit determines whether to learn dropped_cols (train) or apply saved list (val/test).
        """
        if self.corr_threshold is None:
            return X

        if fit:
            correlations = X.apply(lambda col: abs(col.corr(pd.Series(y))))
            self.dropped_cols = correlations[correlations < self.corr_threshold].index.tolist()
            self._save(self.dropped_cols, "corr_dropped_cols.pkl")

            if self.dropped_cols:
                print(f"   [Corr filter] threshold={self.corr_threshold} — "
                      f"dropping {len(self.dropped_cols)} features:")
                for col in self.dropped_cols:
                    print(f"     - {col}  (|r|={correlations[col]:.4f})")
            else:
                print(f"   [Corr filter] threshold={self.corr_threshold} — "
                      f"no features dropped.")
        else:
            self.dropped_cols = self._load("corr_dropped_cols.pkl")

        return X.drop(columns=[c for c in self.dropped_cols if c in X.columns])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> tuple:
        """
        Fit on and transform the training DataFrame.

        Returns
        -------
        X            : np.ndarray,  shape (n_samples, n_features)
        y            : np.ndarray,  shape (n_samples,)
        feature_names: list[str]
        """
        X, y = self._extract_Xy(df)
        X    = self._encode(X, fit=True)
        X    = self._scale(X, fit=True)
        X    = self._apply_corr_filter(X, y, fit=True)

        feature_names = X.columns.tolist()
        self.fitted   = True

        print(f"[{self.dataset_type}] fit_transform complete.")
        print(f"   Shape          : {X.shape}")
        print(f"   Corr threshold : {self.corr_threshold}")

        return X.values, y, feature_names

    def transform(self, df: pd.DataFrame) -> tuple:
        """
        Transform val/test data using fitted artefacts (no fitting).

        Returns
        -------
        X            : np.ndarray,  shape (n_samples, n_features)
        y            : np.ndarray,  shape (n_samples,)
        feature_names: list[str]
        """
        if not self.fitted:
            raise ValueError(
                "Call fit_transform() before transform(), "
                "or load a fitted instance from disk."
            )
        X, y = self._extract_Xy(df)
        X    = self._encode(X, fit=False)
        X    = self._scale(X, fit=False)
        X    = self._apply_corr_filter(X, y, fit=False)

        feature_names = X.columns.tolist()
        return X.values, y, feature_names