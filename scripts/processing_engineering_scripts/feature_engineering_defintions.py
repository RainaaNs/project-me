"""
feature_engineering_definitions.py

Feature engineering definitions for both model paths:

    1. ColdStartFeatureEngineer
       - Used by the MPMN cold-start path.
       - FITS on non-cold-start TRAINING data.
       - TRANSFORMS cold-start train/val/test data using those learned
         parameters.

    2. EstablishedFeatureEngineer
       - Used by the GATEFuse non-cold-start path.
       - FITS on non-cold-start TRAINING data.
       - TRANSFORMS non-cold-start validation/test data.

IMPORTANT
---------
The executable code and DATASET_CONFIG below are the source of truth.
Documentation/specification comments must not override the actual
configuration.

In particular:
    - Bank Complain is KEPT.
    - Bank Satisfaction Score is KEPT.
    - is_cold_start is removed from GATEFuse features.
    - Configured Telco1 features remain configured.
"""


# =============================================================================
# IMPORTS
# =============================================================================

import os
import warnings
import joblib

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler


warnings.filterwarnings("ignore")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _safe_log1p(series):
    """
    Apply log1p safely.

    Negative values are clipped to zero before applying log1p.
    """
    numeric = pd.to_numeric(
        series,
        errors="coerce"
    )

    return np.log1p(
        numeric.clip(lower=0)
    )


def _encode_binary(series):
    """
    Binary encoding used by the ColdStartFeatureEngineer.

    IMPORTANT:
    This preserves the existing cold-start implementation:
        Yes/True/1   -> 1
        No/False/0   -> 0
        Male         -> 1
        Female       -> 0
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

    def encode_value(value):
        if pd.isna(value):
            return np.nan

        if value in mapping:
            return mapping[value]

        value_string = str(value).strip().lower()

        if value_string in mapping:
            return mapping[value_string]

        return value

    return series.map(encode_value)


def _encode_contract(series):
    """
    Encode contract duration ordinally.

    Month-to-Month -> 0
    One Year       -> 1
    Two Year       -> 2
    """
    mapping = {
        "Month-to-Month": 0,
        "One Year": 1,
        "Two Year": 2,
        "month to month": 0,
    }

    def encode_value(value):
        if pd.isna(value):
            return np.nan

        if value in mapping:
            return mapping[value]

        value_string = str(value).strip()
        value_lower = value_string.lower()

        if value_lower in {
            "month-to-month",
            "month to month",
        }:
            return 0

        if value_lower == "one year":
            return 1

        if value_lower == "two year":
            return 2

        return value

    return series.map(encode_value)


def _encode_ternary_service(series, column_name):
    """
    Convert a three-state service variable into two binary variables.

    Example:

        Yes                   -> has_X = 1, no_svc_X = 0
        No                    -> has_X = 0, no_svc_X = 1
        No internet service   -> has_X = 0, no_svc_X = 1
        No phone service      -> has_X = 0, no_svc_X = 1
    """

    values = (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    has_service = (
        values.eq("yes")
        .astype(int)
    )

    no_service = (
        values.isin(
            [
                "no",
                "no internet service",
                "no phone service",
                "none",
                "",
            ]
        )
        .astype(int)
    )

    return pd.DataFrame(
        {
            f"has_{column_name}": has_service,
            f"no_svc_{column_name}": no_service,
        },
        index=series.index,
    )


# =============================================================================
# COLD-START FEATURE ENGINEER
# =============================================================================

class ColdStartFeatureEngineer:
    """
    Feature engineering for the MPMN cold-start path.

    IMPORTANT FITTING STRATEGY
    ---------------------------
    The engineer is fitted on NON-COLD-START training data.

    It is then used to transform:
        - cold-start train
        - cold-start validation
        - cold-start test

    This implements the intended transfer-learning approach where the
    cold-start population receives preprocessing parameters learned from
    established/non-cold-start customers.
    """

    def __init__(
        self,
        dataset_type,
        correlation_threshold=0.95,
    ):
        self.dataset_type = dataset_type.lower()

        self.correlation_threshold = (
            correlation_threshold
        )

        self.fitted = False

        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()

        self.standard_cols = []
        self.minmax_cols = []

        self.ohe_categories = {}

        self.feature_names_out = []

        self.correlation_columns_to_drop = []

        self.target_maps = {}

        self.config = self._get_config()

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    def _get_config(self):

        configs = {

            # -----------------------------------------------------------------
            # BANK
            # -----------------------------------------------------------------

            "bank": {
                "standard": [
                    "CreditScore",
                    "Age",
                    "Balance",
                    "EstimatedSalary",
                    "Point Earned",
                ],

                "minmax": [
                    "Tenure",
                ],

                "binary": [
                    "HasCrCard",
                    "IsActiveMember",
                    "Gender",
                ],

                "ordinal_bin": [
                    "NumOfProducts",
                ],

                "ohe": [
                    "Geography",
                    "Card Type",
                ],

                "engineered": [
                    "has_zero_balance",
                ],

                "target_enc": [],
            },

            # -----------------------------------------------------------------
            # TELCO 1
            # -----------------------------------------------------------------

            "telco1": {
                "standard": [
                    "Age",
                    "Number of Dependents",
                    "Number of Referrals",
                    "Avg Monthly Long Distance Charges",
                    "Avg Monthly GB Download",
                    "Monthly Charge",
                ],

                "minmax": [
                    "Tenure in Months",
                ],

                "binary": [
                    "Gender",
                    "Married",
                    "Phone Service",
                    "Paperless Billing",
                ],

                "contract": [
                    "Contract",
                ],

                "ohe": [
                    "Offer",
                    "Internet Type",
                    "Payment Method",
                    "Internet Service",
                ],

                "ternary": [
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
            },

            # -----------------------------------------------------------------
            # TELCO 2
            # -----------------------------------------------------------------

            "telco2": {
                "standard": [
                    "MonthlyCharges",
                ],

                "minmax": [
                    "tenure",
                ],

                "binary": [
                    "gender",
                    "SeniorCitizen",
                    "Partner",
                    "Dependents",
                    "PhoneService",
                    "PaperlessBilling",
                ],

                "contract": [
                    "Contract",
                ],

                "ohe": [
                    "InternetService",
                    "PaymentMethod",
                ],

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
            },
        }

        if self.dataset_type not in configs:
            raise ValueError(
                f"Unsupported dataset_type '{self.dataset_type}'. "
                f"Expected one of: {list(configs.keys())}"
            )

        return configs[self.dataset_type]

    # =========================================================================
    # FEATURE ENGINEERING
    # =========================================================================

    def _engineer_features(self, df):
        """
        Create engineered features before encoding/scaling.
        """

        X = df.copy()

        # ---------------------------------------------------------------------
        # Bank: zero-balance indicator
        # ---------------------------------------------------------------------

        if "Balance" in X.columns:

            balance = pd.to_numeric(
                X["Balance"],
                errors="coerce"
            )

            X["has_zero_balance"] = (
                balance
                .fillna(0)
                .eq(0)
                .astype(int)
            )

        # ---------------------------------------------------------------------
        # Bank: cap NumOfProducts at 3
        # ---------------------------------------------------------------------

        if "NumOfProducts" in X.columns:

            X["NumOfProducts"] = (
                pd.to_numeric(
                    X["NumOfProducts"],
                    errors="coerce"
                )
                .clip(upper=3)
            )

        # ---------------------------------------------------------------------
        # Existing log transformations
        # ---------------------------------------------------------------------

        log_columns = [
            "Number of Referrals",
            "Avg Monthly GB Download",
            "Total Charges",
            "TotalCharges",
        ]

        for column in log_columns:

            if column in X.columns:
                X[column] = _safe_log1p(
                    X[column]
                )

        return X

    # =========================================================================
    # ENCODING
    # =========================================================================

    def _apply_encoding(
        self,
        X,
        fit=False,
    ):
        """
        Apply scaling, binary encoding, contract encoding,
        one-hot encoding, ternary service encoding, and
        ordinal processing.
        """

        X = X.copy()

        # ---------------------------------------------------------------------
        # STANDARD SCALING
        # ---------------------------------------------------------------------

        standard_cols = [
            column
            for column in self.config.get(
                "standard",
                []
            )
            if column in X.columns
        ]

        if standard_cols:

            X[standard_cols] = X[
                standard_cols
            ].apply(
                pd.to_numeric,
                errors="coerce",
            )

            if fit:

                self.standard_scaler.fit(
                    X[standard_cols]
                )

                self.standard_cols = (
                    standard_cols
                )

            X[standard_cols] = (
                self.standard_scaler.transform(
                    X[standard_cols]
                )
            )

        # ---------------------------------------------------------------------
        # MIN-MAX SCALING
        # ---------------------------------------------------------------------

        minmax_cols = [
            column
            for column in self.config.get(
                "minmax",
                []
            )
            if column in X.columns
        ]

        if minmax_cols:

            X[minmax_cols] = X[
                minmax_cols
            ].apply(
                pd.to_numeric,
                errors="coerce",
            )

            if fit:

                self.minmax_scaler.fit(
                    X[minmax_cols]
                )

                self.minmax_cols = (
                    minmax_cols
                )

            X[minmax_cols] = (
                self.minmax_scaler.transform(
                    X[minmax_cols]
                )
            )

        # ---------------------------------------------------------------------
        # BINARY ENCODING
        # ---------------------------------------------------------------------

        for column in self.config.get(
            "binary",
            []
        ):

            if column in X.columns:

                X[column] = _encode_binary(
                    X[column]
                )

        # ---------------------------------------------------------------------
        # CONTRACT ENCODING
        # ---------------------------------------------------------------------

        for column in self.config.get(
            "contract",
            []
        ):

            if column in X.columns:

                X[column] = _encode_contract(
                    X[column]
                )

        # ---------------------------------------------------------------------
        # ONE-HOT ENCODING
        # ---------------------------------------------------------------------

        for column in self.config.get(
            "ohe",
            []
        ):

            if column not in X.columns:
                continue

            if fit:

                categories = sorted(
                    X[column]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                self.ohe_categories[
                    column
                ] = categories

            else:

                categories = (
                    self.ohe_categories.get(
                        column,
                        []
                    )
                )

            # Preserve the original behavior:
            # first learned category is omitted.
            categories_to_create = (
                categories[1:]
            )

            for category in categories_to_create:

                dummy_name = (
                    f"{column}_{category}"
                )

                X[dummy_name] = (
                    X[column]
                    .astype(str)
                    .eq(category)
                    .astype(int)
                )

            X.drop(
                columns=[column],
                inplace=True,
            )

        # ---------------------------------------------------------------------
        # TERNARY SERVICE ENCODING
        # ---------------------------------------------------------------------

        for column in self.config.get(
            "ternary",
            []
        ):

            if column not in X.columns:
                continue

            encoded = (
                _encode_ternary_service(
                    X[column],
                    column,
                )
            )

            X = pd.concat(
                [
                    X.drop(
                        columns=[column]
                    ),
                    encoded,
                ],
                axis=1,
            )

        # ---------------------------------------------------------------------
        # ORDINAL BIN
        # ---------------------------------------------------------------------

        for column in self.config.get(
            "ordinal_bin",
            []
        ):

            if column not in X.columns:
                continue

            X[column] = (
                pd.to_numeric(
                    X[column],
                    errors="coerce",
                )
                .clip(upper=3)
            )

        # ---------------------------------------------------------------------
        # ENGINEERED FEATURES
        # ---------------------------------------------------------------------

        for column in self.config.get(
            "engineered",
            []
        ):

            if column in X.columns:

                X[column] = pd.to_numeric(
                    X[column],
                    errors="coerce",
                )

        # ---------------------------------------------------------------------
        # Convert remaining object columns
        # ---------------------------------------------------------------------

        for column in X.columns:

            if X[column].dtype == "object":

                X[column] = pd.to_numeric(
                    X[column],
                    errors="coerce",
                )

        return X

    # =========================================================================
    # TARGET EXTRACTION
    # =========================================================================

    def _extract_target(self, df):
        """
        Extract the churn target.

        Supports:
            Churn
            Exited
            Churn Label
        """

        possible_targets = [
            "Churn",
            "Exited",
            "Churn Label",
        ]

        target_column = next(
            (
                column
                for column in possible_targets
                if column in df.columns
            ),
            None,
        )

        if target_column is None:

            raise ValueError(
                "No target column found. "
                f"Expected one of: {possible_targets}"
            )

        y = df[target_column].copy()

        if y.dtype == object:

            y = (
                y.astype(str)
                .str.strip()
                .str.lower()
                .map(
                    {
                        "yes": 1,
                        "no": 0,
                        "true": 1,
                        "false": 0,
                        "1": 1,
                        "0": 0,
                    }
                )
            )

        y = pd.to_numeric(
            y,
            errors="coerce",
        )

        if y.isna().any():

            raise ValueError(
                f"Unable to encode all target values "
                f"for column '{target_column}'."
            )

        return y.astype(int).values

    # =========================================================================
    # CORRELATION FILTER
    # =========================================================================

    def _fit_corr_filter(
        self,
        X,
        y=None,
    ):
        """
        Learn highly correlated columns from the fitting population.
        """

        if (
            self.correlation_threshold
            is None
        ):

            self.correlation_columns_to_drop = []

            return X

        numeric_X = X.select_dtypes(
            include=[np.number]
        )

        if numeric_X.shape[1] <= 1:

            self.correlation_columns_to_drop = []

            return X

        corr_matrix = (
            numeric_X.corr().abs()
        )

        upper = corr_matrix.where(
            np.triu(
                np.ones(
                    corr_matrix.shape
                ),
                k=1,
            ).astype(bool)
        )

        to_drop = [
            column
            for column in upper.columns
            if any(
                upper[column]
                > self.correlation_threshold
            )
        ]

        self.correlation_columns_to_drop = (
            to_drop
        )

        return X.drop(
            columns=to_drop,
            errors="ignore",
        )

    def _apply_corr_filter(self, X):
        """
        Apply the correlation-filter decisions learned during fit().
        """

        if not self.correlation_columns_to_drop:

            return X

        return X.drop(
            columns=(
                self.correlation_columns_to_drop
            ),
            errors="ignore",
        )

    # =========================================================================
    # COLD-START FIT
    # =========================================================================

    def fit(self, df):
        """
        Fit the ColdStartFeatureEngineer.

        IMPORTANT:
        This method is intentionally separate from fit_transform().

        feature_engineering.py uses:

            cs_engineer.fit(train_non_cold)

        followed by:

            cs_engineer.transform(train_cold)
            cs_engineer.transform(val_cold)
            cs_engineer.transform(test_cold)

        Therefore this method learns ALL transformation parameters from
        non-cold-start TRAINING data without producing an MPMN training
        output from that non-cold population.
        """

        X = self._engineer_features(
            df
        )

        # Fit all encoders/scalers.
        X = self._apply_encoding(
            X,
            fit=True,
        )

        # Fit correlation filtering.
        X = self._fit_corr_filter(
            X
        )

        # Clean temporary numerical problems so the final feature
        # structure is known.
        X = X.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        X = X.fillna(0)

        # Store exact feature order.
        self.feature_names_out = (
            X.columns.tolist()
        )

        self.fitted = True

        return self

    # =========================================================================
    # COLD-START FIT + TRANSFORM
    # =========================================================================

    def fit_transform(self, df):
        """
        Fit and transform the supplied data.

        This method is retained for compatibility and convenience.

        For the actual MPMN pipeline, feature_engineering.py should use
        fit(non_cold_train) followed by transform(cold_data).
        """

        self.fit(df)

        X, y, feature_names = (
            self.transform(df)
        )

        return (
            X,
            y,
            feature_names,
        )

    # =========================================================================
    # COLD-START TRANSFORM
    # =========================================================================

    def transform(self, df):
        """
        Transform data using parameters learned by fit().
        """

        if not self.fitted:

            raise RuntimeError(
                "ColdStartFeatureEngineer must be fitted "
                "before transform() is called."
            )

        X = self._engineer_features(
            df
        )

        X = self._apply_encoding(
            X,
            fit=False,
        )

        X = self._apply_corr_filter(
            X
        )

        # ---------------------------------------------------------------------
        # Guarantee identical feature space and feature order.
        # ---------------------------------------------------------------------

        X = X.reindex(
            columns=self.feature_names_out,
            fill_value=0,
        )

        X = X.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        X = X.fillna(0)

        y = self._extract_target(
            df
        )

        return (
            X.values,
            y,
            self.feature_names_out,
        )


# =============================================================================
# ESTABLISHED / NON-COLD-START FEATURE ENGINEER
# =============================================================================

class EstablishedFeatureEngineer:
    """
    Feature engineering for the GATEFuse non-cold-start path.

    This class follows DATASET_CONFIG exactly.

    The following are intentionally retained where configured:
        - Bank Complain
        - Bank Satisfaction Score
        - Telco1 Satisfaction Score
        - Telco1 Latitude
        - Telco1 Longitude
        - Telco1 Population
        - Telco1 Total Revenue
    """

    def __init__(
        self,
        dataset_type,
        scalers_dir,
        corr_threshold=None,
    ):

        self.dataset_type = (
            dataset_type.lower()
        )

        self.scalers_dir = (
            scalers_dir
        )

        self.corr_threshold = (
            corr_threshold
        )

        self.fitted = False

        self._cfg = DATASET_CONFIG[
            self.dataset_type
        ]

        self.std_scaler = (
            StandardScaler()
        )

        self.ohe_columns = []

        self.dropped_cols = []

    # =========================================================================
    # PATHS
    # =========================================================================

    @property
    def scaler_path(self):

        return os.path.join(
            self.scalers_dir,
            self._cfg[
                "scalers_subdir"
            ],
            "std_scaler.pkl",
        )

    @property
    def corr_drop_path(self):

        return os.path.join(
            self.scalers_dir,
            self._cfg[
                "scalers_subdir"
            ],
            "corr_dropped_cols.pkl",
        )

    # =========================================================================
    # TARGET + X EXTRACTION
    # =========================================================================

    def _extract_Xy(self, df):
        """
        Extract X and y.

        Removes:
            - dataset ID columns
            - configured drop columns
            - target column

        In particular, is_cold_start is removed because it appears in
        drop_cols for all datasets.
        """

        cfg = self._cfg

        label_col = cfg[
            "label_col"
        ]

        if label_col not in df.columns:

            raise ValueError(
                f"Target column '{label_col}' "
                f"not found for dataset "
                f"'{self.dataset_type}'."
            )

        cols_to_drop = (
            cfg.get("id_cols", [])
            + cfg.get("drop_cols", [])
            + [label_col]
        )

        # Remove duplicate column names while preserving order.
        cols_to_drop = list(
            dict.fromkeys(
                cols_to_drop
            )
        )

        cols_to_drop = [
            column
            for column in cols_to_drop
            if column in df.columns
        ]

        # ---------------------------------------------------------------------
        # Target encoding
        # ---------------------------------------------------------------------

        y = (
            df[label_col]
            .map(
                {
                    "Yes": 1,
                    "No": 0,
                    1: 1,
                    0: 0,
                    "1": 1,
                    "0": 0,
                }
            )
        )

        # Handle lowercase/string variants if necessary.
        if y.isna().any():

            y = (
                df[label_col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(
                    {
                        "yes": 1,
                        "no": 0,
                        "true": 1,
                        "false": 0,
                        "1": 1,
                        "0": 0,
                    }
                )
            )

        if y.isna().any():

            raise ValueError(
                f"Could not encode all target values "
                f"for dataset '{self.dataset_type}'."
            )

        y = y.astype(int).values

        # ---------------------------------------------------------------------
        # Feature matrix
        # ---------------------------------------------------------------------

        X = df.drop(
            columns=cols_to_drop
        ).copy()

        return X, y

    # =========================================================================
    # ENCODING
    # =========================================================================

    def _encode(
        self,
        X,
        fit=False,
    ):
        """
        Apply the dataset-specific encoding pipeline.
        """

        cfg = self._cfg

        X = X.copy()

        # ---------------------------------------------------------------------
        # Fill missing values
        # ---------------------------------------------------------------------

        for column, fill_value in (
            cfg.get(
                "fillna",
                {}
            ).items()
        ):

            if column in X.columns:

                X[column] = (
                    X[column]
                    .fillna(fill_value)
                )

        # ---------------------------------------------------------------------
        # Bank ordinal columns
        # ---------------------------------------------------------------------

        for column, mapping in (
            cfg.get(
                "ordinal_cols",
                {}
            ).items()
        ):

            if column not in X.columns:
                continue

            values = (
                X[column]
                .astype(str)
                .str.upper()
            )

            X[column] = values.map(
                mapping
            )

        # ---------------------------------------------------------------------
        # Binary columns
        # ---------------------------------------------------------------------

        binary_cols = cfg.get(
            "binary_cols",
            {}
        )

        # Bank uses a dictionary:
        #
        # {
        #     "Gender": {...},
        #     "HasCrCard": {...},
        #     ...
        # }
        if isinstance(
            binary_cols,
            dict
        ):

            for column, mapping in (
                binary_cols.items()
            ):

                if column not in X.columns:
                    continue

                X[column] = X[
                    column
                ].map(mapping)

        # Telco datasets use a list and a shared binary_map.
        elif isinstance(
            binary_cols,
            list
        ):

            binary_map = cfg.get(
                "binary_map",
                {
                    "Yes": 1,
                    "No": 0,
                    "Male": 0,
                    "Female": 1,
                },
            )

            for column in binary_cols:

                if column not in X.columns:
                    continue

                X[column] = X[
                    column
                ].map(binary_map)

        # ---------------------------------------------------------------------
        # Integer columns
        # ---------------------------------------------------------------------

        for column in cfg.get(
            "int_cols",
            []
        ):

            if column not in X.columns:
                continue

            X[column] = pd.to_numeric(
                X[column],
                errors="coerce",
            )

        # ---------------------------------------------------------------------
        # One-hot encoding
        # ---------------------------------------------------------------------

        categorical_cols = [
            column
            for column in cfg.get(
                "categorical_cols",
                []
            )
            if column in X.columns
        ]

        if fit:

            if categorical_cols:

                X = pd.get_dummies(
                    X,
                    columns=categorical_cols,
                    drop_first=False,
                )

            self.ohe_columns = (
                X.columns.tolist()
            )

        else:

            if categorical_cols:

                X = pd.get_dummies(
                    X,
                    columns=categorical_cols,
                    drop_first=False,
                )

            # Guarantee the exact same columns as training.
            X = X.reindex(
                columns=self.ohe_columns,
                fill_value=0,
            )

        # ---------------------------------------------------------------------
        # Convert dummy columns to integers
        # ---------------------------------------------------------------------

        dummy_regex = cfg.get(
            "dummy_regex"
        )

        if dummy_regex:

            dummy_columns = X.filter(
                regex=dummy_regex
            ).columns

            for column in dummy_columns:

                X[column] = (
                    X[column]
                    .astype(int)
                )

        return X

    # =========================================================================
    # STANDARD SCALING
    # =========================================================================

    def _scale(
        self,
        X,
        fit=False,
    ):
        """
        Standard-scale the configured numerical columns.
        """

        cfg = self._cfg

        numerical_cols = [
            column
            for column in cfg.get(
                "std_numerical_cols",
                []
            )
            if column in X.columns
        ]

        if not numerical_cols:

            return X

        X[numerical_cols] = (
            X[numerical_cols]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
        )

        if fit:

            self.std_scaler.fit(
                X[numerical_cols]
            )

        else:

            if not self.fitted:

                raise RuntimeError(
                    "EstablishedFeatureEngineer "
                    "must be fitted before scaling."
                )

        X[numerical_cols] = (
            self.std_scaler.transform(
                X[numerical_cols]
            )
        )

        return X

    # =========================================================================
    # SAVE / LOAD SCALER
    # =========================================================================

    def _save_scaler(self):

        scaler_dir = os.path.dirname(
            self.scaler_path
        )

        os.makedirs(
            scaler_dir,
            exist_ok=True,
        )

        joblib.dump(
            self.std_scaler,
            self.scaler_path,
        )

    def _load_scaler(self):

        if not os.path.exists(
            self.scaler_path
        ):

            raise FileNotFoundError(
                f"Scaler file not found: "
                f"{self.scaler_path}"
            )

        self.std_scaler = (
            joblib.load(
                self.scaler_path
            )
        )

    # =========================================================================
    # CORRELATION FILTER
    # =========================================================================

    def _apply_corr_filter(
        self,
        X,
        y=None,
        fit=False,
    ):
        """
        Learn/apply correlation-based feature filtering.

        If corr_threshold is None, this stage does nothing.
        """

        if self.corr_threshold is None:

            return X

        if fit:

            numeric_X = (
                X.select_dtypes(
                    include=[np.number]
                )
            )

            if numeric_X.shape[1] <= 1:

                self.dropped_cols = []

            else:

                corr_matrix = (
                    numeric_X
                    .corr()
                    .abs()
                )

                upper = (
                    corr_matrix.where(
                        np.triu(
                            np.ones(
                                corr_matrix.shape
                            ),
                            k=1,
                        ).astype(bool)
                    )
                )

                self.dropped_cols = [
                    column
                    for column in upper.columns
                    if any(
                        upper[column]
                        > self.corr_threshold
                    )
                ]

            os.makedirs(
                os.path.dirname(
                    self.corr_drop_path
                ),
                exist_ok=True,
            )

            joblib.dump(
                self.dropped_cols,
                self.corr_drop_path,
            )

        else:

            if not self.dropped_cols:

                if os.path.exists(
                    self.corr_drop_path
                ):

                    self.dropped_cols = (
                        joblib.load(
                            self.corr_drop_path
                        )
                    )

        return X.drop(
            columns=self.dropped_cols,
            errors="ignore",
        )

    # =========================================================================
    # FIT + TRANSFORM
    # =========================================================================

    def fit_transform(self, df):
        """
        Fit the established/non-cold-start feature engineer and transform
        the training data.
        """

        X, y = self._extract_Xy(
            df
        )

        # Encode
        X = self._encode(
            X,
            fit=True,
        )

        # Scale
        X = self._scale(
            X,
            fit=True,
        )

        # Correlation filtering
        X = self._apply_corr_filter(
            X,
            y=y,
            fit=True,
        )

        # Numerical cleanup
        X = X.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        X = X.fillna(0)

        # Persist scaler
        self._save_scaler()

        # Store final feature columns.
        self.ohe_columns = (
            X.columns.tolist()
        )

        self.fitted = True

        feature_names = (
            X.columns.tolist()
        )

        return (
            X.values,
            y,
            feature_names,
        )

    # =========================================================================
    # TRANSFORM
    # =========================================================================

    def transform(self, df):
        """
        Transform validation/test data using the parameters learned from
        non-cold-start training data.
        """

        if not self.fitted:

            raise RuntimeError(
                "EstablishedFeatureEngineer "
                "must be fitted before transform()."
            )

        X, y = self._extract_Xy(
            df
        )

        # Encode using learned categorical structure.
        X = self._encode(
            X,
            fit=False,
        )

        # Scale using learned scaler.
        X = self._scale(
            X,
            fit=False,
        )

        # Apply learned correlation filter.
        X = self._apply_corr_filter(
            X,
            y=None,
            fit=False,
        )

        # Numerical cleanup.
        X = X.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        X = X.fillna(0)

        feature_names = (
            X.columns.tolist()
        )

        return (
            X.values,
            y,
            feature_names,
        )


# =============================================================================
# ESTABLISHED / NON-COLD-START DATASET CONFIGURATION
# =============================================================================

DATASET_CONFIG = {

    # =========================================================================
    # BANK
    # =========================================================================

    "bank": {

        "label_col": "Churn",

        "id_cols": [
            "RowNumber",
            "CustomerId",
            "Surname",
        ],

        "drop_cols": [
            "is_cold_start",
        ],

        "ordinal_cols": {

            "Card Type": {
                "SILVER": 1,
                "GOLD": 2,
                "PLATINUM": 3,
                "DIAMOND": 4,
            },

        },

        "binary_cols": {

            "Gender": {
                "Male": 0,
                "Female": 1,
            },

            "HasCrCard": {
                1: 1,
                0: 0,
                "1": 1,
                "0": 0,
            },

            "IsActiveMember": {
                1: 1,
                0: 0,
                "1": 1,
                "0": 0,
            },

            # -------------------------------------------------------------
            # KEEP COMPLAIN
            # -------------------------------------------------------------

            "Complain": {
                1: 1,
                0: 0,
                "1": 1,
                "0": 0,
            },

        },

        "int_cols": [
            "NumOfProducts",
        ],

        "fillna": {},

        "categorical_cols": [
            "Geography",
        ],

        "dummy_regex": "^Geography_",

        # -------------------------------------------------------------
        # KEEP SATISFACTION SCORE AND COMPLAIN
        # -------------------------------------------------------------

        "std_numerical_cols": [
            "CreditScore",
            "Age",
            "Tenure",
            "Balance",
            "EstimatedSalary",
            "Point Earned",
            "Satisfaction Score",
            "Complain",
        ],

        "scalers_subdir": "bank_scalers",
    },


    # =========================================================================
    # TELCO 1
    # =========================================================================

    "telco1": {

        "label_col": "Churn",

        "id_cols": [
            "Customer ID",
            "Customer Status",
            "Churn Score",
            "CLTV",
            "Churn Category",
            "Churn Reason",
            "Quarter",
            "City",
            "Zip Code",
            "Country",
            "State",
        ],

        "drop_cols": [
            "is_cold_start",
        ],

        "ordinal_cols": {},

        "binary_map": {
            "Yes": 1,
            "No": 0,
            "Male": 0,
            "Female": 1,
        },

        "binary_cols": [
            "Gender",
            "Married",
            "Under 30",
            "Senior Citizen",
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

        "fillna": {
            "Internet Type": "No Internet",
            "Offer": "No Offer",
        },

        "categorical_cols": [
            "Offer",
            "Internet Type",
            "Contract",
            "Payment Method",
        ],

        "dummy_regex": (
            "^(Offer_|Internet Type_|Contract_|Payment Method_)"
        ),

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
            "Total Revenue",
            "Satisfaction Score",
            "Latitude",
            "Longitude",
            "Population",
        ],

        "scalers_subdir": "telco1_scalers",
    },


    # =========================================================================
    # TELCO 2
    # =========================================================================

    "telco2": {

        "label_col": "Churn",

        "id_cols": [
            "customerID",
        ],

        "drop_cols": [
            "is_cold_start",
        ],

        "ordinal_cols": {},

        "binary_map": {
            "Yes": 1,
            "No": 0,
            "Male": 0,
            "Female": 1,
        },

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

        "std_numerical_cols": [
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
        ],

        "scalers_subdir": "telco2_scalers",
    },
}