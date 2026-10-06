import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class MinimalPreprocessor:
    """
    Phase 1: Minimal Preprocessing.

    This module performs ONLY preprocessing that is necessary before
    feature engineering and cold-start detection.

    Responsibilities:
        1. Remove duplicate customers/rows.
        2. Fix known data-type problems.
        3. Perform structural missing-value handling.
        4. Standardize the target column to 'Churn'.
        5. Fill missing values only in columns required for
           cold-start detection.

    IMPORTANT:
        No model feature selection is performed here.

        Decisions about:
            - leakage features
            - redundant features
            - geographical features
            - feature importance
            - feature encoding
            - scaling
            - engineered features
            - correlation filtering

        belong to feature_engineering_definitions.py.
    """

    def __init__(self, target_column="Churn", routing_columns=None):
        if routing_columns is None:
            routing_columns = ["tenure", "referrals", "Offer"]

        self.original_target_column = target_column
        self.routing_columns = routing_columns

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        print("=" * 70)
        print("MINIMAL PREPROCESSING")
        print("=" * 70)

        df_clean = df.copy()
        initial_rows = len(df_clean)

        # ── 1. Duplicate Removal ──────────────────────────────────────────────
        # Prefer Customer ID when available; otherwise remove exact duplicates.
        id_col = next(
            (
                c
                for c in df_clean.columns
                if "customer" in c.lower() and "id" in c.lower()
            ),
            None,
        )

        if id_col:
            n_dupe_ids = df_clean.duplicated(subset=[id_col]).sum()

            if n_dupe_ids > 0:
                print(
                    f"  Found {n_dupe_ids} duplicate Customer IDs. "
                    f"Keeping first occurrence."
                )

                df_clean = df_clean.drop_duplicates(
                    subset=[id_col],
                    keep="first",
                )
            else:
                print("✓ No duplicate Customer IDs found.")

        else:
            before = len(df_clean)

            df_clean = df_clean.drop_duplicates()

            removed = before - len(df_clean)

            if removed > 0:
                print(f"  Removed {removed} duplicate rows.")
            else:
                print("✓ No duplicate rows found.")

        if len(df_clean) < initial_rows:
            print(
                f"✓ Removed {initial_rows - len(df_clean)} duplicate rows."
            )

        # ── 2. Fix TotalCharges type ─────────────────────────────────────────
        # Telco 2 may store TotalCharges as a string.
        #
        # This is a data-type correction, not feature selection.
        if (
            "TotalCharges" in df_clean.columns
            and df_clean["TotalCharges"].dtype == object
        ):
            n_before = df_clean["TotalCharges"].isnull().sum()

            df_clean["TotalCharges"] = pd.to_numeric(
                df_clean["TotalCharges"],
                errors="coerce",
            )

            n_coerced = (
                df_clean["TotalCharges"].isnull().sum()
                - n_before
            )

            if n_coerced > 0:
                print(
                    f"✓ TotalCharges: coerced {n_coerced} "
                    f"unparseable values to NaN."
                )

            # For Telco 2, blank TotalCharges values correspond to
            # zero-tenure customers and are treated as zero.
            df_clean["TotalCharges"] = (
                df_clean["TotalCharges"].fillna(0.0)
            )

        # ── 3. Structural Imputation: Internet Type ──────────────────────────
        # When Internet Service = 'No', Internet Type may be structurally null.
        if (
            "Internet Type" in df_clean.columns
            and "Internet Service" in df_clean.columns
        ):
            mask = df_clean["Internet Type"].isnull()

            if mask.sum() > 0:
                df_clean.loc[
                    mask,
                    "Internet Type"
                ] = "No Internet Service"

                print(
                    f"✓ Internet Type: {mask.sum()} structural nulls "
                    f"→ 'No Internet Service'"
                )

        # ── 4. Standardize Target Variable ───────────────────────────────────
        if self.original_target_column in df_clean.columns:

            if self.original_target_column != "Churn":
                print(
                    f"✓ Renaming target "
                    f"'{self.original_target_column}' → 'Churn'"
                )

                df_clean = df_clean.rename(
                    columns={
                        self.original_target_column: "Churn"
                    }
                )

        # ── 5. Fill Routing-Critical Nulls ────────────────────────────────────
        # Only columns needed by the cold-start detector are handled here.
        #
        # All other missing values remain untouched and are handled later
        # by feature_engineering_definitions.py.
        print("\n✓ Checking routing-critical columns:")

        for col in self.routing_columns:

            match = next(
                (
                    c
                    for c in df_clean.columns
                    if c.lower() == col.lower()
                ),
                None,
            )

            if match is None:
                print(
                    f"  Note: '{col}' not found "
                    f"(detector will handle gracefully)"
                )
                continue

            missing = df_clean[match].isnull().sum()

            if missing == 0:
                continue

            if df_clean[match].dtype == object:
                df_clean[match] = df_clean[match].fillna("None")
                fill_val = "'None'"
            else:
                df_clean[match] = df_clean[match].fillna(0)
                fill_val = "0"

            print(
                f"  - {match}: {missing} nulls → {fill_val}"
            )

        print(
            f"\n✓ Minimal preprocessing complete:"
            f"\n  Rows    : {len(df_clean)}"
            f"\n  Columns : {len(df_clean.columns)}"
            f"\n  No model features were dropped."
        )

        return df_clean


class RobustColdStartDetector:
    """
    Identifies customers with insufficient interaction history.

    This class ONLY determines whether a customer is cold-start or
    non-cold-start. It does not perform feature selection.
    """

    def __init__(
        self,
        strategy: str = "generic",
        tenure_threshold: int = 2,
        tenure_col: str = "tenure",
        referral_col: str = "referrals",
        offer_col: str = "offer",
        contract_col: str = "Contract",
        total_charges_col: str = "TotalCharges",
        num_products_col: str = "NumOfProducts",
        is_active_col: str = "IsActiveMember",
    ):
        self.strategy = strategy
        self.tenure_threshold = tenure_threshold

        self.cols = {
            "tenure": tenure_col,
            "referrals": referral_col,
            "offer": offer_col,
            "contract": contract_col,
            "total_charges": total_charges_col,
            "num_products": num_products_col,
            "is_active": is_active_col,
        }

    def detect(self, df: pd.DataFrame) -> pd.Series:

        print("=" * 70)
        print(
            f"ROBUST COLD-START DETECTION "
            f"(Strategy: {self.strategy.upper()})"
        )
        print("=" * 70)

        cold_start_flags = []

        for _, row in df.iterrows():

            is_cold = False

            tenure_name = self.cols["tenure"]

            tenure_val = (
                row.get(tenure_name)
                if tenure_name in df.columns
                else np.nan
            )

            if pd.isna(tenure_val):
                tenure_val = 0

            # ── BANK ─────────────────────────────────────────────────────────
            if self.strategy == "bank":

                if tenure_val <= 1:
                    is_cold = True

                elif tenure_val <= 2:

                    if (
                        self.cols["num_products"] in df.columns
                        and self.cols["is_active"] in df.columns
                    ):

                        n_prod = row[
                            self.cols["num_products"]
                        ]

                        is_active = row[
                            self.cols["is_active"]
                        ]

                        if (
                            n_prod == 1
                            and is_active == 0
                        ):
                            is_cold = True

            # ── TELCO DEPTH ──────────────────────────────────────────────────
            elif self.strategy == "telco_depth":

                if tenure_val < 2:
                    is_cold = True

                elif tenure_val < 3:

                    if self.cols["contract"] in df.columns:

                        contract_value = str(
                            row[self.cols["contract"]]
                        ).lower()

                        if "month" in contract_value:
                            is_cold = True

                if (
                    not is_cold
                    and self.cols["total_charges"] in df.columns
                ):

                    try:
                        total_charges = float(
                            str(
                                row[
                                    self.cols["total_charges"]
                                ]
                            ).strip()
                            or 0
                        )

                        if total_charges == 0:
                            is_cold = True

                    except Exception:
                        pass

            # ── GENERIC ──────────────────────────────────────────────────────
            else:

                if tenure_val < self.tenure_threshold:
                    is_cold = True

                elif tenure_val < (
                    self.tenure_threshold + 1
                ):

                    if (
                        self.cols["referrals"] in df.columns
                        and self.cols["offer"] in df.columns
                    ):

                        refs = row[
                            self.cols["referrals"]
                        ]

                        offer = str(
                            row[self.cols["offer"]]
                        ).lower()

                        if (
                            refs == 0
                            and offer in [
                                "none",
                                "nan",
                                "",
                            ]
                        ):
                            is_cold = True

            cold_start_flags.append(
                1 if is_cold else 0
            )

        cold_series = pd.Series(
            cold_start_flags,
            index=df.index,
        )

        pct = (
            cold_series.sum()
            / len(df)
            * 100
            if len(df) > 0
            else 0
        )

        print(
            f"✓ Results: {cold_series.sum()} "
            f"Cold-Start ({pct:.1f}%)"
        )

        return cold_series


class DataRouter:
    """
    Attaches the is_cold_start flag and creates stratified
    train/validation/test splits.

    No rows are permanently removed based on cold-start status.
    Both populations remain in the same prepared dataset until
    feature_engineering.py separates them for their respective
    model paths.
    """

    def __init__(self, target_column="Churn"):
        # The preprocessing stage standardizes the target to Churn.
        self.target_column = "Churn"
        self.stats = {}

    def _get_churn_rate(
        self,
        df: pd.DataFrame
    ) -> float:

        if (
            len(df) == 0
            or self.target_column not in df.columns
        ):
            return 0.0

        vals = df[self.target_column]

        if vals.dtype == "object":
            vals = vals.map(
                {
                    "Yes": 1,
                    "No": 0,
                    "yes": 1,
                    "no": 0,
                    1: 1,
                    0: 0,
                }
            )

        return vals.mean() * 100

    def route(
        self,
        df: pd.DataFrame,
        cold_start_flags: pd.Series,
    ) -> pd.DataFrame:

        print("=" * 70)
        print(
            "DATA ROUTING "
            "(Flag-based — unified DataFrame)"
        )
        print("=" * 70)

        df_flagged = df.copy()

        df_flagged["is_cold_start"] = (
            cold_start_flags.values
        )

        n_cold = df_flagged[
            "is_cold_start"
        ].sum()

        n_non_cold = (
            len(df_flagged) - n_cold
        )

        cold_pct = (
            n_cold / len(df_flagged) * 100
            if len(df_flagged) > 0
            else 0
        )

        self.stats = {
            "total_samples": len(df_flagged),
            "cold_start_count": int(n_cold),
            "cold_start_percentage": cold_pct,
            "non_cold_start_count": int(n_non_cold),
            "cold_churn_rate": self._get_churn_rate(
                df_flagged[
                    df_flagged["is_cold_start"] == 1
                ]
            ),
            "non_cold_churn_rate": self._get_churn_rate(
                df_flagged[
                    df_flagged["is_cold_start"] == 0
                ]
            ),
        }

        print("✓ Routing Summary:")

        print(
            f"  - Total          : "
            f"{len(df_flagged)}"
        )

        print(
            f"  - Cold-Start     : "
            f"{n_cold} ({cold_pct:.1f}%)  "
            f"| Churn rate: "
            f"{self.stats['cold_churn_rate']:.1f}%"
        )

        print(
            f"  - Non-Cold-Start : "
            f"{n_non_cold} ({100 - cold_pct:.1f}%)  "
            f"| Churn rate: "
            f"{self.stats['non_cold_churn_rate']:.1f}%"
        )

        return df_flagged

    def split(
        self,
        df: pd.DataFrame,
        train_size: float = 0.70,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42,
    ) -> dict:

        assert abs(
            train_size + val_size + test_size - 1.0
        ) < 1e-6, (
            "train_size + val_size + test_size "
            "must sum to 1.0"
        )

        assert "is_cold_start" in df.columns, (
            "Call route() before split() — "
            "is_cold_start column is missing."
        )

        print("=" * 70)
        print(
            "STRATIFIED TRAIN / VAL / TEST SPLIT "
            "(70 / 15 / 15)"
        )
        print("=" * 70)

        churn_col = self.target_column

        strat_key = (
            df[churn_col].astype(str)
            + "_"
            + df["is_cold_start"].astype(str)
        )

        # ── Test split ────────────────────────────────────────────────────────
        df_trainval, df_test = train_test_split(
            df,
            test_size=test_size,
            stratify=strat_key,
            random_state=random_state,
        )

        # ── Train / validation split ──────────────────────────────────────────
        val_relative = (
            val_size
            / (train_size + val_size)
        )

        strat_key_trainval = (
            df_trainval[churn_col].astype(str)
            + "_"
            + df_trainval["is_cold_start"].astype(str)
        )

        df_train, df_val = train_test_split(
            df_trainval,
            test_size=val_relative,
            stratify=strat_key_trainval,
            random_state=random_state,
        )

        # ── Reporting ─────────────────────────────────────────────────────────
        def _report(name, d):

            n_cold = d[
                "is_cold_start"
            ].sum()

            cr_cold = self._get_churn_rate(
                d[
                    d["is_cold_start"] == 1
                ]
            )

            cr_warm = self._get_churn_rate(
                d[
                    d["is_cold_start"] == 0
                ]
            )

            cold_percentage = (
                n_cold / len(d) * 100
                if len(d) > 0
                else 0
            )

            print(
                f"  {name:<8}: "
                f"{len(d):>5} rows  | "
                f"Cold: {n_cold:>4} "
                f"({cold_percentage:.1f}%)  | "
                f"Churn (cold): {cr_cold:.1f}%  | "
                f"Churn (warm): {cr_warm:.1f}%"
            )

        _report("Train", df_train)
        _report("Val", df_val)
        _report("Test", df_test)

        return {
            "train": df_train,
            "val": df_val,
            "test": df_test,
        }

    def get_routing_summary(self) -> dict:
        return self.stats
