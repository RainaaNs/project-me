import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class MinimalPreprocessor:
    """
    Phase 1: Minimal Preprocessing & Cleaning.

    Addressed Checklist Items:
    1. Duplicates: Checks and removes duplicate Customer IDs.
    2. Feature Selection: Removes known LEAKAGE and REDUNDANT columns.
    3. Missing Values: Handles routing-critical columns + structural imputation.
    4. Type Fixes: Coerces TotalCharges (Telco 2) from string to numeric.

    Changes from v1:
    - [CRITICAL] Added 'Complain' to leakage_columns (near-deterministic of churn in Bank).
    - [CRITICAL] Added Telco 1 redundant binary columns to drop list:
        'Under 30', 'Senior Citizen', 'Referred a Friend'
    - [CRITICAL] Added post-hoc Telco 1 columns: 'Churn Score', 'CLTV', 'Total Revenue'
    - [MEDIUM]   Added geo noise columns for Telco 1: 'Country', 'State', 'City',
                 'Zip Code', 'Population'
    - [MEDIUM]   Structural imputation: 'Internet Type' null → 'No Internet Service'
    - [MEDIUM]   TotalCharges string → numeric coercion for Telco 2

    Note: Encoding and Scaling are reserved for Phase 2 (Feature Engineering)
    to allow model-specific transformation strategies.
    """

    def __init__(self, target_column="Churn", routing_columns=None):
        if routing_columns is None:
            routing_columns = ["tenure", "referrals", "Offer"]
        self.original_target_column = target_column
        self.routing_columns = routing_columns

        # ── LEAKAGE FEATURES ─────────────────────────────────────────────────
        # Columns that directly reveal the outcome or are computed post-churn.
        self.leakage_columns = [
            "Churn Category",  # Post-hoc: explains WHY they churned
            "Churn Reason",  # Post-hoc: explains WHY they churned
            "Churn Score",  # Post-hoc: model output, not a predictor
            "CLTV",  # Post-hoc: computed using churn outcome
            "Quarter",  # Data collection artefact
            "Customer Status",  # Directly encodes churn label
            "Lat Long",  # Raw geo string
            "Latitude",  # Geo noise (not predictive at customer level)
            "Longitude",  # Geo noise (not predictive at customer level)
            # ── Bank: near-deterministic leakage ────────────────────────────
            # 99.5% of complainers churn, 99.9% of non-complainers don't.
            # This column is recorded AFTER the churn decision is made.
            "Complain",
        ]

        # ── REDUNDANT FEATURES (Telco 1 specific) ────────────────────────────
        # These are 100% derivable from other columns already in the dataset.
        # Keeping both inflates the feature space with duplicate signal and
        # biases distance metrics in the MPMN embedding space.
        self.redundant_columns = [
            "Under 30",  # = (Age < 30) — exact match
            "Senior Citizen",  # = (Age >= 65) — exact match
            # "Dependents",      # = (Number of Dependents > 0) — kept, used in Telco_2
            # "Referred a Friend", # = (Number of Referrals > 0) — exact match
            "Total Revenue",  # ≈ Total Charges − Total Refunds (r=0.97)
        ]

        # ── GEO / ADMIN NOISE (Telco 1 specific) ─────────────────────────────
        # City/State/Zip are too high-cardinality to encode usefully without
        # a dedicated geo-embedding. Population is a noisy proxy.
        self.geo_columns = [
            "Country",
            "State",
            "City",
            "Zip Code",
            "Population",
        ]

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        print("=" * 70)
        print("MINIMAL PREPROCESSING (Cleaning & Leakage Removal)")
        print("=" * 70)

        df_clean = df.copy()
        initial_rows = len(df_clean)

        # ── 1. Standardize ID Column ──────────────────────────────────────────
        id_col = next(
            (
                c
                for c in df_clean.columns
                if "customer" in c.lower() and "id" in c.lower()
            ),
            None,
        )

        # ── 2. Duplicate Removal ──────────────────────────────────────────────
        if id_col:
            n_dupe_ids = df_clean.duplicated(subset=[id_col]).sum()
            if n_dupe_ids > 0:
                print(
                    f"  Found {n_dupe_ids} duplicate Customer IDs! Keeping first occurrence."
                )
                df_clean = df_clean.drop_duplicates(subset=[id_col], keep="first")
        else:
            df_clean = df_clean.drop_duplicates()

        if len(df_clean) < initial_rows:
            print(f"✓ Removed {initial_rows - len(df_clean)} duplicates.")
        else:
            print("✓ No duplicates found.")

        # ── 3. Fix TotalCharges type (Telco 2: stored as string with spaces) ──
        if (
            "TotalCharges" in df_clean.columns
            and df_clean["TotalCharges"].dtype == object
        ):
            n_before = df_clean["TotalCharges"].isnull().sum()
            df_clean["TotalCharges"] = pd.to_numeric(
                df_clean["TotalCharges"], errors="coerce"
            )
            n_coerced = df_clean["TotalCharges"].isnull().sum() - n_before
            if n_coerced > 0:
                print(
                    f"\n✓ TotalCharges: coerced {n_coerced} unparseable strings → NaN "
                    f"(these are tenure=0 rows, will be filled with 0.0)"
                )
            df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0.0)

        # ── 4. Structural Imputation: Internet Type ───────────────────────────
        if (
            "Internet Type" in df_clean.columns
            and "Internet Service" in df_clean.columns
        ):
            mask = df_clean["Internet Type"].isnull()
            if mask.sum() > 0:
                df_clean.loc[mask, "Internet Type"] = "No Internet Service"
                print(
                    f"\n✓ Internet Type: {mask.sum()} structural nulls → 'No Internet Service'"
                )

        # ── 5. Drop Leakage Columns ───────────────────────────────────────────
        print("\n✓ Removing Leakage Columns:")
        cols_to_drop = [c for c in self.leakage_columns if c in df_clean.columns]
        if cols_to_drop:
            df_clean = df_clean.drop(columns=cols_to_drop)
            print(f"  - Dropped: {cols_to_drop}")
        else:
            print("  - No leakage columns found (Clean).")

        # ── 6. Drop Redundant Columns ─────────────────────────────────────────
        print("\n✓ Removing Redundant Columns:")
        redundant_to_drop = [c for c in self.redundant_columns if c in df_clean.columns]
        if redundant_to_drop:
            df_clean = df_clean.drop(columns=redundant_to_drop)
            print(f"  - Dropped: {redundant_to_drop}")
        else:
            print("  - No redundant columns found.")

        # ── 7. Drop Geo / Admin Noise ─────────────────────────────────────────
        print("\n✓ Removing Geo/Admin Noise Columns:")
        geo_to_drop = [c for c in self.geo_columns if c in df_clean.columns]
        if geo_to_drop:
            df_clean = df_clean.drop(columns=geo_to_drop)
            print(f"  - Dropped: {geo_to_drop}")
        else:
            print("  - No geo columns found.")

        # ── 8. Standardize Target Variable to 'Churn' ─────────────────────────
        if self.original_target_column in df_clean.columns:
            if self.original_target_column != "Churn":
                print(f"\n✓ Renaming target '{self.original_target_column}' → 'Churn'")
                df_clean = df_clean.rename(
                    columns={self.original_target_column: "Churn"}
                )

        # ── 9. Handle Missing Values in Routing Columns ───────────────────────
        print("\n✓ Handling missing values in routing columns:")
        for col in self.routing_columns:
            match = next(
                (c for c in df_clean.columns if c.lower() == col.lower()), None
            )
            if match:
                missing = df_clean[match].isnull().sum()
                if missing > 0:
                    if df_clean[match].dtype == "object":
                        df_clean[match] = df_clean[match].fillna("None")
                        fill_val = "'None'"
                    else:
                        df_clean[match] = df_clean[match].fillna(0)
                        fill_val = "0"
                    print(f"  - {match}: {missing} missing → filled with {fill_val}")
            else:
                print(f"  Note: '{col}' column not found (Detector will handle this)")

        print(
            f"\n✓ Cleaning complete: {len(df_clean)} samples, {len(df_clean.columns)} columns ready for routing"
        )
        return df_clean


class RobustColdStartDetector:
    """
    Robustly identifies customers with insufficient interaction history.
    No changes from v1 — detection logic is correct.
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
        print(f"ROBUST COLD-START DETECTION (Strategy: {self.strategy.upper()})")
        print("=" * 70)

        cold_start_flags = []

        for idx, row in df.iterrows():
            is_cold = False

            tenure_name = self.cols["tenure"]
            tenure_val = row.get(tenure_name) if tenure_name in df.columns else np.nan
            if pd.isna(tenure_val):
                tenure_val = 0

            # --- BANK STRATEGY ---
            if self.strategy == "bank":
                if tenure_val <= 1:
                    is_cold = True
                elif tenure_val <= 2:
                    if (
                        self.cols["num_products"] in df.columns
                        and self.cols["is_active"] in df.columns
                    ):
                        n_prod = row[self.cols["num_products"]]
                        is_active = row[self.cols["is_active"]]
                        if (n_prod == 1) and (is_active == 0):
                            is_cold = True

            # --- TELCO DEPTH STRATEGY ---
            elif self.strategy == "telco_depth":
                if tenure_val < 2:
                    is_cold = True
                elif tenure_val < 3:
                    if self.cols["contract"] in df.columns:
                        if "month" in str(row[self.cols["contract"]]).lower():
                            is_cold = True
                if not is_cold and self.cols["total_charges"] in df.columns:
                    try:
                        if (
                            float(str(row[self.cols["total_charges"]]).strip() or 0)
                            == 0
                        ):
                            is_cold = True
                    except Exception:
                        pass

            # --- GENERIC STRATEGY ---
            else:
                if tenure_val < self.tenure_threshold:
                    is_cold = True
                elif tenure_val < (self.tenure_threshold + 1):
                    if (
                        self.cols["referrals"] in df.columns
                        and self.cols["offer"] in df.columns
                    ):
                        refs = row[self.cols["referrals"]]
                        offer = str(row[self.cols["offer"]]).lower()
                        if (refs == 0) and (offer in ["none", "nan", ""]):
                            is_cold = True

            cold_start_flags.append(1 if is_cold else 0)

        cold_series = pd.Series(cold_start_flags, index=df.index)
        pct = (cold_series.sum() / len(df)) * 100
        print(f"✓ Results: {cold_series.sum()} Cold-Start ({pct:.1f}%)")
        return cold_series


class DataRouter:
    """
    Flags customers as cold-start / non-cold-start and returns a SINGLE unified
    DataFrame with an 'is_cold_start' column.

    Changes from v1:
    - [HIGH] route() now returns one unified DataFrame with is_cold_start column
      instead of two separate DataFrames. This preserves a shared feature space
      for transfer learning and makes train/val/test splitting cleaner.
    - [HIGH] Added split() method: stratified 70/15/15 train/val/test split,
      stratified on both Churn label AND is_cold_start flag to preserve
      cold-start proportions and churn rates across all three folds.
    """

    def __init__(self, target_column="Churn"):
        self.target_column = "Churn"
        self.stats = {}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_churn_rate(self, df: pd.DataFrame) -> float:
        if len(df) == 0 or self.target_column not in df.columns:
            return 0.0
        vals = df[self.target_column]
        if vals.dtype == "object":
            vals = vals.map({"Yes": 1, "No": 0, "yes": 1, "no": 0, 1: 1, 0: 0})
        return vals.mean() * 100

    # ── Main Methods ──────────────────────────────────────────────────────────

    def route(self, df: pd.DataFrame, cold_start_flags: pd.Series) -> pd.DataFrame:
        """
        Attaches is_cold_start flag to the DataFrame and returns it whole.
        No rows are dropped or separated.
        """
        print("=" * 70)
        print("DATA ROUTING  (Flag-based — unified DataFrame)")
        print("=" * 70)

        df_flagged = df.copy()
        df_flagged["is_cold_start"] = cold_start_flags.values

        n_cold = df_flagged["is_cold_start"].sum()
        n_non_cold = len(df_flagged) - n_cold
        cold_pct = (n_cold / len(df_flagged)) * 100

        self.stats = {
            "total_samples": len(df_flagged),
            "cold_start_count": int(n_cold),
            "cold_start_percentage": cold_pct,
            "non_cold_start_count": int(n_non_cold),
            "cold_churn_rate": self._get_churn_rate(
                df_flagged[df_flagged["is_cold_start"] == 1]
            ),
            "non_cold_churn_rate": self._get_churn_rate(
                df_flagged[df_flagged["is_cold_start"] == 0]
            ),
        }

        print("✓ Routing Summary:")
        print(f"  - Total          : {len(df_flagged)}")
        print(
            f"  - Cold-Start     : {n_cold} ({cold_pct:.1f}%)  "
            f"| Churn rate: {self.stats['cold_churn_rate']:.1f}%"
        )
        print(
            f"  - Non-Cold-Start : {n_non_cold} ({100 - cold_pct:.1f}%)  "
            f"| Churn rate: {self.stats['non_cold_churn_rate']:.1f}%"
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
        """
        Stratified 70/15/15 train/val/test split.

        Stratification key = Churn + is_cold_start combined, so that all three
        folds preserve:
          - The overall churn rate
          - The cold-start proportion
          - The churn rate within cold-start users

        Returns a dict with keys: 'train', 'val', 'test'
        Each value is a DataFrame containing all original columns + is_cold_start.
        """
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, (
            "train_size + val_size + test_size must sum to 1.0"
        )
        assert "is_cold_start" in df.columns, (
            "Call route() before split() — is_cold_start column is missing."
        )

        print("=" * 70)
        print("STRATIFIED TRAIN / VAL / TEST SPLIT  (70 / 15 / 15)")
        print("=" * 70)

        churn_col = self.target_column
        strat_key = df[churn_col].astype(str) + "_" + df["is_cold_start"].astype(str)

        df_trainval, df_test = train_test_split(
            df,
            test_size=test_size,
            stratify=strat_key,
            random_state=random_state,
        )

        val_relative = val_size / (train_size + val_size)
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

        def _report(name, d):
            n_c = d["is_cold_start"].sum()
            cr_cold = self._get_churn_rate(d[d["is_cold_start"] == 1])
            cr_warm = self._get_churn_rate(d[d["is_cold_start"] == 0])
            print(
                f"  {name:<8}: {len(d):>5} rows  | "
                f"Cold: {n_c:>4} ({n_c / len(d) * 100:.1f}%)  | "
                f"Churn (cold): {cr_cold:.1f}%  | "
                f"Churn (warm): {cr_warm:.1f}%"
            )

        _report("Train", df_train)
        _report("Val", df_val)
        _report("Test", df_test)

        return {"train": df_train, "val": df_val, "test": df_test}

    def get_routing_summary(self) -> dict:
        return self.stats
