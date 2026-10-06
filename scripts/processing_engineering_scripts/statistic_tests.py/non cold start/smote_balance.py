"""
smote_balance.py — SMOTE Balancing for Non-Cold-Start Training Data

Applies SMOTE (k=5 neighbors) to the training CSV produced by feature
engineering, replicating the balancing strategy of El Attar & El-Hajj (2026).

SMOTE is applied ONLY to the training data. Val and test CSVs are copied
to the output folder unchanged with their original imbalanced distributions.

Order in pipeline:
    feature_engineering.py
        → smote_balance.py        (this file)
            → non_cold_start_train.py
                → non_cold_start_test.py

Usage:
    Set INPUT_SUBSET, OUTPUT_SUBSET, and BASE in __main__, then run:
        python smote_balance.py
"""

import os
import sys
import shutil
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE

# FOR IMPORTING MODELS, SUPPOSED TO BE 4 LEVELS UP
# Add project root to path so 'models' module can be found
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.abspath(os.path.abspath(__file__))))))


# Columns that are not features
DROP_COLS = ["Churn", "is_cold_start"]

DATASETS = ["bank", "telco1", "telco2"]


def apply_smote(
    input_dir:     str,
    output_dir:    str,
    dataset:       str,
    input_subset:  str,
    output_subset: str,
    k_neighbors:   int = 5,
    random_state:  int = 42,
    sampling_strategy=0.75,
):
    
    """
    Applies SMOTE to the training split of a single dataset.


    Parameters
    ----------
    input_dir     : root processed directory (e.g. datasets/processed)
    output_dir    : root output directory (can be same as input_dir)
    dataset       : "bank", "telco1", or "telco2"
    input_subset  : subfolder inside gatefuse_ready/ to read from
                    e.g. "baseline"
    output_subset : subfolder inside gatefuse_ready/ to write to
                    e.g. "experiment_set1/smote"
    k_neighbors   : number of nearest neighbors for SMOTE (default 5,
                    matching El Attar & El-Hajj 2026)
    random_state  : random seed for reproducibility
    sampling_strategy="auto",
    sampling_strategy=0.5      # Minority becomes 50% of the majority
    sampling_strategy=0.75     # Minority becomes 75% of the majority
    sampling_strategy=1.0 
    """
    print(f"\n{'=' * 60}")
    print(f"  SMOTE BALANCING — {dataset.upper()}")
    print(f"  Input  : {input_subset}")
    print(f"  Output : {output_subset}")
    print(f"  k      : {k_neighbors}")
    print(f"{'=' * 60}\n")

    input_base  = os.path.join(input_dir, input_subset,  dataset, "gatefuse_ready" )
    output_base = os.path.join(output_dir, output_subset, dataset, "gatefuse_ready")
    os.makedirs(output_base, exist_ok=True)

    # ── Load training data ────────────────────────────────────────────────────
    train_path = os.path.join(input_base, "train.csv")
    df_train   = pd.read_csv(train_path)

    drop  = [c for c in DROP_COLS if c in df_train.columns]
    X     = df_train.drop(columns=drop)
    y     = df_train["Churn"]

    print(f"[SMOTE] Original training distribution:")
    print(f"   {Counter(y.values)}")
    print(f"   Churn rate: {y.mean() * 100:.1f}%")
    print(f"   Shape     : {X.shape}\n")

    # ── Apply SMOTE ───────────────────────────────────────────────────────────
    smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state, sampling_strategy=sampling_strategy,)
    X_balanced, y_balanced = smote.fit_resample(X, y)

    print(f"[SMOTE] Balanced training distribution:")
    print(f"   {Counter(y_balanced)}")
    print(f"   Churn rate: {y_balanced.mean() * 100:.1f}%")
    print(f"   Shape     : {X_balanced.shape}\n")

    # ── Save balanced training CSV ────────────────────────────────────────────
    df_balanced = pd.DataFrame(X_balanced, columns=X.columns)
    df_balanced["Churn"] = y_balanced.values

    train_out = os.path.join(output_base, "train.csv")
    df_balanced.to_csv(train_out, index=False)
    print(f"[SMOTE] Saved balanced train.csv → {train_out}")

    # ── Copy val and test unchanged ───────────────────────────────────────────
    for split in ("val", "test"):
        src = os.path.join(input_base, f"{split}.csv")
        dst = os.path.join(output_base, f"{split}.csv")
        shutil.copy2(src, dst)
        df_split = pd.read_csv(dst)
        print(
            f"[SMOTE] Copied {split}.csv → {dst}  "
            f"(shape: {df_split.shape}, "
            f"churn rate: {df_split['Churn'].mean() * 100:.1f}%)"
        )

    print(f"\n[SMOTE] {dataset.upper()} complete.\n")


if __name__ == "__main__":

    # ── Change these to switch experiments ────────────────────────────────────
    INPUT_SUBSET  = "baseline"              # folder to read from
    OUTPUT_SUBSET = "smote_variation_0.75" # folder to write to
    # ─────────────────────────────────────────────────────────────────────────

    INPUT_DIR  = os.path.join("datasets", "processed")
    OUTPUT_DIR = os.path.join("datasets", "processed")

    for dataset in DATASETS:
        apply_smote(
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR,
            dataset=dataset,
            input_subset=INPUT_SUBSET,
            output_subset=OUTPUT_SUBSET,
            k_neighbors=5,    # replicating El Attar & El-Hajj (2026)
            random_state=42,
            sampling_strategy=0.75,
        )

    print("=" * 60)
    print("  ALL DATASETS COMPLETE")
    print("=" * 60)