"""
adasyn_balance.py — ADASYN Balancing for Non-Cold-Start Training Data

Applies ADASYN to the training CSV produced by feature engineering,
replicating the balancing strategy of Liu et al. (2024).

ADASYN differs from plain SMOTE in that it generates more synthetic samples
near the decision boundary — instances that are harder to classify receive
more synthetic neighbours, making the model focus on difficult cases.

ADASYN is applied ONLY to the training data. Val and test CSVs are copied
to the output folder unchanged with their original imbalanced distributions.

Order in pipeline:
    feature_engineering.py
        → adasyn_balance.py        (this file)
            → non_cold_start_train.py
                → non_cold_start_test.py

Input:
    {INPUT_DIR}/{dataset}/gatefuse_ready/{input_subset}/train.csv
    {INPUT_DIR}/{dataset}/gatefuse_ready/{input_subset}/val.csv
    {INPUT_DIR}/{dataset}/gatefuse_ready/{input_subset}/test.csv

Output:
    {OUTPUT_DIR}/{dataset}/gatefuse_ready/{output_subset}/train.csv  ← ADASYN balanced
    {OUTPUT_DIR}/{dataset}/gatefuse_ready/{output_subset}/val.csv    ← unchanged
    {OUTPUT_DIR}/{dataset}/gatefuse_ready/{output_subset}/test.csv   ← unchanged

Usage:
    Set INPUT_SUBSET, OUTPUT_SUBSET, and BASE in __main__, then run:
        python adasyn_balance.py
"""

import os
import sys
import shutil
import pandas as pd
from collections import Counter
from imblearn.over_sampling import ADASYN


# Columns that are not features
DROP_COLS = ["Churn", "is_cold_start"]

DATASETS = ["bank", "telco1", "telco2"]


def apply_adasyn(
    input_dir:     str,
    output_dir:    str,
    dataset:       str,
    input_subset:  str,
    output_subset: str,
    n_neighbors:   int = 5,
    random_state:  int = 42,
):
    """
    Applies ADASYN to the training split of a single dataset.

    Parameters
    ----------
    input_dir     : root processed directory (e.g. datasets/processed)
    output_dir    : root output directory (can be same as input_dir)
    dataset       : "bank", "telco1", or "telco2"
    input_subset  : subfolder inside gatefuse_ready/ to read from
                    e.g. "baseline"
    output_subset : subfolder inside gatefuse_ready/ to write to
                    e.g. "experiment_set2/adasyn"
    n_neighbors   : number of nearest neighbors for ADASYN (default 5,
                    matching Liu et al. 2024)
    random_state  : random seed for reproducibility

    Key difference from SMOTE:
        SMOTE generates the same number of synthetic samples for every
        minority instance. ADASYN generates more synthetic samples for
        minority instances that are harder to classify (those surrounded
        by more majority class neighbours), focusing the model's learning
        on the difficult boundary region.
    """
    print(f"\n{'=' * 60}")
    print(f"  ADASYN BALANCING — {dataset.upper()}")
    print(f"  Input  : {input_subset}")
    print(f"  Output : {output_subset}")
    print(f"  k      : {n_neighbors}")
    print(f"{'=' * 60}\n")

    input_base  = os.path.join(input_dir, input_subset, dataset, "gatefuse_ready")
    output_base = os.path.join(output_dir,  output_subset, dataset, "gatefuse_ready")
    os.makedirs(output_base, exist_ok=True)

    # ── Load training data ────────────────────────────────────────────────────
    train_path = os.path.join(input_base, "train.csv")
    df_train   = pd.read_csv(train_path)

    drop = [c for c in DROP_COLS if c in df_train.columns]
    X    = df_train.drop(columns=drop)
    y    = df_train["Churn"]

    print(f"[ADASYN] Original training distribution:")
    print(f"   {Counter(y.values)}")
    print(f"   Churn rate: {y.mean() * 100:.1f}%")
    print(f"   Shape     : {X.shape}\n")

    # ── Apply ADASYN ──────────────────────────────────────────────────────────
    # Unlike SMOTE which targets exact 50/50 balance, ADASYN targets an
    # approximate balance determined by its sampling_strategy parameter
    # (default 1.0 = full balance). More synthetic samples are generated
    # near the decision boundary where misclassification is more likely.
    adasyn = ADASYN(n_neighbors=n_neighbors, random_state=random_state)

    try:
        X_balanced, y_balanced = adasyn.fit_resample(X, y)
    except ValueError as e:
        print(f"[ADASYN] WARNING: {e}")
        print(f"[ADASYN] Falling back to n_neighbors=3 for {dataset}...")
        adasyn = ADASYN(n_neighbors=3, random_state=random_state)
        X_balanced, y_balanced = adasyn.fit_resample(X, y)

    print(f"[ADASYN] Balanced training distribution:")
    print(f"   {Counter(y_balanced)}")
    print(f"   Churn rate: {y_balanced.mean() * 100:.1f}%")
    print(f"   Shape     : {X_balanced.shape}\n")

    # ── Save balanced training CSV ────────────────────────────────────────────
    df_balanced = pd.DataFrame(X_balanced, columns=X.columns)
    df_balanced["Churn"] = y_balanced.values

    train_out = os.path.join(output_base, "train.csv")
    df_balanced.to_csv(train_out, index=False)
    print(f"[ADASYN] Saved balanced train.csv → {train_out}")

    # ── Copy val and test unchanged ───────────────────────────────────────────
    for split in ("val", "test"):
        src = os.path.join(input_base, f"{split}.csv")
        dst = os.path.join(output_base, f"{split}.csv")
        shutil.copy2(src, dst)
        df_split = pd.read_csv(dst)
        print(
            f"[ADASYN] Copied {split}.csv → {dst}  "
            f"(shape: {df_split.shape}, "
            f"churn rate: {df_split['Churn'].mean() * 100:.1f}%)"
        )

    print(f"\n[ADASYN] {dataset.upper()} complete.\n")


if __name__ == "__main__":
    BASE = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
        )
    )

    # ── Change these to switch experiments ────────────────────────────────────
    INPUT_SUBSET  = "baseline"               # folder to read from
    OUTPUT_SUBSET = "adasyn" # folder to write to
    # ─────────────────────────────────────────────────────────────────────────

    INPUT_DIR  = os.path.join(BASE, "datasets", "processed")
    OUTPUT_DIR = os.path.join(BASE, "datasets", "processed")

    for dataset in DATASETS:
        apply_adasyn(
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR,
            dataset=dataset,
            input_subset=INPUT_SUBSET,
            output_subset=OUTPUT_SUBSET,
            n_neighbors=5,    # replicating Liu et al. (2024)
            random_state=42,
        )

    print("=" * 60)
    print("  ALL DATASETS COMPLETE")
    print("=" * 60)