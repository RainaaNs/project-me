"""
feature_engineering.py — Step 4: Feature Engineering (Both Paths)

Takes the train/val/test CSVs from prepare_data.py and produces
model-ready outputs for both paths:

    4a. Cold-Start path (MPMN)
        - Fits ColdStartFeatureEngineer on non-cold-start TRAINING data
        - Transforms cold-start train/val/test rows
        - Saves numpy arrays as .npz files → data/mpmn_ready/

    4b. Non-Cold-Start path (GATEFuse)
        - Fits NonColdStartFeatureEngineer on non-cold-start TRAINING data
        - Transforms non-cold-start train/val/test rows
        - Saves CSVs with named columns → data/gatefuse_ready/
        - Saves groups.json (feature group → column index mapping)

Why 4a before 4b (or together):
    Both engineers fit on the same non-cold-start training data.
    The cold-start engineer specifically uses non-cold scaler parameters
    to transform cold-start users (transfer learning).
    Running them in one script avoids fitting twice.

Usage:
    python feature_engineering.py --prepared-dir data/prepared/bank
                                  --dataset bank
                                  --out-dir data/processed/bank

    python feature_engineering.py --prepared-dir data/prepared/telco1
                                  --dataset telco1
                                  --out-dir data/processed/telco1

    python feature_engineering.py --prepared-dir data/prepared/telco2
                                  --dataset telco2
                                  --out-dir data/processed/telco2

Input (from prepare_data.py):
    {prepared-dir}/train.csv   — full training split with is_cold_start
    {prepared-dir}/val.csv     — full val split with is_cold_start
    {prepared-dir}/test.csv    — full test split with is_cold_start

Output:
    {out-dir}/mpmn_ready/train.npz       — cold-start train arrays (X, y)
    {out-dir}/mpmn_ready/val.npz         — cold-start val arrays
    {out-dir}/mpmn_ready/test.npz        — cold-start test arrays
    {out-dir}/gatefuse_ready/train.csv   — non-cold-start train features
    {out-dir}/gatefuse_ready/val.csv     — non-cold-start val features
    {out-dir}/gatefuse_ready/test.csv    — non-cold-start test features
    {out-dir}/gatefuse_ready/groups.json — feature group → index mapping
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from feature_engineering_defintions import ColdStartFeatureEngineer, NonColdStartFeatureEngineer


# ── Dataset type mapping ──────────────────────────────────────────────────────
# ColdStartFeatureEngineer and NonColdStartFeatureEngineer use dataset_type
# strings internally to select their per-dataset configs.
DATASET_TYPE_MAP = {
    "telco1": "telco1",
    "telco2": "telco2",
    "bank":    "bank",
}
# ─────────────────────────────────────────────────────────────────────────────


def run(prepared_dir: str, dataset: str, out_dir: str):
    if dataset not in DATASET_TYPE_MAP:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Must be one of: {list(DATASET_TYPE_MAP.keys())}"
        )

    dataset_type = DATASET_TYPE_MAP[dataset]

    print(f"\n{'='*70}")
    print(f"  FEATURE ENGINEERING — {dataset}")
    print(f"{'='*70}\n")

    # ── Load prepared splits ──────────────────────────────────────────────────
    df_train = pd.read_csv(os.path.join(prepared_dir, "train.csv"))
    df_val   = pd.read_csv(os.path.join(prepared_dir, "val.csv"))
    df_test  = pd.read_csv(os.path.join(prepared_dir, "test.csv"))

    print(f"[FE] Loaded splits — Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
    print(f"[FE] Cold-start counts — "
          f"Train: {df_train['is_cold_start'].sum()} | "
          f"Val: {df_val['is_cold_start'].sum()} | "
          f"Test: {df_test['is_cold_start'].sum()}\n")

    # ── Separate by cold-start flag ───────────────────────────────────────────
    train_cold     = df_train[df_train["is_cold_start"] == 1].copy()
    train_non_cold = df_train[df_train["is_cold_start"] == 0].copy()
    val_cold       = df_val[df_val["is_cold_start"] == 1].copy()
    val_non_cold   = df_val[df_val["is_cold_start"] == 0].copy()
    test_cold      = df_test[df_test["is_cold_start"] == 1].copy()
    test_non_cold  = df_test[df_test["is_cold_start"] == 0].copy()

    mpmn_dir     = os.path.join(out_dir, "mpmn_ready")
    gatefuse_dir = os.path.join(out_dir, "gatefuse_ready")
    os.makedirs(mpmn_dir,     exist_ok=True)
    os.makedirs(gatefuse_dir, exist_ok=True)

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 4a — COLD-START PATH (MPMN)
    # Fit on non-cold-start training data, transform cold-start rows.
    # ═════════════════════════════════════════════════════════════════════════
    print(f"{'─'*70}")
    print(f"  STEP 4a — Cold-Start Feature Engineering (MPMN path)")
    print(f"{'─'*70}\n")

    cs_engineer = ColdStartFeatureEngineer(dataset_type=dataset_type)

    # Fit on non-cold-start training data (transfer learning)
    print(f"[CS-FE] Fitting on {len(train_non_cold)} non-cold-start training samples...")
    cs_engineer.fit(train_non_cold)

    # Transform cold-start splits
    for split_name, df_split in [("train", train_cold), ("val", val_cold), ("test", test_cold)]:
        if len(df_split) == 0:
            print(f"[CS-FE] WARNING: No cold-start observations in {split_name} split — skipping.")
            continue

        X, y, feature_names = cs_engineer.transform(df_split)
        out_path = os.path.join(mpmn_dir, f"{split_name}.npz")
        np.savez(out_path, X=X, y=y)
        print(f"[CS-FE] Saved {split_name}.npz → {out_path}  "
              f"(shape: {X.shape}, churn rate: {y.mean()*100:.1f}%)")

    # Save feature names for reference
    feature_names_path = os.path.join(mpmn_dir, "feature_names.json")
    with open(feature_names_path, "w") as f:
        json.dump(cs_engineer.feature_names_out, f, indent=2)
    print(f"[CS-FE] Feature names → {feature_names_path}")
    print(f"[CS-FE] Total features: {len(cs_engineer.feature_names_out)}\n")
    print("[CS-FE] Feature list:")
    for i, name in enumerate(cs_engineer.feature_names_out):
        print(f"         {i+1:2}. {name}")
    print()

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 4b — NON-COLD-START PATH (GATEFuse)
    # Fit on non-cold-start training data, transform non-cold-start rows.
    # ═════════════════════════════════════════════════════════════════════════
    print(f"{'─'*70}")
    print(f"  STEP 4b — Non-Cold-Start Feature Engineering (GATEFuse path)")
    print(f"{'─'*70}\n")

    est_engineer = NonColdStartFeatureEngineer(dataset_type=dataset_type)

    # Fit and transform training data
    print(f"[NCS-FE] Fitting on {len(train_non_cold)} non-cold-start training samples...")
    X_train, y_train, feat_names, feat_groups = est_engineer.fit_transform(train_non_cold)

    # Save training CSV
    df_train_out = pd.DataFrame(X_train, columns=feat_names)
    df_train_out["Churn"] = y_train
    train_out_path = os.path.join(gatefuse_dir, "train.csv")
    df_train_out.to_csv(train_out_path, index=False)
    print(f"[NCS-FE] Saved train.csv → {train_out_path}  "
          f"(shape: {X_train.shape}, churn rate: {y_train.mean()*100:.1f}%)")

    # Transform val and test
    for split_name, df_split in [("val", val_non_cold), ("test", test_non_cold)]:
        if len(df_split) == 0:
            print(f"[NCS-FE] WARNING: No non-cold-start observations in {split_name} — skipping.")
            continue

        X, y, _, _ = est_engineer.transform(df_split)
        df_out = pd.DataFrame(X, columns=feat_names)
        df_out["Churn"] = y
        out_path = os.path.join(gatefuse_dir, f"{split_name}.csv")
        df_out.to_csv(out_path, index=False)
        print(f"[NCS-FE] Saved {split_name}.csv → {out_path}  "
              f"(shape: {X.shape}, churn rate: {y.mean()*100:.1f}%)")

    # Save feature groups (group name → column indices)
    groups_path = os.path.join(gatefuse_dir, "groups.json")
    with open(groups_path, "w") as f:
        json.dump(feat_groups, f, indent=2)
    print(f"[NCS-FE] Feature groups → {groups_path}")
    print(f"[NCS-FE] Group sizes: { {k: len(v) for k, v in feat_groups.items()} }")
    print(f"[NCS-FE] Total features: {len(feat_names)}\n")
    print("[NCS-FE] Feature list:")
    for i, name in enumerate(feat_names):
        print(f"          {i+1:2}. {name}")
    print()
    
    print("[NCS-FE] Features by group:")
    for group_name, indices in feat_groups.items():
        print(f"          {group_name}:")
    for idx in indices:
        print(f"            [{idx:2}] {feat_names[idx]}")
    print()

    print(f"{'='*70}")
    print(f"  FEATURE ENGINEERING COMPLETE — {dataset}")
    print(f"  MPMN ready    : {mpmn_dir}")
    print(f"  GATEFuse ready: {gatefuse_dir}")
    print(f"{'='*70}\n")



if __name__ == "__main__":
    datasets = [
        {
            "prepared_dir": "../../datasets/prepared/bank",
            "dataset": "bank",
            "out_dir": "../../datasets/processed/bank",
        },
        {
            "prepared_dir": "../../datasets/prepared/telco1",
            "dataset": "telco1",
            "out_dir": "../../datasets/processed/telco1",
        },
        {
            "prepared_dir": "../../datasets/prepared/telco2",
            "dataset": "telco2",
            "out_dir": "../../datasets/processed/telco2",
        },
    ]

    for ds in datasets:
        run(
            prepared_dir=ds["prepared_dir"],
            dataset=ds["dataset"],
            out_dir=ds["out_dir"],
        )






