"""
prepare_data.py — Steps 1–3: Clean → Detect Cold-Start → Route & Split

Runs the full preprocessing pipeline on a raw dataset and produces
three output CSVs: train.csv, val.csv, test.csv — each containing all
original columns plus the is_cold_start flag.

These output CSVs are the input to feature_engineering.py (Step 4).

Usage:
    python prepare_data.py --data data/raw/bank_customer_churn.csv
                           --dataset bank
                           --out-dir data/prepared/bank

    python prepare_data.py --data data/raw/telco_customer_churn_1.csv
                           --dataset telco1
                           --out-dir data/prepared/telco1

    python prepare_data.py --data data/raw/telco_customer_churn_2.csv
                           --dataset telco2
                           --out-dir data/prepared/telco2

Dataset-specific notes:
    bank    → strategy='bank',       target_col='Exited'
    telco1 → strategy='telco_depth', target_col='Churn Label'
    telco2 → strategy='generic',     target_col='Churn'

Output:
    {out-dir}/train.csv   — 70% of data, with is_cold_start column
    {out-dir}/val.csv     — 15% of data, with is_cold_start column
    {out-dir}/test.csv    — 15% of data, with is_cold_start column
    {out-dir}/routing_summary.json — cold-start stats
"""

import argparse
import json
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from the cold-start project's processing module.
# Adjust this path if processing_module.py lives elsewhere in your structure.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from processing_module import MinimalPreprocessor, RobustColdStartDetector, DataRouter


# ── Dataset-specific config ───────────────────────────────────────────────────
DATASET_CONFIGS = {
    "telco1": {
        "target_col":       "Churn Label",
        "strategy":         "telco_depth",
        "tenure_col":       "Tenure in Months",
        "referral_col":     "Number of Referrals",
        "offer_col":        "Offer",
        "contract_col":     "Contract",
        "total_charges_col":"Total Charges",
        "routing_columns":  ["Tenure in Months", "Number of Referrals", "Offer"],
    },
    "telco2": {
        "target_col":       "Churn",
        "strategy":         "generic",
        "tenure_col":       "tenure",
        "referral_col":     "referrals",   # not present — detector handles gracefully
        "offer_col":        "offer",       # not present — detector handles gracefully
        "contract_col":     "Contract",
        "total_charges_col":"TotalCharges",
        "routing_columns":  ["tenure"],
    },
    "bank": {
        "target_col":       "Exited",
        "strategy":         "bank",
        "tenure_col":       "Tenure",
        "referral_col":     "referrals",   # not present — detector handles gracefully
        "offer_col":        "offer",       # not present
        "contract_col":     "Contract",    # not present
        "total_charges_col":"TotalCharges",# not present
        "num_products_col": "NumOfProducts",
        "is_active_col":    "IsActiveMember",
        "routing_columns":  ["Tenure"],
    },
}
# ─────────────────────────────────────────────────────────────────────────────


def run(data_path: str, dataset: str, out_dir: str):
    if dataset not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Must be one of: {list(DATASET_CONFIGS.keys())}"
        )

    cfg = DATASET_CONFIGS[dataset]

    print(f"\n{'='*70}")
    print(f"  PREPARE DATA — {dataset}")
    print(f"{'='*70}\n")

    os.makedirs(out_dir, exist_ok=True)

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    df_raw = pd.read_csv(data_path)
    print(f"[Prepare] Loaded {len(df_raw)} rows from: {data_path}\n")

    # ── Step 1: Clean ─────────────────────────────────────────────────────────
    preprocessor = MinimalPreprocessor(
        target_column=cfg["target_col"],
        routing_columns=cfg["routing_columns"],
    )
    df_clean = preprocessor.clean(df_raw)

    # ── Step 2: Detect cold-start ─────────────────────────────────────────────
    detector_kwargs = {
        "strategy":          cfg["strategy"],
        "tenure_col":        cfg["tenure_col"],
        "referral_col":      cfg.get("referral_col", "referrals"),
        "offer_col":         cfg.get("offer_col", "offer"),
        "contract_col":      cfg.get("contract_col", "Contract"),
        "total_charges_col": cfg.get("total_charges_col", "TotalCharges"),
    }
    if dataset == "bank":
        detector_kwargs["num_products_col"] = cfg.get("num_products_col", "NumOfProducts")
        detector_kwargs["is_active_col"]    = cfg.get("is_active_col", "IsActiveMember")

    detector = RobustColdStartDetector(**detector_kwargs)
    cold_start_flags = detector.detect(df_clean)

    # ── Step 3: Route & Split ─────────────────────────────────────────────────
    router     = DataRouter(target_column="Churn")
    df_flagged = router.route(df_clean, cold_start_flags)
    splits     = router.split(df_flagged)

    # ── Save outputs ──────────────────────────────────────────────────────────
    for split_name, df_split in splits.items():
        out_path = os.path.join(out_dir, f"{split_name}.csv")
        df_split.to_csv(out_path, index=False)
        print(f"[Prepare] Saved {split_name}.csv → {out_path}  ({len(df_split)} rows)")

    # Save routing summary
    summary = router.get_routing_summary()
    summary_path = os.path.join(out_dir, "routing_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Prepare] Routing summary → {summary_path}")

    print(f"\n[Prepare] Done. Outputs in: {out_dir}\n")
    return splits


if __name__ == "__main__":
    run(
        data_path="../../datasets/original_datasets/telco2.csv",
        dataset="telco2",
        out_dir="../../datasets/prepared/telco2",
    )

