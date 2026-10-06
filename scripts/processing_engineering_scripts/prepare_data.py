"""
prepare_data.py — Step 1: Prepare Data

Pipeline:

    Raw Data
        ↓
    Minimal Cleaning
        ↓
    Cold-Start Detection
        ↓
    Unified DataFrame + is_cold_start
        ↓
    Stratified Train / Validation / Test Split

IMPORTANT:
    This stage does NOT perform model feature selection.

    All model-specific feature decisions are reserved for:
        feature_engineering_definitions.py

This script produces:

    {out-dir}/train.csv
    {out-dir}/val.csv
    {out-dir}/test.csv
    {out-dir}/routing_summary.json
    {out-dir}/prepare_log.txt

Each CSV contains:
    - cleaned original columns
    - standardized 'Churn' target
    - 'is_cold_start' flag

The cold-start and non-cold-start populations are NOT separated
into different files at this stage. The feature_engineering.py
stage performs that separation.
"""

import json
import os
import sys
import pandas as pd


# ── Make local processing_module.py importable ────────────────────────────────
SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from processing_module import (
    MinimalPreprocessor,
    RobustColdStartDetector,
    DataRouter,
)


# ── Dataset-specific configuration ────────────────────────────────────────────
DATASET_CONFIGS = {

    "telco1": {
        "target_col": "Churn Label",
        "strategy": "telco_depth",

        "tenure_col": "Tenure in Months",
        "referral_col": "Number of Referrals",
        "offer_col": "Offer",
        "contract_col": "Contract",
        "total_charges_col": "Total Charges",

        "routing_columns": [
            "Tenure in Months",
            "Number of Referrals",
            "Offer",
        ],
    },

    "telco2": {
        "target_col": "Churn",
        "strategy": "generic",

        "tenure_col": "tenure",
        "referral_col": "referrals",
        "offer_col": "offer",
        "contract_col": "Contract",
        "total_charges_col": "TotalCharges",

        "routing_columns": [
            "tenure",
        ],
    },

    "bank": {
        "target_col": "Exited",
        "strategy": "bank",

        "tenure_col": "Tenure",

        # These do not exist in the Bank dataset, but the detector
        # handles missing columns gracefully.
        "referral_col": "referrals",
        "offer_col": "offer",
        "contract_col": "Contract",
        "total_charges_col": "TotalCharges",

        "num_products_col": "NumOfProducts",
        "is_active_col": "IsActiveMember",

        "routing_columns": [
            "Tenure",
        ],
    },
}


def run(
    data_path: str,
    dataset: str,
    out_dir: str,
):

    if dataset not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Must be one of: "
            f"{list(DATASET_CONFIGS.keys())}"
        )

    cfg = DATASET_CONFIGS[dataset]

    print("\n" + "=" * 70)
    print(f"  PREPARE DATA — {dataset}")
    print("=" * 70 + "\n")

    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 1 — LOAD RAW DATA
    # ═════════════════════════════════════════════════════════════════════════

    df_raw = pd.read_csv(data_path)

    print(
        f"[Prepare] Loaded {len(df_raw)} rows "
        f"from: {data_path}\n"
    )

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 2 — MINIMAL CLEANING
    # ═════════════════════════════════════════════════════════════════════════

    preprocessor = MinimalPreprocessor(
        target_column=cfg["target_col"],
        routing_columns=cfg["routing_columns"],
    )

    df_clean = preprocessor.clean(
        df_raw
    )

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 3 — COLD-START DETECTION
    # ═════════════════════════════════════════════════════════════════════════

    detector_kwargs = {
        "strategy": cfg["strategy"],
        "tenure_col": cfg["tenure_col"],
        "referral_col": cfg.get(
            "referral_col",
            "referrals",
        ),
        "offer_col": cfg.get(
            "offer_col",
            "offer",
        ),
        "contract_col": cfg.get(
            "contract_col",
            "Contract",
        ),
        "total_charges_col": cfg.get(
            "total_charges_col",
            "TotalCharges",
        ),
    }

    # Bank-specific cold-start detection fields.
    if dataset == "bank":

        detector_kwargs[
            "num_products_col"
        ] = cfg.get(
            "num_products_col",
            "NumOfProducts",
        )

        detector_kwargs[
            "is_active_col"
        ] = cfg.get(
            "is_active_col",
            "IsActiveMember",
        )

    detector = RobustColdStartDetector(
        **detector_kwargs
    )

    cold_start_flags = detector.detect(
        df_clean
    )

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 4 — ATTACH ROUTING FLAG + SPLIT
    # ═════════════════════════════════════════════════════════════════════════

    router = DataRouter(
        target_column="Churn"
    )

    # IMPORTANT:
    # This does NOT create separate cold/non-cold datasets.
    # It attaches a flag to the unified dataframe.
    df_flagged = router.route(
        df_clean,
        cold_start_flags,
    )

    splits = router.split(
        df_flagged
    )

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 5 — SAVE PREPARED DATA
    # ═════════════════════════════════════════════════════════════════════════

    for split_name, df_split in splits.items():

        out_path = os.path.join(
            out_dir,
            f"{split_name}.csv",
        )

        df_split.to_csv(
            out_path,
            index=False,
        )

        print(
            f"[Prepare] Saved {split_name}.csv → "
            f"{out_path} "
            f"({len(df_split)} rows, "
            f"{len(df_split.columns)} columns)"
        )

    # ── Save routing summary ──────────────────────────────────────────────────
    summary_path = os.path.join(
        out_dir,
        "routing_summary.json",
    )

    with open(
        summary_path,
        "w",
    ) as f:

        json.dump(
            router.get_routing_summary(),
            f,
            indent=2,
        )

    print(
        f"[Prepare] Routing summary → "
        f"{summary_path}"
    )

    print(
        f"\n[Prepare] Done. "
        f"Outputs in: {out_dir}\n"
    )

    return splits


def run_with_logging(
    data_path: str,
    dataset: str,
    out_dir: str,
):
    """
    Runs prepare_data.py while saving console output
    to a dataset-specific log file.
    """

    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    log_path = os.path.join(
        out_dir,
        "prepare_log.txt",
    )

    with open(
        log_path,
        "w",
    ) as log_file:

        original_stdout = sys.stdout
        sys.stdout = log_file

        try:
            result = run(
                data_path,
                dataset,
                out_dir,
            )

        finally:
            sys.stdout = original_stdout

    print(
        f"✓ {dataset} done — "
        f"log saved to {log_path}"
    )

    return result


if __name__ == "__main__":

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    datasets = [

        {
            "data_path":
                os.path.join(PROJECT_ROOT, "datasets", "original_datasets", "telco1.csv"),
            "dataset":
                "telco1",
            "out_dir":
                os.path.join(PROJECT_ROOT, "datasets", "prepared", "telco1"),
        },

        {
            "data_path":
                os.path.join(PROJECT_ROOT, "datasets", "original_datasets", "telco2.csv"),
            "dataset":
                "telco2",
            "out_dir":
                os.path.join(PROJECT_ROOT, "datasets", "prepared", "telco2"),
        },

        {
            "data_path":
                os.path.join(PROJECT_ROOT, "datasets", "original_datasets", "bank.csv"),
            "dataset":
                "bank",
            "out_dir":
                os.path.join(PROJECT_ROOT, "datasets", "prepared", "bank"),
        },

    ]

    for ds in datasets:

        run(
            data_path=ds["data_path"],
            dataset=ds["dataset"],
            out_dir=ds["out_dir"],
        )
