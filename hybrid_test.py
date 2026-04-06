"""
hybrid_test.py — Hybrid Model Testing Orchestrator

Single entry point for evaluating the full hybrid system on a held-out test set.

Usage:
    python hybrid_test.py --data path/to/test.csv --dataset Telco_1
                          --cold-model checkpoints/cold_start_model.pth
                          --non-cold-model checkpoints/non_cold_start_model.pt
                          --support-data path/to/train.csv

What it does:
    1. Loads the pre-engineered test dataset (must include cold_start_flag)
    2. Uses CSDM to split observations by cold_start_flag
    3. Cold-start subset    → cold_start_test.py   (MPMN+VML)
    4. Non-cold-start subset → non_cold_start_test.py (GATEFuse)
    5. Merges predictions back by original index
    6. Prints a unified classification report + AUC

Note on --support-data:
    MPMN is a few-shot model — it needs a labelled support pool to build
    class prototypes at test time. Pass the training CSV here so the cold-start
    test script can sample support examples from it. If not provided, the test
    set itself is used as the support pool (less ideal but functional).

Expected folder structure:
    project_root/
    ├── hybrid_test.py
    ├── csdm.py
    ├── scripts/
    │   ├── cold_start_test.py
    │   └── non_cold_start_test.py
    └── checkpoints/
        ├── cold_start_model.pth
        └── non_cold_start_model.pt
"""

import argparse
import os
import sys
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from csdm import split_by_cold_start, merge_results
from scripts.cold_start_test import test as cold_test
from scripts.non_cold_start_test import test as non_cold_test


def run(
    data_path: str,
    dataset: str,
    cold_model_path: str,
    non_cold_model_path: str,
    support_data_path: str = None,
    label_col: str = "Churn",
):
    print(f"\n{'='*60}")
    print(f"  HYBRID TESTER — Dataset: {dataset}")
    print(f"{'='*60}\n")

    # ── 1. Load test data ─────────────────────────────────────────────────────
    df = pd.read_csv(data_path)
    print(f"[Hybrid] Loaded {len(df)} test observations from: {data_path}\n")

    # ── 2. Load support data for MPMN prototype building ─────────────────────
    support_df = None
    if support_data_path:
        full_support = pd.read_csv(support_data_path)
        # Only cold-start observations are needed as the support pool
        support_df = full_support[full_support["cold_start_flag"] == 1].copy()
        print(f"[Hybrid] Support pool loaded: {len(support_df)} cold-start observations "
              f"from: {support_data_path}\n")

    # ── 3. Route by cold_start_flag ───────────────────────────────────────────
    cold_df, non_cold_df = split_by_cold_start(df)

    cold_preds     = pd.DataFrame()
    non_cold_preds = pd.DataFrame()

    # ── 4. Cold-start path (MPMN+VML) ────────────────────────────────────────
    print(f"\n[Hybrid] ── Cold-Start Path (MPMN+VML) ──")
    if len(cold_df) == 0:
        print("[Hybrid] No cold-start observations in test set.")
    else:
        cold_preds = cold_test(
            df=cold_df,
            dataset=dataset,
            model_path=cold_model_path,
            return_proba=True,
            support_df=support_df,
            label_col=label_col,
        )

    # ── 5. Non-cold-start path (GATEFuse) ─────────────────────────────────────────
    print(f"\n[Hybrid] ── Non-Cold-Start Path (GATEFuse) ──")
    if len(non_cold_df) == 0:
        print("[Hybrid] No non-cold-start observations in test set.")
    else:
        non_cold_preds = non_cold_test(
            df=non_cold_df,
            dataset=dataset,
            model_path=non_cold_model_path,
            return_proba=True,
            label_col=label_col,
        )

    # ── 6. Merge & unified report ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  UNIFIED TEST RESULTS — {dataset}")
    print(f"{'='*60}\n")

    all_preds = merge_results(cold_preds, non_cold_preds)

    has_labels = label_col in df.columns
    if has_labels:
        y_true = df.loc[all_preds.index, label_col]
        y_pred = all_preds["predicted_label"]
        y_prob = all_preds.get("predicted_proba", None)

        print(classification_report(y_true, y_pred, target_names=["No Churn", "Churn"]))

        if y_prob is not None:
            auc = roc_auc_score(y_true, y_prob)
            print(f"  AUC-ROC (unified): {auc*100:.2f}%")
    else:
        print(f"  Predictions generated for {len(all_preds)} observations.")
        print(f"  Churn predicted (1): {(all_preds['predicted_label'] == 1).sum()}")
        print(f"  No churn     (0)   : {(all_preds['predicted_label'] == 0).sum()}")

    print()
    return all_preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Churn Model — Testing")
    parser.add_argument("--data",             required=True,
                        help="Path to pre-engineered test CSV (must include cold_start_flag)")
    parser.add_argument("--dataset",          required=True,
                        help="Dataset name: Telco_1, Telco_2, or Bank")
    parser.add_argument("--cold-model",       required=True,
                        help="Path to saved cold-start model checkpoint (.pth)")
    parser.add_argument("--non-cold-model",   required=True,
                        help="Path to saved non-cold-start model checkpoint (.pt)")
    parser.add_argument("--support-data",     default=None,
                        help="Path to training CSV used as MPMN support pool (recommended)")
    parser.add_argument("--label-col",        default="Churn",
                        help="Name of the ground-truth label column")
    args = parser.parse_args()

    run(
        data_path=args.data,
        dataset=args.dataset,
        cold_model_path=args.cold_model,
        non_cold_model_path=args.non_cold_model,
        support_data_path=args.support_data,
        label_col=args.label_col,
    )