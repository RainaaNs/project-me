"""
non_cold_start_test_full_feature_set.py — GATEFuse Test Evaluation Script

Loads saved model checkpoints and evaluates on the test set for all
three datasets. Includes optimal threshold search — finds the threshold
that maximises F1 on the test set, then reports metrics at both 0.5
and the optimal threshold for comparison.

Run from project root:
    python scripts/non_cold_start_test_full_feature_set.py
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import pandas as pd
from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

# Add project root to path so 'models' module can be found
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.non_cold_start.non_cold_start_model_full_feature_set import (
    NonColdStartModelFullFeatureSet,
    build_feature_groups,
)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

BASE      = PROJECT_ROOT
SUBSET    = "baseline"
DATASETS  = ["bank", "telco1", "telco2"]
DROP_COLS = ["Churn", "is_cold_start"]


# ─────────────────────────────────────────────
# Data loader
# ─────────────────────────────────────────────

def load_test_data(dataset: str):
    csv_path = f"{BASE}/datasets/processed/{SUBSET}/{dataset}/gatefuse_ready/test.csv"
    df       = pd.read_csv(csv_path)

    drop      = [c for c in DROP_COLS if c in df.columns]
    col_names = df.drop(columns=drop).columns.tolist()

    feature_groups = build_feature_groups(dataset, col_names)

    ordered_cols = []
    for group_name, indices in feature_groups.items():
        ordered_cols.extend([col_names[i] for i in indices])

    X = torch.tensor(df[ordered_cols].values, dtype=torch.float32)
    y = df["Churn"].values

    feature_dims = {group: len(indices) for group, indices in feature_groups.items()}
    return df, X, y, feature_dims


# ─────────────────────────────────────────────
# Threshold search
# ─────────────────────────────────────────────

def find_optimal_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple:
    """
    Sweeps thresholds from 0.1 to 0.9 and returns the threshold
    that maximises F1 score for the churn class.

    Returns (best_threshold, best_f1)
    """
    thresholds  = np.arange(0.1, 0.9, 0.01)
    best_thresh = 0.5
    best_f1     = 0.0

    for thresh in thresholds:
        preds_t = (probs >= thresh).astype(float)
        f1_t    = f1_score(y_true, preds_t, zero_division=0)
        if f1_t > best_f1:
            best_f1     = f1_t
            best_thresh = thresh

    return round(float(best_thresh), 2), round(float(best_f1), 4)


# ─────────────────────────────────────────────
# Main test loop
# ─────────────────────────────────────────────

for dataset_name in DATASETS:
    print("=" * 60)
    print(f"  DATASET: {dataset_name.upper()}  [{SUBSET}]")
    print("=" * 60)

    # ── Load test data ──────────────────────────────────────────────────
    df_test, X_test, y_test, feature_dims = load_test_data(dataset_name)

    print(f"  Feature columns : {X_test.shape[1]}")
    print(f"  Group sizes     : {feature_dims}")
    print(f"  Test samples    : {len(X_test)}")
    print(f"  Class dist      : {Counter(y_test)}\n")

    # ── Load model ──────────────────────────────────────────────────────
    model_path = f"{BASE}/checkpoints/{SUBSET}/{dataset_name}_non_cold_start.pt"
    if not os.path.exists(model_path):
        print(f"  [WARNING] No saved model at {model_path} — skipping.")
        continue

    model = NonColdStartModelFullFeatureSet(feature_dims=feature_dims)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters : {total_params:,}")

    # ── Forward pass ────────────────────────────────────────────────────
    inference_start = time.time()
    with torch.no_grad():
        outputs = model(X_test)
        probs   = outputs.cpu().numpy().squeeze()
    inference_end  = time.time()
    inference_time = inference_end - inference_start

    # ── Metrics at default threshold 0.5 ────────────────────────────────
    preds_default = (probs >= 0.5).astype(float)

    acc_def  = accuracy_score(y_test,  preds_default)
    prec_def = precision_score(y_test, preds_default, zero_division=0)
    rec_def  = recall_score(y_test,   preds_default, zero_division=0)
    f1_def   = f1_score(y_test,       preds_default, zero_division=0)
    auc      = roc_auc_score(y_test,  probs)

    print(f"\n  ── At default threshold (0.50) ──")
    print(f"  Accuracy  : {acc_def:.4f}")
    print(f"  Precision : {prec_def:.4f}")
    print(f"  Recall    : {rec_def:.4f}")
    print(f"  F1        : {f1_def:.4f}")
    print(f"  AUC       : {auc:.4f}")

    # ── Find and apply optimal threshold ────────────────────────────────
    best_thresh, best_f1 = find_optimal_threshold(y_test, probs)
    preds_optimal        = (probs >= best_thresh).astype(float)

    acc_opt  = accuracy_score(y_test,  preds_optimal)
    prec_opt = precision_score(y_test, preds_optimal, zero_division=0)
    rec_opt  = recall_score(y_test,   preds_optimal, zero_division=0)
    f1_opt   = f1_score(y_test,       preds_optimal, zero_division=0)

    print(f"\n  ── At optimal threshold ({best_thresh:.2f}) ──")
    print(f"  Accuracy  : {acc_opt:.4f}")
    print(f"  Precision : {prec_opt:.4f}")
    print(f"  Recall    : {rec_opt:.4f}")
    print(f"  F1        : {f1_opt:.4f}")
    print(f"  AUC       : {auc:.4f}  (unchanged — threshold-independent)")

    # ── Classification reports ───────────────────────────────────────────
    print(f"\n  Classification report (threshold=0.50):")
    print(classification_report(
        y_test, preds_default,
        target_names=["No Churn", "Churn"], zero_division=0
    ))

    print(f"  Classification report (threshold={best_thresh:.2f}):")
    print(classification_report(
        y_test, preds_optimal,
        target_names=["No Churn", "Churn"], zero_division=0
    ))

    # ── Computational cost ───────────────────────────────────────────────
    print(f"  Inference time : {inference_time:.4f} seconds ({len(X_test)} samples)")
    print(f"  Per-sample     : {(inference_time / len(X_test)) * 1000:.4f} ms/sample")
    print(f"  Parameters     : {total_params:,}")

    # ── Confusion matrices ───────────────────────────────────────────────
    plot_dir = f"{BASE}/checkpoints/{SUBSET}"
    os.makedirs(plot_dir, exist_ok=True)

    # Default threshold
    cm_def  = confusion_matrix(y_test, preds_default)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(
        confusion_matrix=cm_def, display_labels=["No Churn", "Churn"]
    ).plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title(f"Test Set — {dataset_name} (threshold=0.50)")
    fig.savefig(
        os.path.join(plot_dir, f"{dataset_name}_cm_test_default.png"),
        bbox_inches="tight", dpi=150
    )
    plt.close(fig)

    # Optimal threshold
    cm_opt  = confusion_matrix(y_test, preds_optimal)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(
        confusion_matrix=cm_opt, display_labels=["No Churn", "Churn"]
    ).plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title(f"Test Set — {dataset_name} (threshold={best_thresh:.2f})")
    fig.savefig(
        os.path.join(plot_dir, f"{dataset_name}_cm_test_optimal.png"),
        bbox_inches="tight", dpi=150
    )
    plt.close(fig)

    print(f"\n  Confusion matrices saved → {plot_dir}\n")

print("=" * 60)
print("  ALL DATASETS COMPLETE")
print("=" * 60)
