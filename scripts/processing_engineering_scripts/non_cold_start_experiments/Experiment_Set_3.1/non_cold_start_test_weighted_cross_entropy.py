"""
non_cold_start_test.py — GATEFuse Test Evaluation Script

Loads saved model checkpoints and evaluates on the test set for all
three datasets. Mirrors the data loading and group derivation logic
from non_cold_start_train.py exactly.

Run from the project root:
    python scripts/non_cold_start_test.py
"""

import os
import sys
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
# Add project root to path so 'models' module can be found
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
        )
    )
)


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "models", "non_cold_start", "other versions"))
from non_cold_start_model_weighted_cross_entropy import NonColdStartModelWeightedCrossEntropy, build_feature_groups

# ─────────────────────────────────────────────
# Config — update BASE and SUBSET to match your run
# ─────────────────────────────────────────────

BASE   = r"PROJECT_ROOT"
SUBSET = "weighted_cross_entropy"   # ← change this to match the experiment you want to test

DATASETS = ["bank", "telco1", "telco2"]
DROP_COLS = ["Churn", "is_cold_start"]


# ─────────────────────────────────────────────
# Data loader — mirrors train exactly
# ─────────────────────────────────────────────

def load_test_data(dataset: str):
    """
    Loads test.csv for a dataset, derives feature groups from actual
    columns using build_feature_groups(), and reorders columns into
    group order so positional slicing in split_into_groups() is correct.
    """
    csv_path = f"{BASE}/datasets/processed/baseline/{dataset}/gatefuse_ready/test.csv"
    df = pd.read_csv(csv_path)

    drop      = [c for c in DROP_COLS if c in df.columns]
    col_names = df.drop(columns=drop).columns.tolist()

    feature_groups = build_feature_groups(dataset, col_names)

    # Reorder to match group order: Profile → Contract → Billing → Usage
    ordered_cols = []
    for group_name, indices in feature_groups.items():
        ordered_cols.extend([col_names[i] for i in indices])

    X = torch.tensor(df[ordered_cols].values, dtype=torch.float32)
    y = df["Churn"].values

    feature_dims = {group: len(indices) for group, indices in feature_groups.items()}

    return df, X, y, feature_dims


# ─────────────────────────────────────────────
# Main test loop
# ─────────────────────────────────────────────

for dataset_name in DATASETS:
    print("=" * 60)
    print(f"  DATASET: {dataset_name}  [{SUBSET}]")
    print("=" * 60)

    # ── Load test data ──────────────────────────────────────────────
    df_test, X_test, y_test, feature_dims = load_test_data(dataset_name)

    print(f"  Feature columns : {X_test.shape[1]}")
    print(f"  Group sizes     : {feature_dims}")
    print(f"  Test samples    : {len(X_test)}")
    print(f"  Class dist      : {Counter(y_test)}\n")

    # ── Rebuild model and load saved weights ────────────────────────
    model_path = f"{BASE}/checkpoints/{SUBSET}/{dataset_name}_non_cold_start.pt"
    if not os.path.exists(model_path):
        print(f"  [WARNING] No saved model found at {model_path} — skipping.")
        continue

    model = NonColdStartModelWeightedCrossEntropy(feature_dims=feature_dims)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # ── Forward pass ────────────────────────────────────────────────
    with torch.no_grad():
        outputs    = model(X_test)
        
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).float()

        probs = probs.cpu().numpy()
        preds = preds.cpu().numpy()

    # ── Metrics ─────────────────────────────────────────────────────
    acc  = accuracy_score(y_test,  preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec  = recall_score(y_test,   preds, zero_division=0)
    f1   = f1_score(y_test,       preds, zero_division=0)
    auc  = roc_auc_score(y_test,  probs)

    print(f"  Test Accuracy  : {acc:.4f}")
    print(f"  Test Precision : {prec:.4f}")
    print(f"  Test Recall    : {rec:.4f}")
    print(f"  Test F1 Score  : {f1:.4f}")
    print(f"  Test AUC       : {auc:.4f}")
    print(f"\n{classification_report(y_test, preds, target_names=['No Churn', 'Churn'], zero_division=0)}")

    # ── Confusion matrix — saved to file ────────────────────────────
    cm      = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["No Churn", "Churn"]
    ).plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title(f"Confusion Matrix - Test Set ({dataset_name}) [{SUBSET}]")

    plot_dir  = f"{BASE}/checkpoints/{SUBSET}"
    plot_path = os.path.join(plot_dir, f"{dataset_name}_cm_test.png")
    fig.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved → {plot_path}\n")

print("=" * 60)
print("  ALL DATASETS COMPLETE")
print("=" * 60)