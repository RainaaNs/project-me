"""
non_cold_start_train.py — GATEFuse Training Script

Called by hybrid_train.py. Trains the NonColdStartModel on a single dataset
using the feature-engineered CSVs from the specified subset folder.

Do NOT run this directly — use hybrid_train.py instead.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from collections import Counter
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)



import os
import sys

# Add project root to path so 'models' module can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))




PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "models", "non_cold_start", "other versions"))
from non_cold_start_model_targ_corr_threshold import NonColdStartModelTargCorrThreshold, build_feature_groups

# Columns to drop before feeding into the model
DROP_COLS = ["Churn", "is_cold_start"]

# Per-dataset training configs
TRAIN_CONFIGS = {
    "telco1": {"pos_weight": 1.0, "lr": 0.001, "num_epochs": 40, "batch_size": 32},
    "telco2": {"pos_weight": 1.0, "lr": 0.001, "num_epochs": 40, "batch_size": 32},
    "bank":   {"pos_weight": 1.0, "lr": 0.001, "num_epochs": 40, "batch_size": 32},
}


# ─────────────────────────────────────────────
# Data loader
# ─────────────────────────────────────────────

def load_data(csv_path: str, dataset: str):
    """
    Loads a feature-engineered CSV and derives feature groups from the
    actual columns present using build_feature_groups() — no hardcoded
    column lists, no groups.json needed.

    NonColdStartMode.split_into_groups() slices X positionally in the
    order: Profile → Contract → Billing → Usage.
    Columns are reordered here to match that order before building the
    tensor, so every feature slice lands in the correct group encoder.

    Returns (df_raw, X tensor, y tensor, feature_groups dict, feature_dims dict).
    """
    df = pd.read_csv(csv_path)

    # Get actual feature column names (everything except label columns)
    drop = [c for c in DROP_COLS if c in df.columns]
    col_names = df.drop(columns=drop).columns.tolist()

    # Derive group → index mapping from actual columns via keyword rules
    feature_groups = build_feature_groups(dataset, col_names)

    # Reorder columns so they are in group order: Profile, Contract, Billing, Usage
    # This ensures positional slicing in split_into_groups() is correct
    ordered_cols = []
    for group_name, indices in feature_groups.items():
        ordered_cols.extend([col_names[i] for i in indices])

    # Build tensors using the reordered column list
    X = torch.tensor(df[ordered_cols].values, dtype=torch.float32)
    y = torch.tensor(df["Churn"].values, dtype=torch.float32).unsqueeze(1)

    # feature_dims: how many features per group, in group order
    feature_dims = {group: len(indices) for group, indices in feature_groups.items()}

    return df, X, y, feature_dims


# ─────────────────────────────────────────────
# Main train function
# ─────────────────────────────────────────────

def train(
    train_path: str,
    val_path:   str,
    dataset:    str,
    save_path:  str,
) -> dict:
    """
    Trains the GATEFuse model for a single dataset.

    Parameters
    ----------
    train_path : str  — path to train.csv inside the subset folder
    val_path   : str  — path to val.csv inside the subset folder
    dataset    : str  — "bank", "telco1", or "telco2"
    save_path  : str  — path where the trained .pt file will be saved

    Returns
    -------
    dict with keys: val_auc, val_f1, val_accuracy, val_precision, val_recall
    """
    cfg        = TRAIN_CONFIGS[dataset]
    num_epochs = cfg["num_epochs"]
    batch_size = cfg["batch_size"]
    lr         = cfg["lr"]

    print(f"\n[GATEFuse] Dataset    : {dataset}")
    print(f"[GATEFuse] Train path : {train_path}")
    print(f"[GATEFuse] Save path  : {save_path}")
    print(f"[GATEFuse] Epochs     : {num_epochs} | LR: {lr} | Batch: {batch_size}\n")

    # ── Load data ──────────────────────────────────────────────────────────────
    df_train, X_train, y_train, feature_dims = load_data(train_path, dataset)
    df_val,   X_val,   y_val,   _            = load_data(val_path,   dataset)

    print(f"[GATEFuse] Feature columns : {X_train.shape[1]}")
    print(f"[GATEFuse] Group sizes     : {feature_dims}")
    print(f"[GATEFuse] Train samples   : {len(X_train)} | Val: {len(X_val)}\n")

    # ── Model, loss, optimiser ─────────────────────────────────────────────────
    # by feature_dims which comes from the actual CSV columns
    model     = NonColdStartModelTargCorrThreshold(feature_dims=feature_dims)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct    = 0
        total      = 0

        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i : i + batch_size]
            batch_y = y_train[i : i + batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss    = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch_X)
            preds       = (outputs >= 0.5).float()
            correct    += (preds == batch_y).sum().item()
            total      += batch_y.size(0)

        epoch_loss /= len(X_train)
        train_acc   = correct / total * 100
        print(
            f"  Epoch {epoch + 1:02d}/{num_epochs} | "
            f"Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}%"
        )

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss    = criterion(val_outputs, y_val).item()
                val_preds   = (val_outputs >= 0.5).float()
                val_acc     = (val_preds == y_val).float().mean().item() * 100
            print(f"    --> Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # ── Save model ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n[GATEFuse] Model saved → {save_path}")

    # ── Evaluation ─────────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        val_probs         = model(X_val).numpy()
        val_preds_final   = (val_probs >= 0.5).astype(float)
        train_preds_final = (model(X_train) >= 0.5).float().cpu().numpy()

    y_val_np = df_val["Churn"].values

    val_accuracy  = accuracy_score(y_val_np,  val_preds_final)
    val_precision = precision_score(y_val_np, val_preds_final, zero_division=0)
    val_recall    = recall_score(y_val_np,    val_preds_final, zero_division=0)
    val_f1        = f1_score(y_val_np,        val_preds_final, zero_division=0)
    val_auc       = roc_auc_score(y_val_np,   val_probs)

    # ── Confusion matrices ─────────────────────────────────────────────────────
    plot_dir = os.path.dirname(save_path)

    cm_train = confusion_matrix(df_train["Churn"].values, train_preds_final)
    fig, ax  = plt.subplots()
    ConfusionMatrixDisplay(
        confusion_matrix=cm_train, display_labels=["No Churn", "Churn"]
    ).plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title(f"Confusion Matrix - Training Set ({dataset})")
    fig.savefig(
        os.path.join(plot_dir, f"{dataset}_cm_train.png"),
        bbox_inches="tight", dpi=150
    )
    plt.close(fig)

    cm_val  = confusion_matrix(y_val_np, val_preds_final)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(
        confusion_matrix=cm_val, display_labels=["No Churn", "Churn"]
    ).plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title(f"Confusion Matrix - Validation Set ({dataset})")
    fig.savefig(
        os.path.join(plot_dir, f"{dataset}_cm_val.png"),
        bbox_inches="tight", dpi=150
    )
    plt.close(fig)

    # ── Classification report ──────────────────────────────────────────────────
    print(f"\n[GATEFuse] Class distribution (val): {Counter(y_val_np)}")
    print(classification_report(
        y_val_np, val_preds_final,
        target_names=["No Churn", "Churn"], zero_division=0
    ))
    print(f"[GATEFuse] AUC       : {val_auc:.4f}")
    print(f"[GATEFuse] F1        : {val_f1:.4f}")
    print(f"[GATEFuse] Accuracy  : {val_accuracy:.4f}")
    print(f"[GATEFuse] Precision : {val_precision:.4f}")
    print(f"[GATEFuse] Recall    : {val_recall:.4f}")

    return {
        "val_auc":       val_auc,
        "val_f1":        val_f1,
        "val_accuracy":  val_accuracy,
        "val_precision": val_precision,
        "val_recall":    val_recall,
    }

if __name__ == "__main__":
    BASE = r"PROJECT_ROOT"
    for dataset in ["bank", "telco1", "telco2"]:
        train(
            train_path=f"{BASE}/datasets/processed/target_corr_threshold/{dataset}/gatefuse_ready/train.csv",
            val_path=f"{BASE}/datasets/processed/target_corr_threshold/{dataset}/gatefuse_ready/val.csv",
            dataset=dataset,
            save_path=f"{BASE}/checkpoints/target_corr_threshold/{dataset}_non_cold_start.pt",
        )