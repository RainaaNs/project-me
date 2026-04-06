"""
scripts/non_cold_start_train.py — Non-Cold-Start Training Script (GATEFuse)

Reads directly from CSV files produced by feature_engineering.py.

Expected inputs:
    train_path : path to gatefuse_ready/train.csv
    val_path   : path to gatefuse_ready/val.csv

Each CSV contains named feature columns plus a 'Churn' column.

Uses BCEWithLogitsLoss with per-dataset pos_weight.
The GATEFuse classifier returns raw logits — sigmoid is applied
externally at inference time, not inside the model.

Dependencies (must be importable from project root):
    models/non_cold_start_model.py → NonColdStartModel, route_feature_groups
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
    accuracy_score, precision_score, recall_score
)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.non_cold_start_model import NonColdStartModel, route_feature_groups


# ── Per-dataset training configs ──────────────────────────────────────────────
TRAIN_CONFIGS = {
    "telco1": {"pos_weight": 4.0, "lr": 0.001, "num_epochs": 20, "batch_size": 32},
    "telco2": {"pos_weight": 3.0, "lr": 0.001, "num_epochs": 20, "batch_size": 32},
    "bank":    {"pos_weight": 3.0, "lr": 0.001, "num_epochs": 20, "batch_size": 32},
}

DATASET_NAME_MAP = {
    "telco1": "telco1",
    "telco2": "telco2",
    "bank":    "bank",
}
# ─────────────────────────────────────────────────────────────────────────────


def train(
    train_path: str,
    val_path: str,
    dataset: str,
    save_path: str,
    label_col: str = "Churn",
) -> dict:
    """
    Train the GATEFuse non-cold-start model with weighted cross-entropy loss.

    Args:
        train_path : Path to gatefuse_ready/train.csv
        val_path   : Path to gatefuse_ready/val.csv
        dataset    : One of 'telco1', 'telco2', 'bank'
        save_path  : Where to save the model state dict (.pt)
        label_col  : Name of the target column (default: 'Churn')

    Returns:
        dict: val_auc, val_f1, val_accuracy, val_precision, val_recall
    """

    if dataset not in TRAIN_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Must be one of: {list(TRAIN_CONFIGS.keys())}"
        )

    cfg        = TRAIN_CONFIGS[dataset]
    model_name = DATASET_NAME_MAP[dataset]

    print(f"\n{'='*60}")
    print(f"  NON-COLD-START TRAINING (GATEFuse) — {dataset}")
    print(f"  pos_weight={cfg['pos_weight']} | lr={cfg['lr']} | "
          f"epochs={cfg['num_epochs']} | batch={cfg['batch_size']}")
    print(f"{'='*60}\n")

    # ── Load CSV files ────────────────────────────────────────────────────────
    df_train = pd.read_csv(train_path)
    df_val   = pd.read_csv(val_path)

    feature_cols = [c for c in df_train.columns if c != label_col]

    X_train = torch.tensor(df_train[feature_cols].values, dtype=torch.float32)
    y_train = torch.tensor(df_train[label_col].values,    dtype=torch.float32).unsqueeze(1)
    X_val   = torch.tensor(df_val[feature_cols].values,   dtype=torch.float32)
    y_val   = torch.tensor(df_val[label_col].values,      dtype=torch.float32).unsqueeze(1)

    print(f"[NCS-Train] Train: {len(X_train)} | Val: {len(X_val)} | Features: {len(feature_cols)}")
    print(f"[NCS-Train] Churn rate — Train: {df_train[label_col].mean()*100:.1f}% | "
          f"Val: {df_val[label_col].mean()*100:.1f}%\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    groups       = route_feature_groups(model_name)
    feature_dims = {group: len(cols) for group, cols in groups.items()}

    expected = sum(feature_dims.values())
    actual   = len(feature_cols)
    if expected != actual:
        raise ValueError(
            f"Feature mismatch for {dataset}: model expects {expected} features "
            f"but CSV has {actual}. Check feature engineering."
        )

    model      = NonColdStartModel(dataset_name=model_name, feature_dims=feature_dims)
    pos_weight = torch.tensor([cfg['pos_weight']], dtype=torch.float32)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = optim.Adam(model.parameters(), lr=cfg['lr'])

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[NCS-Train] Model built | pos_weight={cfg['pos_weight']} | "
          f"Parameters: {n_params:,}\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    num_epochs = cfg['num_epochs']
    batch_size = cfg['batch_size']

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct    = 0
        total      = 0

        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i + batch_size]
            batch_y = y_train[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss    = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch_X)
            preds   = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == batch_y).sum().item()
            total   += batch_y.size(0)

        epoch_loss /= len(X_train)
        train_acc   = correct / total * 100

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss    = criterion(val_outputs, y_val).item()
                val_preds   = (torch.sigmoid(val_outputs) >= 0.5).float()
                val_acc     = (val_preds == y_val).float().mean().item() * 100
            print(f"  Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        else:
            print(f"  Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}%")

    # ── Validation evaluation ─────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_probs   = torch.sigmoid(val_outputs).cpu().numpy().flatten()
        val_preds   = (val_probs >= 0.5).astype(int)

    y_val_np = df_val[label_col].values

    val_auc       = roc_auc_score(y_val_np, val_probs)
    val_f1        = f1_score(y_val_np, val_preds, average='macro')
    val_accuracy  = accuracy_score(y_val_np, val_preds)
    val_precision = precision_score(y_val_np, val_preds, zero_division=0)
    val_recall    = recall_score(y_val_np, val_preds, zero_division=0)

    print(f"\n[NCS-Train] Validation Results:")
    print(classification_report(y_val_np, val_preds, target_names=["No Churn", "Churn"]))
    print(f"  AUC: {val_auc*100:.2f}%")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"[NCS-Train] Model saved → {save_path}\n")

    return {
        'val_auc':       val_auc,
        'val_f1':        val_f1,
        'val_accuracy':  val_accuracy,
        'val_precision': val_precision,
        'val_recall':    val_recall,
    }