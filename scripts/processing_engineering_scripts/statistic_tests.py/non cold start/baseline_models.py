"""
baseline_models_comparison.py — Standard ML baselines for comparison

Runs a set of standard, widely-used models on the same data/split used
for the ablation study, so you can report them alongside Stage 1-5 in
your dissertation as external reference points (not part of the
ablation itself — this answers "how do standard methods compare?").

Models included:
  - Logistic Regression      (simplest possible baseline, linear)
  - Random Forest            (standard tree ensemble, handles
                               nonlinearity/interactions natively)
  - Gradient Boosting        (sklearn's built-in boosting; if you have
                               xgboost installed, swap in XGBClassifier
                               for a stronger/faster version)
  - Vanilla MLP (PyTorch)    (plain feed-forward NN, no grouping/
                               attention/gating at all — a second,
                               framework-consistent baseline alongside
                               your Stage 1 BaselineModel)

Uses the SAME load_data() / column-ordering logic as your ablation
training script, so results are directly comparable.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, accuracy_score,
    classification_report,
)

from models.non_cold_start.non_cold_start_model_full_feature_set import build_feature_groups

DROP_COLS = ["Churn", "is_cold_start"]


# ─────────────────────────────────────────────
# Data loading (mirrors your ablation load_data)
# ─────────────────────────────────────────────

def load_data(csv_path: str, dataset: str):
    df = pd.read_csv(csv_path)
    drop = [c for c in DROP_COLS if c in df.columns]
    col_names = df.drop(columns=drop).columns.tolist()

    feature_groups = build_feature_groups(dataset, col_names)
    ordered_cols = []
    for group_name, indices in feature_groups.items():
        ordered_cols.extend([col_names[i] for i in indices])

    X = df[ordered_cols].values.astype(np.float32)
    y = df["Churn"].values.astype(np.float32)
    return X, y


# ─────────────────────────────────────────────
# Vanilla MLP (PyTorch) — plain feed-forward,
# no grouping/attention/gating whatsoever
# ─────────────────────────────────────────────

class VanillaMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 32)):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_vanilla_mlp(X_train, y_train, X_val, y_val, num_epochs=40, lr=0.001, batch_size=32):
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_t   = torch.tensor(X_val, dtype=torch.float32)

    model = VanillaMLP(input_dim=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        for i in range(0, len(X_train_t), batch_size):
            bx = X_train_t[i:i + batch_size]
            by = y_train_t[i:i + batch_size]
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_probs = model(X_val_t).numpy().ravel()
    return val_probs


# ─────────────────────────────────────────────
# Evaluation helper — identical metric set as
# your ablation script, for direct comparability
# ─────────────────────────────────────────────

def evaluate(y_true, probs, model_name):
    preds = (probs >= 0.5).astype(float)
    auc = roc_auc_score(y_true, probs)
    f1 = f1_score(y_true, preds, zero_division=0)
    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    accuracy = accuracy_score(y_true, preds)

    print(f"\n{'=' * 60}")
    print(f"  {model_name}")
    print(f"{'=' * 60}")
    print(classification_report(y_true, preds, target_names=["No Churn", "Churn"], zero_division=0))
    print(f"AUC       : {auc:.4f}")
    print(f"F1        : {f1:.4f}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")

    return {
        "model": model_name, "auc": auc, "f1": f1,
        "accuracy": accuracy, "precision": precision, "recall": recall,
    }


# ─────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────

def run_all_baselines(train_path, val_path, dataset):
    print(f"\n{'#' * 60}")
    print(f"  DATASET: {dataset}")
    print(f"{'#' * 60}")

    X_train, y_train = load_data(train_path, dataset)
    X_val, y_val = load_data(val_path, dataset)

    # Standardize features -- important for Logistic Regression and MLP,
    # harmless for tree-based models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    results = []

    # 1. Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, class_weight=None)
    lr_model.fit(X_train_scaled, y_train)
    lr_probs = lr_model.predict_proba(X_val_scaled)[:, 1]
    results.append(evaluate(y_val, lr_probs, "Logistic Regression"))

    # 2. Random Forest
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_model.fit(X_train, y_train)  # trees don't need scaling
    rf_probs = rf_model.predict_proba(X_val)[:, 1]
    results.append(evaluate(y_val, rf_probs, "Random Forest"))

    # 3. Gradient Boosting
    gb_model = GradientBoostingClassifier(n_estimators=200, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_probs = gb_model.predict_proba(X_val)[:, 1]
    results.append(evaluate(y_val, gb_probs, "Gradient Boosting"))

    # 4. Vanilla MLP (PyTorch, no grouping/attention/gating)
    mlp_probs = train_vanilla_mlp(X_train_scaled, y_train, X_val_scaled, y_val)
    results.append(evaluate(y_val, mlp_probs, "Vanilla MLP (PyTorch)"))

    # Summary table
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY — {dataset}")
    print(f"{'=' * 60}")
    print(f"{'Model':<25}{'AUC':>8}{'F1':>8}{'Acc':>8}{'Prec':>8}{'Rec':>8}")
    for r in results:
        print(f"{r['model']:<25}{r['auc']:>8.4f}{r['f1']:>8.4f}{r['accuracy']:>8.4f}{r['precision']:>8.4f}{r['recall']:>8.4f}")

    return results


if __name__ == "__main__":
    BASE = PROJECT_ROOT

    for dataset in ["bank", "telco1", "telco2"]:
        run_all_baselines(
            train_path=f"{BASE}/datasets/processed/baseline/{dataset}/gatefuse_ready/train.csv",
            val_path=f"{BASE}/datasets/processed/baseline/{dataset}/gatefuse_ready/val.csv",
            dataset=dataset,
        )
