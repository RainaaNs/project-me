from models.non_cold_start.model import NonColdStartModel, route_feature_groups
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import Counter

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

DATASETS = ["bank", "telco1", "telco2"]

DATA_ROOT = "C:/Users/larte/Desktop/UG/FINAL_YEAR_PROJECT/project-me/datasets/processed"
MODEL_ROOT = "C:/Users/larte/Desktop/UG/FINAL_YEAR_PROJECT/project-me/saved_models"

NUM_EPOCHS = 20
BATCH_SIZE = 32
LR = 0.001

# Columns to always drop before feeding into the model
DROP_COLS = ["Churn", "is_cold_start"]


# ─────────────────────────────────────────────
# Helper: load CSV and drop non-feature columns
# ─────────────────────────────────────────────


def load_data(dataset_name, split):
    path = f"{DATA_ROOT}/{dataset_name}/gatefuse_ready/{split}.csv"
    df = pd.read_csv(path)

    # Drop Churn and is_cold_start (is_cold_start is a routing flag, not a feature)
    drop = [c for c in DROP_COLS if c in df.columns]
    X = torch.tensor(df.drop(columns=drop).values, dtype=torch.float32)
    y = torch.tensor(df["Churn"].values, dtype=torch.float32).unsqueeze(1)
    return df, X, y


# ─────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────

for dataset_name in DATASETS:
    print("=" * 60)
    print(f"  DATASET: {dataset_name}")
    print("=" * 60)

    # ── Load data ──
    df_train, X_train, y_train = load_data(dataset_name, "train")
    df_val, X_val, y_val = load_data(dataset_name, "val")

    # ── Build feature_dims from group config ──
    groups = route_feature_groups(dataset_name)
    feature_dims = {group: len(cols) for group, cols in groups.items()}

    # ── Sanity check: dataset columns vs model expectation ──
    actual_cols = df_train.drop(
        columns=[c for c in DROP_COLS if c in df_train.columns]
    ).shape[1]
    expected_cols = sum(feature_dims.values())
    print(f"  Actual feature columns : {actual_cols}")
    print(f"  Expected by model      : {expected_cols}")
    if actual_cols != expected_cols:
        print(f"  [WARNING] Mismatch for {dataset_name} — skipping.")
        continue

    # ── Instantiate model, loss, optimizer ──
    model = NonColdStartModel(dataset_name=dataset_name, feature_dims=feature_dims)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ── Training ──
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for i in range(0, len(X_train), BATCH_SIZE):
            batch_X = X_train[i : i + BATCH_SIZE]
            batch_y = y_train[i : i + BATCH_SIZE]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch_X)
            preds = (outputs >= 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        epoch_loss /= len(X_train)
        train_acc = correct / total * 100
        print(
            f"  Epoch {epoch + 1:02d}/{NUM_EPOCHS} | Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}%"
        )

        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
                val_preds = (val_outputs >= 0.5).float()
                val_acc = (val_preds == y_val).float().mean().item() * 100
            print(f"    --> Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # ── Save model ──
    save_path = f"{MODEL_ROOT}/trained_non_cold_{dataset_name}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"  Model saved → {save_path}")

    # ── Confusion Matrix: Training Set ──
    model.eval()
    with torch.no_grad():
        train_preds_final = (model(X_train) >= 0.5).float().cpu().numpy()

    cm_train = confusion_matrix(df_train["Churn"].values, train_preds_final)
    plt.figure()
    ConfusionMatrixDisplay(
        confusion_matrix=cm_train, display_labels=["No Churn", "Churn"]
    ).plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - Training Set ({dataset_name})")
    plt.show()

    # ── Confusion Matrix: Validation Set ──
    with torch.no_grad():
        val_preds_final = (model(X_val) >= 0.5).float().cpu().numpy()

    cm_val = confusion_matrix(df_val["Churn"].values, val_preds_final)
    plt.figure()
    ConfusionMatrixDisplay(
        confusion_matrix=cm_val, display_labels=["No Churn", "Churn"]
    ).plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - Validation Set ({dataset_name})")
    plt.show()

    # ── Classification Report & AUC ──
    print(f"\n  Class distribution (val): {Counter(df_val['Churn'].values)}")
    print(
        classification_report(
            df_val["Churn"].values, val_preds_final, target_names=["No Churn", "Churn"]
        )
    )
    with torch.no_grad():
        val_probs = model(X_val).numpy()
    print(f"  AUC: {roc_auc_score(df_val['Churn'].values, val_probs):.4f}")
    print()

print("=" * 60)
print("  ALL DATASETS COMPLETE")
print("=" * 60)
