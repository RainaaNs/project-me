import pandas as pd
import torch
from models.non_cold_start.model import NonColdStartModel, route_feature_groups
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

DATASETS = ["bank", "telco1", "telco2"]

DATA_ROOT = "C:/Users/larte/Desktop/UG/FINAL_YEAR_PROJECT/project-me/datasets/processed"
MODEL_ROOT = "C:/Users/larte/Desktop/UG/FINAL_YEAR_PROJECT/project-me/saved_models"

DROP_COLS = ["Churn", "is_cold_start"]


# ─────────────────────────────────────────────
# Main test loop
# ─────────────────────────────────────────────

for dataset_name in DATASETS:
    print("=" * 60)
    print(f"  DATASET: {dataset_name}")
    print("=" * 60)

    # ── Load test data ──
    path = f"{DATA_ROOT}/{dataset_name}/gatefuse_ready/test.csv"
    df_test = pd.read_csv(path)

    drop = [c for c in DROP_COLS if c in df_test.columns]
    X_test = torch.tensor(df_test.drop(columns=drop).values, dtype=torch.float32)
    y_test = df_test["Churn"].values

    # ── Build feature_dims from group config ──
    groups = route_feature_groups(dataset_name)
    feature_dims = {group: len(cols) for group, cols in groups.items()}

    # ── Sanity check ──
    actual_cols = X_test.shape[1]
    expected_cols = sum(feature_dims.values())
    print(f"  Actual feature columns : {actual_cols}")
    print(f"  Expected by model      : {expected_cols}")
    if actual_cols != expected_cols:
        print(f"  [WARNING] Mismatch for {dataset_name} — skipping.")
        continue

    # ── Rebuild model and load saved weights ──
    model = NonColdStartModel(dataset_name=dataset_name, feature_dims=feature_dims)
    model.load_state_dict(
        torch.load(f"{MODEL_ROOT}/trained_non_cold_{dataset_name}.pt")
    )
    model.eval()

    # ── Forward pass ──
    with torch.no_grad():
        outputs = model(X_test)
        preds = (outputs >= 0.5).float().cpu().numpy()
        outputs_np = outputs.cpu().numpy()

    # ── Metrics ──
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, outputs_np)
    cm = confusion_matrix(y_test, preds)

    print(f"\n  Test Accuracy : {acc:.4f}")
    print(f"  Test F1 Score : {f1:.4f}")
    print(f"  Test AUC      : {auc:.4f}")
    print(
        f"\n{classification_report(y_test, preds, target_names=['No Churn', 'Churn'])}"
    )

    # ── Confusion Matrix ──
    plt.figure()
    ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["No Churn", "Churn"]
    ).plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - Test Set ({dataset_name})")
    plt.show()

    print()

print("=" * 60)
print("  ALL DATASETS COMPLETE")
print("=" * 60)
