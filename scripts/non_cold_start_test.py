"""
scripts/non_cold_start_test.py — Non-Cold-Start Test / Inference Script (GATEFuse)

Wraps the GATEFuse inference logic as a callable test() function
for use by hybrid_test.py and hybrid_predict.py.

Mirrors weightloss_test.py but:
    - Accepts a DataFrame instead of a hardcoded CSV path
    - Returns a DataFrame with predicted_label and predicted_proba
      (preserving original index for merge_results())
    - Applies sigmoid to raw logits (weightloss_model returns logits, not probs)

Dependencies:
    models/non_cold_start/weightloss_model.py → NonColdStartModel, route_feature_groups
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
    accuracy_score, precision_score, recall_score, confusion_matrix
)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.non_cold_start_model import NonColdStartModel, route_feature_groups

# Dataset name map — hybrid system uses Telco_1/Telco_2/Bank,
# weightloss_model FEATURE_GROUPS uses telco1/telco2/bank
DATASET_NAME_MAP = {
    "Telco_1": "telco1",
    "Telco_2": "telco2",
    "Bank":    "bank",
}
# ─────────────────────────────────────────────────────────────────────────────


def test(
    df: pd.DataFrame,
    dataset: str,
    model_path: str,
    return_proba: bool = True,
    label_col: str = "Churn",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Run inference using the saved GATEFuse (weighted loss) model.

    Args:
        df           : Non-cold-start test observations (pre-engineered).
                       Include label_col for metric reporting; omit for pure inference.
        dataset      : One of 'Telco_1', 'Telco_2', 'Bank'
        model_path   : Path to saved .pt model state dict from non_cold_start_train.py
        return_proba : If True, include 'predicted_proba' column in output
        label_col    : Name of the ground-truth label column (default: 'Churn')
        threshold    : Decision threshold for converting probability → label (default: 0.5)

    Returns:
        pd.DataFrame with columns:
            - predicted_label  (int: 0 or 1)
            - predicted_proba  (float, sigmoid output) — if return_proba=True
        Index matches input df's index (for merge_results() in hybrid_test.py).
    """

    if dataset not in DATASET_NAME_MAP:
        raise ValueError(f"Unknown dataset '{dataset}'. Must be one of: {list(DATASET_NAME_MAP.keys())}")

    model_name = DATASET_NAME_MAP[dataset]

    print(f"\n[NCS-Test] Dataset: {dataset} | Threshold: {threshold}")

    # ── Prepare features ──────────────────────────────────────────────────────
    feature_cols = [c for c in df.columns
                    if c != label_col and c != 'cold_start_flag' and c != 'is_cold_start']

    X_test = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    has_labels = label_col in df.columns
    y_test = df[label_col].values if has_labels else None

    # ── Rebuild model ─────────────────────────────────────────────────────────
    groups       = route_feature_groups(model_name)
    feature_dims = {group: len(cols) for group, cols in groups.items()}

    # Verify feature count
    expected = sum(feature_dims.values())
    actual   = len(feature_cols)
    if expected != actual:
        raise ValueError(
            f"Feature mismatch for {dataset}: model expects {expected} features "
            f"but DataFrame has {actual}. Check feature engineering."
        )

    model = NonColdStartModel(dataset_name=model_name, feature_dims=feature_dims)
    model.load_state_dict(
        torch.load(model_path, map_location='cpu')
    )
    model.eval()

    print(f"[NCS-Test] Checkpoint loaded from: {model_path}")

    # ── Forward pass ──────────────────────────────────────────────────────────
    with torch.no_grad():
        logits      = model(X_test)                              # raw logits
        probs       = torch.sigmoid(logits).cpu().numpy().flatten()  # probabilities
        final_preds = (probs >= threshold).astype(int)

    print(f"[NCS-Test] Done | Churn predicted: {final_preds.sum()} / {len(final_preds)}")

    # ── Metrics (if ground truth available) ───────────────────────────────────
    if has_labels:
        auc       = roc_auc_score(y_test, probs)
        f1        = f1_score(y_test, final_preds, average='macro')
        accuracy  = accuracy_score(y_test, final_preds)
        precision = precision_score(y_test, final_preds, zero_division=0)
        recall    = recall_score(y_test, final_preds, zero_division=0)
        cm        = confusion_matrix(y_test, final_preds)

        print(f"\n[NCS-Test] Test Results (threshold={threshold}):")
        print(classification_report(y_test, final_preds, target_names=["No Churn", "Churn"]))
        print(f"  AUC-ROC : {auc*100:.2f}%")
        print(f"  Confusion Matrix: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}\n")

    # ── Return DataFrame ──────────────────────────────────────────────────────
    result = pd.DataFrame(
        {"predicted_label": final_preds},
        index=df.index   # preserve original index for merge_results()
    )
    if return_proba:
        result["predicted_proba"] = probs

    return result