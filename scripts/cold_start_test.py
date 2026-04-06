"""
scripts/cold_start_test.py — Cold-Start Test / Inference Script (MPMN+VML)

Wraps the MPMN episodic evaluation loop as a callable test() function
for use by hybrid_test.py and hybrid_predict.py.

How MPMN inference works:
    MPMN is a few-shot model — it cannot classify a query observation in
    isolation. It needs a support set (a small pool of labelled examples)
    to build class prototypes first, then measures distance from the query
    to each prototype to produce a prediction.

    At test time, we use the training data (passed in as support_df) as the
    support pool. Each episode samples n_support examples per class from this
    pool, builds prototypes, then classifies a batch of query observations
    from the test set. Results are aggregated across TEST_EPISODES episodes
    and averaged per query observation to produce stable final predictions.

Dependencies:
    src/models.py          → MPMN, DATASET_CONFIGS
    scripts/cold_start_train.py → EpisodeDataset (imported directly)
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
    accuracy_score, precision_score, recall_score, confusion_matrix
)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from models import MPMN, DATASET_CONFIGS

# Import EpisodeDataset from cold_start_train to avoid duplication
sys.path.append(os.path.dirname(__file__))
from cold_start_train import EpisodeDataset

TEST_EPISODES = 200   # number of episodic evaluations (same as notebook)
# ─────────────────────────────────────────────────────────────────────────────


def test(
    df: pd.DataFrame,
    dataset: str,
    model_path: str,
    return_proba: bool = True,
    support_df: pd.DataFrame = None,
    label_col: str = "Churn",
) -> pd.DataFrame:
    """
    Run episodic inference using the saved MPMN+VML model.

    Args:
        df           : Cold-start test observations (pre-engineered).
                       Must contain label_col if ground-truth metrics are needed.
        dataset      : One of 'Telco_1', 'Telco_2', 'Bank'
        model_path   : Path to saved .pth checkpoint from cold_start_train.py
        return_proba : If True, include 'predicted_proba' column in output
        support_df   : Labelled pool used to build prototypes. If None, a portion
                       of df itself is used as the support pool (deployment mode).
        label_col    : Name of the ground-truth label column (default: 'Churn')

    Returns:
        pd.DataFrame with columns:
            - predicted_label  (int: 0 or 1)
            - predicted_proba  (float, P(churn)) — if return_proba=True
        Index matches input df's index (for merge_results() in hybrid_test.py).
    """

    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset}'. Must be one of: {list(DATASET_CONFIGS.keys())}")

    cfg = DATASET_CONFIGS[dataset]

    print(f"\n[CS-Test] Dataset: {dataset} | Episodes: {TEST_EPISODES} | "
          f"Threshold: {cfg['decision_threshold']}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    checkpoint = torch.load(model_path, map_location='cpu')
    input_dim  = checkpoint['input_dim']
    saved_cfg  = checkpoint.get('config', cfg)

    model = MPMN(
        input_dim,
        saved_cfg['hidden_dim'],
        saved_cfg['latent_dim'],
        saved_cfg['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    final_temp = (torch.nn.functional.softplus(model.log_temp) + 0.01).item()
    print(f"[CS-Test] Checkpoint loaded | learned temperature: {final_temp:.3f}")

    # ── Prepare feature arrays ────────────────────────────────────────────────
    # Use saved feature_cols if available, otherwise infer
    feature_cols = checkpoint.get('feature_cols', None)
    if feature_cols is None:
        feature_cols = [c for c in df.columns
                        if c != label_col and c != 'cold_start_flag' and c != 'is_cold_start']

    X_test = df[feature_cols].values.astype(np.float32)

    # y_test: used for support pool building and (if available) metric reporting
    has_labels = label_col in df.columns
    y_test = df[label_col].values.astype(int) if has_labels else None

    # ── Support pool ──────────────────────────────────────────────────────────
    # MPMN needs labelled examples to form prototypes.
    # If a separate support_df is provided (e.g. training data), use it.
    # Otherwise fall back to using the test df itself (deployment / no-label mode).
    if support_df is not None and label_col in support_df.columns:
        X_sup = support_df[feature_cols].values.astype(np.float32)
        y_sup = support_df[label_col].values.astype(int)
        print(f"[CS-Test] Support pool: {len(X_sup)} observations (separate df)")
    elif has_labels:
        X_sup = X_test
        y_sup = y_test
        print(f"[CS-Test] Support pool: test df itself ({len(X_sup)} obs) — "
              f"no separate support provided")
    else:
        raise ValueError(
            "MPMN requires a labelled support pool to build prototypes. "
            "Either pass support_df or ensure df contains the label column."
        )

    # ── Episodic inference ────────────────────────────────────────────────────
    # Strategy: run TEST_EPISODES episodes. Each episode:
    #   1. Sample n_support per class from support pool → build prototypes
    #   2. Use ALL test observations as the query set
    #   3. Collect per-observation probabilities
    # Final probability = mean across all episodes (reduces variance)

    n_support   = saved_cfg['n_support']
    threshold   = saved_cfg['decision_threshold']
    idx_0_sup   = np.where(y_sup == 0)[0]
    idx_1_sup   = np.where(y_sup == 1)[0]

    if len(idx_0_sup) < n_support or len(idx_1_sup) < n_support:
        raise ValueError(
            f"Support pool too small: need {n_support} per class, "
            f"have {len(idx_0_sup)} (class 0) and {len(idx_1_sup)} (class 1)."
        )

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    X_sup_tensor  = torch.tensor(X_sup,  dtype=torch.float32)
    y_sup_tensor  = torch.tensor(y_sup,  dtype=torch.long)

    all_probs = np.zeros((TEST_EPISODES, len(X_test)))

    print(f"[CS-Test] Running {TEST_EPISODES} episodes...")

    with torch.no_grad():
        for ep in range(TEST_EPISODES):
            # Sample support set
            s0 = np.random.choice(idx_0_sup, n_support, replace=False)
            s1 = np.random.choice(idx_1_sup, n_support, replace=False)
            sup_idx = np.concatenate([s0, s1])

            sup_X = X_sup_tensor[sup_idx]
            sup_y = y_sup_tensor[sup_idx]

            # Full test set as query
            logits, _, _, _, _, _ = model(sup_X, sup_y, X_test_tensor)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs[ep] = probs

    # Average across episodes
    mean_probs = all_probs.mean(axis=0)          # (n_test,)
    final_preds = (mean_probs >= threshold).astype(int)

    print(f"[CS-Test] Done | Churn predicted: {final_preds.sum()} / {len(final_preds)}")

    # ── Metrics (if ground truth available) ───────────────────────────────────
    if has_labels:
        auc       = roc_auc_score(y_test, mean_probs)
        f1        = f1_score(y_test, final_preds, average='macro')
        accuracy  = accuracy_score(y_test, final_preds)
        precision = precision_score(y_test, final_preds, zero_division=0)
        recall    = recall_score(y_test, final_preds, zero_division=0)
        cm        = confusion_matrix(y_test, final_preds)

        print(f"\n[CS-Test] Test Results (threshold={threshold}):")
        print(classification_report(y_test, final_preds, target_names=["No Churn", "Churn"]))
        print(f"  AUC-ROC : {auc*100:.2f}%")
        print(f"  Confusion Matrix: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}\n")

    # ── Return DataFrame ──────────────────────────────────────────────────────
    result = pd.DataFrame(
        {"predicted_label": final_preds},
        index=df.index   # preserve original index for merge_results()
    )
    if return_proba:
        result["predicted_proba"] = mean_probs

    return result