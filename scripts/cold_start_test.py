import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    brier_score_loss,
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.cold_start.cold_start_model import MPMN, DATASET_CONFIGS


# ── 1. CALIBRATION METRIC (ECE) ───────────────────────────────────────────────
def compute_ece(probs, labels, n_bins=10):
    """Computes Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    return ece


# ── 2. EPISODIC MPMN TESTER ───────────────────────────────────────────────────
def evaluate_mpmn_episodic(model, X_test, y_test, cfg, n_support, n_episodes=500):
    """Evaluates MPMN model across n_episodes to compute 95% Confidence Intervals."""
    model.eval()
    idx_0 = np.where(y_test == 0)[0]
    idx_1 = np.where(y_test == 1)[0]

    aucs, f1s, eces = [], [], []

    with torch.no_grad():
        for _ in range(n_episodes):
            # Sample support set
            sup_idx_0 = np.random.choice(idx_0, n_support, replace=False)
            sup_idx_1 = np.random.choice(idx_1, n_support, replace=False)
            sup_idx = np.concatenate([sup_idx_0, sup_idx_1])

            # Query set is the rest
            qry_idx = np.setdiff1d(np.arange(len(y_test)), sup_idx)

            sup_X = torch.tensor(X_test[sup_idx], dtype=torch.float32)
            sup_y = torch.tensor(y_test[sup_idx], dtype=torch.long)
            qry_X = torch.tensor(X_test[qry_idx], dtype=torch.float32)
            qry_y = y_test[qry_idx]

            # Forward pass
            logits, _, _, _, _, _ = model(sup_X, sup_y, qry_X)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= cfg["decision_threshold"]).astype(int)

            aucs.append(roc_auc_score(qry_y, probs))
            f1s.append(f1_score(qry_y, preds, average="macro"))
            eces.append(compute_ece(probs, qry_y))

    # Calculate 95% Confidence Intervals
    f1_mean, f1_ci = np.mean(f1s), 1.96 * np.std(f1s) / np.sqrt(n_episodes)
    auc_mean, auc_ci = np.mean(aucs), 1.96 * np.std(aucs) / np.sqrt(n_episodes)
    ece_mean = np.mean(eces)

    return f1_mean, f1_ci, auc_mean, auc_ci, ece_mean


# ── 3. BASELINE EVALUATOR (Random Forest & XGBoost) ──────────────────────────
def evaluate_baselines(X_test, y_test, n_support, n_runs=100):
    """
    Trains standard non-episodic models on the exact same k-shot support size
    and evaluates on the remaining query test set.
    """
    idx_0 = np.where(y_test == 0)[0]
    idx_1 = np.where(y_test == 1)[0]

    rf_f1s, xgb_f1s = [], []

    for _ in range(n_runs):
        sup_idx_0 = np.random.choice(idx_0, n_support, replace=False)
        sup_idx_1 = np.random.choice(idx_1, n_support, replace=False)
        sup_idx = np.concatenate([sup_idx_0, sup_idx_1])
        qry_idx = np.setdiff1d(np.arange(len(y_test)), sup_idx)

        X_sup, y_sup = X_test[sup_idx], y_test[sup_idx]
        X_qry, y_qry = X_test[qry_idx], y_test[qry_idx]

        # Random Forest
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X_sup, y_sup)
        rf_preds = rf.predict(X_qry)
        rf_f1s.append(f1_score(y_qry, rf_preds, average="macro"))

        # XGBoost
        xgb = XGBClassifier(
            n_estimators=50, max_depth=3, eval_metric="logloss", random_state=42
        )
        xgb.fit(X_sup, y_sup)
        xgb_preds = xgb.predict(X_qry)
        xgb_f1s.append(f1_score(y_qry, xgb_preds, average="macro"))

    return np.mean(rf_f1s), np.mean(xgb_f1s)


# ── 4. MAIN BENCHMARKING PIPELINE ─────────────────────────────────────────────
def run_benchmark(dataset_name, checkpoint_path, test_npz_path):
    print(f"\n============================================================")
    print(f" BENCHMARKING COLD-START PERFORMANCE — {dataset_name.upper()}")
    print(f"============================================================")

    # Load Saved MPMN Checkpoint
    checkpoint = torch.load(checkpoint_path)
    cfg = checkpoint["config"]
    input_dim = checkpoint["input_dim"]

    model = MPMN(input_dim, cfg["hidden_dim"], cfg["latent_dim"], cfg["dropout"])
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load Test Data
    test_data = np.load(test_npz_path)
    X_test, y_test = test_data["X"].astype(np.float32), test_data["y"].astype(int)

    # 1. Evaluate Model at Configured K-shot with Confidence Intervals
    f1_m, f1_ci, auc_m, auc_ci, ece_m = evaluate_mpmn_episodic(
        model, X_test, y_test, cfg, n_support=cfg["n_support"]
    )
    print(f"\n[MPMN+VML Performance @ {cfg['n_support']}-shot]:")
    print(f"  Macro F1: {f1_m * 100:.2f}% (± {f1_ci * 100:.2f}%)")
    print(f"  ROC-AUC:  {auc_m * 100:.2f}% (± {auc_ci * 100:.2f}%)")
    print(f"  ECE Calibration Error: {ece_m:.4f}")

    # 2. Multi-Shot Efficiency Curve (K-Shot comparison with Baselines)
    k_shots = [3, 5, 10, 15]
    results = []

    print(f"\n[Running N-Shot Curve & Baseline Comparisons...]")
    for k in k_shots:
        if len(np.where(y_test == 0)[0]) < k or len(np.where(y_test == 1)[0]) < k:
            continue

        f1_mpmn, _, _, _, _ = evaluate_mpmn_episodic(
            model, X_test, y_test, cfg, n_support=k, n_episodes=200
        )
        f1_rf, f1_xgb = evaluate_baselines(X_test, y_test, n_support=k, n_runs=50)

        results.append(
            {
                "K-Shot": k,
                "MPMN+VML (Ours)": f"{f1_mpmn * 100:.2f}%",
                "Random Forest": f"{f1_rf * 100:.2f}%",
                "XGBoost": f"{f1_xgb * 100:.2f}%",
            }
        )

    # Print Comparison Table
    df_results = pd.DataFrame(results)
    print("\n--- Model Comparison across Support Sizes (Macro F1) ---")
    print(df_results.to_string(index=False))


if __name__ == "__main__":
    # Example test runs
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets_to_test = [
        {
            "name": "telco1",
            "model_path": os.path.join(PROJECT_ROOT, "checkpoints", "cold_start_", "mpmn_telco1.pth"),
            "test_npz": os.path.join(PROJECT_ROOT, "datasets", "processed", "telco1", "mpmn_ready", "test.npz"),
        },
        {
            "name": "bank",
            "model_path": os.path.join(PROJECT_ROOT, "checkpoints", "cold_start_", "mpmn_bank.pth"),
            "test_npz": os.path.join(PROJECT_ROOT, "datasets", "processed", "bank", "mpmn_ready", "test.npz"),
        },
        {
            "name": "telco2",
            "model_path": os.path.join(PROJECT_ROOT, "checkpoints", "cold_start_", "mpmn_telco2.pth"),
            "test_npz": os.path.join(PROJECT_ROOT, "datasets", "processed", "telco2", "mpmn_ready", "test.npz"),
        },
    ]

    for ds in datasets_to_test:
        if os.path.exists(ds["model_path"]) and os.path.exists(ds["test_npz"]):
            run_benchmark(ds["name"], ds["model_path"], ds["test_npz"])
        else:
            print(f"Skipping {ds['name']}: Missing model or test file.")
