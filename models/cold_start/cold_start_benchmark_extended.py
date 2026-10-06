"""
scripts/cold_start_benchmark_extended.py — Extended benchmarking pipeline

Builds on your existing cold_start_test.py, with two additions:

1. VANILLA PROTOTYPICAL NETWORK BASELINE
   RF and XGBoost are trained fresh from just the k-shot support set —
   that's a fair comparison for them, since they have no other source of
   knowledge. But it is NOT a fair comparison for "does meta-learning
   help" — that question needs a baseline that was ALSO meta-trained on
   the same episodic task distribution, just without the VML machinery.
   That's exactly ABLATION_VARIANTS["Vanilla ProtoNet (floor)"] from
   ablation_model.py. This script loads its checkpoint (produced by
   run_ablation_study.py) instead of retraining it, so numbers are
   consistent between your ablation table and this comparison table.

2. COMPUTATIONAL COST COLUMNS
   Params, average inference time per episode (ms), and — for RF/XGBoost —
   average fit+predict time per episode (since they retrain every episode,
   unlike the meta-learned models which train once).

Prerequisite: run run_ablation_study.py first (or train a Vanilla ProtoNet
checkpoint some other way) so a comparison checkpoint exists per dataset.
If none is found, this script trains one on the spot using the same
protocol, so it still works standalone — just slower.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Same robust, cwd-independent path scheme as run_ablation_study.py —
# both scripts must agree on where checkpoints live regardless of which
# folder they sit in or where you run python from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
MODELS_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "cold_start_")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets", "processed")

sys.path.append(PROJECT_ROOT)
from models.cold_start.cold_start_model import MPMN, DATASET_CONFIGS
from models.cold_start.ablation_model import AblatableMPMN, ABLATION_VARIANTS


def compute_ece(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (probs > lo) & (probs <= hi)
        prop = np.mean(in_bin)
        if prop > 0:
            acc = np.mean(labels[in_bin])
            conf = np.mean(probs[in_bin])
            ece += np.abs(acc - conf) * prop
    return ece


def evaluate_meta_model(model, X_test, y_test, cfg, n_support, n_episodes=300):
    """Episodic evaluation for any meta-learned model (MPMN or ProtoNet),
    with per-episode inference timing. Returns a dict — see
    run_ablation_study.py's evaluate_variant for the same pattern; both
    report accuracy/precision/recall/f1/auc/ece so the metric set matches
    GATEFuse's tables (4.1-4.3)."""
    model.eval()
    idx_0 = np.where(y_test == 0)[0]
    idx_1 = np.where(y_test == 1)[0]

    accs, precs, recs, aucs, f1s, eces, infer_times = [], [], [], [], [], [], []

    with torch.no_grad():
        for _ in range(n_episodes):
            sup_idx_0 = np.random.choice(idx_0, n_support, replace=False)
            sup_idx_1 = np.random.choice(idx_1, n_support, replace=False)
            sup_idx = np.concatenate([sup_idx_0, sup_idx_1])
            qry_idx = np.setdiff1d(np.arange(len(y_test)), sup_idx)

            sup_X = torch.tensor(X_test[sup_idx], dtype=torch.float32)
            sup_y = torch.tensor(y_test[sup_idx], dtype=torch.long)
            qry_X = torch.tensor(X_test[qry_idx], dtype=torch.float32)
            qry_y = y_test[qry_idx]

            t0 = time.perf_counter()
            logits, _, _, _, _, _ = model(sup_X, sup_y, qry_X)
            infer_times.append(time.perf_counter() - t0)

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= cfg["decision_threshold"]).astype(int)

            accs.append(accuracy_score(qry_y, preds))
            precs.append(precision_score(qry_y, preds, zero_division=0))
            recs.append(recall_score(qry_y, preds, zero_division=0))
            aucs.append(roc_auc_score(qry_y, probs))
            f1s.append(f1_score(qry_y, preds, average="macro"))
            eces.append(compute_ece(probs, qry_y))

    def _mean_ci(vals):
        return np.mean(vals), 1.96 * np.std(vals) / np.sqrt(n_episodes)

    acc_mean, acc_ci = _mean_ci(accs)
    prec_mean, prec_ci = _mean_ci(precs)
    rec_mean, rec_ci = _mean_ci(recs)
    f1_mean, f1_ci = _mean_ci(f1s)
    auc_mean, auc_ci = _mean_ci(aucs)

    return {
        "accuracy": acc_mean,
        "accuracy_ci": acc_ci,
        "precision": prec_mean,
        "precision_ci": prec_ci,
        "recall": rec_mean,
        "recall_ci": rec_ci,
        "f1": f1_mean,
        "f1_ci": f1_ci,
        "auc": auc_mean,
        "auc_ci": auc_ci,
        "ece": np.mean(eces),
        "infer_ms": np.mean(infer_times) * 1000,
    }


def evaluate_baselines_with_cost(X_test, y_test, n_support, n_runs=100):
    """Same as your evaluate_baselines, plus fit+predict timing — this is
    the fair cost comparison, since RF/XGBoost pay full train cost on
    every single episode while the meta-learned models pay it once. Now
    also reports accuracy/precision/recall for metric-set parity with the
    meta-learned models' table."""
    idx_0 = np.where(y_test == 0)[0]
    idx_1 = np.where(y_test == 1)[0]

    rf_metrics, xgb_metrics = (
        {"acc": [], "prec": [], "rec": [], "f1": [], "t": []},
        {"acc": [], "prec": [], "rec": [], "f1": [], "t": []},
    )

    for _ in range(n_runs):
        sup_idx_0 = np.random.choice(idx_0, n_support, replace=False)
        sup_idx_1 = np.random.choice(idx_1, n_support, replace=False)
        sup_idx = np.concatenate([sup_idx_0, sup_idx_1])
        qry_idx = np.setdiff1d(np.arange(len(y_test)), sup_idx)

        X_sup, y_sup = X_test[sup_idx], y_test[sup_idx]
        X_qry, y_qry = X_test[qry_idx], y_test[qry_idx]

        t0 = time.perf_counter()
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X_sup, y_sup)
        rf_preds = rf.predict(X_qry)
        rf_metrics["t"].append(time.perf_counter() - t0)
        rf_metrics["acc"].append(accuracy_score(y_qry, rf_preds))
        rf_metrics["prec"].append(precision_score(y_qry, rf_preds, zero_division=0))
        rf_metrics["rec"].append(recall_score(y_qry, rf_preds, zero_division=0))
        rf_metrics["f1"].append(f1_score(y_qry, rf_preds, average="macro"))

        t0 = time.perf_counter()
        xgb = XGBClassifier(
            n_estimators=50, max_depth=3, eval_metric="logloss", random_state=42
        )
        xgb.fit(X_sup, y_sup)
        xgb_preds = xgb.predict(X_qry)
        xgb_metrics["t"].append(time.perf_counter() - t0)
        xgb_metrics["acc"].append(accuracy_score(y_qry, xgb_preds))
        xgb_metrics["prec"].append(precision_score(y_qry, xgb_preds, zero_division=0))
        xgb_metrics["rec"].append(recall_score(y_qry, xgb_preds, zero_division=0))
        xgb_metrics["f1"].append(f1_score(y_qry, xgb_preds, average="macro"))

    def _summarize(d):
        return {
            "accuracy": np.mean(d["acc"]),
            "precision": np.mean(d["prec"]),
            "recall": np.mean(d["rec"]),
            "f1": np.mean(d["f1"]),
            "cost_ms": np.mean(d["t"]) * 1000,
        }

    return _summarize(rf_metrics), _summarize(xgb_metrics)


def load_or_train_protonet(
    dataset_name,
    cfg,
    input_dim,
    protonet_ckpt_path,
    X_train=None,
    y_train=None,
    X_val=None,
    y_val=None,
):
    """Loads a Vanilla ProtoNet checkpoint if run_ablation_study.py already
    produced one. Otherwise trains one on the spot (requires train/val data
    to be passed in) so this script still works standalone."""
    if os.path.exists(protonet_ckpt_path):
        ckpt = torch.load(protonet_ckpt_path)
        model = AblatableMPMN(
            input_dim,
            cfg["hidden_dim"],
            cfg["latent_dim"],
            cfg["dropout"],
            **{
                k: v
                for k, v in ABLATION_VARIANTS["Vanilla ProtoNet (floor)"].items()
                if k != "use_kl"
            },
        )
        model.load_state_dict(ckpt["model_state_dict"])
        return model

    if X_train is None:
        raise FileNotFoundError(
            f"No ProtoNet checkpoint at {protonet_ckpt_path} and no train/val data given. "
            f"Run run_ablation_study.py first, or pass train/val npz paths."
        )

    # Fallback: quick on-the-spot training (import kept local to avoid a
    # hard dependency when a checkpoint already exists)
    from scripts.run_ablation_study import train_variant

    flags = ABLATION_VARIANTS["Vanilla ProtoNet (floor)"]
    model, _, _ = train_variant(X_train, y_train, X_val, y_val, cfg, flags, seed=0)
    os.makedirs(os.path.dirname(protonet_ckpt_path) or ".", exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "config": cfg}, protonet_ckpt_path
    )
    return model


def run_extended_benchmark(
    dataset_name, mpmn_ckpt_path, test_npz_path, protonet_ckpt_path
):
    print(f"\n{'=' * 70}")
    print(f" EXTENDED COLD-START BENCHMARK — {dataset_name.upper()}")
    print(f"{'=' * 70}")

    cfg_and_dim = torch.load(mpmn_ckpt_path)
    cfg = cfg_and_dim["config"]
    input_dim = cfg_and_dim["input_dim"]

    mpmn = MPMN(input_dim, cfg["hidden_dim"], cfg["latent_dim"], cfg["dropout"])
    mpmn.load_state_dict(cfg_and_dim["model_state_dict"])
    mpmn_params = sum(p.numel() for p in mpmn.parameters() if p.requires_grad)

    protonet = load_or_train_protonet(dataset_name, cfg, input_dim, protonet_ckpt_path)
    protonet_params = sum(p.numel() for p in protonet.parameters() if p.requires_grad)

    test_data = np.load(test_npz_path)
    X_test, y_test = test_data["X"].astype(np.float32), test_data["y"].astype(int)
    n_support = cfg["n_support"]

    mpmn_m = evaluate_meta_model(mpmn, X_test, y_test, cfg, n_support)
    proto_m = evaluate_meta_model(protonet, X_test, y_test, cfg, n_support)
    rf_m, xgb_m = evaluate_baselines_with_cost(X_test, y_test, n_support)

    rows = [
        {
            "Model": "MPMN+VML (Ours)",
            "Accuracy": f"{mpmn_m['accuracy'] * 100:.2f}% (±{mpmn_m['accuracy_ci'] * 100:.2f})",
            "Precision": f"{mpmn_m['precision'] * 100:.2f}% (±{mpmn_m['precision_ci'] * 100:.2f})",
            "Recall": f"{mpmn_m['recall'] * 100:.2f}% (±{mpmn_m['recall_ci'] * 100:.2f})",
            "Macro F1": f"{mpmn_m['f1'] * 100:.2f}% (±{mpmn_m['f1_ci'] * 100:.2f})",
            "ROC-AUC": f"{mpmn_m['auc'] * 100:.2f}%",
            "ECE": f"{mpmn_m['ece']:.4f}",
            "Params": f"{mpmn_params:,}",
            "Cost/Episode (ms)": f"{mpmn_m['infer_ms']:.2f} (inference only — trained once)",
        },
        {
            "Model": "Vanilla ProtoNet",
            "Accuracy": f"{proto_m['accuracy'] * 100:.2f}% (±{proto_m['accuracy_ci'] * 100:.2f})",
            "Precision": f"{proto_m['precision'] * 100:.2f}% (±{proto_m['precision_ci'] * 100:.2f})",
            "Recall": f"{proto_m['recall'] * 100:.2f}% (±{proto_m['recall_ci'] * 100:.2f})",
            "Macro F1": f"{proto_m['f1'] * 100:.2f}% (±{proto_m['f1_ci'] * 100:.2f})",
            "ROC-AUC": f"{proto_m['auc'] * 100:.2f}%",
            "ECE": f"{proto_m['ece']:.4f}",
            "Params": f"{protonet_params:,}",
            "Cost/Episode (ms)": f"{proto_m['infer_ms']:.2f} (inference only — trained once)",
        },
        {
            "Model": "Random Forest",
            "Accuracy": f"{rf_m['accuracy'] * 100:.2f}%",
            "Precision": f"{rf_m['precision'] * 100:.2f}%",
            "Recall": f"{rf_m['recall'] * 100:.2f}%",
            "Macro F1": f"{rf_m['f1'] * 100:.2f}%",
            "ROC-AUC": "n/a",
            "ECE": "n/a",
            "Params": "n/a (50 trees, retrained per episode)",
            "Cost/Episode (ms)": f"{rf_m['cost_ms']:.2f} (fit + predict, every episode)",
        },
        {
            "Model": "XGBoost",
            "Accuracy": f"{xgb_m['accuracy'] * 100:.2f}%",
            "Precision": f"{xgb_m['precision'] * 100:.2f}%",
            "Recall": f"{xgb_m['recall'] * 100:.2f}%",
            "Macro F1": f"{xgb_m['f1'] * 100:.2f}%",
            "ROC-AUC": "n/a",
            "ECE": "n/a",
            "Params": "n/a (50 trees, retrained per episode)",
            "Cost/Episode (ms)": f"{xgb_m['cost_ms']:.2f} (fit + predict, every episode)",
        },
    ]

    df = pd.DataFrame(rows)
    print(f"\n[{n_support}-shot comparison, including computational cost]")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    datasets = [
        {
            "name": "telco1",
            "mpmn_ckpt": os.path.join(MODELS_DIR, "mpmn_telco1.pth"),
            "test_npz": os.path.join(DATASETS_DIR, "telco1", "mpmn_ready", "test.npz"),
            "protonet_ckpt": os.path.join(MODELS_DIR, "protonet_telco1.pth"),
        },
        {
            "name": "telco2",
            "mpmn_ckpt": os.path.join(MODELS_DIR, "mpmn_telco2.pth"),
            "test_npz": os.path.join(DATASETS_DIR, "telco2", "mpmn_ready", "test.npz"),
            "protonet_ckpt": os.path.join(MODELS_DIR, "protonet_telco2.pth"),
        },
        {
            "name": "bank",
            "mpmn_ckpt": os.path.join(MODELS_DIR, "mpmn_bank.pth"),
            "test_npz": os.path.join(DATASETS_DIR, "bank", "mpmn_ready", "test.npz"),
            "protonet_ckpt": os.path.join(MODELS_DIR, "protonet_bank.pth"),
        },
    ]

    all_dfs = []
    for ds in datasets:
        if os.path.exists(ds["mpmn_ckpt"]) and os.path.exists(ds["test_npz"]):
            df = run_extended_benchmark(
                ds["name"], ds["mpmn_ckpt"], ds["test_npz"], ds["protonet_ckpt"]
            )
            df.insert(0, "Dataset", ds["name"])
            all_dfs.append(df)
        else:
            print(
                f"Skipping {ds['name']}: missing checkpoint ({ds['mpmn_ckpt']}) or test file."
            )

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_csv = os.path.join(RESULTS_DIR, "extended_benchmark_results.csv")
        combined.to_csv(out_csv, index=False)
        print(f"\nSaved → {out_csv}")
