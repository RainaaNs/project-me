"""
scripts/basic_distance_baseline.py — "Does meta-learning itself help?"

This is the third ablation your thesis draft describes but hasn't been run
yet: "replacing the prototypical network core with a basic distance metric
classifier, to measure the role of meta-learning."

It's a DIFFERENT question from "MPMN without VML." That ablation keeps the
learned encoder and episodic meta-training, and only strips the variational
machinery. This one strips the learned encoder AND meta-training entirely:

    Full MPMN+VML          → learned embedding + variational distance
    MPMN without VML       → learned embedding + plain Euclidean distance
    THIS baseline           → raw features (no learning at all) + plain
                               Euclidean distance to per-class centroids

No training happens here — there are no parameters to fit. For each
episode: take the support set's raw (already-scaled) features, compute a
per-class mean vector (centroid) directly in raw feature space, and
classify each query by nearest centroid. This is evaluated with the exact
same episodic protocol (same n_support, same N_TEST_EPISODES, same test
set) as every other variant in your ablation table, so it's a fair,
directly-comparable floor.

If this baseline is close to MPMN's numbers, meta-learning/the encoder
isn't adding much — your prototypes are already separable in raw feature
space. If it's much worse, that's direct evidence the learned embedding
(independent of VML) is doing real work.

Usage:
    python basic_distance_baseline.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from scipy.special import softmax

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets", "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

sys.path.append(PROJECT_ROOT)
from models.cold_start.cold_start_model import DATASET_CONFIGS

N_TEST_EPISODES = 300  # matches evaluate_variant elsewhere in your pipeline


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


def evaluate_nearest_centroid(
    X_test, y_test, n_support, threshold, n_episodes=N_TEST_EPISODES
):
    idx_0 = np.where(y_test == 0)[0]
    idx_1 = np.where(y_test == 1)[0]

    accs, precs, recs, aucs, f1s, eces, infer_times = [], [], [], [], [], [], []

    for _ in range(n_episodes):
        sup_idx_0 = np.random.choice(idx_0, n_support, replace=False)
        sup_idx_1 = np.random.choice(idx_1, n_support, replace=False)
        qry_idx = np.setdiff1d(
            np.arange(len(y_test)), np.concatenate([sup_idx_0, sup_idx_1])
        )

        proto_0 = X_test[sup_idx_0].mean(axis=0)
        proto_1 = X_test[sup_idx_1].mean(axis=0)
        qry_X = X_test[qry_idx]
        qry_y = y_test[qry_idx]

        t0 = time.perf_counter()
        # Plain squared Euclidean distance to each raw-feature centroid —
        # no learned embedding, no variance weighting, no temperature.
        dist_0 = ((qry_X - proto_0) ** 2).sum(axis=1)
        dist_1 = ((qry_X - proto_1) ** 2).sum(axis=1)
        logits = np.stack([-dist_0, -dist_1], axis=1)
        probs_both = softmax(logits, axis=1)
        probs = probs_both[:, 1]
        infer_times.append(time.perf_counter() - t0)

        preds = (probs >= threshold).astype(int)

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


def run_dataset(dataset_name, test_path):
    print(
        f"\n{'=' * 70}\n BASIC DISTANCE-METRIC BASELINE — {dataset_name.upper()}\n{'=' * 70}"
    )
    cfg = DATASET_CONFIGS[dataset_name]

    test_data = np.load(test_path)
    X_test, y_test = test_data["X"].astype(np.float32), test_data["y"].astype(int)

    m = evaluate_nearest_centroid(
        X_test, y_test, cfg["n_support"], cfg["decision_threshold"]
    )
    print(
        f"  Acc: {m['accuracy'] * 100:.2f}% (±{m['accuracy_ci'] * 100:.2f}) | "
        f"Prec: {m['precision'] * 100:.2f}% (±{m['precision_ci'] * 100:.2f}) | "
        f"Rec: {m['recall'] * 100:.2f}% (±{m['recall_ci'] * 100:.2f}) | "
        f"F1: {m['f1'] * 100:.2f}% (±{m['f1_ci'] * 100:.2f}) | "
        f"AUC: {m['auc'] * 100:.2f}% (±{m['auc_ci'] * 100:.2f}) | "
        f"ECE: {m['ece']:.4f} | infer/episode: {m['infer_ms']:.3f}ms"
    )

    return {
        "Dataset": dataset_name,
        "Variant": "Basic Distance Metric (no meta-learning)",
        "Accuracy": f"{m['accuracy'] * 100:.2f}% (±{m['accuracy_ci'] * 100:.2f})",
        "Precision": f"{m['precision'] * 100:.2f}% (±{m['precision_ci'] * 100:.2f})",
        "Recall": f"{m['recall'] * 100:.2f}% (±{m['recall_ci'] * 100:.2f})",
        "Macro F1": f"{m['f1'] * 100:.2f}% (±{m['f1_ci'] * 100:.2f})",
        "ROC-AUC": f"{m['auc'] * 100:.2f}% (±{m['auc_ci'] * 100:.2f})",
        "ECE": f"{m['ece']:.4f}",
        "Inference (ms/episode)": f"{m['infer_ms']:.3f}",
        "Params": "0 (no learned parameters)",
    }


if __name__ == "__main__":
    datasets = [
        {
            "name": "telco1",
            "test_path": os.path.join(DATASETS_DIR, "telco1", "mpmn_ready", "test.npz"),
        },
        {
            "name": "telco2",
            "test_path": os.path.join(DATASETS_DIR, "telco2", "mpmn_ready", "test.npz"),
        },
        {
            "name": "bank",
            "test_path": os.path.join(DATASETS_DIR, "bank", "mpmn_ready", "test.npz"),
        },
    ]

    rows = []
    for ds in datasets:
        if os.path.exists(ds["test_path"]):
            rows.append(run_dataset(ds["name"], ds["test_path"]))
        else:
            print(f"Skipping {ds['name']}: {ds['test_path']} not found.")

    if rows:
        df = pd.DataFrame(rows)
        print(f"\n{'=' * 70}\n SUMMARY\n{'=' * 70}")
        print(df.to_string(index=False))
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_csv = os.path.join(RESULTS_DIR, "basic_distance_baseline.csv")
        df.to_csv(out_csv, index=False)
        print(f"\nSaved → {out_csv}")
