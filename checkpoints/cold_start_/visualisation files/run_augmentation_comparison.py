"""
scripts/run_augmentation_comparison.py — Augmented vs. original training data

Crosses two independent axes:

    Data axis:   original train.npz  vs  CTGAN-augmented train_augmented.npz
    Model axis:  MPMN+VML (Full)     vs  MPMN without VML (Point Estimate)

...giving a 2x2 grid per dataset. This is deliberately not folded into
run_ablation_study.py's six-variant grid — crossing all six VML variants
with both data variants would be 6x2x3 = 36 training runs. Use this
script when the question is specifically "does augmentation help, and
does it help both variants equally," and run_ablation_study.py /
mpmn_vs_no_vml.py for the VML-component questions.

If a dataset has no train_augmented.npz, that cell is skipped and noted
in the output rather than silently failing — telco2 in your current
pipeline only has train.npz per the __main__ block in cold_start_train.py.

Usage:
    python run_augmentation_comparison.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets", "processed")

sys.path.append(PROJECT_ROOT)
sys.path.append(SCRIPT_DIR)
from models.cold_start.cold_start_model import DATASET_CONFIGS
from models.cold_start.ablation_model import ABLATION_VARIANTS
from run_ablation_study import train_variant, evaluate_variant

MODEL_VARIANTS = {
    "MPMN+VML (Full)": ABLATION_VARIANTS["Full (MPMN+VML)"],
    "MPMN without VML": ABLATION_VARIANTS["No Variational Path (Point Estimate)"],
}

# Same reasoning as run_ablation_study.py: a single seed can land on a
# degenerate solution by chance. 3 seeds lets you tell "this variant is
# worse" apart from "this one run collapsed."
N_SEEDS = 3

# A trained binary classifier landing at/near 50% AUC isn't "a bit worse" —
# it's indistinguishable from random guessing, which usually means training
# collapsed (e.g. posterior collapse: logits become ~constant regardless of
# input) rather than "the mechanism doesn't help here." Flag it loudly.
COLLAPSE_AUC_THRESHOLD = 0.55


def run_cell(dataset_name, data_label, train_path, val_path, test_path, cfg):
    """Train + evaluate both model variants on one data variant (original
    or augmented), across N_SEEDS seeds each. Returns a list of result rows."""
    train_data = np.load(train_path)
    val_data = np.load(val_path)
    test_data = np.load(test_path)

    X_train, y_train = train_data["X"].astype(np.float32), train_data["y"].astype(int)
    X_val, y_val = val_data["X"].astype(np.float32), val_data["y"].astype(int)
    X_test, y_test = test_data["X"].astype(np.float32), test_data["y"].astype(int)

    pool_0 = int((y_train == 0).sum())
    pool_1 = int((y_train == 1).sum())
    print(
        f"\n  [{data_label}] pool: {len(X_train)} samples "
        f"(class0={pool_0}, class1={pool_1}, churn rate={y_train.mean() * 100:.1f}%)"
    )

    rows = []
    for variant_name, flags in MODEL_VARIANTS.items():
        print(f"  Training {variant_name} on {data_label} data ({N_SEEDS} seeds) ...")

        seed_metrics, seed_temps = [], []
        n_params = None

        for seed in range(N_SEEDS):
            try:
                model, n_params, train_time = train_variant(
                    X_train, y_train, X_val, y_val, cfg, flags, seed=seed
                )
            except AssertionError as e:
                print(f"    seed {seed}: skipped — {e}")
                continue

            m = evaluate_variant(model, X_test, y_test, cfg)
            temp = F.softplus(model.log_temp).item() + 0.01

            flag = (
                "  <-- COLLAPSE (near-random AUC)"
                if m["auc"] < COLLAPSE_AUC_THRESHOLD
                else ""
            )
            print(
                f"    seed {seed}: Acc {m['accuracy'] * 100:.2f}% | Prec {m['precision'] * 100:.2f}% | "
                f"Rec {m['recall'] * 100:.2f}% | F1 {m['f1'] * 100:.2f}% | AUC {m['auc'] * 100:.2f}% | "
                f"ECE {m['ece']:.4f} | T={temp:.3f} | {train_time:.1f}s{flag}"
            )

            seed_metrics.append(m)
            seed_temps.append(temp)

        if not seed_metrics:
            continue

        def _agg(key):
            vals = [m[key] for m in seed_metrics]
            return np.mean(vals), np.std(vals)

        acc_mean, acc_std = _agg("accuracy")
        prec_mean, prec_std = _agg("precision")
        rec_mean, rec_std = _agg("recall")
        f1_mean, f1_std = _agg("f1")
        auc_mean, auc_std = _agg("auc")
        n_collapsed = sum(1 for m in seed_metrics if m["auc"] < COLLAPSE_AUC_THRESHOLD)

        print(
            f"    → mean over {len(seed_metrics)} seeds: F1 {f1_mean * 100:.2f}% (±{f1_std * 100:.2f}) | "
            f"AUC {auc_mean * 100:.2f}% (±{auc_std * 100:.2f})"
            + (
                f" | {n_collapsed}/{len(seed_metrics)} seeds collapsed"
                if n_collapsed
                else ""
            )
        )

        rows.append(
            {
                "Dataset": dataset_name,
                "Data Variant": data_label,
                "Model Variant": variant_name,
                "Train Pool Size": len(X_train),
                "Churn Rate": f"{y_train.mean() * 100:.1f}%",
                "Accuracy": f"{acc_mean * 100:.2f}% (±{acc_std * 100:.2f})",
                "Precision": f"{prec_mean * 100:.2f}% (±{prec_std * 100:.2f})",
                "Recall": f"{rec_mean * 100:.2f}% (±{rec_std * 100:.2f})",
                "Macro F1": f"{f1_mean * 100:.2f}% (±{f1_std * 100:.2f})",
                "ROC-AUC": f"{auc_mean * 100:.2f}% (±{auc_std * 100:.2f})",
                "ECE": f"{np.mean([m['ece'] for m in seed_metrics]):.4f}",
                "Temperature (mean)": f"{np.mean(seed_temps):.3f}",
                "Seeds Collapsed": f"{n_collapsed}/{len(seed_metrics)}",
                "Params": f"{n_params:,}",
            }
        )
    return rows


def run_dataset_grid(dataset_name, original_path, augmented_path, val_path, test_path):
    print(f"\n{'=' * 70}")
    print(f" AUGMENTED vs ORIGINAL — {dataset_name.upper()}")
    print(f"{'=' * 70}")

    cfg = DATASET_CONFIGS[dataset_name]
    rows = []

    if os.path.exists(original_path):
        rows += run_cell(
            dataset_name, "Original", original_path, val_path, test_path, cfg
        )
    else:
        print(f"  Skipping Original: {original_path} not found.")

    if os.path.exists(augmented_path):
        rows += run_cell(
            dataset_name, "Augmented", augmented_path, val_path, test_path, cfg
        )
    else:
        print(
            f"  Skipping Augmented: {augmented_path} not found "
            f"(no CTGAN output for this dataset)."
        )

    return pd.DataFrame(rows)


def _npz(dataset, filename):
    return os.path.join(DATASETS_DIR, dataset, "mpmn_ready", filename)


if __name__ == "__main__":
    datasets = [
        {
            "name": "telco1",
            "original": _npz("telco1", "train.npz"),
            "augmented": _npz("telco1", "train_augmented.npz"),
            "val_path": _npz("telco1", "val.npz"),
            "test_path": _npz("telco1", "test.npz"),
        },
        {
            "name": "telco2",
            "original": _npz("telco2", "train.npz"),
            "augmented": _npz("telco2", "train_augmented.npz"),
            "val_path": _npz("telco2", "val.npz"),
            "test_path": _npz("telco2", "test.npz"),
        },
        {
            "name": "bank",
            "original": _npz("bank", "train.npz"),
            "augmented": _npz("bank", "train_augmented.npz"),
            "val_path": _npz("bank", "val.npz"),
            "test_path": _npz("bank", "test.npz"),
        },
    ]

    all_dfs = []
    for ds in datasets:
        if os.path.exists(ds["val_path"]) and os.path.exists(ds["test_path"]):
            df = run_dataset_grid(
                ds["name"],
                ds["original"],
                ds["augmented"],
                ds["val_path"],
                ds["test_path"],
            )
            if not df.empty:
                all_dfs.append(df)
        else:
            print(f"Skipping {ds['name']}: missing val or test npz.")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        print(f"\n{'=' * 70}\n FULL 2x2 GRID (all datasets)\n{'=' * 70}")
        print(combined.to_string(index=False))
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_csv = os.path.join(RESULTS_DIR, "augmentation_comparison.csv")
        combined.to_csv(out_csv, index=False)
        print(f"\nSaved → {out_csv}")
