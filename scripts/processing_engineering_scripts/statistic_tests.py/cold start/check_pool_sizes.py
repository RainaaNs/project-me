"""
scripts/check_pool_sizes.py — Quick diagnostic for the Telco-1 pool-size
discrepancy (584->1,500 per actual script output vs 508->2,032 per
cold_start_model.py's inline comments).

Just loads whatever .npz files currently exist and reports their actual
sizes plus modification times, so you can see directly which number is
current rather than guessing from comments.

Usage:
    python check_pool_sizes.py
"""

import os
import time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets", "processed")


def check(dataset, filename):
    path = os.path.join(DATASETS_DIR, dataset, "mpmn_ready", filename)
    if not os.path.exists(path):
        print(f"  {filename:25s} MISSING at {path}")
        return
    data = np.load(path)
    n = len(data["y"])
    n0 = int((data["y"] == 0).sum())
    n1 = int((data["y"] == 1).sum())
    n_features = data["X"].shape[1]
    mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path)))
    print(
        f"  {filename:25s} n={n:5d} (class0={n0}, class1={n1}) features={n_features:3d}   last modified: {mtime}"
    )


if __name__ == "__main__":
    for dataset in ["telco1", "telco2", "bank"]:
        print(f"\n{dataset.upper()}")
        check(dataset, "train.npz")
        check(dataset, "train_augmented.npz")
        check(dataset, "val.npz")
        check(dataset, "test.npz")
