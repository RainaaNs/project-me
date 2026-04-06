"""
hybrid_train.py — Hybrid Model Training Orchestrator

Single entry point for training the full hybrid system.
Passes file paths directly to each training script — no runtime
splitting needed since the two paths have different feature sets
and file formats after feature engineering.

Usage:
    python hybrid_train.py --dataset Bank
                           --processed-dir datasets/processed/bank

    python hybrid_train.py --dataset Telco_1
                           --processed-dir datasets/processed/telco1

    python hybrid_train.py --dataset Telco_2
                           --processed-dir datasets/processed/telco2

Expected folder structure under --processed-dir:
    mpmn_ready/
        train.npz
        val.npz
    gatefuse_ready/
        train.csv
        val.csv

Checkpoints saved to:
    checkpoints/{dataset}_cold_start.pth
    checkpoints/{dataset}_non_cold_start.pt
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.cold_start_train import train as cold_train
from scripts.non_cold_start_train import train as non_cold_train


def run(dataset: str, processed_dir: str, checkpoint_dir: str = "checkpoints"):

    print(f"\n{'='*60}")
    print(f"  HYBRID TRAINER — {dataset}")
    print(f"{'='*60}\n")

    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── File paths ────────────────────────────────────────────────────────────
    mpmn_train  = os.path.join(processed_dir, "mpmn_ready", "train.npz")
    mpmn_val    = os.path.join(processed_dir, "mpmn_ready", "val.npz")
    gf_train    = os.path.join(processed_dir, "gatefuse_ready", "train.csv")
    gf_val      = os.path.join(processed_dir, "gatefuse_ready", "val.csv")

    cold_save     = os.path.join(checkpoint_dir, f"{dataset}_cold_start.pth")
    non_cold_save = os.path.join(checkpoint_dir, f"{dataset}_non_cold_start.pt")

    # ── Validate paths exist ──────────────────────────────────────────────────
    for path in [mpmn_train, mpmn_val, gf_train, gf_val]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Expected file not found: {path}\n"
                f"Run feature_engineering.py for {dataset} first."
            )

    # ── Cold-start path (MPMN+VML) ────────────────────────────────────────────
    print(f"[Hybrid] ── Cold-Start Path (MPMN+VML) ──")
    cold_results = cold_train(
        train_path=mpmn_train,
        val_path=mpmn_val,
        dataset=dataset,
        save_path=cold_save,
    )

    # ── Non-cold-start path (GATEFuse) ────────────────────────────────────────
    print(f"[Hybrid] ── Non-Cold-Start Path (GATEFuse) ──")
    non_cold_results = non_cold_train(
        train_path=gf_train,
        val_path=gf_val,
        dataset=dataset,
        save_path=non_cold_save,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE — {dataset}")
    print(f"{'='*60}")
    print(f"\n  Cold-Start (MPMN+VML):")
    print(f"    AUC   : {cold_results['val_auc']*100:.2f}%")
    print(f"    F1    : {cold_results['val_f1']*100:.2f}%")
    print(f"    Recall: {cold_results['val_recall']*100:.2f}%")
    print(f"    Saved : {cold_save}")
    print(f"\n  Non-Cold-Start (GATEFuse):")
    print(f"    AUC   : {non_cold_results['val_auc']*100:.2f}%")
    print(f"    F1    : {non_cold_results['val_f1']*100:.2f}%")
    print(f"    Recall: {non_cold_results['val_recall']*100:.2f}%")
    print(f"    Saved : {non_cold_save}")
    print()


if __name__ == "__main__":
    run(
        dataset="telco2",
        processed_dir="datasets/processed/telco2",
        checkpoint_dir="checkpoints",
    )