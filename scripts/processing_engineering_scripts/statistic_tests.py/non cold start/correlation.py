import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

DATASETS = {
    "bank":   f"{BASE}/datasets/processed/baseline/bank/gatefuse_ready/train.csv",
    "telco1": f"{BASE}/datasets/processed/baseline/telco1/gatefuse_ready/train.csv",
    "telco2": f"{BASE}/datasets/processed/baseline/telco2/gatefuse_ready/train.csv",
}

OUTPUT_DIR = f"{BASE}/correlation_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DROP_COLS = ["is_cold_start"]

for dataset_name, path in DATASETS.items():
    print(f"\nProcessing {dataset_name}...")
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # ── Target correlation ────────────────────────────────────────────────────
    target_corr = df.corr()["Churn"].drop("Churn").sort_values()

    fig, ax = plt.subplots(figsize=(10, max(6, len(target_corr) * 0.25)))
    colors = ["#d73027" if v < 0 else "#1a9850" for v in target_corr.values]
    ax.barh(target_corr.index, target_corr.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"{dataset_name.upper()} — Target Correlation with Churn", fontsize=13)
    ax.set_xlabel("Pearson Correlation")
    plt.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, f"{dataset_name}_target_correlation.png"),
        dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print(f"  Saved target correlation plot")

    # ── Pairwise correlation heatmap ──────────────────────────────────────────
    corr_matrix = df.corr()

    fig, ax = plt.subplots(figsize=(max(12, len(corr_matrix) * 0.35),
                                    max(10, len(corr_matrix) * 0.35)))
    sns.heatmap(
        corr_matrix,
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        square=True,
        linewidths=0.3,
        ax=ax,
        cbar_kws={"shrink": 0.5},
        xticklabels=True,
        yticklabels=True,
    )
    ax.set_title(f"{dataset_name.upper()} — Pairwise Feature Correlation", fontsize=13)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, f"{dataset_name}_pairwise_correlation.png"),
        dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print(f"  Saved pairwise correlation heatmap")

print(f"\nAll plots saved to {OUTPUT_DIR}")
