"""
scripts/plot_utils.py — Confusion matrix, ROC curve, and reliability
diagram for any episodically-evaluated model.

Why "pooled across episodes" instead of one single confusion matrix:
Your evaluation protocol resamples a new random support/query split every
episode (that's how you get the 95% CI on F1/AUC elsewhere in your
pipeline). A single query example can land in different episodes with
different support sets, so there's no one canonical train/test pass to
draw a confusion matrix from the way you would for an ordinary classifier.
The standard way to visualize an episodic evaluation is to pool the
(prediction, true label) pairs across many episodes into one running set,
then compute the confusion matrix / ROC / calibration curve over that
pooled set. This is what generate_diagnostic_plots() below does — it's
the same underlying evaluation loop as evaluate_variant(), just also
keeping the raw predictions instead of only summary statistics.

Usage (per variant, after training):
    from scripts.plot_utils import generate_diagnostic_plots
    generate_diagnostic_plots(model, X_test, y_test, cfg,
                               dataset_name="telco1",
                               variant_name="Full (MPMN+VML)",
                               save_dir=FIGURES_DIR)
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # no display needed — just save PNGs
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc


def _slugify(name):
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")


def _pooled_episodic_predictions(model, X_test, y_test, cfg, n_episodes=100):
    """Same episodic sampling as evaluate_variant, but returns pooled raw
    arrays (probs, preds, labels) instead of only summary statistics."""
    model.eval()
    idx_0 = np.where(y_test == 0)[0]
    idx_1 = np.where(y_test == 1)[0]
    n_support = cfg["n_support"]
    threshold = cfg["decision_threshold"]

    all_probs, all_labels = [], []

    with torch.no_grad():
        for _ in range(n_episodes):
            sup_idx_0 = np.random.choice(idx_0, n_support, replace=False)
            sup_idx_1 = np.random.choice(idx_1, n_support, replace=False)
            sup_idx = np.concatenate([sup_idx_0, sup_idx_1])
            qry_idx = np.setdiff1d(np.arange(len(y_test)), sup_idx)

            sup_X = torch.tensor(X_test[sup_idx], dtype=torch.float32)
            sup_y = torch.tensor(y_test[sup_idx], dtype=torch.long)
            qry_X = torch.tensor(X_test[qry_idx], dtype=torch.float32)

            logits, _, _, _, _, _ = model(sup_X, sup_y, qry_X)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            all_probs.append(probs)
            all_labels.append(y_test[qry_idx])

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = (probs >= threshold).astype(int)
    return probs, preds, labels


def plot_confusion_matrix(labels, preds, title, save_path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Churn", "Churn"])
    ax.set_yticklabels(["No Churn", "Churn"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(labels, probs, title, save_path):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_reliability_diagram(labels, probs, title, save_path, n_bins=10):
    """Same binning convention as compute_ece elsewhere in your pipeline,
    so the diagram and the ECE number in your results table agree."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers, bin_accs, bin_confs, bin_counts = [], [], [], []
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (probs > lo) & (probs <= hi)
        if np.sum(in_bin) > 0:
            bin_centers.append((lo + hi) / 2)
            bin_accs.append(np.mean(labels[in_bin]))
            bin_confs.append(np.mean(probs[in_bin]))
            bin_counts.append(np.sum(in_bin))

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.bar(bin_centers, bin_accs, width=1 / n_bins * 0.9, alpha=0.7,
           edgecolor="black", label="Observed")
    ax.set_xlabel("Predicted probability (churn)")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def generate_diagnostic_plots(model, X_test, y_test, cfg, dataset_name, variant_name,
                               save_dir, n_episodes=100):
    """Generates and saves confusion matrix, ROC curve, and reliability
    diagram for one (dataset, variant) pair. Returns the pooled arrays in
    case you want additional custom plots."""
    os.makedirs(save_dir, exist_ok=True)
    probs, preds, labels = _pooled_episodic_predictions(model, X_test, y_test, cfg, n_episodes)

    slug = f"{dataset_name}_{_slugify(variant_name)}"
    label_prefix = f"{dataset_name.upper()} — {variant_name}"

    plot_confusion_matrix(labels, preds, f"Confusion Matrix\n{label_prefix}",
                           os.path.join(save_dir, f"{slug}_confusion_matrix.png"))
    plot_roc_curve(labels, probs, f"ROC Curve\n{label_prefix}",
                    os.path.join(save_dir, f"{slug}_roc_curve.png"))
    plot_reliability_diagram(labels, probs, f"Reliability Diagram\n{label_prefix}",
                              os.path.join(save_dir, f"{slug}_reliability.png"))

    print(f"    Saved 3 diagnostic plots → {save_dir}/{slug}_*.png")
    return probs, preds, labels
