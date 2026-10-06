"""
non_cold_start_train.py — GATEFuse Training Script

Called by hybrid_train.py. Trains the NonColdStartModel on a single dataset
using the feature-engineered CSVs from the specified subset folder.

Do NOT run this directly — use hybrid_train.py instead.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from collections import Counter
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)



import os
import sys

# Add project root to path so 'models' module can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from models.non_cold_start.non_cold_start_model_full_feature_set import NonColdStartModelFullFeatureSet, build_feature_groups

# Columns to drop before feeding into the model
DROP_COLS = ["Churn", "is_cold_start"]

# Per-dataset training configs
TRAIN_CONFIGS = {
    "telco1": {"pos_weight": 1.0, "lr": 0.001, "num_epochs": 40, "batch_size": 32},
    "telco2": {"pos_weight": 1.0, "lr": 0.001, "num_epochs": 40, "batch_size": 32},
    "bank":   {"pos_weight": 1.0, "lr": 0.001, "num_epochs": 40, "batch_size": 32},
}


 
# ─────────────────────────────────────────────
# Curve plotting helper NEWWWWWWWW
# ─────────────────────────────────────────────
 
def _plot_curve(train_vals, val_vals, metric_name, dataset, plot_dir, eval_every):
    """
    Plots a single learning curve with train (blue) and val (orange) on one graph.
    Annotates the best val point and the final val point with exact values.
    X-axis is epoch number; only evaluated epochs are plotted.
    """
    epochs = [(i + 1) * eval_every for i in range(len(val_vals))]
 
    # Trim train_vals to same length as val_vals (one per eval checkpoint)
    # train_vals already sampled at same cadence
    fig, ax = plt.subplots(figsize=(9, 5))
 
    ax.plot(epochs, train_vals, color="#2563EB", linewidth=2,
            marker="o", markersize=4, label=f"Train {metric_name}")
    ax.plot(epochs, val_vals,   color="#F97316", linewidth=2,
            marker="s", markersize=4, label=f"Val {metric_name}")
 
    # Annotate best val point
    best_idx   = val_vals.index(min(val_vals) if "loss" in metric_name.lower()
                                else max(val_vals))
    best_epoch = epochs[best_idx]
    best_val   = val_vals[best_idx]
    ax.annotate(
        f"Best val\n{best_val:.4f}",
        xy=(best_epoch, best_val),
        xytext=(best_epoch + max(1, len(epochs) // 8), best_val),
        fontsize=8,
        color="#F97316",
        arrowprops=dict(arrowstyle="->", color="#F97316", lw=1.2),
    )
 
    # Annotate final val point (if different from best)
    if best_idx != len(val_vals) - 1:
        final_epoch = epochs[-1]
        final_val   = val_vals[-1]
        ax.annotate(
            f"Final\n{final_val:.4f}",
            xy=(final_epoch, final_val),
            xytext=(final_epoch - max(1, len(epochs) // 8), final_val),
            fontsize=8,
            color="#F97316",
            ha="right",
            arrowprops=dict(arrowstyle="->", color="#F97316", lw=1.2),
        )
 
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_title(f"{metric_name} — {dataset}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)

     # ── Set y-axis range based on metric type ─────────────────────────
    if "loss" in metric_name.lower():
        ax.set_ylim(0.0, 1.2)
    elif "accuracy" in metric_name.lower():
        ax.set_ylim(75, 100)

    fig.tight_layout()
 
    filename = f"{dataset}_curve_{metric_name.lower().replace(' ', '_')}.png"
    fig.savefig(os.path.join(plot_dir, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[GATEFuse] Curve saved → {os.path.join(plot_dir, filename)}")


 

# ─────────────────────────────────────────────
# Data loader
# ─────────────────────────────────────────────

def load_data(csv_path: str, dataset: str):
    """
    Loads a feature-engineered CSV and derives feature groups from the
    actual columns present using build_feature_groups() — no hardcoded
    column lists, no groups.json needed.

    NonColdStartMode.split_into_groups() slices X positionally in the
    order: Profile → Contract → Billing → Usage.
    Columns are reordered here to match that order before building the
    tensor, so every feature slice lands in the correct group encoder.

    Returns (df_raw, X tensor, y tensor, feature_groups dict, feature_dims dict).
    """
    df = pd.read_csv(csv_path)

    # Get actual feature column names (everything except label columns)
    drop = [c for c in DROP_COLS if c in df.columns]
    col_names = df.drop(columns=drop).columns.tolist()

    # Derive group → index mapping from actual columns via keyword rules
    feature_groups = build_feature_groups(dataset, col_names)

    # Reorder columns so they are in group order: Profile, Contract, Billing, Usage
    # This ensures positional slicing in split_into_groups() is correct
    ordered_cols = []
    for group_name, indices in feature_groups.items():
        ordered_cols.extend([col_names[i] for i in indices])

    # Build tensors using the reordered column list
    X = torch.tensor(df[ordered_cols].values, dtype=torch.float32)
    y = torch.tensor(df["Churn"].values, dtype=torch.float32).unsqueeze(1)

    # feature_dims: how many features per group, in group order
    feature_dims = {group: len(indices) for group, indices in feature_groups.items()}

    return df, X, y, feature_dims


# ─────────────────────────────────────────────
# Main train function
# ─────────────────────────────────────────────

def train(
    train_path: str,
    val_path:   str,
    dataset:    str,
    save_path:  str,
) -> dict:
    """
    Trains the GATEFuse model for a single dataset.

    Parameters
    ----------
    train_path : str  — path to train.csv inside the subset folder
    val_path   : str  — path to val.csv inside the subset folder
    dataset    : str  — "bank", "telco1", or "telco2"
    save_path  : str  — path where the trained .pt file will be saved

    Returns
    -------
    dict with keys: val_auc, val_f1, val_accuracy, val_precision, val_recall
    """
    cfg        = TRAIN_CONFIGS[dataset]
    num_epochs = cfg["num_epochs"]
    batch_size = cfg["batch_size"]
    lr         = cfg["lr"]

    print(f"\n[GATEFuse] Dataset    : {dataset}")
    print(f"[GATEFuse] Train path : {train_path}")
    print(f"[GATEFuse] Save path  : {save_path}")
    print(f"[GATEFuse] Epochs     : {num_epochs} | LR: {lr} | Batch: {batch_size}\n")

    # ── Load data ──────────────────────────────────────────────────────────────
    df_train, X_train, y_train, feature_dims = load_data(train_path, dataset)
    df_val,   X_val,   y_val,   _            = load_data(val_path,   dataset)

    print(f"[GATEFuse] Feature columns : {X_train.shape[1]}")
    print(f"[GATEFuse] Group sizes     : {feature_dims}")
    print(f"[GATEFuse] Train samples   : {len(X_train)} | Val: {len(X_val)}\n")

    # ── Model, loss, optimiser ─────────────────────────────────────────────────
    # by feature_dims which comes from the actual CSV columns
    model     = NonColdStartModelFullFeatureSet(feature_dims=feature_dims)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)






    # ── History — recorded at every EVAL_EVERY epochs NEWWWWWWW─────────────────────────
    history = {
        "train_loss":      [],   # average BCE loss over training batches
        "val_loss":        [],   # BCE loss on full val set
        "train_accuracy":  [],   # accuracy on training set
        "val_accuracy":    [],   # accuracy on val set
        "train_f1":        [],   # F1 on training set
        "val_f1":          [],   # F1 on val set
        "train_precision": [],
        "val_precision":   [],
        "train_recall":    [],
        "val_recall":      [],
    }
    EVAL_EVERY = 5 







    # ── Training loop ──────────────────────────────────────────────────────────

    # ------------------------------------------------------------------------------------------
     # ── NEW ADDITION  ────────────────────────────────────────────────────────────
    import time
    train_start = time.time()
    # ------------------------------------------------------------------------------------------
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct    = 0
        total      = 0

        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i : i + batch_size]
            batch_y = y_train[i : i + batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss    = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch_X)
            preds       = (outputs >= 0.5).float()
            correct    += (preds == batch_y).sum().item()
            total      += batch_y.size(0)

        epoch_loss /= len(X_train)
        train_acc   = correct / total * 100
        print(
            f"  Epoch {epoch + 1:02d}/{num_epochs} | "
            f"Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}%"
        )

        # if (epoch + 1) % 5 == 0:
        #     model.eval()
        #     with torch.no_grad():
        #         val_outputs = model(X_val)
        #         val_loss    = criterion(val_outputs, y_val).item()
        #         val_preds   = (val_outputs >= 0.5).float()
        #         val_acc     = (val_preds == y_val).float().mean().item() * 100
        #     print(f"    --> Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")


        # ── Full metric evaluation every EVAL_EVERY epochs NEWWWWWWW ────────────────────
        if (epoch + 1) % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                # Val
                val_out   = model(X_val)
                val_loss  = criterion(val_out, y_val).item()
                val_probs_ep = val_out.numpy().ravel()
                val_pred_ep  = (val_probs_ep >= 0.5).astype(float)
 
                # Train (full pass, no grad)
                tr_out    = model(X_train)
                tr_loss   = criterion(tr_out, y_train).item()
                tr_probs  = tr_out.numpy().ravel()
                tr_pred   = (tr_probs >= 0.5).astype(float)
 
            y_val_ep = y_val.numpy().ravel()
            y_tr_ep  = y_train.numpy().ravel()
 
            # Record into history
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(val_loss)
            history["train_accuracy"].append(accuracy_score(y_tr_ep,  tr_pred)  * 100)
            history["val_accuracy"].append(  accuracy_score(y_val_ep, val_pred_ep) * 100)
            history["train_f1"].append(      f1_score(y_tr_ep,  tr_pred,   zero_division=0))
            history["val_f1"].append(        f1_score(y_val_ep, val_pred_ep, zero_division=0))
            history["train_precision"].append(precision_score(y_tr_ep,  tr_pred,   zero_division=0))
            history["val_precision"].append(  precision_score(y_val_ep, val_pred_ep, zero_division=0))
            history["train_recall"].append(   recall_score(y_tr_ep,  tr_pred,   zero_division=0))
            history["val_recall"].append(     recall_score(y_val_ep, val_pred_ep, zero_division=0))
 
            print(
                f"    --> Val Loss: {val_loss:.4f} | Val Acc: {history['val_accuracy'][-1]:.2f}% | "
                f"Val F1: {history['val_f1'][-1]:.4f} | Val Recall: {history['val_recall'][-1]:.4f}")







   # ------------------------------------------------------------------------------------------
    train_end = time.time()
    print(f"\n[GATEFuse] Training time : {train_end - train_start:.2f} seconds")
   # ------------------------------------------------------------------------------------------




    # ── Save model ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n[GATEFuse] Model saved → {save_path}")





    # ── Plot learning curves NEWWWWWWW ───────────────────────────────────────────────────
    plot_dir = os.path.dirname(save_path)
    _plot_curve(history["train_loss"],      history["val_loss"],      "Loss",      dataset, plot_dir, EVAL_EVERY)
    _plot_curve(history["train_accuracy"],  history["val_accuracy"],  "Accuracy",  dataset, plot_dir, EVAL_EVERY)





    # ── Evaluation ─────────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        val_probs         = model(X_val).numpy()
        val_preds_final   = (val_probs >= 0.5).astype(float)
        train_preds_final = (model(X_train) >= 0.5).float().cpu().numpy()

    y_val_np = df_val["Churn"].values

    val_accuracy  = accuracy_score(y_val_np,  val_preds_final)
    val_precision = precision_score(y_val_np, val_preds_final, zero_division=0)
    val_recall    = recall_score(y_val_np,    val_preds_final, zero_division=0)
    val_f1        = f1_score(y_val_np,        val_preds_final, zero_division=0)
    val_auc       = roc_auc_score(y_val_np,   val_probs)




    # ── Confusion matrices ─────────────────────────────────────────────────────
    plot_dir = os.path.dirname(save_path)

    cm_train = confusion_matrix(df_train["Churn"].values, train_preds_final)
    fig, ax  = plt.subplots()
    ConfusionMatrixDisplay(
        confusion_matrix=cm_train, display_labels=["No Churn", "Churn"]
    ).plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title(f"Confusion Matrix - Training Set ({dataset})")
    fig.savefig(
        os.path.join(plot_dir, f"{dataset}_cm_train.png"),
        bbox_inches="tight", dpi=150
    )
    plt.close(fig)

    cm_val  = confusion_matrix(y_val_np, val_preds_final)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(
        confusion_matrix=cm_val, display_labels=["No Churn", "Churn"]
    ).plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title(f"Confusion Matrix - Validation Set ({dataset})")
    fig.savefig(
        os.path.join(plot_dir, f"{dataset}_cm_val.png"),
        bbox_inches="tight", dpi=150
    )
    plt.close(fig)

    # ── Classification report ──────────────────────────────────────────────────
    print(f"\n[GATEFuse] Class distribution (val): {Counter(y_val_np)}")
    print(classification_report(
        y_val_np, val_preds_final,
        target_names=["No Churn", "Churn"], zero_division=0
    ))
    print(f"[GATEFuse] AUC       : {val_auc:.4f}")
    print(f"[GATEFuse] F1        : {val_f1:.4f}")
    print(f"[GATEFuse] Accuracy  : {val_accuracy:.4f}")
    print(f"[GATEFuse] Precision : {val_precision:.4f}")
    print(f"[GATEFuse] Recall    : {val_recall:.4f}")

    return {
        "val_auc":       val_auc,
        "val_f1":        val_f1,
        "val_accuracy":  val_accuracy,
        "val_precision": val_precision,
        "val_recall":    val_recall,
         "history":       history,
    }

if __name__ == "__main__":
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for dataset in ["bank", "telco1", "telco2"]:
        train(
            train_path=f"{BASE}/datasets/processed/baseline/{dataset}/gatefuse_ready/train.csv",
            val_path=f"{BASE}/datasets/processed/baseline/{dataset}/gatefuse_ready/val.csv",
            dataset=dataset,
            save_path=f"{BASE}/checkpoints/baseline/{dataset}_non_cold_start.pt",
        )
