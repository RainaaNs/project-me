"""
scripts/run_ablation_study.py — Ablation study for MPMN+VML

For each dataset (telco1, telco2, bank) and each variant in
ABLATION_VARIANTS, this script:

    1. Trains a fresh AblatableMPMN from scratch using the SAME episodic
       training loop, hyperparameters, and early-stopping rule as your
       main cold_start_train.py — the only thing that changes between
       runs is which mechanisms are switched on.
    2. Evaluates it with the same episodic tester used in cold_start_test.py
       (evaluate_mpmn_episodic-style: n_episodes random support/query
       splits, 95% CI).
    3. Records Macro F1, ROC-AUC, ECE, wall-clock training time, and
       parameter count.

Output: one comparison table per dataset, plus a combined CSV you can
paste straight into the dissertation's Results/Discussion chapter.

Usage:
    python run_ablation_study.py

Assumes the same directory layout as cold_start_train.py / cold_start_test.py:
    project-me/
      models/cold_start_model.py
      models/ablation_model.py
      scripts/cold_start_train.py
      scripts/run_ablation_study.py   <- this file
      datasets/processed/<name>/mpmn_ready/{train,train_augmented,val,test}.npz
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from collections import deque
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# Robust, cwd-independent paths: works whether this file lives in scripts/
# or models/, and regardless of the directory you run python from — this
# is what broke last time (checkpoints landed in different places depending
# on how the script was invoked).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "cold_start_")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets", "processed")

sys.path.append(PROJECT_ROOT)
sys.path.append(SCRIPT_DIR)
from models.cold_start.cold_start_model import DATASET_CONFIGS, MPMN
from models.cold_start.ablation_model import (
    AblatableMPMN,
    compute_ablatable_loss,
    ABLATION_VARIANTS,
)
from scripts.cold_start_train import EpisodeDataset
from plot_utils import generate_diagnostic_plots
from torch.utils.data import DataLoader

FIGURES_DIR = os.path.join(PROJECT_ROOT, "results", "figures")

# Set True only when models/mpmn_<dataset>.pth is confirmed trained on the
# SAME data (original vs. augmented) this run is using — otherwise Full
# gets silently retrained from scratch alongside every other variant.
LOAD_EXISTING_FULL_CHECKPOINT = False

# ── Training budget — MATCHED to cold_start_train.py on purpose.
# Using a smaller budget here (as an earlier version of this script did)
# quietly disadvantages the more complex variants (Full VML has an extra
# KL term and sampling noise to anneal in) relative to simpler ones that
# converge faster. Comparing variants fairly means giving them the same
# budget your production checkpoints get.
EPOCHS = 150
TRAIN_EPISODES = 500
VAL_EPISODES = 150
GRAD_CLIP = 1.0
N_TEST_EPISODES = 300  # for final held-out episodic evaluation

# ── Seeds per variant. A single training run's outcome can differ a lot
# from another just due to random init / early-stopping timing — the CI
# in evaluate_variant only captures episode-sampling noise at eval time,
# not this run-to-run variance. Train each variant N_SEEDS times and
# report the mean ± std across seeds so a lucky/unlucky single run
# doesn't drive your conclusions. Set to 1 for a quick sanity check.
N_SEEDS = 3


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


def train_variant(X_train, y_train, X_val, y_val, cfg, variant_flags, seed=None):
    """Train one AblatableMPMN variant. Mirrors cold_start_train.py's loop,
    minus checkpointing/logging noise, plus wall-clock timing."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    input_dim = X_train.shape[1]

    train_ds = EpisodeDataset(
        X_train,
        y_train,
        cfg["n_support"],
        cfg["n_query_train"],
        TRAIN_EPISODES,
        balanced=True,
    )
    val_ds = EpisodeDataset(
        X_val,
        y_val,
        cfg["n_support"],
        cfg["n_query_eval"],
        VAL_EPISODES,
        balanced=False,
    )
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1)

    model = AblatableMPMN(
        input_dim,
        cfg["hidden_dim"],
        cfg["latent_dim"],
        cfg["dropout"],
        use_variational=variant_flags["use_variational"],
        use_temperature=variant_flags["use_temperature"],
        use_uncertainty_weighting=variant_flags["use_uncertainty_weighting"],
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )

    anneal_epochs = max(1, int(EPOCHS * cfg["anneal_frac"]))
    best_val_loss = float("inf")
    best_state = None
    patience_count = 0
    val_window = deque(maxlen=cfg["smooth_window"])

    t_start = time.perf_counter()

    for epoch in range(EPOCHS):
        beta = min(cfg["beta_max"], cfg["beta_max"] * (epoch + 1) / anneal_epochs)

        model.train()
        for batch in train_loader:
            sup_X, sup_y, qry_X, qry_y = [b.squeeze(0) for b in batch]
            logits, q_mean, q_logvar, sup_means, sup_logvars, _ = model(
                sup_X, sup_y, qry_X
            )
            loss, _, _ = compute_ablatable_loss(
                logits,
                qry_y,
                q_mean,
                q_logvar,
                sup_means,
                sup_logvars,
                beta=beta,
                use_kl=variant_flags["use_kl"],
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        scheduler.step()

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                sup_X, sup_y, qry_X, qry_y = [b.squeeze(0) for b in batch]
                logits, q_mean, q_logvar, sup_means, sup_logvars, _ = model(
                    sup_X, sup_y, qry_X
                )
                loss, _, _ = compute_ablatable_loss(
                    logits,
                    qry_y,
                    q_mean,
                    q_logvar,
                    sup_means,
                    sup_logvars,
                    beta=beta,
                    use_kl=variant_flags["use_kl"],
                )
                v_loss += loss.item()
        avg_val = v_loss / len(val_loader)
        val_window.append(avg_val)
        smooth_val = np.mean(val_window)

        if smooth_val < best_val_loss:
            best_val_loss = smooth_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= cfg["patience"]:
                break

    train_time = time.perf_counter() - t_start
    model.load_state_dict(best_state)
    return model, n_params, train_time


def evaluate_variant(model, X_test, y_test, cfg, n_episodes=N_TEST_EPISODES):
    """Episodic held-out test evaluation, same protocol as cold_start_test.py.
    Also times inference (per-episode forward pass) as a computational-cost
    metric.

    Returns a dict with accuracy/precision/recall/f1/auc/ece — matching the
    metric set GATEFuse's tables (4.1-4.3) report, so the cold-start tables
    (4.2.x) use the same columns rather than a different subset. Each of
    accuracy/precision/recall/f1/auc comes with a 95% CI over the
    n_episodes evaluation episodes; ece and inference time are reported as
    means only (consistent with how they're already used elsewhere)."""
    model.eval()
    idx_0 = np.where(y_test == 0)[0]
    idx_1 = np.where(y_test == 1)[0]

    accs, precs, recs, f1s, aucs, eces, infer_times = [], [], [], [], [], [], []
    n_support = cfg["n_support"]
    threshold = cfg["decision_threshold"]

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
            preds = (probs >= threshold).astype(int)

            accs.append(accuracy_score(qry_y, preds))
            precs.append(precision_score(qry_y, preds, zero_division=0))
            recs.append(recall_score(qry_y, preds, zero_division=0))
            f1s.append(f1_score(qry_y, preds, average="macro"))
            aucs.append(roc_auc_score(qry_y, probs))
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


def _slugify(name):
    return (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("+", "plus")
    )


def run_dataset_ablation(dataset_name, train_path, val_path, test_path):
    print(f"\n{'=' * 70}")
    print(f" ABLATION STUDY — {dataset_name.upper()}")
    print(f"{'=' * 70}")

    cfg = DATASET_CONFIGS[dataset_name]

    train_data = np.load(train_path)
    val_data = np.load(val_path)
    test_data = np.load(test_path)

    X_train, y_train = train_data["X"].astype(np.float32), train_data["y"].astype(int)
    X_val, y_val = val_data["X"].astype(np.float32), val_data["y"].astype(int)
    X_test, y_test = test_data["X"].astype(np.float32), test_data["y"].astype(int)

    os.makedirs(MODELS_DIR, exist_ok=True)

    rows = []
    for variant_name, flags in ABLATION_VARIANTS.items():
        existing_ckpt = os.path.join(MODELS_DIR, f"mpmn_{dataset_name}.pth")
        # LOAD_EXISTING_FULL_CHECKPOINT is False by default: mpmn_<dataset>.pth
        # was trained on CTGAN-augmented data via cold_start_train.py, which
        # you've since found hurts performance on Telco-1/Telco-2 and
        # outright collapses Telco-2. Loading it here would silently mix an
        # augmented-trained model into a run you're deliberately doing on
        # original (non-augmented) data. Full is now retrained from scratch
        # like every other variant, on whatever train_path is passed below.
        # Flip this back to True only if you want Full to reuse a checkpoint
        # you've confirmed was trained on the same data this run is using.
        use_existing = (
            LOAD_EXISTING_FULL_CHECKPOINT
            and variant_name == "Full (MPMN+VML)"
            and os.path.exists(existing_ckpt)
        )

        if use_existing:
            # AblatableMPMN with all three flags True is architecturally
            # identical to MPMN (same encoder, same log_temp, same math) —
            # so instead of retraining "Full" from scratch, load the
            # checkpoint you already trained and validated with
            # cold_start_train.py. This makes the "Full" row your actual
            # shipped model's real performance, not a fresh run that could
            # land differently, and skips 150 epochs x N_SEEDS of redundant
            # compute.
            print(
                f"\n[{dataset_name}] {variant_name}: loading existing checkpoint "
                f"({existing_ckpt}) instead of retraining ..."
            )
            ckpt = torch.load(existing_ckpt)
            model = MPMN(
                X_train.shape[1], cfg["hidden_dim"], cfg["latent_dim"], cfg["dropout"]
            )
            model.load_state_dict(ckpt["model_state_dict"])
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            m = evaluate_variant(model, X_test, y_test, cfg)
            print(
                f"    loaded: Acc {m['accuracy'] * 100:.2f}% | Prec {m['precision'] * 100:.2f}% | "
                f"Rec {m['recall'] * 100:.2f}% | F1 {m['f1'] * 100:.2f}% | AUC {m['auc'] * 100:.2f}% | ECE {m['ece']:.4f}"
            )

            seed_metrics = [m]
            seed_times, seed_infers = [0.0], [m["infer_ms"]]
            best_model = model
        else:
            print(
                f"\n[{dataset_name}] Training variant: {variant_name} "
                f"({N_SEEDS} seed{'s' if N_SEEDS > 1 else ''}) ..."
            )

            seed_metrics, seed_times, seed_infers = [], [], []
            best_model, best_f1, n_params = None, -1.0, None

            for seed in range(N_SEEDS):
                model, n_params, train_time = train_variant(
                    X_train, y_train, X_val, y_val, cfg, flags, seed=seed
                )
                m = evaluate_variant(model, X_test, y_test, cfg)
                print(
                    f"    seed {seed}: Acc {m['accuracy'] * 100:.2f}% | Prec {m['precision'] * 100:.2f}% | "
                    f"Rec {m['recall'] * 100:.2f}% | F1 {m['f1'] * 100:.2f}% | AUC {m['auc'] * 100:.2f}% | "
                    f"ECE {m['ece']:.4f} | {train_time:.1f}s"
                )

                seed_metrics.append(m)
                seed_times.append(train_time)
                seed_infers.append(m["infer_ms"])
                if m["f1"] > best_f1:
                    best_f1, best_model = m["f1"], model

        # Generate confusion matrix / ROC / reliability diagram for the
        # best (or loaded) model of this variant. Pooled across 100
        # episodes on the held-out test set — see plot_utils.py for why
        # pooling is the right approach for an episodic evaluation.
        generate_diagnostic_plots(
            best_model,
            X_test,
            y_test,
            cfg,
            dataset_name=dataset_name,
            variant_name=variant_name,
            save_dir=FIGURES_DIR,
        )

        def _agg(key):
            vals = [m[key] for m in seed_metrics]
            return np.mean(vals), np.std(vals)

        acc_mean, acc_std = _agg("accuracy")
        prec_mean, prec_std = _agg("precision")
        rec_mean, rec_std = _agg("recall")
        f1_mean, f1_std = _agg("f1")
        auc_mean, auc_std = _agg("auc")
        ece_mean = np.mean([m["ece"] for m in seed_metrics])

        n_seeds_used = 1 if use_existing else N_SEEDS
        label = (
            "existing checkpoint" if use_existing else f"mean over {n_seeds_used} seeds"
        )
        print(
            f"  → {label}: Acc {acc_mean * 100:.2f}% (±{acc_std * 100:.2f}) | "
            f"Prec {prec_mean * 100:.2f}% (±{prec_std * 100:.2f}) | "
            f"Rec {rec_mean * 100:.2f}% (±{rec_std * 100:.2f}) | "
            f"F1 {f1_mean * 100:.2f}% (±{f1_std * 100:.2f}) | "
            f"AUC {auc_mean * 100:.2f}% (±{auc_std * 100:.2f}) | ECE {ece_mean:.4f} | params: {n_params:,}"
        )

        if use_existing:
            print(f"  (not re-saving — already at {existing_ckpt})")
        else:
            # Save the best-seed checkpoint. This is what
            # cold_start_benchmark_extended.py looks for — in particular
            # "Vanilla ProtoNet (floor)" needs to land at
            # models/protonet_<dataset>.pth for that script to find it.
            ckpt_name = (
                "protonet"
                if variant_name == "Vanilla ProtoNet (floor)"
                else f"ablation_{_slugify(variant_name)}"
            )
            ckpt_path = os.path.join(MODELS_DIR, f"{ckpt_name}_{dataset_name}.pth")
            torch.save(
                {
                    "model_state_dict": best_model.state_dict(),
                    "config": cfg,
                    "variant": variant_name,
                },
                ckpt_path,
            )
            print(f"  Saved best-seed checkpoint → {ckpt_path}")

        rows.append(
            {
                "Dataset": dataset_name,
                "Variant": variant_name,
                "Accuracy": f"{acc_mean * 100:.2f}% (±{acc_std * 100:.2f})",
                "Precision": f"{prec_mean * 100:.2f}% (±{prec_std * 100:.2f})",
                "Recall": f"{rec_mean * 100:.2f}% (±{rec_std * 100:.2f})",
                "Macro F1": f"{f1_mean * 100:.2f}% (±{f1_std * 100:.2f})",
                "ROC-AUC": f"{auc_mean * 100:.2f}% (±{auc_std * 100:.2f})",
                "ECE": f"{ece_mean:.4f}",
                "Train Time (s, avg)": "n/a (loaded)"
                if use_existing
                else f"{np.mean(seed_times):.1f}",
                "Inference (ms/episode)": f"{np.mean(seed_infers):.2f}",
                "Params": f"{n_params:,}",
                "N Seeds": n_seeds_used,
            }
        )

    return pd.DataFrame(rows)


def _npz(dataset, filename):
    return os.path.join(DATASETS_DIR, dataset, "mpmn_ready", filename)


if __name__ == "__main__":
    datasets = [
        {
            "name": "telco1",
            "train_path": _npz("telco1", "train.npz"),
            "val_path": _npz("telco1", "val.npz"),
            "test_path": _npz("telco1", "test.npz"),
        },
        {
            "name": "telco2",
            "train_path": _npz("telco2", "train.npz"),
            "val_path": _npz("telco2", "val.npz"),
            "test_path": _npz("telco2", "test.npz"),
        },
        {
            "name": "bank",
            "train_path": _npz("bank", "train.npz"),
            "val_path": _npz("bank", "val.npz"),
            "test_path": _npz("bank", "test.npz"),
        },
    ]

    all_results = []
    for ds in datasets:
        if all(os.path.exists(ds[k]) for k in ("train_path", "val_path", "test_path")):
            df = run_dataset_ablation(
                ds["name"], ds["train_path"], ds["val_path"], ds["test_path"]
            )
            all_results.append(df)
        else:
            print(f"Skipping {ds['name']}: missing one of train/val/test npz files.")

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        print(f"\n{'=' * 70}")
        print(" FULL ABLATION TABLE (all datasets)")
        print(f"{'=' * 70}")
        print(combined.to_string(index=False))
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_csv = os.path.join(RESULTS_DIR, "ablation_study_results.csv")
        combined.to_csv(out_csv, index=False)
        print(f"\nSaved → {out_csv}")
