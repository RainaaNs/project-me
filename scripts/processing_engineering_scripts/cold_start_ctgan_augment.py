"""
ctgan_augment.py — CTGAN Augmentation for Cold-Start Training Data

Augments the cold-start training pool (mpmn_ready/train.npz) using CTGAN
synthetic data generation, then saves the result as train_augmented.npz.

CTGAN trains on whatever feature set is in train.npz — no per-feature drop
list. Whatever ColdStartFeatureEngineer produced is what gets augmented.

TENURE INDEX:
    The tenure feature index is computed automatically from
    feature_names.json. You do not need to set it manually.

Requirements:
    pip install sdv

Usage:
    Run from processing_engineering_scripts/ or project root.
    Change the bottom block for each dataset and run.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


# ── Target augmented pool sizes ───────────────────────────────────────────────
TARGET_TRAIN_SIZE = {
    "telco1": 1500,
    "telco2": 1800,
    "bank":    3000,
}

# CTGAN config
CTGAN_CONFIG = {
    "epochs":            300,
    "batch_size":        500,
    "generator_dim":     (256, 256),
    "discriminator_dim": (256, 256),
    "verbose":           True,
}

# Tenure column name per dataset (as it appears in feature_names.json).
# Used to filter synthetic rows down to the cold-start tenure range.
TENURE_COL_NAME = {
    "telco1": "Tenure in Months",
    "telco2": "tenure",
    "bank":   "Tenure",
}

COLD_START_TENURE_THRESHOLD = 2     # months
REBALANCE_THRESHOLD         = 0.10
SEED                        = 42
# ─────────────────────────────────────────────────────────────────────────────


def run(dataset: str, mpmn_dir: str, model_dir: str):
    """
    Run CTGAN augmentation for one dataset.

    Args:
        dataset   : One of 'telco1', 'telco2', 'bank'
        mpmn_dir  : Path to mpmn_ready/ folder
        model_dir : Directory to save/load CTGAN model (.pkl)
    """
    try:
        from sdv.single_table import CTGANSynthesizer
        from sdv.metadata import SingleTableMetadata
        from sklearn.metrics import roc_auc_score
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        print("ERROR: sdv is not installed. Run: pip install sdv")
        sys.exit(1)

    if dataset not in TARGET_TRAIN_SIZE:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Must be one of: {list(TARGET_TRAIN_SIZE.keys())}"
        )

    os.makedirs(model_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  CTGAN AUGMENTATION — {dataset}")
    print(f"{'='*60}\n")

    # ── Load paths ────────────────────────────────────────────────────────────
    train_path      = os.path.join(mpmn_dir, "train.npz")
    val_path        = os.path.join(mpmn_dir, "val.npz")
    test_path       = os.path.join(mpmn_dir, "test.npz")
    feat_names_path = os.path.join(mpmn_dir, "feature_names.json")

    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Expected file not found: {path}\n"
                f"Run feature_engineering.py for {dataset} first."
            )
    if not os.path.exists(feat_names_path):
        raise FileNotFoundError(
            f"feature_names.json not found in {mpmn_dir}.\n"
            f"Run feature_engineering.py for {dataset} first."
        )

    # ── Load feature names ────────────────────────────────────────────────────
    with open(feat_names_path) as f:
        feat_names = json.load(f)

    # ── Load real cold-start training data ────────────────────────────────────
    train_d        = np.load(train_path)
    X_real, y_real = train_d["X"].astype(np.float32), train_d["y"].astype(int)
    n_real         = len(X_real)

    print(f"[CTGAN] Real cold-start train  : {n_real}")
    print(f"[CTGAN] Features              : {len(feat_names)}")
    print(f"[CTGAN] Churn rate (real)      : {y_real.mean()*100:.1f}%")

    # ── Locate tenure feature index ───────────────────────────────────────────
    tenure_col_name = TENURE_COL_NAME[dataset]
    if tenure_col_name not in feat_names:
        raise ValueError(
            f"Tenure column '{tenure_col_name}' not found in feature_names.json.\n"
            f"Available columns: {feat_names}"
        )
    tenure_idx = feat_names.index(tenure_col_name)
    print(f"[CTGAN] Tenure col             : '{tenure_col_name}' at index {tenure_idx}\n")

    target_size = TARGET_TRAIN_SIZE[dataset]
    n_synthetic = max(0, target_size - n_real)

    print(f"[CTGAN] Target augmented size  : {target_size}")
    print(f"[CTGAN] Synthetic needed       : {n_synthetic}")

    if n_synthetic <= 0:
        print(f"[CTGAN] Already at target size. No augmentation needed.")
        return

    # ── Build DataFrame for CTGAN ─────────────────────────────────────────────
    df_ctgan = pd.DataFrame(X_real, columns=feat_names)
    df_ctgan["__label__"] = y_real.astype(int)

    # ── Train or load CTGAN ───────────────────────────────────────────────────
    # Each dataset has its own saved model. If your feature set changes (e.g.
    # because ColdStartFeatureEngineer was updated), delete the old .pkl to
    # force retraining.
    model_path = os.path.join(model_dir, f"ctgan_{dataset}.pkl")

    if os.path.exists(model_path):
        print(f"[CTGAN] Loading saved model from: {model_path}")
        synthesizer = CTGANSynthesizer.load(model_path)
        print(f"[CTGAN] Model loaded.\n")
    else:
        print(f"[CTGAN] Training CTGAN ({CTGAN_CONFIG['epochs']} epochs)...")
        print(f"[CTGAN] Expected time: 5–15 minutes on CPU.\n")

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df_ctgan)
        metadata.update_column(column_name="__label__", sdtype="categorical")

        synthesizer = CTGANSynthesizer(
            metadata,
            epochs            = CTGAN_CONFIG["epochs"],
            batch_size        = CTGAN_CONFIG["batch_size"],
            generator_dim     = CTGAN_CONFIG["generator_dim"],
            discriminator_dim = CTGAN_CONFIG["discriminator_dim"],
            verbose           = CTGAN_CONFIG["verbose"],
            cuda              = False,
        )
        synthesizer.fit(df_ctgan)
        synthesizer.save(model_path)
        print(f"\n[CTGAN] Model saved → {model_path}\n")

    # ── Generate synthetic rows and filter cold-start ─────────────────────────
    n_generate = n_synthetic * 10
    print(f"[CTGAN] Generating {n_generate} synthetic rows...")
    synth_df = synthesizer.sample(num_rows=n_generate)

    # MinMaxScaler maps tenure to [0,1]; cold-start = ≤ 2 months → ≈ 2/72 + buffer.
    norm_threshold = COLD_START_TENURE_THRESHOLD / 72.0 + 0.01

    if tenure_col_name not in synth_df.columns:
        raise ValueError(
            f"Tenure column '{tenure_col_name}' not found in CTGAN output.\n"
            f"Available columns: {list(synth_df.columns)}"
        )

    cold_mask  = synth_df[tenure_col_name] <= norm_threshold
    synth_cold = synth_df[cold_mask].copy()

    print(f"[CTGAN] Tenure col: '{tenure_col_name}' | norm threshold: {norm_threshold:.4f}")
    print(f"[CTGAN] After cold-start filter: {len(synth_cold)} / {n_generate} "
          f"({100*len(synth_cold)/n_generate:.1f}%)")

    n_to_use = min(n_synthetic, len(synth_cold))
    if len(synth_cold) < n_synthetic:
        print(f"[CTGAN] WARNING: Only {len(synth_cold)} cold-start rows available. Using all.")

    synth_sample = synth_cold.sample(n=n_to_use, random_state=SEED).reset_index(drop=True)

    X_synth = synth_sample[feat_names].values.astype(np.float32)
    y_synth = synth_sample["__label__"].values.astype(int)

    # CTGAN can produce values slightly outside [0,1]; clip back.
    X_synth = np.clip(X_synth, 0.0, 1.0)

    if X_synth.shape[1] != X_real.shape[1]:
        raise ValueError(
            f"Feature count mismatch: synth={X_synth.shape[1]}, real={X_real.shape[1]}"
        )

    # ── Combine real + synthetic ──────────────────────────────────────────────
    X_aug = np.vstack([X_real, X_synth])
    y_aug = np.concatenate([y_real, y_synth])

    rng         = np.random.default_rng(seed=SEED)
    shuffle_idx = rng.permutation(len(X_aug))
    X_aug       = X_aug[shuffle_idx]
    y_aug       = y_aug[shuffle_idx]

    print(f"\n[CTGAN] Real:      {n_real} rows | churn: {y_real.mean()*100:.1f}%")
    print(f"[CTGAN] Synthetic: {n_to_use} rows | churn: {y_synth.mean()*100:.1f}%")
    print(f"[CTGAN] Combined:  {len(X_aug)} rows | churn: {y_aug.mean()*100:.1f}%")

    # ── Rebalance if synthetic churn rate deviates too much ───────────────────
    real_churn  = y_real.mean()
    synth_churn = y_synth.mean()
    gap         = abs(real_churn - synth_churn)

    if gap > REBALANCE_THRESHOLD:
        print(f"\n[CTGAN] Churn gap {gap*100:.1f}pp > {REBALANCE_THRESHOLD*100:.0f}pp "
              f"— rebalancing synthetic portion...")

        synth_churn_idx   = np.where(y_synth == 1)[0]
        synth_nochurn_idx = np.where(y_synth == 0)[0]
        n_nochurn         = len(synth_nochurn_idx)
        n_churn_target    = int(round(n_nochurn * real_churn / (1 - real_churn)))

        churn_oversample_idx = rng.choice(
            synth_churn_idx, size=n_churn_target, replace=True
        )
        X_synth_rebal = np.vstack([
            X_synth[synth_nochurn_idx],
            X_synth[churn_oversample_idx]
        ])
        y_synth_rebal = np.concatenate([
            y_synth[synth_nochurn_idx],
            y_synth[churn_oversample_idx]
        ])

        X_aug = np.vstack([X_real, X_synth_rebal])
        y_aug = np.concatenate([y_real, y_synth_rebal])

        shuffle_idx = rng.permutation(len(X_aug))
        X_aug       = X_aug[shuffle_idx]
        y_aug       = y_aug[shuffle_idx]

        print(f"[CTGAN] After rebalancing: {len(X_aug)} rows | "
              f"churn: {y_aug.mean()*100:.1f}%")

    # ── Save augmented file ───────────────────────────────────────────────────
    out_path = os.path.join(mpmn_dir, "train_augmented.npz")
    np.savez(out_path, X=X_aug, y=y_aug)

    print(f"\n[CTGAN] Saved → {out_path}")
    print(f"[CTGAN] Feature set unchanged from feature_names.json — "
          f"{len(feat_names)} features.")
    print(f"[CTGAN] val.npz and test.npz are untouched — real data only.")

    # ── TSTR quality check ────────────────────────────────────────────────────
    print(f"\n[CTGAN] Running TSTR quality check...")

    test_d = np.load(test_path)
    X_test = test_d["X"].astype(np.float32)
    y_test = test_d["y"].astype(int)

    rf_real = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    rf_real.fit(X_real, y_real)
    trtr_auc = roc_auc_score(y_test, rf_real.predict_proba(X_test)[:, 1]) * 100

    rf_aug = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    rf_aug.fit(X_aug, y_aug)
    tstr_auc = roc_auc_score(y_test, rf_aug.predict_proba(X_test)[:, 1]) * 100

    gap_auc = tstr_auc - trtr_auc
    print(f"  TRTR AUC (train real, test real) : {trtr_auc:.2f}%  [reference]")
    print(f"  TSTR AUC (train aug,  test real) : {tstr_auc:.2f}%  [quality check]")
    print(f"  Gap: {gap_auc:+.2f}%")

    if abs(gap_auc) <= 5:
        print(f"  [GOOD] AUC gap <= 5%")
    elif abs(gap_auc) <= 10:
        print(f"  [ACCEPTABLE] AUC gap 5–10%")
    else:
        print(f"  [WARNING] AUC gap > 10% — consider retraining CTGAN with more epochs.")

    print(f"\n{'='*60}")
    print(f"  AUGMENTATION COMPLETE — {dataset}")
    print(f"  Features used  : {len(feat_names)}")
    print(f"  Real           : {n_real}")
    print(f"  Synthetic      : {n_to_use}")
    print(f"  Combined       : {len(X_aug)}")
    print(f"  Churn (aug)    : {y_aug.mean()*100:.1f}%")
    print(f"  Saved          : {out_path}")
    print(f"{'='*60}\n")


# Replace with your directory paths before running.
if __name__ == "__main__":
    run(
        dataset="telco2",
        mpmn_dir=r"C:\Users\T-Plug\Desktop\ML Mini Project\hybrid_churn_prediction_project\datasets\processed\telco2\mpmn_ready",
        model_dir=r"C:\Users\T-Plug\Desktop\ML Mini Project\hybrid_churn_prediction_project\models\ctgan",
    )