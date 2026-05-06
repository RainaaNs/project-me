"""
scripts/cold_start_train.py — Cold-Start Training Script (MPMN+VML)

Reads directly from .npz files produced by feature_engineering.py.

Expected inputs:
    train_path : path to mpmn_ready/train.npz
    val_path   : path to mpmn_ready/val.npz

Each .npz file contains:
    X : numpy array of shape (n_samples, n_features)
    y : numpy array of shape (n_samples,) with binary labels

Dependencies (must be importable from project root):
    models/cold_start_model.py → MPMN, compute_vml_loss, DATASET_CONFIGS
"""

import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import deque
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score
)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.cold_start_model import MPMN, compute_vml_loss, DATASET_CONFIGS

# ── Global training constants ─────────────────────────────────────────────────
EPOCHS         = 150
TRAIN_EPISODES = 500
VAL_EPISODES   = 150
GRAD_CLIP      = 1.0
# ─────────────────────────────────────────────────────────────────────────────


class EpisodeDataset(Dataset):
    """
    Episodic task generator.
    balanced=True  → equal class sizes in query (training)
    balanced=False → real class proportions in query (evaluation)
    """

    def __init__(self, X, y, n_support=5, n_query=15, n_episodes=500, balanced=True):
        self.X          = torch.tensor(X, dtype=torch.float32)
        self.y          = torch.tensor(y, dtype=torch.long)
        self.n_support  = n_support
        self.n_query    = n_query
        self.n_episodes = n_episodes
        self.balanced   = balanced
        self.idx_0      = np.where(y == 0)[0]
        self.idx_1      = np.where(y == 1)[0]
        min_needed      = n_support + (n_query if balanced else 1)
        assert len(self.idx_0) >= min_needed, (
            f"Class 0 has {len(self.idx_0)} samples — need at least {min_needed}"
        )
        assert len(self.idx_1) >= min_needed, (
            f"Class 1 has {len(self.idx_1)} samples — need at least {min_needed}"
        )

    def __len__(self):
        return self.n_episodes

    def __getitem__(self, idx):
        sup_idx_0 = np.random.choice(self.idx_0, self.n_support, replace=False)
        sup_idx_1 = np.random.choice(self.idx_1, self.n_support, replace=False)
        avail_0   = np.setdiff1d(self.idx_0, sup_idx_0)
        avail_1   = np.setdiff1d(self.idx_1, sup_idx_1)

        if self.balanced:
            qry_idx_0 = np.random.choice(avail_0, min(self.n_query, len(avail_0)), replace=False)
            qry_idx_1 = np.random.choice(avail_1, min(self.n_query, len(avail_1)), replace=False)
        else:
            total = len(avail_0) + len(avail_1)
            n     = min(self.n_query, total)
            n_q0  = min(max(1, round(n * len(avail_0) / total)), len(avail_0))
            n_q1  = min(max(1, n - n_q0), len(avail_1))
            qry_idx_0 = np.random.choice(avail_0, n_q0, replace=False)
            qry_idx_1 = np.random.choice(avail_1, n_q1, replace=False)

        return (
            torch.cat([self.X[sup_idx_0], self.X[sup_idx_1]]),
            torch.cat([self.y[sup_idx_0], self.y[sup_idx_1]]),
            torch.cat([self.X[qry_idx_0], self.X[qry_idx_1]]),
            torch.cat([self.y[qry_idx_0], self.y[qry_idx_1]]),
        )


def train(
    train_path: str,
    val_path: str,
    dataset: str,
    save_path: str,
) -> dict:
    """
    Train the MPMN+VML cold-start model.

    Args:
        train_path : Path to mpmn_ready/train.npz
        val_path   : Path to mpmn_ready/val.npz
        dataset    : One of 'telco1', 'telco2', 'bank'
        save_path  : Where to save the model checkpoint (.pth)

    Returns:
        dict: val_auc, val_f1, val_accuracy, val_precision, val_recall, temperature
    """

    if dataset not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Must be one of: {list(DATASET_CONFIGS.keys())}"
        )

    cfg = DATASET_CONFIGS[dataset]

    print(f"\n{'='*60}")
    print(f"  COLD-START TRAINING (MPMN+VML) — {dataset}")
    print(f"  hidden={cfg['hidden_dim']} | latent={cfg['latent_dim']} | "
          f"dropout={cfg['dropout']} | n_support={cfg['n_support']}")
    print(f"  threshold={cfg['decision_threshold']} | patience={cfg['patience']}")
    print(f"{'='*60}\n")

    # ── Load .npz files ───────────────────────────────────────────────────────
    # Prefer train_augmented.npz if it exists (produced by ctgan_augment.py).
    # Falls back to train.npz if augmentation has not been run yet.
    augmented_path = train_path.replace("train.npz", "train_augmented.npz")
    if os.path.exists(augmented_path):
        print(f"[CS-Train] Using augmented training pool: {augmented_path}")
        train_data = np.load(augmented_path)
    else:
        print(f"[CS-Train] No augmented file found — using: {train_path}")
        print(f"[CS-Train] Run ctgan_augment.py to improve cold-start pool size.")
        train_data = np.load(train_path)

    val_data = np.load(val_path)

    X_train, y_train = train_data['X'].astype(np.float32), train_data['y'].astype(int)
    X_val,   y_val   = val_data['X'].astype(np.float32),   val_data['y'].astype(int)

    input_dim = X_train.shape[1]
    print(f"[CS-Train] Train: {len(X_train)} | Val: {len(X_val)} | Features: {input_dim}")
    print(f"[CS-Train] Churn rate — Train: {y_train.mean()*100:.1f}% | "
          f"Val: {y_val.mean()*100:.1f}%\n")

    # ── Episode datasets ──────────────────────────────────────────────────────
    train_ds = EpisodeDataset(
        X_train, y_train, cfg['n_support'], cfg['n_query_train'],
        TRAIN_EPISODES, balanced=True
    )
    val_ds = EpisodeDataset(
        X_val, y_val, cfg['n_support'], cfg['n_query_eval'],
        VAL_EPISODES, balanced=False
    )

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=1)

    # ── Model ─────────────────────────────────────────────────────────────────
    model    = MPMN(input_dim, cfg['hidden_dim'], cfg['latent_dim'], cfg['dropout'])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[CS-Train] Model parameters: {n_params:,}")

    optimizer = optim.Adam(
        model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    anneal_epochs  = max(1, int(EPOCHS * cfg['anneal_frac']))
    best_val_loss  = float('inf')
    patience_count = 0
    best_state     = None
    history = {
        'train_loss': [], 'val_loss': [], 'val_loss_smooth': [],
        'ce_loss': [], 'kl_loss': [], 'beta': [], 'temperature': []
    }
    val_window = deque(maxlen=cfg['smooth_window'])

    print(f"[CS-Train] Training up to {EPOCHS} epochs | "
          f"patience={cfg['patience']} | smooth_window={cfg['smooth_window']}\n")

    for epoch in range(EPOCHS):
        beta = min(cfg['beta_max'], cfg['beta_max'] * (epoch + 1) / anneal_epochs)

        # Train
        model.train()
        t_loss = t_ce = t_kl = 0.0
        for batch in train_loader:
            sup_X, sup_y, qry_X, qry_y = [b.squeeze(0) for b in batch]
            logits, q_mean, q_logvar, sup_means, sup_logvars, _ = model(sup_X, sup_y, qry_X)
            loss, ce, kl = compute_vml_loss(
                logits, qry_y, q_mean, q_logvar, sup_means, sup_logvars, beta
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            t_loss += loss.item(); t_ce += ce.item(); t_kl += kl.item()

        scheduler.step()
        n = len(train_loader)
        history['train_loss'].append(t_loss / n)
        history['ce_loss'].append(t_ce / n)
        history['kl_loss'].append(t_kl / n)
        history['beta'].append(beta)

        # Validate
        model.eval()
        v_loss = 0.0; cur_temp = 0.0
        with torch.no_grad():
            for batch in val_loader:
                sup_X, sup_y, qry_X, qry_y = [b.squeeze(0) for b in batch]
                logits, q_mean, q_logvar, sup_means, sup_logvars, temp = model(
                    sup_X, sup_y, qry_X
                )
                loss, _, _ = compute_vml_loss(
                    logits, qry_y, q_mean, q_logvar, sup_means, sup_logvars, beta
                )
                v_loss   += loss.item()
                cur_temp += temp.item()

        avg_val  = v_loss  / len(val_loader)
        avg_temp = cur_temp / len(val_loader)
        history['val_loss'].append(avg_val)
        history['temperature'].append(avg_temp)

        val_window.append(avg_val)
        smooth_val = np.mean(val_window)
        history['val_loss_smooth'].append(smooth_val)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:>3}/{EPOCHS} | "
                  f"train:{history['train_loss'][-1]:.4f} | "
                  f"val:{avg_val:.4f} | val_smooth:{smooth_val:.4f} | "
                  f"T:{avg_temp:.3f} | beta:{beta:.5f}")

        if smooth_val < best_val_loss:
            best_val_loss  = smooth_val
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= cfg['patience']:
                print(f"  Early stop at epoch {epoch+1} "
                      f"(smoothed val stagnant for {cfg['patience']} epochs)")
                break

    # ── Restore best checkpoint ───────────────────────────────────────────────
    model.load_state_dict(best_state)
    final_temp = (torch.nn.functional.softplus(model.log_temp) + 0.01).item()
    print(f"\n[CS-Train] Best checkpoint restored | "
          f"smoothed val loss: {best_val_loss:.4f} | temperature: {final_temp:.3f}")

    # ── Validation evaluation ─────────────────────────────────────────────────
    threshold = cfg['decision_threshold']
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            sup_X, sup_y, qry_X, qry_y = [b.squeeze(0) for b in batch]
            logits, _, _, _, _, _ = model(sup_X, sup_y, qry_X)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs >= threshold).long()
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(qry_y.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    val_auc       = roc_auc_score(all_labels, all_probs)
    val_f1        = f1_score(all_labels, all_preds, average='macro')
    val_accuracy  = (all_preds == all_labels).mean()
    val_precision = precision_score(all_labels, all_preds, zero_division=0)
    val_recall    = recall_score(all_labels, all_preds, zero_division=0)

    print(f"\n[CS-Train] Validation Results (threshold={threshold}):")
    print(f"  AUC: {val_auc*100:.2f}% | F1: {val_f1*100:.2f}% | "
          f"Acc: {val_accuracy*100:.2f}% | "
          f"Precision: {val_precision*100:.2f}% | Recall: {val_recall*100:.2f}%")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim':        input_dim,
        'config':           cfg,
        'dataset':          dataset,
        'history':          history,
        'results': {
            'val_auc':       val_auc,
            'val_f1':        val_f1,
            'val_accuracy':  val_accuracy,
            'val_precision': val_precision,
            'val_recall':    val_recall,
            'temperature':   final_temp,
        }
    }, save_path)
    print(f"[CS-Train] Model saved → {save_path}\n")

    return {
        'val_auc':       val_auc,
        'val_f1':        val_f1,
        'val_accuracy':  val_accuracy,
        'val_precision': val_precision,
        'val_recall':    val_recall,
        'temperature':   final_temp,
    }

if __name__ == "__main__":
    train(
        train_path=r"C:\Users\T-Plug\Desktop\ML Mini Project\hybrid_churn_prediction_project\datasets\processed\telco2\mpmn_ready\train.npz",
        val_path  =r"C:\Users\T-Plug\Desktop\ML Mini Project\hybrid_churn_prediction_project\datasets\processed\telco2\mpmn_ready\val.npz",
        dataset   ="bank",
        save_path =r"C:\Users\T-Plug\Desktop\ML Mini Project\hybrid_churn_prediction_project\models\cold_start\telco2.pth",
    )