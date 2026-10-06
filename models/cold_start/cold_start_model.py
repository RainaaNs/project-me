import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# DATASET-SPECIFIC HYPERPARAMETER CONFIGS
# ─────────────────────────────────────────────────────────────────────────────
#
# v4 changes per dataset:
#
#  All datasets:
#    • patience raised 20 → 35. Early stopping was reacting to val loss NOISE
#      rather than true degradation. Episodic val loss has high variance because
#      each epoch re-samples different support/query sets. A single bad epoch
#      was triggering patience when the model hadn't actually deteriorated.
#      Smoothed early stopping (5-epoch rolling average) also added in notebook.
#
#    • temperature scaling added to MPMN (learnable parameter T).
#      Converts raw negative-distance logits → logits/T before softmax.
#      Lets the model learn how sharp/soft its predictions should be,
#      which is critical for imbalanced datasets like bank.
#
#  telco2:
#    • n_support raised 5 → 8. With only 20 features and heavy structural
#      zeros, more support examples give the prototype more signal.
#    • weight_decay raised to 2e-3 for stronger regularisation.
#
#  bank:
#    • decision_threshold added: 0.30 instead of default 0.50.
#      With 25% real churn rate, a threshold of 0.50 is too conservative —
#      the model must be very confident before predicting churn, causing it
#      to miss most churners. Lowering to 0.30 matches the prior better.
#      Threshold was chosen as slightly above the real churn rate (0.25)
#      to balance precision and recall.
#    • patience raised further to 40 — bank val loss was oscillating with
#      no clear trend (genuine noise from small minority class), not overfitting.
#
# v5b changes per dataset (augmented pool enables larger support sets):
#
#  Rationale: CTGAN augmentation grew the cold-start training pool 3–4x.
#  With a larger pool, episodes can draw more support examples per class
#  without exhausting the pool, yielding more stable class prototypes.
#  n_query_train stays at 15 — gradient signal from 15 queries per episode
#  is already sufficient; increasing adds compute without benefit.
#
#  telco1: n_support 5 → 10. Pool grew 4x (508 → 2032). Churn prototype
#    was the unstable one — more support examples directly stabilise it.
#
#  telco2: n_support 8 → 12. Pool grew 3x (598 → 1800). Structural zero
#    problem means each support example carries less signal; more examples
#    compensate. decision_threshold raised 0.45 → 0.48 — augmented churn
#    rate (53.8%) is closer to balanced; original 0.45 was slightly aggressive.
#
#  bank: n_support 10 → 15. Pool grew 2.5x (1171 → 3000). Minority churn
#    class (25%) needs more prototype examples for stable representation.
#    decision_threshold raised 0.30 → 0.38 — v5 showed T=5.165 caused
#    over-prediction at 0.30; raising threshold corrects precision/recall
#    balance without touching the architecture.

DATASET_CONFIGS = {
    "telco1": {
        "hidden_dim": 64,
        "latent_dim": 32,
        "dropout": 0.35,
        "n_support": 10,        # v5b: ↑ was 5 — pool grew 4x (508→2032)
        "n_query_train": 15,
        "n_query_eval": 30,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "beta_max": 1e-3,
        "anneal_frac": 0.1,
        "patience": 35,
        "smooth_window": 5,     # rolling average window for val loss
        "decision_threshold": 0.50,  # standard threshold (balanced dataset)
    },
    "telco2": {
        "hidden_dim": 32,
        "latent_dim": 16,
        "dropout": 0.5,
        "n_support": 12,        # v5b: ↑ was 8 — pool grew 3x (598→1800)
        "n_query_train": 15,
        "n_query_eval": 30,
        "lr": 5e-4,
        "weight_decay": 2e-3,
        "beta_max": 1e-3,
        "anneal_frac": 0.1,
        "patience": 35,
        "smooth_window": 5,
        "decision_threshold": 0.48,  # v5b: ↑ was 0.45 — augmented churn rate closer to balanced
    },
    "bank": {
        "hidden_dim": 64,
        "latent_dim": 32,
        "dropout": 0.4,
        "n_support": 15,        # v5b: ↑ was 10 — pool grew 2.5x (1171→3000)
        "n_query_train": 15,
        "n_query_eval": 30,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "beta_max": 1e-3,
        "anneal_frac": 0.1,
        "patience": 40,
        "smooth_window": 7,     # wider smoothing for noisier bank signal
        "decision_threshold": 0.38,  # v5b: ↑ was 0.30 — corrects over-prediction from high T
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# ENCODER
# ─────────────────────────────────────────────────────────────────────────────


class VariationalEncoder(nn.Module):
    """
    Maps input features → Probabilistic Latent Distribution (mean, logvar).

    Uses LayerNorm instead of BatchNorm1d — invariant to batch size, which is
    essential for episodic meta-learning where support and query sets differ
    in size every call.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        dropout: float = 0.35,
    ):
        super().__init__()

        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.fc_mean   = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Start logvar near -2 → sigma ≈ 0.37. Prevents KL explosion early.
        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.constant_(self.fc_logvar.bias, -2.0)

    def forward(self, x: torch.Tensor):
        h      = self.shared_encoder(x)
        mean   = self.fc_mean(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-8.0, max=4.0)
        return mean, logvar


# ─────────────────────────────────────────────────────────────────────────────
# MPMN
# ─────────────────────────────────────────────────────────────────────────────


class MPMN(nn.Module):
    """
    Meta-Prototypical Matching Network with Variational Metric Learning (VML).

    v4 addition: learnable temperature scaling.
    ──────────────────────────────────────────
    logits = -dists / T   where T is a learnable scalar > 0.

    Raw distances can be on very different scales depending on the latent
    space geometry. Without temperature scaling, the softmax can be either
    too peaked (overconfident) or too flat (underconfident), causing poor
    calibration especially for imbalanced datasets.

    T is initialised to 1.0 (no effect) and learned jointly with the encoder.
    A T > 1 softens predictions; T < 1 sharpens them.
    The model learns whichever is appropriate for the dataset.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        dropout: float = 0.35,
    ):
        super().__init__()
        self.encoder    = VariationalEncoder(input_dim, hidden_dim, latent_dim, dropout)
        self.latent_dim = latent_dim

        # Learnable temperature — initialised to 1 (identity), constrained > 0
        # via softplus in forward so it never goes negative.
        self.log_temp = nn.Parameter(torch.zeros(1))

    # ── Reparameterization ────────────────────────────────────────────────────

    def reparameterize(
        self, mean: torch.Tensor, logvar: torch.Tensor, num_samples: int = 1
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(mean.size(0), num_samples, mean.size(1), device=mean.device)
        return mean.unsqueeze(1) + std.unsqueeze(1) * eps

    # ── Prototype computation ─────────────────────────────────────────────────

    def compute_prototypes(
        self, support_X: torch.Tensor, support_y: torch.Tensor, num_samples: int = 5
    ):
        """
        Encode support set → per-class prototype distributions.

        Proto variance via Law of Total Variance:
            Var[Z_class] = E[sigma²] + Var[mu]   (aleatoric + epistemic)
        """
        sup_means, sup_logvars = self.encoder(support_X)
        z_avg = self.reparameterize(sup_means, sup_logvars, num_samples).mean(dim=1)

        unique_labels          = torch.unique(support_y)
        prototypes, proto_vars = [], []

        for label in unique_labels:
            mask     = support_y == label
            class_z  = z_avg[mask]
            class_lv = sup_logvars[mask]
            class_mu = sup_means[mask]

            if class_z.size(0) == 0:
                continue

            proto      = class_z.mean(dim=0)
            aleatoric  = torch.exp(class_lv).mean(dim=0)
            epistemic  = (
                class_mu.var(dim=0, unbiased=False)
                if class_mu.size(0) > 1
                else torch.zeros_like(aleatoric)
            )

            prototypes.append(proto)
            proto_vars.append(aleatoric + epistemic)

        return (
            torch.stack(prototypes),
            torch.stack(proto_vars),
            sup_means,
            sup_logvars,
            unique_labels,
        )

    # ── Distance ──────────────────────────────────────────────────────────────

    def variational_distance(
        self,
        query_mean:   torch.Tensor,
        query_logvar: torch.Tensor,
        prototype:    torch.Tensor,
        proto_var:    torch.Tensor,
        num_samples:  int = 10,
    ) -> torch.Tensor:
        """
        Uncertainty-weighted Mahalanobis-style distance:
            dist = E_z~q [ Σ_d  (z_d − proto_d)² / (proto_var_d + query_var_d) ]
        """
        query_samples = self.reparameterize(query_mean, query_logvar, num_samples)

        query_var = torch.exp(query_logvar)
        total_var = query_var.unsqueeze(1) + proto_var.unsqueeze(0).unsqueeze(0) + 1e-6

        diff        = query_samples - prototype.unsqueeze(0).unsqueeze(0)
        weighted_sq = (diff ** 2) / total_var

        return weighted_sq.sum(dim=2).mean(dim=1)   # [n_query]

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        support_X:   torch.Tensor,
        support_y:   torch.Tensor,
        query_X:     torch.Tensor,
        num_samples: int = 10,
    ):
        """
        Returns
        -------
        logits      : [n_query, n_classes]   temperature-scaled negative distances
        q_mean      : [n_query, latent]
        q_logvar    : [n_query, latent]
        sup_means   : [n_support, latent]
        sup_logvars : [n_support, latent]
        temperature : scalar float (for logging)
        """
        prototypes, proto_vars, sup_means, sup_logvars, _ = self.compute_prototypes(
            support_X, support_y, num_samples
        )

        q_mean, q_logvar = self.encoder(query_X)

        n_classes = prototypes.size(0)
        dists     = torch.zeros(query_X.size(0), n_classes, device=query_X.device)

        for c in range(n_classes):
            dists[:, c] = self.variational_distance(
                q_mean, q_logvar, prototypes[c], proto_vars[c], num_samples
            )

        # Temperature scaling: T = softplus(log_temp) + 0.01 keeps T > 0
        temperature = F.softplus(self.log_temp) + 0.01
        logits      = -dists / temperature

        return logits, q_mean, q_logvar, sup_means, sup_logvars, temperature


# ─────────────────────────────────────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────────────────────────────────────


def compute_vml_loss(
    logits, query_y, q_mean, q_logvar, sup_means, sup_logvars, beta=0.0
):
    """
    Loss = CrossEntropy + beta × KL(support + query → N(0, I))

    KL regularises BOTH encoder paths (support + query concatenated).
    beta annealed from 0 → beta_max via training loop.
    """
    ce_loss = F.cross_entropy(logits, query_y)

    all_means   = torch.cat([sup_means, q_mean],   dim=0)
    all_logvars = torch.cat([sup_logvars, q_logvar], dim=0)

    kl = (
        (-0.5 * (1 + all_logvars - all_means.pow(2) - all_logvars.exp()))
        .sum(dim=1)
        .mean()
    )

    return ce_loss + beta * kl, ce_loss, kl