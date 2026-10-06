"""
models/ablation_model.py — Configurable MPMN+VML for ablation studies

Reuses your existing VariationalEncoder (imported from cold_start_model.py)
so the encoder architecture — and therefore parameter count — stays fixed
across ablations. Only the *mechanisms built on top of the encoder* are
toggled, which is what makes the comparison fair: differences in
performance can be attributed to the removed mechanism, not to a smaller
network.

Three independent flags:

    use_variational            reparameterization sampling on/off.
                                 False → z = encoder mean (deterministic
                                 point estimate), no sampling noise.

    use_temperature             learnable softmax temperature on/off.
                                 False → logits = -dists (T fixed at 1).

    use_uncertainty_weighting   variance-weighted (Mahalanobis-style)
                                 distance on/off. False → plain squared
                                 Euclidean distance (denominator = 1).

Setting all three False collapses this exactly to a vanilla Prototypical
Network (Snell et al. 2017): deterministic embeddings, Euclidean distance,
unscaled softmax.

ABLATION_VARIANTS below defines the six configurations used in the study:
the full model, three single-component removals, a "no variational path
at all" variant, and the vanilla ProtoNet floor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.cold_start.cold_start_model import VariationalEncoder


class AblatableMPMN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        dropout: float = 0.35,
        use_variational: bool = True,
        use_temperature: bool = True,
        use_uncertainty_weighting: bool = True,
    ):
        super().__init__()
        self.encoder = VariationalEncoder(input_dim, hidden_dim, latent_dim, dropout)
        self.latent_dim = latent_dim

        self.use_variational = use_variational
        self.use_temperature = use_temperature
        self.use_uncertainty_weighting = use_uncertainty_weighting

        # Kept as a real nn.Parameter (not a python float) even when unused,
        # so state_dict shape is stable if you ever want to load/compare
        # checkpoints across variants. It's just never applied when
        # use_temperature=False.
        self.log_temp = nn.Parameter(torch.zeros(1))

    # ── Reparameterization ────────────────────────────────────────────────
    def reparameterize(self, mean, logvar, num_samples=1):
        if not self.use_variational:
            return mean.unsqueeze(1).expand(-1, num_samples, -1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(mean.size(0), num_samples, mean.size(1), device=mean.device)
        return mean.unsqueeze(1) + std.unsqueeze(1) * eps

    # ── Prototype computation ─────────────────────────────────────────────
    def compute_prototypes(self, support_X, support_y, num_samples=5):
        sup_means, sup_logvars = self.encoder(support_X)
        z_avg = self.reparameterize(sup_means, sup_logvars, num_samples).mean(dim=1)

        unique_labels = torch.unique(support_y)
        prototypes, proto_vars = [], []

        for label in unique_labels:
            mask = support_y == label
            class_z = z_avg[mask]
            class_lv = sup_logvars[mask]
            class_mu = sup_means[mask]

            if class_z.size(0) == 0:
                continue

            proto = class_z.mean(dim=0)

            if self.use_uncertainty_weighting:
                aleatoric = torch.exp(class_lv).mean(dim=0)
                epistemic = (
                    class_mu.var(dim=0, unbiased=False)
                    if class_mu.size(0) > 1
                    else torch.zeros_like(aleatoric)
                )
                proto_var = aleatoric + epistemic
            else:
                proto_var = torch.zeros(self.latent_dim, device=class_z.device)

            prototypes.append(proto)
            proto_vars.append(proto_var)

        return (
            torch.stack(prototypes),
            torch.stack(proto_vars),
            sup_means,
            sup_logvars,
            unique_labels,
        )

    # ── Distance ──────────────────────────────────────────────────────────
    def variational_distance(self, query_mean, query_logvar, prototype, proto_var, num_samples=10):
        query_samples = self.reparameterize(query_mean, query_logvar, num_samples)

        if self.use_uncertainty_weighting:
            query_var = torch.exp(query_logvar)
            total_var = query_var.unsqueeze(1) + proto_var.unsqueeze(0).unsqueeze(0) + 1e-6
        else:
            total_var = torch.ones_like(query_samples)

        diff = query_samples - prototype.unsqueeze(0).unsqueeze(0)
        weighted_sq = (diff ** 2) / total_var
        return weighted_sq.sum(dim=2).mean(dim=1)

    # ── Forward ───────────────────────────────────────────────────────────
    def forward(self, support_X, support_y, query_X, num_samples=10):
        prototypes, proto_vars, sup_means, sup_logvars, _ = self.compute_prototypes(
            support_X, support_y, num_samples
        )
        q_mean, q_logvar = self.encoder(query_X)

        n_classes = prototypes.size(0)
        dists = torch.zeros(query_X.size(0), n_classes, device=query_X.device)
        for c in range(n_classes):
            dists[:, c] = self.variational_distance(
                q_mean, q_logvar, prototypes[c], proto_vars[c], num_samples
            )

        if self.use_temperature:
            temperature = F.softplus(self.log_temp) + 0.01
            logits = -dists / temperature
        else:
            temperature = torch.tensor(1.0, device=query_X.device)
            logits = -dists

        return logits, q_mean, q_logvar, sup_means, sup_logvars, temperature


def compute_ablatable_loss(logits, query_y, q_mean, q_logvar, sup_means, sup_logvars, beta=0.0, use_kl=True):
    """Same as compute_vml_loss, with a use_kl switch. When False, beta is
    forced to zero regardless of the anneal schedule passed in — this is
    the 'No KL Regularization' ablation."""
    ce_loss = F.cross_entropy(logits, query_y)

    if not use_kl:
        zero = torch.tensor(0.0, device=logits.device)
        return ce_loss, ce_loss, zero

    all_means = torch.cat([sup_means, q_mean], dim=0)
    all_logvars = torch.cat([sup_logvars, q_logvar], dim=0)
    kl = (
        (-0.5 * (1 + all_logvars - all_means.pow(2) - all_logvars.exp()))
        .sum(dim=1)
        .mean()
    )
    return ce_loss + beta * kl, ce_loss, kl


# ─────────────────────────────────────────────────────────────────────────
# ABLATION LADDER
# ─────────────────────────────────────────────────────────────────────────
# Ordered from full model to vanilla ProtoNet floor. Each row after the
# first removes exactly one more mechanism than a "clean" single-component
# ablation would need, EXCEPT rows 2-4 which are single-component removals
# from the full model (for isolating each component's individual
# contribution), and rows 5-6 which stack removals (for showing the
# cumulative degradation down to the ProtoNet floor).
ABLATION_VARIANTS = {
    "Full (MPMN+VML)": dict(
        use_variational=True, use_temperature=True, use_uncertainty_weighting=True, use_kl=True,
    ),
    "No Temperature Scaling": dict(
        use_variational=True, use_temperature=False, use_uncertainty_weighting=True, use_kl=True,
    ),
    "No Uncertainty Weighting": dict(
        use_variational=True, use_temperature=True, use_uncertainty_weighting=False, use_kl=True,
    ),
    "No KL Regularization": dict(
        use_variational=True, use_temperature=True, use_uncertainty_weighting=True, use_kl=False,
    ),
    "No Variational Path (Point Estimate)": dict(
        use_variational=False, use_temperature=True, use_uncertainty_weighting=False, use_kl=False,
    ),
    "Vanilla ProtoNet (floor)": dict(
        use_variational=False, use_temperature=False, use_uncertainty_weighting=False, use_kl=False,
    ),
}
