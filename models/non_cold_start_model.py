import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUPS
# ─────────────────────────────────────────────────────────────────────────────
# Column names match the actual gatefuse_ready CSV output exactly.
# Group sizes must match what EstablishedFeatureEngineer produces:
#   Bank    : Profile=7, Contract=1, Billing=3, Usage=3  → total 14
#   Telco_1 : Profile=5, Contract=6, Billing=6, Usage=5  → total 22 (excl. Churn)
#   Telco_2 : Profile=4, Contract=4, Billing=6, Usage=1  → total 15 (excl. Churn)

FEATURE_GROUPS = {

    "bank": {
        "Profile":  [
            "CreditScore",
            "Age",
            "Geography_Germany",
            "Geography_Spain",
            "Geography_France",
            "Gender",
            "EstimatedSalary",
            "Satisfaction Score",
        ],
        "Contract": [
            "Tenure",
        ],
        "Billing":  [
            "Balance",
            "has_zero_balance",
            "Point Earned",
        ],
        "Usage":    [
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
        ],
    },

    "telco1": {
        "Profile":  [
            "Gender",
            "Age",
            "Married",
            "Number of Dependents",
            "Satisfaction Score",
        ],
        "Contract": [
            "Tenure in Months",
            "Contract",
            "Offer_Offer B",
            "Offer_Offer C",
            "Offer_Offer D",
            "Offer_Offer E",
        ],
        "Billing":  [
            "Monthly Charge",
            "Total Charges",
            "Total Refunds",
            "Total Extra Data Charges",
            "Total Long Distance Charges",
            "Paperless Billing",
        ],
        "Usage":    [
            "Phone Service",
            "Internet Type_DSL",
            "Internet Type_Fiber Optic",
            "Internet Type_No Internet Service",
            "Avg Monthly GB Download",
        ],
    },

    "telco2": {
        "Profile":  [
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
        ],
        "Contract": [
            "tenure",
            "Contract",
            "InternetService_Fiber optic",
            "InternetService_No",
        ],
        "Billing":  [
            "MonthlyCharges",
            "TotalCharges",
            "PaperlessBilling",
            "PaymentMethod_Credit card (automatic)",
            "PaymentMethod_Electronic check",
            "PaymentMethod_Mailed check",
        ],
        "Usage":    [
            "PhoneService",
        ],
    },
}


def route_feature_groups(dataset_name: str) -> dict:
    if dataset_name not in FEATURE_GROUPS:
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Must be one of: {list(FEATURE_GROUPS.keys())}"
        )
    return FEATURE_GROUPS[dataset_name]


# ─────────────────────────────────────────────────────────────────────────────
# BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────────────────────


class GroupEncoder(nn.Module):
    """Per-group encoder with residual connection."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc       = nn.Linear(input_dim, hidden_dim)
        self.residual = (
            nn.Identity()
            if input_dim == hidden_dim
            else nn.Linear(input_dim, hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.fc(x)) + self.residual(x)


class GroupedFeatureEncoder(nn.Module):
    """Applies a GroupEncoder independently to each feature group."""

    def __init__(self, dataset_name: str, feature_dims: dict, hidden_dim: int = 64):
        super().__init__()
        self.groups   = route_feature_groups(dataset_name)
        self.encoders = nn.ModuleDict({
            group: GroupEncoder(feature_dims[group], hidden_dim)
            for group in self.groups
        })

    def forward(self, grouped_inputs: dict) -> dict:
        return {
            group: encoder(grouped_inputs[group])
            for group, encoder in self.encoders.items()
        }


class GroupAttention(nn.Module):
    """Softmax attention over group embeddings."""

    def __init__(self, embedding_dim: int, num_groups: int = 4):
        super().__init__()
        self.attention_fc = nn.ModuleDict({
            group: nn.Linear(embedding_dim, 1)
            for group in ["Profile", "Contract", "Billing", "Usage"]
        })

    def forward(self, group_embeddings: dict):
        scores = torch.cat(
            [self.attention_fc[g](group_embeddings[g])
             for g in ["Profile", "Contract", "Billing", "Usage"]],
            dim=1
        )
        weights = F.softmax(scores, dim=1)
        weighted = {
            g: group_embeddings[g] * weights[:, i].unsqueeze(1)
            for i, g in enumerate(["Profile", "Contract", "Billing", "Usage"])
        }
        return weighted, weights


class GMUGating(nn.Module):
    """Gated Multimodal Unit — sigmoid gate per group."""

    def __init__(self, embedding_dim: int, num_groups: int = 4):
        super().__init__()
        self.gate_fc = nn.ModuleDict({
            group: nn.Linear(embedding_dim, 1)
            for group in ["Profile", "Contract", "Billing", "Usage"]
        })

    def forward(self, weighted_embeddings: dict):
        gated     = {}
        gate_vals = []
        for g in ["Profile", "Contract", "Billing", "Usage"]:
            x        = weighted_embeddings[g]
            gate     = torch.sigmoid(self.gate_fc[g](x))
            gated[g] = x * gate
            gate_vals.append(gate)
        return gated, torch.cat(gate_vals, dim=1)


class GroupFusion(nn.Module):
    """Concatenates all group embeddings and projects to fused_dim."""

    def __init__(self, embedding_dim: int, fused_dim: int = 128, num_groups: int = 4):
        super().__init__()
        self.fusion_fc  = nn.Linear(embedding_dim * num_groups, fused_dim)
        self.activation = nn.ReLU()

    def forward(self, gated_embeddings: dict) -> torch.Tensor:
        concatenated = torch.cat(
            [gated_embeddings[g] for g in ["Profile", "Contract", "Billing", "Usage"]],
            dim=1
        )
        return self.activation(self.fusion_fc(concatenated))


class InteractionModeling(nn.Module):
    """Low-rank Hadamard interaction on the fused vector."""

    def __init__(self, fused_dim: int = 128, interaction_dim: int = 64):
        super().__init__()
        self.proj1      = nn.Linear(fused_dim, interaction_dim)
        self.proj2      = nn.Linear(fused_dim, interaction_dim)
        self.output_fc  = nn.Linear(interaction_dim, interaction_dim)
        self.activation = nn.ReLU()

    def forward(self, fused_vector: torch.Tensor) -> torch.Tensor:
        v1 = self.proj1(fused_vector)
        v2 = self.proj2(fused_vector)
        return self.activation(self.output_fc(v1 * v2))


class ChurnClassifier(nn.Module):
    """
    Final binary classifier — returns raw logits.
    Sigmoid is applied externally (BCEWithLogitsLoss handles it during
    training; torch.sigmoid() is applied at inference time).
    """

    def __init__(self, input_dim: int = 64):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)   # raw logit — NO sigmoid here


# ─────────────────────────────────────────────────────────────────────────────
# MAIN MODEL
# ─────────────────────────────────────────────────────────────────────────────


class NonColdStartModel(nn.Module):
    """
    GATEFuse — Grouped-Attention Feature Encoder with GMU Gating,
    Group Fusion, Interaction Modeling, and Binary Classifier.

    Forward pass returns raw logits (no sigmoid).
    Use BCEWithLogitsLoss during training.
    Apply torch.sigmoid() at inference time.
    """

    def __init__(
        self,
        dataset_name:    str,
        feature_dims:    dict,
        hidden_dim:      int = 64,
        fused_dim:       int = 128,
        interaction_dim: int = 64,
    ):
        super().__init__()

        # Sanitise feature_dims: convert list-of-col-names → int if needed
        self.feature_dims = {
            k: (len(v) if isinstance(v, list) else v)
            for k, v in feature_dims.items()
        }
        self.groups      = route_feature_groups(dataset_name)
        self.group_names = list(self.groups.keys())

        self.encoder     = GroupedFeatureEncoder(dataset_name, self.feature_dims, hidden_dim)
        self.attention   = GroupAttention(hidden_dim, num_groups=len(self.groups))
        self.gmu         = GMUGating(hidden_dim, num_groups=len(self.groups))
        self.fusion      = GroupFusion(hidden_dim, fused_dim=fused_dim, num_groups=len(self.groups))
        self.interaction = InteractionModeling(fused_dim, interaction_dim)
        self.classifier  = ChurnClassifier(input_dim=interaction_dim)

    def split_into_groups(self, X: torch.Tensor) -> dict:
        grouped = {}
        start   = 0
        for group in self.group_names:
            dim            = self.feature_dims[group]
            grouped[group] = X[:, start:start + dim]
            start         += dim
        return grouped

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        grouped     = self.split_into_groups(X)
        encoded     = self.encoder(grouped)
        weighted, _ = self.attention(encoded)
        gated,   _  = self.gmu(weighted)
        fused       = self.fusion(gated)
        interacted  = self.interaction(fused)
        logit       = self.classifier(interacted)
        return logit   # raw logit