import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# GROUP RULES
# ─────────────────────────────────────────────────────────────────────────────
# Keyword-based rules that assign each column to a semantic group at runtime.
# These rules are applied against the ACTUAL column names in the CSV, so they
# work regardless of which features survive correlation-based selection.
#
# Rules are evaluated in order: Profile → Contract → Billing → Usage.
# Any column not matched by any rule goes into a catch-all (Usage) to avoid
# silently dropping features.
#
# When adding a new dataset, add a new entry here following the same pattern.

GROUP_RULES = {
    # accurate
    "bank": {
        "Profile": [
            "Geography_", "Gender", "Age", "EstimatedSalary",
            "CreditScore", "Satisfaction Score",
        ],
        "Contract": [
            "Tenure", "Card Type",
        ],
        "Billing": [
            "Balance", "Point Earned",
        ],
        "Usage": [
            "NumOfProducts", "HasCrCard", "IsActiveMember", "Complain",
        ],
    },

# 
    "telco1": {
        "Profile": [
            "Gender", "Age", "Married", "Number of Dependents",
            "Satisfaction Score", "Under 30", "Senior Citizen", "Latitude", "Longitude"
        ],
        "Contract": [
            "Tenure in Months", "Offer_", "Contract_",
        ],
        "Billing": [
            "Monthly Charge", "Total Charges", "Total Refunds",
            "Total Extra Data Charges", "Total Long Distance Charges",
            "Paperless Billing", "Payment Method_",
            "Avg Monthly Long Distance Charges",
        ],
        "Usage": [
            "Phone Service", "Multiple Lines", "Internet Service",
            "Internet Type_", "Avg Monthly GB Download",
            "Online Security", "Online Backup", "Device Protection Plan",
            "Premium Tech Support", "Streaming TV", "Streaming Movies",
            "Streaming Music", "Unlimited Data", "Referred a Friend",
            "Dependents", "Number of Referrals", 
        ],
    },

    "telco2": {
        "Profile": [
            "gender", "SeniorCitizen_", "Partner", "Dependents",
        ],
        "Contract": [
            "tenure", "Contract_", "InternetService_",
        ],
        "Billing": [
            "MonthlyCharges", "TotalCharges", "PaperlessBilling",
            "PaymentMethod_",
        ],
        "Usage": [
            "PhoneService", "MultipleLines_", "OnlineSecurity_",
            "OnlineBackup_", "DeviceProtection_", "TechSupport_",
            "StreamingTV_", "StreamingMovies_",
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# build_feature_groups
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_groups(dataset_name: str, col_names: list) -> dict:
    """
    Derives group → column index mapping at runtime from actual CSV columns.

    Parameters
    ----------
    dataset_name : str
        One of "bank", "telco1", "telco2".
    col_names : list[str]
        Actual column names from the feature-engineered CSV (excluding Churn).

    Returns
    -------
    dict[str, list[int]]
        Maps each group name to a list of column indices.

    Notes
    -----
    - Rules are matched by prefix/substring — a keyword matches a column if
      the column name starts with or equals the keyword.
    - Columns are assigned to the FIRST group whose rules match them.
    - Any column not matched by any rule is assigned to Usage as a fallback,
      so no features are silently dropped.
    - Prints a warning for any unmatched columns so you can update GROUP_RULES.
    """
    if dataset_name not in GROUP_RULES:
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Must be one of: {list(GROUP_RULES.keys())}"
        )

    rules = GROUP_RULES[dataset_name]
    group_names = list(rules.keys())

    # Build result dict
    feature_groups = {g: [] for g in group_names}
    assigned = set()

    for idx, col in enumerate(col_names):
        matched = False
        for group_name in group_names:
            keywords = rules[group_name]
            if any(col.startswith(kw) or col == kw for kw in keywords):
                feature_groups[group_name].append(idx)
                assigned.add(col)
                matched = True
                break
        if not matched:
            # Fallback: assign to last group (Usage) and warn
            fallback = group_names[-1]
            feature_groups[fallback].append(idx)
            print(f"  [GROUP WARNING] '{col}' not matched by any rule → assigned to '{fallback}'")

    # Print group sizes for verification
    print(f"  [Groups] { {g: len(v) for g, v in feature_groups.items()} }")

    return feature_groups


# ─────────────────────────────────────────────
# Group Encoder
# ─────────────────────────────────────────────

class GroupEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.residual = (
            nn.Identity()
            if input_dim == hidden_dim
            else nn.Linear(input_dim, hidden_dim)
        )

    def forward(self, x):
        return F.relu(self.fc(x)) + self.residual(x)


class GroupedFeatureEncoder(nn.Module):
    def __init__(self, feature_dims: dict, hidden_dim: int = 64):
        """
        Parameters
        ----------
        feature_dims : dict[str, int]
            Maps group name → number of features in that group.
            Derived from build_feature_groups() at runtime.
        """
        super().__init__()
        self.group_names = list(feature_dims.keys())
        self.encoders = nn.ModuleDict(
            {
                group_name: GroupEncoder(
                    input_dim=feature_dims[group_name], hidden_dim=hidden_dim
                )
                for group_name in self.group_names
            }
        )

    def forward(self, grouped_inputs):
        return {
            group_name: encoder(grouped_inputs[group_name])
            for group_name, encoder in self.encoders.items()
        }


# ─────────────────────────────────────────────
# Group Attention
# ─────────────────────────────────────────────

class GroupAttention(nn.Module):
    def __init__(self, embedding_dim, group_names):
        super().__init__()
        self.group_names = group_names
        self.attention_fc = nn.ModuleDict(
            {group: nn.Linear(embedding_dim, 1) for group in group_names}
        )

    def forward(self, group_embeddings):
        scores = torch.cat(
            [self.attention_fc[g](group_embeddings[g]) for g in self.group_names], dim=1
        )
        weights = F.softmax(scores, dim=1)
        weighted_embeddings = {
            g: group_embeddings[g] * weights[:, i].unsqueeze(1)
            for i, g in enumerate(self.group_names)
        }
        return weighted_embeddings, weights


# ─────────────────────────────────────────────
# GMU Gating
# ─────────────────────────────────────────────

class GMUGating(nn.Module):
    def __init__(self, embedding_dim, group_names):
        super().__init__()
        self.group_names = group_names
        self.gate_fc = nn.ModuleDict(
            {group: nn.Linear(embedding_dim, 1) for group in group_names}
        )

    def forward(self, weighted_embeddings):
        gated_embeddings = {}
        gate_values = []
        for g in self.group_names:
            x = weighted_embeddings[g]
            gate = torch.sigmoid(self.gate_fc[g](x))
            gated_embeddings[g] = x * gate
            gate_values.append(gate)
        return gated_embeddings, torch.cat(gate_values, dim=1)


# ─────────────────────────────────────────────
# Fusion
# ─────────────────────────────────────────────

class GroupFusion(nn.Module):
    def __init__(self, embedding_dim, group_names, fused_dim=128):
        super().__init__()
        self.group_names = group_names
        num_groups = len(group_names)
        self.fusion_fc = nn.Linear(embedding_dim * num_groups, fused_dim)
        self.activation = nn.ReLU()

    def forward(self, gated_embeddings):
        concatenated = torch.cat(
            [gated_embeddings[g] for g in self.group_names], dim=1
        )
        return self.activation(self.fusion_fc(concatenated))


# ─────────────────────────────────────────────
# Interaction Modeling
# ─────────────────────────────────────────────

class InteractionModeling(nn.Module):
    def __init__(self, fused_dim=128, interaction_dim=64):
        super().__init__()
        self.proj1 = nn.Linear(fused_dim, interaction_dim)
        self.proj2 = nn.Linear(fused_dim, interaction_dim)
        self.output_fc = nn.Linear(interaction_dim, interaction_dim)
        self.activation = nn.ReLU()

    def forward(self, fused_vector):
        interaction = self.proj1(fused_vector) * self.proj2(fused_vector)
        return self.activation(self.output_fc(interaction))


# ─────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────

class ChurnClassifier(nn.Module):
    def __init__(self, input_dim=64):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(x))


# ─────────────────────────────────────────────
# Main Model
# ─────────────────────────────────────────────

class NonColdStartModelTargCorrThreshold(nn.Module):
    def __init__(
        self,
        feature_dims: dict,
        hidden_dim: int = 64,
        fused_dim: int = 128,
        interaction_dim: int = 64,
    ):
        """
        Parameters
        ----------
        feature_dims : dict[str, int]
            Maps group name → number of features in that group.
            Derived from build_feature_groups() at runtime in the training script.
            e.g. {"Profile": 8, "Contract": 2, "Billing": 2, "Usage": 3}
        """
        super().__init__()
        self.group_names  = list(feature_dims.keys())
        self.feature_dims = feature_dims

        self.encoder     = GroupedFeatureEncoder(feature_dims, hidden_dim)
        self.attention   = GroupAttention(hidden_dim, self.group_names)
        self.gmu         = GMUGating(hidden_dim, self.group_names)
        self.fusion      = GroupFusion(hidden_dim, self.group_names, fused_dim)
        self.interaction = InteractionModeling(fused_dim, interaction_dim)
        self.classifier  = ChurnClassifier(input_dim=interaction_dim)

    def split_into_groups(self, X: torch.Tensor) -> dict:
        grouped_inputs = {}
        start_idx = 0
        for group_name in self.group_names:
            dim = self.feature_dims[group_name]
            grouped_inputs[group_name] = X[:, start_idx : start_idx + dim]
            start_idx += dim
        return grouped_inputs

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        grouped_inputs = self.split_into_groups(X)
        encoded        = self.encoder(grouped_inputs)
        weighted, _    = self.attention(encoded)
        gated, _       = self.gmu(weighted)
        fused          = self.fusion(gated)
        interacted     = self.interaction(fused)
        prob           = self.classifier(interacted)
        return prob