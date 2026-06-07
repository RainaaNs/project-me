import torch
import torch.nn as nn
import torch.nn.functional as F


<<<<<<< HEAD
# ─────────────────────────────────────────────
# Feature Group Definitions
# ─────────────────────────────────────────────
=======
# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUPS
# ─────────────────────────────────────────────────────────────────────────────
# Column names match the gatefuse_ready CSV output produced by
# NonColdStartFeatureEngineer.group_layout exactly.
#

# Group sizes (excluding Churn):
#   bank   : Profile=6  Contract=2  Billing=3   Usage=4    →  15 features
#   telco1 : Profile=5  Contract=8  Billing=9   Usage=16   →  38 features
#   telco2 : Profile=4  Contract=3  Billing=6   Usage=10   →  23 features
#
# If a name here doesn't match an output column from the engineer, training
# will fail with a "Feature mismatch" error. Keep these two files in lockstep.
>>>>>>> d66af3786a85f4f752e0806f0347a5c4e2599045

FEATURE_GROUPS = {
    "bank": {
        "Profile": [
            "Geography_Spain",
            "Geography_France",
            "Geography_Germany",
            "Gender",
            "Age",
            "Satisfaction Score",
        ],
        "Contract": ["Tenure", "Card Type"],
        "Billing": ["Balance", "EstimatedSalary", "CreditScore"],
        "Usage": [
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            # "Complain",
            "Point Earned",
        ],
    },
    "telco1": {
        "Profile": [
            "Gender",
            "Age",
            "Married",
            "Dependents",
            "Number of Dependents",
            "Satisfaction Score",
        ],
        "Contract": [
            "Tenure in Months",
            "Offer_No Offer",
            "Offer_Offer A",
            "Offer_Offer B",
            "Offer_Offer C",
            "Offer_Offer D",
            "Offer_Offer E",
            "Unlimited Data",
            "Contract_Month-to-Month",
            "Contract_One Year",
            "Contract_Two Year",
        ],
        "Billing": [
            "Avg Monthly Long Distance Charges",
            "Paperless Billing",
            "Payment Method_Bank Withdrawal",
            "Payment Method_Credit Card",
            "Payment Method_Mailed Check",
            "Monthly Charge",
            "Total Charges",
            "Total Refunds",
            "Total Extra Data Charges",
            "Total Long Distance Charges",
        ],
        "Usage": [
            "Referred a Friend",
            "Number of Referrals",
            "Phone Service",
            "Multiple Lines",
            "Internet Service",
            "Internet Type_DSL",
            "Internet Type_Cable",
            "Internet Type_Fiber Optic",
            "Internet Type_No Internet",
            "Avg Monthly GB Download",
            "Online Security",
            "Online Backup",
            "Device Protection Plan",
            "Premium Tech Support",
            "Streaming TV",
            "Streaming Movies",
            "Streaming Music",
        ],
    },
    # Updated to match NCS-FE output (41 features, is_cold_start excluded)
    "telco2": {
        "Profile": [
            "gender",  # 1
            "Partner",  # 2
            "SeniorCitizen_0",  # 9
            "SeniorCitizen_1",  # 10
        ],
        "Contract": [
            "tenure",  # 3
            "Contract_Month-to-month",  # 14
            "Contract_One year",  # 15
            "Contract_Two year",  # 16
            "InternetService_DSL",  # 11
            "InternetService_Fiber optic",  # 12
            "InternetService_No",  # 13
        ],
        "Billing": [
            "PaperlessBilling",  # 5
            "MonthlyCharges",  # 6
            "TotalCharges",  # 7
            "PaymentMethod_Bank transfer (automatic)",  # 17
            "PaymentMethod_Credit card (automatic)",  # 18
            "PaymentMethod_Electronic check",  # 19
            "PaymentMethod_Mailed check",  # 20
        ],
        "Usage": [
            "PhoneService",  # 4
            "MultipleLines_No",  # 27
            "MultipleLines_No phone service",  # 28
            "MultipleLines_Yes",  # 29
            "OnlineSecurity_No",  # 21
            "OnlineSecurity_No internet service",  # 22
            "OnlineSecurity_Yes",  # 23
            "OnlineBackup_No",  # 30
            "OnlineBackup_No internet service",  # 31
            "OnlineBackup_Yes",  # 32
            "DeviceProtection_No",  # 33
            "DeviceProtection_No internet service",  # 34
            "DeviceProtection_Yes",  # 35
            "TechSupport_No",  # 24
            "TechSupport_No internet service",  # 25
            "TechSupport_Yes",  # 26
            "StreamingTV_No",  # 36
            "StreamingTV_No internet service",  # 37
            "StreamingTV_Yes",  # 38
            "StreamingMovies_No",  # 39
            "StreamingMovies_No internet service",  # 40
            "StreamingMovies_Yes",  # 41
        ],
    },
}


def route_feature_groups(dataset_name: str):
    if dataset_name not in FEATURE_GROUPS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return FEATURE_GROUPS[dataset_name]


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
    def __init__(self, dataset_name, feature_dims, hidden_dim=64):
        super().__init__()
        self.groups = route_feature_groups(dataset_name)
        self.encoders = nn.ModuleDict(
            {
                group_name: GroupEncoder(
                    input_dim=feature_dims[group_name], hidden_dim=hidden_dim
                )
                for group_name in self.groups
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
        )  # (batch_size, num_groups)

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
        concatenated = torch.cat([gated_embeddings[g] for g in self.group_names], dim=1)
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


class NonColdStartModel(nn.Module):
    def __init__(
        self,
        dataset_name,
        feature_dims,
        hidden_dim=64,
        fused_dim=128,
        interaction_dim=64,
    ):
        super().__init__()
        self.groups = route_feature_groups(dataset_name)
        self.group_names = list(self.groups.keys())  # dynamic, not hardcoded
        self.feature_dims = feature_dims

        self.encoder = GroupedFeatureEncoder(dataset_name, feature_dims, hidden_dim)
        self.attention = GroupAttention(hidden_dim, self.group_names)
        self.gmu = GMUGating(hidden_dim, self.group_names)
        self.fusion = GroupFusion(hidden_dim, self.group_names, fused_dim)
        self.interaction = InteractionModeling(fused_dim, interaction_dim)
        self.classifier = ChurnClassifier(input_dim=interaction_dim)

    def split_into_groups(self, X):
        grouped_inputs = {}
        start_idx = 0
        for group_name in self.group_names:
            dim = self.feature_dims[group_name]
            grouped_inputs[group_name] = X[:, start_idx : start_idx + dim]
            start_idx += dim
        return grouped_inputs

    def forward(self, X):
        grouped_inputs = self.split_into_groups(X)
        encoded = self.encoder(grouped_inputs)
        weighted, attn_weights = self.attention(encoded)
        gated, gate_values = self.gmu(weighted)
        fused = self.fusion(gated)
        interacted = self.interaction(fused)
        prob = self.classifier(interacted)
        return prob
