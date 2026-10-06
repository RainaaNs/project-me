import shap
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.non_cold_start.non_cold_start_model_full_feature_set import (
    NonColdStartModelFullFeatureSet,
    build_feature_groups
)

DROP_COLS = ["Churn", "is_cold_start"]


def load_data(csv_path, dataset):
    df = pd.read_csv(csv_path)

    drop = [c for c in DROP_COLS if c in df.columns]
    col_names = df.drop(columns=drop).columns.tolist()

    feature_groups = build_feature_groups(dataset, col_names)

    ordered_cols = []
    for group, indices in feature_groups.items():
        ordered_cols.extend([col_names[i] for i in indices])

    X = torch.tensor(df[ordered_cols].values, dtype=torch.float32)

    feature_dims = {
        group: len(indices)
        for group, indices in feature_groups.items()
    }

    return X, ordered_cols, feature_dims


# def analyse_model(
#     dataset,
#     model_path,
#     val_csv,
#     output_folder,
# ):

#     X, feature_names, feature_dims = load_data(val_csv, dataset)

#     model = NonColdStartModelFullFeatureSet(
#         feature_dims=feature_dims
#     )

#     model.load_state_dict(torch.load(model_path))

#     model.eval()

#     background = X[:100]
#     samples = X[:200]

#     explainer = shap.GradientExplainer(
#         model,
#         background
#     )

#     shap_values = explainer.shap_values(
#         samples
#     )

#     # GradientExplainer can return a list (one array per output) even for
#     # single-output models -- normalize to a single 2D array if so.
#     if isinstance(shap_values, list):
#         shap_values = shap_values[0]

#     plt.figure()

#     shap.summary_plot(
#         shap_values,
#         samples.numpy(),
#         feature_names=feature_names,
#         plot_type="bar",
#         show=False,
#     )

#     plt.tight_layout()

#     plt.savefig(
#         f"{output_folder}/{dataset}_feature_importance.png",
#         dpi=300
#     )

#     plt.close()

#     print("Done.")

def analyse_model(
    dataset,
    model_path,
    val_csv,
    output_folder,
):

    X, feature_names, feature_dims = load_data(val_csv, dataset)

    model = NonColdStartModelFullFeatureSet(
        feature_dims=feature_dims
    )

    model.load_state_dict(torch.load(model_path))

    model.eval()

    background = X[:100]
    samples = X[:200]

    explainer = shap.GradientExplainer(
        model,
        background
    )

    shap_values = explainer.shap_values(
        samples
    )

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = shap_values.squeeze()

    # Sanity check: this MUST match (num_samples, num_features)
    print(f"[Debug] shap_values shape: {shap_values.shape}")
    print(f"[Debug] samples shape:     {samples.shape}")
    print(f"[Debug] num feature_names: {len(feature_names)}")

    assert shap_values.ndim == 2, f"Expected 2D shap_values, got {shap_values.shape}"
    assert shap_values.shape[1] == len(feature_names), (
        f"Feature count mismatch: shap_values has {shap_values.shape[1]} "
        f"features but feature_names has {len(feature_names)}"
    )

    plt.figure()

    shap.summary_plot(
        shap_values,
        samples.numpy(),
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        f"{output_folder}/{dataset}_feature_importance.png",
        dpi=300
    )

    plt.close()

    print("Done.")

if __name__ == "__main__":

    BASE = PROJECT_ROOT

    analyse_model(

        dataset="bank",

        model_path=f"{BASE}/checkpoints/baseline/bank_non_cold_start.pt",

        val_csv=f"{BASE}/datasets/processed/baseline/bank/gatefuse_ready/val.csv",

        output_folder=f"{BASE}/analysis",

    )
