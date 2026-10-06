from pathlib import Path

src = Path("/mnt/data/Pasted code(2).py")
text = src.read_text(encoding="utf-8")

# Build a standalone evaluation-only version from the uploaded experiment.
# We retain the original model/config imports and evaluation logic, but replace
# the training/checkpoint branch with direct loading of existing checkpoints.

start = text.index("def run_dataset_ablation(")
end = text.index("\ndef _npz(", start)

new_func = r'''def run_dataset_ablation(dataset_name, train_path, val_path, test_path):
    """
    Evaluation-only rerun.

    Loads already-trained .pth checkpoints from models/ and evaluates them.
    NO MODEL IS TRAINED OR OVERWRITTEN.
    """
    print(f"\n{'=' * 70}")
    print(f" EVALUATION-ONLY RERUN — {dataset_name.upper()}")
    print(f"{'=' * 70}")

    cfg = DATASET_CONFIGS[dataset_name]

    train_data = np.load(train_path)
    test_data = np.load(test_path)

    X_train = train_data["X"].astype(np.float32)
    X_test = test_data["X"].astype(np.float32)
    y_test = test_data["y"].astype(int)

    rows = []

    for variant_name, flags in ABLATION_VARIANTS.items():

        # Match the filenames produced by the original ablation script.
        if variant_name == "Full (MPMN+VML)":
            ckpt_name = f"mpmn_{dataset_name}.pth"
        elif variant_name == "Vanilla ProtoNet (floor)":
            ckpt_name = f"protonet_{dataset_name}.pth"
        else:
            ckpt_name = f"ablation_{_slugify(variant_name)}_{dataset_name}.pth"

        ckpt_path = os.path.join(MODELS_DIR, ckpt_name)

        print(f"\n[{dataset_name}] {variant_name}")
        print(f"  Checkpoint: {ckpt_path}")

        if not os.path.exists(ckpt_path):
            print("  WARNING: checkpoint not found — skipping.")
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu")

        # Full MPMN checkpoints were saved using MPMN rather than
        # AblatableMPMN, so load them with the original architecture.
        if variant_name == "Full (MPMN+VML)":
            model = MPMN(
                X_train.shape[1],
                cfg["hidden_dim"],
                cfg["latent_dim"],
                cfg["dropout"],
            )
        else:
            model = AblatableMPMN(
                X_train.shape[1],
                cfg["hidden_dim"],
                cfg["latent_dim"],
                cfg["dropout"],
                use_variational=flags["use_variational"],
                use_temperature=flags["use_temperature"],
                use_uncertainty_weighting=flags["use_uncertainty_weighting"],
            )

        model.load_state_dict(ckpt["model_state_dict"])
        n_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        # Use a fresh, deterministic seed for each model's evaluation.
        # The evaluator itself uses random episodic test splits.
        np.random.seed(42)
        torch.manual_seed(42)

        m = evaluate_variant(
            model,
            X_test,
            y_test,
            cfg,
            n_episodes=N_TEST_EPISODES,
        )

        print(
            f"  Acc {m['accuracy'] * 100:.2f}% | "
            f"Prec {m['precision'] * 100:.2f}% | "
            f"Rec {m['recall'] * 100:.2f}% | "
            f"F1 {m['f1'] * 100:.2f}% | "
            f"AUC {m['auc'] * 100:.2f}% | "
            f"ECE {m['ece']:.4f} | "
            f"Inference {m['infer_ms']:.2f} ms/episode"
        )

        # Generate the same diagnostic plots, but never save/modify models.
        generate_diagnostic_plots(
            model,
            X_test,
            y_test,
            cfg,
            dataset_name=dataset_name,
            variant_name=variant_name,
            save_dir=FIGURES_DIR,
        )

        rows.append({
            "Dataset": dataset_name,
            "Variant": variant_name,
            "Accuracy": f"{m['accuracy'] * 100:.2f}% (±{m['accuracy_ci'] * 100:.2f})",
            "Precision": f"{m['precision'] * 100:.2f}% (±{m['precision_ci'] * 100:.2f})",
            "Recall": f"{m['recall'] * 100:.2f}% (±{m['recall_ci'] * 100:.2f})",
            "Macro F1": f"{m['f1'] * 100:.2f}% (±{m['f1_ci'] * 100:.2f})",
            "ROC-AUC": f"{m['auc'] * 100:.2f}% (±{m['auc_ci'] * 100:.2f})",
            "ECE": f"{m['ece']:.4f}",
            "Train Time (s, avg)": "n/a (existing checkpoint)",
            "Inference (ms/episode)": f"{m['infer_ms']:.2f}",
            "Params": f"{n_params:,}",
            "N Seeds": 0,
            "Checkpoint": ckpt_name,
        })

    return pd.DataFrame(rows)
'''

new_text = text[:start] + new_func + text[end:]

# Remove the old training-oriented setting/comment and make the script explicit.
new_text = new_text.replace(
    "LOAD_EXISTING_FULL_CHECKPOINT = False",
    "LOAD_EXISTING_FULL_CHECKPOINT = True  # retained for compatibility; this script loads ALL existing checkpoints",
)

# Make the purpose obvious at the top.
marker = "import os\n"
insert = """# ================================================================
# EVALUATION-ONLY VERSION
# Loads existing .pth checkpoints. It does NOT train or overwrite models.
# ================================================================
"""
new_text = new_text.replace(marker, marker + insert, 1)

out = Path("/mnt/data/rerun_existing_models.py")
out.write_text(new_text, encoding="utf-8")

print(f"Created: {out}")
print(
    "This version evaluates existing checkpoints and does not train or save model checkpoints."
)
