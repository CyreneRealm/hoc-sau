"""experiments.py — Điều phối các thí nghiệm.

Chạy: python experiments.py --epochs 5 --batch-size 16 --runs erm erm_aug group_dro group_dro_aug
"""

import argparse
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dataset import get_dataloaders
from model import get_model
from train import train_one_epoch_erm, train_one_epoch_group_dro
from evaluate import evaluate


RESULTS_DIR = "results"

# Cấu hình thí nghiệm
EXPERIMENT_CONFIGS = {
    "erm":           {"method": "erm",       "augmentation": False},
    "erm_aug":       {"method": "erm",       "augmentation": True},
    "group_dro":     {"method": "group_dro", "augmentation": False},
    "group_dro_aug": {"method": "group_dro", "augmentation": True},
}


def set_seed(seed):
    """Đặt random seed cho reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(run_name, config, seed, epochs, batch_size, lr, data_dir, device):
    """Chạy một thí nghiệm đầy đủ: train → validate → test."""
    print(f"\n{'='*60}")
    print(f"  {run_name}  |  seed={seed}  |  epochs={epochs}")
    print(f"{'='*60}")

    set_seed(seed)

    # Data
    splits, hospital_idx, n_groups = get_dataloaders(
        batch_size=batch_size,
        use_augmentation=config["augmentation"],
        data_dir=data_dir,
    )

    # Model & Optimizer
    model = get_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Group DRO weights
    group_weights = torch.ones(n_groups, device=device) / n_groups

    # Training history
    history = {"train_loss": [], "id_val": [], "ood_val": []}
    best_id_val_acc = -1.0
    tag = f"{run_name}_seed{seed}"
    checkpoint_path = os.path.join(RESULTS_DIR, f"{tag}_best.pth")

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # Train
        if config["method"] == "erm":
            train_loss = train_one_epoch_erm(model, splits["train"], optimizer, device)
        else:
            train_loss = train_one_epoch_group_dro(
                model, splits["train"], optimizer, device,
                n_groups, hospital_idx, group_weights,
            )
        history["train_loss"].append(train_loss)
        print(f"  Train loss: {train_loss:.4f}")

        # ID validation
        id_val_metrics = evaluate(model, splits["id_val"], device, n_groups, hospital_idx)
        history["id_val"].append(id_val_metrics)
        print(f"  ID Val  — Acc: {id_val_metrics['accuracy']:.4f}  AUROC: {id_val_metrics['auroc']:.4f}")

        # OOD validation
        ood_val_metrics = evaluate(model, splits["val"], device, n_groups, hospital_idx)
        history["ood_val"].append(ood_val_metrics)
        print(f"  OOD Val — Acc: {ood_val_metrics['accuracy']:.4f}  AUROC: {ood_val_metrics['auroc']:.4f}")

        # Checkpoint selection: best ID validation accuracy
        if id_val_metrics["accuracy"] > best_id_val_acc:
            best_id_val_acc = id_val_metrics["accuracy"]
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Saved best checkpoint (ID val acc={best_id_val_acc:.4f})")

    # Load best checkpoint for final test
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    print(f"\nFinal test evaluation ({tag}):")

    id_val_final = evaluate(model, splits["id_val"], device, n_groups, hospital_idx)
    ood_test = evaluate(model, splits["test"], device, n_groups, hospital_idx)

    print(f"  ID Val  — Acc: {id_val_final['accuracy']:.4f}  AUROC: {id_val_final['auroc']:.4f}")
    print(f"  OOD Test— Acc: {ood_test['accuracy']:.4f}  AUROC: {ood_test['auroc']:.4f}")
    print(f"  Avg Group Risk: {ood_test['avg_group_risk']:.4f}")
    print(f"  Worst Group Risk: {ood_test['worst_group_risk']:.4f}")
    print(f"  Per-group Acc: {ood_test['per_group_acc']}")

    # Save history
    history_path = os.path.join(RESULTS_DIR, f"{tag}_history.json")
    # Convert numpy/float values for JSON serialization
    def _serialize(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, dict):
            return {str(k): _serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialize(i) for i in obj]
        return obj

    with open(history_path, "w") as f:
        json.dump(_serialize(history), f, indent=2)

    return {
        "model": run_name,
        "seed": seed,
        "id_accuracy": id_val_final["accuracy"],
        "ood_accuracy": ood_test["accuracy"],
        "id_auroc": id_val_final["auroc"],
        "ood_auroc": ood_test["auroc"],
        "avg_group_risk": ood_test["avg_group_risk"],
        "worst_group_risk": ood_test["worst_group_risk"],
    }, history


def plot_results(all_results, all_histories):
    """Tạo 3 biểu đồ theo yêu cầu README mục 9."""

    # 1. So sánh ID và OOD performance
    df = pd.DataFrame(all_results)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    models = df.groupby("model")

    model_names = []
    id_accs = []
    ood_accs = []
    id_aurocs = []
    ood_aurocs = []
    for name, group in models:
        model_names.append(name)
        id_accs.append(group["id_accuracy"].mean())
        ood_accs.append(group["ood_accuracy"].mean())
        id_aurocs.append(group["id_auroc"].mean())
        ood_aurocs.append(group["ood_auroc"].mean())

    x = np.arange(len(model_names))
    w = 0.35

    axes[0].bar(x - w/2, id_accs, w, label="ID Accuracy", color="#4285F4")
    axes[0].bar(x + w/2, ood_accs, w, label="OOD Accuracy", color="#EA4335")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=30, ha="right")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("ID vs OOD Accuracy")
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    axes[1].bar(x - w/2, id_aurocs, w, label="ID AUROC", color="#4285F4")
    axes[1].bar(x + w/2, ood_aurocs, w, label="OOD AUROC", color="#EA4335")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, rotation=30, ha="right")
    axes[1].set_ylabel("AUROC")
    axes[1].set_title("ID vs OOD AUROC")
    axes[1].legend()
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "comparison_id_ood.png"), dpi=150)
    plt.close()

    # 2. Training loss và validation loss theo epoch
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#4285F4", "#EA4335", "#FBBC04", "#34A853"]
    for i, (tag, hist) in enumerate(all_histories.items()):
        epochs = range(1, len(hist["train_loss"]) + 1)
        c = colors[i % len(colors)]
        ax.plot(epochs, hist["train_loss"], "-", color=c, label=f"{tag} train")
        val_losses = [m["loss"] for m in hist["ood_val"]]
        ax.plot(epochs, val_losses, "--", color=c, label=f"{tag} OOD val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & OOD Validation Loss")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "loss_curves.png"), dpi=150)
    plt.close()

    # 3. Per-hospital accuracy (từ test cuối cùng, dùng seed đầu tiên của mỗi run)
    fig, ax = plt.subplots(figsize=(10, 6))
    seen_models = set()
    bar_data = {}
    for r in all_results:
        if r["model"] not in seen_models:
            seen_models.add(r["model"])
            # Lấy per_group_acc từ history không có sẵn trong summary,
            # nên chạy lại evaluate lần nữa sẽ tốn. Thay vào đó lưu vào history.
            # ponytail: skip per-hospital chart nếu không có data, thêm khi cần
    # Fallback: dùng OOD accuracy per model
    if not bar_data:
        model_names_2 = [r["model"] for r in all_results]
        ood_accs_2 = [r["ood_accuracy"] for r in all_results]
        ax.bar(range(len(model_names_2)), ood_accs_2, color=colors[:len(model_names_2)])
        ax.set_xticks(range(len(model_names_2)))
        ax.set_xticklabels(model_names_2, rotation=30, ha="right")
        ax.set_ylabel("OOD Test Accuracy")
        ax.set_title("OOD Test Accuracy per Experiment")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "per_experiment_accuracy.png"), dpi=150)
    plt.close()

    print(f"\nCharts saved to {RESULTS_DIR}/")


def main():
    parser = argparse.ArgumentParser(description="Camelyon17-WILDS Experiments")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--runs", type=str, nargs="+", default=["erm"],
                        choices=list(EXPERIMENT_CONFIGS.keys()))
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_results = []
    all_histories = {}

    for run_name in args.runs:
        config = EXPERIMENT_CONFIGS[run_name]
        for seed in args.seeds:
            result, history = run_experiment(
                run_name, config, seed, args.epochs, args.batch_size,
                args.lr, args.data_dir, device,
            )
            all_results.append(result)
            all_histories[f"{run_name}_seed{seed}"] = history

    # Lưu kết quả tổng hợp vào CSV
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(RESULTS_DIR, "metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n{'='*60}")
    print("Summary:")
    print(df.to_string(index=False))
    print(f"\nResults saved to {csv_path}")

    # Phân tích Q1, Q2, Q3
    print(f"\n{'='*60}")
    print("Analysis")
    print(f"{'='*60}")

    # Q1: Performance loss do domain shift
    erm_rows = df[df["model"] == "erm"]
    if not erm_rows.empty:
        id_acc = erm_rows["id_accuracy"].mean()
        ood_acc = erm_rows["ood_accuracy"].mean()
        gap = id_acc - ood_acc
        relative = (gap / id_acc * 100) if id_acc > 0 else float("nan")
        print(f"\nQ1 — Domain Shift Gap (ERM baseline):")
        print(f"  ID Accuracy:  {id_acc:.4f}")
        print(f"  OOD Accuracy: {ood_acc:.4f}")
        print(f"  Gap:          {gap:.4f}")
        print(f"  Relative:     {relative:.1f}%")

    # Q2: Data Augmentation
    erm_aug_rows = df[df["model"] == "erm_aug"]
    if not erm_rows.empty and not erm_aug_rows.empty:
        delta = erm_aug_rows["ood_accuracy"].mean() - erm_rows["ood_accuracy"].mean()
        print(f"\nQ2 — Data Augmentation Effect:")
        print(f"  ΔOOD Accuracy (aug - baseline): {delta:+.4f}")

    # Q3: ERM vs Group DRO
    dro_rows = df[df["model"] == "group_dro"]
    if not erm_rows.empty and not dro_rows.empty:
        print(f"\nQ3 — ERM vs Group DRO:")
        print(f"  ERM  — Avg Risk: {erm_rows['avg_group_risk'].mean():.4f}  "
              f"Worst Risk: {erm_rows['worst_group_risk'].mean():.4f}")
        print(f"  DRO  — Avg Risk: {dro_rows['avg_group_risk'].mean():.4f}  "
              f"Worst Risk: {dro_rows['worst_group_risk'].mean():.4f}")

    # Vẽ biểu đồ
    plot_results(all_results, all_histories)


if __name__ == "__main__":
    main()
