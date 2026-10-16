"""evaluate.py — Đánh giá mô hình trên validation/test."""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm


@torch.no_grad()
def evaluate(model, dataloader, device, n_groups, hospital_idx):
    """Tính các chỉ số đánh giá.

    Returns:
        dict với keys: accuracy, auroc, loss, avg_group_risk, worst_group_risk,
                       per_group_risk, per_group_acc
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    all_groups = []
    all_losses = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        x, y, metadata = batch
        x, y = x.to(device), y.to(device)
        groups = metadata[:, hospital_idx].long()

        logits = model(x)
        probs = F.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)
        losses = F.cross_entropy(logits, y, reduction="none")

        all_preds.append(preds.cpu())
        all_probs.append(probs.cpu())
        all_labels.append(y.cpu())
        all_groups.append(groups)
        all_losses.append(losses.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_groups = torch.cat(all_groups).numpy()
    all_losses = torch.cat(all_losses).numpy()

    # Overall metrics
    accuracy = accuracy_score(all_labels, all_preds)
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = float("nan")  # ponytail: single-class split edge case
    avg_loss = float(all_losses.mean())

    # Per-group metrics
    per_group_risk = {}
    per_group_acc = {}
    group_risks = []
    for g in range(n_groups):
        mask = all_groups == g
        if mask.any():
            per_group_risk[g] = float(all_losses[mask].mean())
            per_group_acc[g] = float(accuracy_score(all_labels[mask], all_preds[mask]))
            group_risks.append(per_group_risk[g])

    avg_group_risk = float(np.mean(group_risks)) if group_risks else float("nan")
    worst_group_risk = float(np.max(group_risks)) if group_risks else float("nan")

    return {
        "accuracy": accuracy,
        "auroc": auroc,
        "loss": avg_loss,
        "avg_group_risk": avg_group_risk,
        "worst_group_risk": worst_group_risk,
        "per_group_risk": per_group_risk,
        "per_group_acc": per_group_acc,
    }
