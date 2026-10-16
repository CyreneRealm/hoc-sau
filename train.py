"""train.py — Huấn luyện một epoch: ERM hoặc Group DRO."""

import torch
import torch.nn.functional as F
from tqdm import tqdm


def train_one_epoch_erm(model, dataloader, optimizer, device):
    """ERM: tối thiểu hóa cross-entropy loss trung bình."""
    model.train()
    total_loss = 0.0
    n_samples = 0

    for batch in tqdm(dataloader, desc="Training (ERM)", leave=False):
        x, y, metadata = batch
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        n_samples += x.size(0)

    return total_loss / n_samples


def train_one_epoch_group_dro(model, dataloader, optimizer, device,
                              n_groups, hospital_idx, group_weights,
                              dro_step_size=0.01):
    """Group DRO: tối ưu worst-domain risk qua cập nhật trọng số nhóm.

    group_weights: tensor shape (n_groups,) — trọng số hiện tại, được cập nhật in-place.
    dro_step_size: tốc độ cập nhật trọng số nhóm.
    """
    model.train()
    total_loss = 0.0
    n_samples = 0

    for batch in tqdm(dataloader, desc="Training (DRO)", leave=False):
        x, y, metadata = batch
        x, y = x.to(device), y.to(device)
        groups = metadata[:, hospital_idx].long().to(device)

        optimizer.zero_grad()
        logits = model(x)
        losses = F.cross_entropy(logits, y, reduction="none")

        # Tính loss trung bình mỗi nhóm
        group_losses = torch.zeros(n_groups, device=device)
        group_counts = torch.zeros(n_groups, device=device)
        for g in range(n_groups):
            mask = (groups == g)
            if mask.any():
                group_losses[g] = losses[mask].mean()
                group_counts[g] = mask.sum()

        # Cập nhật trọng số nhóm: tăng trọng số domain có loss cao
        # ponytail: multiplicative update đơn giản, thay bằng Sagawa et al. nếu cần chuẩn hơn
        present = group_counts > 0
        if present.any():
            group_weights[present] *= torch.exp(dro_step_size * group_losses[present].detach())
            group_weights /= group_weights.sum()  # normalize

        # Weighted loss
        loss = (group_weights * group_losses).sum()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        n_samples += x.size(0)

    return total_loss / max(n_samples, 1)
