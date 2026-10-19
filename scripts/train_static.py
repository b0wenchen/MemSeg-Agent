#!/usr/bin/env python
"""Train only MemSeg-Agent static pseudo-observations for one patient split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from memseg_agent.data import NiftiSliceDataset, patient_five_fold_split
from memseg_agent.model import MemSegAgent
from memseg_agent.utils import dice_from_logits, load_yaml, resolve_device, set_seed


def binary_target(labels: torch.Tensor, target_class: int) -> torch.Tensor:
    return (labels == target_class).float().unsqueeze(1)


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Differentiable Dice loss used to optimize pseudo-observations."""
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1 - ((2 * intersection + eps) / (denominator + eps)).mean()


@torch.no_grad()
def validate(model: MemSegAgent, loader: DataLoader, target_class: int, device: torch.device) -> float:
    model.eval()
    values: list[float] = []
    for image, label in loader:
        logits, _ = model(image.to(device))
        target = F.interpolate(binary_target(label.to(device), target_class), size=logits.shape[-2:], mode="nearest")
        values.extend(dice_from_logits(logits, target).cpu().tolist())
    return sum(values) / max(1, len(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--fold", type=int, default=0, help="0-indexed outer test fold")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg.get("device"))
    dataset = NiftiSliceDataset(cfg["data_root"], num_classes=cfg["num_classes"])
    split = patient_five_fold_split(dataset, args.fold, seed=cfg["seed"])
    print(f"patients total/train/val/test = {dataset.patient_count}/{len(split.train_patients)}/{len(split.val_patients)}/{len(split.test_patients)}")
    train_loader = DataLoader(split.train, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val_loader = DataLoader(split.val, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    model = MemSegAgent(cfg["sam2_config"], cfg["sam2_checkpoint"], cfg["target_class"]).to(device)
    optimizer = AdamW([model.mem_img_logits, model.mem_mask_logits], lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    output = Path(cfg["output_dir"]) / f"fold_{args.fold + 1}"
    output.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        for image, label in tqdm(train_loader, desc=f"epoch {epoch}/{cfg['epochs']}"):
            logits, objectness = model(image.to(device))
            target = F.interpolate(binary_target(label.to(device), cfg["target_class"]), size=logits.shape[-2:], mode="nearest")
            has_object = target.flatten(1).any(dim=1).float()
            dice_loss = soft_dice_loss(logits, target)
            object_loss = F.binary_cross_entropy_with_logits(objectness.flatten(), has_object)
            loss = dice_loss + cfg["object_loss_weight"] * object_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        score = validate(model, val_loader, cfg["target_class"], device)
        print(f"epoch={epoch} validation_slice_dice={score:.4f}")
        if score > best:
            best = score
            torch.save({"epoch": epoch, "val_slice_dice": score, "static_memory": model.static_state_dict(), "config": cfg}, output / "best_static_memory.pt")
    print(f"best validation slice Dice: {best:.4f}; test patients were never used for training or selection")


if __name__ == "__main__":
    main()
