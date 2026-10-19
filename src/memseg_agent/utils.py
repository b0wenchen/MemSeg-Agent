"""Small, dependency-light utilities shared by command-line entry points."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(value: str | None = None) -> torch.device:
    if value is None or value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def dice_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Dice per batch element for binary logits and binary targets."""
    prediction = (torch.sigmoid(logits) >= 0.5).float()
    target = target.float()
    dims = tuple(range(1, prediction.ndim))
    intersection = (prediction * target).sum(dim=dims)
    denominator = prediction.sum(dim=dims) + target.sum(dim=dims)
    return (2 * intersection + eps) / (denominator + eps)


def binary_dice(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    prediction = prediction.bool()
    target = target.bool()
    inter = (prediction & target).sum().item()
    total = prediction.sum().item() + target.sum().item()
    return float((2 * inter + eps) / (total + eps))


def dice_from_counts(twice_intersection: int, denominator: int) -> float:
    """Volume Dice: both empty is a correct prediction (1.0)."""
    return twice_intersection / denominator if denominator else 1.0
