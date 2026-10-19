"""Communication-safe aggregation for static-memory checkpoints only."""

from __future__ import annotations

from collections.abc import Sequence

import torch


MEMORY_KEYS = ("mem_img_logits", "mem_mask_logits")


def fedavg_static_memories(states: Sequence[dict[str, torch.Tensor]], weights: Sequence[float] | None = None) -> dict[str, torch.Tensor]:
    """FedAvg for the two pseudo-observation tensors; never aggregates SAM2 weights."""
    if not states:
        raise ValueError("At least one client state is required")
    if weights is None:
        weights = [1.0] * len(states)
    if len(states) != len(weights) or any(weight < 0 for weight in weights):
        raise ValueError("states and non-negative weights must have matching lengths")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("At least one aggregation weight must be positive")
    result: dict[str, torch.Tensor] = {}
    for key in MEMORY_KEYS:
        if not all(key in state for state in states):
            raise KeyError(f"Every client state must contain {key}")
        result[key] = sum(state[key].detach().float().cpu() * (weight / total) for state, weight in zip(states, weights))
    return result
