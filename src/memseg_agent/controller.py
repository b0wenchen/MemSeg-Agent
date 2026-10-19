"""Minimal explicit policy for composing static, few-shot and working memories."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MemoryPlan:
    use_static: bool
    use_few_shot: bool
    use_working: bool


class MemoryController:
    """A transparent controller policy; callers can replace it with a learned policy."""

    def __init__(self, static_similarity_threshold: float = 0.85) -> None:
        self.static_similarity_threshold = static_similarity_threshold

    def select(self, static_similarity: float | None, has_few_shot: bool, has_working: bool) -> MemoryPlan:
        aligned = static_similarity is not None and static_similarity >= self.static_similarity_threshold
        return MemoryPlan(use_static=True, use_few_shot=has_few_shot and not aligned, use_working=has_working)

    def compose(self, plan: MemoryPlan, few_shot: list[tuple[torch.Tensor, torch.Tensor]], working: list[tuple[torch.Tensor, torch.Tensor]]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return ([*few_shot] if plan.use_few_shot else []) + ([*working] if plan.use_working else [])
