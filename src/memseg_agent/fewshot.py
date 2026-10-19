"""Few-shot support-set encoding in the shared SAM2 memory space."""

from __future__ import annotations

import torch


@torch.no_grad()
def encode_support_set(model, supports: list[tuple[torch.Tensor, torch.Tensor]]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Convert labeled support image/mask pairs to non-parametric memory tokens."""
    return [model.encode_observation(image, mask) for image, mask in supports]
