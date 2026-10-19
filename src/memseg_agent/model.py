"""Frozen-SAM2 segmentation model conditioned by static and working memories."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MemSegAgent(nn.Module):
    """SAM2 with trainable pseudo-observations and composable memory tokens.

    Static supervision optimizes only ``mem_img_logits`` and
    ``mem_mask_logits``. SAM2 and its memory encoder stay frozen. Corrected
    observations can be encoded into the same space through
    :meth:`encode_observation` and composed at inference time.
    """

    def __init__(self, sam2_config: str | Path, sam2_checkpoint: str | Path, target_class: int = 1) -> None:
        super().__init__()
        try:
            from sam2.build_sam import build_sam2
        except ImportError as exc:  # pragma: no cover - external dependency
            raise ImportError("Install SAM2: pip install git+https://github.com/facebookresearch/sam2.git") from exc
        self.target_class = target_class
        self.sam2 = build_sam2(str(sam2_config), str(sam2_checkpoint), device="cpu")
        self.sam2.eval()
        for parameter in self.sam2.parameters():
            parameter.requires_grad_(False)
        self.mem_img_logits = nn.Parameter(torch.randn(1, 3, 1024, 1024) * 0.02)
        self.mem_mask_logits = nn.Parameter(torch.randn(1, 1, 1024, 1024) * 0.02 + 2.0)

    def train(self, mode: bool = True) -> "MemSegAgent":
        super().train(mode)
        self.sam2.eval()  # Frozen normalization/dropout layers always use inference behavior.
        return self

    @property
    def device(self) -> torch.device:
        return self.mem_img_logits.device

    @staticmethod
    def prepare_image(image: torch.Tensor | np.ndarray, size: int = 1024) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        if image.ndim == 2:
            image = image[None, None].repeat(1, 3, 1, 1)
        elif image.ndim == 3 and image.shape[0] == 3:
            image = image[None]
        elif image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected [3,H,W] or [B,3,H,W], got {tuple(image.shape)}")
        image = image.float()
        if image.max() > 1:
            image = image / 255.0
        return F.interpolate(image, size=(size, size), mode="bilinear", align_corners=False)

    def _forward_image(self, image: torch.Tensor) -> dict:
        output = self.sam2.image_encoder(image)
        output["backbone_fpn"][0] = self.sam2.sam_mask_decoder.conv_s0(output["backbone_fpn"][0])
        output["backbone_fpn"][1] = self.sam2.sam_mask_decoder.conv_s1(output["backbone_fpn"][1])
        return output

    def _tokens_from_memory_encoder(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._forward_image(image)["backbone_fpn"][-1]
        result = self.sam2.memory_encoder(pix_feat=features, masks=mask, skip_mask_sigmoid=True)
        memory = result["vision_features"].flatten(2).permute(2, 0, 1)
        position = result["vision_pos_enc"][-1].flatten(2).permute(2, 0, 1)
        return memory, position

    def static_tokens(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Build differentiable tokens from the optimized pseudo-observation."""
        return self._tokens_from_memory_encoder(torch.sigmoid(self.mem_img_logits), torch.sigmoid(self.mem_mask_logits))

    @torch.no_grad()
    def encode_observation(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one corrected 2D observation into SAM2 memory-token space."""
        image = self.prepare_image(image).to(self.device)
        if mask.ndim == 2:
            mask = mask[None, None]
        elif mask.ndim == 3:
            mask = mask[:, None]
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError(f"Expected binary mask [H,W] or [B,1,H,W], got {tuple(mask.shape)}")
        mask = mask.to(self.device).float()
        if mask.max() > 1:
            mask = (mask == self.target_class).float()
        mask = F.interpolate(mask, size=(1024, 1024), mode="nearest")
        return self._tokens_from_memory_encoder(image, mask * 20.0 - 10.0)

    @staticmethod
    def _repeat_batch(tokens: torch.Tensor, batch_size: int) -> torch.Tensor:
        if tokens.shape[1] == batch_size:
            return tokens
        if tokens.shape[1] != 1:
            raise ValueError(f"Memory batch size must be 1 or {batch_size}, got {tokens.shape[1]}")
        return tokens.repeat(1, batch_size, 1)

    def forward(
        self,
        image: torch.Tensor,
        working_memory: Iterable[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_static_memory: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.prepare_image(image).to(self.device)
        backbone = self._forward_image(image)
        feature_maps = backbone["backbone_fpn"][-3:]
        pos_maps = backbone["vision_pos_enc"][-3:]
        current = [feature_maps[-1].flatten(2).permute(2, 0, 1)]
        current_pos = [pos_maps[-1].flatten(2).permute(2, 0, 1)]
        batch_size = image.shape[0]

        memory_parts: list[torch.Tensor] = []
        position_parts: list[torch.Tensor] = []
        if use_static_memory:
            memory, position = self.static_tokens()
            memory_parts.append(self._repeat_batch(memory, batch_size))
            position_parts.append(self._repeat_batch(position, batch_size))
        if working_memory is not None:
            for memory, position in working_memory:
                memory_parts.append(self._repeat_batch(memory.to(self.device), batch_size))
                position_parts.append(self._repeat_batch(position.to(self.device), batch_size))
        if memory_parts:
            enriched = self.sam2.memory_attention(curr=current, curr_pos=current_pos, memory=torch.cat(memory_parts, 0), memory_pos=torch.cat(position_parts, 0), num_obj_ptr_tokens=0)
        else:
            enriched = current[0]

        height, width = feature_maps[-1].shape[-2:]
        enriched = enriched.permute(1, 2, 0).reshape(batch_size, enriched.shape[-1], height, width)
        points = torch.zeros(batch_size, 1, 2, device=self.device)
        point_labels = -torch.ones(batch_size, 1, dtype=torch.int32, device=self.device)
        sparse, dense = self.sam2.sam_prompt_encoder(points=(points, point_labels), boxes=None, masks=None)
        masks, ious, _, objectness = self.sam2.sam_mask_decoder(
            image_embeddings=enriched,
            image_pe=self.sam2.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True,
            repeat_image=False,
            high_res_features=[backbone["backbone_fpn"][0], backbone["backbone_fpn"][1]],
        )
        floor = -6.0 if self.training else -1024.0
        object_probability = torch.sigmoid(objectness).view(-1, 1, 1, 1)
        masks = object_probability * masks + (1 - object_probability) * floor
        weights = F.softmax(ious, dim=-1)[:, :, None, None]
        return (weights * masks.float()).sum(dim=1, keepdim=True), objectness

    def static_state_dict(self) -> dict[str, torch.Tensor]:
        return {"mem_img_logits": self.mem_img_logits.detach().cpu(), "mem_mask_logits": self.mem_mask_logits.detach().cpu()}

    @torch.no_grad()
    def load_static_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        required = {"mem_img_logits", "mem_mask_logits"}
        if not required.issubset(state):
            raise KeyError(f"Static checkpoint needs {sorted(required)}, found {sorted(state)}")
        self.mem_img_logits.copy_(state["mem_img_logits"].to(self.device))
        self.mem_mask_logits.copy_(state["mem_mask_logits"].to(self.device))
