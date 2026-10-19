"""Retrieval and gated test-time working-memory updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F


class ImageEmbedder(Protocol):
    def encode(self, image: torch.Tensor) -> torch.Tensor: ...


class DINOv3Embedder:
    """Optional Hugging Face DINOv3 embedder used for memory retrieval."""

    def __init__(self, model_id: str, device: torch.device) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install optional retrieval dependencies: pip install -e '.[retrieval]'") from exc
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 4:
            image = image[0]
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"Expected [3,H,W], got {tuple(image.shape)}")
        array = image.detach().float().cpu()
        array = (array * 255 if array.max() <= 1 else array).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        inputs = self.processor(images=array, return_tensors="pt")
        output = self.model(**{key: value.to(self.device) for key, value in inputs.items()})
        vector = output.pooler_output if getattr(output, "pooler_output", None) is not None else output.last_hidden_state[:, 0]
        return F.normalize(vector.squeeze(0).float(), dim=0).cpu()


@dataclass
class WorkingMemoryEntry:
    key: torch.Tensor
    memory: torch.Tensor
    position: torch.Tensor
    age: int


class WorkingMemoryBuffer:
    """Bounded top-k memory store with similarity-gated EMA delta updates.

    A new corrected observation is added if it is dissimilar to existing keys;
    otherwise its SAM2 memory token is blended with the nearest entry. ``dice``
    is the pre-correction prediction Dice, so lower Dice produces a stronger
    correction. Ground-truth corrections are supplied by the caller; this class
    never treats a model prediction as a human correction.
    """

    def __init__(
        self,
        embedder: ImageEmbedder,
        capacity: int = 64,
        top_k: int = 3,
        add_similarity_threshold: float = 0.95,
        similarity_midpoint: float = 0.90,
        similarity_temperature: float = 0.03,
        min_foreground_pixels: int = 100,
    ) -> None:
        self.embedder = embedder
        self.capacity = capacity
        self.top_k = top_k
        self.add_similarity_threshold = add_similarity_threshold
        self.similarity_midpoint = similarity_midpoint
        self.similarity_temperature = similarity_temperature
        self.min_foreground_pixels = min_foreground_pixels
        self.entries: list[WorkingMemoryEntry] = []
        self._clock = 0

    def __len__(self) -> int:
        return len(self.entries)

    def _key(self, image: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.embedder.encode(image).detach().float().cpu(), dim=0)

    def retrieve(self, image: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if not self.entries:
            return []
        key = self._key(image)
        similarities = torch.stack([torch.dot(key, entry.key) for entry in self.entries])
        count = min(self.top_k, len(self.entries))
        indices = torch.topk(similarities, k=count).indices.tolist()
        self._clock += 1
        selected: list[tuple[torch.Tensor, torch.Tensor]] = []
        for index in indices:
            self.entries[index].age = self._clock
            selected.append((self.entries[index].memory, self.entries[index].position))
        return selected

    def update(self, model, image: torch.Tensor, corrected_mask: torch.Tensor, dice: float) -> dict[str, float | int | str | None]:
        """Write a clinician-corrected observation using gated delta adaptation."""
        mask = corrected_mask if corrected_mask.max() <= 1 else (corrected_mask == model.target_class)
        if int(mask.sum().item()) < self.min_foreground_pixels:
            return {"action": "skipped_small_foreground", "index": None, "similarity": None, "old_weight": None}
        key = self._key(image)
        memory, position = model.encode_observation(image, mask)
        self._clock += 1
        if not self.entries:
            self.entries.append(WorkingMemoryEntry(key, memory.cpu(), position.cpu(), self._clock))
            return {"action": "added", "index": 0, "similarity": None, "old_weight": None}
        similarities = torch.stack([torch.dot(key, entry.key) for entry in self.entries])
        similarity, index_tensor = torch.max(similarities, dim=0)
        similarity_value, index = float(similarity), int(index_tensor)
        if similarity_value < self.add_similarity_threshold:
            self._append_or_evict(WorkingMemoryEntry(key, memory.cpu(), position.cpu(), self._clock))
            return {"action": "added", "index": len(self.entries) - 1, "similarity": similarity_value, "old_weight": None}
        # Faithful to the original gated-delta rule: high similarity and a less
        # severe error retain more of the existing entry.
        retain = torch.sigmoid(torch.tensor((similarity_value - self.similarity_midpoint) / self.similarity_temperature)).item()
        retain *= max(0.0, min(1.0, dice / 0.5))
        retain = max(0.05, min(0.95, retain))
        entry = self.entries[index]
        entry.key = F.normalize(retain * entry.key + (1 - retain) * key, dim=0)
        entry.memory = (retain * entry.memory + (1 - retain) * memory.cpu()).detach()
        # Positional encodings are coordinate-dependent and retained unchanged.
        entry.age = self._clock
        return {"action": "ema_updated", "index": index, "similarity": similarity_value, "old_weight": retain}

    def _append_or_evict(self, entry: WorkingMemoryEntry) -> None:
        if len(self.entries) >= self.capacity:
            self.entries.pop(min(range(len(self.entries)), key=lambda i: self.entries[i].age))
        self.entries.append(entry)

    def state_dict(self) -> dict:
        return {"entries": [{"key": e.key, "memory": e.memory, "position": e.position, "age": e.age} for e in self.entries], "clock": self._clock}

    def load_state_dict(self, state: dict) -> None:
        self.entries = [WorkingMemoryEntry(item["key"], item["memory"], item["position"], item["age"]) for item in state.get("entries", [])]
        self._clock = int(state.get("clock", 0))
