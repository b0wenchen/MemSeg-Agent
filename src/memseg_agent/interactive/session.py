"""Host-independent segmentation session. No static memory or automatic feedback.

The interactive bank preserves individual corrected cases instead of performing
the experimental EMA merge: users can inspect, remove and reuse exact examples.
Persistence exports only reviewed image/mask pairs, not executable checkpoints.
"""

from __future__ import annotations

import hashlib
import io
import json
import threading
import uuid
import zipfile
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MAX_PIXELS = 2048 * 2048
MAX_CASES = 32
MAX_ARCHIVE_BYTES = 192 * 1024 * 1024


def normalize_image(image):
    """Use explicit uint8 HWC RGB / HW grayscale inputs; no hidden normalization."""
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise ValueError("Use uint8 images; explicitly normalize other intensity ranges first")
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected a 2D grayscale image or an H×W×3 RGB image")
    if min(array.shape[:2]) < 2 or array.shape[0] * array.shape[1] > MAX_PIXELS:
        raise ValueError("Images must contain between 2×2 and 2048×2048 pixels")
    return np.ascontiguousarray(array).copy()


def binary_mask(mask, shape):
    mask = np.asarray(mask)
    if mask.shape != tuple(shape):
        raise ValueError("Mask dimensions must exactly match the image")
    if not np.isin(mask, [0, 1, 255]).all():
        raise ValueError("Masks must be binary (0/1 or 0/255); select one target first")
    return (mask > 0).copy()


def png_bytes(array):
    output = io.BytesIO()
    Image.fromarray(np.asarray(array)).save(output, format="PNG")
    return output.getvalue()


def decoded_array(image, *, mask=False):
    if image.mode not in ("1", "L", "RGB", "RGBA", "P"):
        raise ValueError("Use 8-bit images; normalize high-bit-depth data explicitly")
    if mask:
        rgba = np.asarray(image.convert("RGBA"))
        if not np.all(rgba[..., 3] == 255) or not np.all(rgba[..., :3] == rgba[..., :1]):
            raise ValueError("Use an opaque grayscale binary mask, not a colored overlay")
        return binary_mask(rgba[..., 0], rgba.shape[:2]).astype(np.uint8)
    return np.asarray(image.convert("RGB")).copy()


def read_png(payload, mask=False):
    with Image.open(io.BytesIO(payload)) as image:
        if image.format != "PNG" or image.width * image.height > MAX_PIXELS:
            raise ValueError("Expected a PNG with at most 2048×2048 pixels")
        return decoded_array(image, mask=mask)


@dataclass
class MemoryCase:
    id: str
    name: str
    bank: str
    fingerprint: str
    image: np.ndarray
    mask: np.ndarray
    key: torch.Tensor
    tokens: tuple[torch.Tensor, torch.Tensor]


class SegmentationSession:
    """One target, one local user; both banks are isolated per session.

    ``model`` must expose encode_observation and the MemSegAgent forward API.
    ``embedder`` must expose encode(image_3hw). All model use is inference-only.
    """

    def __init__(self, model, embedder, target="Foreground", top_k=2, capacity=MAX_CASES):
        if not 1 <= top_k <= 4 or not 1 <= capacity <= MAX_CASES:
            raise ValueError("top_k must be 1–4 and capacity 1–32")
        self.model = model.eval()
        self.embedder = embedder
        self.target = str(target).strip()[:120] or "Foreground"
        self.top_k, self.capacity = top_k, capacity
        self.cases: list[MemoryCase] = []
        self.current = None
        self.prediction = None
        self.lock = threading.RLock()

    @staticmethod
    def tensor(image):
        return torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255

    def _encode(self, image, mask):
        with torch.inference_mode():
            tensor = self.tensor(image)
            key = F.normalize(self.embedder.encode(tensor).detach().float().cpu().flatten(), dim=0)
            memory, position = self.model.encode_observation(tensor, torch.from_numpy(mask.copy()))
            if not torch.isfinite(key).all() or key.norm() == 0:
                raise ValueError("Invalid retrieval embedding")
            return key, (memory.detach().cpu(), position.detach().cpu())

    def _add(self, image, mask, name, bank):
        image = normalize_image(image)
        mask = binary_mask(mask, image.shape[:2])
        if bank == "fewshot" and not mask.any():
            raise ValueError("A few-shot example must contain foreground")
        fingerprint = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
        existing = next((c for c in self.cases if c.fingerprint == fingerprint), None)
        if existing and existing.bank != bank:
            raise ValueError("This image already belongs to the other memory bank")
        if not existing and len(self.cases) >= self.capacity:
            raise ValueError("Memory is full; remove a case before adding another")
        key, tokens = self._encode(image, mask)
        case = MemoryCase(
            existing.id if existing else uuid.uuid4().hex,
            str(name)[:120],
            bank,
            fingerprint,
            image,
            mask,
            key,
            tokens,
        )
        if existing:
            self.cases[self.cases.index(existing)] = case
        else:
            self.cases.append(case)
        return case.id

    def add_support(self, image, mask, name="Support"):
        with self.lock:
            return self._add(image, mask, name, "fewshot")

    def set_query(self, image, name="Query"):
        with self.lock:
            self.current = {
                "id": uuid.uuid4().hex,
                "name": str(name)[:120],
                "image": normalize_image(image),
            }
            self.prediction = None
            return self.current["id"]

    def _query(self, case_id):
        if self.current is None or self.current["id"] != case_id:
            raise ValueError("Query changed; reload the current image before editing")
        return self.current

    def predict(self, case_id):
        with self.lock:
            query = self._query(case_id)
            prediction, retrieved = self.predict_image(query["image"])
            self.prediction = prediction.copy()
            return prediction, retrieved

    def predict_image(self, image):
        """Infer a slice atomically without replacing the browser's current query."""
        with self.lock, torch.inference_mode():
            image = normalize_image(image)
            if not self.cases:
                raise ValueError("Add at least one labeled few-shot example first")
            tensor = self.tensor(image)
            key = F.normalize(self.embedder.encode(tensor).float().cpu().flatten(), dim=0)
            if not torch.isfinite(key).all() or key.norm() == 0:
                raise ValueError("Invalid retrieval embedding")
            retrieved = []
            for bank in ("fewshot", "working"):
                scored = [(float(torch.dot(key, c.key)), c) for c in self.cases if c.bank == bank]
                retrieved.extend(
                    sorted(scored, key=lambda item: item[0], reverse=True)[: self.top_k]
                )
            logits, _ = self.model(
                tensor[None],
                working_memory=[c.tokens for _, c in retrieved],
                use_static_memory=False,
            )
            probability = F.interpolate(
                logits.sigmoid(),
                size=image.shape[:2],
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            if not torch.isfinite(probability).all():
                raise ValueError("Model produced non-finite predictions")
            prediction = (probability >= 0.5).cpu().numpy().copy()
            return prediction, [
                {"id": c.id, "name": c.name, "bank": c.bank, "similarity": score}
                for score, c in retrieved
            ]

    def confirm_correction(self, case_id, mask, *, confirmed=False):
        """Explicit confirmation is required, including for an all-background mask."""
        if confirmed is not True:
            raise ValueError("Confirm that the mask has been reviewed before saving")
        with self.lock:
            query = self._query(case_id)
            return self._add(query["image"], mask, query["name"], "working")

    def add_correction(self, image, mask, name="Correction", *, confirmed=False):
        """Save an explicitly reviewed image/mask pair from an external editor."""
        if confirmed is not True:
            raise ValueError("Confirm that the mask has been reviewed before saving")
        with self.lock:
            return self._add(image, mask, name, "working")

    def remove(self, memory_id):
        with self.lock:
            if not any(c.id == memory_id for c in self.cases):
                raise ValueError("Memory case not found")
            self.cases = [c for c in self.cases if c.id != memory_id]

    def reset(self, target=None):
        with self.lock:
            self.cases, self.current, self.prediction = [], None, None
            if target is not None:
                self.target = str(target).strip()[:120] or "Foreground"

    def summary(self):
        with self.lock:
            return {
                "target": self.target,
                "capacity": self.capacity,
                "top_k": self.top_k,
                "current_id": self.current["id"] if self.current else None,
                "memories": [
                    {
                        "id": c.id,
                        "name": c.name,
                        "bank": c.bank,
                        "shape": list(c.mask.shape),
                        "foreground_pixels": int(c.mask.sum()),
                    }
                    for c in self.cases
                ],
            }

    def export_archive(self):
        """Portable reviewed examples only. Treat this archive as sensitive data."""
        with self.lock:
            output = io.BytesIO()
            manifest = {"format": "memseg-reviewed-cases-v1", "target": self.target, "cases": []}
            with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
                total_size = 0
                for i, case in enumerate(self.cases):
                    image_name, mask_name = f"{i}/image.png", f"{i}/mask.png"
                    image_data = png_bytes(case.image)
                    mask_data = png_bytes(case.mask.astype(np.uint8) * 255)
                    total_size += len(image_data) + len(mask_data)
                    # Reserve room for the manifest and ZIP headers so every
                    # successful export can be imported under the same limits.
                    if total_size > MAX_ARCHIVE_BYTES - 1024 * 1024:
                        raise ValueError(
                            "Export exceeds 192 MiB; download masks individually and reduce the bank size"
                        )
                    archive.writestr(image_name, image_data)
                    archive.writestr(mask_name, mask_data)
                    manifest["cases"].append(
                        {
                            "name": case.name,
                            "bank": case.bank,
                            "image": image_name,
                            "mask": mask_name,
                        }
                    )
                archive.writestr("manifest.json", json.dumps(manifest))
            return output.getvalue()

    def import_archive(self, payload):
        """Atomic import into an empty session; re-encode with the current model.

        Never loads pickle/PT files or extracts paths to disk. Bounded ZIP sizes
        reject compression bombs before decompressing image members.
        """
        with self.lock:
            if self.cases or self.current:
                raise ValueError("Start a new empty session before importing memories")
            if len(payload) > MAX_ARCHIVE_BYTES:
                raise ValueError("Archive is too large")
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                info = archive.infolist()
                if (
                    len(info) > 2 * self.capacity + 1
                    or sum(x.file_size for x in info) > MAX_ARCHIVE_BYTES
                ):
                    raise ValueError("Archive exceeds memory limits")
                if archive.getinfo("manifest.json").file_size > 64000:
                    raise ValueError("Manifest too large")
                manifest = json.loads(archive.read("manifest.json"))
                if manifest.get("format") != "memseg-reviewed-cases-v1" or not isinstance(
                    manifest.get("cases"), list
                ):
                    raise ValueError("Unsupported memory archive")
                if len(manifest["cases"]) > self.capacity:
                    raise ValueError("Too many memory cases")
                staged = SegmentationSession(
                    self.model,
                    self.embedder,
                    manifest.get("target", "Foreground"),
                    self.top_k,
                    self.capacity,
                )
                for item in manifest["cases"]:
                    if item["bank"] not in ("fewshot", "working"):
                        raise ValueError("Unknown memory bank")
                    image = read_png(archive.read(item["image"]))
                    mask = read_png(archive.read(item["mask"]), mask=True)
                    staged._add(image, mask, item["name"], item["bank"])
                self.cases, self.target = staged.cases, staged.target
