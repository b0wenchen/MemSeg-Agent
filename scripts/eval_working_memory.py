#!/usr/bin/env python
"""Evaluate static memory with simulated clinician-corrected working memory."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from memseg_agent.data import NiftiSliceDataset, iter_volume_slices, patient_five_fold_split
from memseg_agent.model import MemSegAgent
from memseg_agent.utils import binary_dice, load_yaml, resolve_device, set_seed
from memseg_agent.working_memory import DINOv3Embedder, WorkingMemoryBuffer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--static-checkpoint", required=True)
    parser.add_argument("--static-only", action="store_true", help="Evaluate either memory parameterization without feedback or DINO")
    parser.add_argument("--feedback-dice-threshold", type=float, default=0.5)
    parser.add_argument("--save-working-memory", default=None)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg.get("device"))
    dataset = NiftiSliceDataset(cfg["data_root"], num_classes=cfg["num_classes"])
    split = patient_five_fold_split(dataset, args.fold, seed=cfg["seed"])
    checkpoint = torch.load(args.static_checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("static_memory", checkpoint)
    memory_mode = "direct_tokens" if "learnable_memory" in state or "memory" in state else "pseudo_observation"
    if memory_mode == "direct_tokens" and not args.static_only:
        parser.error("Use --static-only for direct-token checkpoints; our working-memory workflow uses pseudo-observation training.")
    model = MemSegAgent(cfg["sam2_config"], cfg["sam2_checkpoint"], cfg["target_class"], memory_mode=memory_mode).to(device).eval()
    model.load_static_state_dict(state)
    buffer = None
    if not args.static_only:
        buffer = WorkingMemoryBuffer(DINOv3Embedder(cfg["dino_model_id"], device), capacity=cfg["working_memory_capacity"], top_k=cfg["working_memory_top_k"], add_similarity_threshold=cfg["working_memory_add_similarity"], min_foreground_pixels=cfg["working_memory_min_foreground"])
    totals: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    updates = 0
    for info, image, label in tqdm(iter_volume_slices(dataset, set(split.test_patients)), total=len(split.test)):
        with torch.no_grad():
            logits, _ = model(image.unsqueeze(0).to(device), working_memory=buffer.retrieve(image) if buffer is not None else None)
            prediction = F.interpolate(torch.sigmoid(logits), size=label.shape, mode="bilinear", align_corners=False)[0, 0] >= 0.5
        target = label == cfg["target_class"]
        dice = binary_dice(prediction.cpu(), target)
        totals[info.volume_index][0] += int((prediction.cpu() & target).sum()) * 2
        totals[info.volume_index][1] += int(prediction.sum()) + int(target.sum())
        # This is an oracle-feedback simulation: the label is a clinician correction.
        if buffer is not None and dice < args.feedback_dice_threshold:
            result = buffer.update(model, image, target, dice)
            updates += result["action"] != "skipped_small_foreground"
    volume_dice = [numerator / max(denominator, 1) for numerator, denominator in totals.values()]
    print(f"test volumes={len(volume_dice)} mean_volume_dice={sum(volume_dice) / max(1, len(volume_dice)):.4f} working_memory_entries={len(buffer) if buffer is not None else 0} updates={updates}")
    if args.save_working_memory and buffer is not None:
        path = Path(args.save_working_memory)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(buffer.state_dict(), path)


if __name__ == "__main__":
    main()
