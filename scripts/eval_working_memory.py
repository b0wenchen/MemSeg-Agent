#!/usr/bin/env python
"""Evaluate static memory with simulated clinician-corrected working memory."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from memseg_agent.data import NiftiSliceDataset, iter_volume_slices
from memseg_agent.model import MemSegAgent
from memseg_agent.protocol import get_split
from memseg_agent.controller import MemoryController
from memseg_agent.fewshot import encode_support_set
from memseg_agent.utils import binary_dice, dice_from_counts, load_yaml, resolve_device, set_seed
from memseg_agent.working_memory import DINOv3Embedder, WorkingMemoryBuffer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--static-checkpoint", required=True)
    parser.add_argument("--static-only", action="store_true", help="Disable working-memory feedback; explicit supports/prototypes remain available")
    parser.add_argument("--feedback-dice-threshold", type=float, default=0.5)
    parser.add_argument("--save-working-memory", default=None)
    parser.add_argument("--external", action="store_true", help="Evaluate ALL external volumes, with no train/validation split")
    parser.add_argument("--split-manifest")
    parser.add_argument("--eval-profile", choices=["training", "cross_domain"], default=None, help="Defaults to cross_domain for external tests, training otherwise")
    parser.add_argument("--output", default="outputs/evaluation.json")
    parser.add_argument("--support-root", help="Separate labeled support set; never sample supports from test")
    parser.add_argument("--support-slices", type=int, default=1)
    parser.add_argument("--prototype-root", help="Training-domain images for DINO prototype-based controller routing")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg.get("device"))
    external = args.external or cfg.get("evaluation_set") == "external"
    if external and args.split_manifest:
        parser.error("External evaluation does not accept a train/validation/test split.")
    dataset = NiftiSliceDataset(cfg["data_root"], num_classes=cfg["num_classes"], mapping_id_column=cfg.get("mapping_id_column", "patient_id"))
    if external:
        test_patients = set(dataset.volume_patient_ids)
        expected = cfg.get("expected_external_volumes")
        if expected is not None and len(dataset.volumes) != expected:
            parser.error(f"Expected {expected} external volumes, found {len(dataset.volumes)}")
    else:
        test_patients = set(get_split(dataset, args.fold, cfg["seed"], args.split_manifest).test_patients)
    checkpoint = torch.load(args.static_checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("static_memory", checkpoint)
    memory_mode = "direct_tokens" if "learnable_memory" in state or "memory" in state else "pseudo_observation"
    if memory_mode == "direct_tokens" and not args.static_only:
        parser.error("Use --static-only for direct-token checkpoints; our working-memory workflow uses pseudo-observation training.")
    if external and cfg.get("dataset_name", "").lower() == "cardiacuda" and cfg["target_class"] not in (1, 3):
        parser.error("CardiacUDA supports CAMUS-aligned LV=1 and LA=3 only.")
    profile = args.eval_profile or cfg.get("eval_profile", "cross_domain" if external else "training")
    model = MemSegAgent(cfg["sam2_config"], cfg["sam2_checkpoint"], cfg["target_class"], memory_mode=memory_mode, eval_profile=profile).to(device).eval()
    model.load_static_state_dict(state)
    buffer = None
    embedder = DINOv3Embedder(cfg["dino_model_id"], device) if not args.static_only or args.prototype_root else None
    if not args.static_only:
        buffer = WorkingMemoryBuffer(embedder, capacity=cfg["working_memory_capacity"], top_k=cfg["working_memory_top_k"], add_similarity_threshold=cfg["working_memory_add_similarity"], min_foreground_pixels=cfg["working_memory_min_foreground"])
    supports = []
    support_records = []
    if args.support_root:
        if args.support_slices < 1:
            parser.error("--support-slices must be positive")
        support_data = NiftiSliceDataset(args.support_root, cfg["num_classes"])
        if set(support_data.volume_patient_ids) & test_patients:
            parser.error("Support patients overlap test patients; use stable, dataset-qualified patient IDs.")
        for info, image, label in iter_volume_slices(support_data, set(support_data.volume_patient_ids)):
            mask = label == cfg["target_class"]
            if mask.any():
                supports.append((image, mask))
                support_records.append({"patient_id": info.patient_id, "volume": info.volume_name, "slice": info.slice_index})
            if len(supports) == args.support_slices:
                break
        if len(supports) != args.support_slices:
            parser.error("Not enough foreground-containing support slices")
        supports = encode_support_set(model, supports)
    prototype = None
    if args.prototype_root:
        proto_data = NiftiSliceDataset(args.prototype_root, cfg["num_classes"])
        if set(proto_data.volume_patient_ids) & test_patients:
            parser.error("Prototype patients overlap test patients")
        keys = [embedder.encode(proto_data[i][0]) for i in range(min(16, len(proto_data)))]
        prototype = F.normalize(torch.stack(keys).mean(0), dim=-1)
    controller = MemoryController(cfg.get("static_similarity_threshold", 0.85))
    totals: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    updates = 0
    route_counts = defaultdict(int)
    for info, image, label in tqdm(iter_volume_slices(dataset, test_patients)):
        with torch.no_grad():
            working = buffer.retrieve(image) if buffer is not None else []
            similarity = float(F.cosine_similarity(embedder.encode(image).reshape(1, -1), prototype.reshape(1, -1))) if prototype is not None else None
            plan = controller.select(similarity, bool(supports), bool(working))
            route_counts[f"fewshot={plan.use_few_shot},working={plan.use_working}"] += 1
            logits, _ = model(image.unsqueeze(0).to(device), working_memory=controller.compose(plan, supports, working), use_static_memory=plan.use_static)
            prediction = F.interpolate(torch.sigmoid(logits), size=label.shape, mode="bilinear", align_corners=False)[0, 0] >= 0.5
        target = label == cfg["target_class"]
        dice = binary_dice(prediction.cpu(), target)
        totals[info.volume_index][0] += int((prediction.cpu() & target).sum()) * 2
        totals[info.volume_index][1] += int(prediction.sum()) + int(target.sum())
        # This is an oracle-feedback simulation: the label is a clinician correction.
        if buffer is not None and dice < args.feedback_dice_threshold:
            result = buffer.update(model, image, target, dice)
            updates += result["action"] != "skipped_small_foreground"
    volume_dice = [dice_from_counts(numerator, denominator) for numerator, denominator in totals.values()]
    report = {"evaluation_set": "external" if external else "in_domain", "eval_profile": profile, "test_patients": sorted(test_patients), "support_slices": support_records, "routing": dict(route_counts), "oracle_feedback": buffer is not None, "updates": updates, "volumes": [{"volume_index": i, "patient_id": dataset.volume_patient_ids[i], "dice": dice_from_counts(*counts)} for i, counts in totals.items()], "mean_volume_dice": sum(volume_dice) / len(volume_dice), "config": cfg, "static_checkpoint": args.static_checkpoint}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(f"test volumes={len(volume_dice)} mean_volume_dice={sum(volume_dice) / max(1, len(volume_dice)):.4f} working_memory_entries={len(buffer) if buffer is not None else 0} updates={updates}")
    if args.save_working_memory and buffer is not None:
        path = Path(args.save_working_memory)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(buffer.state_dict(), path)


if __name__ == "__main__":
    main()
