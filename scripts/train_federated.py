"""Single-process, patient-partitioned FedAvg simulation (no network deployment)."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from memseg_agent.data import NiftiSliceDataset
from memseg_agent.protocol import get_split, split_dict
from memseg_agent.model import MemSegAgent
from memseg_agent.federated import fedavg_static_memories
from memseg_agent.utils import load_yaml, set_seed, resolve_device
from train_static import binary_target, soft_dice_loss, validate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--split-manifest")
    p.add_argument("--sites", type=int, default=4)
    p.add_argument("--site-manifest", help="JSON list of lists of training patient IDs, one list per site")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--local-epochs", type=int, default=1)
    args = p.parse_args()
    cfg = load_yaml(args.config)
    if cfg.get("evaluation_set") == "external":
        p.error("External data is test-only")
    cfg["memory_mode"] = "pseudo_observation"
    cfg["federated_rounds"] = args.rounds
    cfg["federated_local_epochs"] = args.local_epochs
    if min(args.sites, args.rounds, args.local_epochs) < 1:
        p.error("sites, rounds and local-epochs must be positive")
    set_seed(cfg["seed"])
    device = resolve_device(cfg.get("device"))
    dataset = NiftiSliceDataset(cfg["data_root"], cfg["num_classes"])
    split = get_split(dataset, args.fold, cfg["seed"], args.split_manifest)
    ids = sorted(split.train_patients)
    if args.site_manifest:
        sites = json.loads(Path(args.site_manifest).read_text())
    else:
        np.random.default_rng(cfg["seed"]).shuffle(ids)
        sites = [part.tolist() for part in np.array_split(ids, args.sites)]
    flat = [pid for site in sites for pid in site]
    if not sites or any(not site for site in sites) or len(set(flat)) != len(flat) or set(flat) != split.train_patients:
        p.error("Sites must partition all and only training patients, without overlap or empty sites")
    subsets = [Subset(dataset, [i for i, info in enumerate(dataset.slice_info) if info.patient_id in site]) for site in sites]
    model = MemSegAgent(cfg["sam2_config"], cfg["sam2_checkpoint"], cfg["target_class"], memory_mode="pseudo_observation").to(device)
    global_state = model.static_state_dict()
    out = Path(cfg["output_dir"]) / "federated" / f"fold_{args.fold + 1}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "protocol.json").write_text(json.dumps({"split": split_dict(split), "sites": sites, "aggregation_weight": "training_slices", "local_optimizer": "AdamW reset each round"}, indent=2))
    val_loader = DataLoader(split.val, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"])
    best = -1
    history = []
    for round_id in range(1, args.rounds + 1):
        states = []
        for subset in subsets:
            model.load_static_state_dict(global_state)
            model.train()
            optimizer = torch.optim.AdamW(model.memory_parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
            loader = DataLoader(subset, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
            for _ in range(args.local_epochs):
                for image, label in loader:
                    logits, objectness = model(image.to(device))
                    target = F.interpolate(binary_target(label.to(device), cfg["target_class"]), size=logits.shape[-2:], mode="nearest")
                    loss = soft_dice_loss(logits, target) + cfg["object_loss_weight"] * F.binary_cross_entropy_with_logits(objectness.flatten(), target.flatten(1).any(1).float())
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
            states.append(model.static_state_dict())
        global_state = fedavg_static_memories(states, [len(s) for s in subsets])
        model.load_static_state_dict(global_state)
        score = validate(model, val_loader, cfg["target_class"], device)
        payload = {"static_memory": global_state, "round": round_id, "val_slice_dice": score, "config": cfg, "split": split_dict(split), "sites": sites}
        torch.save(payload, out / "last_static_memory.pt")
        if score > best:
            best = score
            torch.save(payload, out / "best_static_memory.pt")
        history.append({"round": round_id, "val_slice_dice": score})
        (out / "metrics.json").write_text(json.dumps(history, indent=2))
        print(history[-1], flush=True)


if __name__ == "__main__":
    main()
