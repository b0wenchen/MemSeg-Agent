"""Serializable patient splits shared by training and evaluation."""
import json
import math
from pathlib import Path

import numpy as np
from torch.utils.data import Subset

from .data import PatientSplit, patient_five_fold_split


def get_split(dataset, fold=0, seed=42, manifest=None, train_fraction=1.0):
    if not 0 < train_fraction <= 1:
        raise ValueError("train_fraction must be in (0, 1]")
    if manifest:
        groups = json.loads(Path(manifest).read_text())
        groups = {key: frozenset(groups[key]) for key in ("train", "val", "test")}
    else:
        split = patient_five_fold_split(dataset, fold, seed)
        groups = {key: getattr(split, key + "_patients") for key in ("train", "val", "test")}
    known = set(dataset.volume_patient_ids)
    for name, ids in groups.items():
        if not ids or not ids <= known:
            raise ValueError(f"Empty or unknown patients in {name} split")
    if groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"]:
        raise ValueError("Patient overlap in split manifest")
    if train_fraction < 1:
        ids = sorted(groups["train"])
        np.random.default_rng(seed).shuffle(ids)
        groups["train"] = frozenset(ids[:max(1, math.ceil(len(ids) * train_fraction))])
    subsets = [Subset(dataset, [i for i, info in enumerate(dataset.slice_info) if info.patient_id in groups[k]]) for k in ("train", "val", "test")]
    return PatientSplit(*subsets, groups["train"], groups["val"], groups["test"])


def split_dict(split):
    return {k: sorted(getattr(split, k + "_patients")) for k in ("train", "val", "test")}
