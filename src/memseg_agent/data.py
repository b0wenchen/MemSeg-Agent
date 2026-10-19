"""Patient-safe NIfTI loading and 3:1:1 split utilities."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import nibabel as nib
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset, Subset


def _numeric_key(path: Path, prefix: str) -> tuple[int, str]:
    stem = re.sub(r"\.nii(\.gz)?$", "", path.name)
    value = re.sub(rf"^{re.escape(prefix)}", "", stem)
    return (int(value), value) if value.isdigit() else (10**12, value)


@dataclass(frozen=True)
class SliceInfo:
    volume_index: int
    slice_index: int
    patient_id: str
    volume_name: str


class NiftiSliceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Paired NIfTI volumes with optional patient identities from a mapping CSV."""

    def __init__(self, root: str | Path, num_classes: int | None = None) -> None:
        self.root = Path(root)
        self.num_classes = num_classes
        self._patient_by_key = self._read_mapping()
        image_paths = sorted(self.root.glob("image_*.nii*"), key=lambda p: _numeric_key(p, "image_"))
        label_paths = sorted(self.root.glob("label_*.nii*"), key=lambda p: _numeric_key(p, "label_"))
        image_by_key = {_numeric_key(p, "image_")[1]: p for p in image_paths}
        label_by_key = {_numeric_key(p, "label_")[1]: p for p in label_paths}
        keys = sorted(set(image_by_key) & set(label_by_key), key=lambda x: (int(x) if x.isdigit() else 10**12, x))
        if not keys:
            raise FileNotFoundError(f"No paired image_<id>.nii[.gz]/label_<id>.nii[.gz] files under {self.root}")

        self.volumes: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
        self.volume_patient_ids: list[str] = []
        self.slice_info: list[SliceInfo] = []
        self.volume_slice_indices: list[list[int]] = []
        for volume_index, key in enumerate(keys):
            image = np.asarray(nib.load(str(image_by_key[key])).get_fdata(), dtype=np.float32)
            label = np.asarray(nib.load(str(label_by_key[key])).get_fdata(), dtype=np.int64)
            if image.ndim == 2:
                image, label = image[..., None], label[..., None]
            if image.ndim != 3 or label.shape != image.shape:
                raise ValueError(f"Expected paired 3D arrays for key={key}; got {image.shape} and {label.shape}")
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            if self.num_classes is not None:
                label = np.clip(label, 0, self.num_classes - 1)
            patient_id = self._patient_by_key.get(key, key)
            slices: list[tuple[torch.Tensor, torch.Tensor]] = []
            flat_indices: list[int] = []
            for slice_index in range(image.shape[-1]):
                x = torch.from_numpy(np.repeat(image[..., slice_index][None], 3, axis=0).copy())
                y = torch.from_numpy(label[..., slice_index].copy()).long()
                slices.append((x, y))
                flat_indices.append(len(self.slice_info))
                self.slice_info.append(SliceInfo(volume_index, slice_index, patient_id, image_by_key[key].stem))
            self.volumes.append(slices)
            self.volume_patient_ids.append(patient_id)
            self.volume_slice_indices.append(flat_indices)

    def _read_mapping(self) -> dict[str, str]:
        mapping_path = self.root / "index_mapping.csv"
        if not mapping_path.exists():
            return {}
        with mapping_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"new_index", "patient_id"}
            if not required.issubset(reader.fieldnames or set()):
                raise ValueError(f"{mapping_path} needs columns {sorted(required)}")
            return {str(int(float(row["new_index"]))): str(row["patient_id"]) for row in reader}

    def __len__(self) -> int:
        return len(self.slice_info)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        info = self.slice_info[index]
        return self.volumes[info.volume_index][info.slice_index]

    @property
    def patient_count(self) -> int:
        return len(set(self.volume_patient_ids))


@dataclass(frozen=True)
class PatientSplit:
    train: Subset
    val: Subset
    test: Subset
    train_patients: frozenset[str]
    val_patients: frozenset[str]
    test_patients: frozenset[str]


def patient_five_fold_split(dataset: NiftiSliceDataset, fold: int, seed: int = 42) -> PatientSplit:
    """A deterministic patient-level 3:1:1 split, using one of five test folds."""
    patient_ids = np.asarray(dataset.volume_patient_ids, dtype=object)
    volume_ids = np.arange(len(dataset.volumes))
    folds = list(GroupKFold(n_splits=5).split(volume_ids, groups=patient_ids))
    if not 0 <= fold < len(folds):
        raise ValueError(f"fold must be in [0, {len(folds) - 1}], got {fold}")
    train_val_positions, test_positions = folds[fold]
    train_val_patients = np.unique(patient_ids[train_val_positions]).copy()
    rng = np.random.default_rng(seed + fold)
    rng.shuffle(train_val_patients)
    n_val = max(1, len(train_val_patients) // 4)
    val_patients = frozenset(map(str, train_val_patients[:n_val]))
    train_patients = frozenset(map(str, train_val_patients[n_val:]))
    test_patients = frozenset(map(str, patient_ids[test_positions]))
    if train_patients & val_patients or train_patients & test_patients or val_patients & test_patients:
        raise RuntimeError("Patient leakage detected while building split")

    def collect(patients: frozenset[str]) -> list[int]:
        return [i for i, item in enumerate(dataset.slice_info) if item.patient_id in patients]

    return PatientSplit(Subset(dataset, collect(train_patients)), Subset(dataset, collect(val_patients)), Subset(dataset, collect(test_patients)), train_patients, val_patients, test_patients)


def iter_volume_slices(dataset: NiftiSliceDataset, patient_ids: set[str]) -> Iterator[tuple[SliceInfo, torch.Tensor, torch.Tensor]]:
    for index, info in enumerate(dataset.slice_info):
        if info.patient_id in patient_ids:
            image, label = dataset[index]
            yield info, image, label
