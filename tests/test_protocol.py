import csv
import json
from types import SimpleNamespace
import nibabel as nib
import numpy as np
import pytest
from memseg_agent.data import NiftiSliceDataset
from memseg_agent.protocol import get_split, split_dict
from memseg_agent.utils import dice_from_counts


class ToyDataset:
    def __init__(self):
        self.volume_patient_ids = [f"p{i}" for i in range(20) for _ in range(2)]
        self.volumes = [None] * 40
        self.slice_info = [SimpleNamespace(patient_id=pid) for pid in self.volume_patient_ids]


def test_patient_split_and_fraction(tmp_path):
    dataset = ToyDataset()
    full = get_split(dataset)
    assert (len(full.train_patients), len(full.val_patients), len(full.test_patients)) == (12, 4, 4)
    small = get_split(dataset, train_fraction=0.1)
    assert len(small.train_patients) == 2
    assert small.train_patients <= full.train_patients
    assert small.val_patients == full.val_patients
    assert small.test_patients == full.test_patients
    path = tmp_path / "split.json"
    path.write_text(json.dumps(split_dict(full)))
    assert split_dict(get_split(dataset, manifest=path)) == split_dict(full)
    bad = split_dict(full)
    bad["train"].append(bad["test"][0])
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="overlap"):
        get_split(dataset, manifest=path)


def test_external_mapping(tmp_path):
    for kind in ("image", "label"):
        nib.save(nib.Nifti1Image(np.zeros((4, 4, 2)), np.eye(4)), tmp_path / f"{kind}_1.nii.gz")
    with (tmp_path / "index_mapping.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["new_index", "base"])
        writer.writerow([1, "external-case-1"])
    dataset = NiftiSliceDataset(tmp_path, mapping_id_column="base")
    assert dataset.volume_patient_ids == ["external-case-1"]
    assert len(dataset) == 2


def test_empty_volume_dice():
    assert dice_from_counts(0, 0) == 1
    assert dice_from_counts(0, 3) == 0
    assert dice_from_counts(4, 4) == 1
