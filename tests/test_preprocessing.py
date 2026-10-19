import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")
from memseg_agent.preprocessing.chaos import preprocess_chaos
from memseg_agent.preprocessing.camus import preprocess_camus_2d
from memseg_agent.preprocessing.acdc import preprocess_acdc


def pair(image_path, label_path, shape):
    arr = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    label = np.zeros(shape, dtype=np.uint8)
    label[..., 2:6, 2:6] = 1
    sitk.WriteImage(sitk.GetImageFromArray(arr), str(image_path))
    sitk.WriteImage(sitk.GetImageFromArray(label), str(label_path))


@pytest.mark.parametrize("dataset", ["camus", "acdc", "chaos"])
def test_preprocessing_pair(tmp_path, dataset):
    root, out = tmp_path / "raw", tmp_path / "output"
    root.mkdir()
    out.mkdir()
    if dataset == "chaos":
        pair(root / "image_1.nii.gz", root / "label_1.nii.gz", (3, 12, 12))
        preprocess_chaos(root, out, spacing=(1, 1, 1), crop=8)
    else:
        patient = root / "patient001"
        patient.mkdir()
        if dataset == "camus":
            pair(patient / "patient001_2CH_ED.nii.gz", patient / "patient001_2CH_ED_gt.nii.gz", (12, 12))
            preprocess_camus_2d(str(root), str(out), crop_hw=(8, 8), new_spacing_xy=(1, 1), logging=False)
        else:
            (patient / "Info.cfg").write_text("ED: 1\nES: 2\n")
            for frame in (1, 2):
                pair(patient / f"patient001_frame{frame:02d}.nii.gz", patient / f"patient001_frame{frame:02d}_gt.nii.gz", (3, 12, 12))
            preprocess_acdc(str(root), str(out), crop_hw=(8, 8), new_spacing=(1, 1, 1), logging=False)
    image = sitk.ReadImage(str(out / "image_1.nii.gz"))
    label = sitk.ReadImage(str(out / "label_1.nii.gz"))
    assert image.GetSize()[:2] == (8, 8)
    assert image.GetSize() == label.GetSize()
    assert (out / "index_mapping.csv").exists()


def test_cardiacuda_preprocessing(tmp_path):
    pytest.importorskip("cv2")
    import nibabel as nib
    from memseg_agent.preprocessing.cardiacuda import preprocess_cardiacuda_allframe_video
    root, out = tmp_path / "raw", tmp_path / "output"
    root.mkdir()
    image = np.ones((12, 12, 2), dtype=np.float32)
    labels = np.zeros((12, 12, 2), dtype=np.int16)
    labels[3:9, 3:9, :] = 1
    for suffix, array in (("image", image), ("label", labels)):
        nib.save(nib.Nifti1Image(array, np.eye(4)), root / f"patient-1-4_{suffix}.nii.gz")
    preprocess_cardiacuda_allframe_video(str(root), str(out), resize_hw=(12, 12), crop_hw=(8, 8), logging=False)
    assert nib.load(out / "image_1.nii.gz").shape == (8, 8, 2)
    assert np.max(nib.load(out / "label_1.nii.gz").get_fdata()) == 1
