"""CHAOS T2 conversion, preserving the archived center-crop/spacing protocol.

The archived workflow uses dcm2niix, a PNG row flip, 99.5th percentile
clipping, label-wise linear resampling with rounding, and image-center crops.
Always visually inspect the converted image/label alignment before training.
"""
import csv
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import SimpleITK as sitk
from PIL import Image
from .acdc import resample_by_res, resample_lb_by_res, image_crop, copy_spacing_ori


def convert_raw(root, output):
    for patient in sorted(Path(root).iterdir()):
        series = patient / "T2SPIR"
        if not series.is_dir():
            continue
        subprocess.run(["dcm2niix", "-z", "y", "-f", f"image_{patient.name}", "-o", str(output), str(series / "DICOM_anon")], check=True)
        reference = sitk.ReadImage(str(output / f"image_{patient.name}.nii.gz"))
        pngs = sorted((series / "Ground").glob("*.png"), key=lambda p: int(p.stem.split("-")[-1]))
        if not pngs:
            raise ValueError(f"No PNG labels for {patient.name}")
        labels = np.flip(np.stack([np.asarray(Image.open(p)) for p in pngs]), axis=1).copy()
        # Fixed palette avoids shifting organ IDs when an organ is absent.
        if not set(np.unique(labels)) <= {0, 63, 126, 189, 252}:
            raise ValueError(f"Unexpected PNG palette for {patient.name}")
        labels = (labels // 63).astype(np.uint8)
        if labels.shape != sitk.GetArrayFromImage(reference).shape:
            raise ValueError(f"DICOM/PNG shape mismatch for {patient.name}")
        mask = sitk.GetImageFromArray(labels)
        mask.CopyInformation(reference)
        sitk.WriteImage(mask, str(output / f"label_{patient.name}.nii.gz"))


def preprocess_chaos(root, output, spacing=(1.25, 1.25, 7.7), crop=256, raw=False):
    output = Path(output)
    if raw:
        with tempfile.TemporaryDirectory(prefix="memseg-chaos-") as temporary:
            convert_raw(root, Path(temporary))
            return preprocess_chaos(temporary, output, spacing, crop)
    images = sorted(Path(root).glob("image_*.nii*"))
    if not images:
        raise ValueError("No converted CHAOS image_<id>.nii[.gz] files")
    with (output / "index_mapping.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["new_index", "patient_id"])
        for path in images:
            pid = path.name.removeprefix("image_").split(".")[0]
            label_path = path.with_name(path.name.replace("image_", "label_", 1))
            image, label = sitk.ReadImage(str(path)), sitk.ReadImage(str(label_path))
            array = sitk.GetArrayFromImage(image).astype(np.float32)
            clipped = sitk.GetImageFromArray(np.minimum(array, np.percentile(array, 99.5)))
            clipped.CopyInformation(image)
            resampled = resample_by_res(clipped, spacing, logging=False)
            mask = resample_lb_by_res(label, spacing, ref_img=resampled, logging=False)
            for kind, obj in (("image", resampled), ("label", mask)):
                arr = sitk.GetArrayFromImage(obj)
                center = (arr.shape[1] // 2, arr.shape[2] // 2)
                cropped = image_crop(arr.transpose(1, 2, 0), (crop, crop), center, padval=float(arr.min()) if kind == "image" else 0).transpose(2, 0, 1)
                result = copy_spacing_ori(resampled, sitk.GetImageFromArray(cropped))
                sitk.WriteImage(result, str(output / f"{kind}_{pid}.nii.gz"))
            writer.writerow([pid, pid])
