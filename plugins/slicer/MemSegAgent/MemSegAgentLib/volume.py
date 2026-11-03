"""Explicit KJI slice indexing and fixed-window normalization, without Slicer."""

import numpy as np


def slice_at(volume, axis, index):
    if volume.ndim != 3 or axis not in (0, 1, 2):
        raise ValueError("Select a scalar 3D volume and a native K/J/I plane")
    if not 0 <= index < volume.shape[axis]:
        raise ValueError("Slice index is outside the volume")
    selection = [slice(None)] * 3
    selection[axis] = index
    return volume[tuple(selection)]


def put_slice(volume, mask, axis, index):
    target = slice_at(volume, axis, index)
    if mask.shape != target.shape or not np.isin(mask, [0, 1]).all():
        raise ValueError("Prediction must be a binary mask with matching slice dimensions")
    target[:] = mask


def normalize_slice(image, low, high):
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError("Intensity maximum must be greater than minimum")
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("Use a finite scalar slice; remove NaN/Inf values first")
    # One explicit intensity window is used across slices, never per-slice min/max.
    scaled = np.clip((image.astype(np.float64) - low) / (high - low), 0, 1)
    return np.rint(scaled * 255).astype(np.uint8)


def segment_range(
    client, volume, original_mask, axis, indices, low, high, *, cancelled=None, progress=None
):
    """Stage a complete result; cancellation/failure never mutates the input mask."""
    if original_mask.shape != volume.shape:
        raise ValueError("Segmentation must use the source volume geometry")
    result = original_mask.copy()
    retrieved = []
    indices = list(indices)
    for completed, index in enumerate(indices, 1):
        if cancelled and cancelled():
            raise InterruptedError("Cancelled; no predicted voxels were applied")
        image = normalize_slice(slice_at(volume, axis, index), low, high)
        mask, retrieved = client.predict(image)
        if cancelled and cancelled():
            raise InterruptedError("Cancelled; no predicted voxels were applied")
        put_slice(result, mask, axis, index)
        if progress:
            progress(completed, len(indices))
    return result, retrieved
