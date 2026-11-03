"""Bounded uint8 slice transport for desktop clients without PNG dependencies."""

import base64

import numpy as np

from .session import MAX_PIXELS


def pack_array(array):
    array = np.ascontiguousarray(array, dtype=np.uint8)
    return {
        "dtype": "uint8",
        "shape": list(array.shape),
        "data": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def unpack_array(payload, *, mask=False):
    if not isinstance(payload, dict) or payload.get("dtype") != "uint8":
        raise ValueError("Expected a uint8 slice")
    shape = payload.get("shape")
    if not isinstance(shape, list) or len(shape) not in (2, 3):
        raise ValueError("Expected an H×W or H×W×3 slice")
    if any(type(size) is not int or size < 1 for size in shape):
        raise ValueError("Invalid slice dimensions")
    if min(shape[:2]) < 2 or shape[0] * shape[1] > MAX_PIXELS:
        raise ValueError("Slice exceeds pixel limits")
    if (mask and len(shape) != 2) or (len(shape) == 3 and shape[2] != 3):
        raise ValueError("Use a grayscale/RGB image and a single-channel mask")
    size = int(np.prod(shape))
    encoded = payload.get("data")
    if not isinstance(encoded, str) or len(encoded) != 4 * ((size + 2) // 3):
        raise ValueError("Slice payload length does not match its dimensions")
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) != size:
        raise ValueError("Invalid slice byte count")
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()
