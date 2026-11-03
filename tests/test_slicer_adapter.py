"""Desktop adapter tests without importing Slicer or downloading model weights."""

import sys
import threading
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plugins/slicer/MemSegAgent"))
from MemSegAgentLib.client import MemSegClient, NoRedirect, pack_array, unpack_mask
from MemSegAgentLib.volume import normalize_slice, put_slice, segment_range, slice_at


class ThresholdClient:
    def predict(self, image):
        return (image > 127).astype(np.uint8), [{"bank": "fewshot"}]


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_native_planes_keep_axis_order_and_outside_voxels(axis):
    volume = np.arange(5 * 7 * 9).reshape(5, 7, 9)
    original = np.ones_like(volume, dtype=np.uint8)
    predicted, _ = segment_range(ThresholdClient(), volume, original, axis, [2], 0, 314)
    expected = original.copy()
    # 157 maps to round(127.5) == 128 in the explicit uint8 intensity window.
    put_slice(expected, (slice_at(volume, axis, 2) >= 157).astype(np.uint8), axis, 2)
    np.testing.assert_array_equal(predicted, expected)
    assert original.all()


def test_fixed_intensity_window_and_invalid_values():
    first = np.array([[-1000.0, 0], [500.0, 1000.0]])
    second = np.array([[0.0, 10], [30.0, 40.0]])
    assert normalize_slice(first, -1000, 1000)[0, 1] == normalize_slice(second, -1000, 1000)[0, 0]
    for low, high in ((1, 1), (2, 1), (float("nan"), 10)):
        with pytest.raises(ValueError):
            normalize_slice(first, low, high)
    first[0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        normalize_slice(first, 0, 1)


def test_cancelled_range_does_not_apply_partial_result():
    original = np.zeros((3, 4, 5), dtype=np.uint8)
    volume = np.ones_like(original) * 200
    cancel = threading.Event()
    with pytest.raises(InterruptedError):
        segment_range(
            ThresholdClient(),
            volume,
            original,
            0,
            range(3),
            0,
            255,
            cancelled=cancel.is_set,
            progress=lambda step, total: cancel.set(),
        )
    assert not original.any()


def test_failed_range_and_shape_mismatch_leave_input_untouched():
    class FailedClient:
        def predict(self, image):
            raise RuntimeError("Backend unavailable")

    original = np.zeros((3, 4, 5), dtype=np.uint8)
    with pytest.raises(RuntimeError):
        segment_range(FailedClient(), original, original, 0, [1], 0, 255)
    assert not original.any()
    with pytest.raises(ValueError):
        put_slice(original, np.ones((5, 4), dtype=np.uint8), 0, 1)
    with pytest.raises(ValueError):
        slice_at(original, 0, -1)


def test_binary_wire_roundtrip_and_corruption():
    mask = np.zeros((5, 7), dtype=np.uint8)
    mask[1:3, 2:5] = 1
    payload = pack_array(mask)
    np.testing.assert_array_equal(unpack_mask(payload, mask.shape), mask)
    with pytest.raises(ValueError):
        unpack_mask(payload, (7, 5))
    payload["data"] = "bad"
    with pytest.raises(ValueError):
        unpack_mask(payload, mask.shape)
    with pytest.raises(ValueError):
        unpack_mask(pack_array(mask * 2), mask.shape)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "http://127.0.0.1.evil.test",
        "http://user@localhost:8766",
        "http://localhost:8766/path",
        "http://localhost:8766?token=secret",
    ],
)
def test_client_rejects_external_or_ambiguous_endpoints(url):
    with pytest.raises(ValueError):
        MemSegClient(url)


def test_no_redirects_and_no_unconfirmed_write():
    with pytest.raises(ValueError, match="redirects"):
        NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.com")
    with pytest.raises(ValueError, match="confirm"):
        MemSegClient().add_correction(
            np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8), "Unreviewed"
        )
