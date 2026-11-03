"""Run inside real Slicer; synthetic data only, no model downloads.

Slicer --no-splash --ignore-slicerrc --additional-module-paths <module-directory>
       --python-script <this-file>
"""

import json
import os
import time
import traceback

import numpy as np
import slicer
import vtk


def run():
    from MemSegAgent import MemSegAgentLogic
    from MemSegAgentLib.volume import put_slice

    slicer.util.selectModule("MemSegAgent")
    widget = slicer.modules.memsegagent.widgetRepresentation().self()
    assert widget.editor is not None
    assert widget.cancelButton.enabled is False
    volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Synthetic volume")
    array = np.arange(9 * 13 * 17, dtype=np.int16).reshape(9, 13, 17)
    slicer.util.updateVolumeFromArray(volume, array)
    angle = 0.37
    direction = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, -1]]
    )
    matrix = vtk.vtkMatrix4x4()
    for i in range(3):
        for j in range(3):
            matrix.SetElement(i, j, direction[i, j] * [0.8, 1.4, 2.2][j])
        matrix.SetElement(i, 3, [21.0, -43.0, 17.0][i])
    volume.SetIJKToRASMatrix(matrix)
    volume.CreateDefaultDisplayNodes()
    geometry = MemSegAgentLogic.geometry(volume)
    widget.volumeSelector.setCurrentNode(volume)
    widget.onNewSegment()
    segmentation = widget.segmentationSelector.currentNode()
    assert widget.editor.sourceVolumeNode() == volume
    segment_id = widget.editor.currentSegmentID()
    original = MemSegAgentLogic.mask_array(volume, segmentation, segment_id)
    assert original.shape == array.shape and not original.any()
    second_id = segmentation.GetSegmentation().AddEmptySegment("", "Untouched segment")
    second = np.zeros_like(array, dtype=np.uint8)
    second[2:5, 3:7, 4:9] = 1
    MemSegAgentLogic.apply_mask(volume, segmentation, second_id, second)
    for axis in range(3):
        mask = np.zeros_like(array, dtype=np.uint8)
        shape = tuple(size for i, size in enumerate(array.shape) if i != axis)
        plane = np.zeros(shape, dtype=np.uint8)
        plane[1:4, 2:6] = 1
        put_slice(mask, plane, axis, 5)
        MemSegAgentLogic.apply_mask(volume, segmentation, segment_id, mask)
        np.testing.assert_array_equal(
            MemSegAgentLogic.mask_array(volume, segmentation, segment_id), mask
        )
        np.testing.assert_array_equal(
            MemSegAgentLogic.mask_array(volume, segmentation, second_id), second
        )
        assert MemSegAgentLogic.geometry(volume) == geometry
        MemSegAgentLogic.show_slice(volume, axis, 5)
        view_matrix = slicer.app.layoutManager().sliceWidget("Red").mrmlSliceNode().GetSliceToRAS()
        ras = [view_matrix.GetElement(i, 3) for i in range(3)] + [1]
        inverse = vtk.vtkMatrix4x4()
        volume.GetRASToIJKMatrix(inverse)
        ijk = inverse.MultiplyPoint(ras)
        assert abs(ijk[2 - axis] - 5) < 1e-6
        MemSegAgentLogic.verify_review_plane(volume, axis, 5)
        try:
            MemSegAgentLogic.verify_review_plane(volume, axis, 4)
            raise AssertionError("Wrong review plane was accepted")
        except ValueError:
            pass
    transform = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
    volume.SetAndObserveTransformNodeID(transform.GetID())
    try:
        MemSegAgentLogic.validate_volume(volume)
        raise AssertionError("Unhardened transform was not rejected")
    except ValueError:
        pass
    volume.SetAndObserveTransformNodeID(None)
    widget.axis.currentIndex = 0
    widget.index.value = 5
    widget.editor.setCurrentSegmentID(segment_id)
    widget.editor.setActiveEffectByName("Paint")
    assert widget.editor.activeEffect() is not None
    widget.editor.setActiveEffectByName("")
    backend = os.environ.get("MEMSEG_TEST_BACKEND")
    if backend:
        from MemSegAgentLib.client import MemSegClient

        client = MemSegClient(backend)
        assert not client.connect()["memories"], "Use a fresh, dedicated test backend"
        errors = []
        original_confirm, original_error = slicer.util.confirmYesNoDisplay, slicer.util.errorDisplay
        slicer.util.confirmYesNoDisplay = lambda *args, **kwargs: True
        slicer.util.errorDisplay = lambda message, **kwargs: errors.append(message)

        def wait():
            deadline = time.monotonic() + 180
            while widget._busy and time.monotonic() < deadline:
                slicer.app.processEvents()
                time.sleep(0.02)
            assert not widget._busy, "UI request timed out"

        try:
            widget.endpoint.text = backend
            widget.onConnect()
            wait()
            assert widget._client is not None
            reviewed = np.zeros_like(array, dtype=np.uint8)
            reviewed[2, 3:8, 4:11] = 1
            MemSegAgentLogic.apply_mask(volume, segmentation, segment_id, reviewed)
            widget.index.value = 2
            widget.onRemember("fewshot")
            wait()
            assert len(client.connect()["memories"]) == 1
            widget.index.value = 3
            widget.onPredict(False)
            wait()
            assert "Applied 1" in widget.status.text, widget.status.text
            assert len(client.connect()["memories"]) == 1
            correction = MemSegAgentLogic.mask_array(volume, segmentation, segment_id)
            correction[3] = 0
            correction[3, 2:9, 3:10] = 1
            MemSegAgentLogic.apply_mask(volume, segmentation, segment_id, correction)
            widget.onRemember("working")
            wait()
            assert len(client.connect()["memories"]) == 2
            widget.index.value = 4
            widget.onPredict(False)
            wait()
            assert "working:" in widget.retrieval.text, widget.retrieval.text
            widget.startIndex.value, widget.endIndex.value = 5, 6
            widget.onPredict(True)
            wait()
            assert "Applied 2" in widget.status.text, widget.status.text
            before_cancel = MemSegAgentLogic.mask_array(volume, segmentation, segment_id)
            widget.onPredict(True)
            widget.onCancel()
            wait()
            np.testing.assert_array_equal(
                MemSegAgentLogic.mask_array(volume, segmentation, segment_id), before_cancel
            )
            # Simulate an edit from another module while the worker is running.
            widget.onPredict(False)
            edited = before_cancel.copy()
            edited[0, 0, 0] = 1 - edited[0, 0, 0]
            MemSegAgentLogic.apply_mask(volume, segmentation, segment_id, edited)
            wait()
            np.testing.assert_array_equal(
                MemSegAgentLogic.mask_array(volume, segmentation, segment_id), edited
            )
            assert len(errors) == 1 and "changed during inference" in errors[0], errors
            archive = client.export_archive()
            client.request("reset", {"confirmed": True, "target": "Test reset"})
            state = client.import_archive(archive)
            assert {item["bank"] for item in state["memories"]} == {"fewshot", "working"}
            widget._update_state(state)
            print(
                "SLICER LIVE BACKEND PASS: support, inference, correction, working retrieval, range, cancellation, concurrent-edit guard, archive",
                flush=True,
            )
        finally:
            slicer.util.confirmYesNoDisplay, slicer.util.errorDisplay = (
                original_confirm,
                original_error,
            )
    screenshot = os.environ.get("MEMSEG_SLICER_SCREENSHOT")
    if screenshot:
        slicer.util.mainWindow().resize(1450, 1000)
        slicer.app.processEvents()
        slicer.util.mainWindow().grab().save(screenshot)
    # Exercise actual scene lifecycle, including recreation of the editor node.
    slicer.mrmlScene.Clear(0)
    assert widget.editorNode.GetScene() is not None
    print(
        json.dumps(
            {
                "slicer": slicer.app.applicationVersion,
                "passed": [
                    "widget load",
                    "empty segment",
                    "K/J/I geometry roundtrips",
                    "anisotropic oblique volume",
                    "other segment preserved",
                    "native plane navigation",
                    "transform rejection",
                    "Segment Editor Paint",
                    "scene reset",
                ],
            }
        ),
        flush=True,
    )


try:
    run()
except Exception:  # noqa: BLE001 -- translate any smoke-test failure into a process exit code
    traceback.print_exc()
    slicer.app.exit(1)
else:
    slicer.app.exit(0)
