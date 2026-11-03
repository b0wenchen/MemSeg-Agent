"""3D Slicer adapter for few-shot segmentation and reviewed working memory."""

import queue
import threading
from pathlib import Path

import ctk
import numpy as np
import qt
import slicer
import vtk
from MemSegAgentLib.client import MAX_ARCHIVE_BYTES, MemSegClient
from MemSegAgentLib.volume import normalize_slice, segment_range, slice_at
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleWidget,
)
from slicer.util import VTKObservationMixin


class MemSegAgent(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "MemSeg Agent"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = ["Segmentations", "SegmentEditor"]
        self.parent.contributors = ["MemSeg-Agent authors"]
        self.parent.helpText = (
            "Provide labeled slices, segment a volume, and explicitly save reviewed corrections. "
            "A separate local MemSeg backend runs the frozen models. "
            '<a href="https://github.com/b0wenchen/MemSeg-Agent/tree/main/plugins/slicer">Guide</a>.'
        )
        self.parent.acknowledgementText = "We thank NVIDIA for supporting this research."


class MemSegAgentLogic(ScriptedLoadableModuleLogic):
    @staticmethod
    def validate_volume(volume):
        if volume is None or volume.GetImageData() is None:
            raise ValueError("Select a scalar volume first")
        if volume.GetParentTransformNode() is not None:
            raise ValueError("Harden the source volume transform before using MemSeg")
        array = slicer.util.arrayFromVolume(volume)
        if array.ndim != 3:
            raise ValueError("Use a scalar volume; select one frame of a sequence first")
        return array

    @classmethod
    def mask_array(cls, volume, segmentation, segment_id):
        array = cls.validate_volume(volume)
        if (
            segmentation is None
            or not segment_id
            or segmentation.GetSegmentation().GetSegment(segment_id) is None
        ):
            raise ValueError("Select or create a target segment in Segment Editor")
        if segmentation.GetParentTransformNode() is not None:
            raise ValueError("Harden the segmentation transform before using MemSeg")
        # Always request the source volume grid: segment internal labelmaps may
        # be cropped, differently oriented, or shared with other segments.
        mask = slicer.util.arrayFromSegmentBinaryLabelmap(segmentation, segment_id, volume)
        if mask.shape != array.shape:
            raise ValueError("Unable to align the segmentation to the source volume")
        return (mask > 0).astype(np.uint8)

    @classmethod
    def apply_mask(cls, volume, segmentation, segment_id, mask):
        source = cls.validate_volume(volume)
        if segmentation.GetParentTransformNode() is not None:
            raise ValueError("Harden the segmentation transform first")
        if mask.shape != source.shape or not np.isin(mask, [0, 1]).all():
            raise ValueError("Result does not match the source volume geometry")
        if segmentation.GetSegmentation().GetSegment(segment_id) is None:
            raise ValueError("Target segment was removed")
        slicer.util.updateSegmentBinaryLabelmapFromArray(mask, segmentation, segment_id, volume)

    @staticmethod
    def geometry(volume):
        matrix = vtk.vtkMatrix4x4()
        volume.GetIJKToRASMatrix(matrix)
        return tuple(matrix.GetElement(i, j) for i in range(4) for j in range(4)) + tuple(
            volume.GetImageData().GetExtent()
        )

    @classmethod
    def verify_review_plane(cls, volume, axis, index):
        """Do not label a different slice after the user scrolls/reformats Red."""
        cls.validate_volume(volume)
        view = slicer.app.layoutManager().sliceWidget("Red")
        if view is None:
            raise ValueError("Show the selected native plane in Red before reviewing it")
        plane = view.mrmlSliceNode().GetSliceToRAS()
        inverse = vtk.vtkMatrix4x4()
        volume.GetRASToIJKMatrix(inverse)
        center = inverse.MultiplyPoint([plane.GetElement(i, 3) for i in range(3)] + [1])
        ijk_axis = 2 - axis
        extent_start = volume.GetImageData().GetExtent()[2 * ijk_axis]
        normal = np.array([plane.GetElement(i, 2) for i in range(3)])
        grid_normal = np.array([inverse.GetElement(ijk_axis, i) for i in range(3)])
        cosine = abs(float(np.dot(normal, grid_normal))) / (
            np.linalg.norm(normal) * np.linalg.norm(grid_normal)
        )
        if abs(center[ijk_axis] - extent_start - index) > 0.05 or cosine < 0.99999:
            raise ValueError(
                "Red view is on a different or oblique plane. Click 'Show selected plane in Red view', review that slice, then save."
            )

    @classmethod
    def show_slice(cls, volume, axis, index):
        array = cls.validate_volume(volume)
        slice_at(array, axis, index)  # validates index before changing a view
        layout = slicer.app.layoutManager()
        if layout.sliceWidget("Red") is None:
            layout.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        slicer.util.setSliceViewerLayers(background=volume)
        widget = layout.sliceWidget("Red")
        widget.sliceLogic().FitSliceToAll()
        matrix = vtk.vtkMatrix4x4()
        volume.GetIJKToRASMatrix(matrix)
        ijk_to_ras = np.array([[matrix.GetElement(i, j) for j in range(4)] for i in range(4)])
        extent = volume.GetImageData().GetExtent()
        center = np.array(
            [
                (extent[0] + extent[1]) / 2,
                (extent[2] + extent[3]) / 2,
                (extent[4] + extent[5]) / 2,
                1,
            ]
        )
        ijk_axis = 2 - axis
        center[ijk_axis] = extent[2 * ijk_axis] + index
        row_axis, col_axis = [(1, 0), (2, 0), (2, 1)][axis]
        x = ijk_to_ras[:3, col_axis]
        y = ijk_to_ras[:3, row_axis]
        x = x / np.linalg.norm(x)
        y = y / np.linalg.norm(y)
        # A sheared grid cannot be represented by an orthogonal editing plane.
        if abs(float(np.dot(x, y))) > 1e-5:
            raise ValueError("Resample sheared voxel grids to an orthogonal grid first")
        plane = np.eye(4)
        plane[:3, 0], plane[:3, 1], plane[:3, 2] = x, y, np.cross(x, y)
        plane[:3, 3] = (ijk_to_ras @ center)[:3]
        slice_node = widget.mrmlSliceNode()
        for i in range(4):
            for j in range(4):
                slice_node.GetSliceToRAS().SetElement(i, j, float(plane[i, j]))
        slice_node.UpdateMatrices()


class MemSegAgentWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self._busy = False
        self._closed = False
        self._windowInitialized = False
        self._cancel = threading.Event()
        self._events = queue.Queue()
        self._client = None
        self._state = None

    def setup(self):
        super().setup()
        self.logic = MemSegAgentLogic()
        self.status = qt.QLabel("Start the local MemSeg backend, then connect.")
        self.status.wordWrap = True
        self.layout.addWidget(self.status)

        self.connection = ctk.ctkCollapsibleButton()
        self.connection.text = "1. Backend and task"
        self.layout.addWidget(self.connection)
        form = qt.QFormLayout(self.connection)
        self.endpoint = qt.QLineEdit("http://127.0.0.1:8766")
        self.target = qt.QLineEdit("Foreground")
        self.connectButton = qt.QPushButton("Connect / refresh memories")
        self.resetButton = qt.QPushButton("Start new memory task")
        self.counts = qt.QLabel("No backend connected")
        form.addRow("Local backend:", self.endpoint)
        form.addRow(self.connectButton)
        form.addRow("Task name:", self.target)
        form.addRow(self.resetButton)
        form.addRow(self.counts)

        self.inputs = ctk.ctkCollapsibleButton()
        self.inputs.text = "2. Volume, target segment, and native plane"
        self.layout.addWidget(self.inputs)
        form = qt.QFormLayout(self.inputs)
        self.volumeSelector = slicer.qMRMLNodeComboBox()
        self.volumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.volumeSelector.addEnabled = False
        self.volumeSelector.removeEnabled = False
        self.volumeSelector.noneEnabled = True
        self.volumeSelector.setMRMLScene(slicer.mrmlScene)
        self.segmentationSelector = slicer.qMRMLNodeComboBox()
        self.segmentationSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.segmentationSelector.addEnabled = False
        self.segmentationSelector.removeEnabled = False
        self.segmentationSelector.noneEnabled = True
        self.segmentationSelector.setMRMLScene(slicer.mrmlScene)
        self.newSegment = qt.QPushButton("Create a new target segmentation")
        self.axis = qt.QComboBox()
        self.axis.addItems(["Native K planes", "Native J planes", "Native I planes"])
        self.index = qt.QSpinBox()
        self.index.setRange(0, 0)
        self.goButton = qt.QPushButton("Show selected plane in Red view")
        self.low = qt.QDoubleSpinBox()
        self.high = qt.QDoubleSpinBox()
        for control in (self.low, self.high):
            control.setRange(-1e12, 1e12)
            control.setDecimals(4)
        self.high.value = 255
        self.windowButton = qt.QPushButton("Set intensity window from this volume")
        form.addRow("Source volume:", self.volumeSelector)
        form.addRow("Segmentation:", self.segmentationSelector)
        form.addRow(self.newSegment)
        form.addRow("Plane family:", self.axis)
        form.addRow("Slice index (zero-based):", self.index)
        form.addRow(self.goButton)
        form.addRow("Intensity minimum:", self.low)
        form.addRow("Intensity maximum:", self.high)
        form.addRow(self.windowButton)
        hint = qt.QLabel(
            "Inference uses the native plane and index above, not an arbitrary oblique viewport. One intensity window is shared across all slices."
        )
        hint.wordWrap = True
        form.addRow(hint)

        self.editorPanel = ctk.ctkCollapsibleButton()
        self.editorPanel.text = "3. Draw / correct with Segment Editor"
        self.layout.addWidget(self.editorPanel)
        self.editor = slicer.qMRMLSegmentEditorWidget()
        self.editor.setMRMLScene(slicer.mrmlScene)
        self.editor.setSegmentationNodeSelectorVisible(False)
        self.editor.setSourceVolumeNodeSelectorVisible(False)
        self.editor.setMaximumNumberOfUndoStates(10)
        self.editor.setUndoEnabled(True)
        qt.QVBoxLayout(self.editorPanel).addWidget(self.editor)
        self._initialize_editor_node()

        self.actions = ctk.ctkCollapsibleButton()
        self.actions.text = "4. Segment and remember reviewed slices"
        self.layout.addWidget(self.actions)
        form = qt.QFormLayout(self.actions)
        self.supportButton = qt.QPushButton("Add reviewed slice to few-shot memory")
        self.predictButton = qt.QPushButton("Segment selected slice")
        self.startIndex = qt.QSpinBox()
        self.endIndex = qt.QSpinBox()
        self.rangeButton = qt.QPushButton("Segment inclusive slice range")
        self.correctButton = qt.QPushButton("Save reviewed slice to working memory")
        self.retrieval = qt.QLabel("No prediction yet.")
        self.retrieval.wordWrap = True
        form.addRow(self.supportButton)
        form.addRow(self.predictButton)
        form.addRow("Range start:", self.startIndex)
        form.addRow("Range end:", self.endIndex)
        form.addRow(self.rangeButton)
        form.addRow(self.correctButton)
        form.addRow(self.retrieval)

        self.memoriesPanel = ctk.ctkCollapsibleButton()
        self.memoriesPanel.text = "Reviewed memories and local archives"
        self.memoriesPanel.collapsed = True
        self.layout.addWidget(self.memoriesPanel)
        form = qt.QFormLayout(self.memoriesPanel)
        self.memoryList = qt.QListWidget()
        self.removeButton = qt.QPushButton("Remove selected memory")
        self.exportButton = qt.QPushButton("Export reviewed memories (.zip)")
        self.importButton = qt.QPushButton("Import into empty backend (.zip)")
        form.addRow(self.memoryList)
        for button in (self.removeButton, self.exportButton, self.importButton):
            form.addRow(button)
        self.progress = qt.QProgressBar()
        self.progress.visible = False
        self.cancelButton = qt.QPushButton("Cancel inference (discard staged predictions)")
        self.cancelButton.enabled = False
        self.layout.addWidget(self.progress)
        self.layout.addWidget(self.cancelButton)
        note = qt.QLabel(
            "Research use. Review every result. Predictions never enter memory automatically. Save segmentation nodes with Slicer's Save dialog; export memory separately."
        )
        note.wordWrap = True
        self.layout.addWidget(note)
        self.layout.addStretch(1)

        self.connectButton.connect("clicked()", self.onConnect)
        self.resetButton.connect("clicked()", self.onReset)
        self.newSegment.connect("clicked()", self.onNewSegment)
        self.goButton.connect("clicked()", lambda: self._ui(self._show_slice))
        self.windowButton.connect("clicked()", lambda: self._ui(self._set_window))
        self.volumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onVolumeChanged)
        self.segmentationSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.onSegmentationChanged
        )
        self.axis.connect("currentIndexChanged(int)", self.onPlaneChanged)
        self.index.connect("valueChanged(int)", lambda unused: self._ui(self._show_slice))
        self.predictButton.connect("clicked()", lambda: self.onPredict(False))
        self.rangeButton.connect("clicked()", lambda: self.onPredict(True))
        self.supportButton.connect("clicked()", lambda: self.onRemember("fewshot"))
        self.correctButton.connect("clicked()", lambda: self.onRemember("working"))
        self.cancelButton.connect("clicked()", self.onCancel)
        self.removeButton.connect("clicked()", self.onRemove)
        self.exportButton.connect("clicked()", self.onExport)
        self.importButton.connect("clicked()", self.onImport)
        self.timer = qt.QTimer()
        self.timer.setInterval(100)
        self.timer.connect("timeout()", self._poll)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        self.onVolumeChanged()

    def _initialize_editor_node(self):
        self.editorNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentEditorNode", "MemSegEditor"
        )
        self.editorNode.SetOverwriteMode(slicer.vtkMRMLSegmentEditorNode.OverwriteNone)
        self.editor.setMRMLSegmentEditorNode(self.editorNode)

    def _ui(self, function):
        try:
            function()
        except Exception as exc:  # noqa: BLE001 -- report errors at the Qt callback boundary
            self.status.text = str(exc)
            slicer.util.errorDisplay(str(exc))

    def _require_client(self):
        if self._client is None or self._client.endpoint != self.endpoint.text.rstrip("/"):
            raise ValueError("Connect to the selected backend first")
        return self._client

    def _selection(self):
        volume = self.volumeSelector.currentNode()
        segmentation = self.segmentationSelector.currentNode()
        segment_id = self.editor.currentSegmentID()
        array = self.logic.validate_volume(volume)
        mask = self.logic.mask_array(volume, segmentation, segment_id)
        return volume, segmentation, segment_id, array, mask

    def _show_slice(self):
        volume = self.volumeSelector.currentNode()
        if volume is not None:
            self.logic.show_slice(volume, self.axis.currentIndex, self.index.value)

    def _set_window(self):
        volume = self.volumeSelector.currentNode()
        array = self.logic.validate_volume(volume)
        if not np.isfinite(array).all():
            raise ValueError("Remove NaN/Inf values from the source volume first")
        low, high = float(array.min()), float(array.max())
        if high <= low:
            raise ValueError("The volume has no intensity variation")
        self.low.value, self.high.value = low, high
        self._windowInitialized = True

    def onVolumeChanged(self, unused=None):
        volume = self.volumeSelector.currentNode()
        if self.segmentationSelector.currentNode() is not None:
            self.editor.setSourceVolumeNode(volume)
        if volume is not None:
            if not self._windowInitialized:
                self._ui(self._set_window)
            self.onPlaneChanged()

    def onSegmentationChanged(self, unused=None):
        self.editor.setSegmentationNode(self.segmentationSelector.currentNode())
        if self.segmentationSelector.currentNode() is not None:
            self.editor.setSourceVolumeNode(self.volumeSelector.currentNode())

    def onPlaneChanged(self, unused=None):
        volume = self.volumeSelector.currentNode()
        if volume is None:
            return
        try:
            array = self.logic.validate_volume(volume)
            maximum = array.shape[self.axis.currentIndex] - 1
            for control in (self.index, self.startIndex, self.endIndex):
                control.setRange(0, maximum)
            self.endIndex.value = maximum
            self._show_slice()
        except ValueError as exc:
            self.status.text = str(exc)

    def onNewSegment(self):
        def create():
            volume = self.volumeSelector.currentNode()
            self.logic.validate_volume(volume)
            segmentation = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSegmentationNode", "MemSeg result"
            )
            segmentation.CreateDefaultDisplayNodes()
            segmentation.SetReferenceImageGeometryParameterFromVolumeNode(volume)
            segment_id = segmentation.GetSegmentation().AddEmptySegment(
                "", self.target.text.strip() or "Foreground", [0.2, 0.85, 0.65]
            )
            self.segmentationSelector.setCurrentNode(segmentation)
            self.editor.setCurrentSegmentID(segment_id)
            self._show_slice()

        self._ui(create)

    def _set_busy(self, value, cancellable=False):
        self._busy = value
        for widget in (
            self.connection,
            self.inputs,
            self.editorPanel,
            self.actions,
            self.memoriesPanel,
        ):
            widget.enabled = not value
        self.progress.visible = value
        self.cancelButton.enabled = value and cancellable

    def _start(self, work, done, text, *, cancellable=False):
        if self._busy:
            return
        self._cancel = threading.Event()
        cancel, events = self._cancel, self._events
        self._done = done
        self._set_busy(True, cancellable)
        self.status.text = text
        self.progress.setRange(0, 0)

        def run():
            try:
                value = work(cancel, lambda step, total: events.put(("progress", (step, total))))
                events.put(("done", value))
            except Exception as exc:  # noqa: BLE001 -- always release the UI after a worker failure
                events.put(("error", str(exc)))

        threading.Thread(target=run, daemon=True).start()
        self.timer.start()

    def _poll(self):
        try:
            while True:
                kind, value = self._events.get_nowait()
                if kind == "progress":
                    self.progress.setRange(0, value[1])
                    self.progress.value = value[0]
                else:
                    self.timer.stop()
                    self._set_busy(False)
                    if kind == "error":
                        self.status.text = value
                    elif self._cancel.is_set():
                        self.status.text = "Cancelled; staged predictions discarded."
                    else:
                        self._ui(lambda result=value: self._done(result))
        except queue.Empty:
            pass

    def _update_state(self, state):
        self._state = state
        self.target.text = state["target"]
        few = sum(item["bank"] == "fewshot" for item in state["memories"])
        work = len(state["memories"]) - few
        self.counts.text = f"Few-shot: {few} | Working: {work} | Capacity: {state['capacity']}"
        self.memoryList.clear()
        for item in state["memories"]:
            row = qt.QListWidgetItem(f"{item['bank']} · {item['name']}")
            row.setData(qt.Qt.UserRole, item["id"])
            self.memoryList.addItem(row)
        self.status.text = "Backend ready. Only explicitly reviewed slices are stored."

    def onConnect(self):
        def connect():
            client = MemSegClient(self.endpoint.text)

            def done(state):
                self._client = client
                self._update_state(state)

            self._start(lambda cancel, progress: client.connect(), done, "Connecting…")

        self._ui(connect)

    def onReset(self):
        def reset():
            client = self._require_client()
            if not slicer.util.confirmYesNoDisplay(
                "Clear both backend memory banks? Export them first if needed. Slicer segmentation nodes will be kept."
            ):
                return
            target = self.target.text
            self._start(
                lambda cancel, progress: client.request(
                    "reset", {"confirmed": True, "target": target}
                ),
                self._update_state,
                "Starting new memory task…",
            )

        self._ui(reset)

    def onRemember(self, bank):
        def remember():
            client = self._require_client()
            volume, segmentation, segment_id, array, mask = self._selection()
            axis, index = self.axis.currentIndex, self.index.value
            self.logic.verify_review_plane(volume, axis, index)
            image = normalize_slice(slice_at(array, axis, index), self.low.value, self.high.value)
            reviewed = slice_at(mask, axis, index).copy()
            segment_name = segmentation.GetSegmentation().GetSegment(segment_id).GetName()
            plane = "KJI"[axis]
            if bank == "fewshot" and not reviewed.any():
                raise ValueError(
                    "Draw foreground on the selected slice before adding a support example"
                )
            if not slicer.util.confirmYesNoDisplay(
                f"Have you reviewed '{segment_name}' on {volume.GetName()}, {plane}={index}? Save exactly this slice to {bank} memory for task '{self._state['target']}'?"
            ):
                return
            name = f"{volume.GetName()} {plane}={index} [{self.low.value:g},{self.high.value:g}]"

            def work(cancel, progress):
                if bank == "fewshot":
                    return client.add_support(image, reviewed, name)
                return client.add_correction(image, reviewed, name, confirmed=True)

            self._start(work, self._update_state, "Encoding the reviewed slice…")

        self._ui(remember)

    def onPredict(self, use_range):
        def predict():
            client = self._require_client()
            volume, segmentation, segment_id, array, original = self._selection()
            axis = self.axis.currentIndex
            first, last = (
                (self.startIndex.value, self.endIndex.value)
                if use_range
                else (self.index.value, self.index.value)
            )
            if last < first:
                raise ValueError("Range end must be at least the range start")
            if any(
                slice_at(original, axis, index).any() for index in range(first, last + 1)
            ) and not slicer.util.confirmYesNoDisplay(
                "Replace the selected segment's existing mask in this slice range? Other slices and segments are preserved; Segment Editor Undo is available."
            ):
                return
            snapshot = array.copy()
            low, high = self.low.value, self.high.value
            geometry = self.logic.geometry(volume)
            image_mtime = volume.GetImageData().GetMTime()

            def work(cancel, progress):
                return segment_range(
                    client,
                    snapshot,
                    original,
                    axis,
                    range(first, last + 1),
                    low,
                    high,
                    cancelled=cancel.is_set,
                    progress=progress,
                )

            def done(result):
                if (
                    volume.GetScene() is None
                    or segmentation.GetScene() is None
                    or volume.GetImageData() is None
                    or image_mtime != volume.GetImageData().GetMTime()
                    or geometry != self.logic.geometry(volume)
                    or not np.array_equal(
                        original, self.logic.mask_array(volume, segmentation, segment_id)
                    )
                ):
                    raise ValueError(
                        "Source or segmentation changed during inference; result discarded to preserve edits"
                    )
                self.editor.saveStateForUndo()
                self.logic.apply_mask(volume, segmentation, segment_id, result[0])
                self.retrieval.text = "Last slice used: " + "; ".join(
                    f"{item['bank']}: {item['name']} ({item['similarity']:.3f})"
                    for item in result[1]
                )
                self.status.text = f"Applied {last - first + 1} predicted slice(s). Review and correct them; no memory was written."

            self._start(work, done, "Running slice inference…", cancellable=True)

        self._ui(predict)

    def onCancel(self):
        self._cancel.set()
        self.status.text = (
            "Cancelling after the current request; staged predictions will be discarded."
        )

    def onRemove(self):
        def remove():
            client = self._require_client()
            row = self.memoryList.currentItem()
            if row is None:
                raise ValueError("Select a memory to remove")
            if not slicer.util.confirmYesNoDisplay(
                "Remove this reviewed case from backend memory?"
            ):
                return
            memory_id = row.data(qt.Qt.UserRole)
            self._start(
                lambda cancel, progress: client.request("remove", {"id": memory_id}),
                self._update_state,
                "Removing memory…",
            )

        self._ui(remove)

    def onExport(self):
        def export():
            client = self._require_client()
            path = qt.QFileDialog.getSaveFileName(
                slicer.util.mainWindow(),
                "Export images and masks to a secure location",
                "memseg-reviewed-memories.zip",
                "ZIP archive (*.zip)",
            )
            if not path:
                return

            def work(cancel, progress):
                Path(path).write_bytes(client.export_archive())

            self._start(
                work,
                lambda result: setattr(
                    self.status,
                    "text",
                    "Reviewed images and masks exported. This file is not encrypted.",
                ),
                "Exporting reviewed memories…",
            )

        self._ui(export)

    def onImport(self):
        def import_archive():
            client = self._require_client()
            path = qt.QFileDialog.getOpenFileName(
                slicer.util.mainWindow(), "Import into an empty backend", "", "ZIP archive (*.zip)"
            )
            if not path:
                return
            if Path(path).stat().st_size > MAX_ARCHIVE_BYTES:
                raise ValueError("Archive exceeds 192 MiB")
            self._start(
                lambda cancel, progress: client.import_archive(Path(path).read_bytes()),
                self._update_state,
                "Importing and re-encoding memories…",
            )

        self._ui(import_archive)

    def onSceneStartClose(self, caller=None, event=None):
        self._cancel.set()
        self.editor.setActiveEffectByName("")
        self.editor.setMRMLSegmentEditorNode(None)

    def onSceneEndClose(self, caller=None, event=None):
        if not self._closed:
            self._initialize_editor_node()
            self._windowInitialized = False
            self.status.text = (
                "Scene cleared. Backend memories persist; start a new memory task to clear them."
            )

    def enter(self):
        self.editor.setupViewObservations()

    def exit(self):
        self.editor.setActiveEffectByName("")
        self.editor.removeViewObservations()

    def cleanup(self):
        self._closed = True
        self._cancel.set()
        self.timer.stop()
        self.removeObservers()
        self.editor.removeViewObservations()
        self.editor.setMRMLScene(None)
