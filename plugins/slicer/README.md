# MemSeg Agent for 3D Slicer

We bring our few-shot and human-reviewed working-memory workflow into 3D Slicer.
Load a volume, provide a labeled slice, segment new slices, and correct them with
Slicer's Segment Editor. Explicitly saved corrections become examples for later
queries. Our plugin uses frozen SAM2 and DINOv3 in a separate Python backend;
Slicer itself does not need PyTorch, Transformers, or model weights installed.

**This is a Slicer volume workflow with slice-wise 2D inference.** We preserve the
source volume grid when writing results, but we do not introduce a native 3D
backbone or guarantee continuity between predicted slices. The plugin allocates
no static memory and performs no fine-tuning.

## Setup

We tested the extension with **3D Slicer 5.12.4 on Linux**. Other Slicer versions
and Windows/macOS have not been validated in this release. Download Slicer from
its [official website](https://download.slicer.org/).

### 1. Start a dedicated backend

Use an ordinary Python environment outside Slicer. First install PyTorch and
torchvision for your GPU, then install this repository and SAM2:

```bash
git clone https://github.com/b0wenchen/MemSeg-Agent.git
cd MemSeg-Agent
python -m pip install -e '.[workbench]'
python -m pip install 'git+https://github.com/facebookresearch/sam2.git'
memseg-workbench \
  --sam2-checkpoint /path/to/sam2.1_hiera_tiny.pt \
  --target "Left ventricle" --device cuda --port 8766
```

Obtain the matching SAM2.1 Tiny weights from the
[official SAM2 repository](https://github.com/facebookresearch/sam2).
The default DINOv3 retrieval model is
[`facebook/dinov3-vitb16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m);
accept its access terms and authenticate if required. First startup may download
model files. With weights cached, `HF_HUB_OFFLINE=1` supports offline startup.

The default Slicer endpoint is `http://127.0.0.1:8766`. We recommend a separate
backend process from the browser workbench because memory banks are shared by
every client connected to the same process. For a remote GPU server:

```bash
ssh -N -L 8766:127.0.0.1:8766 your-user@your-gpu-server
```

The client accepts loopback HTTP addresses only and disables HTTP proxies and
redirects. Use an SSH tunnel for remote inference; the server does not expose a
public port. No images are uploaded to Hugging Face or GitHub.

### 2. Load the Slicer module

In Slicer, open **Edit → Application Settings → Modules → Additional module
paths** and add the directory containing `MemSegAgent.py`:

```text
/path/to/MemSeg-Agent/plugins/slicer/MemSegAgent
```

Restart Slicer. Select **Segmentation → MemSeg Agent** from the module selector.
This source installation does not require compilation or pip installs inside
Slicer. It is not yet listed in the Slicer Extensions Manager. Our source ZIP
must be extracted and loaded with Additional module paths; it is not a compiled
Extensions Manager package. We also provide standard extension/module
`CMakeLists.txt` files for maintainers with a matching Slicer build environment.

### 3. Connect and prepare a target

1. In **Backend and task**, connect to `http://127.0.0.1:8766` and check the two
   memory counts. The task name labels the session; it is not a text prompt.
2. Load a scalar volume through Slicer, for example NIfTI or NRRD. DICOM is
   handled by Slicer's normal DICOM import workflow. Select the source volume.
3. Select an existing segmentation and target segment, or choose **Create a new
   target segmentation**. Import labelmap annotations through Slicer's
   Segmentations module if needed. Each session represents one binary target.
4. Choose a native **K, J, or I** plane and a zero-based slice index. Click
   **Show selected plane in Red view**. These are voxel-grid axes, not fixed
   anatomical axial/coronal/sagittal labels for every acquisition.
5. Check the intensity minimum/maximum. We initialize them from the first
   volume and keep them when changing volumes. Use **Set intensity window from
   this volume** or enter a consistent window explicitly when appropriate.

We normalize each requested slice through that window to 8-bit grayscale and
send it to the backend. The same window is used for all slices in a range; there
is no independent per-slice auto-normalization. Input slices must have at most
4,194,304 pixels. The original Slicer volume is not modified.

## Few-shot → prediction → correction → working memory

1. **Provide a support slice.** Draw the selected target using Segment Editor's
   Paint/Draw tools, or select a segment imported from an annotation. Review the
   selected native slice and choose **Add reviewed slice to few-shot memory**.
   Confirm the displayed volume, target, and index. A support slice must contain
   foreground; only this slice is stored, not the entire labelmap volume.
2. **Segment.** Select another slice or volume of the same task. Choose
   **Segment selected slice**, or set an inclusive start/end range and choose
   **Segment inclusive slice range**. We retrieve few-shot and working examples
   separately for each slice. The panel lists retrievals for the last slice.
3. **Correct.** Use the embedded Segment Editor to edit the result. Paint,
   Erase, Draw, and Undo remain available. Click **Show selected plane in Red
   view** before reviewing the slice you intend to save.
4. **Remember.** Choose **Save reviewed slice to working memory** and explicitly
   confirm. We store its normalized image and corrected mask. An intentionally
   reviewed all-background mask is allowed here. Unannotated slices are never
   automatically interpreted as negative examples.
5. **Continue.** Subsequent predictions can retrieve the confirmed correction.
   Nothing is automatically added to memory after inference. The two banks hold
   up to 32 cases in total, using `--top-k` examples per bank (default 2).

If you scroll Red to another slice or rotate it to an oblique plane independently
of the plugin controls, saving is blocked until you show and review the selected
native plane again. This prevents a reviewed viewport from being paired with the
wrong slice index. Use consistent orientation, plane family, and normalization
for your examples and queries; the plugin does not anatomically register cases.

## Geometry, edits, and cancellation

We read arrays in Slicer's **KJI** order and read/write each segment using the
source volume as the reference grid. This retains its origin, spacing, axis
directions, and extent. Internal cropped segmentation arrays are never treated
as full-volume arrays. Writes affect only the selected segment and requested
slices. Existing foreground in the requested range triggers an overwrite
confirmation, and a Segment Editor undo state is saved before applying results.
The editor uses `OverwriteNone` so manual corrections can overlap other segments
without erasing them.

Inference runs in a background Python worker; Qt and MRML access stays on the
main thread. Predictions are staged and applied together only after the entire
requested range succeeds. Cancellation waits for the current HTTP request and
discards the staged range. If the source volume, geometry, or selected segment's
mask changes during inference, we discard the result to preserve those edits.

Harden parent transforms before using the plugin. Resample sheared voxel grids
to an orthogonal grid before native-plane editing. Oblique acquisitions with
orthogonal voxel axes and nonuniform voxel spacing are supported. For sequences,
select a single scalar frame; automatic 4D traversal is not implemented.

## Save results and memories

- Use Slicer's **Save** dialog to save the segmentation, for example as
  `.seg.nrrd`, or use its standard labelmap export. This retains spatial context.
- Use **Reviewed memories and local archives** to remove individual examples or
  export/import the memory ZIP. These archives contain normalized 2D image/mask
  pairs and task labels, not the original 3D source geometry or pretrained weights.
- Import into an empty backend. The backend re-encodes all reviewed pairs with
  the current models; the archive format is shared with the browser workbench.
- Closing/clearing the Slicer scene does not clear the backend banks. **Start new
  memory task** explicitly clears them while preserving Slicer segmentations.
  Stopping the backend loses unexported memories. Scene files do not save these
  remote banks or reconnect to the backend automatically.

Archives contain image data and are not encrypted. Keep images, annotations,
saved scenes, and memory ZIPs out of source releases. Use separate processes for
independent users/tasks; this backend is a single-user local application.

## Validation

Our checks cover real Slicer module loading, Segment Editor activation, K/J/I
array round-trips, oblique anisotropic geometry, preservation of other segments,
native-plane selection, transform rejection, and scene reset. We also exercised
the Slicer UI against real SAM2.1 Tiny and DINOv3 for support upload, prediction,
reviewed correction, working-memory retrieval, range inference, cancellation,
concurrent-edit protection, and archive round-trips using synthetic test data.
These are functional checks, not a segmentation-quality or clinical benchmark.

Run the lightweight tests from the repository:

```bash
python -m pip install -e '.[workbench,dev]'
python -m pytest tests/test_interactive.py tests/test_slicer_adapter.py -q
```

Run the geometry/UI smoke test inside Slicer (no backend required):

```bash
/path/to/Slicer --no-splash --ignore-slicerrc \
  --additional-module-paths /path/to/MemSeg-Agent/plugins/slicer/MemSegAgent \
  --python-script /path/to/MemSeg-Agent/plugins/slicer/tests/slicer_smoke.py
```

On a headless Linux host, prefix the command with `xvfb-run -a`. For the full
backend loop, set `MEMSEG_TEST_BACKEND=http://127.0.0.1:18766` and start a **fresh,
dedicated test backend** on that port. The test adds synthetic cases and resets
that backend during archive testing; never point it at an active user session.

We reuse the original observation-mask encoding and `cross_domain` decoder.
Our interactive banks retain individual reviewed cases; they do not implement
the paper's Dice-dependent EMA merge. No claim of reproducing the full paper's
evaluation protocol or improving every subsequent case is implied.

## Preparing a Hugging Face release

From the source repository:

```bash
python scripts/package_slicer.py --output-dir outputs/slicer-release
```

This produces `MemSeg-Agent-Slicer/`, a matching source ZIP, SHA256 checksums, and
a manifest with file hashes and source revision. It includes the Slicer module,
backend source, usage guides, tests, license, and a Hugging Face README/card.
Only allowlisted source files are packaged. Checkpoints, patient data, caches,
and local memory banks are excluded. The command performs no network upload.

When ready, create your intended Hugging Face repository and upload the generated
directory. The source bundle is intended for local installation, not a hosted
Space or Inference Endpoint. The [release card](../../release/huggingface/README.md)
documents model provenance and current limitations. Choose the final repository
namespace at publication time; no Hugging Face repository is provisioned here.

We thank NVIDIA for supporting this research. Our source is under the
[MIT license](../../LICENSE); SAM2, DINOv3, and Slicer retain their own licenses.
See [our paper](https://arxiv.org/abs/2603.05873) for the underlying method.

Implementation references: [Slicer segmentation arrays](https://slicer.readthedocs.io/en/latest/developer_guide/script_repository/segmentations.html),
[Slicer extension structure](https://slicer.readthedocs.io/en/latest/developer_guide/extensions.html),
and [Hugging Face cards](https://huggingface.co/docs/hub/model-cards).
