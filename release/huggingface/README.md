---
license: mit
language:
  - en
tags:
  - 3d-slicer
  - medical-image-segmentation
  - few-shot-learning
  - working-memory
  - sam2
  - dinov3
  - research
---

# MemSeg Agent for 3D Slicer

We provide an open-source Slicer extension and local inference backend for
segmentation guided by few-shot examples and human-reviewed working memory.
After providing a labeled slice, users segment new slices or ranges, correct
the results in Segment Editor, and explicitly save corrected cases to guide
future predictions.

**This is a source-code release, not a new pretrained checkpoint or hosted
inference service.** We run frozen SAM2 and DINOv3 with no static-memory training.
The plugin processes 3D volumes slice by slice through a 2D segmentation model.

- [Installation and workflow](https://github.com/b0wenchen/MemSeg-Agent/tree/main/plugins/slicer)
- [Source repository](https://github.com/b0wenchen/MemSeg-Agent)
- [Paper](https://arxiv.org/abs/2603.05873)
- [Project page](https://b0wenchen.github.io/MemSeg-Agent/)

## Quick start

Download/extract this source bundle. In a separate Python environment with a
matching PyTorch installation:

```bash
python -m pip install -e '.[workbench]'
python -m pip install 'git+https://github.com/facebookresearch/sam2.git'
memseg-workbench --sam2-checkpoint /path/to/sam2.1_hiera_tiny.pt \
  --target "Left ventricle" --device cuda --port 8766
```

In Slicer's **Application Settings → Modules → Additional module paths**, add
`plugins/slicer/MemSegAgent` from the extracted bundle. Restart Slicer and select
**Segmentation → MemSeg Agent**. Connect to `http://127.0.0.1:8766`. Follow the
linked guide to load a scalar volume, select a target segment, provide a support
slice, run inference, and save reviewed corrections. Use an SSH tunnel if the
backend runs on a GPU server. PyTorch is not installed inside Slicer.

## Model provenance and licenses

We use pretrained [SAM2.1 Hiera Tiny](https://github.com/facebookresearch/sam2)
for segmentation and
[`facebook/dinov3-vitb16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)
for retrieval. Obtain weights from their providers and follow their license and
access requirements. No SAM2/DINOv3 weights, training datasets, or patient cases
are redistributed in this source bundle. Our code is MIT-licensed; that license
does not relicense the external models or Slicer.

The plugin performs no weight optimization. Its adaptation state consists of
user-provided and explicitly reviewed normalized image/mask pairs and their
encoded representations. These user-specific memory banks are not part of the
release. Our paper's experimental results are not claimed as benchmark results
for this interactive plugin configuration.

## Scope and validation

We tested on Slicer 5.12.4/Linux with real SAM2.1 Tiny and DINOv3, including the
few-shot/correction loop, volume geometry, preservation of other segments,
cancellation, concurrent-edit protection, and memory archive round-trips.
Synthetic test data is used for the Slicer integration test. Automated session
tests use small model doubles. We do not report a clinical validation or a
segmentation-quality benchmark for this release.

Current limits include one binary target per memory task, scalar volumes,
native voxel planes, an explicit intensity window, at most 32 reviewed cases,
and slice-wise 2D inference without guaranteed 3D continuity. Parent transforms
must be hardened; sheared grids must be resampled for native-plane editing.
Windows/macOS and other Slicer versions have not been validated. We retain exact
reviewed cases rather than the experimental Dice-dependent EMA merge.

Results require human review. This research software is not a clinical device.
Performance depends on acquisition, anatomy, preprocessing, annotation quality,
and the suitability of the support examples. Additional memories do not
guarantee improvement. The backend is local and single-user; Hugging Face is
used to distribute source, not to receive user images. Exported memories contain
image data, are unencrypted, and must not be included in public uploads.

We thank NVIDIA for supporting this research.
