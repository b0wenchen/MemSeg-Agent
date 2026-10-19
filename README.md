# MemSeg-Agent

We present **MemSeg-Agent**, a memory-augmented framework that
shifts medical-image segmentation adaptation from backbone weights to memory
space. See [our paper](https://arxiv.org/abs/2603.05873).

> Research code only; not a clinical device or clinical decision-support tool.

## Modules

| Module | Location | Updated parameters/state |
| --- | --- | --- |
| Supervised static memory | `scripts/train_static.py` | Direct memory tokens or pseudo-image/mask logits; SAM2 is frozen |
| Few-shot memory | `memseg_agent.fewshot` | Non-parametric tokens from support pairs |
| Test-time working memory | `scripts/eval_working_memory.py` | Bounded DINOv3 retrieval buffer with gated EMA updates |
| Agentic composition | `memseg_agent.controller` | Transparent static/few-shot/working routing policy |
| Federated static memory | `scripts/fedavg_memories.py` | Only the two static-memory tensors are averaged |

Raw datasets, annotations, checkpoints, experiment outputs, local paths, and
vendored SAM2/MONAI/DINO code are intentionally excluded. Install external
dependencies and use all datasets under their respective licenses.

## Installation

```bash
git clone https://github.com/b0wenchen/MemSeg-Agent.git
cd MemSeg-Agent
# First install the PyTorch build matching your CUDA runtime.
pip install -r requirements.txt
```

Download a SAM2 checkpoint from the
[official SAM2 repository](https://github.com/facebookresearch/sam2), then
copy `configs/default.yaml` and set `sam2_checkpoint` to your local weight file.
Keep `sam2_config` as the SAM2 package config name (for example,
`configs/sam2.1/sam2.1_hiera_t.yaml`). The optional DINOv3 retrieval model is downloaded from Hugging
Face and may require access approval.

## Data Description

We evaluate static-memory segmentation on ACDC, CAMUS, and CHAOS, and use
CardiacUDA as an external benchmark for cross-domain evaluation. The counts
below describe the subsets used in our experiments, not necessarily the full
datasets available from their providers.

| Dataset and source | Modality and subset used | Patients | Images / sequences used | Segmentation targets |
| --- | --- | ---: | --- | --- |
| [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/) | Short-axis cardiac cine MRI; annotated ED/ES phases from the training cohort | 100 | 200 3D volumes (two phases per patient) | Right ventricle (RV), myocardium (Myo), left ventricle (LV) |
| [CAMUS](https://www.creatis.insa-lyon.fr/Challenge/camus/) | 2D echocardiography; apical two- and four-chamber views at ED/ES | 500 | 2,000 annotated 2D frames (two views × two phases per patient) | LV, Myo, left atrium (LA) |
| [CHAOS](https://arxiv.org/abs/2001.06535) | Abdominal T2-SPIR MRI; labeled training subset | 20 | 20 3D volumes (one per patient) | Liver, right kidney (RK), left kidney (LK), spleen |
| [CardiacUDA](https://github.com/XiaoweiXu/CardiacUDA-dataset) | External echocardiography; selected annotated apical four-chamber sequences | 8 | 8 sequences containing 428 annotated frames | LV and LA, aligned to the CAMUS label space |

ED and ES denote end-diastole and end-systole. We use only the annotated ED/ES
frames from CAMUS, whereas our CardiacUDA subset contains annotated frames
throughout the selected sequences. We store each CAMUS frame as a singleton-depth
NIfTI volume and each CardiacUDA sequence as a stack with time along the last
axis. These storage units must not be counted as additional patients.

For cross-domain evaluation, we train static memory on CAMUS and evaluate on
the selected CardiacUDA subset. We evaluate LV and LA only; we do not treat
unavailable CardiacUDA myocardium annotations as negative myocardium examples.
When evaluating working memory, we use corrections after scoring each prediction,
as described in the test-time evaluation section below.

### Labels and preprocessing

We train one binary segmentation task at a time using `target_class`. Background
is `0`; the preprocessed label conventions are:

| Dataset | Foreground label IDs | `num_classes` |
| --- | --- | ---: |
| ACDC | `1`: RV, `2`: Myo, `3`: LV | 4 |
| CAMUS | `1`: LV, `2`: Myo, `3`: LA | 4 |
| CHAOS | `1`: liver, `2`: RK, `3`: LK, `4`: spleen | 5 |
| CardiacUDA (our remapped subset) | `1`: LV, `3`: LA; other structures mapped to background | 4 |

Our experiments use preprocessed 256 × 256 slices/frames with upper-intensity
clipping at the 99.5th percentile. The loader performs per-volume min–max
normalization and repeats the grayscale image across three channels; the model
then resizes inputs to 1024 × 1024 for SAM2. We perform training and inference
slice by slice and aggregate predictions for volume/sequence-level evaluation.

The current release expects preprocessed NIfTI files; it does not yet include
the dataset-specific preprocessing scripts. Our original preprocessing includes
annotation-centered crops (a shared crop across frames for CardiacUDA), so the
preprocessing protocol must be preserved when comparing results. The paper
describes 0.5 × 0.5 mm in-plane resampling, but the archived ACDC script defaults
to 1.25 × 1.25 × 7.70 mm and the CardiacUDA export uses pixel resizing with unit
spacing. We have not yet reconciled these settings against every reported run;
the current loader does not enforce physical-spacing resampling.

### Input format and patient-level splits

We use paired volumes and a patient-identity mapping:

```text
dataset/
├── image_1.nii.gz
├── label_1.nii.gz
├── ...
└── index_mapping.csv  # columns: new_index, patient_id
```

We keep every patient's phases, views, and slices in the same split. For the
in-domain datasets, our supplied splitter uses a patient-level 3:1:1
train/validation/test protocol: an outer five-fold partition selects the test
fold, then one quarter of the remaining patients form the validation set.
This gives 60/20/20 patients for ACDC, 300/100/100 for CAMUS, and 12/4/4 for
CHAOS per fold. These are our experimental splits, not the providers' official
challenge splits. Our external CardiacUDA benchmark uses all eight selected
patients for testing rather than applying this in-domain split. The current
evaluation CLI still uses the in-domain splitter; a full external-set entry
point has not yet been included in this release.

For ACDC and CAMUS, provide a complete `index_mapping.csv` with `new_index`
and `patient_id`, so phases and views remain grouped. For CHAOS, each file ID
already identifies a single patient; the loader can use this ID when no mapping
is present. We do not redistribute images or annotations. Obtain them from
the linked dataset providers and follow their access and citation requirements.

## Static-memory training

### Choosing a static-memory parameterization

We study two static-memory parameterizations in our ablations:

| Intended use | Optimization target | Memory representation |
| --- | --- | --- |
| Static-only segmentation | Directly train memory tokens (and their positional parameters) | Compact memory (our “2M” memory size) |
| Static memory followed by test-time working-memory aggregation | Train pseudo-image and pseudo-mask logits through the frozen encoder | Encoder-generated static memory, composed with encoded corrected observations |

Our ablation results show that the two approaches achieve similar static
segmentation performance. For static-only training, we directly optimize the
compact memory. For subsequent test-time working-memory aggregation, we train
pseudo-image and pseudo-mask logits through the frozen encoder.

Here, “2M” denotes our compact memory size; it
does not denote the number of trainable pseudo-image and pseudo-mask logits.
Storage in bytes depends on tensor shapes, dtype, and whether positional
encodings are included, and should be reported separately from parameter count.

We provide both routes through `MemSegAgent` and `scripts/train_static.py`.
They share memory attention, decoding, loss, and the training loop. Direct-token
training initializes and optimizes `learnable_memory` and `learnable_memory_pos`;
pseudo-observation training initializes logits and encodes them at each forward
pass. Select the route with `memory_mode` in YAML or `--memory-mode` on the CLI.

```bash
cp configs/default.yaml configs/acdc_lv.yaml
# Edit paths, target_class, num_classes, and output_dir.
python scripts/train_static.py --config configs/acdc_lv.yaml --fold 0 --memory-mode pseudo_observation
# Static-only direct-token ablation:
python scripts/train_static.py --config configs/acdc_lv.yaml --fold 0 --memory-mode direct_tokens
```

We save checkpoints under `<output_dir>/<memory_mode>/fold_<N>/` so the two
experiments remain separate. Each checkpoint stores the selected memory
parameters and configuration, with no SAM2 backbone weights.

Evaluate a direct-token checkpoint without DINO or test-time feedback:

```bash
python scripts/eval_working_memory.py --config configs/acdc_lv.yaml --fold 0 \
  --static-only \
  --static-checkpoint outputs/static_memory/direct_tokens/fold_1/best_static_memory.pt
```

We also support `--static-only` with pseudo-observation checkpoints for comparison.

## Test-time working memory

```bash
python scripts/eval_working_memory.py \
  --config configs/acdc_lv.yaml \
  --fold 0 \
  --static-checkpoint outputs/static_memory/pseudo_observation/fold_1/best_static_memory.pt \
  --feedback-dice-threshold 0.5 \
  --save-working-memory outputs/working_memory/fold_1.pt
```

This evaluation is an **oracle-feedback simulation**: a ground-truth slice is
used only when its prediction Dice is below the threshold, as a stand-in for a
clinician correction. In deployment, replace this with an audited human
feedback workflow; model predictions are never written as corrections.

## Federated memory-only aggregation

```bash
python scripts/fedavg_memories.py \
  --inputs client_a.pt client_b.pt client_c.pt client_d.pt \
  --weights 120 95 110 102 \
  --output global_static_memory.pt
```

The helper aggregates only pseudo-observation tensors; no raw data, SAM2
parameters, DINO features, or working-memory entries are communicated.
Its communication payload therefore follows the pseudo-observation tensors,
not the compact “2M” memory size discussed for static-only direct-token training.

## Verification

```bash
python -m pytest -q
python -m compileall -q src scripts
```

## Citation

```bibtex
@article{chen2026memsegagent,
  title={Shifting Adaptation from Weight Space to Memory Space: A Memory-Augmented Agent for Medical Image Segmentation},
  author={Chen, Bowen and Gao, Qiaohui and Wan, Shaowen and others},
  journal={arXiv preprint arXiv:2603.05873},
  year={2026}
}
```

Please also cite and comply with the licenses of SAM2, DINOv3, and the data
sources used in an experiment.
