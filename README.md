# MemSeg-Agent

Official implementation of **MemSeg-Agent**, a memory-augmented framework that
shifts medical-image segmentation adaptation from backbone weights to memory
space. See [the paper](https://arxiv.org/abs/2603.05873).

> Research code only; not a clinical device or clinical decision-support tool.

## Modules

| Module | Location | Updated parameters/state |
| --- | --- | --- |
| Supervised static memory | `scripts/train_static.py` | Pseudo-image and pseudo-mask logits; SAM2 is frozen |
| Few-shot memory | `memseg_agent.fewshot` | Non-parametric tokens from support pairs |
| Test-time working memory | `scripts/eval_working_memory.py` | Bounded DINOv3 retrieval buffer with gated EMA updates |
| Agentic composition | `memseg_agent.controller` | Transparent static/few-shot/working routing policy |
| Federated static memory | `scripts/fedavg_memories.py` | Only the two static-memory tensors are averaged |

Raw datasets, annotations, checkpoints, experiment outputs, local paths, and
vendored SAM2/MONAI/DINO code are intentionally excluded. Install external
dependencies and use all datasets under their respective licenses.

## Installation

```bash
git clone https://github.com/<YOUR-ACCOUNT>/MemSeg-Agent.git
cd MemSeg-Agent
# First install the PyTorch build matching your CUDA runtime.
pip install -r requirements.txt
```

Download a SAM2 checkpoint from the
[official SAM2 repository](https://github.com/facebookresearch/sam2), then
copy `configs/default.yaml` and set `sam2_config` and `sam2_checkpoint` to
your local files. The optional DINOv3 retrieval model is downloaded from Hugging
Face and may require access approval.

## Data and patient-level protocol

The loader expects paired volumes and, for multi-volume patients, a mapping:

```text
dataset/
├── image_1.nii.gz
├── label_1.nii.gz
├── ...
└── index_mapping.csv  # columns: new_index, patient_id
```

The supplied splitter prevents one patient's multiple volumes from crossing a
boundary and implements a patient-level 3:1:1 train/validation/test protocol:
an outer five-fold partition selects the test fold, then one quarter of the
remaining patients are validation patients.

The workspace preprocessing used in the paper contains ACDC: **100 patients,
200 ED/ES volumes**; and CAMUS: **500 patients, 2,000 2CH/4CH × ED/ES volumes**.
Neither dataset is redistributed here.

## Static-memory training

### Choosing a static-memory parameterization

The authors studied two parameterizations in their ablations:

| Intended use | Optimization target | Memory representation |
| --- | --- | --- |
| Static-only segmentation | Directly train memory tokens (and their positional parameters) | Compact memory, described as “2M” memory size by the authors |
| Static memory followed by test-time working-memory aggregation | Train pseudo-image and pseudo-mask logits through the frozen encoder | Encoder-generated static memory, composed with encoded corrected observations |

According to the authors' ablation findings, the two approaches achieve
similar static segmentation performance. This is an empirical observation,
not a guarantee of identical predictions or performance in every setting.
For the authors' workflow that subsequently aggregates test-time working
memory, use the pseudo-observation training route.

Here, “2M” denotes the compact memory size in the authors' terminology; it
does not denote the number of trainable pseudo-image and pseudo-mask logits.
Storage in bytes depends on tensor shapes, dtype, and whether positional
encodings are included, and should be reported separately from parameter count.

**Release scope:** `MemSegAgent` and `scripts/train_static.py` currently implement
the pseudo-observation route. The original workspace's `5F_Supmem_3.py` contains
the direct-token ablation (`learnable_memory`, `learnable_memory_pos`); a
direct-token training entry point has not yet been ported into this release.

```bash
cp configs/default.yaml configs/acdc_lv.yaml
# Edit paths, target_class, num_classes, and output_dir.
python scripts/train_static.py --config configs/acdc_lv.yaml --fold 0
```

The best checkpoint stores only `mem_img_logits` and `mem_mask_logits`, never
the SAM2 backbone.

## Test-time working memory

```bash
python scripts/eval_working_memory.py \
  --config configs/acdc_lv.yaml \
  --fold 0 \
  --static-checkpoint outputs/static_memory/fold_1/best_static_memory.pt \
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
