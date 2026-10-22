# R4DGS: Referring Segmentation in 4D Gaussian Splatting

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-brightgreen.svg)](#quick-start)
[![ACM MM 2026](https://img.shields.io/badge/ACM%20MM-2026-8A2BE2.svg)](#citation)

[Project Page](https://trump0412.github.io/R4DGS/) | [Paper](https://doi.org/10.1145/3767308.3836021) | [R4D-Bench-QA](https://huggingface.co/datasets/LiYacheng/r4d-bench-qa) | [Citation](#citation)

**Bangpu Chen, Yaxuan Li, Shirui Peng, Xiangtian Si, Liuxin Chu, Xitong Cao, Hongbo Jin, Jiayu Ding**

</div>

ReferGaussian performs natural-language referring segmentation in dynamic 4D Gaussian scenes. It uses a Qwen-based Refer-Planner, Grounded-SAM2 masks, and training-free multi-frame Gaussian entity lifting over a frozen standard 4DGS scene. Final masks are rendered from the selected Gaussian entities with synchronized Stage-1 boundary gating.

<p align="center">
  <img src="docs/assets/teaser2.webp" width="92%" alt="ReferGaussian qualitative results"/>
</p>

## Method

<p align="center">
  <img src="docs/assets/framework.webp" width="96%" alt="ReferGaussian framework"/>
</p>

ReferGaussian first grounds language into tracked multi-frame masks, lifts their consistent spatial evidence into Gaussian entities, stores them in EntityBank, and reasons over entity identity and temporal state. Stage-1 masks guide entity assignment and boundary gating for the final Gaussian-rendered predictions.

## Results

Results reported in the paper (%, higher is better).

| Benchmark | Method | Acc | vIoU |
|---|---|---:|---:|
| R4D-Bench-QA | Segment then Splat | 55.6 | 28.4 |
| R4D-Bench-QA | 4D LangSplat | 58.4 | 32.1 |
| R4D-Bench-QA | **ReferGaussian** | **76.5** | **34.4** |
| 4D LangSplat HyperNeRF, 4 scenes | 4D LangSplat | 90.83 | **72.26** |
| 4D LangSplat HyperNeRF, 4 scenes | **ReferGaussian** | **93.01** | 62.95 |

## Quick Start

Requirements: Linux, Miniconda, NVIDIA GPU, CUDA 12.1, and COLMAP.

### 1. Installation

```bash
git clone https://github.com/Trump0412/R4DGS.git
cd R4DGS
bash scripts/setup.sh
source scripts/common.sh
```

### 2. Data Preparation

```bash
bash scripts/download_4dlangsplat_annotations.sh
bash scripts/download_r4d_bench_qa.sh
bash scripts/prepare_hypernerf.sh
```

For R4D-Bench-QA, download the six scenes below from [Neural 3D Video](https://github.com/facebookresearch/Neural_3D_Video) into `data/dynerf/<scene>/`:

```text
data/dynerf/<scene>/
├── cam00.mp4
├── cam01.mp4
├── ...
└── poses_bounds.npy
```

```bash
for scene in coffee_martini cook_spinach cut_roasted_beef flame_salmon_1 flame_steak sear_steak; do
  gs_python scripts/prepare_dynerf.py --dataset-dir "${GS_DATA_ROOT}/dynerf/${scene}"
done
```

### 3. Model Weights

```bash
bash scripts/download_models.sh
```

### 4. 4DGS Training

Train the four scenes for the 4D LangSplat benchmark:

```bash
for scene in misc/americano interp/chickchicken misc/espresso misc/split-cookie; do
  gs_python scripts/train_4dgs.py --dataset hypernerf --scene "${scene}" --gpu 0
done
```

For R4D-Bench-QA, also train:

```bash
for scene in interp/cut-lemon1 misc/keyboard interp/torchocolate; do
  gs_python scripts/train_4dgs.py --dataset hypernerf --scene "${scene}" --gpu 0
done

for scene in coffee_martini cook_spinach cut_roasted_beef flame_salmon_1 flame_steak sear_steak; do
  gs_python scripts/train_4dgs.py --dataset dynerf --scene "${scene}" --gpu 0
done
```

Checkpoints and test renders are saved to `runs/baseline_4dgs/<dataset>/<scene>/`.

### 5. Inference and Evaluation

| Benchmark | Scene data | Scenes | Queries | CLI |
|---|---|---:|---:|---|
| 4D LangSplat | HyperNeRF | 4 | 9 | `4dlangsplat` |
| R4D-Bench-QA | HyperNeRF + Neural 3D Video | 12 | 89 | `r4d-bench` |

The 4D LangSplat split covers `americano`, `chickchicken`, `espresso`, and `split-cookie`. R4D-Bench-QA contains 36 temporal, 29 multi-target/reasoning, and 24 zero-target queries.

```bash
# 4D LangSplat: 4 scenes, 9 queries
gs_python scripts/run_benchmark.py 4dlangsplat --gpus 0

# R4D-Bench-QA: 12 scenes, 89 queries
gs_python scripts/run_benchmark.py r4d-bench --gpus 0
```

Use `--gpus 0 1 2 3` for multiple GPUs and `--output reports/<run-name>` to choose the output directory. Each command reports Acc and vIoU; see [metric definitions](docs/METRICS.md).

## Citation

```bibtex
@inproceedings{chen2026r4dgs,
  title     = {R4DGS: Referring Segmentation in 4D Gaussian Splatting},
  author    = {Chen, Bangpu and Li, Yaxuan and Peng, Shirui and Si, Xiangtian and Chu, Liuxin and Cao, Xitong and Jin, Hongbo and Ding, Jiayu},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  year      = {2026},
  doi       = {10.1145/3767308.3836021}
}
```
