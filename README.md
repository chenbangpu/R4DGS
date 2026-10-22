# R4DGS: Referring Segmentation in 4D Gaussian Splatting

[Project Page](https://trump0412.github.io/R4DGS/) | [Paper](https://doi.org/10.1145/3767308.3836021) | [R4D-Bench-QA](https://huggingface.co/datasets/LiYacheng/r4d-bench-qa)

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

```bash
# 4D LangSplat: 4 scenes, 9 queries
gs_python scripts/run_benchmark.py 4dlangsplat --gpus 0

# R4D-Bench-QA: 12 scenes, 89 queries
gs_python scripts/run_benchmark.py r4d-bench --gpus 0
```

Use `--gpus 0 1 2 3` for multiple GPUs. Each command prints Acc, vIoU, and the evaluation report path.
