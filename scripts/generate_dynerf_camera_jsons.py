#!/usr/bin/env python3
"""Create camera metadata required by the DyNeRF referring pipeline.

Neural 3D Video / DyNeRF scenes provide ``poses_bounds.npy`` in LLFF format,
while the referring pipeline also expects one Nerfies-style ``camera/*.json``
file per image.  The supported monocular release layout uses a fixed ``cam00``
camera, so every frame receives the same calibrated camera record.

Example:
    python scripts/generate_dynerf_camera_jsons.py \
        --dataset-dir data/dynerf/coffee_martini
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from refergaussian.dataset_contract import dynerf_camera_rotation, writable_path, DYNERF_CAMERA_CONTRACT


def load_llff_poses(dataset_dir: Path) -> tuple[np.ndarray, float, float, float]:
    """Load poses_bounds.npy and return (c2w_all, H, W, focal) for all cameras."""
    poses_arr = np.load(dataset_dir / "poses_bounds.npy")
    poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
    H, W, focal = poses[0, :, -1]
    return poses, float(H), float(W), float(focal)


def llff_c2w_to_nerfies_camera(
    c2w_raw: np.ndarray,
    H: float,
    W: float,
    focal: float,
    actual_H: int,
    actual_W: int,
) -> dict:
    """Convert raw LLFF pose using exactly the pinned 4DGS axis convention."""
    R_c2w = dynerf_camera_rotation(c2w_raw)
    C = c2w_raw[:3, 3].astype(float)
    R_w2c = R_c2w.T

    # Pinned upstream uses 2704 / image_width, independently of metadata W.
    focal_scaled_x = focal / (2704.0 / actual_W)
    # Principal point: assume image center
    cx = actual_W / 2.0
    cy = actual_H / 2.0

    camera_json = {
        "refergaussian_camera_contract": DYNERF_CAMERA_CONTRACT,
        "orientation": R_w2c.tolist(),
        "position": C.tolist(),
        "focal_length": float(focal_scaled_x),
        "principal_point": [float(cx), float(cy)],
        "skew": 0.0,
        "pixel_aspect_ratio": 1.0,
        "radial_distortion": [0.0, 0.0, 0.0],
        "tangential_distortion": [0.0, 0.0],
        "image_size": [int(actual_W), int(actual_H)],
    }
    return camera_json


def get_actual_image_size(dataset_dir: Path, cam_name: str = "cam00") -> tuple[int, int]:
    """Get actual image size from cam00/images/ directory."""
    cam_dir = dataset_dir / cam_name / "images"
    imgs = sorted(cam_dir.glob("*.png"))
    if not imgs:
        raise FileNotFoundError(f"No PNG images found under {cam_dir}")
    from PIL import Image as PILImage
    with PILImage.open(imgs[0]) as img:
        W, H = img.size
    return int(W), int(H)


def get_frame_ids(dataset_dir: Path, cam_name: str = "cam00") -> list[str]:
    """Get list of frame image_ids from cam00/images/."""
    cam_dir = dataset_dir / cam_name / "images"
    imgs = sorted(cam_dir.glob("*.png"))
    return [img.stem for img in imgs]


def generate_camera_jsons(
    dataset_dir: Path,
    cam_index: int = 0,
    dry_run: bool = False,
) -> None:
    """Generate camera/XXXX.json files for a dynerf dataset."""
    dataset_dir = Path(dataset_dir).resolve() if dry_run else writable_path(dataset_dir)

    if not (dataset_dir / "poses_bounds.npy").exists():
        raise FileNotFoundError(f"poses_bounds.npy not found in {dataset_dir}")

    poses, H_meta, W_meta, focal_meta = load_llff_poses(dataset_dir)
    if cam_index != 0:
        raise ValueError("The released DyNeRF protocol evaluates cam00; --cam-index must be 0")
    cam_name = "cam00"
    if not (dataset_dir / cam_name / "images").is_dir():
        raise FileNotFoundError("Extract frames first with scripts/prepare_dynerf.py")

    # Get the c2w for the selected camera
    c2w_raw = poses[cam_index, :, :4]  # (3, 4)
    print(f"c2w_raw for cam_index={cam_index}:\n{c2w_raw}")

    # Get actual image size from disk
    actual_W, actual_H = get_actual_image_size(dataset_dir, cam_name)
    print(f"Actual image size: {actual_W}x{actual_H}")

    # Build camera JSON
    camera_json = llff_c2w_to_nerfies_camera(
        c2w_raw, H_meta, W_meta, focal_meta, actual_H, actual_W
    )

    # Get all frame IDs
    frame_ids = get_frame_ids(dataset_dir, cam_name)
    print(f"N frames: {len(frame_ids)}")

    # Output directory
    camera_dir = dataset_dir / "camera"
    if not dry_run:
        writable_path(camera_dir)
    if not dry_run:
        camera_dir.mkdir(exist_ok=True)

    # Write one JSON per frame (all identical since camera doesn't move for monocular video)
    for frame_id in frame_ids:
        out_path = camera_dir / f"{frame_id}.json"
        if not dry_run:
            writable_path(out_path)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(camera_json, f, indent=2)

    if not dry_run:
        print(f"Generated {len(frame_ids)} camera JSON files in {camera_dir}")
        # Verify
        sample = camera_dir / f"{frame_ids[0]}.json"
        loaded = json.loads(sample.read_text())
        print(f"Sample {sample.name}:")
        print(f"  focal_length: {loaded['focal_length']:.2f}")
        print(f"  principal_point: {loaded['principal_point']}")
        print(f"  image_size: {loaded['image_size']}")
    else:
        print(f"[dry-run] Would generate {len(frame_ids)} camera JSONs in {camera_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate dynerf camera JSONs for query pipeline")
    parser.add_argument("--dataset-dir", required=True, help="Path to dynerf dataset directory")
    parser.add_argument("--cam-index", type=int, default=0, help="Camera index in poses_bounds.npy to use (default: 0=cam00)")
    parser.add_argument("--dry-run", action="store_true", help="Don't write files, just print what would be done")
    args = parser.parse_args()

    generate_camera_jsons(
        dataset_dir=Path(args.dataset_dir),
        cam_index=args.cam_index,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
