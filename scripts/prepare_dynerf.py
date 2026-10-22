#!/usr/bin/env python3
"""Prepare the pinned 4DGS DyNeRF layout: frames, COLMAP point cloud, cameras."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from refergaussian.dataset_contract import (
    DYNERF_FRAME_COUNT, DYNERF_IMAGE_SIZE, validate_dynerf_layout, writable_path,
)
from generate_dynerf_camera_jsons import generate_camera_jsons


def extract_frames(video: Path) -> None:
    import cv2
    from PIL import Image
    image_dir = writable_path(video.parent / video.stem / "images")
    expected = [f"{i:04d}.png" for i in range(DYNERF_FRAME_COUNT)]
    if image_dir.exists():
        if sorted(p.name for p in image_dir.iterdir()) != expected:
            raise ValueError(f"Incomplete frame directory {image_dir}; move it aside before retrying")
        for name in expected:
            with Image.open(image_dir / name) as image:
                if image.size != DYNERF_IMAGE_SIZE:
                    raise ValueError(f"Wrong frame size: {image_dir / name}; expected {DYNERF_IMAGE_SIZE}")
        return
    image_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="frames-", dir=image_dir.parent) as temporary:
        capture = cv2.VideoCapture(str(video))
        try:
            for name in expected:
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"{video}: fewer than {DYNERF_FRAME_COUNT} frames")
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image.resize(DYNERF_IMAGE_SIZE, Image.Resampling.LANCZOS).save(Path(temporary) / name)
        finally:
            capture.release()
        # Rename only this newly generated directory. Never overwrite source frames.
        Path(temporary).rename(image_dir)


def prepare_scene(dataset_dir: Path) -> None:
    import numpy as np
    dataset_dir = writable_path(dataset_dir)
    poses = np.load(dataset_dir / "poses_bounds.npy", allow_pickle=False)
    videos = sorted(dataset_dir.glob("cam*.mp4"))
    if not videos or videos[0].stem != "cam00" or len(videos) != poses.shape[0]:
        raise ValueError("Expected cam00.mp4 onward, one video for every poses_bounds.npy row")
    for video in videos:
        extract_frames(video)
    output_ply = writable_path(dataset_dir / "points3D_downsample2.ply")
    if not output_ply.is_file():
        upstream = ROOT / "external" / "4DGaussians"
        if not (upstream / "colmap.sh").is_file() or shutil.which("colmap") is None:
            raise FileNotFoundError("Install COLMAP and run scripts/setup.sh before point-cloud preparation")
        # Upstream colmap.sh removes intermediate folders: run it only on our
        # fresh scratch copy, containing poses and the first frame of each camera.
        with tempfile.TemporaryDirectory(prefix="dynerf-colmap-", dir=dataset_dir.parent) as temporary:
            scratch = writable_path(temporary)
            shutil.copy2(dataset_dir / "poses_bounds.npy", scratch / "poses_bounds.npy")
            for video in videos:
                target = scratch / video.stem / "images"
                target.mkdir(parents=True)
                shutil.copy2(dataset_dir / video.stem / "images" / "0000.png", target / "0000.png")
            env = dict(os.environ)
            env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            subprocess.run(["bash", "-e", "colmap.sh", str(scratch), "llff"], cwd=upstream, env=env, check=True)
            candidate = scratch / "points.ply"
            subprocess.run([sys.executable, "-B", str(upstream / "scripts/downsample_point.py"),
                            str(scratch / "colmap/dense/workspace/fused.ply"), str(candidate)],
                           cwd=upstream, env=env, check=True)
            from plyfile import PlyData
            if len(PlyData.read(str(candidate))["vertex"]) == 0:
                raise ValueError("COLMAP produced an empty point cloud")
            shutil.copy2(candidate, output_ply)
    generate_camera_jsons(dataset_dir)
    errors = validate_dynerf_layout(dataset_dir)
    if errors:
        raise ValueError("\n".join(errors))
    print(f"[ready] {dataset_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true", help="Validate without writing or running COLMAP")
    args = parser.parse_args()
    if args.check_only:
        errors = validate_dynerf_layout(args.dataset_dir)
        if errors:
            raise SystemExit("\n".join(errors))
        print(f"[ready] {args.dataset_dir}")
    else:
        prepare_scene(args.dataset_dir)


if __name__ == "__main__":
    main()
