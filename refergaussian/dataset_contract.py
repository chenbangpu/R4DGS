"""Camera and frame conventions of the pinned upstream 4DGS data loaders."""
from __future__ import annotations

from pathlib import Path
import numpy as np

DYNERF_FRAME_COUNT = 300
DYNERF_IMAGE_SIZE = (1352, 1014)
DYNERF_CAMERA_CONTRACT = "upstream_llff_axes_v1"


def writable_path(path: str | Path) -> Path:
    """Never write to the permanent read-only source mount, including symlinks."""
    resolved = Path(path).expanduser().resolve()
    if resolved == Path("/share") or Path("/share") in resolved.parents:
        raise ValueError(f"Read-only source path: {resolved}; use a working copy outside /share")
    return resolved


def dynerf_time_values(count: int) -> np.ndarray:
    if not 0 <= count <= DYNERF_FRAME_COUNT:
        raise ValueError(f"Pinned DyNeRF loader supports at most {DYNERF_FRAME_COUNT} frames, got {count}")
    return np.arange(count, dtype=np.float32) / float(DYNERF_FRAME_COUNT)


def dynerf_camera_rotation(raw_c2w: np.ndarray) -> np.ndarray:
    """Graphdeco R (transposed world-to-camera), matching upstream load_meta."""
    raw = np.asarray(raw_c2w, dtype=np.float32)
    if raw.shape != (3, 4):
        raise ValueError(f"Expected LLFF c2w [3,4], got {raw.shape}")
    return np.stack((raw[:, 1], raw[:, 0], -raw[:, 2]), axis=1)


def validate_dynerf_layout(dataset_dir: Path, *, cameras: bool = True) -> list[str]:
    """Read-only validation of the prepared, 300-frame cam00 protocol."""
    from PIL import Image
    import json
    errors = []
    for name in ("poses_bounds.npy", "points3D_downsample2.ply"):
        if not (dataset_dir / name).is_file():
            errors.append(f"missing {dataset_dir / name}; run scripts/prepare_dynerf.py")
    videos = sorted(dataset_dir.glob("cam*.mp4"))
    if not videos or videos[0].stem != "cam00":
        errors.append(f"missing cam00.mp4 / camera videos under {dataset_dir}")
    if (dataset_dir / "poses_bounds.npy").is_file():
        try:
            if np.load(dataset_dir / "poses_bounds.npy", allow_pickle=False).shape[0] != len(videos):
                errors.append("camera video count differs from poses_bounds.npy")
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    expected = [f"{i:04d}.png" for i in range(DYNERF_FRAME_COUNT)]
    for video in videos:
        image_dir = dataset_dir / video.stem / "images"
        paths = sorted(image_dir.glob("*.png"))
        if [p.name for p in paths] != expected:
            errors.append(f"{image_dir}: expected 0000.png through 0299.png")
        elif paths:
            for path in (paths[0], paths[-1]):
                with Image.open(path) as image:
                    if image.size != DYNERF_IMAGE_SIZE:
                        errors.append(f"{path}: expected image size {DYNERF_IMAGE_SIZE}, got {image.size}")
    if cameras:
        for i in range(DYNERF_FRAME_COUNT):
            path = dataset_dir / "camera" / f"{i:04d}.json"
            if not path.is_file():
                errors.append(f"missing camera {path}")
                break
            if json.loads(path.read_text()).get("refergaussian_camera_contract") != DYNERF_CAMERA_CONTRACT:
                errors.append(f"stale camera convention in {path}; regenerate camera JSONs")
                break
    return errors
