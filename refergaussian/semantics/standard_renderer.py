"""Full-scene 4DGS semantic rendering with the pinned upstream CUDA rasterizer.

Membership changes color only. Non-selected Gaussians retain their opacity,
shape and depth, so they occlude selected Gaussians. No 2D evidence enters here.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from refergaussian.checkpoint import checkpoint_identity
from .renderer_geometry import _load_renderer_runtime, _renderer_projection_metadata
from .source_images import resolve_dataset_image_entries
from .semantic_renderer import binarize_alpha_map

from refergaussian.prediction_contract import RENDERER_CONTRACT


def deform_full_scene(gaussians, time_value: float) -> dict[str, torch.Tensor]:
    """Same fine-stage deformation/activation order as pinned gaussian_renderer."""
    xyz = gaussians.get_xyz
    time = torch.full((xyz.shape[0], 1), float(time_value), dtype=xyz.dtype, device=xyz.device)
    with torch.no_grad():
        centers, scales, rotations, opacity, _ = gaussians._deformation(
            xyz, gaussians._scaling, gaussians._rotation, gaussians._opacity,
            gaussians.get_features, time,
        )
        return {
            "means3D": centers,
            "means2D": torch.zeros_like(centers),
            "scales": gaussians.scaling_activation(scales),
            "rotations": gaussians.rotation_activation(rotations),
            "opacities": gaussians.opacity_activation(opacity),
        }


def rasterize_membership(state, gaussian_ids, rasterizer) -> torch.Tensor:
    """Render a binary membership channel without removing any occluding points."""
    centers = state["means3D"]
    ids = torch.as_tensor(gaussian_ids, device=centers.device, dtype=torch.long).reshape(-1)
    if ids.numel() == 0 or torch.any(ids < 0) or torch.any(ids >= centers.shape[0]):
        raise ValueError("Selected Gaussian IDs must be nonempty and within the full checkpoint")
    colors = torch.zeros_like(centers)
    colors[ids] = 1.0
    with torch.no_grad():
        # The upstream wrapper passes both SH and override_color; the extension
        # requires exactly one. Supply semantic RGB and no SH explicitly.
        image, _radii, _depth = rasterizer(
            **state, shs=None, colors_precomp=colors, cov3D_precomp=None,
        )
    if image.ndim != 3 or image.shape[0] != 3 or not torch.isfinite(image).all():
        raise RuntimeError("Invalid semantic image from the upstream Gaussian rasterizer")
    alpha = image[0]
    if torch.any(alpha < -1e-5) or torch.any(alpha > 1.0 + 1e-5):
        raise RuntimeError("Semantic contribution outside [0,1]; expected front-to-back compositing")
    return alpha.clamp(0.0, 1.0)


class Standard4DGSSemanticRenderer:
    def __init__(self, run_dir: Path, dataset_dir: Path):
        self.identity = checkpoint_identity(run_dir)
        self.args, self.gaussians, _, _, self.iteration = _load_renderer_runtime(run_dir)
        self.dataset_dir = Path(dataset_dir)
        self.entries = {str(row["image_id"]): row for row in resolve_dataset_image_entries(dataset_dir)}
        self._key = None
        self._state = None
        self._rasterizer = None

    def mask(self, gaussian_ids, image_id: str, time_value: float, image_size,
             *, relative_threshold: float, absolute_threshold: float) -> np.ndarray:
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

        entry = self.entries.get(str(image_id))
        if entry is None or not np.isclose(float(entry["time_value"]), time_value, atol=1e-6, rtol=0):
            raise ValueError(f"Frame/time does not match the pinned source-camera contract: {image_id}, {time_value}")
        key = (str(image_id), float(time_value), tuple(image_size))
        if key != self._key:
            world_view, full_projection, _ = _renderer_projection_metadata(self.dataset_dir, entry)
            device = self.gaussians.get_xyz.device
            world = torch.as_tensor(world_view, device=device, dtype=torch.float32)
            full = torch.as_tensor(full_projection, device=device, dtype=torch.float32)
            projection = torch.linalg.solve(world, full)
            settings = GaussianRasterizationSettings(
                image_height=int(image_size[1]), image_width=int(image_size[0]),
                tanfovx=float(1.0 / projection[0, 0]), tanfovy=float(1.0 / projection[1, 1]),
                bg=torch.zeros(3, device=device), scale_modifier=1.0,
                viewmatrix=world, projmatrix=full, sh_degree=0,
                campos=world.inverse()[3, :3], prefiltered=False, debug=False,
            )
            self._state = deform_full_scene(self.gaussians, time_value)
            self._rasterizer = GaussianRasterizer(raster_settings=settings)
            self._key = key
        alpha = rasterize_membership(self._state, gaussian_ids, self._rasterizer)
        return binarize_alpha_map(alpha, relative_threshold, absolute_threshold).cpu().numpy().astype(bool)

    def provenance(self) -> dict:
        return {
            "contract": RENDERER_CONTRACT,
            "checkpoint_identity": self.identity,
            "full_scene_occlusion": True,
            "deformation_time": "exact_source_frame",
            "boundary_gating": "synchronized_stage1_clip_after_rasterization",
        }
