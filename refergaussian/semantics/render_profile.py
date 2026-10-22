"""Load render settings from the same shell profiles used by query workers."""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import subprocess

FORMAL_PROFILES = frozenset({
    "public_time_boundary_gated_v5", "boundary_gated_gaussian_v5",
    "public_time_boundary_gated_v5_numeric", "boundary_gated_gaussian_v5_numeric",
    "r4d_boundary_gated_v5", "r4d_multi_instance_boundary_v6",
    "r4d_renderer_geometry_v7", "r4d_renderer_consistent",
})
RENDER_QUERY_KEYS = frozenset({
    "QUERY_REQUIRE_RENDERER_GEOMETRY", "QUERY_USE_RENDERER_GEOMETRY",
    "QUERY_RENDER_TORCH_MORPHOLOGY", "QUERY_RENDER_TORCH_MORPHOLOGY_DEVICE",
})


@lru_cache(maxsize=16)
def formal_render_settings(profile: str) -> dict[str, str]:
    if profile not in FORMAL_PROFILES:
        raise ValueError(f"Not a formal render profile: {profile}")
    script = Path(__file__).resolve().parents[2] / "scripts/query_eval_profiles.sh"
    clean = {k: v for k, v in os.environ.items()
             if not k.startswith(("GS_QUERY_", "QUERY_", "REFERGAUSSIAN_QUERY_"))}
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; apply_query_eval_profile "$2" || exit; env -0',
         "render-profile", str(script), profile], env=clean, check=True, capture_output=True,
    )
    values = dict(row.decode().split("=", 1) for row in result.stdout.split(b"\0") if row)
    return {k: v for k, v in values.items()
            if k.startswith("GS_QUERY_") or k in RENDER_QUERY_KEYS
            or k == "REFERGAUSSIAN_QUERY_EVAL_PROFILE"}


def apply_formal_render_settings(profile: str) -> dict[str, str]:
    values = dict(formal_render_settings(profile))
    for key in list(os.environ):
        if key.startswith("GS_QUERY_") or key in RENDER_QUERY_KEYS:
            os.environ.pop(key)
    os.environ.update(values)
    return values
