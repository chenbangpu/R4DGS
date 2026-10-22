"""Fail closed when formal evaluation is given predictions from an older renderer."""

RENDERER_CONTRACT = "upstream_4dgs_full_scene_semantic_v1"


def validate_standard_prediction(payload: dict) -> None:
    renderer = payload.get("semantic_renderer") or {}
    if renderer.get("contract") != RENDERER_CONTRACT:
        raise ValueError("Prediction lacks the standard 4DGS renderer contract; rerun in a new output directory")
    if payload.get("selection_status") == "semantic_empty":
        if not payload.get("empty_selection") or any(f.get("query_active") for f in payload.get("frames", [])):
            raise ValueError("Invalid semantic-empty prediction")
        return
    if payload.get("selection_status") != "resolved" or not renderer.get("full_scene_occlusion"):
        raise ValueError("Resolved predictions must use full-scene Gaussian occlusion")
    if not renderer.get("checkpoint_identity"):
        raise ValueError("Missing rendered checkpoint identity")
    coverage = payload.get("stage1_boundary_coverage") or {}
    if coverage.get("required") and (
        coverage.get("missing_match_count", 0) or coverage.get("stale_match_count", 0)
    ):
        raise ValueError("Incomplete synchronized Stage-1 boundaries; rerun the failed query")
    if any(f.get("used_direct_stage1_mask") for f in payload.get("frames", [])):
        raise ValueError("Direct Stage-1 masks are not valid formal predictions")
