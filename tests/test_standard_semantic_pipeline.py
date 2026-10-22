"""CPU numerical oracles for the CUDA renderer adapter and public contracts.

The rasterizer double implements one pixel's analytic front-to-back blend.
These tests verify adapter inputs, not the compiled CUDA kernel itself.
"""
import json
from contextlib import ExitStack
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from refergaussian.dataset_contract import dynerf_time_values, writable_path
from refergaussian.semantics.standard_renderer import deform_full_scene, rasterize_membership
from refergaussian.semantics.query_render import (
    _apply_render_profile_env_defaults, _fuse_query_and_cloud_masks, _fusion_options_for_profile,
    _dynerf_test_ids_from_renders,
    QueryTrack, _query_track_match_for_time,
)
from refergaussian.semantics.render_profile import formal_render_settings
from scripts.generate_dynerf_camera_jsons import llff_c2w_to_nerfies_camera

ROOT = Path(__file__).resolve().parents[1]


class StandardSemanticPipelineTest(unittest.TestCase):
    def state(self):
        return {"means3D": torch.tensor([[0., 0., 1.], [0., 0., 2.]]),
                "means2D": torch.zeros(2, 3), "scales": torch.ones(2, 3),
                "rotations": torch.tensor([[1., 0., 0., 0.]] * 2),
                "opacities": torch.tensor([[0.8], [0.8]])}

    def oracle(self, **kwargs):
        self.assertIsNone(kwargs["shs"])
        self.assertEqual(len(kwargs["means3D"]), 2, "unselected occluders must remain")
        rgb, transmittance = torch.zeros(3), 1.0
        for index in torch.argsort(kwargs["means3D"][:, 2]):
            alpha = kwargs["opacities"][index, 0]
            rgb += transmittance * alpha * kwargs["colors_precomp"][index]
            transmittance *= 1.0 - alpha
        return rgb[:, None, None], None, None

    def test_membership_uses_full_scene_occlusion_and_depth(self):
        state = self.state()
        self.assertAlmostEqual(float(rasterize_membership(state, [1], self.oracle)), 0.16, places=6)
        self.assertAlmostEqual(float(rasterize_membership(state, [0, 1], self.oracle)), 0.96, places=6)
        state["means3D"][:, 2] = torch.tensor([2., 1.])
        self.assertAlmostEqual(float(rasterize_membership(state, [1], self.oracle)), 0.8, places=6)

    def test_deformation_evaluates_requested_time_without_quantization(self):
        model = types.SimpleNamespace(get_xyz=torch.zeros(2, 3), _scaling=torch.zeros(2, 3),
                                      _rotation=torch.tensor([[1., 0., 0., 0.]] * 2),
                                      _opacity=torch.zeros(2, 1), get_features=torch.zeros(2, 1, 3),
                                      scaling_activation=torch.exp, rotation_activation=lambda x: x,
                                      opacity_activation=torch.sigmoid)
        def deformation(xyz, scale, rotation, opacity, sh, time):
            return xyz + time, scale + time, rotation, opacity + time, sh
        model._deformation = deformation
        state = deform_full_scene(model, 17 / 300)
        torch.testing.assert_close(state["means3D"], torch.full((2, 3), 17 / 300))
        torch.testing.assert_close(state["scales"], torch.full((2, 3), 17 / 300).exp())

    def test_boundary_gate_cannot_add_pixels_or_replace_missing_projection(self):
        with patch.dict(os.environ, {}, clear=True):
            _apply_render_profile_env_defaults("public_time_boundary_gated_v5_numeric")
            options = _fusion_options_for_profile("public_time_boundary_gated_v5_numeric")
            cloud = np.zeros((80, 80), bool); cloud[10:70, 10:70] = True; cloud[35:45, 35:45] = False
            stage1 = np.zeros_like(cloud); stage1[20:60, 20:60] = True
            mask, *_ = _fuse_query_and_cloud_masks(stage1, cloud, fusion_options=options)
            self.assertIsNotNone(mask)
            self.assertFalse(np.any(mask & ~cloud))
            mask, *_ = _fuse_query_and_cloud_masks(stage1, None, fusion_options=options)
            self.assertIsNone(mask)

    def test_rerender_and_shell_use_identical_render_parameters(self):
        for profile in ("r4d_renderer_consistent", "public_time_boundary_gated_v5_numeric"):
            with patch.dict(os.environ, {"GS_QUERY_FINAL_ERODE_KERNEL": "99"}, clear=True):
                _apply_render_profile_env_defaults(profile)
                actual = {k: os.environ[k] for k in formal_render_settings(profile)}
                self.assertEqual(actual, formal_render_settings(profile))
                self.assertEqual(actual["QUERY_REQUIRE_RENDERER_GEOMETRY"], "1")
                self.assertEqual(actual["GS_QUERY_CLOUD_RENDER_MODE"], "standard_4dgs")
        self.assertEqual(formal_render_settings("r4d_renderer_consistent")["GS_QUERY_FINAL_ERODE_KERNEL"], "9")

    def test_dynerf_rotation_and_camera_center_match_upstream_algebra(self):
        raw = np.array([[0., -1., 0., 3.], [1., 0., 0., 4.], [0., 0., 1., 5.]])
        pose = np.concatenate([raw[:, 1:2], -raw[:, :1], raw[:, 2:4]], axis=-1)
        expected = -pose[:, :3]; expected[:, 0] *= -1
        camera = llff_c2w_to_nerfies_camera(raw, 2028, 2704, 2000, 1014, 1352)
        np.testing.assert_allclose(np.array(camera["orientation"]).T, expected)
        np.testing.assert_allclose(camera["position"], raw[:, 3])
        self.assertEqual(camera["focal_length"], 1000)
        # Apply the generated and upstream w2c to the same world points.
        points = np.array([[4., 2., 8.], [0., 1., 9.]])
        np.testing.assert_allclose((points - raw[:, 3]) @ expected,
                                   (points - camera["position"]) @ np.array(camera["orientation"]).T)

    def test_dynerf_render_names_map_to_source_ids_and_original_times(self):
        with tempfile.TemporaryDirectory() as temp:
            for i in range(300):
                (Path(temp) / f"{i:05d}.png").touch()
            ids, times = _dynerf_test_ids_from_renders(Path(temp))
        self.assertEqual((ids[0], ids[-1]), ("0000", "0299"))
        np.testing.assert_array_equal(times, dynerf_time_values(300))
        self.assertAlmostEqual(float(times[-1]), 299 / 300, places=6)

    @unittest.skipUnless(os.name == "posix", "POSIX source-mount guard")
    def test_symlink_cannot_write_into_read_only_source_mount(self):
        with tempfile.TemporaryDirectory() as temp:
            link = Path(temp) / "source"
            link.symlink_to("/share", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "Read-only"):
                writable_path(link / "new-cache")

    def test_download_scripts_call_supported_snapshot_api(self):
        for name in ("download_4dlangsplat_annotations.sh", "download_r4d_bench_qa.sh"):
            source = (ROOT / "scripts" / name).read_text()
            python = source.split("\nfrom __future__ import annotations\n", 1)[1].split("\nPY\n", 1)[0]
            with tempfile.TemporaryDirectory() as temp:
                calls = []
                def download(repo_id, *, repo_type, revision, local_dir, max_workers, etag_timeout,
                             token=None, allow_patterns=None):
                    calls.append(repo_id)
                    return local_dir
                module = types.ModuleType("huggingface_hub")
                module.snapshot_download = download
                module.HfApi = lambda: types.SimpleNamespace(dataset_info=lambda *a, **kw: types.SimpleNamespace(sha="pinned"))
                argv = [name, "test/dataset", temp, "", "pinned"] if "4dlangsplat" in name else [name, "test/dataset", temp, "pinned"]
                with patch.dict(sys.modules, {"huggingface_hub": module}), patch.object(sys, "argv", argv):
                    exec(compile(python, name, "exec"), {})
                self.assertEqual(calls, ["test/dataset"])

    def test_boundary_gate_does_not_use_another_camera_at_the_same_time(self):
        track = QueryTrack(phrase="target", frames=[
            {"image_id": "000001", "time_value": 0.5, "mask_path": "must-not-be-opened"},
        ])
        mask, meta = _query_track_match_for_time(track, 0.5, strict=True, image_id="000002")
        self.assertIsNone(mask)
        self.assertEqual(meta["stage1_match_mode"], "missing_exact_camera_frame")

    def test_formal_evaluator_rejects_old_projection_outputs(self):
        from refergaussian.prediction_contract import validate_standard_prediction, RENDERER_CONTRACT
        with self.assertRaisesRegex(ValueError, "renderer contract"):
            validate_standard_prediction({"selection_status": "resolved"})
        good = {"selection_status": "resolved", "semantic_renderer": {
            "contract": RENDERER_CONTRACT, "full_scene_occlusion": True,
            "checkpoint_identity": {"iteration": 14000},
        }, "frames": [{"used_direct_stage1_mask": False}]}
        validate_standard_prediction(good)
        good["frames"][0]["used_direct_stage1_mask"] = True
        with self.assertRaisesRegex(ValueError, "Stage-1"):
            validate_standard_prediction(good)
        good["frames"][0]["used_direct_stage1_mask"] = False
        good["stage1_boundary_coverage"] = {"required": True, "missing_match_count": 1}
        with self.assertRaisesRegex(ValueError, "synchronized"):
            validate_standard_prediction(good)

    def test_frame_extraction_is_atomic_and_reuses_only_complete_images(self):
        # Small synthetic video interface, no GPU or installed codec required.
        sys.path.insert(0, str(ROOT / "scripts"))
        import prepare_dynerf as prep
        class Video:
            def read(self): return True, np.zeros((8, 8, 3), dtype=np.uint8)
            def release(self): pass
        with tempfile.TemporaryDirectory() as temp, patch.object(prep, "DYNERF_FRAME_COUNT", 3), \
                patch.object(prep, "DYNERF_IMAGE_SIZE", (16, 12)), patch("cv2.VideoCapture", return_value=Video()):
            video = Path(temp) / "cam00.mp4"
            prep.extract_frames(video)
            frames = video.parent / "cam00/images"
            self.assertEqual(sorted(p.name for p in frames.iterdir()), ["0000.png", "0001.png", "0002.png"])
            prep.extract_frames(video)
            (frames / "0001.png").unlink()
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                prep.extract_frames(video)

    def test_runtime_checkpoint_loading_does_not_construct_dataset_scene(self):
        from refergaussian.semantics.renderer_geometry import _load_renderer_runtime
        from types import SimpleNamespace as NS
        calls = []
        class Args:
            def __init__(self, parser): pass
            def extract(self, values): return values
        class Model:
            def __init__(self, degree, args): self._deformation = NS(eval=lambda: calls.append("eval"))
            def load_ply(self, path): calls.append(Path(path).name)
            def load_model(self, path): calls.append(Path(path).name)
        arguments = types.ModuleType("arguments")
        arguments.ModelHiddenParams = arguments.ModelParams = arguments.PipelineParams = Args
        renderer = types.ModuleType("gaussian_renderer"); renderer.GaussianModel = Model
        # No scene.Scene double is provided: any dataset construction fails.
        scene = types.ModuleType("scene")
        with patch.dict(sys.modules, {"arguments": arguments, "gaussian_renderer": renderer, "scene": scene}), \
                patch("refergaussian.semantics.renderer_geometry._prepare_external_imports"), \
                patch("refergaussian.checkpoint.saved_model_args", return_value={"sh_degree": 3}), \
                patch("refergaussian.checkpoint.latest_iteration_dir", return_value=Path("/run/point_cloud/iteration_14000")):
            _, _, _, dataset_scene, iteration = _load_renderer_runtime(Path("/run"))
        self.assertIsNone(dataset_scene)
        self.assertEqual(iteration, 14000)
        self.assertEqual(calls, ["point_cloud.ply", "iteration_14000", "eval"])

    def test_final_export_connects_selection_gate_and_evaluation_contract(self):
        from PIL import Image
        from refergaussian.semantics import query_render as qr
        from refergaussian.prediction_contract import validate_standard_prediction, RENDERER_CONTRACT
        cloud = np.zeros((80, 80), bool)
        cloud[10:70, 10:70] = True
        cloud[35:45, 35:45] = False
        stage1 = np.zeros_like(cloud)
        stage1[20:60, 20:60] = True
        calls = []
        class Renderer:
            def __init__(self, *args): pass
            def mask(self, ids, image_id, time, size, **kwargs):
                calls.append((list(ids), image_id, time, size))
                return cloud.copy()
            def provenance(self):
                return {"contract": RENDERER_CONTRACT, "full_scene_occlusion": True,
                        "checkpoint_identity": {"iteration": 14000}}
        camera = types.SimpleNamespace(points_to_local_points=lambda points: points,
                                       image_size=np.array([80, 80]), focal_length=40.)
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            renders = root / "renders"; renders.mkdir()
            Image.new("RGB", (80, 80)).save(renders / "00000.png")
            gate_path = root / "stage1.png"
            Image.fromarray(stage1.astype(np.uint8) * 255).save(gate_path)
            selection = root / "selection.json"
            track = qr.TrackSample(7, np.array([0.]), np.array([[0., 0., 2.]]),
                                   np.array([[-1., -1., 1.]]), np.array([[1., 1., 3.]]),
                                   np.ones(1), np.ones(1))
            query_track = qr.QueryTrack("target", [{"image_id": "0000", "time_value": 0.,
                "frame_index": 0, "active": True, "mask_path": str(gate_path)}])
            entity_cloud = qr.EntityCloud(7, np.array([0.]), np.zeros((1, 1, 3)),
                                         np.ones(1), np.ones((1, 3)), gaussian_ids=np.array([1]))
            stack.enter_context(patch.dict(os.environ, {"QUERY_FAST_VALIDATION_ONLY": "1"}, clear=True))
            replacements = {
                "_find_render_dir": lambda *_: renders,
                "_load_tracks": lambda *_: {7: track},
                "_load_entity_static_texts": lambda *_: {7: "target"},
                "_load_entity_phrase_hints": lambda *_: {7: ["target"]},
                "_resolve_query_tracks": lambda *_: {"target": query_track},
                "_load_entity_clouds": lambda *_: {0: entity_cloud},
                "_camera_class": lambda: types.SimpleNamespace(from_json=lambda *_: camera),
                "_project_points_to_image": lambda *a, **kw: np.array([[40., 40.]]),
                "_project_box": lambda *a, **kw: None,
            }
            for key, value in replacements.items():
                stack.enter_context(patch.object(qr, key, value))
            stack.enter_context(patch("refergaussian.semantics.standard_renderer.Standard4DGSSemanticRenderer", Renderer))
            selection.write_text(json.dumps({"selection_status": "resolved", "query": "target",
                "selected": [{"id": 7, "role": "entity", "segments": [[0, 0]]}]}))
            output = qr.render_hypernerf_query_video(root, root, selection, root / "positive",
                eval_profile="public_time_boundary_gated_v5_numeric")
            payload = json.loads((output / "validation.json").read_text())
            validate_standard_prediction(payload)
            with Image.open(output / "binary_masks/00000.png") as image:
                final = np.asarray(image) > 0
            self.assertTrue(final.any())
            self.assertFalse(np.any(final & ~cloud))
            self.assertEqual(calls, [([1], "0000", 0., (80, 80))])
            self.assertEqual(payload["frames"][0]["roles"][0]["cloud_opacity_source"],
                             "exact_time_deformed_checkpoint")
            # A semantically rejected query exports an explicit inactive result;
            # it must neither construct a renderer nor fabricate a positive mask.
            selection.write_text(json.dumps({"selection_status": "semantic_empty", "selected": []}))
            output = qr.render_hypernerf_query_video(root, root, selection, root / "negative",
                eval_profile="public_time_boundary_gated_v5_numeric")
            payload = json.loads((output / "validation.json").read_text())
            validate_standard_prediction(payload)
            self.assertTrue(payload["empty_selection"])
            self.assertFalse(payload["frames"][0]["query_active"])
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
