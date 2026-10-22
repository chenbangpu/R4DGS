# R4DGS Release Reproduction Protocol

This repository releases the ReferGaussian referring-inference and evaluation
pipeline. It treats a standard 4D Gaussian scene as a frozen input and includes
a thin wrapper around the pinned upstream trainer for end-to-end reproduction.

## Scope

The supported path contains:

1. Refer-Planner query decomposition;
2. Grounded-SAM2 multi-frame mask evidence;
3. training-free mask-supported Gaussian entity lifting;
4. EntityBank selection and boundary-gated standard 4DGS mask rendering;
5. strict time-sensitive Public and R4D-Bench-QA evaluators.

The repository intentionally excludes custom reconstruction code, scene-quality
metrics, scene checkpoints, model weights, rendered benchmark outputs,
experiment logs, and ablation runners.

## Frozen 4DGS Input

Use the pinned, unmodified upstream 4DGaussians checkout prepared by:

```bash
bash scripts/bootstrap_external.sh
bash scripts/setup_4dgs_env.sh cuda121
source scripts/common.sh
gs_python scripts/train_4dgs.py --dataset hypernerf --scene misc/americano
```

Each query-ready scene run must provide:

```text
<run_dir>/cfg_args
<run_dir>/point_cloud/iteration_<N>/point_cloud.ply
<run_dir>/point_cloud/iteration_<N>/deformation.pth
<run_dir>/test/ours_<N>/renders/*.png
```

`scripts/validate_4dgs_run.py` and batch preflight reject missing inputs. The
checkpoint is never optimized by ReferGaussian. The latest numeric fine-stage
iteration is selected consistently; a partial later iteration is an error.

The query pipeline calls `prepare_4dgs_run.py` before lifting. It derives a
64-time trajectory bank from the frozen upstream deformation network and stores
a checkpoint/configuration identity beside it. Cached legacy polynomial
trajectories are not accepted. `temporal_params.pth` and `config.yaml` are not
required. Network structure is restored from `cfg_args` without overwriting it
with parser defaults. See the README for preparation and migration commands.

This correction changes the geometry consumed by downstream inference. Fresh
Public and R4D aggregate GPU runs are required before claiming that paper
metrics are reproduced by this revision.

## Protocol Identities

The camera-ready R4D-Bench-QA protocol contains 89 English queries over 12 scenes:

- 36 temporal/dynamic queries;
- 29 multi-target or reasoning queries;
- 24 zero-target or distractor queries.

The camera-ready Public protocol is `release_public4_extension`: 4 HyperNeRF
scenes and 9 English time-sensitive queries.

These are the only formal release protocols. Counts, source hashes, category
counts, and scene lists are frozen in `configs/benchmarks/release_protocols.json`.

## Strict Execution

Formal runs must:

1. build a versioned protocol and manifest;
2. run `scripts/preflight_query_batch.py --strict-release`;
3. use one serial worker per GPU;
4. preserve `batch_provenance.json` and every query validation record;
5. evaluate with `--require-complete --require-standard-renderer`;
6. write all new outputs to a new, untracked directory.

Missing queries, unresolved selections, missing masks, or source-camera
mismatches are errors. They cannot be silently skipped when producing an
aggregate.

## Empty-Set Semantics

A zero-target query receives a perfect empty-set score only when both ground
truth and prediction are empty. An empty prediction for non-empty ground truth
scores zero. Missing evidence is an evaluation error, not an empty prediction.

## Asset Policy

Git tracks source code, tests, benchmark protocol metadata, documentation, and
project-page figures only. `.gitignore` excludes `data/`, `models/`, `runs/`,
`reports/`, experiment directories, checkpoints, point clouds, archives, and
videos. `scripts/check_release.py` independently rejects those artifact types
if they are accidentally staged.

Metric formulas and aggregation rules are defined in [METRICS.md](METRICS.md).

## Final mask contract (2026-10-22)

Formal predictions carry `upstream_4dgs_full_scene_semantic_v1`. The pinned
GaussianRasterizer receives the complete deformed scene and binary entity
membership as precomputed RGB, with SH disabled for this semantic pass.
Non-selected Gaussians retain their opacity and participate in occlusion.
Thresholded projections are clipped by synchronized Stage-1 boundary masks;
cleanup may remove pixels but cannot add pixels outside Gaussian support.
This is a boundary-gated method and requires Stage-1 evidence in the evaluated
view. It is not a claim of segmentation independent of target-view 2D evidence.

Both benchmarks evaluate deformation at the source-frame time. The 64-sample
scene bank is only for entity search and motion summaries. Final rendering uses
no selected-only all-frame geometry cache. Shell and Python rendering consume
the same formal profile, and validation records its effective render settings
and checkpoint identity. `run_benchmark.py` passes `--require-standard-renderer`
to the evaluators so older approximate outputs cannot enter a new formal report.
Explicitly resolved semantic-empty queries are the only no-entity exception.

DyNeRF preparation uses cam00, 300 frames at 1352x1014, source IDs `0000` through
`0299`, upstream LLFF axis conversion and timestamps `i/300`. Rebuild older
camera metadata and use fresh query output directories after updating.

The metric formulas and query sets are unchanged. The rendered predictions
have changed and require fresh GPU evaluation before comparison to paper scores.
