# Source Snapshot

- Source workspace: `/workspace/code/PSL-perf`
- Source HEAD: `ec85e80886e2f0494acb87781e075334bcf2ba58`
- Source branch/state: `feat/mapper-budget-v1`
- Snapshot branch: `source/perf-worktree-20261019`
- Snapshot created (UTC): `2026-10-19T09:26:29Z`
- Includes uncommitted source: **yes**
- Uncommitted modified source files included: `10`
- Uncommitted new/untracked source files included: `3`
- Included source files: `684`
- Included source bytes: `12931565`

## Purpose

This is a pure-source recovery snapshot. It is intentionally a new, single-commit
Git history and does not inherit the original repository history.

## Included

First-party source, headers, CUDA sources, CMake files, configuration files,
scripts, documentation, and necessary third-party source/licences present in
the actual working tree, including uncommitted source changes.

`lib/` and similar directories are not removed by name; only compiled libraries
and other non-source files are excluded by content/extension rules.

## Excluded

- `.git` metadata and all original history
- build trees, CMakeFiles, compiled objects/libraries/executables
- results, abl_out, evidence, experiment archives, run logs and generated reports
- datasets, vocabulary files, weights, checkpoints and generated media
- JPG/JPEG/PNG, video, PLY/OBJ, point clouds, model outputs and compressed archives
- result-bearing handoff/report documents

## Restoration

This snapshot does **not** contain local dependencies, datasets, vocabulary files,
weights, checkpoints or compiled outputs. After cloning, install dependencies,
restore the required data/vocabulary outside the source tree, then configure and
build again (for example with the project CMake/build scripts).

## Uncommitted Source Included

The following working-tree changes are included:
- M `cfg/gaussian_mapper/RGB-D/Replica/office1.yaml`
- M `cfg/gaussian_mapper/RGB-D/Replica/office2.yaml`
- M `cfg/gaussian_mapper/RGB-D/Replica/office3.yaml`
- M `cfg/gaussian_mapper/RGB-D/Replica/office4.yaml`
- M `cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml`
- M `cfg/gaussian_mapper/RGB-D/Replica/room0.yaml`
- M `cfg/gaussian_mapper/RGB-D/Replica/room1.yaml`
- M `cfg/gaussian_mapper/RGB-D/Replica/room2.yaml`
- M `examples/eval_common_views.cpp`
- M `scripts/eval_common_views.py`
- A `scripts/rescore_common_views.py`
- A `scripts/run_renderer_speed.sh`
- A `scripts/run_replica_rgbd_fast_all.sh`

## Verification

`SOURCE_MANIFEST.sha256` lists the SHA-256 of every file in this snapshot except
the manifest itself. Verify with:

```bash
sha256sum -c SOURCE_MANIFEST.sha256
```
