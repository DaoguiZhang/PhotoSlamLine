# Source Snapshot

- Source workspace: `/workspace/code/PSL-fgm-ablation`
- Source HEAD: `0e2f7fc4001019c61984fc7e266b80f188c8a12e`
- Source branch/state: `feat/fixed-geometry-mapper-ablation`
- Snapshot branch: `source/fgm-ablation-20261019`
- Snapshot created (UTC): `2026-10-19T09:26:24Z`
- Includes uncommitted source: **yes**
- Uncommitted modified source files included: `4`
- Uncommitted new/untracked source files included: `5`
- Included source files: `659`
- Included source bytes: `12403112`

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
- M `include/gaussian_mapper_line.h`
- M `include/gaussian_model_line.h`
- M `src/gaussian_mapper_line.cpp`
- M `src/gaussian_model_line.cpp`
- A `scripts/analyze_fgm_curves.py`
- A `scripts/lpips_fgm.py`
- A `scripts/run_frozen_abl.sh`
- A `scripts/run_frozen_abl_perproc.sh`
- A `scripts/summarize_frozen_abl.py`

## Verification

`SOURCE_MANIFEST.sha256` lists the SHA-256 of every file in this snapshot except
the manifest itself. Verify with:

```bash
sha256sum -c SOURCE_MANIFEST.sha256
```
