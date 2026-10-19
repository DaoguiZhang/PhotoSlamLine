# Source Snapshot

- Source workspace: `/workspace/code/PSL-verify-old`
- Source HEAD: `a3022387b3c7acca1876e21a8d2d02dbe6ff06e5`
- Source branch/state: `(detached)` (detached)
- Snapshot branch: `source/verify-old-20261019`
- Snapshot created (UTC): `2026-10-19T09:26:47Z`
- Includes uncommitted source: **no**
- Uncommitted modified source files included: `0`
- Uncommitted new/untracked source files included: `0`
- Included source files: `536`
- Included source bytes: `11439365`

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

None. The snapshot matches the source HEAD tree for all included files.

## Verification

`SOURCE_MANIFEST.sha256` lists the SHA-256 of every file in this snapshot except
the manifest itself. Verify with:

```bash
sha256sum -c SOURCE_MANIFEST.sha256
```
