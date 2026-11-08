# archive/raster-perf-v5 — manifest

Anchor commit: `a92a21ae6d` (branch `feat/mapper-budget-v1`).

## What is stored here

| Item | Path |
|---|---|
| Commit id | `SOURCE_COMMIT.txt` |
| Source diff vs raster-perf-v4 (`4530435fee`), scoped to code/config/scripts/reports | `diff_vs_v4.patch` (32 KB) |
| Files touched by that diff | `changed_files.txt` (5 files) |
| Config in force | `cfg/replica_rgbd.yaml` |
| Binary identifier | `binaries.sha256` (`bin/replica_rgbd`) |
| Text metrics (logs, csv, yaml, per-frame eval) | `metrics/` |
| Checksums of everything above | `CHECKSUMS.sha256` |

## What is NOT copied here (too large, left in place)

The rendered evaluation images and the full run trees stay under
`/workspace/code/PSL-perf/abl_out/` (disk is at 96%). They are reproducible from
the commands in the report, and LPIPS can be recomputed from them without
retraining:

- `abl_out/gold_r0/`            (FP64 golden self-check run, 131 MB)
- `abl_out/v5gate/`             (100-step A/C gate, 100 MB)
- `abl_out/v5p_r0/`             (5000-step A→C, 390 MB)
- `abl_out/v5p_r1/`             (5000-step C→A, 396 MB)

`metrics/` keeps every `per_frame.csv`, `gaussian_training_log.txt`, `run.log`
and `*.used.yaml` from those four trees, so the numbers in the report are
checkable without the images.

## Note on history hygiene

Commit `a92a21ae6d` was created with `git add -A` and therefore also captured
the pre-existing `abl_out/` result trees (16,888 files). That commit id is kept
unchanged because this round is anchored on it. The following commit
`15ac6ee227` adds `.gitignore` (`abl_out/`) and removes those paths from
tracking while keeping the files on disk. `diff_vs_v4.patch` above is scoped to
source paths only, so it does not contain the result trees.
