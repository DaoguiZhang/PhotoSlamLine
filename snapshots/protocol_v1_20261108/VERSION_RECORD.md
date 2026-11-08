# Locked version record — full GPU mapping, 5 repeats (2026-11-05)

Status: **LOCKED for the formal batch**. Any code change after this record must
go to a new source tree, a new build and a new output root.

## 1. Identity

| item | value |
|---|---|
| candidate repository (untouched) | `/workspace/code/Photo-SLAM-L` |
| branch | `fix/line-rescue-tracking` |
| base commit | `0e2f7fc4001019c61984fc7e266b80f188c8a12e` |
| locked source tree | `/workspace/code/PSL-fullmap-lock-v1` |
| experiment / output root | `/workspace/code/PSL-fullmap-exp-v1` |
| build type | `Release` (top level and ORB-SLAM3; both `CMAKE_BUILD_TYPE=Release`) |
| base metadata | `PSL-fullmap-lock-v1/_lock/{base-commit,base-branch,base-dirty-status,base-dirty-diffstat}.txt` |
| included source diff | `PSL-fullmap-lock-v1/_lock/included-changes.patch` |
| artifact hashes | `PSL-fullmap-lock-v1/_lock/artefact_hashes_final.txt` |
| per-cell config hashes | `PSL-fullmap-exp-v1/config_hashes.csv` |

The locked tree was created with `git archive HEAD`, so it contains **only
tracked files at the base commit**. No `pull`, `reset`, `stash`, or bulk commit
was performed on the candidate.

## 2. Audit of the candidate's uncommitted state

`git status` listed 150 modified paths, but only the following are real content
changes; the rest are mode-bit churn (`100644 -> 100755`).

| path | what it really was | decision |
|---|---|---|
| `ORB-SLAM3/include/KeyFrame.h` | **corruption**: pasted log lines (`LocalMapping:: RunWithLine...`, `free(): invalid pointer`) replaced `// sophus poses` and the `mTcw` declaration. It cannot compile. | repaired to HEAD in the candidate; locked tree already has HEAD content |
| `cfg/gaussian_mapper/RGB-D/Replica/office0.yaml` | diagnostic change `refined_gaussian_max_iter_num: 0` (disables final refinement) | reverted to the committed `10000`; not included |
| `tests/cull_chain_invariant.cc` | +78 lines of new tests | test-only; not built by the mapping batch; excluded |
| `scripts/round_stats.py` | analysis-script bug fix | not used by the batch; excluded |
| `todolist.txt` | +171 lines of notes | excluded |
| `evidence/**.tar.gz` | regenerated archives | excluded |
| all other modified paths | file-mode only (`644 -> 755`) | excluded; no build/experiment effect |
| `ORB-SLAM3/include/LineMode.h` | **mode-bit only**; content identical to HEAD | no functional change included |
| untracked files | `.cache/`, `Temp/`, `myresult/`, `no_viewer/`, root statistics files, `*_asan/`, a few diagnostic scripts | excluded; none needed for build or paper runs |

## 3. Included source changes (3 files, 134 diff lines)

1. `ORB-SLAM3/src/Frame.cc` — `ComputeLineStereoFromRGBD`: validate the
   **rounded** endpoint indices before reading the depth buffer
   (`cvRound(rows-0.4)==rows` was a heap over-read).
2. `ORB-SLAM3/src/KeyFrame.cc` —
   * size `mvpMapLines` at construction (`assign(NL, nullptr)`) and bound
     `GetKeyLineInfo()` by the minimum of `NL`, `mvKeyLines.size()` and
     `mvpMapLines.size()` (the "line container" crash);
   * `<algorithm>` include.
3. `src/gaussian_mapper_line.cpp` — two empty-model guards found during the
   small samples:
   * the inactive-geometry depth cache must not flush into a still-empty model
     (mono runs before `Mapper.min_num_initial_map_kfs`), otherwise
     `densificationPostfixWithLineAwareness` concatenates undefined shadow
     tensors and segfaults. The cache is kept and retried.
   * if initial mapping never started, skip final refinement and shutdown
     render/PLY instead of rendering from an undefined model. The run exits 0
     with `Total Iterations: 0` and is marked **invalid** by the runner.

The two crash-fix hunks are taken from commit `d7e4e9fd2d` (branch
`eval/cross-dataset-frontend`) **by source hunk only**. That branch was not
merged; its other files (e.g. `eval/CRASH_FIXES.md`) are not in the locked tree.

`LineMode.h` is unchanged from HEAD. `KeyFrame.h` and the Replica `office0.yaml`
are identical to HEAD in the locked tree.

## 4. Build and artifact identity

Build commands (all in the isolated tree):

```bash
# DBoW2, g2o, Sophus: cmake -DCMAKE_BUILD_TYPE=Release && cmake --build . -j8
# ORB-SLAM3:
cmake -S ORB-SLAM3 -B ORB-SLAM3/build -DCMAKE_BUILD_TYPE=Release
cmake --build ORB-SLAM3/build -j8
# Photo-SLAM-L:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
```

Key SHA-256 values (full list in `_lock/artefact_hashes_final.txt`):

| artifact | SHA-256 |
|---|---|
| `bin/replica_mono` | `1393a4386f09c4384f908dcda83a8550580241bf8bca16079d4b6e4189f5c643` |
| `bin/replica_rgbd` | `6731676589095cb371d99fbd40745346a91a0e74510726839440ff5b84e4e0e1` |
| `bin/tum_mono` | `62030ae53d644e31b4dc141ce6e77515d83e083e22e57f0e8497f1433340292b` |
| `bin/tum_rgbd` | `d33717f248ada712efa0d0454716c8f54d42f96970634d1157fcf368195a2c2b` |
| `bin/eth3d_rgbd` | `3dea3ec2c57414ec305668069b70c0f1d916faa13703b56ed2de496d6ca069c6` |
| `bin/eth3d_mono` | `e471184e381b22d4653c4e6bad878a1feb09c1432536b6f274d8480eb14d0971` |
| `lib/libgaussian_mapper.so` | `98cc87928caa592823650048aa78d1940f36cc913db3d02ec7534566aa5a904a` |
| `ORB-SLAM3/lib/libORB_SLAM3.so` | `017b659e0ea94370eeac6365bd9f8305887fbdf22a2513bf7095eb1a6593cd31` |
| `ORB-SLAM3/Thirdparty/g2o/lib/libg2o.so` | `e8f7d966cf616a20450932c8e29d8d0d46b8f3d110564a425ee6b54bd8f35716` |
| `ORB-SLAM3/Thirdparty/DBoW2/lib/libDBoW2.so` | `7bb660badc49dfd5c43631fc7482cbbb6686bc013586b3a8bb6ecd98d688da0f` |

`ldd` on the entries resolves `libORB_SLAM3.so`, `libg2o.so`, `libDBoW2.so`,
`libgaussian_mapper.so`, `libgaussian_viewer.so`, `libimgui.so`,
`libsimple_knn.so`, `libcuda_rasterizer.so` to
`/workspace/code/PSL-fullmap-lock-v1/...`; no `*_asan` or old `Photo-SLAM-L`
library appears. Torch resolves to the system
`/usr/local/lib/python3.10/dist-packages/torch/lib`.

## 5. ETH3D calibration, depth scale and Gaussian config (audited)

* Intrinsics in all ETH3D ORB configs: `fx=fy=726.28741455078`,
  `cx=354.6496887207`, `cy=186.46566772461`, `739x458`; matches each
  sequence's `calibration.txt`.
* `RGBD.DepthMapFactor: 5000.0` in all 11 official ETH3D RGB-D configs. This is
  the authoritative value: the upstream Photo-SLAM configs all use 5000, and
  the raw depth PNG is 16-bit with `max=18721` on `desk_3`.
  The earlier manifest line claiming "depth factor 1000 (mm)" was retracted by
  commit `047ebd9223`; it was a generated-config error, never executed.
  No ETH3D depth factor was changed in this version.
* ETH3D RGB-D uses **per-scene** Gaussian YAMLs
  (`cfg/gaussian_mapper/RGB-D/ETH3D/eth3d_<scene>.yaml`).
* TUM RGB-D uses the generic `tum_rgbd.yaml`; TUM mono uses `tum_mono.yaml`;
  Replica mono uses `replica_mono.yaml`.
* All Gaussian YAMLs used by the manifest contain the three required
  `Optimization.weight_line_*` keys (presence checked by the manifest builder).

## 6. Full vs OFF: one binary, one behaviour, one weight vector

Same binary, same ORB config, same sequence/association, same seed, same
`PHOTO_SLAM_LINE_MODE=2`. The only difference is the Gaussian YAML:

| condition | Gaussian YAML | effective weights |
|---|---|---|
| Full | official YAML | `coherence`, `shape_ecc`, `shape_ori` = official values (0.001/0.002/0.005; TUM generic uses 0.005/0.002/0.005) |
| OFF | generated copy | all three = `0.0` |

`make_off_cfg.py` rewrites only those three values and the manifest builder
verifies the diff is exactly six +/- lines (three keys). Code gate
(`gaussian_mapper_line.cpp`):

```cpp
if (lineLossEvery() > 0 && (w_ori > 0.0f || w_shape_ecc > 0.0f) && ...)
```

so with all three zero the whole `L_str` block is skipped. No other switch is
set; `useDirectionalLineDensify()` defaults to **true** when
`PHOTO_SLAM_ABL_DIRECTIONAL_DENSIFY` is unset, so directional line densification
is identical in both conditions.

Observed evidence (small samples, `PHOTO_SLAM_DEBUG_LINE_LOSS=1` only there):

| run | `[LINE-LOSS]` lines |
|---|---|
| Replica mono office0, Full | 1094 |
| Replica mono office0, OFF | 0 |
| TUM RGB-D fr3, Full | 3244 |
| TUM RGB-D fr3, OFF | 0 |

## 7. Small-sample acceptance (before locking)

Every accepted run: `exit=0`, `Total Iterations > 0`, non-empty `psnr.txt`,
`point_cloud.ply` with `element vertex > 0`.

| cell | cond | iterations | mean PSNR | PLY vertices | PLY size | accept |
|---|---|---:|---:|---:|---:|---|
| Replica mono office0, 200f | Full | 547 | 30.0319 | 8618 | 2.1 MB | YES |
| Replica mono office0, 200f | OFF | 661 | 30.3924 | 8612 | 2.1 MB | YES |
| TUM RGB-D fr3 long office, 200f | Full | 1702 | 26.0093 | 12697 | 3.2 MB | YES |
| TUM RGB-D fr3 long office, 200f | OFF | 1820 | 26.5510 | 12831 | 3.2 MB | YES |
| ETH3D RGB-D desk_3, 200f | Full | 11210 | 25.4117 | 33735 | 8.4 MB | YES |
| ETH3D mono table_7, 600f | Full | 13098 | 27.2753 | 22052 | 5.5 MB | YES |

The 200-frame ETH3D **mono** run reaches only ~21 KFs while its Gaussian config
requires `Mapper.min_num_initial_map_kfs: 28`. It therefore never starts
mapping; with the new guard it exits 0, reports `Gaussian model is empty`, and
produces no PLY. That case is correctly **not** accepted. The 600-frame view
reaches the threshold. The formal runner marks any 0-iteration cell invalid.

These 200-frame numbers are acceptance checks only, not effect estimates.

## 8. Formal batch manifest and run protocol

Manifest: `PSL-fullmap-exp-v1/manifest.csv` (30 cells, full sequences).

| family | modality | cells |
|---|---|---|
| Replica | mono | 8 (`office0..4`, `room0..2`), 2000 frames each |
| TUM | RGB-D | 3 (`fr1_desk`, `fr2_xyz`, `fr3_long_office_household`) |
| TUM | mono | 3 (same sequences) |
| ETH3D | RGB-D | 11 (all official scenes: desk_3, mannequin_1/3, planar_2/3, plant_scene_1, sfm_bench, sfm_garden, sfm_house_loop, sfm_lab_room_1, table_7) |
| ETH3D | mono | 5 (desk_3, planar_2, planar_3, sfm_garden, table_7) |

5 repeats x 2 conditions x 30 cells = **300 runs**, serial.

Runtime switches (no diagnostic switches):

```bash
PHOTO_SLAM_LINE_MODE=2
PHOTO_SLAM_LINE_LOOP=0
PHOTO_SLAM_SHADOW_LINE=0
PHOTO_SLAM_SLAM_ONLY=0
PHOTO_SLAM_NO_PACING=0
PHOTO_SLAM_SEED=<rep-1>
LD_LIBRARY_PATH=<locked>/ORB-SLAM3/lib:<locked>/ORB-SLAM3/Thirdparty/g2o/lib:<locked>/ORB-SLAM3/Thirdparty/DBoW2/lib:<locked>/lib
```

Repeats are the outer loop; condition order alternates per repeat
(odd `full,off`; even `off,full`). Full and OFF for a cell are adjacent.
Output: `runs/formal-v1/<cell>/r<rep>_<cond>/` with `stdout.log`,
`run.meta.json`, `status.json`, the exact used YAMLs, trajectory and PLY.
Per-run summary: `runs/formal-v1/summary.csv`. The batch aborts if any binary or
library hash changes. Completed runs are skipped on resume; a non-empty
incomplete directory is never overwritten.

Launch:

```bash
nohup env REPEATS=5 RUN_TIMEOUT=7200 \
  python3 /workspace/code/PSL-fullmap-exp-v1/scripts/formal_batch.py \
  > /workspace/code/PSL-fullmap-exp-v1/logs/formal_batch.log 2>&1 &
```

## 9. Batch launch record

* launched: `2026-11-05T19:56Z`
* detached PID: `2740886` (also in `logs/formal_batch.pid`)
* log: `logs/formal_batch.log`
* progress: `runs/formal-v1/summary.csv` (one row per finished run)
* resume: re-run the same command; completed `status.json` rows with
  `accept=true` are skipped, non-empty incomplete directories are not
  overwritten.

## 10. Rules carried into the batch

* No binary or library is replaced while the batch runs.
* Any code fix requires a new source tree, a new build and a new output root;
  it must not be mixed into `formal-v1`.
* The official Photo-SLAM baseline is pinned **separately** at
  `f8bfb2f0809c003ccc3fd577dc43c576fcafa4ac`
  (`https://github.com/HuajianUP/Photo-SLAM.git`, 2024-06-12) in
  `/workspace/code/PSL-upstream-official`. It is not part of this manifest and
  must not be compared until its own build and hashes are recorded.
  Its build currently fails against the installed PyTorch 2.12
  (`c10::guts::to_string` removed; `torch::cat(..., int)` signature), after a
  baseline-only `#include <cfloat>` fix for CUDA 12.x. See
  `PSL-upstream-official/_baseline/BASELINE_RECORD.md`. No baseline numbers are
  claimed in this round.
