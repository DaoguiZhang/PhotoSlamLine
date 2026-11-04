# Frozen evaluation version (v_final_1)

Frozen at 2026-11-04, before the first formal run of the 492-row manifest. Every
identifier below is a hash of a file that exists on disk; an identifier that is
not listed here is not part of the frozen version.

## 1. Binaries (flag-consistent unified Release build)

Built by `scripts/build_unified.sh`, `build_rc=0`, from the sources in section 2.
Both trees (Release and ASan) use the same include paths, the same defines and
the same g2o; they differ only in optimisation flags and sanitizer. The in-tree
`bin/eval_frontend` and `ORB-SLAM3/lib/libORB_SLAM3.so` are **not** the frozen
artefacts (built earlier with a different flag set).

| role | path | sha256 |
|---|---|---|
| entry (all runs) | `/workspace/eval_builds/v1/entry_rel/eval_frontend` | `60d513316619e5d68de5bca276617b4ef455d575de330faef1b55570dad95977` |
| SLAM library | `/workspace/eval_builds/v1/lib_rel/libORB_SLAM3.so` | `d33731eb233da9078a7896e68801dd188da69f60c35f98fcd41fd1fdab9ee860` |
| g2o | `/workspace/eval_builds/v1/g2o_rel/lib/libg2o.so` | `e8f7d966cf616a20450932c8e29d8d0d46b8f3d110564a425ee6b54bd8f35716` |
| DBoW2 (shared) | `ORB-SLAM3/Thirdparty/DBoW2/lib/libDBoW2.so` | `7bb660badc49dfd5c43631fc7482cbbb6686bc013586b3a8bb6ecd98d688da0f` |
| vocabulary | `ORB-SLAM3/Vocabulary/ORBvoc.txt` | `f8dd027f7a6cb88129821341194d7f2c75b77b3394257ddd0d2229863d1a3570` |

ASan counterparts exist for defect attribution
(`lib_asan 3b93c8b42287ae3b46999911ee65b566b6ecb05f2788c5701ef8bebd340523c9`,
`g2o_asan 770a184c992d0a362b8e5a2c1feffb570e58c66e21fd3e48d21af6aa76b28462`,
`entry_asan a69e6116e74cf8941b4d6ed1966956b7ef42a9874b02756cd3397d6f1e4a5014`).
**No performance or accuracy number in this report comes from an ASan build.**

Loading is verified per invocation with `ldd` (`libORB_SLAM3.so` resolves to
`lib_rel`, `libg2o.so` to `g2o_rel`). A hash in a manifest is not evidence of
what was loaded; the `ldd` output is.

## 2. Sources

| file | sha256 |
|---|---|
| `examples/eval_frontend.cpp` | `fa4df99260124005238cf696784101175b2559573115263a7bca948d10b41a4b` |
| `ORB-SLAM3/src/Optimizer.cc` | `13f2a8c7c5dfd54c5f4c51f22feddc39c8c6ffc86cf31981f3dfcb9f95d34d41` |
| `ORB-SLAM3/src/OptimizableTypes.cpp` | `ea1ac8a08e56a7809bb298825c461dc3b3fe2e47e2d0e890f6545159dc7be7ea` |
| `ORB-SLAM3/src/LineExtractor.cc` | `02af020d1332242f1620de855467a98743a5f5d1686b48e69959046057c6b8e0` |
| `ORB-SLAM3/include/OptimizerVertexIds.h` | `c0707ab796ca7633ca33fec4dce7fd08309c3b32dd8be38c17b20483e10cce3c` |

Delivery branch `eval/cross-dataset-frontend` = `e297ee639a3ad3c52074b19bf38ef1e560e2d1ae`.
The fix branch `fix/line-reproj-triangulate-po-reeval` stays at
`d32141947b4b882a1761455bacf80aaf0dfe411a` and is untouched.

If any of these source hashes changes, the build is no longer this version: it
must be rebuilt, re-verified (including the `ldd` binding) and re-hashed, and the
old hash must not be quoted for the new runs.

## 3. Configuration, inputs and ground truth

* configs: `cfg/` subtree (163 files) hashes to
  `86e96fc99bec4c8b45c98f2d2385c1c6c4edcf077f9c41b9301421d4968d9e71`.
* inputs: absolute dataset paths recorded per row in the manifest
  (`/workspace/code/SEGS-SLAM/datasets/{tum,replica,eth3d}/...`).
* TUM/ETH3D GT: the dataset's own `groundtruth.txt` (TUM samples ~100 Hz; the
  evaluator interpolates in time, never index).
* Replica GT: derived by `eval/derive_replica_gt.py` from the raw `traj.txt`
  under the rule $R_{cv}=R_{c2w}\mathrm{diag}(1,-1,-1)$, $t=i/30$ s. Rule,
  verification and per-scene hashes are in `eval/REPLICA_GT.md`; all derived
  files hash to `4e81b2bd1342d697b077a222fb9c8bad23c5e3b3fa7f397b0e1c3a58cc92dca3`.

## 4. Evaluator

`/workspace/code/PSL-perf/scripts/eval_tum_se3_v2.py`,
sha256 `199862bbeb8862083dd1d2eb62a5043641c5e426c132b2493a18c8ee04b3e3ac`.
Called positionally `<gt> <est>` with `--mode sim3` for mono and `se3` for RGB-D,
`--method interp --max-dt 0.02`. Estimate-centric. Coverage/continuous-tracking
metrics come from `eval/coverage.py`, which shells out to that same evaluator and
uses **all GT-evaluable input frames** as denominator
(`eval/test_coverage.py` T1-T10 all pass; T10 pins "10 inputs, 4 correct => 0.4").

## 5. Environment

Ubuntu 22.04.5, g++ 11.4.0-1ubuntu1~22.04.3, Eigen 3.4.0
(`/usr/include/eigen3`), OpenCV 4.10 (`/usr/local/lib`), Python 3.10.12,
24 cores, NVIDIA GeForce RTX 5090 / driver 595.99.02, CUDA 12.8 (V12.8.61).
Compiler flags: Release `-O3 -DNDEBUG -fPIC -w -std=c++17 -pthread` with
`-DCOMPILEDWITHC11 -DORB_SLAM3_EXPORTS -DREGISTER_TIMES -D_GLIBCXX_USE_CXX11_ABI=1`.

## 6. Manifests and run layout

* manifest: `/workspace/eval_runs/final_runs/MASTER_MANIFEST.csv`,
  492 rows, sha256
  `3534f6d0e75e760709260aa9267462d2317ae2431d748f1776cc3978ca4afcc0`.
  486 internal cells (54 dataset/scene/modality groups x 3 arms x 3 repeats) and
  6 upstream control cells.
* run root: `/workspace/eval_runs/final_runs` -- a new directory, so no
  `status.txt` from the historical 473-item tree or from the three contract runs
  can make the scheduler treat a cell as already done. The historical roots
  (`v2/`, `cabinet/`, `cabinet2/`, `edges/`, `edges2/`) are evidence only and are
  excluded from every statistic in this report.
* execution order: grouped by (dataset, scene, modality); inside a group the 9
  arm x repeat cells are walked in an order rotated left by (group index mod 9).
  Pre-declared, deterministic, and independent of any result.
* environment per run: `EVAL_ARM`, `PHOTO_SLAM_LINE_MODE` (0/1/2, consistency
  enforced by the entry), `PHOTO_SLAM_PO_LINE=1`, `PHOTO_SLAM_LBA_LINE=1`,
  `PHOTO_SLAM_LINE_RESCUE=0`, `PHOTO_SLAM_SLAM_ONLY=1`,
  `PHOTO_SLAM_DEBUG_LINE_EDGES=1`.

## 7. Timing caveat (pre-registered)

`PHOTO_SLAM_DEBUG_LINE_EDGES=1` is kept on for every run so that the line-edge
counts in `eval/EDGE_COUNTS.md` remain verifiable per run. All wall-clock and
`[EVAL-TIMING]` numbers in this report are therefore **measurements under
diagnostic logging**, not "logging off" numbers, and they are labelled as such
wherever they appear. The entry also paces each frame to the sequence's own
timestamp interval unless `PHOTO_SLAM_NO_PACING=1`; pacing is on (unset), so
wall time is dominated by the sequence duration and the compute cost must be read
from the `track_ms` / `drain_ms` buckets, not from wall time.

## 8. Upstream control (6 rows): prepared, not substituted

Upstream Photo-SLAM is cloned at the official commit
`f8bfb2f0809c003ccc3fd577dc43c576fcafa4ac` (2024-06-12, `HuajianUP/Photo-SLAM`)
into `/workspace/upstream/Photo-SLAM`; it has no submodules. It needs libtorch,
CUDA, jsoncpp, glm, glfw3, OpenGL in addition to the ORB-SLAM3 stack, and its
`cuda_rasterizer` is pinned to `CUDA_ARCHITECTURES "75;86"` while this machine is
an RTX 5090 (sm_120), so the arch list has to be adapted and recorded as a build
adaptation. Its 6 rows stay `BLOCKED` in the manifest with an explicit reason.
T0 of this fork is **not** accepted as a substitute for upstream.

Ordering decision: the upstream build is scheduled **after** the internal 486
cells, because compiling with -j on the same 24 cores would inflate the
`track_ms`/wall measurements that are themselves deliverables. This is a
sequencing choice, not a substitution, and it is why the 6 rows can still be
`BLOCKED` while the internal matrix is complete.
