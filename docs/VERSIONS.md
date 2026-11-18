# Version inventory — Photo-SLAM-L / PhotoSlamLine

Snapshot date: **2026-11-18**. Source of truth: `git@github.com:DaoguiZhang/PhotoSlamLine.git`.

This file records *what every version on the development machine is for*, so that
the branches and the on-disk experiment snapshots can be told apart later.

Regenerate the machine-readable part of this inventory with

```bash
scripts/version_inventory.sh            # branches, push status, worktrees, snapshots
```

## 0. What is versioned and what is not

Versioned (uploaded):

* `src/`, `include/`, `cuda_rasterizer/`, `viewer/`, `examples/`, `tests/`
* `ORB-SLAM3/{include,src,tests}` (the vendored ORB-SLAM3 fork used by this project)
* `cfg/` (ORB-SLAM3 and mapper YAMLs), `scripts/`, `docs/`, launcher shell scripts
* small text notes (`readme.txt`, `todolist.txt`)

Never uploaded:

* experimental results — `results/`, `evidence/`, `myresult/`, `Temp/`, `*-results/`,
  `*.ply`, metric dumps (`*_Stats.txt`, `ExecMean.txt`, `LBA_Stats.txt`, ...)
* build output — `build/`, `ORB-SLAM3/build/`, `bin/`, `lib/`, `CMakeCache.txt`,
  `CMakeFiles/`, `*.so`, `*.o`, `compile_commands.json`
* datasets and vocabularies (`ORBvoc.txt`, Replica / TUM / ETH3D / Rover data)
* backups and snapshots of the working tree (`Photo-SLAM-L_backups`, `PSL-*-results`)

`.gitignore` already enforces most of this; the check is
`git status --porcelain | grep -E '\.(ply|png|pth|so|o)$'` → must stay empty.

## 1. The repository at a glance

| item | count |
|---|---|
| local branches | 25 |
| remote branches on `origin` | 33 |
| local tags | 7 (3 of them not on GitHub, see §2) |
| git worktrees on this machine | 19 |
| tracked files | 724 |
| experiment directories under `/workspace/code` | 100 (87 named `PSL-*`) |

The project is a **single** repository with many worktrees — there is no second
copy of the source under version control. Everything else on disk is either a
worktree, a build tree, a results folder, or an untracked source snapshot (§4).

## 2. Branches that are NOT on GitHub (as of this commit)

| branch | commits not on any remote | purpose |
|---|---|---|
| `feat/point-assist-rescue-modes` | 5 | **Main line of the paper.** POINT / ASSIST / ASSIST_RESCUE tracking modes with an independent acceptance test, per-frame four-level line supply-chain diagnostics, point-mask-before-count`fix`, joint point+line solve with post-solve check, and verification of the solved candidate on a disjoint held-out set. Local tag `rescue-experiment-sealed-20261112` points here. |
| `fix/pose-opt-line-residual` | 6 | Superset of the branch above **plus** the pose-optimisation fix: line outliers are judged on the residual at the *current* pose instead of the stale one (`ORB-SLAM3/src/Optimizer.cc`, `LineMode.h`, `Tracking.*`). |
| `fix/pose-residual-refresh` | 1 | The same optimizer fix re-applied on the archived base (`archive/pose-residual-20261113`), used to compare “fix only” against “fix + rescue modes”. |
| `fix/line-reproj-triangulate-po-reeval` | 1 | Eval-only change: merged settlement root, merge-boundary audit and a staged preview report for the re-evaluated line-reprojection / triangulation runs (`eval/build_merged_root.py`, `eval/report_preview.py`). |

Tags that exist locally but not on GitHub: `rescue-experiment-sealed-20261112`,
`d6-fix-backup-20260919`, `pre-mapper-ablation-feat-eth3d-rover-line`.

None of these commits touch results, build output or datasets; they only change
`src/`, `include/`, `ORB-SLAM3/`, `cfg/`, `scripts/`, `tests/` and `eval/` scripts.

## 3. Purpose of every branch (all branches, in chronological order)

| branch | last commit | uploaded | purpose |
|---|---|---|---|
| `photoslaml_5090` | 2026-06-16 | yes | Initial Photo-SLAM-L import (line features on top of Photo-SLAM) as it was first run on the RTX 5090 machine. |
| `feat/stereo-line-pipeline` | 2026-09-10 | yes | Stereo line pipeline (WIP, runtime unverified at the time). |
| `fix/mono-single-image` | 2026-09-10 | yes | Monocular line pipeline + Gaussian-mapping integration fix. |
| `feat/mono-line-loop-closing` | 2026-09-14 | yes | Line-aware loop closing, activated with an A/B/C switch. |
| `exp/mono-office2-diagnosis` | 2026-09-14 | yes | Monocular office2 diagnosis; adds reusable eval and no-overwrite run scripts. |
| `feat/replica-d-verify` | 2026-09-14 | yes | Loop verification fixes: `pMostBoWMatchesKF` update, float Sim3 thresholds, `fr1_room` config. |
| `fix/line-mode-loop-detection` | 2026-09-15 | yes | Guard against degenerate rotation in `ComputeSim3` (SO3::exp NaN crash). |
| `fix/mono-office1-stability` | 2026-09-16 | yes | office1 reproduction, regression and sanitizer runners. |
| `fix/mono-office1-pose-stability` | 2026-09-18 | yes | Line-projection stage counters + writeback / fusion / Sim3-refine unit tests. |
| `feat/line-loop-gaussian-sync` | 2026-09-19 | yes | Line-Gaussian synchronisation and training-crash mitigation. |
| `feat/eth3d-rover-line` | 2026-09-21 | yes | ETH3D / Rover line entries; `REGISTER_TIMES` no longer prints `-nan` on empty stat vectors. |
| `fix/d6-line-gaussian-crash` | 2026-09-21 | yes | Same `REGISTER_TIMES`/`-nan` fix, kept as the narrower bug-fix branch. |
| `fix/line-loop-real` | 2026-09-21 | yes | Line loop branch used for the “real” loop-closing rerun. |
| `perf/compute-latency` | 2026-09-21 | yes | Densify profiling split (sync / maxR / gradAccum / densify) and sampling-alignment fix. |
| `exp/mapper-frozen-ablation` | 2026-10-12 | yes (via archive) | Multi-scene RGB-D + Mono **frozen-geometry mapper** ablation. |
| `feat/fixed-geometry-mapper-ablation` | 2026-10-15 | yes (via archive) | A6-violation classification from a run's own artifacts; ablation bookkeeping. |
| `perf/mapper-min-cost` | 2026-10-16 | yes (via archive) | Mapper minimum-cost round: the audit switch costs one branch per creation call. |
| `feat/mapper-budget-v1` | 2026-10-19 | yes (via archive) | Mapper budget v1 + default-off shutdown single-frame MSE/PSNR diagnostic. |
| `eval/cross-dataset-frontend` | 2026-11-05 | yes | Paper tables, English results text, cabinet coverage audit (frontend side). |
| `fix/line-reproj-triangulate-po-reeval` | 2026-11-05 | **no** | Eval staging/report scripts for the merged settlement root (see §2). |
| `fix/line-rescue-tracking` | 2026-11-10 | yes | Size `mvpMapLines`, validate rounded line endpoints (line rescue tracking baseline). |
| `feat/point-assist-rescue-modes` | 2026-11-12 | **no** | POINT / ASSIST / ASSIST_RESCUE modes + acceptance test (see §2). Current branch. |
| `fix/pose-opt-line-residual` | 2026-11-12 | **no** | Pose-opt line-outlier fix on the current-pose residual (see §2). |
| `fix/pose-residual-refresh` | 2026-11-12 | **no** | Same fix, refreshed on the archived base (see §2). |
| `archive/pose-residual-20261113` | 2026-11-12 | yes | Archive of the pose-residual experiments and the release decision that followed. |

## 4. Source snapshots on disk that are NOT version-controlled

These are working copies (source + build tree) created for individual experiment
rounds. They contain unique `cfg/`, `scripts/` and diagnostics, but no `.git`, so
their diffs are **not** on GitHub. Kept on disk only; candidates for a one-time
“snapshot commit” if their configs are ever needed again.

| directory | size | purpose |
|---|---|---|
| `PSL-formal-main-20261108` | 28 GB | Formal paper-run tree (main); configs for the sealed formal batch. |
| `PSL-pml-ablation` | 28 GB | Point-mapper-loss ablation tree. |
| `PSL-verify` | 19 GB | Verification round (bisect / head2 / new / old siblings below). |
| `PSL-fullmap-exp-v1` | 17 GB | Full-map lock experiment (v1). |
| `PSL-alpha-full-20261112` | 15 GB | α (line weight) sweep, full grid. |
| `PSL-abl-20261110` | 13 GB | First ablation round tree (stage / stage2 / po / lbawin / fr1split siblings). |
| `PSL-alpha-exp-20261113` | 4.7 GB | α = 0.5 decision runs (`A05`), source of `ALPHA_WEIGHT.patch`. |
| `PSL-upstream-official` | 361 MB | Clone of upstream `HuajianUP/Photo-SLAM` (separate git repo, baseline reference). |
| `PSL-final`, `PSL-replay` | 400 MB each | Release/replay trees used to regenerate the shipped tables. |
| `PSL-fullmap-lock-v1` | 394 MB | Full-map lock v1; **holds the `cfg/` files that `PLAN_E2E.csv` points at**. |
| `PSL-alinexp-20261112` | 310 MB | α experiment patch source (`ALPHA_WEIGHT.patch`). |
| `PSL-pslpre-20261112`, `PSL-pslpost-20261112` | ~310 MB each | Pre/post snapshots around the pose-residual-refresh fix. |
| `PSL-lavn`, `PSL-lbadiag`, `PSL-dedupfix`, `PSL-dedupfix-base`, `PSL-camdiag`, `PSL-dirfix` | 24–380 MB | Diagnostic/fix snapshots (line-averaging, LBA diagnostics, dedup, camera, directory-handling fixes). |
| `PSL-initb`, `PSL-initdiag`, `PSL-sticky-20261112`, `PSL-mech-20261112` etc. | 0.2–2.6 GB | Initialisation, sticky/mechanism diagnostics rounds (results-heavy). |
| `PSL-upstream-old-20261026` | 12 MB | Older upstream copy (pre-line baseline). |

Not worth uploading (pure output or backups): `Photo-SLAM-L_backups` (18 GB),
`Photo-slam-l-readme` (38 GB), `PSL-*-results` (8 directories, ~60 GB),
`PSL-lineband-*`, `PSL-densify-formal-results`, `PSL-fgm-formal-results`,
`PSL-initdiag`, `PSL-diag2`, `PSL-statfix`, `PSL-finalstate`, plus every `build/`
tree (`PSL-camdiag-build`, `PSL-dirfix-build`, `PSL-pml-ablation-build`).

## 5. Worktrees on this machine

`scripts/version_inventory.sh` prints the live table (worktree path → branch/commit).
The named worktrees are used as *read-only experiment bases*: each round checks out
a detached HEAD at a fixed commit so that a long run cannot be disturbed by later
edits. Dirty file counts in those worktrees are expected (they carry run outputs
that are ignored/untracked).

## 6. Commit convention

Prefixes used in this repository, so a new commit can be placed consistently:

| prefix | meaning |
|---|---|
| `feat(scope):` | new capability (tracking mode, loop stage, mapper feature) |
| `fix(scope):` | behaviour fix (NaN guard, outlier rule, dedup, directory handling) |
| `perf(scope):` | performance round, with the measured cost stated in the body |
| `diag(scope):` | diagnostics / instrumentation only, no behaviour change |
| `eval:` | evaluation, tables, report scripts |
| `docs:` | documentation, archives, release decisions |
| `test(scope):` | unit / regression / sanitizer runners and repros |

Bodies should state **what was measured** (scene, repetitions, metric) and, when a
change is rejected, **why**. Results themselves stay out of the repository.
