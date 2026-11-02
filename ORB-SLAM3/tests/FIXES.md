# Line/point tracking fixes — what changed and what is verified

This branch contains **implementation fixes only**. It adds no new algorithm, no
tuning, no ablation switch, and no change to thresholds, covisibility, the
initialisation path, rescue, or the Gaussian backend.

## Fixes

### 1. `LocalMapping::CheckLineReprojection` ignored the keyframe pose
`GeometricCamera::project()` is `K * (X/Z, Y/Z)` with **no pose**, but both live
callers pass **world** points (`KeyFrame::UnprojectDepthLine` and the mono
fallback that builds `S3D/E3D` around `Triangulate`'s world output). Every
endpoint was therefore additionally translated by the camera centre. The pose is
now applied exactly once, inside the function, together with non-finite and
`z <= 0` guards and an explicit degeneracy rejection.
Thresholds are unchanged (15 px, cos 20 deg, 1 px).

### 2. Mono fallback passed a unit ray to `GeometricTools::Triangulate`
`Triangulate` reads **only** `x_c(0)` and `x_c(1)`, i.e. it needs normalized
plane coordinates `(u, v, *)`. The mono fallback called `.normalize()` first,
which is not equivalent (only the first term of each DLT row is scaled).
Measured with the real implementation at 3.7 m: `x1` → 4.7e-07,
`x(1/L)` unit ray → 5.1e-03. The `normalize()` was removed at that call site;
the point path already did the right thing. The global semantics of
`Triangulate` were **not** changed.

### 3. `Optimizer::PoseOptimizationWithLine` never refreshed demoted edges
g2o does not update an edge that was demoted with `setLevel(1)`, so `chi2()`
returns a **stale** error. `PoseOptimization` re-computes it explicitly for
exactly that reason; the line-aware copy did not, on either the point edges
(`mvbOutlier`) or the line edges (`mvbLineOutlier`). Both sides now call
`computeError()` for edges that were outliers in the previous iteration.
Proven on a real captured state: 38 → 39 inliers, pose changes.

Two compile-time switches exist so an A/B can be built from identical sources:
`-DPO_LINE_NO_OUTLIER_REEVAL` (point side) and
`-DPO_LINE_NO_LINE_OUTLIER_REEVAL` (line side). Both are OFF by default.

### 4. Diagnostics and record protection
Per-frame tracking CSV (stage, map id, live pose), reset call sites in map
events, a line-edge diagnostic row emitted *before* the line gate so a gated-off
arm is measurable, a MapLine lifecycle ledger keyed by `(mapId, lineId)`, and a
pose-optimisation state capture whose **path is validated before the run** and
whose write failures are counted and reported (`[PO-CAPTURE-SUMMARY] ...
status=FAIL`). A capture can no longer fail silently.

## Tests

`tests/run_cpu_tests.sh` builds and runs everything; no dataset is required.

| test | covers |
|---|---|
| `test_line_reproj_geometry.cpp` | Tcw transform, depth/non-finite/degenerate guards, common-world-transform invariance |
| `test_line_triangulate.cpp` | real `Triangulate`: plane coords vs unit ray vs arbitrary scale, no depth/degeneracy/finiteness guard (recorded as absence) |
| `test_pose_opt_outlier_reeval.cpp` | g2o mechanism: stale chi2 5034.379 vs recomputed 0.025, decision flip |
| `test_pose_opt_real.cpp` | real optimisers: return value == Frame flags, persistent outlier/inlier, empty frame, zero-line parity |
| `test_po_capture_paths.cpp` | missing parent, not-a-directory, unwritable, no-overwrite |

## Known limitations

* `vpEdgesMono_FHR` has no outlier loop at all in `PoseOptimizationWithLine`.
  It is only built when `pFrame->mpCamera2 != nullptr`, i.e. unreachable for
  pure monocular. Not fixed here; do not assume it is supported.
* Line edges still have no reference implementation to compare against; the
  added re-evaluation is the same mechanism as the point side, not a claim of
  parity with a line-specific design.
* `fail_stage` is not recorded on the line-entry reference-keyframe path.
* After these fixes the tracker still loses track a few frames after
  initialisation on `fr3_structure_notexture_near`, on the native
  `mnMatchesInliers >= 30` condition with only ~65 local map points. That is a
  legitimate native rejection, not something these fixes address.

## Licensing / provenance

Derived from ORB-SLAM3 (GPLv3, see `LICENSE`) as vendored in this repository.
The vendored `Thirdparty/` trees are unchanged.
