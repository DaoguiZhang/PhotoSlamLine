# Pose-residual experiments — NOT FOR RELEASE

**Status: archived, negative result. Do not merge, do not ship.**

This directory holds the *record* of an investigation into a real defect in
`Optimizer::PoseOptimizationWithLine()` and of two candidate fixes. **Neither
candidate passed release acceptance.** The patches here are archive material:
they are intentionally *not* applied to this branch, and this branch changes no
source code.

## What was investigated

`PoseOptimizationWithLine()` read `chi2()` straight out of the g2o edge cache.
Edges parked at level 1 are excluded from `initializeOptimization(0)` /
`optimize()`, so their `_error` is never refreshed: a line (or point) that the
current pose had already pulled back inside the threshold stayed an outlier for
the remaining rounds and never re-entered the graph. Non-finite residuals were
also classified as inliers, because `nan > threshold` is false.

## What was confirmed

- The stale-residual misjudgement is **real**. A unit regression over the real
  edge classes and a real `g2o::SparseOptimizer` showed the old code reading
  `chi2 = 1200.332` where the residual at the final pose was `0.000`; a NaN
  residual was kept as an inlier.
- Fixing it changes behaviour: on `TUM fr1/desk` monocular the fixed build
  re-admits roughly 3.9 edges per frame that the old build keeps parked.

## What failed

| Candidate | Outcome |
|---|---|
| FULL (refresh all four judging loops + reject non-finite) | **Rejected.** On `TUM fr1/desk` monocular, three paired repeats gave a window-translation error above the pre-registered noise band in 3/3 pairs and a Sim(3) ATE median 2.9× the baseline. One repeat produced a trajectory only 37 % of the ground-truth path length. |
| LINE-STICKY (refresh all four, points may re-enter, a rejected line stays out for the rest of the call) | **Rejected.** Mechanism checks all passed, but in the formal regression a repeat **lost tracking** (`rc = 0`, 92 × `Fail to track local map`, a second map created, 595/610 poses exported) and Sim(3) ATE reached 20× the baseline in that pair. |
| RGB-D cross-check (`ETH3D sfm_lab_room_1`) | **Unusable as a control.** The *baseline* itself failed the pre-registered gate in 3/3 repeats (two maps, two relocalisations, alignment scale ≈ 0.68). No claim is made about RGB-D. |

The stale-residual misjudgement being real does **not** imply that the fixes are
releasable, and a mechanism-level check passing does **not** imply the fix is
safe. No process crashed in any of these runs: the failures are trajectory /
tracking failures at exit code 0.

## Mechanism diagnosis (why the fixes were not pursued further)

A same-state replay was built: every frame the production
`PoseOptimizationWithLine()` is executed once with the selected policy, then the
frame pose and both outlier arrays are restored and the function is re-run under
the other policies on the **identical** map, initial pose, correspondences and
solver configuration. Over 609 frames:

| policy | point edges re-admitted | line edges re-admitted | non-finite rejections |
|---|---|---|---|
| baseline (no refresh) | 0 | 0 | 0 (guard absent) |
| refresh points only | 2381 | 0 | 0 |
| refresh lines only | 0 | 1 | 0 |
| refresh all (FULL) | 2381 | 1 | 0 |
| refresh all, lines sticky | 2381 | 0 | 0 |

The first pose divergence from the baseline appears at frame 4 (4.5e-5 m) and is
identical for the point-only, FULL and sticky policies; refreshing lines alone
does not diverge until frame 26. **The behavioural difference is carried almost
entirely by point-edge re-admission**, and the non-finite guard never fired on
this sequence. Why point re-admission should worsen monocular scale drift is
**not established** — the end-to-end root cause remains unproven.

## What is not in this directory

Source changes, build trees, build logs, per-run specifications, stdout logs,
trajectories, binaries and the release-decision record are **not published**
here. They remain local to the experiment machines. Only the three patches, this
README, the index and `SHA256SUMS` are public.

## Using PRE versus M0

Both are working configurations; they answer different questions.

| | PRE (baseline pose residuals) | M0 (point-only) |
|---|---|---|
| Line residuals in pose optimisation | yes | none (no line edges are built) |
| Suited to | reproducing the current published behaviour, line-aware experiments | when a stable pose is the priority and line features are not needed |
| Monocular caveat | on `TUM fr1/desk` the monocular configuration frequently lands in a scale-drift mode (ATE 0.10–0.22 m); this is pre-existing, not caused by the patches here | trajectory and coverage were stable in the archived runs |
| ABI note | application and `libORB_SLAM3.so` must come from the same build tree | application and library must still be ABI-compatible; do not pair a newer application with an older library that lacks the members the application references |

## Restore / reproduce

The patches are historical artefacts. **The correct restore baseline is
`7e0eedd04f`, not this branch's HEAD.** See `CHANGE_INDEX.md` for the per-item
baseline, the touched file and function, and the restore procedure.
