# Change index — pose-residual experiments

All three items below are **archive only**. None of them is applied on this
branch, and this branch contains no source change.

> **Restore baseline is `7e0eedd04f` for all three items, not this branch's HEAD.**
> The archive branch is based on `line_slam_branch` (`015bf23962`), whose
> `PoseOptimizationWithLine()` only has three judging loops and an extra
> early-exit, so the FULL/sticky patches do **not** apply to it. That mismatch is
> itself one of the findings of this work.

Common facts:

| item | value |
|---|---|
| Touched file | `ORB-SLAM3/src/Optimizer.cc` (only) |
| Touched function | `Optimizer::PoseOptimizationWithLine(Frame*)` |
| Restore baseline | `7e0eedd04f` (`perf/compute-latency`, contains all four judging loops: mono/stereo point, mono/stereo line) |
| Not touched | thresholds, robust-kernel schedule, matcher, rescue path, α (line weight), default branch |

---

## 1. FULL — residual refresh in all four judging loops

| field | value |
|---|---|
| Patch | `patches/full-optimizer-residual-refresh.patch` (also a commit diff: `14 insertions(+), 4 deletions(-)`) |
| Original commit | `d567197e3a` on `fix/pose-residual-refresh`, cherry-picked with `-x` from `f94543c3f4` |
| Worktree state | local only, **not pushed** (`fix/pose-residual-refresh` exists on one machine only) |
| Change | each of the four loops calls `e->computeError()` before reading `chi2()`, and the outlier test becomes `!std::isfinite(chi2) || chi2 > threshold` |
| Function | `Optimizer::PoseOptimizationWithLine(Frame*)` — `vpEdgesMono`, `vpEdgesStereo`, `vpEdgesLineMono`, `vpEdgesLineStereo` loops |
| Outcome | **rejected** — `TUM fr1/desk` monocular regressed (window translation above the noise band in 3/3 pairs, Sim(3) median 2.9× baseline); the RGB-D cross-check scene could not be used because the baseline itself failed the gate |
| Restore | `git switch --detach 7e0eedd04f && git apply patches/full-optimizer-residual-refresh.patch` (apply the `ORB-SLAM3/` path from the repository root) |

## 2. Mechanism diagnosis — same-state replay harness

| field | value |
|---|---|
| Patch | `patches/mech-same-state-replay.patch` (264 lines, instrumentation only) |
| Worktree state | local only, **uncommitted** in a detached worktree at `7e0eedd04f` |
| Change | adds an env-gated policy switch (`pre` / `point` / `line` / `full` / `sticky`), an optional per-iteration trace, and a replay wrapper that restores the frame pose plus `mvbOutlier` / `mvbLineOutlier` and re-runs the *production* function under each policy on the identical state |
| Function | `Optimizer::PoseOptimizationWithLine(Frame*)` (the original body is kept verbatim and renamed; a thin wrapper is added) |
| Purpose | attribute the behaviour change to a specific edge class without copying the loops into a test |
| Outcome | diagnostic only — with the policy switch unset the behaviour is the untouched baseline; **not a candidate and never shipped** |
| Restore | `git switch --detach 7e0eedd04f && git apply patches/mech-same-state-replay.patch`, then set `PHOTO_SLAM_PO_MECH=all` (+ `PHOTO_SLAM_PO_REFRESH`) and read the `[PoMech]` lines from stderr |
| Result | point-edge re-admissions 2381, line-edge re-admissions 1, non-finite rejections 0 over 609 frames; first pose divergence at frame 4 for point/FULL/sticky, frame 26 for line-only |

## 3. LINE-STICKY — refresh all four, sticky lines

| field | value |
|---|---|
| Patch | `patches/line-sticky-candidate.patch` (`23 insertions(+), 6 deletions(-)`) |
| Worktree state | local only, **uncommitted** in a detached worktree at `7e0eedd04f` |
| Change | all four loops refresh and reject non-finite; point edges may re-enter (`setLevel(0)`); a line edge rejected during the call keeps `mvbLineOutlier = true` and stays at level 1 for the rest of the call, and is re-judged on the next frame |
| Function | `Optimizer::PoseOptimizationWithLine(Frame*)` — the two line loops plus two `std::vector<char>` sticky masks declared next to `nBad` |
| Outcome | **rejected** — mechanism checks passed (four classes refreshed, lines never re-admitted, only one extra line rejection in 609 frames, flags consistent with level), but the formal regression lost tracking in one repeat (`rc = 0`, 92 × `Fail to track local map`, second map created, 595/610 poses exported, Sim(3) ATE 20× baseline in that pair) |
| Restore | `git switch --detach 7e0eedd04f && git apply patches/line-sticky-candidate.patch` |

---

## Local-only material (not published)

The following are kept on the experiment machine and intentionally **not**
uploaded, because they contain absolute build paths, per-run specifications,
raw logs, trajectories and binaries:

- release decision record and pre-registration documents;
- build logs for the baseline, FULL, mechanism and candidate trees;
- the 12-run formal regression (trajectories, `stdout.log`, per-run JSON);
- the 609-frame same-state replay log;
- SHA-256 manifests of the local artefacts.

The published `SHA256SUMS` in this directory covers only the files that are
actually in this directory, using relative paths.
