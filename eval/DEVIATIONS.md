# Deviations from the frozen protocol, and execution findings

Everything here was discovered **after** `PROTOCOL.md` was committed. The
protocol text was not edited; this file records what actually happened.

## D1. Only TUM (fr3 cabinet scenes) was executed — 4 of 6 categories NOT RUN

Executed: TUM monocular and TUM RGB-D on `rgbd_dataset_freiburg3_cabinet` and
`rgbd_dataset_freiburg3_large_cabinet`.

Not executed: Replica mono, Replica RGB-D, ETH3D mono, ETH3D RGB-D.
Reason (engineering, not data): the unified entry `examples/eval_frontend.cpp`
has loaders for TUM-style `rgb.txt` (mono) and TUM/ETH3D-style association files
(rgbd). Replica uses an image-index layout (`results/frame*.png` +
`results/depth*.png`, no timestamps) and its depth scale is not
`RGBD.DepthMapFactor=5000`; both need a dedicated loader and a calibrated config.
That work was not done in this round. **No numbers are reported for these four
categories.**

## D2. [RETRACTED] "RGB-D is slow, ~3 s per frame" — it was the pacing sleep

**This section's original attribution was wrong and is retracted.**

The original text said the 3400 s/run was "the baseline RGB-D path of this
build". That conclusion is withdrawn. Evidence from the staged entry-level
timing on the verified build (60 associated pairs, cabinet RGB-D, serial,
`PHOTO_SLAM_NO_PACING` unset):

| arm | api | read_rgb | cvt | read_depth | track | drain | **pace** | sum |
|---|---|---|---|---|---|---|---|---|
| T0 | `TrackRGBD` | 570 ms | 10 ms | 150 ms | 373 ms | 0.03 ms | **177 866 ms** | 178 968 ms |
| P1 | `TrackRGBDWithLine` | 537 ms | 11 ms | 159 ms | 615 ms | 0.05 ms | **177 651 ms** | 178 974 ms |
| P2 | `TrackRGBDWithLine` | 545 ms | 12 ms | 156 ms | 652 ms | 0.06 ms | **177 625 ms** | 178 991 ms |

99.4 % of the wall clock is the real-time pacing sleep. Tracking is 2–4 ms per
frame; image and depth reading are ~10 ms per frame combined. So RGB-D is **not**
slow — the *input list was not in time order* (see D9), and the entry sleeps
`ts[i+1] - ts[i]` per frame. With a scrambled list that difference is tens of
seconds, which reproduces the 3 s/frame average and the "identical across arms"
pattern (identical because the sleep, not the line pipeline, dominated).

The old claim "identical for T0/P1/P2, therefore a baseline cost" is exactly the
kind of inference that the corrected timing falsifies.

## D9. `gen_assoc.py` wrote the association file out of time order

The greedy acceptance in `gen_assoc.py` walks candidate pairs in order of
increasing time difference, so the accepted list was **not** sorted by timestamp.
The pair *set* was correct (1112/1147 and 984/1011) but the *order* was not, and
every sequential reader — including this entry's pacing and any ORB-SLAM3 example
loader — assumes strictly increasing timestamps.

Consequences that were measured, not assumed:
* the RGB-D runs consumed frames in a non-chronological order, so their
trajectories are not meaningful and are excluded from results;
* pacing slept tens of seconds per frame (D2), which is what "slow RGB-D" was.

Fix: sort the accepted pairs by RGB timestamp before writing
(`scripts/gen_assoc.py`), verified with an explicit monotonicity check
(`out_of_order=0` for both sequences). New hashes:
`fr3_cabinet.txt` `c2523114…`, `fr3_large_cabinet.txt` `5c3b70a3…`
(previously `fcecb5c9…` and `509277a7…`). All RGB-D runs on the corrected file
are a separate version.

## D3. `bc` is not installed — `wall_s=` is empty in rep-1 `status.txt`

Rows written by the first version of `scripts/run_batch.sh` contain
`wall_s=` with no value. Wall time for those runs is taken from the ISO
`start_iso` / `end_iso` stamps instead; the script now uses `python3`.

## D4. Build defect found and fixed (independent build only)

`scripts/build_eval_orb.sh` initially globbed `src/*.cc`, which silently omitted
`src/OptimizableTypes.cpp` (a `.cpp`). The resulting `libORB_SLAM3.so` linked but
left `EdgeSE3ProjectXYZ`, `VertexSim3Expmap`, `MLPnPsolver` etc. undefined. Fixed
by adding `src/*.cpp`; the library used for every reported run is the rebuilt one
`eb716e53f47167bb4c93b6f69ab7ab9fda9ce58fd523ea4ecad4d120e2854580`
(36 translation units).

## D5. No upstream (original Photo-SLAM) arm

Not built, not run. T0 is this code base with `PHOTO_SLAM_LINE_MODE=0`; it is not
a stand-in for upstream.

## D6. Mono and RGB-D use different ORB extractor settings

Already declared in `PROTOCOL.md` §4; restated because it also means the mono and
RGB-D tables cannot be compared with each other.

## D7. The first 9 runs were built from the wrong sources and were discarded

The first batch (mono and RGB-D, `rep1`, run root `/workspace/eval_runs/cabinet`)
was executed with a binary built from a worktree whose **on-disk files did not
match commit `d3214194`**, even though `git rev-parse HEAD` in that worktree
reported `d3214194` and every `meta.txt` recorded `git_head=d3214194`.

Evidence, captured before anything was changed:

```
git diff --numstat  ->  34 files
  0   20  ORB-SLAM3/src/Atlas.cc
 29  187  ORB-SLAM3/src/Tracking.cc
git diff ORB-SLAM3/src/Atlas.cc
  -#include "LineLifecycleLedger.h"
  -#include "MapEventLog.h"
git status --short
   D ORB-SLAM3/include/LineLifecycleLedger.h      (and the other 6 new headers)
```

i.e. the worktree held the **parent** content of the fix commit: the ledger /
map-event instrumentation and the seven new headers were absent. Since
`libORB_SLAM3.so` is compiled from those very files, the first build
(`eb716e53…`) and the first entry binary (`2c7280b9…`) did not contain the
delivered fix, and the 9 runs built on them are not results for this commit.

Recovery:

1. `git checkout -- .` restored the cone; `git diff --numstat` → 0 files, the 7
   headers exist, `Atlas.cc` contains the ledger include.
2. Full rebuild → `libORB_SLAM3.so` `87f72874…`, `bin/eval_frontend` `2f4a2ab5…`.
3. The 18 monocular cells were re-run into a clean root
   `/workspace/eval_runs/cabinet2`; the old root is kept only as evidence.

Lesson recorded for later batches: **a git ref in `meta.txt` is not proof of what
was compiled.** A future runner must hash the source files (or the git tree
object) that the library was actually built from, not `HEAD`.

## D8. Correct-pose coverage and longest-continuous-tracking withheld

`eval/metrics.py` implements them, but its alignment does not reconcile with the
validated evaluator on the same run (`large_cabinet/mono/T0/rep1`): 3.239 m vs
0.1036 m, fitted scale 0.0112 vs 0.0851. Rather than publish a number that
disagrees with the project's own validated tool by 30×, those two metrics are
reported as NOT AVAILABLE in `REPORT.md`. The in-house code is kept in the tree
so the discrepancy can be chased down, but it is not used for any stated result.

