# Cross-dataset front-end evaluation — TUM fr3 cabinet ablation

Executed on the verified fix commit `d3214194` (branch
`fix/line-reproj-triangulate-po-reeval`), rebuilt independently in
`/workspace/deliv/cabinet-eval`. Definitions were frozen in `PROTOCOL.md`
**before** the batch; everything that deviated is in `DEVIATIONS.md`.

Build identity of every reported run:

| artefact | sha256 |
|---|---|
| `libORB_SLAM3.so` | `87f728740a61c0c8b01cf412030e0c63b3f28ed245daff91c6d737485d95af72` |
| `bin/eval_frontend` | `2f4a2ab54f18d6636ee4937ff61319e568ad2bb495e86eb44516e11957d7fa7f` |

## Results

18 monocular cells (2 scenes × 3 arms × 3 repeats), **all exit code 0**, no
crash, no missing artefact, no re-run and no cell replaced.

| scene | modality | arm | runs | accepted coverage median [min,max] | ATE RMSE (m) median [min,max] | per-run ATE |
|---|---|---|---|---|---|---|
| cabinet | mono | P1 | 3 | 0.606 [0.557,0.651] | 0.0065 [0.0016,0.0141] | 0.0016 0.0065 0.0141 |
| cabinet | mono | P2 | 3 | 0.657 [0.353,0.799] | 0.0041 [0.0018,0.0264] | 0.0018 0.0041 0.0264 |
| cabinet | mono | T0 | 3 | 0.625 [0.467,0.660] | 0.0202 [0.0031,0.0500] | 0.0031 0.0202 0.0500 |
| large_cabinet | mono | P1 | 3 | 0.989 [0.839,0.989] | 0.0366 [0.0352,0.0407] | 0.0352 0.0366 0.0407 |
| large_cabinet | mono | P2 | 3 | 0.986 [0.853,0.989] | 0.0357 [0.0327,0.0382] | 0.0327 0.0357 0.0382 |
| large_cabinet | mono | T0 | 3 | 0.989 [0.878,0.989] | 0.0377 [0.0362,0.1036] | 0.0362 0.0377 0.1036 |

Map resets (number of maps created per run, from the run log):

| cell | maps per repeat |
|---|---|
| cabinet mono T0 | 3, 3, 4 |
| cabinet mono P1 | 3, 3, 4 |
| cabinet mono P2 | 3, 3, 2 |
| large_cabinet mono T0 | 1, 1, 2 |
| large_cabinet mono P1 | 1, 1, 2 |
| large_cabinet mono P2 | 2, 1, 1 |

ATE and accepted coverage come from two independent sources:
`CameraTrajectory_TUM.txt` scored by the previously validated evaluator
`eval_tum_se3_v2.py` (`--mode sim3 --method interp --max-dt 0.02`, scale
estimated because monocular scale is unobservable), and `final_state.csv`
written by the run itself. `eval/build_results_table.py` aggregates them; no
number here was typed by hand. Raw per-run lines:
`/workspace/eval_runs/cabinet2_ate.txt` (not committed).

## Interpretation

1. **No arm separates from the others.** In `cabinet` the median accepted
   coverage of T0 / P1 / P2 is 0.625 / 0.606 / 0.657 and the within-cell spread
   (e.g. P2: 0.353–0.799) is larger than the between-arm difference. In
   `large_cabinet` all three arms sit at 0.98–0.99. With n = 3 this data cannot
   support any statement that line edges in PoseOptimization/Local BA help or
   hurt tracking on these scenes.

2. **The ATE medians must not be read as an ordering.** `cabinet` T0 has the
   largest median (0.0202 m) and P2 the smallest (0.0041 m), but T0's range
   reaches down to 0.0031 m and P2's up to 0.0264 m — the ranges overlap almost
   completely. `large_cabinet` is flat at ≈0.036 m for all arms with a single T0
   outlier at 0.1036 m.

3. **ATE here is conditional on tracking.** The exported trajectory contains only
   frames the system accepted, so a cell with 35 % acceptance can post a lower
   ATE than a cell with 66 % acceptance purely because it dropped the hard part
   of the sequence. Accepted coverage must be read first, ATE second.

4. **The two cabinet scenes differ sharply, and the arm split does not explain
   it.** Both run the same monocular pipeline with `_lowfast` extractor settings;
   `cabinet` accepts ~62 % of frames and rebuilds its map 2–3 times,
   `large_cabinet` accepts ~99 % and rebuilds at most once. Nothing here
   attributes that difference to the line pipeline.

## Not delivered, and why (no substitute numbers are given)

* **Correct-pose coverage at the frozen tolerance set {0.05, 0.10, 0.20, 0.50} m,
  and longest continuous correct tracking: NOT AVAILABLE.** An in-house
  implementation in `eval/metrics.py` was written and it does **not** reconcile
  with the validated evaluator: on the identical run
  `large_cabinet/mono/T0/rep1` it produced ATE 3.239 m where
  `eval_tum_se3_v2.py` produces 0.1036 m, with fitted scales 0.0112 vs 0.0851.
  The discrepancy is unresolved, so no coverage-under-tolerance number is
  reported rather than a number that cannot be trusted. The "longest continuous
  correct tracking" metric, defined on top of that tolerance test, inherits the
  same block.
* **RGB-D arms: NOT RUN on the verified build.** Cabinet RGB-D measured
  3400 s / 3399 s / 3399 s per run (≈3.06 s per frame) on the earlier pre-fix
  build, identical for T0/P1/P2, i.e. a baseline RGB-D path cost and not a line
  cost. 18 RGB-D runs would have needed ~17 h. The three RGB-D runs that exist
  were produced by a binary whose sources did not match `d3214194` and are
  excluded. See `DEVIATIONS.md` D2.
* **Replica mono/RGB-D and ETH3D mono/RGB-D: NOT RUN.** No loader for the
  Replica image-index layout, no Replica depth-scale config, no ETH3D configs,
  no schedule budget left. See `DEVIATIONS.md` D1.
* **Upstream Photo-SLAM comparison: NOT RUN.** T0 is this code base with
  `PHOTO_SLAM_LINE_MODE=0`, not the published system.

## Reproducing

See `eval/README.md`. Run root for this report: `/workspace/eval_runs/cabinet2`
(results are not committed).
