# Replica ground truth: derivation rule and verification

Status: **closed**. The derived GT used for every Replica row of the frozen
manifest is produced by `eval/derive_replica_gt.py` from the dataset's raw
`traj.txt`, and every property below is checked mechanically (that script exits
non-zero on any failure).

## Why a derived GT is needed

* `traj.txt` holds 2000 rows of 16 numbers: a row-major 4x4 camera-to-world
  matrix per frame, one row per `results/frame%06d.jpg`.
* The pre-existing `pose_TUM.txt` was compared once and the comparison failed on
  "column arrangement" and stopped there. Reading the code that produced it
  settles the question without guessing: `SEGS-SLAM/eval/shapeReplicaGT.py`
  (the repository's own converter) does exactly

  ```
  c2w = line.reshape(4, 4)            # row-major
  c2w[:3, 1] *= -1                    # negate camera y axis column
  c2w[:3, 2] *= -1                    # negate camera z axis column
  q   = Rotation.from_matrix(c2w[:3,:3]).as_quat()   # scipy -> (x,y,z,w)
  write "i  tx ty tz  qx qy qz qw"
  ```

  So there are two separate facts, and the earlier failure was the second one:

  1. **Camera axes.** The raw matrices are in the renderer's camera axes; the
     x-right / y-down / z-forward axes that ORB-SLAM3 reports are obtained by
     negating the camera **y and z columns**. In matrix terms:
     $R_{\text{cv}} = R_{c2w}\,\mathrm{diag}(1,-1,-1)$.
  2. **Time base.** The first column of `pose_TUM.txt` is the integer **frame
     index**, not seconds. Our entry feeds Replica frames on a synthetic
     $i/30\,\text{s}$ base (declared in `eval_entry.txt`, and consistent with
     `Camera.fps: 30` in the repository's Replica configs). A GT whose time
     column is an index cannot be associated with an estimate in seconds, which
     is why the comparison could not proceed -- not because the columns were
     permuted.

## What this script writes

| file | time column | use |
|---|---|---|
| `<scene>_gt_tum_s.txt` | $i/30$ s | evaluation (matches the entry's time base) |
| `<scene>_gt_index.txt` | $i$ | byte-level comparison with `pose_TUM.txt` |

`pose_TUM.txt` itself is **not modified**.

## Verification performed on all 8 scenes (2000 frames each)

* raw matrices are rigid: max $\lVert R^\top R - I\rVert_\infty \le 1\text{e-}6$,
  $\lvert\det R - 1\rvert \le 1\text{e-}6$, last row $(0,0,0,1)$;
* quaternion from matrix, then matrix from quaternion, reproduces the (converted)
  rotation to **$8.9\times10^{-16}$** worst case; $\lvert\lVert q\rVert-1\rvert = 0$;
* timestamps strictly increasing, $t_0=0$, $t_{1999}=66.6333$ s at 30 fps;
* **agreement with the repository's own conversion**: positions
  differ by $0.000\times10^{0}$ m (bit-identical) and rotations by
  $\le 2.4\times10^{-6}$ deg (print precision), on every scene;
* first column of `pose_TUM.txt` is exactly `0..1999` on every scene
  (`pose_TUM_first_column_is_index = true`);
* continuity, as a check against a transpose / wrong-axis / index slip (all of
  which still yield valid rotations): largest single-frame step
  0.0102-0.0367 m and 2.04-9.51 deg, i.e. smooth camera motion, no jumps.

| scene | traj sha256 (16) | derived GT sha256 (16) | max step m | max step deg |
|---|---|---|---|---|
| office0 | `6fdf15ff9a85626e` | `43713a68cf6723b2` | 0.0217 | 3.790 |
| office1 | `0d99612a5e70d989` | `a19df9c6a83f8514` | 0.0102 | 2.696 |
| office2 | `4b06c484cb814cd0` | `1065fb2d88df2427` | 0.0337 | 2.397 |
| office3 | `b6496454205dfa29` | `2eae76013939bcf7` | 0.0367 | 2.521 |
| office4 | `a86e633630dfc080` | `4bfee40cbd98d5c8` | 0.0252 | 2.207 |
| room0 | `5692a60076e426b4` | `cd16bfdcf89f5154` | 0.0171 | 2.041 |
| room1 | `5c650f90192da13b` | `bfaacfcde5012e21` | 0.0168 | 3.895 |
| room2 | `5edb60cd84aeaba6` | `186309cc3f38fa93` | 0.0231 | 9.510 |

Full record: `/workspace/eval_runs/gt_replica/REPLICA_GT_REPORT.json`
(contains the complete hashes and every check value).

## What changed in the pipeline

`eval/coverage.py` previously fell back to
`/workspace/eval_runs/gt_cache/replica_<scene>.txt`, a copy of `pose_TUM.txt`
whose time column had merely been relabelled to $i/30$ -- an unverified edit
that is superseded here. It now requires
`/workspace/eval_runs/gt_replica/<scene>_gt_tum_s.txt` and aborts with a
pointer to the derivation script if that file is missing, so a Replica metric can
never again be computed against an unverified time base.
