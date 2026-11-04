# TUM mono: export convention, evaluator binding, and rotation check

Scope: `freiburg1_room` monocular, three arms (T0/P1/P2) from the frozen Release
build (see `eval/EDGE_COUNTS.md` for the hashes). This note exists because the
joint coverage numbers for T0 and P1 come out at zero while translation-only
coverage does not, which has to be explained rather than reported.

## 1. What is bound

| item | identity |
|---|---|
| evaluator | `/workspace/code/PSL-perf/scripts/eval_tum_se3_v2.py`, sha256 `199862bbeb8862083dd1d2eb62a5043641c5e426c132b2493a18c8ee04b3e3ac` |
| call | positional `<gt> <est>`, `--mode sim3` (mono) / `se3` (RGB-D), `--method interp --max-dt 0.02` |
| GT | `rgbd_dataset_freiburg1_room/groundtruth.txt` (TUM, 4890 samples, ~100 Hz) |

The evaluator is estimate-centric: it iterates over estimate frames, interpolates
GT at each estimate timestamp (linear + slerp), and fits the alignment
estimate -> GT. Its output schema is
`ate{rmse,mean,median,std,max,p95}`, `rot{rmse,median,max}`,
`rpe[d]{trans_rms,trans_mean,rot_rms,rot_mean}`, plus `pairs`, `unmatched` and
`align_scale`. The per-frame dump is `<prefix>.perframe.txt` with columns
`est_idx ts_est ts_dt_res err_m rot_deg ex ey ez gt_x gt_y gt_z`.

## 2. Export convention (checked in the source, not assumed)

`ORB-SLAM3/src/System.cc:1266` writes `Twc` as

```
t  tx ty tz  q.x q.y q.z q.w
```

i.e. **camera-to-world** in **TUM order (qx qy qz qw)** -- Eigen's internal order
is (w,x,y,z), so the explicit reordering matters and it is present. Two further
properties were confirmed from the same code path:

* the export covers **one map only** (`SelectExportMap`) and prints
  `frame entries by reference-KF map: ...`; frames whose reference keyframe
  belongs to an earlier map are not written. So a run with resets yields a
  single-map trajectory, and the alignment is not mixing world frames. This is
  why `n_accepted` (914/973/834) exceeds the exported count (333/266/399).
* no quaternion renormalisation or axis flip is applied at export.

## 3. Rotation: the number is real, but it is an alignment artefact

Per-frame statistics over the exported, GT-associated frames
(`--mode sim3`, i.e. the frozen protocol):

| arm | exported | median trans err | median rot err (after sim3 align) | frac rot <= 5 deg | joint (<=0.2 m, <=5 deg) |
|---|---|---|---|---|---|
| T0 | 333 | 0.130 m | 15.77 deg | 0.00 | 0.00 |
| P1 | 266 | 0.171 m | 65.92 deg | 0.00 | 0.00 |
| P2 | 399 | 0.050 m | 3.25 deg | 0.80 | 0.80 |

The rotation error is not drift. Repeating the alignment and then asking for the
single rotation that best explains the residual gives:

| arm | singular values of the position cross-covariance (normalised) | constant residual rotation after alignment | residual after removing that constant |
|---|---|---|---|
| T0 | 0.885 / 0.086 / 0.029 | 15.14 deg | median 3.16 deg, p90 12.68 |
| P1 | 0.984 / 0.012 / 0.004 | 65.63 deg | median 9.39 deg, p90 18.13 |
| P2 | 0.862 / 0.125 / 0.013 | 2.80 deg | median 2.74 deg, p90 5.77 |

`align_scale` is 3.63 / 6.88 / 1.49 respectively.

Read together: the P1 and T0 trajectories are nearly one-dimensional (98 % and
89 % of the position variance on a single axis), so the rotation that Umeyama
recovers from positions is essentially unconstrained about that axis. The
alignment therefore lands on an arbitrary rotation, and the resulting
orientation error is almost a **constant offset** (15.1 deg for T0,
65.6 deg for P1) that has nothing to do with how well the system tracked
orientation; the residual after removing it is 3.2 deg / 9.4 deg. P2's
trajectory is two-dimensional enough that the fit is well posed, and its
orientation error (3.25 deg, 80 % of frames within 5 deg) is a genuine result.

## 4. Consequences for reporting (frozen, no threshold changes)

The frozen joint definition (translation <= 0.20 m **and** rotation <= 5 deg,
rotation measured after the position-based Sim(3)/SE(3) alignment) is **not**
changed. It is reported as measured: T0 = 0.000, P1 = 0.000, P2 = 0.231 of all
GT-evaluable inputs.

Because the zero is dominated by an unconstrained alignment degree of freedom
rather than by tracking quality, every table that reports the joint criterion
also reports, from the same runs and the same evaluator:

* translation-only correct coverage (T0 0.200, P1 0.148, P2 0.268 of all
  GT-evaluable inputs);
* the alignment-invariant rotation metric `rpe_rot` (a common rotation cancels
  inside $dR$; d=1: 1.08 deg for P2), which measures orientation drift without
  inheriting the alignment's arbitrary rotation;
* the alignment degeneracy itself (normalised singular values above).

No number in this note is a substitute for another: the joint figure stays 0 for
T0/P1 and is reported as such. What is *not* claimed anywhere is that T0/P1 lose
orientation tracking to that degree -- that would be unsupported, because the
metric used cannot distinguish a constant offset from drift.
