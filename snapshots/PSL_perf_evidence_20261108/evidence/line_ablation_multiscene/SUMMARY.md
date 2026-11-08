# Multi-scene line ablation: fr3_structure_notexture_{near,far} x {RGB-D, Mono}

Builds on commit `8abd176866`. 5 interleaved repeats (r1:A,B,C r2:A,B,C ...),
SLAM-only, all three line-diagnostic streams enabled.

| cond | `PHOTO_SLAM_LINE_MODE` | `PHOTO_SLAM_LINE_RESCUE` |
|---|---|---|
| A | 0 | 0 |
| B | 2 | 0 |
| C | 2 | 1 |

Mono ATE uses **Sim(3)**, RGB-D uses **SE(3)**. SLAM-only ⇒ **no image-quality
metric** (PSNR/SSIM/LPIPS) is produced or reported.

## 1. Evaluation-protocol correction (prerequisite)

The previously quoted "near RGB-D RPE Δ=1 ≈ 0.25 m/s" was **wrong** and is
retracted. That number was computed from **raw, unaligned** estimate positions,
so the relative-translation difference was contaminated by the mismatch between
the SLAM world frame and the GT world frame. The corrected definition (now in
`eval_tum_se3_v2.py`) computes, for each estimate frame pair (k, k+d):

```
dt      = t[k+d] - t[k]
e_trans = || (p_al[k+d] - p_al[k]) - (p_gt[k+d] - p_gt[k]) ||   [m]
e_rot   = angle( dR_gt^T dR_est ),  dR = R[k]^T R[k+d]            [deg]
```
with `p_al` the **aligned** estimate position. `e_rot` is invariant to the
global alignment. Units are stated per column (metres per step, and m/s).

Pose convention verified: `System::SaveTrajectoryTUM` writes **T_wc**
(`twc` + quaternion), matching TUM `groundtruth.txt`, so translation comparison
is valid. Frames whose tracking failed are **not** written, so the trajectory
length is a coverage measure.

Corrected near RGB-D (A1/B1/C1):

| cond | ATE RMSE | per-frame σ | RPE Δ=1 trans | RPE Δ=1 rot | RPE Δ=30 trans | RPE Δ=30 rot |
|---|---|---|---|---|---|---|
| A1 | 0.022824 | 0.007448 | 6.60 mm (0.190 m/s) | 0.426° | 16.6 mm | 0.887° |
| B1 | 0.030187 | 0.010619 | 6.42 mm (0.185 m/s) | 0.413° | — | — |
| C1 | 0.020544 | 0.008446 | 6.40 mm (0.185 m/s) | 0.412° | — | — |

Sanity checks: Sim(3) on an RGB-D run returns scale 1.011 (metric scale is
observable, as expected); Δ=1 RPE is dominated by per-frame jitter while Δ=30
reflects local drift.

A second correction was needed because coverage differs between conditions:
comparing ATE over different time spans is invalid (drift grows with elapsed
time). All headline numbers below are therefore computed over the **common time
window** of each scene×mode (the intersection of every run that produced a
trajectory).

## 2. Data / configuration

| scene | rgb | depth | GT | association | Δt max |
|---|---|---|---|---|---|
| near | 1082 | 1057 | 3645 | 1054 | 6.9 ms |
| far | 814 | 796 | 2729 | 794 | 6.8 ms |

GT is ascending and overlaps the RGB span in both scenes. A bug in the first far
association generation (`rgb/rgb/...`, `depth/depth/...` — the TUM lists already
carry the prefix) made `imread` fail and abort with SIGABRT 134; regenerated and
verified 0 missing files.

Configs (new, scene-specific): relaxed ORB (`nFeatures 5000`, `iniThFAST 5`,
`minThFAST 2`) plus the `Line.*` sampling block, for both Monocular and RGB-D,
near and far. Identical config is used by A/B/C within a scene×mode.

## 3. Common-window results (5 runs each)

| scene | mode | cond | valid | ATE mean (m) | ATE std | RPE Δ=1 (m) | RPE Δ=1 (°) | RPE Δ=30 (m) | RPE Δ=30 (°) |
|---|---|---|---|---|---|---|---|---|---|
| far | mono | A | 5/5 | 0.345195 | 0.108849 | 0.083082 | 0.2714 | 0.239242 | 4.3360 |
| far | mono | B | 5/5 | 0.373148 | 0.069041 | 0.123041 | 0.2781 | 0.233168 | 4.3864 |
| far | mono | C | 5/5 | 0.412351 | 0.111500 | 0.120313 | 0.2810 | 0.268715 | 4.3528 |
| far | rgbd | A | 5/5 | 0.009533 | 0.000343 | 0.007009 | 0.2923 | 0.010542 | 0.4662 |
| far | rgbd | B | 5/5 | 0.010183 | 0.000761 | 0.006931 | 0.2909 | 0.010143 | 0.4518 |
| far | rgbd | C | 5/5 | 0.009878 | 0.000382 | 0.006966 | 0.2928 | 0.010261 | 0.4602 |
| near | mono | A | 3/5 | 0.112541 | 0.045913 | 0.012440 | 0.4966 | 0.087414 | 7.3522 |
| near | mono | B | 5/5 | 0.184594 | 0.074803 | 0.025093 | 0.4512 | 0.129193 | 6.7152 |
| near | mono | C | 4/5 | 0.135806 | 0.046316 | 0.017242 | 0.4620 | 0.107956 | 7.0800 |
| near | rgbd | A | 5/5 | 0.025152 | 0.005389 | 0.006403 | 0.4109 | 0.017026 | 0.8700 |
| near | rgbd | B | 5/5 | 0.022288 | 0.002594 | 0.006497 | 0.4182 | 0.016013 | 0.8448 |
| near | rgbd | C | 5/5 | 0.021775 | 0.003955 | 0.006492 | 0.4176 | 0.015679 | 0.8345 |

Coverage: RGB-D 1.000 in both scenes for all conditions. Far Mono 0.7482 in every
run of every condition. Near Mono: A 0.00/0.57/0.80/0.00/0.65, B 0.3993 x5,
C 0.3993/0.00/0.3993/0.3993/0.3993 (three runs had **no trajectory at all**).

### Effect sizes (Welch two-sided)

| scene | mode | metric | A−B | t | p | B−C (null control) | t | p |
|---|---|---|---|---|---|---|---|---|
| near | rgbd | ATE | +0.002864 | 1.07 | 0.327 | +0.000513 | 0.24 | 0.815 |
| far | rgbd | ATE | −0.000650 | −1.74 | 0.136 | +0.000306 | 0.80 | 0.454 |
| near | mono | ATE | −0.072054 | −1.69 | 0.143 | +0.048788 | 1.20 | 0.271 |
| far | mono | ATE | −0.027953 | −0.49 | 0.643 | −0.039202 | −0.67 | 0.526 |

**Every p > 0.08.** In the two mono cases the *null* comparison (B−C) is as large
as the effect comparison (A−B), which is the clearest possible sign that the
differences are noise.

## 4. Answers to the required questions

**Is the rescue ever exercised?** No, effectively never. Across all 60 runs:
`[LineGate] old_would_fail=1` occurs **0** times, so the original point-only
early returns never fire. `[LineRescue] oldPtVerdict=FAIL` occurs **12** times
(all at the LocalMap stage), i.e. the rescue path *was* attempted 12 times, and
succeeded **0** times (`reason` = too few reliable lines). Therefore
**"rescue benefit = not triggered"**; B/C serves as a regression check only, and
indeed no B/C difference is significant.

**Does A vs B show a change in localization or coverage after lines participate?**
No separable change in either:
* RGB-D (both scenes): coverage identical at 1.000; ATE differences
  (near +2.9 mm, far −0.65 mm) are within the run-to-run spread, and the Δ=1
  translation/rotation RPE is essentially identical across A/B/C
  (near: 6.40/6.50/6.49 mm; far: 7.01/6.93/6.97 mm).
* Lines demonstrably reach the optimizer: `[LineEdgeDiag]` shows 0 line-edge
  calls in every A run (sanity check) versus 462-602 calls/run (RGB-D, ~1300-1400
  line edges) and 341-2341 (Mono).
* Mono is **confounded**: `System::TrackMonocularWithLine` dispatches to a
  *different* function (`GrabImageMonocular`) when `LINE_MODE==0`, so mono A vs
  B/C changes the tracking implementation, not just the line switch. Mono also
  fails outright in 3/15 runs (`SYSTEM-> Reseting active map in monocular case`,
  0 KFs, empty trajectory), and near-Mono B/C stall at exactly 432 frames
  (0.3993 coverage), so its A/B/C comparison is not trustworthy.

**Single-run differences as "benefit"?** Not used. All judgements use interleaved
repeats and effect sizes; within-condition spread is reported next to every
mean.

## 5. Limitations

* n = 5 per condition. With this n, only large effects would be detectable.
* SLAM-side nondeterminism is **not seedable**: `PHOTO_SLAM_SEED` only seeds the
  Gaussian mapper, which is skipped under `PHOTO_SLAM_SLAM_ONLY=1`. ORB-SLAM3's
  LocalMapping/LoopClosing threads interleave with tracking by wall-clock, so
  repeats quantify the spread but cannot remove it.
* Neither scene naturally trips the point gate, so **the rescue cannot be
  evaluated on these sequences at all**. A sequence (or a clearly labelled
  synthetic stress test) that actually fails the point test is required.
* Monocular results are additionally confounded by the pipeline dispatch above
  and by init instability; they are reported for completeness, not as evidence.
* Only SLAM-only runs; no mapping/training and no image-quality claims.
* **No claim of line benefit or harm is made.**

## 6. Artifacts

Scripts: `fr3_ablation_run.sh` (one condition, any scene×mode),
`fr3_ablation_repeats.sh` (interleaved repeats),
`ablation_aggregate.py` (per-scene×mode CSV + summary),
`ablation_common_window.py` (fair common-window comparison + Welch tests),
`eval_tum_se3_v2.py` (corrected SE(3)/Sim(3) ATE, σ, translation+rotation RPE).

Configs: `cfg/ORB_SLAM3/{Monocular,RGB-D}/TUM/tum_freiburg3_structure_notexture_{near,far}_lowfast.yaml`,
`cfg/gaussian_mapper/RGB-D/TUM/tum_freiburg3_structure_notexture_far.yaml`,
`cfg/ORB_SLAM3/RGB-D/TUM/associations/fr3_structure_notexture_far{,_150}.txt`.

Evidence (this folder): `all_runs.csv`, `common_window.csv`,
`{near,far}_{rgbd,mono}.csv`, `gate_rescue_summary.txt`, `logs.tar.gz`
(all 60 raw run logs + run.meta).
