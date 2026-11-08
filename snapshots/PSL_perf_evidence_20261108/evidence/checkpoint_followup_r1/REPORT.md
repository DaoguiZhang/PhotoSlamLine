# room1 checkpoint follow-up (r1): offline rescore + controlled continuation

## Scope and controls

- One A/C pair from the previous online run was reused: same source, config,
  seed and checkpoint protocol; only the raster backend differed.
- Existing snapshots were not resumed as optimizers. They were rendered offline
  by `eval_checkpoint_snapshot`, which loads the saved model tensors, the saved
  training cameras and the saved undistort masks, and calls the existing C++
  `loss_utils::psnr_gaussian_splatting`.
- The controlled continuation was a single necessary live C run. The complete
  live state was captured after natural final refinement, then the same state
  was replayed as two fixed-topology arms (A and C) with one pre-generated view
  sequence.
- No loss, LR rule, densify threshold, geometry threshold or production
  refinement budget was changed.

## 1. Offline re-render matches the checkpoint-time score

`offline_crosscheck.csv` joins the offline render with the checkpoint-time
`psnr_gaussian_splatting.txt` per keyframe. All 8 checkpoints have the same frame
counts; the largest absolute per-frame difference is **0.00661 dB**, and every
mean difference is within **0.00018 dB**. The full-frame means agree within
0.0002 dB (not necessarily at the fourth printed decimal).

This validates the offline path (snapshot tensors + saved cameras + saved masks)
and means the earlier checkpoint-time values do not need correction.

### Fixed 12 early common frames (offline native GS PSNR)

| checkpoint | n | A mean | C mean | A−C mean | A−C median |
|---|---:|---:|---:|---:|---:|
| 500 | 12 | 35.8217 | 37.5558 | **−1.7342** | −1.4701 |
| 1000 | 12 | 36.0691 | 36.9497 | **−0.8806** | −1.0671 |
| 1500 | 12 | 29.8121 | 26.6103 | **+3.2019** | +3.4407 |
| end | 12 | 34.3403 | 32.5927 | **+1.7476** | +1.9184 |

Negative A−C means C is higher. The previous checkpoint-time table is reproduced
within 0.0003 dB.

## 2. 1000 → 1500: camera, scale and opacity

`camera_fixed12_1000_1500.csv` uses the relative pose
`inv(Tcw_1000) * Tcw_1500` for the same image within each run. Absolute camera
matrices are not compared between A and C; their map world frames differ.

Summary for the fixed 12 frames:

| side | rot max (deg) | rot mean (deg) | trans max (m) | trans mean (m) | ΔPSNR mean (dB) |
|---|---:|---:|---:|---:|---:|
| A | 0.1167 | 0.0733 | 0.00484 | 0.00255 | **−6.257** |
| C | 0.0939 | 0.0560 | 0.00343 | 0.00216 | **−10.339** |

The strongest single check is `frame000000.jpg`: its camera is **exactly
unchanged** (0° and 0 m) in both runs, yet its PSNR drops by **3.61 dB in A** and
**10.50 dB in C** from 1000 to 1500. The quality drop is therefore not explained
by the measured camera updates. The camera/relative-pose correlation is weak and
unstable at n=12; it is descriptive only, not causal.

Model statistics (`model_stats.csv`, 1000 → 1500):

| side | bbox diag | median exp(scale_x) | max exp(scale) | opacity mean | opacity median |
|---|---:|---:|---:|---:|---:|
| A | 8.286 → 8.318 | 0.001895 → 0.001515 | 2.287 → 1.404 | 0.663 → 0.684 | 0.889 → 0.963 |
| C | 8.369 → 8.331 | 0.002073 → 0.002023 | 12.496 → 11.888 | 0.654 → 0.639 | 0.860 → 0.903 |

The scene extent is stable; there is no global model rescale. Median opacity
rises in both runs, so there is no global opacity collapse. C carries much larger
scale outliers (max about 12 vs A about 1.4–2.3), and C has many more line
Gaussians (1313 → 1844 vs A 925 → 977), but these are descriptive and are
confounded with topology. The existing data cannot attribute the 1500 dip to a
specific scale or opacity event.

**Conclusion for this stage:** the large 1000 → 1500 drop is model/topology/
training-side, not a visible camera-pose jump. The local-BA/camera-consistency
branch is not supported by this evidence.

## 3. Snapshot completeness

`snapshot_completeness.txt` records the exact limitation: each r0 `snapshot.pt`
has 16 parameter/metadata tensors, but **no Adam moments, no optimizer LR
options and no per-keyframe usage state**. It is therefore a rendering
checkpoint, not a resumable optimizer checkpoint. Rebuilding Adam from a PLY was
not done.

The r0 `end` checkpoint is also not the process terminal state: it was captured
at input frame 2000 during final refinement (C: 8337 updates), whereas the live
final state in this run is 8712 updates.

For the controlled experiment, the live state was therefore captured in-process.
`undertrain_r1/terminal_state.txt` shows all 6 Adam groups with
`exp_avg`, `exp_avg_sq` and LR, plus CPU/CUDA RNG and 168 keyframes. The Adam step
is 8625 while `frozen_opt_steps` is 8712; the difference is 87, and no
per-iteration reconciliation is available. Do not treat the two counters as a
one-to-one identity.

## 4. Controlled fixed-topology continuation from the C terminal

Protocol:

- Source terminal: one live C run, `PHOTO_SLAM_RASTER_BACKEND=C`, same config and
  seed, no LOOP/GBA (0 `Loop Closure Detected`, 0 `[GBA-Apply]`), no append
  refinement. The log has 334 `LocalBundleAdjustmentWithLine` lines, but no
  optimizer-step/data-frame consumption mapping.
- Arms: `END_A` (backend A) and `END_C` (backend C), both restored from the same
  in-memory `RecoverySnapshot`.
- Camera poses frozen; Gaussian topology frozen at 450378 total / 2080 line
  Gaussians. Loss, opacity/scaling optimization and LR rules unchanged.
- Same 1230-step pre-generated sequence for both arms. Evaluation at 0, 100,
  400, 800 and 1230 real updates on 12 fixed CURRENT training KFs.
- The 12 views are training views, not a held-out set.

The requested earlier 12-frame set could not be reused at the live terminal:
only 4 of those images still existed as current KFs, so the run fell back to the
first 12 current map-order KFs. `eval_views.txt` records the exact list; this is
not the same 12-frame set as the r0 fixed-set table.

| updates | A native GS PSNR | C native GS PSNR | A−C |
|---:|---:|---:|---:|
| 0 | 33.4330 | 33.4330 | 0.0000 |
| 100 | 33.7872 | 33.7861 | +0.0011 |
| 400 | 35.5500 | 35.5401 | +0.0099 |
| 800 | 35.9617 | 35.9146 | +0.0471 |
| 1230 | 35.3638 | 35.3629 | +0.0009 |

Actual training opportunities in each arm (`train_levels.csv`, `train_sh.csv`):

- pyramid level 1: 335 updates
- pyramid level 2: 590 updates
- pyramid level 3 (1200×680): 305 updates
- active SH degree 3: all 1230 updates
- total topology: unchanged at 450378 / 2080

Both arms improve strongly from the terminal state (about **+1.93 dB** at 1230
updates; peak at 800 is about +2.53 dB), and A/C remain almost identical
(A−C ≤ 0.047 dB). There is **no sustained A/C divergence** after the shared
terminal state. The 1230-update point is lower than 800 in both arms; extra
training is not monotonic on these fixed views.

## 5. Interpretation against the pre-registered decision rules

1. **C terminal still has unfinished fitting?** Yes on these fixed training
   views: +1.93 dB after 1230 fixed-topology updates. This supports late
   training/opportunity loss as a contributor.
2. **Does A/C diverge continuously from the same state?** No: both improve and
   remain within 0.047 dB; the C-specific degradation branch is not supported.
3. **Are camera updates responsible for a quality jump?** No: the zero-motion
   frame still drops 3.61/10.50 dB, and all measured relative camera updates are
   tiny (<0.12°, <5 mm). The local-BA/camera-consistency branch is not supported
   by this pair.
4. **Does this prove the whole 1.75 dB end gap is a step-count deficit?** No.
   1230 is the prior A/C update difference, not equal compute; the arms are not
   compared to a separately trained A terminal; the 1230 point is not monotonic;
   and the evaluation views are training views.

The supported statement is narrower: **the C terminal state still contains
substantial fit headroom, and a shared A/C continuation from it evolves almost
identically. This is consistent with lost training opportunity in C's slow
interval, but does not establish that all of the online gap is step-count
caused.** The next fix should target the slow interval / lost training
opportunities rather than blindly increasing fixed refinement.

## 6. Next online run: minimum counters

A lightweight `PHOTO_SLAM_STAGE_COUNTERS=1` path was added. At each online
checkpoint and at natural end it writes:

- `stage_updates.csv`: per-pyramid-level and per-SH update counts, plus total
  training pixels;
- `kf_usage.csv`: per-KF used and remaining counts;
- `progress.csv`: input frame/total, mapper iteration, optimizer steps, KF and
  Gaussian counts, mapping-queue pending, shutdown/stopped flags, stop reason.

These counters were compiled and the existing contract test still passes, but
they were not enabled in the r1 run. LocalBA consumption step/frame and camera
change range are still **not available**; `progress.csv` labels that field
`not_available_in_this_build`. They must be added in the ORB-SLAM3 LocalBA path
or derived from a new event log; the existing logs cannot recover them.

The next necessary online run should enable `PHOTO_SLAM_STAGE_COUNTERS=1` and
add the LocalBA event record before making a causal claim about the C slow
interval.

## 7. Evidence and provenance

- `offline_rescore/` — eight checkpoint offline renders (`offline_psnr.csv`,
  `offline_summary.txt`); snapshots remain in `/tmp/psl-room1-control`.
- `offline_crosscheck.csv`, `fixed12_offline.csv`
- `camera_fixed12_1000_1500.csv`, `camera_fixed12_summary.csv`
- `model_stats.csv`
- `snapshot_completeness.txt`
- `undertrain_r1/` — `terminal_state.txt`, `summary.csv`, `eval_views.txt`,
  `run.log`, `END_A|END_C/metrics.csv`, per-step `per_frame.csv` (includes
  `psnr_gs`), `train_levels.csv`, `train_sh.csv`, `train_views.csv`,
  `env.txt`
- `undertrain_r0_eval3/` — first live attempt; only 3 requested fixed views
  survived, so it is retained but not used for conclusions.
- `instrumentation.patch` SHA-256
  `104a2d702f09885763818519980aa068ebb0286d028a075040d39e3a830f62ad`;
  it applies to production HEAD `ec85e80886e2f0494acb87781e075334bcf2ba58`.
- Current source/binary hashes: `source_bin_hashes.txt`.

Limitations: the r1 binary was overwritten by the later stage-counter build, so
its exact binary hash was not retained; the run is pinned by its log, source
patch and terminal state. The r1 terminal is not the same process state as the
r0 `end` checkpoint. The controlled arms are not an equal-compute A/C online
replay. No pre-500 checkpoint exists, so the earliest formation of the online
A/C difference remains bounded to before frame 500.
