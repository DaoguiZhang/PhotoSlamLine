# gba-recovery-v1 — controlled two-arm GBA correction experiment

Base commit: `9be7e400e1` (applyGbaCorrections) + this round's harness.
Worktree at run time: modified experiment code, built with
`scripts/build_two_level.sh` (ORB-SLAM3 first, then top level; both DSOs
reported the same `MappingOperation` destructor layout `0x27d`).

## 1. Single real event and shared input

The experiment triggers inside the first `GlobalBA` event, **before** the
production correction, and restores the pre-event state afterwards.

Event (from the run's `[GBA-Apply]` line):

```
evt=1 map=0 kfs=245 in_scene=245 deltas=245
moved=104159 nonzero_moved=99881 line_moved=238
translation_mean=0.00334 m translation_max=0.17229 m
ref_in_event=104159 ref_scene_not_event=169
ref_absent_scene=0 ref_unassigned=0 ref_newer_than_event=0
count_before=104328 count_after=104328 created_during_event=0
```

The 169 not-corrected rows are anchored to keyframes that are present in the
Gaussian scene but absent from the event payload (keyframes marked bad/culled at
GBA time). They are reported, not assigned a guessed anchor. No row is anchored
to a keyframe created after the event (`ref_newer_than_event=0`). The production
`sync_ms` in this experiment run included the offline arms because the timer
started before the call; the timer was moved after the experiment (post-run fix,
production-only log field). The arm timings below are measured inside the arms.

Saved in-process before any correction: all Gaussian parameter tensors, the six
Adam `(step, exp_avg, exp_avg_sq)` states, `iteration_`, `active_sh_`/
`default_sh_`, EMA loss, `kfs_used_times_`, and per-keyframe pose /
`remaining_times_of_use_` / Gaussian-pyramid budget / tracking error. The event
payload (per-KF `before`, `after`, `delta` 4x4) is written to
`event_evt1.txt`; the common-state identity to `common_state.txt`.

## 2. Repeat check

Two PoseOnly arms ran 10 steps from the same snapshot:

```
step0_param_exact,1
step0_adam_exact,1
schedule_exact,1
quality_tol,1
post_step_param_max_abs_diff,0.0390588
post_step_adam_max_abs_diff,0.00103617
match,1
```

The start state (parameters + Adam), the fixed render at step 0 and the complete
schedule/training-view sequence are identical. Post-step tensors drift because the
CUDA rasteriser backward uses non-deterministic atomic reduction order; that
drift is reported but is **not** a restore failure. It is why the experiment
compares 500-step curves rather than requiring bitwise equality after training.

## 3. Arms and controls

- **PoseOnly**: corrected camera poses for all keyframes; Gaussians keep pre-event geometry.
- **PoseAndMap**: identical corrected poses + `applyGbaCorrections` (ref-keyframe grouped SE(3): centres, quaternion, line anchor, line direction, line cache; xyz Adam first moment rotated; scaling and SH untouched).
- Both arms: same initialization, same 500-step pre-generated training-view sequence, same 24 evenly spaced evaluation views (excluded from training), 500 real optimizer updates, backend C, structural line loss weights 0, opacity reset 0, fixed Gaussian count (no add/clone/split/prune).
- Order 1: PoseOnly → PoseAndMap. Order 2 (same snapshot): PoseAndMap → PoseOnly.
- Evaluation at completed steps 0, 10, 50, 100, 250, 500. Step 0 is after the camera update and the arm-specific sync.

## 4. Results

Paired deltas = PoseAndMap − PoseOnly.

Order 1:

| step | dPSNR (dB) | dSSIM | dLPIPS |
|---:|---:|---:|---:|
| 0 | **−0.0231** | **−0.001227** | **+0.001110** |
| 10 | −0.0258 | −0.001291 | +0.000943 |
| 50 | −0.0372 | −0.001373 | +0.002018 |
| 100 | +0.0156 | −0.000408 | +0.000618 |
| 250 | −0.0483 | −0.000680 | +0.000419 |
| 500 | −0.0485 | −0.000161 | +0.000343 |

Order 2 (order swapped, same snapshot):

| step | dPSNR (dB) | dSSIM | dLPIPS |
|---:|---:|---:|---:|
| 0 | −0.0231 | −0.001227 | +0.001110 |
| 10 | −0.0259 | −0.001291 | +0.000946 |
| 50 | −0.0373 | −0.001445 | +0.002042 |
| 100 | +0.0211 | −0.000677 | +0.000784 |
| 250 | −0.0595 | −0.000796 | +0.001440 |
| 500 | −0.0023 | +0.000162 | −0.000051 |

Absolute endpoints (order 1): PSNR 14.845 → 20.923 (PoseOnly) vs 14.822 →
20.874 (PoseAndMap); SSIM 0.5788 → 0.7260 vs 0.5775 → 0.7259; LPIPS 0.5464 →
0.3341 vs 0.5475 → 0.3344.

Timing (recovery = sync + training, evaluation excluded):

| order | arm | sync s | train s | total s |
|---|---|---:|---:|---:|
| 1 | PoseOnly | 0.0157 | 3.6787 | **3.6944** |
| 1 | PoseAndMap | 0.1074 | 3.6796 | 3.7869 |
| 2 | PoseOnly_r2 | 0.0158 | 3.7128 | **3.7286** |
| 2 | PoseAndMap_r2 | 0.1057 | 3.7468 | 3.8525 |

Curves: `reports/gba_recovery_v1/curves/psnr_vs_step.png`,
`ssim_vs_step.png`, `lpips_vs_step.png`, `psnr_vs_total_time.png`.

## 5. Decision (honest, negative result)

1. **Immediate state after sync (step 0)**: PoseAndMap is slightly *worse* in both
   orders (PSNR −0.023 dB, SSIM −0.0012, LPIPS +0.0011). The correction did not
   improve the synchronized reconstruction state for this event.
2. **Same actual steps**: no stable gain. Most points are slightly worse; the only
   positive PSNR point (step 100) is not reproduced as a consistent advantage, and
   the step-500 difference changes sign between orders (−0.0485 vs −0.0023 dB),
   i.e. within run-to-run variation.
3. **Same total recovery time**: PoseAndMap spends ~0.09–0.12 s more on the
   correction and is not more accurate at that time. It is not an acceleration.

The event is small (mean translation 3.3 mm), so the null/slightly-negative result
is plausible: the single-anchor rigid approximation adds map motion without
reducing the error that dominates the short recovery window. Per the pre-registered
rule, no quality threshold time is reported and no threshold is selected after the
fact. Correctness fixes are retained; no further modules are added to chase a
positive result.

## 6. Limitations

- One event, one sequence (TUM fr3_long_office_household), one pair per order.
- 24 fixed evaluation keyframes, evenly spaced; they are excluded from training.
- The renderer returns RGB only, so valid-depth error is **not available**; it is
  not substituted.
- LPIPS uses AlexNet on CPU because the image's CUDA/cuDNN combination rejects the
  LPIPS forward (`CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`); it is computed on
  the exact saved render/GT PNGs.
- Single-reference-keyframe rigid approximation: all Gaussians anchored at one
  keyframe receive the same SE(3) delta; per-point GBA adjustment is not reproduced.
- Initial Gaussian colours come from cached MapPoint/MapLine colours, not a
  photometric optimisation.

## 7. Raw artifacts and command

```
reports/gba_recovery_v1/summary.csv
reports/gba_recovery_v1/lpips.csv
reports/gba_recovery_v1/repeat_check.csv
reports/gba_recovery_v1/common_state.txt
reports/gba_recovery_v1/event_evt1.txt
reports/gba_recovery_v1/<arm>/metrics.csv
reports/gba_recovery_v1/<arm>/per_frame_eval_step*.csv
reports/gba_recovery_v1/curves/*.png
```

Run command:

```bash
cd /workspace/code/PSL-perf
./scripts/build_two_level.sh
PHOTO_SLAM_RASTER_BACKEND=C PHOTO_SLAM_LINE_LOSS_EVERY=0 \
PHOTO_SLAM_GBA_RECOVERY=abl_out/gba_recovery_v1 \
PHOTO_SLAM_GBA_RECOVERY_STEPS=500 \
PHOTO_SLAM_GBA_RECOVERY_EVAL_STEPS=0,10,50,100,250,500 \
PHOTO_SLAM_GBA_RECOVERY_VIEWS=24 \
  bin/tum_rgbd ORB-SLAM3/Vocabulary/ORBvoc.txt \
  cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
  cfg/gaussian_mapper/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
  /workspace/code/SEGS-SLAM/datasets/tum/rgbd_dataset_freiburg3_long_office_household \
  cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg3_long_office_household.txt \
  /tmp/gba_recovery_run no_viewer 2>&1 | tee /tmp/gba_recovery.log
python3 scripts/lpips_gba_recovery.py abl_out/gba_recovery_v1 --device cpu
python3 scripts/plot_gba_recovery.py abl_out/gba_recovery_v1
```
