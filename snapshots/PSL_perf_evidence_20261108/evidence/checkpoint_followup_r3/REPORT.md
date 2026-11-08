# room1 H candidate round (r3): A/C/H same build

## Boundary

- The formal quality/counter table is the `h-round3` same-build A/C/H batch.
  Earlier `h-round1`/`h-round2` were counter/dispatch smoke runs with incomplete
  stage or Adam-boundary files; they are not used for quality or stage claims.
- Existing extreme-row counterfactual, continuation and online checkpoint
  evidence are retained.
- r2's line-eccentricity statement is corrected: the original
  `mean(s_perp)/(s_parallel+1e-6)` term still acts on every line Gaussian, but
  the current snapshots and fixed anomaly-row counterfactual do **not support
  it as the main cause** of this round's drop.  It was not removed.
- The extreme-row opacity counterfactual is not a production pruning
  recommendation, and it is not expanded in this round.
- No loss, pruning threshold, precision, training budget or fixed refinement
  was changed.
- `input_end` (`frame_end`, input frame 2000 during refinement) and
  `natural_end` (after final refinement) are separate snapshots.

## 1. H candidate implementation

H is an independent experiment option, default off:

- `active_sh_degree < 3` -> existing C backward kernel;
- `active_sh_degree >= 3` -> existing A backward kernel.

No CUDA formula, loss, precision, densify or budget is changed.  The selection
is in host code in `BACKWARD::render`; the existing CPU `active_sh_degree_` is
passed via `SetRasterSh()`.  There is no GPU readback.  The selection is
resolved per backward call, not by the old one-time `g_raster_backend` cache.
The new env value is `PHOTO_SLAM_RASTER_BACKEND=H`; unset/A/C behavior remains
unchanged.

Host-side dispatch acceptance (`test_raster_h_dispatch`):

```text
sh=0 C, sh=1 C, sh=2 C, sh=3 A
explicit A/C pins still select A/C
PASS
```

`test_raster_h_backward` additionally runs one synthetic GPU backward with H at
SH=0 and the explicit C pin, then H at SH=3 and the explicit A pin; the four
output arrays are bitwise equal in both pairs (`PASS`, A=1/C=1).  Thus the
selected H path is the same existing kernel, not a third formula.  This test
does not replace the full run.  In the formal H run the four switches were:

```text
sh=0 C
sh=1 C
sh=2 C
sh=3 A
```

Actual H counters at natural end: A=7065, C=2999, switches=4, last=A/sh=3.
The stage counter backend counts are exactly A=7065, C=2999; their sum is the
10064 optimizer updates.  This confirms one selected backward backend per
update; selfcheck, golden and profilers were not enabled.  The H path calls the
same existing A/C launch functions, not a third formula.

## 2. Stage counters are real, not a plan

`PHOTO_SLAM_STAGE_COUNTERS=1` was set for every formal A/C/H run.  The files
are non-empty at input_end and natural_end:

- `stage_updates.csv`: levels, SH, backend and training pixels;
- `kf_usage.csv`: per-KF used/remaining;
- `progress.csv`: input frame, total, mapper iteration, optimizer steps,
  KFs/Gaussians, queue, shutdown/stopped, stop reason, LocalBA status;
- `adam_boundary.csv`: all six Adam group steps/LRs/param counts;
- `raster_h_counts.csv`: actual H backward counts and switches.

Natural-end stage table:

| arm | updates | pixels | level0/1/2/3 | SH0/1/2/3 | backend |
|---|---:|---:|---|---|---|
| A | 9770 | 2.939B | 2655/2526/1582/3007 | 999/1000/1000/6771 | A=9770 |
| C | 8772 | 2.639B | 2685/2113/1218/2756 | 999/1000/1000/5773 | C=8772 |
| H | 10064 | 3.182B | 2700/2506/1543/3315 | 999/1000/1000/7065 | A=7065, C=2999 |

All runs ended with `input_frame=2000/2000`, `queue_pending=0`,
`slam_shutdown=1`, `mapper_stopped=0`, `stop_reason=slam_shutdown`.
LocalBA consumption step/frame/camera range is still
`not_available_in_this_build`; the logs only give 354/356/360
`LocalBundleAdjustmentWithLine` lines, with no step mapping.

## 3. Input_end vs natural_end

| arm | input_end steps/KF | natural_end steps/KF | fixed-5 input -> natural | all-common input -> natural | full input -> natural |
|---|---:|---:|---:|---:|---:|
| A | 9769 / 178 | 9770 / 179 | 34.3997 -> 34.3817 | 33.6612 -> 33.2870 | 31.4687 -> 30.9933 |
| C | 8771 / 180 | 8772 / 181 | 31.1567 -> 31.1781 | 30.4873 -> 30.3740 | 28.6557 -> 28.5023 |
| H | 10063 / 181 | 10064 / 182 | 34.4491 -> 34.4147 | 33.1975 -> 32.4609 | 30.7579 -> 30.0720 |

The input-frame-2000 checkpoint is not the training terminal.  Full-set and
all-common means move when the KF set changes; the fixed-5 curve is the stable
comparison here.

## 4. Native C++ GS PSNR (offline after training)

All values are recomputed from the saved snapshots with
`eval_checkpoint_snapshot` and the existing C++ `psnr_gaussian_splatting`.
The fixed early set has only **5 images common to A/C/H** in this round; the
original 12 cannot be reported as a 12-frame curve because the others were
culled.  All-common n is the image-name intersection at that checkpoint.

| checkpoint | all-common n | A | C | H | fixed-5 A/C/H |
|---|---:|---:|---:|---:|---|
| 500 | 6 | 29.284 | 34.087 | 32.775 | 28.381 / 33.629 / 32.293 |
| 1000 | 8 | 34.653 | 35.940 | 35.055 | 35.454 / 36.746 / 35.741 |
| 1500 | 11 | 32.091 | 26.947 | 30.967 | 32.843 / 27.506 / 31.619 |
| input_end | 12 | 33.661 | 30.487 | 33.198 | 34.400 / 31.157 / 34.449 |
| natural_end | 12 | 33.287 | 30.374 | 32.461 | 34.382 / 31.178 / 34.415 |

Full-set natural-end means (different KF sets, descriptive only):
A `30.993`, C `28.502`, H `30.072`.  Frame0 natural-end:
A `28.746`, C `28.626`, H `32.435` (one frame; not a general claim).

## 5. Answers to the four questions

1. **Did H reduce C's loss of updates in the later phase?**  Yes relative to
   C: H has 10064 updates vs C 8772 (+1292) and 3.182B training pixels vs
   2.639B (+20.6%).  H uses C for 2999 updates and A for 7065.  But H is also
   +294 updates over A (9770) while its all-common quality is lower, so extra
   updates alone do not explain the result.
2. **Is H's native training PSNR close to same-batch A?**  On the fixed-5
   early set, H is essentially A at input/natural end (+0.049/-0.034 dB) and
   better than C; on all 12 common images H is **0.826 dB below A** at natural
   end; full-set H is 0.921 dB below A.  It is not a general match.
3. **Does H preserve early benefit and avoid late regression?**  Partly.  At
   500/1000 H is 1.33/1.01 dB below C on fixed-5 (it does not fully preserve
   C's early advantage), and at 1500 it is 1.22 dB below A but 4.11 dB above C.
   At end it is approximately A on fixed-5, but below A on all-common/full.
   The frame0 improvement is local.
4. **Do LOOP/GBA or KF differences still limit causality?**  Yes.  A has zero
   loop/GBA; C and H each have one loop and one GBA application.  C: 121 KFs /
   121 deltas; H: 117/117.  The GBA deformation was disabled, but camera poses
   were updated; `moved=0`, `pose_mismatch=0`.  LocalBA consumption is now
   recorded per event with step/iteration/KF range and pose deltas; it does not
   account for the per-100 Adam gap.  H-vs-A is therefore confounded, and
   H-vs-C has different event KFs/deltas.

Run cost (not a speed claim): total operation A/C/H = `89.02 / 89.05 / 89.99 s`;
mapping = `65.82 / 66.90 / 65.16 s`; KFs = `179 / 181 / 182`; Gaussians =
`496296 / 466707 / 531605`; line Gaussians = `1422 / 1028 / 836`.

Snapshot capture overhead is recorded separately in `capture_overhead.csv`:
input-frame save 0.03-0.12 s and read-only render 0.52-2.83 s per checkpoint;
natural-end save 0.07-0.11 s with no render.  Stage-counter host file writes
were not separately timed, and the capture necessarily perturbs online timing;
the quality runs are diagnostic, not zero-overhead production timings.

## 5b. LocalBA consumption record

`adam_topo_diag/localba_consumption.csv` is a payload-only record written just
before the LocalBA pose update.  The short A run produced 33 rows in 3 events:

| event | optimizer_steps | iteration | input_frame | KFs | KF range | max rot | max trans |
|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | 1 | 1 | 114 | 10 | 0-9 | 0.1782 deg | 0.006477 |
| 2 | 1 | 1 | 114 | 11 | 0-10 | 0.0790 deg | 0.001965 |
| 3 | 124 | 124 | 122 | 12 | 0-11 | 0.1457 deg | 0.005837 |

Each row records `old/new` quaternion and translation, `rot_deg`,
`trans`, `map_points` and `line_points`; event 2 also exercises the new-KF
path, whose old-pose columns are empty and whose new pose is populated.  The
record does not read the GPU and changes no pose.  It confirms that LocalBA
consumption happens at optimizer steps 1/1/124, not at every 100-step boundary,
so it cannot explain the per-100 Adam-step deficit.

## 5c. Global/loop GBA actual increments

`event_increment_summary.csv` was regenerated from the actual event dumps:

| arm | loop KFs | pose changed | moved rows | nonzero moved | line rows | max trans | max rot | global deltas / moved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 |
| C | 119 | 0 | 359170 | 0 | 1055 | 2.67e-07 | 5.60e-06 | 121 / 0 |
| H | 116 | 112 | 374605 | 336452 | 1084 | 7.10e-03 | 0.2162 | 117 / 0 |

The C loop event is numerically zero at grid/float precision; the H loop event
is not: 112 KFs moved, with mean translation 0.00275 and maximum rotation
0.216 deg.  Both global-GBA applications report `moved=0`, and
`pose_mismatch=0` only means the checked pose entries matched, not that every
increment was zero.  A complete per-KF pre-event pose map was not saved for
C/H; the event dump now preserves the payload (`PHOTO_SLAM_EVENT_DUMP_DIR`) and
the existing `PHOTO_SLAM_GBA_RECOVERY` path can capture live `PoseOnly`/`Map`
state if another run is authorized.  Until then, C/H remain event-confounded.

## 6. Judgment

H is a **partial positive candidate**, not a validated system improvement:

- it recovers C's late fixed-frame behavior and does not regress below A on the
  fixed-5 set at the end;
- it does not beat A on all-common 12 or full-set quality;
- it loses part of C's early advantage;
- it is not faster in wall time and uses more training pixels;
- LOOP/GBA/KF differences prevent a clean causal attribution.

Therefore the result does **not** justify replacing A.  A second swapped-order
round was not run because H did not show a clear overall advantage over A; the
negative/partial result is retained.  Normal operation should continue with A.
No threshold scan, extra fixed refinement, loss change or pruning change was
made.

## 7. Adam per-100 diagnosis

`adam_step_audit.txt` showed the host-vs-Adam difference is already 23-28 at
frame 500 and grows to 97/87/100 for A/C/H at natural end; all six groups share
the same step.  A short A-only run (`PHOTO_SLAM_DIAG_STOP_STEPS=200`) now
records tensor identity, gradient presence and Adam step at four boundaries:
`after_backward`, `after_topology`, `before_step`, `after_step`.

| iteration | boundary | rows | grad defined | Adam step | host steps | TensorImpl |
|---:|---|---:|---:|---:|---:|---|
| 100 | after_backward | 3029 (all 6) | 1 | 99 | 99 | initial |
| 100 | after_topology | 3419 (all 6) | 0 | 99 | 99 | **all 6 changed** |
| 100 | before_step | 3419 (all 6) | 0 | 99 | 99 | unchanged |
| 100 | after_step | 3419 (all 6) | 0 | 99 | 100 | unchanged |
| 200 | after_backward | 3989 (all 6) | 1 | 198 | 199 | initial |
| 200 | after_topology | 4322 (all 6) | 0 | 198 | 199 | **all 6 changed** |
| 200 | after_step | 4322 (all 6) | 0 | 198 | 200 | unchanged |

`prunes.csv` reports row changes `3029->2737`, `4101->3419`,
`3989->3430`, `5214->4322`, and `replacements.csv` records tensor 5 being
replaced at steps 50/99/149/198.  The runtime log says
`densify_from_iter=0` even though the YAML says 600: the code reads
`Optimization.densify_from_iter_` while the YAML key is
`densify_from_iter`, so topology starts at iteration 100 rather than 600.

Conclusion: the 100-boundary order is `backward -> topology replacement ->
optimizer.step()`.  The topology replacement swaps every one of the six
parameter tensors and their gradients are undefined; Adam's state remains on
the new tensors but sees `grad=None`, so the step is a no-op for all groups
while the host step counter still increments.  This exactly accounts for the
100th-update loss at each topology boundary (and the observed 198 vs 199 at
iteration 200), and it is a pre-existing order, not an H/C regression.

The `densify_from_iter` key mismatch is a real configuration defect, but it is
recorded only: moving the optimizer step before topology, copying parent
gradients across the replacement, or changing the densify schedule would change
training behaviour, which this convergence task forbids.  No optimizer,
densification, loss, LR or refinement change was made.

## 8. Artifacts

- `quality_summary.csv`, `fixed_early_curve.csv`, `common_frames.csv`
- `stage_summary.csv`, `event_summary.csv`, `event_increment_summary.csv`,
  `adam_step_audit.txt`
- `adam_topo_diag/`: `adam_topo.csv`, `replacements.csv`, `prunes.csv`,
  `runtime_config.txt`, `localba_consumption.csv`, `run.log`, `out/`
- `offline/<arm>_<checkpoint>/offline_psnr.csv`, `offline_summary.txt`
- `runs/<arm>/run.log`, timing files
- `checkpoints/<arm>/<checkpoint>/` metadata, cameras, masks ids, stage,
  Adam, progress and KF usage files
- `snapshot_sha256.txt` (snapshots remain in
  `/tmp/psl-room1-control/h-round3/<arm>/<checkpoint>/snapshot.pt`)
- `bin/`, `libgaussian_mapper.so`, `libcuda_rasterizer.so`, `config/` and
  `artifact_manifest.sha256` preserve the exact formal build.
- `diagnostic_build/` and `diagnostic_manifest.sha256` preserve the later
  diagnostic build containing the Adam/LocalBA/event instrumentation.
- `test_raster_h_dispatch` and `test_raster_h_backward` source/log/binary.
- `instrumentation.patch` SHA-256
  `3b3f2d4caf50d7c4ef66590ac3dbe38da2e628cc5678d6390e47eb7b7be0fdb1`;
  it applies to production HEAD
  `ec85e80886e2f0494acb87781e075334bcf2ba58`.

Limitations: one formal round only; no swapped-order round because H did not
show a clear overall advantage; fixed curve has n=5; event confound;
center/state metrics not a Shapley decomposition; offline scoring is on saved
snapshots, not a new online run.  The GBA payload entry was added and compiled
but no complete per-KF pre-event pose map was captured for C/H.  The Adam
diagnosis used a 200-step A-only run; it identifies the mechanism but was not
replayed through a full 50k run.
