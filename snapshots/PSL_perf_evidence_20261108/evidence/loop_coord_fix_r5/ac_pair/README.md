# room1 A/C diagnostic pair

Same candidate `src-fix`, same `room1_orb.yaml` / `room1_gaussian.yaml`,
`PHOTO_SLAM_SEED=0`, `PHOTO_SLAM_GBA_APPLY_CORRECTION=0`,
`PHOTO_SLAM_APPEND_STEPS=0`, `PHOTO_SLAM_LINE_MODE=2`, only
`PHOTO_SLAM_RASTER_BACKEND` differs. Both runs exited 0.

This is a diagnostic pair, not acceptance: A and C produced different tracking
trajectories (183 vs 182 KFs), and only C triggered a LOOP. The PSNR gap cannot
be attributed to the LOOP fix or to the raster backend.

| backend | KF | GS | line GS | steps | native mean/last | GS mean/last | LOOP | GBA |
|---|---:|---:|---:|---:|---:|---:|---|---|
| A | 183 | 546704 | 2725 | 10289 | 30.6536 / 28.3001 | 30.6610 / 28.3001 | none | none |
| C | 182 | 472461 | 818 | 9189 | 29.9071 / 24.7993 | 29.9150 / 24.8131 | 1 | 1 (zero delta) |

## Matched training frames (same `img_name`)

`common_frames.csv` contains every frame present in both `cameras.json` files and
in both PSNR files. There are **17** such frames (15 before `frame001300`, 2
after). The native GS PSNR difference is A−C:

- common-frame mean: A `32.3404`, C `31.3077`, **A−C `+1.0327 dB`**,
  median `+0.9844`; A higher on 13/17, C higher on 4/17.
- The 15 frames before `frame001300` and 2 after are an **image-time
  classification only**. Every score in this file was rendered at shutdown;
  the split does not show that quality was already different before the LOOP.
  The same applies to the `+1.0942 / +0.5709 dB` subgroup means; they are not
  event-time or historical-quality evidence.
- post-loop subgroup has only n=2, so it is not a LOOP-effect estimate.
- **Last matched frame** is `frame001625.jpg` (A camera id 163, C camera id
  163): native GS A `28.2450`, C `27.9186`, **A−C `+0.3264 dB`**; native
  ordinary A `28.2244`, C `27.8901`. This row is the last common data frame, not
  the last line of either PSNR file. The full-file means above remain the
  system-level description.

The two runs have different KF counts (183 vs 182), so only this matched subset
is paired; all other per-frame comparisons would be invalid.

C event position: LOOP current KF **124**, consumed at `ProcessNewKeyFrame mnId
125`; GBA follows that KF's LBA before KF 126. The log has no wall-clock
timestamps (last `Frame id: 1300` before the event), so only KF/frame positions
are reported. A has no LOOP/GBA.

C event lines:

```text
[LineLoop-Gaussian] candidates=1053 correctedLineGaussians=1053 kfsTotal=120
 kfsWithPkf=120 kfsCorrected=120 kfsSkippedByGate=0 corrected=371525
 skippedBad=0 skippedNull=0 duplicateSkipped=0 nonFinite=0
 maxTranslation=0.00539564 maxRotationDeg=0.155592
 meanTranslation=0.00174113 meanRotationDeg=0.0674584 meanScaleRatio=1
[LoopSync] evt=1 kfs=120 kfsPoseChanged=116 noScene=0 movedRows=371525
 nonzeroMovedRows=347164 lineRowsMoved=1053 unassignedRows=0
 missingEventRows=4703 staleRows=0 scale=1 count=376228->376228
[GBA-Apply] deformation DISABLED ... 121 deltas not applied
[GBA-Apply] evt=1 map=0 kfs=121 in_scene=121 deltas=121 moved=0
 nonzero_moved=0 line_moved=0 ... translation_mean=0.00000
 translation_max=0.00000 ... APPLIED
```

`APPLIED` means the payload was accepted; with zero translation and
`GBA_APPLY_CORRECTION=0`, no map row moved and the GBA is not evidence of a
nonzero correction. A has no `[LoopSync]` / `[GBA-Apply]` lines.

Command:

```bash
cd /tmp/psl-room1-control/src-fix
for b in A C; do
  env -u CUDA_LAUNCH_BLOCKING -u PHOTO_SLAM_RASTER_AGG_SELFCHECK \
      -u PHOTO_SLAM_RASTER_PROF -u PHOTO_SLAM_GPU_PROF \
      -u PHOTO_SLAM_FROZEN_ABLATION \
      PHOTO_SLAM_LINE_MODE=2 PHOTO_SLAM_GBA_APPLY_CORRECTION=0 \
      PHOTO_SLAM_APPEND_STEPS=0 PHOTO_SLAM_SEED=0 \
      PHOTO_SLAM_PSNR_DIAG=1 PHOTO_SLAM_PSNR_DIAG_FID=0 \
      PHOTO_SLAM_RASTER_BACKEND=$b \
      timeout --signal=TERM --kill-after=30s 1800 \
      bin/replica_rgbd \
        /workspace/code/PSL-perf/ORB-SLAM3/Vocabulary/ORBvoc.txt \
        /tmp/psl-room1-control/config/room1_orb.yaml \
        /tmp/psl-room1-control/config/room1_gaussian.yaml \
        /workspace/code/SEGS-SLAM/datasets/replica/room1 \
        /tmp/psl-room1-control/results/fix-pair-$b no_viewer
done
```

Files: `fix-pair-{A,C}.log`, `psnr-{A,C}.txt`, `psnr_gs-{A,C}.txt`,
`quad_metrics-{A,C}.txt`, `eval_guard_shutdown-{A,C}.txt`.
