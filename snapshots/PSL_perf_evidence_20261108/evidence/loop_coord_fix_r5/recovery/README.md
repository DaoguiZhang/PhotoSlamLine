# Recovery / coverage / GBA evidence

## Legacy vs Fixed 100 steps (real LOOP)

`recovery/loop_recovery_summary.csv`

| arm | step | ordinary PSNR | psnr_gaussian_splatting |
|---|---:|---:|---:|
| Legacy PoseOnly | 0 | 28.1042196 | 28.1115733 |
| Legacy PoseOnly | 100 | 31.43890246 | 31.44993854 |
| Fixed PoseAndMap | 0 | 28.90320388 | 28.91266044 |
| Fixed PoseAndMap | 100 | 31.53920007 | 31.55057915 |

All four arms have camera fingerprint `340f17cbe078514f`.
Event: `[LoopSync] evt=1 kfs=114 kfsPoseChanged=111
movedRows=387773 nonzero=360934 line=610 missingEventRows=5047`.

## Coverage / replay

`recovery/replay_coverage.log`:

```
[LoopCoverage] evt=1 rows=385356 refInEvent=373356
 refSceneNotEvent=12000 refAbsentScene=0 refUnassigned=0
 refNewerThanEvent=0 kfsNoScene=0
[LoopReplay] evt=1 same_moved=0 same_stale=373356
 older_moved=0 older_stale=373356
```

`recovery/missing_visibility.csv` is a **visibility count only**, not an
occlusion-contribution measurement. It shows fid56 with 11667 missing-ref rows
visible; fid55 is not in this 24-view grid.

## GBA

- r5 GBA step0 (event quality, ordinary PSNR): before 28.2804,
  pose-only 24.0143 (−4.2662), corrected map 29.2305 (+0.9501). No 100-step
  nonzero GBA recovery completed.
- GBA recovery runs r2/r4 reached a GBA event after a real LOOP, but the GBA
  payload had zero pose deltas (`PoseAndMap moved=0`); they are not evidence of
  a GBA-induced drop.
