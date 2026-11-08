# room1 LOOP → Gaussian mapping coordinate-consistency fix (evidence)

## Event

- Event type: `MappingOperation::LoopClosingBA`
- ORB-SLAM3 payload event id: `0` (LoopClosingBA does not set one)
- Mapper-local coordinate event id: `1` (`[LoopSync] evt=1`)
- Run: `/tmp/psl-room1-control/src-fix-audit`, backend `C`, room1, seed 0,
  `PHOTO_SLAM_LINE_MODE=2`, `PHOTO_SLAM_GBA_APPLY_CORRECTION=0`.
- The event is real, not synthetic. `event_scope.csv` has 117 scene KFs;
  113 changed pose. Max translation `0.0361479 m`, max rotation
  `0.899438 deg`; mean `0.0185401 m` / `0.605442 deg`. Both maxima are below
  the old gate (`0.10 m`, `2 deg`), so the legacy path skipped every map
  correction.

## Same-event quality (24 fixed views, no optimizer step)

`summary.csv`:

| arm | mean PSNR | delta vs before |
|---|---:|---:|
| before camera + before model | 30.58399487 | 0 |
| after camera + before model (legacy gate-skipped map) | 23.47381488 | **-7.110179981** |
| after camera + corrected model (fixed path) | 28.98292383 | **-1.60107104** |

Recovery: `+5.50910894 dB`; residual `-1.60107104 dB`.
23 of 24 views moved. Moved-view means: before `30.574`, legacy `23.1547`,
fixed `28.9052`; worst fixed view fid 55 `-5.16902 dB`.

## Production correction coverage (same event)

`run.log`:

```
[LineLoop-Gaussian] candidates=1262 correctedLineGaussians=1262
 kfsTotal=117 kfsWithPkf=117 kfsCorrected=117 kfsSkippedByGate=0
 corrected=386317 skippedBad=0 skippedNull=0 duplicateSkipped=0
 nonFinite=0 maxTranslation=0.0361479 maxRotationDeg=0.899438
 meanTranslation=0.0172877 meanRotationDeg=0.605443 meanScaleRatio=1
[LoopSync] evt=1 kfs=117 kfsPoseChanged=113 noScene=0
 movedRows=386317 nonzeroMovedRows=351371 lineRowsMoved=1262
 unassignedRows=0 missingEventRows=3801 staleRows=0 scale=1
 count=390118->390118
```

The row count is unchanged. `missingEventRows=3801` are rows whose
`ref_kf_id_` is a scene KF absent from this LOOP payload; they are deliberately
left in their old frame (reported, not guessed).

## Clean LOOP regression (no event diagnostics)

- `clean_loop_regression.log` — real LOOP, no diagnostic env.
- `clean_loop_quad_metrics.txt`.
- `clean_loop_psnr_gaussian_splatting.txt`.

```
[LoopSync] evt=1 kfs=118 kfsPoseChanged=117
 movedRows=373284 nonzeroMovedRows=346367 lineRowsMoved=1107
 unassignedRows=0 missingEventRows=4927 staleRows=0 scale=1
 count=378211->378211
```

`optimizer_steps=9146`, `gaussians=491017`, `line_gaussians=1143`,
`mapping_seconds=71.373`; PSNR n=180 mean `29.2972`, last `25.8999`.

## Files

- `summary.csv` — same-event arm means.
- `step0_per_view.csv` — per-view PSNR/SSIM/camera delta.
- `event_scope.csv` — per-KF loop delta.
- `run.log` — full diagnostic run log.
- `clean_loop_regression.log` / `clean_loop_*` — clean LOOP regression.
- `gba_evt1_summary.csv` — the subsequent GBA event, included only to show the
  still-disabled GBA deformation is a separate remaining defect.

## Reproduce

```bash
SRC=/tmp/psl-room1-control/src-fix-audit \
OUT=/tmp/psl-room1-control/repro-loop-fix \
bash scripts/repro_loop_coord_fix.sh
```

The diagnostic event audit intentionally runs counterfactual arms on the
pre-event snapshot; its final PSNR is not a production regression metric. The
production regression must run without `PHOTO_SLAM_EVENT_QUALITY_DIR` and
`PHOTO_SLAM_COMMON_ABC_DIR`.
