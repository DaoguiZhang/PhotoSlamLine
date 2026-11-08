# geom-sync-v3 evidence — real GBA applies to existing Gaussians

Build: `scripts/build_two_level.sh` (ORB-SLAM3 then top level).
Both DSOs report `MappingOperation::~MappingOperation` size `0x27d`.

## 1. Contract self-test (`PHOTO_SLAM_SYNC_CONTRACT_TEST=1`, CPU+GPU)

```
[SyncContract] Rz90+trans     recover=0.00e+00 direction=0.00e+00 anchor=0.00e+00 old_expression=2.00e+00
[SyncContract] GPU centre identity       max_abs_vs_cpu=0.00e+00 OK
[SyncContract] GPU centre translation    max_abs_vs_cpu=0.00e+00 OK
[SyncContract] GPU centre Rz90+trans     max_abs_vs_cpu=0.00e+00 OK
[GbaContract][diag] row0=(0.2000,0.9000,2.9000) expected=(0.2000,0.9000,2.9000)
[GbaContract] centre=1 dir=1 anchor=1 residual=1 rows=1 scale=1 covariance=1 cache=1 counts=1 duplicate=1 order=1 : PASS
[SyncContract] worst recover=0.00e+00 direction=0.00e+00 anchor=0.00e+00 gpu_centre=0.00e+00 | old expression error=2.00e+00 | gba_apply=PASS | PASS (new entry)
```

Covers: rigid centre/anchor/rotation, line direction, line residual, scale
invariance, covariance `Σ' = R Σ Rᵀ`, unselected rows unchanged, line-cache
refresh, duplicate replay (`moved=0, stale=2`), and two-event composition order.

## 2. TUM fr3_long_office_household — full sequence, exit 0

Command (same verified entry point):

```bash
PHOTO_SLAM_RASTER_BACKEND=C PHOTO_SLAM_LINE_LOSS_EVERY=0 \
  bin/tum_rgbd ORB-SLAM3/Vocabulary/ORBvoc.txt \
  cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
  cfg/gaussian_mapper/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
  /workspace/code/SEGS-SLAM/datasets/tum/rgbd_dataset_freiburg3_long_office_household \
  cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg3_long_office_household.txt \
  /tmp/gba_tum_final no_viewer
```

Event sequence (full log `reports/gba_tum_final.log`):

```
GBA: Correct MapPoints and MapLines
Local Mapping RELEASE
 ProcessNewKeyFrameWithLine-> mpCurrentKeyFrame: mnId 229
LocalMapping:: RunWithLine: start LocalBundleAdjustmentWithLine_Optimization_Plucker_Reg
[GBA-Apply] evt=1 map=0 kfs=228 in_scene=228 deltas=228 moved=77170 line_moved=134
 stale=0 unassigned=30 missing_event=116 groups=227 scene_missing=0
 pose_mismatch_max=0.0000 count_before=77316 count_after=77316
 created_during_event=0 lm_map_points_created=104096 sync_ms=79.9 APPLIED
LocalMapping:: RunWithLine: end LocalBundleAdjustmentWithLine_Optimization_Plucker_Reg
```

After the event the run continued normally through KF 230, 231, ... and exited 0:

```
 ProcessNewKeyFrameWithLine-> mpCurrentKeyFrame: mnId 230
 ProcessNewKeyFrameWithLine-> mpCurrentKeyFrame: mnId 231
...
[System] 全局地图导出完成！
```

Post-event keyframes processed: 18. `created_during_event=0` is also enforced by
`TORCH_CHECK(count_after == count_before)` inside the callback.

Interpretation of the counters:
- `moved=77170` rows (`line_moved=134`) were actually transformed.
- `unassigned=30` rows have no reference keyframe (-1) and are reported, not guessed.
- `missing_event=116` rows are anchored to keyframes absent from the GBA payload
  (e.g. culled); they are reported and left untouched.
- `stale=0`: no row was already at this event version.
- `pose_mismatch_max=0`: the event's `Tcw_before` equals the scene's pre-event pose.

## 3. office0 regression (Replica RGB-D, GPU path)

`bin/replica_rgbd ... office0` — exit 0, `RefKf` warnings 0, `Torch Error` 0,
normal trajectory/global-map export. No loop closure in this run.

## Limitations (explicit)

- Only the RGB-D SE(3) correction is implemented; mono Sim(3), a new local-BA
  strategy and higher-order SH rotation keep their existing paths.
- Deformation is a per-reference-keyframe rigid approximation: every Gaussian
  anchored at one keyframe receives the same delta. It does not reproduce the
  per-map-point GBA adjustment exactly.
- Rows with unknown or missing anchors are not moved; the counts are logged.
