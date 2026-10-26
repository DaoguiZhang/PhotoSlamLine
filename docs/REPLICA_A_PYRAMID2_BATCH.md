# Replica eight-scene pyramid2 batch

Date: 2026-10-26  
Scope: Replica RGB-D, eight scenes. This is a configuration correction, not a
new acceleration algorithm.

## Configuration rule

Each scene was copied from its own stable canonical config under
`cfg/gaussian_mapper/RGB-D/Replica/A_baseline/<scene>.yaml` to
`cfg/gaussian_mapper/RGB-D/Replica/A_pyramid2/<scene>.yaml`.

The only changed key is:

```diff
-GausPyramid.num_sub_levels: 3
+GausPyramid.num_sub_levels: 2
```

Room1's config was not copied to other scenes. The original `A_baseline`
configs remain untouched.

## Source and runtime identity

- Base repository commit: `ec85e80886e2f0494acb87781e075334bcf2ba58`
- Frozen source snapshot SHA256:
  `1c1b0c71c2f4bf88a97f29194e8c911b4acef691260239dae4920fe3dec42027`
- Frozen binary SHA256:
  `5919857e7a89947b2c5256a8d47ccaf7ca6952f65a9f96218754543d9989d6ae`
- All training used the frozen binary and its companion project libraries.
- No rebuild was used for the training batch.

## Runtime settings

```text
PHOTO_SLAM_RASTER_BACKEND=A
PHOTO_SLAM_LINE_MODE=2
PHOTO_SLAM_LINE_LOOP=0
PHOTO_SLAM_GBA_APPLY_CORRECTION=0
PHOTO_SLAM_APPEND_STEPS=0
PHOTO_SLAM_SEED=0
viewer=no_viewer
diagnostics/checkpoints/selfcheck/profiler=off
```

Serial scene order:

`office0 office1 office2 office3 office4 room0 room1 room2`

## Export completeness

All eight training processes exited 0. Each result contains:

- full Gaussian parameters:
  `*_shutdown/ply/point_cloud/iteration_*/point_cloud.ply`
- camera metadata: `*_shutdown/ply/cameras.json`
- `CameraTrajectory_TUM.txt`
- `KeyFrameTrajectory_TUM.txt`
- `Global_Final_Map_Points.obj`
- `TotalOperationTime.txt`
- `run.log`
- `natural_end/cameras.csv`
- `natural_end/meta.txt`

`input.ply` was not used as the completeness criterion. Internal result
directories were not moved.

## Training GS PSNR and Photo-SLAM-eval

Training values are the native `psnr_gaussian_splatting` full-run means.
Evaluation values come from Photo-SLAM-eval after training on the same result
trees.

| scene | train p2 | train p3 | Δ train | eval p2 | eval p3 | Δ eval |
|---|---:|---:|---:|---:|---:|---:|
| office0 | 40.438919 | 39.450982 | +0.987937 | 39.348686 | 37.892439 | +1.456247 |
| office1 | 41.571293 | 40.430663 | +1.140630 | 39.529468 | 39.060943 | +0.468525 |
| office2 | 32.933991 | 30.100643 | +2.833348 | 32.009902 | 29.340560 | +2.669342 |
| office3 | 33.196308 | 31.454170 | +1.742138 | 32.218576 | 31.112703 | +1.105873 |
| office4 | 36.549065 | 34.564024 | +1.985041 | 35.457077 | 33.182715 | +2.274362 |
| room0 | 31.455996 | 30.786618 | +0.669378 | 30.461621 | 30.343413 | +0.118208 |
| room1 | 35.510036 | 31.984953 | +3.525083 | 33.759717 | 30.907537 | +2.852180 |
| room2 | 35.989110 | 34.406182 | +1.582928 | 34.911356 | 33.067907 | +1.843450 |
| mean | 35.955590 | 34.147279 | **+1.808310** | 34.712050 | 33.113527 | **+1.598523** |

All eight scenes improve in both training GS PSNR and Photo-SLAM-eval PSNR.
SSIM is higher and LPIPS is lower for all eight. Translation APE is mixed
(3 scenes better, 5 worse), so trajectory accuracy is not claimed as a uniform
improvement.

## Independent source build check

The source snapshot was copied to an isolated temporary directory and built
without reusing the frozen `bin/` or `lib/`.

Built successfully:

1. `ORB-SLAM3/Thirdparty/DBoW2`
2. `ORB-SLAM3/Thirdparty/g2o`
3. `ORB-SLAM3/Thirdparty/Sophus`
4. `ORB-SLAM3`
5. top-level `replica_rgbd`

The fresh `replica_rgbd` SHA256 was
`d8c5f105061ec204f147e634a7d1e3440fbd27c63b948fe2c65213b5cba05147`.
It differs from the frozen binary because it is a fresh build in a separate
directory; the build check only establishes that the uploaded source tree is
self-contained for compilation.

`ORBvoc.txt` is runtime data, not a build input. It is intentionally not
committed; pass it with `--vocab` (single-scene script) or `VOCAB=...`
(eight-scene batch script).

## Limits

- Verified on the eight Replica RGB-D scenes only.
- Not claimed to generalize to TUM/EuRoC/ETH3D or other sensor modes.
- Not a system-level speedup claim.
