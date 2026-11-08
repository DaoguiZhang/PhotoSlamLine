# Replica eight-scene pyramid2 batch report

Date: 2026-10-26

## Scope

- Eight Replica RGB-D scenes, one run each, serial.
- Each scene used its own stable canonical config copied to
  `evidence/replica_A_pyramid2_20261026/config/<scene>_pyramid2.yaml`.
- Only change: `GausPyramid.num_sub_levels: 3 -> 2`.
- Frozen A binary and companion libraries; no rebuild for training.
- No loss/LR/densify/optimizer changes.

## Run identity

- Frozen binary:
  `/workspace/code/PSL-perf/evidence/frozen_A_clean_20261023/bin/replica_rgbd`
- SHA256:
  `5919857e7a89947b2c5256a8d47ccaf7ca6952f65a9f96218754543d9989d6ae`
- Result root:
  `/workspace/code/PSL-perf/results/replica_A_pyramid2_20261026_A_pyramid2/replica_rgbd_0`

All eight training processes exited 0. The raw `summary.csv` recorded
`export_ok=0` because the first shell checker used `find -size +1000000`
(512-byte blocks) instead of `-size +1000000c`. The checker bug is fixed in the
saved runner. The corrected check in `export_check.csv` shows all eight pass.

## Export completeness

Each scene has:

- full Gaussian PLY under
  `*_shutdown/ply/point_cloud/iteration_*/point_cloud.ply`
- `*_shutdown/ply/cameras.json`
- `CameraTrajectory_TUM.txt`
- `KeyFrameTrajectory_TUM.txt`
- `Global_Final_Map_Points.obj`
- `TotalOperationTime.txt`
- `run.log`
- `natural_end/cameras.csv`
- `natural_end/meta.txt`

No internal directory was moved; `input.ply` was not used as the criterion.

## Training / native GS PSNR and Photo-SLAM-eval

Training values are full-run native `psnr_gaussian_splatting` means.
Evaluation values come from Photo-SLAM-eval on the same result trees.

| scene | train p2 | train p3 | Δ train | eval p2 | eval p3 | Δ eval | SSIM Δ | LPIPS Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| office0 | 40.438919 | 39.450982 | +0.987937 | 39.348686 | 37.892439 | +1.456247 | +0.009915 | -0.012504 |
| office1 | 41.571293 | 40.430663 | +1.140630 | 39.529468 | 39.060943 | +0.468525 | +0.003103 | -0.012577 |
| office2 | 32.933991 | 30.100643 | +2.833348 | 32.009902 | 29.340560 | +2.669342 | +0.019035 | -0.029144 |
| office3 | 33.196308 | 31.454170 | +1.742138 | 32.218576 | 31.112703 | +1.105873 | +0.017760 | -0.013041 |
| office4 | 36.549065 | 34.564024 | +1.985041 | 35.457077 | 33.182715 | +2.274362 | +0.017739 | -0.024839 |
| room0 | 31.455996 | 30.786618 | +0.669378 | 30.461621 | 30.343413 | +0.118208 | +0.022194 | -0.010878 |
| room1 | 35.510036 | 31.984953 | +3.525083 | 33.759717 | 30.907537 | +2.852180 | +0.031248 | -0.027373 |
| room2 | 35.989110 | 34.406182 | +1.582928 | 34.911356 | 33.067907 | +1.843450 | +0.021231 | -0.023403 |
| mean | 35.955590 | 34.147279 | **+1.808310** | 34.712050 | 33.113527 | **+1.598523** | +0.017778 | -0.019220 |

All eight improve training GS PSNR, eval PSNR, SSIM, and LPIPS.
Translation APE is mixed (3 scenes better, 5 worse); no uniform trajectory
improvement is claimed.

## Photo-SLAM-eval harness

The external Photo-SLAM-eval project needs local compatibility handling in this
container:

- skip non-directory entries such as `eval_guard_shutdown.txt` when scanning a
  result root;
- disable cuDNN for offline SSIM because the installed cuDNN sublibrary set is
  mismatched (`CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`);
- pin `evo==1.34.3` for the expected `ape(est_name=...)` API;
- install isolated Python dependencies under `eval_site/`.

These do not change PSNR/SSIM/LPIPS formulas or the training code.

## Independent source build check

Built from the uploaded source snapshot in an isolated directory:

- DBoW2, g2o, Sophus, ORB-SLAM3, top-level `replica_rgbd`: all succeeded.
- Fresh binary SHA256:
  `d8c5f105061ec204f147e634a7d1e3440fbd27c63b948fe2c65213b5cba05147`

This confirms the source-only snapshot is buildable without excluded `lib/` or
archives. The frozen training binary was not replaced.

## GitHub source update

Pushed and remote-verified:

- branch: `source/room1-pyramid2-20261026`
- commit: `260bc860cb3aff43bd4304a8a4e5f472ab9dc596`
- remote file count: 428
- remote prohibited-artifact scan: no matches

## Limits

- Replica RGB-D eight scenes only.
- No claim of generalization to other datasets or of system acceleration.
- This is a verified configuration correction, not a new algorithm.
