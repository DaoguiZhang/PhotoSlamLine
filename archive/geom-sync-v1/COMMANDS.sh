#!/usr/bin/env bash
# geom-sync-v1 baseline commands (backend C explicit; A remains the rollback)
cd /workspace/code/PSL-perf || exit 1
PHOTO_SLAM_RASTER_BACKEND=C PHOTO_SLAM_LINE_LOSS_EVERY=0 \
  bash scripts/run_online_ab.sh C abl_out/online_C office0
PHOTO_SLAM_RASTER_BACKEND=A PHOTO_SLAM_LINE_LOSS_EVERY=0 \
  bash scripts/run_online_ab.sh A abl_out/online_A office0
