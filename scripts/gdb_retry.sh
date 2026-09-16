#!/bin/bash
# Retry gdb until a SIGSEGV is caught, up to N attempts.
# Usage: gdb_retry.sh <attempts>
set -u
cd /workspace/code/Photo-SLAM-L
export LD_LIBRARY_PATH=ORB-SLAM3/lib:ORB-SLAM3/Thirdparty/g2o/lib:ORB-SLAM3/Thirdparty/DBoW2/lib:lib
export PHOTO_SLAM_LINE_MODE=2 PHOTO_SLAM_LINE_LOOP=0 PHOTO_SLAM_DEBUG_LINE_LOOP=0
N="${1:-6}"
for i in $(seq 1 "$N"); do
  out="/tmp/gdb_retry_${i}.log"
  echo "=== attempt $i ===" > "$out"
  timeout 420 gdb -batch \
    -ex "set pagination off" \
    -ex run \
    -ex "thread apply all bt full" \
    --args bin/tum_rgbd \
      ORB-SLAM3/Vocabulary/ORBvoc.txt \
      cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
      cfg/gaussian_mapper/RGB-D/TUM/tum_freiburg3_long_office_household_diag.yaml \
      /workspace/code/SEGS-SLAM/datasets/tum/rgbd_dataset_freiburg3_long_office_household \
      cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg3_long_office_household.txt \
      "/tmp/gdb_retry_${i}/" no_viewer >> "$out" 2>&1
  if grep -qE 'Program received signal|SIGSEGV|SIGABRT' "$out"; then
    echo "CRASH_CAUGHT in attempt $i" | tee -a "$out"
    break
  else
    echo "attempt $i clean (no crash)" | tee -a "$out"
  fi
done
echo "DONE"
