#!/bin/bash
# 5x full line-mode run on fr3 to characterize the LBA segfault.
set -u
cd /workspace/code/Photo-SLAM-L
export LD_LIBRARY_PATH=ORB-SLAM3/lib:ORB-SLAM3/Thirdparty/g2o/lib:ORB-SLAM3/Thirdparty/DBoW2/lib:lib
export PHOTO_SLAM_LINE_MODE=2 PHOTO_SLAM_LINE_LOOP=0 PHOTO_SLAM_DEBUG_LINE_LOOP=1
for i in 1 2 3 4 5; do
  out="/tmp/fr3_stab_${i}"
  mkdir -p "$out"
  log="${out}/run.log"
  echo "=== run $i start $(date +%H:%M:%S) ===" > "$log"
  timeout 420 bin/tum_rgbd \
    ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
    cfg/gaussian_mapper/RGB-D/TUM/tum_freiburg3_long_office_household_diag.yaml \
    /workspace/code/SEGS-SLAM/datasets/tum/rgbd_dataset_freiburg3_long_office_household \
    cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg3_long_office_household.txt \
    "$out/" no_viewer >> "$log" 2>&1
  ec=$?
  lastkf=$(grep -oE 'KF [0-9]+' "$log" | tail -1)
  lastsec=$(grep -oE 'LBA-SEC\] [0-9]+-[a-z-]+' "$log" | tail -1)
  lost=$(grep -cE 'Tracking lost|Reset|relocal' "$log")
  echo "=== run $i done exit=$ec lastKf=$lastkf lastLbaSec=$lastsec lostReset=$lost ===" | tee -a "$log"
  # stop early if it crashed (segfault) so we can inspect promptly
  if [ "$ec" -eq 139 ]; then
    echo "SEGFAULT on run $i" | tee -a "$log"
    break
  fi
done
echo "BATCH DONE"
