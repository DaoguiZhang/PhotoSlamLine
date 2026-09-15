#!/bin/bash
# SLAM-only R0/R1 isolation on fr3_long_office_household.
# PHOTO_SLAM_SLAM_ONLY=1 skips the Gaussian mapper (added to tum_rgbd.cpp).
#   R0: LINE_MODE=0 (TrackWithLine + standard LocalMapping::Run)
#   R1: LINE_MODE=1 (TrackWithLine + RunWithLine, standard point LBA via fix)
set -u
cd /workspace/code/Photo-SLAM-L
export LD_LIBRARY_PATH=ORB-SLAM3/lib:ORB-SLAM3/Thirdparty/g2o/lib:ORB-SLAM3/Thirdparty/DBoW2/lib:lib
SEQ=/workspace/code/SEGS-SLAM/datasets/tum/rgbd_dataset_freiburg3_long_office_household
ASSOC=cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg3_long_office_household.txt
ORBCFG=cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml
GAUSCFG=cfg/gaussian_mapper/RGB-D/TUM/tum_freiburg3_long_office_household_diag.yaml
VOC=ORB-SLAM3/Vocabulary/ORBvoc.txt
ROOT=${1:-/tmp/fr3_slamonly}

run_one () {
  local tag=$1; local lm=$2
  local out="$ROOT/${tag}"
  mkdir -p "$out"
  export PHOTO_SLAM_LINE_MODE=$lm PHOTO_SLAM_LINE_LOOP=0 PHOTO_SLAM_PO_LINE=1 PHOTO_SLAM_LBA_LINE=1 PHOTO_SLAM_DEBUG_LINE_LOOP=1
  export PHOTO_SLAM_SLAM_ONLY=1
  echo "=== $tag (LM=$lm SLAM_ONLY=1) start $(date +%H:%M:%S) ===" > "$out/run.log"
  timeout 300 bin/tum_rgbd "$VOC" "$ORBCFG" "$GAUSCFG" "$SEQ" "$ASSOC" "$out/" no_viewer >> "$out/run.log" 2>&1
  echo "EXIT=$?" >> "$out/run.log"
  echo "=== $tag done $(date +%H:%M:%S) ==="
}

run_one R0 0
run_one R1 1
echo "SLAM-ONLY DONE"
