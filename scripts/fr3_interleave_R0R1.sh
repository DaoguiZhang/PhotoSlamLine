#!/bin/bash
# Interleaved R0/R1 baseline on fr3_long_office_household.
# Order: R0,R1,R0,R1,R0,R1 — each in its own directory (never overwritten).
# R0: LINE_MODE=0 (point-only).  R1: LINE_MODE=1 (line frontend, no line edges).
# Both use LINE_LOOP=0 (standard point loop closing) and DEBUG=1.
# Per-frame CSV is recorded via PHOTO_SLAM_TRACK_CSV for divergence analysis.
set -u
cd /workspace/code/Photo-SLAM-L
export LD_LIBRARY_PATH=ORB-SLAM3/lib:ORB-SLAM3/Thirdparty/g2o/lib:ORB-SLAM3/Thirdparty/DBoW2/lib:lib
SEQ=/workspace/code/SEGS-SLAM/datasets/tum/rgbd_dataset_freiburg3_long_office_household
ASSOC=cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg3_long_office_household.txt
ORBCFG=cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml
GAUSCFG=cfg/gaussian_mapper/RGB-D/TUM/tum_freiburg3_long_office_household_diag.yaml
VOC=ORB-SLAM3/Vocabulary/ORBvoc.txt
ROOT=${1:-/tmp/fr3_interleave}

run_one () {
  local tag=$1; local lm=$2
  local out="$ROOT/${tag}"
  mkdir -p "$out"
  export PHOTO_SLAM_LINE_MODE=$lm PHOTO_SLAM_LINE_LOOP=0 PHOTO_SLAM_PO_LINE=1 PHOTO_SLAM_LBA_LINE=1 PHOTO_SLAM_DEBUG_LINE_LOOP=1
  export PHOTO_SLAM_TRACK_CSV="$out/track.csv"
  echo "=== $tag (LM=$lm) start $(date +%H:%M:%S) ===" > "$out/run.log"
  timeout 420 bin/tum_rgbd "$VOC" "$ORBCFG" "$GAUSCFG" "$SEQ" "$ASSOC" "$out/" no_viewer >> "$out/run.log" 2>&1
  echo "EXIT=$?" >> "$out/run.log"
  echo "=== $tag done $(date +%H:%M:%S) ==="
}

run_one R0_i1 0
run_one R1_i1 1
run_one R0_i2 0
run_one R1_i2 1
run_one R0_i3 0
run_one R1_i3 1
echo "INTERLEAVE DONE"
