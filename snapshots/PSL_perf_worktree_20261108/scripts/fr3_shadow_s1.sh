#!/bin/bash
# S1 shadow-line isolation: line frontend (LSD+matching) + pure point backend.
# PHOTO_SLAM_LINE_MODE=1 + PHOTO_SLAM_SHADOW_LINE=1 + PHOTO_SLAM_SLAM_ONLY=1.
set -u
cd /workspace/code/Photo-SLAM-L
export LD_LIBRARY_PATH=ORB-SLAM3/lib:ORB-SLAM3/Thirdparty/g2o/lib:ORB-SLAM3/Thirdparty/DBoW2/lib:lib
SEQ=/workspace/code/SEGS-SLAM/datasets/tum/rgbd_dataset_freiburg3_long_office_household
ASSOC=cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg3_long_office_household.txt
ORBCFG=cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml
GAUSCFG=cfg/gaussian_mapper/RGB-D/TUM/tum_freiburg3_long_office_household_diag.yaml
VOC=ORB-SLAM3/Vocabulary/ORBvoc.txt
OUT=${1:-/tmp/fr3_s1}
mkdir -p "$OUT"
export PHOTO_SLAM_LINE_MODE=1 PHOTO_SLAM_LINE_LOOP=0 PHOTO_SLAM_PO_LINE=1 PHOTO_SLAM_LBA_LINE=1 PHOTO_SLAM_DEBUG_LINE_LOOP=1
export PHOTO_SLAM_SHADOW_LINE=1 PHOTO_SLAM_SLAM_ONLY=1
echo "=== S1 (LM=1 SHADOW=1 SLAM_ONLY=1) start $(date +%H:%M:%S) ===" > "$OUT/run.log"
timeout 300 bin/tum_rgbd "$VOC" "$ORBCFG" "$GAUSCFG" "$SEQ" "$ASSOC" "$OUT/" no_viewer >> "$OUT/run.log" 2>&1
echo "EXIT=$?" >> "$OUT/run.log"
echo "=== S1 done $(date +%H:%M:%S) ==="
