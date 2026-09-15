#!/bin/bash
# Acceptance batch: R0x3, R1x3, R4x3 on fr3_long_office_household.
# Each run in its own dir. Reports loop-detected, GBA write-back, KF count, ATE, EXIT.
set -u
cd /workspace/code/Photo-SLAM-L
export LD_LIBRARY_PATH=ORB-SLAM3/lib:ORB-SLAM3/Thirdparty/g2o/lib:ORB-SLAM3/Thirdparty/DBoW2/lib:lib
SEQ=/workspace/code/SEGS-SLAM/datasets/tum/rgbd_dataset_freiburg3_long_office_household
ASSOC=cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg3_long_office_household.txt
ORBCFG=cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml
GAUSCFG=cfg/gaussian_mapper/RGB-D/TUM/tum_freiburg3_long_office_household_diag.yaml
VOC=ORB-SLAM3/Vocabulary/ORBvoc.txt
ROOT=${1:-/tmp/fr3_accept}
GT=/workspace/code/SEGS-SLAM/datasets/tum/rgbd_dataset_freiburg3_long_office_household/groundtruth.txt

run_one () {
  local tag=$1; local lm=$2
  local out="$ROOT/${tag}"
  mkdir -p "$out"
  export PHOTO_SLAM_LINE_MODE=$lm PHOTO_SLAM_LINE_LOOP=0 PHOTO_SLAM_PO_LINE=1 PHOTO_SLAM_LBA_LINE=1 PHOTO_SLAM_DEBUG_LINE_LOOP=1
  echo "=== $tag (LM=$lm) start $(date +%H:%M:%S) ===" > "$out/run.log"
  timeout 420 bin/tum_rgbd "$VOC" "$ORBCFG" "$GAUSCFG" "$SEQ" "$ASSOC" "$out/" no_viewer >> "$out/run.log" 2>&1
  echo "EXIT=$?" >> "$out/run.log"
  echo "=== $tag done $(date +%H:%M:%S) ==="
}

for i in 1 2 3; do run_one R0_$i 0; run_one R1_$i 1; run_one R4_$i 2; done
echo "ACCEPTANCE DONE"

# Report
for i in 1 2 3; do
  for t in R0 R1 R4; do
    out="$ROOT/${t}_$i"
    loop=$(grep -cE 'Loop detected' "$out/run.log" 2>/dev/null)
    gba=$(grep -c 'GBA' "$out/run.log" 2>/dev/null)
    kf=$(grep -oE 'KF [0-9]+' "$out/run.log" 2>/dev/null | tail -1)
    exitc=$(grep EXIT "$out/run.log" 2>/dev/null | tail -1)
    ate=$(python3 scripts/eval_tum_sim3.py "$GT" "$out/CameraTrajectory_TUM.txt" "$out/ate.txt" 2>/dev/null | tail -1)
    echo "$t_$i: loop=$loop gba=$gba $kf $exitc | $ate"
  done
done
