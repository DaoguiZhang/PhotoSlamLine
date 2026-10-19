#!/bin/bash
# R0-R4 RGB-D isolation matrix on fr3_long_office_household.
# All runs use LINE_LOOP=0 (standard point loop closing) and DEBUG=1.
#   R0: LINE_MODE=0 (point-only, no lines)
#   R1: LINE_MODE=1 (line frontend, no line edges in PO/LBA)
#   R2: LINE_MODE=2 PO_LINE=1 LBA_LINE=0 (line PO, point LBA)
#   R3: LINE_MODE=2 PO_LINE=0 LBA_LINE=1 (point PO, line LBA)
#   R4: LINE_MODE=2 PO_LINE=1 LBA_LINE=1 (full point-line)
set -u
cd /workspace/code/Photo-SLAM-L
export LD_LIBRARY_PATH=ORB-SLAM3/lib:ORB-SLAM3/Thirdparty/g2o/lib:ORB-SLAM3/Thirdparty/DBoW2/lib:lib
SEQ=/workspace/code/SEGS-SLAM/datasets/tum/rgbd_dataset_freiburg3_long_office_household
ASSOC=cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg3_long_office_household.txt
ORBCFG=cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml
GAUSCFG=cfg/gaussian_mapper/RGB-D/TUM/tum_freiburg3_long_office_household_diag.yaml
VOC=ORB-SLAM3/Vocabulary/ORBvoc.txt

run_one () {
  local tag=$1; local lm=$2; local po=$3; local lba=$4
  local out="/tmp/fr3_matrix_${tag}"
  mkdir -p "$out"
  export PHOTO_SLAM_LINE_MODE=$lm PHOTO_SLAM_LINE_LOOP=0 PHOTO_SLAM_PO_LINE=$po PHOTO_SLAM_LBA_LINE=$lba PHOTO_SLAM_DEBUG_LINE_LOOP=1
  echo "=== $tag (LM=$lm PO=$po LBA=$lba) start $(date +%H:%M:%S) ===" > "$out/run.log"
  timeout 420 bin/tum_rgbd "$VOC" "$ORBCFG" "$GAUSCFG" "$SEQ" "$ASSOC" "$out/" no_viewer >> "$out/run.log" 2>&1
  echo "EXIT=$?" >> "$out/run.log"
  echo "=== $tag done $(date +%H:%M:%S) ==="
}

case "${1:-all}" in
  R0) run_one R0 0 1 1 ;;
  R1) run_one R1 1 1 1 ;;
  R2) run_one R2 2 1 0 ;;
  R3) run_one R3 2 0 1 ;;
  R4) run_one R4 2 1 1 ;;
  all)
    run_one R0 0 1 1
    run_one R1 1 1 1
    run_one R2 2 1 0
    run_one R3 2 0 1
    run_one R4 2 1 1
    ;;
esac
echo "MATRIX DONE"
