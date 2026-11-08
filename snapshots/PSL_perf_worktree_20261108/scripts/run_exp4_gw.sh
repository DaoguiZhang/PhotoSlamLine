#!/usr/bin/env bash
# G/W numeric candidates (with the map-mutex fix). C0 = exp3/C0 (already run).
# G: PHOTO_SLAM_LBA_TAU=5.991 (tighten LBA line outlier gate only)
# W: PHOTO_SLAM_LBA_W=0.1, TAU=0.9, DELTA=sqrt(0.5991) (reduce LBA line contribution)
set -u
BIN=/workspace/code/Photo-SLAM-L/bin/replica_mono
VOC=/workspace/code/Photo-SLAM-L/ORB-SLAM3/Vocabulary/ORBvoc.txt
SL=/workspace/code/Photo-SLAM-L/cfg/ORB_SLAM3/Monocular/Replica/office2.yaml
GS=/workspace/code/Photo-SLAM-L/cfg/gaussian_mapper/Monocular/Replica/replica_mono_diag.yaml
DS=/workspace/code/SEGS-SLAM/datasets/replica/office2
GT=/workspace/code/SEGS-SLAM/datasets/replica/office2/pose_TUM.txt
EV=/workspace/code/Photo-SLAM-L/scripts/eval_ate_rpe.py
BASE=/workspace/code/Photo-SLAM-L/results/diagnosis/office2/exp4

run_one () {  # $1=group $2=runindex ; extra env via EXTRA_ENV
  local GRP=$1 R=$2
  local OUT="$BASE/$GRP/run$R"
  rm -rf "$OUT"; mkdir -p "$OUT/cwd"
  ( cd "$OUT/cwd" && env PHOTO_SLAM_LINE_MODE=2 PHOTO_SLAM_PO_LINE=1 PHOTO_SLAM_LBA_LINE=1 \
      PHOTO_SLAM_DEBUG_MONO_INIT=1 PHOTO_SLAM_DEBUG_LINE_EDGES=1 $EXTRA_ENV \
      "$BIN" "$VOC" "$SL" "$GS" "$DS" "$OUT/" no_viewer ) > "$OUT/run.log" 2>&1
  local EC=$?
  echo "exit=$EC" >> "$OUT/run.log"
  if [ -f "$OUT/CameraTrajectory_TUM.txt" ]; then
    python3 "$EV" "$GT" "$OUT/CameraTrajectory_TUM.txt" "$OUT/ate.txt" "$OUT/rpe.txt" "$OUT/ape.txt" >> "$OUT/run.log" 2>&1
  fi
  echo "$GRP/run$R exit=$EC" >> "$BASE/_status.txt"
}

: > "$BASE/_status.txt"
for R in 1 2 3; do
  EXTRA_ENV="PHOTO_SLAM_LBA_TAU=5.991"; run_one G $R
  EXTRA_ENV="PHOTO_SLAM_LBA_W=0.1 PHOTO_SLAM_LBA_TAU=0.9 PHOTO_SLAM_LBA_DELTA=0.77401"; run_one W $R
done
echo "ALL_DONE" >> "$BASE/_status.txt"
