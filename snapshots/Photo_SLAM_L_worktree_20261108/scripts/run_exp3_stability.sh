#!/usr/bin/env bash
# Stability verification + fixed-logging P/C0 baseline (with the map-mutex fix).
set -u
BIN=/workspace/code/Photo-SLAM-L/bin/replica_mono
VOC=/workspace/code/Photo-SLAM-L/ORB-SLAM3/Vocabulary/ORBvoc.txt
SL=/workspace/code/Photo-SLAM-L/cfg/ORB_SLAM3/Monocular/Replica/office2.yaml
GS=/workspace/code/Photo-SLAM-L/cfg/gaussian_mapper/Monocular/Replica/replica_mono_diag.yaml
DS=/workspace/code/SEGS-SLAM/datasets/replica/office2
GT=/workspace/code/SEGS-SLAM/datasets/replica/office2/pose_TUM.txt
EV=/workspace/code/Photo-SLAM-L/scripts/eval_ate_rpe.py
BASE=/workspace/code/Photo-SLAM-L/results/diagnosis/office2/exp3

run_one () {  # $1=PO $2=LBA $3=group $4=runindex
  local PO=$1 LBA=$2 GRP=$3 R=$4
  local OUT="$BASE/$GRP/run$R"
  rm -rf "$OUT"; mkdir -p "$OUT/cwd"
  local T0=$(date +%s)
  ( cd "$OUT/cwd" && env PHOTO_SLAM_LINE_MODE=2 PHOTO_SLAM_PO_LINE=$PO PHOTO_SLAM_LBA_LINE=$LBA \
      PHOTO_SLAM_DEBUG_MONO_INIT=1 PHOTO_SLAM_DEBUG_LINE_EDGES=1 \
      "$BIN" "$VOC" "$SL" "$GS" "$DS" "$OUT/" no_viewer ) > "$OUT/run.log" 2>&1
  local EC=$?
  local T1=$(date +%s)
  echo "exit=$EC wall=$((T1-T0))s" >> "$OUT/run.log"
  for f in 2_Keyframe_Camera_before.txt 2_Keyframe_Camera_after.txt; do
    [ -f "$OUT/cwd/$f" ] && cp "$OUT/cwd/$f" "$OUT/$f"
  done
  if [ -f "$OUT/CameraTrajectory_TUM.txt" ]; then
    python3 "$EV" "$GT" "$OUT/CameraTrajectory_TUM.txt" "$OUT/ate.txt" "$OUT/rpe.txt" "$OUT/ape.txt" >> "$OUT/run.log" 2>&1
  fi
  echo "$GRP/run$R exit=$EC" >> "$BASE/_status.txt"
}

: > "$BASE/_status.txt"
# 1) L stability: PO=0 LBA=1 with diagnostics ON (historical crash condition), 5x
for R in 1 2 3 4 5; do run_one 0 1 L_stability $R; done
# 2) P and C0 with unified logging (diagnostics ON), 3x each, interleaved
for R in 1 2 3; do run_one 1 0 P $R; run_one 1 1 C0 $R; done
echo "ALL_DONE" >> "$BASE/_status.txt"
