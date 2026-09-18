#!/bin/bash
# P6: TUM fr3 RGB-D loop-closing matrix (P0 / LB / LC) with Gaussian ON.
#
#   P0 = PHOTO_SLAM_LINE_LOOP=0  (point-only loop closing; C0 baseline)
#   LB = PHOTO_SLAM_LINE_LOOP=1  (line map/Gaussian sync carried by point loop)
#   LC = PHOTO_SLAM_LINE_LOOP=2  (line residuals in Sim(3) refine + line GBA)
#
# Layout: round 1 = P0,LB,LC (1x). Then 3x interleaved (P0,LB,LC,P0,LB,LC,...).
# Per run: EXIT, loop/candidate KF, point/line match pairs, Sim3 line pairs,
# corrected/fused lines, GBA line vertices/edges, Gaussian corrected/nonFinite,
# SE3/Sim3 ATE, RPE, mapping time, GPU peak VRAM.
#
# Usage:
#   ROOT=/path/to/results bash scripts/p6_rgbd_loop_matrix.sh
#   ROOT=... PHASE1_ONLY=1 bash scripts/p6_rgbd_loop_matrix.sh   # only the 1x round
set -u
cd /workspace/code/Photo-SLAM-L
export LD_LIBRARY_PATH=ORB-SLAM3/lib:ORB-SLAM3/Thirdparty/g2o/lib:ORB-SLAM3/Thirdparty/DBoW2/lib:lib

SEQ=/workspace/code/SEGS-SLAM/datasets/tum/rgbd_dataset_freiburg3_long_office_household
ASSOC=cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg3_long_office_household.txt
ORBCFG=cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml
GAUSCFG=cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml
VOC=ORB-SLAM3/Vocabulary/ORBvoc.txt
GT="$SEQ/groundtruth.txt"
ROOT="${ROOT:-results/p6_rgbd_loop_matrix}"
EVAL=scripts/p6_eval_tum.py

run_one () {
  local tag=$1; local loop=$2
  local out="$ROOT/$tag"
  mkdir -p "$out"
  export PHOTO_SLAM_LINE_MODE=2 PHOTO_SLAM_LINE_LOOP=$loop \
         PHOTO_SLAM_PO_LINE=1 PHOTO_SLAM_LBA_LINE=1 \
         PHOTO_SLAM_SHADOW_LINE=0 PHOTO_SLAM_SLAM_ONLY=0 \
         PHOTO_SLAM_DEBUG_LINE_LOOP=1
  echo "=== $tag (LINE_LOOP=$loop) start $(date +%H:%M:%S) ===" > "$out/run.log"
  bin/tum_rgbd "$VOC" "$ORBCFG" "$GAUSCFG" "$SEQ" "$ASSOC" "$out/" no_viewer >> "$out/run.log" 2>&1
  echo "EXIT=$?" >> "$out/run.log"
  echo "=== $tag done $(date +%H:%M:%S) ==="

  # --- metric extraction ------------------------------------------------
  local m="$out/metrics.txt"
  {
    echo "tag=$tag"
    echo "line_loop=$loop"
    grep -E '^EXIT=' "$out/run.log" | tail -1
    echo "loop_detected=$(grep -cE 'Loop detected' "$out/run.log")"
    echo "loop_kf=$(grep -oE 'Loop: mpCurrentKF -> [0-9]+' "$out/run.log" | tail -1 | grep -oE '[0-9]+$')"
    echo "candidate_accepted=$(grep -oE 'candidate accepted: loop=[0-9]+ merge=[0-9]+ matchedPts=[0-9]+ matchedLines=[0-9]+' "$out/run.log" | tail -1)"
    echo "sim3_lines=$(grep -oE '\[LineLoop\]\[Sim3\] linePairs=[0-9]+ lineInlierPairs=[0-9]+' "$out/run.log" | tail -1)"
    echo "correct_loop=$(grep -oE 'CorrectLoopWithLine: scale=[0-9.e+-]+ KFs=[0-9]+ linesCorrected=[0-9]+ linesFused=[0-9]+' "$out/run.log" | tail -1)"
    echo "gba_lines=$(grep -oE '\[LineLoop\]\[GBA\] lineVertices=[0-9]+ lineEdges=[0-9]+ vertexAddFailures=[0-9]+ mapLines=[0-9]+' "$out/run.log" | tail -1)"
    echo "gba_writeback=$(grep -oE '\[LineLoop\] GBA writeback done: linesUpdated=[0-9]+' "$out/run.log" | tail -1)"
    echo "gaussian=$(grep -oE '\[LineLoop-Gaussian\].*' "$out/run.log" | tail -1)"
    echo "mapping_time=$(cat "$out/total_mapping_time.txt" 2>/dev/null | head -1)"
    echo "gpu_peak_mb=$(cat "$out/GpuPeakUsageMB.txt" 2>/dev/null | head -1)"
  } > "$m"

  if [ -f "$out/CameraTrajectory_TUM.txt" ]; then
    python3 "$EVAL" "$GT" "$out/CameraTrajectory_TUM.txt" "$out/eval.txt" >> "$out/run.log" 2>&1
    cat "$out/eval.txt" 2>/dev/null >> "$m"
  fi
}

ROUND1_TAGS="P0 LB LC"
run_round1 () {
  run_one P0_r1 0
  run_one LB_r1 1
  run_one LC_r1 2
}

run_rounds () {
  for i in 2 3 4; do
    run_one P0_r$i 0
    run_one LB_r$i 1
    run_one LC_r$i 2
  done
}

case "${1:-all}" in
  round1) run_round1 ;;
  rounds) run_rounds ;;
  all)
    run_round1
    if [ "${PHASE1_ONLY:-0}" != "1" ]; then
      run_rounds
    fi
    ;;
esac

# --- aggregate report ---------------------------------------------------
echo "==================== P6 REPORT ===================="
for m in "$ROOT"/*/metrics.txt; do
  [ -f "$m" ] || continue
  tag=$(grep '^tag=' "$m" | cut -d= -f2)
  exitc=$(grep '^EXIT=' "$m" | tail -1)
  loopkf=$(grep '^loop_kf=' "$m" | cut -d= -f2)
  cand=$(grep '^candidate_accepted=' "$m" | cut -d= -f2-)
  sim3l=$(grep '^sim3_lines=' "$m" | cut -d= -f2-)
  corr=$(grep '^correct_loop=' "$m" | cut -d= -f2-)
  gba=$(grep '^gba_lines=' "$m" | cut -d= -f2-)
  gauss=$(grep '^gaussian=' "$m" | cut -d= -f2-)
  se3=$(grep '^se3_ate_rmse_m=' "$m" | cut -d= -f2)
  sim3ate=$(grep '^sim3_ate_rmse_m=' "$m" | cut -d= -f2)
  rpe=$(grep '^rpe_rmse_m=' "$m" | cut -d= -f2)
  echo "$tag | $exitc | loopKF=$loopkf | $cand | $sim3l | $corr | $gba | $gauss | SE3=$se3 Sim3=$sim3ate RPE=$rpe"
done
echo "P6 DONE"
