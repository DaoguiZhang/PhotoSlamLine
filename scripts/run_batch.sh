#!/usr/bin/env bash
# Serialised, resumable batch for the pre-registered cabinet ablation
# (2 scenes x 2 modalities x 3 arms x 3 repeats = 36 runs).
#
# Properties required by PROTOCOL.md:
#   * one process at a time, unique directory per run, refuses to overwrite
#   * arm order rotated per repeat (T0,P1,P2 / P1,P2,T0 / P2,T0,P1)
#   * every run records command, env, times, exit code and artefact hashes
#   * a failed run is recorded as failed and NOT retried to improve the number
set -uo pipefail

TREE=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-/workspace/eval_runs/cabinet}
VOCAB=$TREE/ORB-SLAM3/Vocabulary/ORBvoc.txt
BIN=$TREE/bin/eval_frontend
LIB=$TREE/ORB-SLAM3/lib/libORB_SLAM3.so
DATA=/workspace/code/SEGS-SLAM/datasets/tum
ASSOC=$TREE/cfg/ORB_SLAM3/RGB-D/TUM/associations
GAUSS=$TREE/cfg/gaussian_mapper/Monocular/TUM/ignored_for_slam_only.yaml

mkdir -p "$OUT"

CONF=(
  "cabinet:mono:cfg/ORB_SLAM3/Monocular/TUM/tum_freiburg3_cabinet.yaml"
  "cabinet:rgbd:cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_cabinet.yaml"
  "large_cabinet:mono:cfg/ORB_SLAM3/Monocular/TUM/tum_freiburg3_large_cabinet.yaml"
  "large_cabinet:rgbd:cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_large_cabinet.yaml"
)

arm_order() {   # $1 = repeat index (1..3)
  case "$1" in
    1) echo "T0 P1 P2" ;;
    2) echo "P1 P2 T0" ;;
    3) echo "P2 T0 P1" ;;
  esac
}
arm_mode() {
  case "$1" in T0) echo 0 ;; P1) echo 1 ;; P2) echo 2 ;; *) echo "" ;; esac
}

run_one() {
  local scene=$1 mod=$2 arm=$3 rep=$4 cfg=$5
  local dir=$OUT/${scene}/${mod}/${arm}/rep${rep}
  if ! mkdir -p "$(dirname "$dir")" 2>/dev/null || ! mkdir "$dir" 2>/dev/null; then
    echo "[SKIP] exists: $dir"; return 0
  fi
  if [ -f "$dir/status.txt" ]; then echo "[SKIP] done: $dir"; return 0; fi

  local assoc=""
  if [ "$mod" = "rgbd" ]; then assoc=$ASSOC/fr3_${scene}.txt; fi
  local seq=$DATA/rgbd_dataset_freiburg3_${scene}

  {
    echo "scene=$scene"
    echo "modality=$mod"
    echo "arm=$arm"
    echo "line_mode=$(arm_mode "$arm")"
    echo "repeat=$rep"
    echo "dir=$dir"
    echo "seq=$seq"
    echo "cfg=$TREE/$cfg"
    echo "assoc=$assoc"
    echo "vocab=$VOCAB"
    echo "binary=$BIN"
    echo "binary_sha256=$(sha256sum "$BIN" | cut -d' ' -f1)"
    echo "lib=$LIB"
    echo "lib_sha256=$(sha256sum "$LIB" | cut -d' ' -f1)"
    echo "cfg_sha256=$(sha256sum "$TREE/$cfg" | cut -d' ' -f1)"
    [ -n "$assoc" ] && echo "assoc_sha256=$(sha256sum "$assoc" | cut -d' ' -f1)"
    echo "git_head=$(git -C "$TREE" rev-parse HEAD)"
    echo "start_iso=$(date -Is)"
  } > "$dir/meta.txt"

  local log=$dir/stdout.log
  local t0 t1 rc
  t0=$(date +%s.%N)
  if [ "$mod" = "rgbd" ]; then
    EVAL_ARM="$arm" PHOTO_SLAM_LINE_MODE=$(arm_mode "$arm") PHOTO_SLAM_PO_LINE=1 PHOTO_SLAM_LBA_LINE=1 \
      PHOTO_SLAM_LINE_RESCUE=0 PHOTO_SLAM_SLAM_ONLY=1 \
      "$BIN" "$VOCAB" "$TREE/$cfg" "$GAUSS" "$seq" "$dir" rgbd "$assoc" > "$log" 2>&1
  else
    EVAL_ARM="$arm" PHOTO_SLAM_LINE_MODE=$(arm_mode "$arm") PHOTO_SLAM_PO_LINE=1 PHOTO_SLAM_LBA_LINE=1 \
      PHOTO_SLAM_LINE_RESCUE=0 PHOTO_SLAM_SLAM_ONLY=1 \
      "$BIN" "$VOCAB" "$TREE/$cfg" "$GAUSS" "$seq" "$dir" mono > "$log" 2>&1
  fi
  rc=$?
  t1=$(date +%s.%N)

  {
    echo "exit_code=$rc"
    echo "end_iso=$(date -Is)"
    echo "wall_s=$(python3 -c "print(round($t1-$t0,1))")"
    echo "line_mode_effective=$(grep -o 'PHOTO_SLAM_LINE_MODE=[0-9]' "$log" | head -1)"
    echo "frames_processed=$(grep -o 'frames_processed=[0-9]*' "$log" | tail -1)"
    echo "final_state=$(grep -o 'final_state=[0-9-]*' "$log" | tail -1)"
    echo "maps=$(grep -c 'SaveKeyFrameTrajectoryTUM:   map' "$log")"
    echo "artefacts:"
    for f in final_state.csv CameraTrajectory_TUM.txt KeyFrameTrajectory_TUM.txt \
             TrackingTime.txt eval_entry.txt; do
      if [ -f "$dir/$f" ]; then echo "  $f $(sha256sum "$dir/$f" | cut -d' ' -f1)"; else echo "  $f MISSING"; fi
    done
  } > "$dir/status.txt"
  echo "[DONE] $scene/$mod/$arm/rep$rep exit=$rc wall=$(python3 -c "print(round($t1-$t0,1))")s"
}

MODS=${MODS:-"mono rgbd"}
REPS=${REPS:-"1 2 3"}
for rep in $REPS; do
  for entry in "${CONF[@]}"; do
    IFS=: read -r scene mod cfg <<< "$entry"
    case " $MODS " in *" $mod "*) ;; *) continue ;; esac
    for arm in $(arm_order "$rep"); do
      run_one "$scene" "$mod" "$arm" "$rep" "$cfg"
    done
  done
done
echo "=== batch complete ==="
