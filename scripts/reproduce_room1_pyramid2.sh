#!/usr/bin/env bash
# Reproduce the verified room1 pyramid2 candidate run.
#
# Required local inputs:
#   --binary   path to a built replica_rgbd
#   --dataset  path to the room1 sequence root (contains results/frame*.jpg)
#   --output   new output directory
#
# Optional:
#   --vocab    path to ORBvoc.txt (default: <repo>/ORB-SLAM3/Vocabulary/ORBvoc.txt)
#   --lib-dir  directory containing the binary's project shared libraries
#              (default: directory containing --binary)
#   --orb-config, --gaussian-config
#   --seed     default 0
#
# This script intentionally uses only arguments; it has no dependency on any
# local evidence/ or bin/ archive path.
set -u

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=""
DATA=""
OUT=""
VOCAB="${ROOT}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
LIB_DIR=""
ORB_CFG="${ROOT}/cfg/ORB_SLAM3/RGB-D/Replica/room1.yaml"
GAUSS_CFG="${ROOT}/cfg/gaussian_mapper/RGB-D/Replica/A_baseline/room1_pyramid2_candidate.yaml"
SEED=0

usage() {
    cat <<'EOF'
Usage: reproduce_room1_pyramid2.sh --binary BIN --dataset DATA --output OUT [options]

Required:
  --binary BIN       built replica_rgbd executable
  --dataset DATA     room1 sequence root (contains results/frame*.jpg)
  --output OUT       new output directory

Optional:
  --vocab FILE       ORBvoc.txt (default: <repo>/ORB-SLAM3/Vocabulary/ORBvoc.txt)
  --lib-dir DIR      directory with project shared libraries; default: BIN directory
  --orb-config FILE  default: cfg/ORB_SLAM3/RGB-D/Replica/room1.yaml
  --gaussian-config FILE
                     default: cfg/gaussian_mapper/RGB-D/Replica/A_baseline/room1_pyramid2_candidate.yaml
  --seed N           default: 0
EOF
    exit "${1:-2}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --binary) BIN=${2:-}; shift 2 ;;
        --dataset) DATA=${2:-}; shift 2 ;;
        --output) OUT=${2:-}; shift 2 ;;
        --vocab) VOCAB=${2:-}; shift 2 ;;
        --lib-dir) LIB_DIR=${2:-}; shift 2 ;;
        --orb-config) ORB_CFG=${2:-}; shift 2 ;;
        --gaussian-config) GAUSS_CFG=${2:-}; shift 2 ;;
        --seed) SEED=${2:-}; shift 2 ;;
        -h|--help) usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 2 ;;
    esac
done

[ -n "$BIN" ] || { echo "--binary is required" >&2; usage 2; }
[ -n "$DATA" ] || { echo "--dataset is required" >&2; usage 2; }
[ -n "$OUT" ] || { echo "--output is required" >&2; usage 2; }
[ -x "$BIN" ] || { echo "not executable: $BIN" >&2; exit 2; }
[ -f "$VOCAB" ] || { echo "missing vocabulary: $VOCAB (pass --vocab)" >&2; exit 2; }
[ -f "$ORB_CFG" ] || { echo "missing ORB config: $ORB_CFG" >&2; exit 2; }
[ -f "$GAUSS_CFG" ] || { echo "missing Gaussian config: $GAUSS_CFG" >&2; exit 2; }
[ -d "$DATA" ] || { echo "missing dataset: $DATA" >&2; exit 2; }
[ -n "$LIB_DIR" ] || LIB_DIR=$(cd "$(dirname "$BIN")" && pwd)
[ -d "$LIB_DIR" ] || { echo "missing library directory: $LIB_DIR" >&2; exit 2; }
if [ -e "$OUT" ] && [ -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
    echo "refusing to write into non-empty output: $OUT" >&2
    exit 2
fi
mkdir -p "$OUT"

{
    echo "PHOTO_SLAM_RASTER_BACKEND=A"
    echo "PHOTO_SLAM_LINE_MODE=2"
    echo "PHOTO_SLAM_LINE_LOOP=0"
    echo "PHOTO_SLAM_GBA_APPLY_CORRECTION=0"
    echo "PHOTO_SLAM_APPEND_STEPS=0"
    echo "PHOTO_SLAM_SEED=${SEED}"
    echo "viewer=no_viewer"
    echo "BIN=$BIN"
    echo "LIB_DIR=$LIB_DIR"
    echo "VOCAB=$VOCAB"
    echo "ORB_CFG=$ORB_CFG"
    echo "GAUSS_CFG=$GAUSS_CFG"
    echo "DATA=$DATA"
    echo "OUT=$OUT"
} > "$OUT/command.txt"
sha256sum "$BIN" "$GAUSS_CFG" "$ORB_CFG" >> "$OUT/command.txt"

LD_LIBRARY_PATH="${LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
timeout -k 20 600 \
    env -u CUDA_LAUNCH_BLOCKING \
        -u PHOTO_SLAM_STAGE_COUNTERS \
        -u PHOTO_SLAM_ADAM_TOPO_DIAG_DIR \
        -u PHOTO_SLAM_ONLINE_CHECKPOINT_DIR \
        -u PHOTO_SLAM_SHUTDOWN_DIAG \
        -u PHOTO_SLAM_RASTER_AGG_SELFCHECK \
        -u PHOTO_SLAM_RASTER_PROF \
        -u PHOTO_SLAM_GPU_PROF \
        -u PHOTO_SLAM_GBA_RECOVERY_DIAG \
        -u PHOTO_SLAM_ZERO_INDICES \
        -u PHOTO_SLAM_DIFF_DIR \
        PHOTO_SLAM_RASTER_BACKEND=A \
        PHOTO_SLAM_LINE_MODE=2 \
        PHOTO_SLAM_LINE_LOOP=0 \
        PHOTO_SLAM_GBA_APPLY_CORRECTION=0 \
        PHOTO_SLAM_APPEND_STEPS=0 \
        PHOTO_SLAM_SEED="$SEED" \
        "$BIN" "$VOCAB" "$ORB_CFG" "$GAUSS_CFG" "$DATA" "$OUT" no_viewer \
        > "$OUT/run.log" 2>&1
rc=$?
{
    echo "exit_code=$rc"
    echo "result_dir=$OUT"
} > "$OUT/trial_status.txt"
echo "exit_code=$rc result_dir=$OUT"
exit "$rc"
