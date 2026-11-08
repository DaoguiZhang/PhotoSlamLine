#!/usr/bin/env bash
# One whole-config room1 run for a single-factor oldconfig variant.
set -u

if [ $# -ne 1 ]; then
    echo "usage: $0 <variant_tag>" >&2
    exit 2
fi
TAG=$1

ROOT=/workspace/code/PSL-perf
FROZEN=$ROOT/evidence/frozen_A_clean_20261023
BIN=$FROZEN/bin/replica_rgbd
VOCAB=$ROOT/ORB-SLAM3/Vocabulary/ORBvoc.txt
ORB_CFG=$ROOT/cfg/ORB_SLAM3/RGB-D/Replica/room1.yaml
GAUSS_CFG=$ROOT/evidence/repro_room1_oldconfig_20261023/config/variants/room1_oldconfig_${TAG}.yaml
DATA=/workspace/code/SEGS-SLAM/datasets/replica/room1
OUTROOT=$ROOT/results/repro_room1_oldconfig_${TAG}_20261023/replica_rgbd_0
OUT=$OUTROOT/room1

if [ ! -f "$GAUSS_CFG" ]; then
    echo "missing config: $GAUSS_CFG" >&2
    exit 2
fi
mkdir -p "$OUT"
BIN_SHA=$(sha256sum "$BIN" | cut -d' ' -f1)
CFG_SHA=$(sha256sum "$GAUSS_CFG" | cut -d' ' -f1)
{
    echo "# single-factor restoration from room1_oldconfig_effective"
    echo "scene=room1"
    echo "variant=$TAG"
    echo "binary=$BIN"
    echo "binary_sha256=$BIN_SHA"
    echo "orb_cfg=$ORB_CFG"
    echo "gaussian_cfg=$GAUSS_CFG"
    echo "gaussian_cfg_sha256=$CFG_SHA"
    echo "dataset=$DATA"
    echo "images=$(find "$DATA/results" -maxdepth 1 -name 'frame*.jpg' | wc -l)"
    echo "PHOTO_SLAM_LINE_MODE=2"
    echo "PHOTO_SLAM_LINE_LOOP=0"
    echo "PHOTO_SLAM_GBA_APPLY_CORRECTION=0"
    echo "PHOTO_SLAM_APPEND_STEPS=0"
    echo "PHOTO_SLAM_SEED=0"
    echo "PHOTO_SLAM_RASTER_BACKEND=A"
    echo "viewer=no_viewer"
    echo "diagnostics=off"
} > "$OUT/command.txt"

t0=$(date +%s)
LD_LIBRARY_PATH="$FROZEN/bin" timeout -k 20 600 \
    env -u CUDA_LAUNCH_BLOCKING \
        -u PHOTO_SLAM_STAGE_COUNTERS \
        -u PHOTO_SLAM_ADAM_TOPO_DIAG_DIR \
        -u PHOTO_SLAM_ONLINE_CHECKPOINT_DIR \
        -u PHOTO_SLAM_SHUTDOWN_DIAG \
        -u PHOTO_SLAM_RASTER_AGG_SELFCHECK \
        -u PHOTO_SLAM_RASTER_PROF \
        -u PHOTO_SLAM_GPU_PROF \
        PHOTO_SLAM_LINE_MODE=2 \
        PHOTO_SLAM_LINE_LOOP=0 \
        PHOTO_SLAM_GBA_APPLY_CORRECTION=0 \
        PHOTO_SLAM_APPEND_STEPS=0 \
        PHOTO_SLAM_SEED=0 \
        PHOTO_SLAM_RASTER_BACKEND=A \
        "$BIN" "$VOCAB" "$ORB_CFG" "$GAUSS_CFG" "$DATA" "$OUT" no_viewer \
        > "$OUT/run.log" 2>&1
rc=$?
t1=$(date +%s)
echo "variant=$TAG rc=$rc elapsed=$((t1 - t0))s out=$OUT" | tee "$OUT/exit_status.txt"
exit "$rc"
