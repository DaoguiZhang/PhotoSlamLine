#!/usr/bin/env bash
# Reproduce the room1 pyramid2 candidate with the frozen A binary.
# usage: TAG=optional_tag ./reproduce_room1_pyramid2.sh
set -u

ROOT=/workspace/code/PSL-perf
FROZEN=$ROOT/evidence/frozen_A_clean_20261023
BIN=$FROZEN/bin/replica_rgbd
VOCAB=$ROOT/ORB-SLAM3/Vocabulary/ORBvoc.txt
ORB_CFG=$ROOT/cfg/ORB_SLAM3/RGB-D/Replica/room1.yaml
GAUSS_CFG=$ROOT/cfg/gaussian_mapper/RGB-D/Replica/A_baseline/room1_pyramid2_candidate.yaml
DATA=/workspace/code/SEGS-SLAM/datasets/replica/room1
TAG=${TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=$ROOT/results/room1_pyramid2_candidate_20261023/$TAG/room1

if [ -e "$OUT" ]; then
    echo "refusing to overwrite: $OUT" >&2
    exit 2
fi
mkdir -p "$OUT"
{
    echo "TAG=$TAG"
    echo "PHOTO_SLAM_RASTER_BACKEND=A"
    echo "PHOTO_SLAM_LINE_MODE=2"
    echo "PHOTO_SLAM_LINE_LOOP=0"
    echo "PHOTO_SLAM_GBA_APPLY_CORRECTION=0"
    echo "PHOTO_SLAM_APPEND_STEPS=0"
    echo "PHOTO_SLAM_SEED=0"
    echo "LD_LIBRARY_PATH=$FROZEN/bin"
    echo "BIN=$BIN"
    echo "ORB_CFG=$ORB_CFG"
    echo "GAUSS_CFG=$GAUSS_CFG"
    echo "DATA=$DATA"
    echo "OUT=$OUT"
} > "$OUT/command.txt"
sha256sum "$BIN" "$GAUSS_CFG" "$ORB_CFG" >> "$OUT/command.txt"

LD_LIBRARY_PATH="$FROZEN/bin" timeout -k 20 600 \
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
        PHOTO_SLAM_SEED=0 \
        "$BIN" "$VOCAB" "$ORB_CFG" "$GAUSS_CFG" "$DATA" "$OUT" no_viewer \
        > "$OUT/run.log" 2>&1
rc=$?
echo "exit_code=$rc" > "$OUT/trial_status.txt"
echo "result_dir=$OUT" >> "$OUT/trial_status.txt"
exit "$rc"
