#!/usr/bin/env bash
# One clean room1 trial for the pyramid-level comparison.
# usage: run_one_trial.sh <tag> <pyramid3|pyramid2>
set -u

if [ $# -ne 2 ]; then
    echo "usage: $0 <tag> <pyramid3|pyramid2>" >&2
    exit 2
fi
TAG=$1
LEVEL=$2

ROOT=/workspace/code/PSL-perf
FROZEN=$ROOT/evidence/frozen_A_clean_20261023
BIN=$FROZEN/bin/replica_rgbd
VOCAB=$ROOT/ORB-SLAM3/Vocabulary/ORBvoc.txt
ORB_CFG=$ROOT/cfg/ORB_SLAM3/RGB-D/Replica/room1.yaml
GAUSS_CFG=$ROOT/evidence/room1_pyramid_ablation_20261023/config/room1_A8_${LEVEL}.yaml
DATA=/workspace/code/SEGS-SLAM/datasets/replica/room1
OUTROOT=$ROOT/results/room1_pyramid_ablation_20261023
OUT=$OUTROOT/${TAG}_${LEVEL}/room1

if [ ! -x "$BIN" ]; then
    echo "[trial][ERROR] missing binary: $BIN" >&2
    exit 2
fi
if [ ! -f "$GAUSS_CFG" ]; then
    echo "[trial][ERROR] missing config: $GAUSS_CFG" >&2
    exit 2
fi
if [ -e "$OUT" ]; then
    echo "[trial][ERROR] refusing to overwrite: $OUT" >&2
    exit 2
fi

mkdir -p "$OUT"
BIN_SHA=$(sha256sum "$BIN" | cut -d' ' -f1)
CFG_SHA=$(sha256sum "$GAUSS_CFG" | cut -d' ' -f1)
{
    echo "# one clean room1 trial: only GausPyramid.num_sub_levels differs"
    echo "tag=$TAG"
    echo "level=$LEVEL"
    echo "binary=$BIN"
    echo "binary_sha256=$BIN_SHA"
    echo "library_path=$FROZEN/bin"
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
    echo "checkpoints=off"
    echo "selfcheck=off"
    echo "profiler=off"
} > "$OUT/command.txt"
LD_LIBRARY_PATH="$FROZEN/bin" ldd "$BIN" \
    | grep -E 'gaussian_viewer|gaussian_mapper|ORB_SLAM3|imgui|simple_knn|cuda_rasterizer|DBoW2|g2o' \
    > "$OUT/resolved_project_libs.txt"

START_UTC=$(date -u +%FT%TZ)
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
        -u PHOTO_SLAM_GBA_RECOVERY_DIAG \
        -u PHOTO_SLAM_ZERO_INDICES \
        -u PHOTO_SLAM_DIFF_DIR \
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
END_UTC=$(date -u +%FT%TZ)
{
    echo "tag=$TAG"
    echo "level=$LEVEL"
    echo "exit_code=$rc"
    echo "start_utc=$START_UTC"
    echo "end_utc=$END_UTC"
    echo "wall_seconds=$((t1 - t0))"
    echo "result_dir=$OUT"
} > "$OUT/trial_status.txt"
echo "tag=$TAG level=$LEVEL rc=$rc wall=$((t1 - t0))s out=$OUT"
exit "$rc"
