#!/usr/bin/env bash
# Serial eight-scene Replica RGB-D batch using the per-scene A_pyramid2 configs.
#
# Required local inputs:
#   DATA_ROOT=/path/to/replica        (contains office0 ... room2)
#   OUTROOT=/path/to/new/output/root
#
# Optional:
#   BIN=/path/to/replica_rgbd         (default: <repo>/bin/replica_rgbd)
#   LIB_DIR=/path/to/project/libs     (default: directory containing BIN)
#   VOCAB=/path/to/ORBvoc.txt         (default: <repo>/ORB-SLAM3/Vocabulary/ORBvoc.txt)
#   CFG_DIR=/path/to/A_pyramid2       (default: repo A_pyramid2 configs)
#   BATCH_TAG=label                   (recorded only)
#
# This script does not invoke Photo-SLAM-eval and does not depend on any local
# evidence/ or prebuilt archive path.
set -u

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-$ROOT/bin/replica_rgbd}
LIB_DIR=${LIB_DIR:-$(dirname "$BIN")}
VOCAB=${VOCAB:-$ROOT/ORB-SLAM3/Vocabulary/ORBvoc.txt}
CFG_DIR=${CFG_DIR:-$ROOT/cfg/gaussian_mapper/RGB-D/Replica/A_pyramid2}
DATA_ROOT=${DATA_ROOT:-}
OUTROOT=${OUTROOT:-}
BATCH_TAG=${BATCH_TAG:-pyramid2}
SCENES=(office0 office1 office2 office3 office4 room0 room1 room2)
SHUTDOWN_LIMIT=${SHUTDOWN_LIMIT:-900}

[ -n "$DATA_ROOT" ] || { echo "DATA_ROOT is required" >&2; exit 2; }
[ -n "$OUTROOT" ] || { echo "OUTROOT is required" >&2; exit 2; }
[ -x "$BIN" ] || { echo "missing binary: $BIN" >&2; exit 2; }
[ -d "$LIB_DIR" ] || { echo "missing library directory: $LIB_DIR" >&2; exit 2; }
[ -f "$VOCAB" ] || { echo "missing vocabulary: $VOCAB (set VOCAB)" >&2; exit 2; }
mkdir -p "$OUTROOT"
SUMMARY=$OUTROOT/summary.csv
DETAIL=$OUTROOT/summary_detailed.csv
printf 'scene,exit_code,export_ok,log_path,result_dir\n' > "$SUMMARY"
printf 'scene,exit_code,export_ok,elapsed_seconds,result_dir,log_path,orb_cfg,gaussian_cfg,binary_sha256,config_sha256\n' > "$DETAIL"
BIN_SHA=$(sha256sum "$BIN" | cut -d' ' -f1)

check_export() {
    local out=$1
    [ -n "$(find "$out" -type f -path '*/ply/point_cloud/*/point_cloud.ply' -size +1000000c -print -quit 2>/dev/null)" ] || return 1
    [ -n "$(find "$out" -maxdepth 3 -type f -path '*/ply/cameras.json' -print -quit 2>/dev/null)" ] || return 1
    for f in CameraTrajectory_TUM.txt KeyFrameTrajectory_TUM.txt \
             Global_Final_Map_Points.obj TotalOperationTime.txt run.log \
             natural_end/cameras.csv natural_end/meta.txt; do
        [ -s "$out/$f" ] || return 1
    done
}

for scene in "${SCENES[@]}"; do
    ORB_CFG=$ROOT/cfg/ORB_SLAM3/RGB-D/Replica/$scene.yaml
    GAUSS_CFG=$CFG_DIR/$scene.yaml
    DATA=$DATA_ROOT/$scene
    OUT=$OUTROOT/$scene
    LOG=$OUT/run.log

    if [ -e "$OUT" ] && [ -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
        echo "[batch][ERROR] refusing non-empty output: $OUT" >&2
        printf '%s,2,0,%s,%s\n' "$scene" "$LOG" "$OUT" >> "$SUMMARY"
        continue
    fi
    if [ ! -f "$ORB_CFG" ] || [ ! -f "$GAUSS_CFG" ] || [ ! -d "$DATA" ]; then
        echo "[batch][ERROR] missing input for $scene" >&2
        printf '%s,2,0,%s,%s\n' "$scene" "$LOG" "$OUT" >> "$SUMMARY"
        continue
    fi
    mkdir -p "$OUT"
    CFG_SHA=$(sha256sum "$GAUSS_CFG" | cut -d' ' -f1)
    {
        echo "# independent per-scene A_pyramid2 config"
        echo "batch_tag=$BATCH_TAG"
        echo "scene=$scene"
        echo "binary=$BIN"
        echo "binary_sha256=$BIN_SHA"
        echo "library_path=$LIB_DIR"
        echo "orb_cfg=$ORB_CFG"
        echo "gaussian_cfg=$GAUSS_CFG"
        echo "gaussian_cfg_sha256=$CFG_SHA"
        echo "dataset=$DATA"
        echo "PHOTO_SLAM_LINE_MODE=2"
        echo "PHOTO_SLAM_LINE_LOOP=0"
        echo "PHOTO_SLAM_GBA_APPLY_CORRECTION=0"
        echo "PHOTO_SLAM_APPEND_STEPS=0"
        echo "PHOTO_SLAM_SEED=0"
        echo "PHOTO_SLAM_RASTER_BACKEND=A"
        echo "viewer=no_viewer"
        echo "diagnostics=off"
    } > "$OUT/command.txt"
    LD_LIBRARY_PATH="$LIB_DIR" ldd "$BIN" \
        | grep -E 'gaussian_viewer|gaussian_mapper|ORB_SLAM3|imgui|simple_knn|cuda_rasterizer|DBoW2|g2o' \
        > "$OUT/resolved_project_libs.txt" || true

    t0=$(date +%s)
    LD_LIBRARY_PATH="$LIB_DIR" timeout -k 20 "$SHUTDOWN_LIMIT" \
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
            > "$LOG" 2>&1
    rc=$?
    t1=$(date +%s)
    export_ok=0
    check_export "$OUT" && export_ok=1
    printf '%s,%d,%d,%s,%s\n' "$scene" "$rc" "$export_ok" "$LOG" "$OUT" >> "$SUMMARY"
    printf '%s,%d,%d,%d,%s,%s,%s,%s,%s,%s\n' \
        "$scene" "$rc" "$export_ok" "$((t1 - t0))" "$OUT" "$LOG" \
        "$ORB_CFG" "$GAUSS_CFG" "$BIN_SHA" "$CFG_SHA" >> "$DETAIL"
    echo "[batch] scene=$scene rc=$rc export_ok=$export_ok elapsed=$((t1 - t0))s"
done
cat "$SUMMARY"
