#!/usr/bin/env bash
# Replica RGB-D, frozen A backend, clean baseline batch.
#
# Usage:
#   scripts/run_replica_A_baseline.sh [scene ...]
#   BATCH_TAG=my_tag scripts/run_replica_A_baseline.sh
#
# With no scene arguments, runs:
#   office0 office1 office2 office3 office4 room0 room1 room2
#
# This script does NOT call any evaluation project.  It records the real exit
# code, wall time, binary/config hashes and result directory per scene.
set -u

ROOT=${ROOT:-/workspace/code/PSL-perf}
FROZEN=${FROZEN:-$ROOT/evidence/frozen_A_clean_20261023}
BIN=$FROZEN/bin/replica_rgbd
VOCAB=$ROOT/ORB-SLAM3/Vocabulary/ORBvoc.txt
DATA_ROOT=${DATA_ROOT:-/workspace/code/SEGS-SLAM/datasets/replica}
BATCH_TAG=${BATCH_TAG:-$(date +%Y%m%d_%H%M%S)}
OUTROOT=${OUTROOT:-$ROOT/results/replica_A_baseline_$BATCH_TAG/replica_rgbd_0}
SHUTDOWN_LIMIT=${SHUTDOWN_LIMIT:-600}

SCENES=("$@")
if [ ${#SCENES[@]} -eq 0 ]; then
    SCENES=(office0 office1 office2 office3 office4 room0 room1 room2)
fi

if [ ! -x "$BIN" ]; then
    echo "[batch][ERROR] frozen binary missing: $BIN" >&2
    exit 2
fi
if [ ! -f "$VOCAB" ]; then
    echo "[batch][ERROR] vocabulary missing: $VOCAB" >&2
    exit 2
fi
mkdir -p "$OUTROOT"
SUMMARY=$OUTROOT/summary.csv
DETAIL=$OUTROOT/summary_detailed.csv
printf 'scene,exit_code,log_path,result_dir\n' > "$SUMMARY"
printf 'scene,exit_code,elapsed_seconds,result_dir,log_path,orb_cfg,gaussian_cfg,binary_sha256,config_sha256\n' > "$DETAIL"
BIN_SHA=$(sha256sum "$BIN" | cut -d' ' -f1)

for scene in "${SCENES[@]}"; do
    ORB_CFG=$ROOT/cfg/ORB_SLAM3/RGB-D/Replica/$scene.yaml
    GAUSS_CFG=$ROOT/cfg/gaussian_mapper/RGB-D/Replica/A_baseline/$scene.yaml
    DATA=$DATA_ROOT/$scene
    OUT=$OUTROOT/$scene
    mkdir -p "$OUT"
    LOG=$OUT/run.log

    echo "[batch] scene=$scene"
    fail_reason=""
    [ -f "$ORB_CFG" ] || fail_reason="missing ORB config: $ORB_CFG"
    [ -f "$GAUSS_CFG" ] || fail_reason="missing Gaussian config: $GAUSS_CFG"
    [ -d "$DATA/results" ] || fail_reason="missing dataset: $DATA/results"
    image_count=$(find "$DATA/results" -maxdepth 1 -name 'frame*.jpg' 2>/dev/null | wc -l)
    [ "$image_count" -gt 0 ] || fail_reason="no frame images under $DATA/results"

    if [ -n "$fail_reason" ]; then
        echo "[batch][FAIL] $scene: $fail_reason" | tee -a "$LOG"
        printf '%s,2,%s,%s\n' "$scene" "$LOG" "$OUT" >> "$SUMMARY"
        printf '%s,2,0,%s,%s,%s,%s,%s,\n' "$scene" "$OUT" "$LOG" "$ORB_CFG" "$GAUSS_CFG" "$BIN_SHA" >> "$DETAIL"
        continue
    fi
    if [ "${CHECK_ONLY:-0}" = "1" ]; then
        echo "[batch][CHECK] $scene images=$image_count orb=$ORB_CFG gaussian=$GAUSS_CFG out=$OUT"
        printf '%s,0,%s,%s\n' "$scene" "$LOG" "$OUT" >> "$SUMMARY"
        printf '%s,0,0,%s,%s,%s,%s,%s,\n' "$scene" "$OUT" "$LOG" "$ORB_CFG" "$GAUSS_CFG" "$BIN_SHA" >> "$DETAIL"
        continue
    fi

    CFG_SHA=$(sha256sum "$GAUSS_CFG" | cut -d' ' -f1)
    {
        echo "# frozen A clean baseline"
        echo "scene=$scene"
        echo "binary=$BIN"
        echo "binary_sha256=$BIN_SHA"
        echo "orb_cfg=$ORB_CFG"
        echo "gaussian_cfg=$GAUSS_CFG"
        echo "gaussian_cfg_sha256=$CFG_SHA"
        echo "dataset=$DATA"
        echo "images=$image_count"
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
    LD_LIBRARY_PATH="$FROZEN/bin" timeout -k 20 "$SHUTDOWN_LIMIT" \
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
            > "$LOG" 2>&1
    rc=$?
    t1=$(date +%s)
    elapsed=$((t1 - t0))
    printf '%s,%d,%s,%s\n' "$scene" "$rc" "$LOG" "$OUT" >> "$SUMMARY"
    printf '%s,%d,%d,%s,%s,%s,%s,%s,%s\n' \
        "$scene" "$rc" "$elapsed" "$OUT" "$LOG" "$ORB_CFG" "$GAUSS_CFG" "$BIN_SHA" "$CFG_SHA" >> "$DETAIL"

    if [ "$rc" -eq 0 ]; then
        echo "[batch][OK] scene=$scene elapsed=${elapsed}s out=$OUT"
    else
        echo "[batch][FAIL] scene=$scene exit=$rc elapsed=${elapsed}s log=$LOG"
    fi
done

echo "[batch] summary: $SUMMARY"
column -s, -t "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
