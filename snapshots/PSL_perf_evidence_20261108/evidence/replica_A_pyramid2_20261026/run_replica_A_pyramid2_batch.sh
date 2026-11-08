#!/usr/bin/env bash
# Serial room1-8 pyramid2 batch on the frozen A binary.
# No evaluation is invoked here. Original result trees are preserved.
set -u

ROOT=${ROOT:-/workspace/code/PSL-perf}
FROZEN=${FROZEN:-$ROOT/evidence/frozen_A_clean_20261023}
BIN=$FROZEN/bin/replica_rgbd
VOCAB=$ROOT/ORB-SLAM3/Vocabulary/ORBvoc.txt
DATA_ROOT=${DATA_ROOT:-/workspace/code/SEGS-SLAM/datasets/replica}
CFG_DIR=${CFG_DIR:-$ROOT/evidence/replica_A_pyramid2_20261026/config}
BATCH_TAG=${BATCH_TAG:-20261026_A_pyramid2}
OUTROOT=${OUTROOT:-$ROOT/results/replica_A_pyramid2_${BATCH_TAG}/replica_rgbd_0}
SCENES=(office0 office1 office2 office3 office4 room0 room1 room2)

[ -x "$BIN" ] || { echo "missing binary: $BIN" >&2; exit 2; }
[ -f "$VOCAB" ] || { echo "missing vocabulary: $VOCAB" >&2; exit 2; }
mkdir -p "$OUTROOT"
SUMMARY=$OUTROOT/summary.csv
DETAIL=$OUTROOT/summary_detailed.csv
printf 'scene,exit_code,export_ok,log_path,result_dir\n' > "$SUMMARY"
printf 'scene,exit_code,export_ok,elapsed_seconds,result_dir,log_path,orb_cfg,gaussian_cfg,binary_sha256,config_sha256\n' > "$DETAIL"
BIN_SHA=$(sha256sum "$BIN" | cut -d' ' -f1)

check_export() {
    local out=$1
    local ply cams
    ply=$(find "$out" -type f -path '*/ply/point_cloud/*/point_cloud.ply' -size +1000000c -print -quit 2>/dev/null)
    cams=$(find "$out" -maxdepth 3 -type f -path '*/ply/cameras.json' -print -quit 2>/dev/null)
    [ -n "$ply" ] || return 1
    [ -n "$cams" ] || return 1
    for f in CameraTrajectory_TUM.txt KeyFrameTrajectory_TUM.txt \
             Global_Final_Map_Points.obj TotalOperationTime.txt run.log; do
        [ -s "$out/$f" ] || return 1
    done
    [ -s "$out/natural_end/cameras.csv" ] || return 1
    [ -s "$out/natural_end/meta.txt" ] || return 1
    return 0
}

for scene in "${SCENES[@]}"; do
    ORB_CFG=$ROOT/cfg/ORB_SLAM3/RGB-D/Replica/$scene.yaml
    GAUSS_CFG=$CFG_DIR/${scene}_pyramid2.yaml
    DATA=$DATA_ROOT/$scene
    OUT=$OUTROOT/$scene
    LOG=$OUT/run.log

    if [ -e "$OUT" ] && [ -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
        echo "[batch][ERROR] refusing non-empty output: $OUT" >&2
        printf '%s,2,0,%s,%s\n' "$scene" "$LOG" "$OUT" >> "$SUMMARY"
        continue
    fi
    mkdir -p "$OUT"
    CFG_SHA=$(sha256sum "$GAUSS_CFG" | cut -d' ' -f1)
    {
        echo "# frozen A, independent per-scene pyramid2 config"
        echo "scene=$scene"
        echo "binary=$BIN"
        echo "binary_sha256=$BIN_SHA"
        echo "library_path=$FROZEN/bin"
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
    LD_LIBRARY_PATH="$FROZEN/bin" ldd "$BIN" \
        | grep -E 'gaussian_viewer|gaussian_mapper|ORB_SLAM3|imgui|simple_knn|cuda_rasterizer|DBoW2|g2o' \
        > "$OUT/resolved_project_libs.txt"

    t0=$(date +%s)
    LD_LIBRARY_PATH="$FROZEN/bin" timeout -k 20 900 \
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
    echo "[batch] scene=$scene rc=$rc export_ok=$export_ok elapsed=$((t1 - t0))s out=$OUT"
done

column -s, -t "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
