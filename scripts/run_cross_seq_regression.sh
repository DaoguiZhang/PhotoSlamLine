#!/usr/bin/env bash
# =============================================================================
# Cross-sequence regression for the map-mutex stability fix (C0 vs pre-fix C).
#
# Sequences: office0 (mandatory) + room0 (chosen for cross-scene generalization;
#            not used in any prior tuning; 2000 frames + GT, config present).
# Versions:  fix    = post-fix  C0 (map-mutex fix, snapshot c0_fix)
#            prefix = pre-fix   C  (map-mutex fix REVERTED, snapshot c0_prefix)
#            -> the ONLY source difference between the two .so snapshots is the
#               map-mutex lock around LBA line write-back (section 13).
#
# Identical numeric params / data / eval / Gaussian config / logging for both.
# Interleaved serial execution (fix r1, prefix r1, ... fix r3, prefix r3).
#
# Policy: REFUSES to overwrite an existing output directory (no rm -rf).
#         The in-use shared library (ORB-SLAM3/lib) is NOT touched; selection is
#         via LD_LIBRARY_PATH (bin has RUNPATH, so LD_LIBRARY_PATH wins).
# =============================================================================
set -euo pipefail

ROOT=/workspace/code/Photo-SLAM-L
BIN=$ROOT/bin/replica_mono
VOC=$ROOT/ORB-SLAM3/Vocabulary/ORBvoc.txt
GS=$ROOT/cfg/gaussian_mapper/Monocular/Replica/replica_mono_diag.yaml
EV=$ROOT/scripts/eval_ate_rpe.py

SNAP_FIX=$ROOT/results/diagnosis/_snapshots/c0_fix
SNAP_PREFIX=$ROOT/results/diagnosis/_snapshots/c0_prefix
BASE=$ROOT/results/diagnosis/cross_seq

# LD_LIBRARY_PATH: snapshot dir FIRST, then the full set of lib dirs normally
# resolved via RUNPATH so that every other dependency still resolves.
LIBPATH_TAIL="/workspace/code/Photo-SLAM-L/lib:/workspace/code/Photo-SLAM-L/ORB-SLAM3/lib:/workspace/code/Photo-SLAM-L/ORB-SLAM3/Thirdparty/DBoW2/lib:/workspace/code/Photo-SLAM-L/ORB-SLAM3/Thirdparty/g2o/lib:/workspace/code/Photo-SLAM-L/ORB-SLAM3/Thirdparty/Sophus/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/cuda/lib64"

declare -A SL_CFG GT_CFG DS_CFG
SL_CFG[office0]=$ROOT/cfg/ORB_SLAM3/Monocular/Replica/office0.yaml
SL_CFG[room0]=$ROOT/cfg/ORB_SLAM3/Monocular/Replica/room0.yaml
GT_CFG[office0]=/workspace/code/SEGS-SLAM/datasets/replica/office0/pose_TUM.txt
GT_CFG[room0]=/workspace/code/SEGS-SLAM/datasets/replica/room0/pose_TUM.txt
DS_CFG[office0]=/workspace/code/SEGS-SLAM/datasets/replica/office0
DS_CFG[room0]=/workspace/code/SEGS-SLAM/datasets/replica/room0

run_one () {  # $1=seq  $2=version(fix|prefix)  $3=runindex
    local SEQ=$1 VER=$2 R=$3
    local SNAP="$SNAP_FIX"
    [ "$VER" = "prefix" ] && SNAP="$SNAP_PREFIX"
    [ -f "$SNAP/libORB_SLAM3.so" ] || { echo "MISSING SNAPSHOT: $SNAP/libORB_SLAM3.so" >&2; exit 3; }

    local OUT="$BASE/$SEQ/$VER/run$R"
    if [ -e "$OUT" ]; then
        echo "REFUSE: output dir already exists: $OUT" >&2
        exit 2
    fi
    mkdir -p "$OUT/cwd"

    {
        echo "=== metadata ==="
        echo "date: $(date -Is)"
        echo "seq: $SEQ  version: $VER  run: $R"
        echo "libORB_SLAM3.so: $(md5sum "$SNAP/libORB_SLAM3.so" | awk '{print $1}')"
        echo "SL config: ${SL_CFG[$SEQ]}"
        echo "Gaussian config: $GS"
        echo "env: PHOTO_SLAM_LINE_MODE=2 PHOTO_SLAM_PO_LINE=1 PHOTO_SLAM_LBA_LINE=1"
        echo "     PHOTO_SLAM_DEBUG_MONO_INIT=1 PHOTO_SLAM_DEBUG_LINE_EDGES=1"
    } > "$OUT/metadata.txt"

    local T0=$(date +%s)
    local EC=0
    ( cd "$OUT/cwd" && env \
        LD_LIBRARY_PATH="$SNAP:$LIBPATH_TAIL" \
        PHOTO_SLAM_LINE_MODE=2 PHOTO_SLAM_PO_LINE=1 PHOTO_SLAM_LBA_LINE=1 \
        PHOTO_SLAM_DEBUG_MONO_INIT=1 PHOTO_SLAM_DEBUG_LINE_EDGES=1 \
        "$BIN" "$VOC" "${SL_CFG[$SEQ]}" "$GS" "${DS_CFG[$SEQ]}" "$OUT/" no_viewer ) \
        > "$OUT/run.log" 2>&1 || EC=$?
    local T1=$(date +%s)
    echo "exit=$EC wall=$((T1-T0))s" >> "$OUT/run.log"

    for f in 2_Keyframe_Camera_before.txt 2_Keyframe_Camera_after.txt; do
        [ -f "$OUT/cwd/$f" ] && cp "$OUT/cwd/$f" "$OUT/$f"
    done

    if [ -f "$OUT/TrackingTime.txt" ]; then
        python3 - "$OUT/TrackingTime.txt" >> "$OUT/run.log" <<'PY'
import sys, numpy as np
t = np.loadtxt(sys.argv[1])
print(f"[TrackingTime] n={len(t)} mean={t.mean():.4f}s median={np.median(t):.4f}s "
      f"p99={np.percentile(t,99):.4f}s max={t.max():.4f}s sum={t.sum():.2f}s")
PY
    fi

    if [ -f "$OUT/CameraTrajectory_TUM.txt" ]; then
        python3 "$EV" "${GT_CFG[$SEQ]}" "$OUT/CameraTrajectory_TUM.txt" \
            "$OUT/ate.txt" "$OUT/rpe.txt" "$OUT/ape.txt" >> "$OUT/run.log" 2>&1 || \
            echo "EVAL_FAILED" >> "$OUT/run.log"
    fi

    {
        echo "--- reset/lost/reloc scan ---"
        grep -cE "Reseting|reset|LOST|Lost|RELOCALIZ|New Map created" "$OUT/run.log" || true
        echo "--- LineMode startup line ---"
        grep -m1 "\[LineMode\]" "$OUT/run.log" || true
    } >> "$OUT/metadata.txt"

    echo "$SEQ/$VER/run$R exit=$EC" >> "$BASE/_status.txt"
    echo "[done] $SEQ/$VER/run$R exit=$EC"
}

: > "$BASE/_status.txt" 2>/dev/null || { mkdir -p "$BASE"; : > "$BASE/_status.txt"; }

for SEQ in office0 room0; do
    for R in 1 2 3; do
        run_one "$SEQ" fix    "$R"
        run_one "$SEQ" prefix "$R"
    done
done

echo "ALL_DONE" >> "$BASE/_status.txt"
