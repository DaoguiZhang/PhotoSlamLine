#!/usr/bin/env bash
# =============================================================================
# office2 seed reproducibility ablation.
#   group "default": no PHOTO_SLAM_GAUSSIAN_KF_SEED -> original random_device
#                     per-call shuffle (non-deterministic).
#   group "seed0"  : PHOTO_SLAM_GAUSSIAN_KF_SEED=0 -> persistent mt19937(0),
#                     deterministic shuffle GIVEN identical keyframe input list.
# Interleaved serial: default r1, seed0 r1, default r2, seed0 r2, default r3, seed0 r3.
# KF shuffle summaries ([KfShuffleDiag]) captured for determinism analysis.
# Policy: REFUSES to overwrite an existing output directory.
# =============================================================================
set -euo pipefail

ROOT=/workspace/code/Photo-SLAM-L
BIN=$ROOT/bin/replica_mono
VOC=$ROOT/ORB-SLAM3/Vocabulary/ORBvoc.txt
SL=$ROOT/cfg/ORB_SLAM3/Monocular/Replica/office2.yaml
GS=$ROOT/cfg/gaussian_mapper/Monocular/Replica/replica_mono_diag.yaml
DS=/workspace/code/SEGS-SLAM/datasets/replica/office2
GT=$DS/pose_TUM.txt
EV=$ROOT/scripts/eval_ate_rpe.py
BASE=$ROOT/results/diagnosis/office2/seed_ablation
SO=$ROOT/ORB-SLAM3/lib/libORB_SLAM3.so

run_one () {  # $1=group(default|seed0)  $2=runindex
    local GRP=$1 R=$2
    local OUT="$BASE/$GRP/run$R"
    if [ -e "$OUT" ]; then
        echo "REFUSE: output dir already exists: $OUT" >&2
        exit 2
    fi
    mkdir -p "$OUT/cwd"

    local SEED_ENV=()
    [ "$GRP" = "seed0" ] && SEED_ENV=(PHOTO_SLAM_GAUSSIAN_KF_SEED=0)

    {
        echo "=== metadata ==="
        echo "date: $(date -Is)"
        echo "seq: office2  group: $GRP  run: $R"
        echo "seed: ${SEED_ENV[*]:-(default random_device)}"
        echo "libORB_SLAM3.so: $(md5sum "$SO" | awk '{print $1}')"
        echo "bin/replica_mono: $(md5sum "$BIN" | awk '{print $1}')"
        echo "env: PHOTO_SLAM_LINE_MODE=2 PO_LINE=1 LBA_LINE=1 DEBUG_MONO_INIT=1 DEBUG_LINE_EDGES=1 DEBUG_KF_SHUFFLE=1"
    } > "$OUT/metadata.txt"

    local T0=$(date +%s)
    local EC=0
    ( cd "$OUT/cwd" && env \
        "${SEED_ENV[@]}" \
        PHOTO_SLAM_LINE_MODE=2 PHOTO_SLAM_PO_LINE=1 PHOTO_SLAM_LBA_LINE=1 \
        PHOTO_SLAM_DEBUG_MONO_INIT=1 PHOTO_SLAM_DEBUG_LINE_EDGES=1 \
        PHOTO_SLAM_DEBUG_KF_SHUFFLE=1 \
        "$BIN" "$VOC" "$SL" "$GS" "$DS" "$OUT/" no_viewer ) \
        > "$OUT/run.log" 2>&1 || EC=$?
    local T1=$(date +%s)
    echo "exit=$EC wall=$((T1-T0))s" >> "$OUT/run.log"

    if [ -f "$OUT/TrackingTime.txt" ]; then
        python3 - "$OUT/TrackingTime.txt" >> "$OUT/run.log" <<'PY'
import sys, numpy as np
t = np.loadtxt(sys.argv[1])
print(f"[TrackingTime] n={len(t)} mean={t.mean():.4f}s median={np.median(t):.4f}s "
      f"p99={np.percentile(t,99):.4f}s max={t.max():.4f}s sum={t.sum():.2f}s")
PY
    fi

    if [ -f "$OUT/CameraTrajectory_TUM.txt" ]; then
        python3 "$EV" "$GT" "$OUT/CameraTrajectory_TUM.txt" \
            "$OUT/ate.txt" "$OUT/rpe.txt" "$OUT/ape.txt" >> "$OUT/run.log" 2>&1 || \
            echo "EVAL_FAILED" >> "$OUT/run.log"
    fi

    # Extract shuffle summaries (lightweight digest of RNG output + input list)
    grep -aE "\[KfShuffleDiag\]" "$OUT/run.log" > "$OUT/kf_shuffle_diag.txt" || true

    echo "$GRP/run$R exit=$EC" >> "$BASE/_status.txt"
    echo "[done] $GRP/run$R exit=$EC"
}

mkdir -p "$BASE"
: > "$BASE/_status.txt"
for R in 1 2 3; do
    run_one default "$R"
    run_one seed0   "$R"
done
echo "ALL_DONE" >> "$BASE/_status.txt"
