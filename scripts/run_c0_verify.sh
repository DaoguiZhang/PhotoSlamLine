#!/usr/bin/env bash
# =============================================================================
# C0 acceptance run for office2 Mono (final version):
#   PHOTO_SLAM_LINE_MODE=2, PO_LINE=1, LBA_LINE=1, LBA params at C0 defaults,
#   map-mutex stability fix in place, diagnostics ON.
# This is a VERSION ACCEPTANCE run (not an ATE target): success = exit 0,
# correct PO/LBA line-edge activity, expected coverage, no reset/deadlock.
#
# Policy: REFUSES to overwrite an existing output directory (no rm -rf).
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
BASE=$ROOT/results/diagnosis/office2/c0_verify
SO=$ROOT/ORB-SLAM3/lib/libORB_SLAM3.so

OUT=$BASE/run1
if [ -e "$OUT" ]; then
    echo "REFUSE: output dir already exists: $OUT" >&2
    exit 2
fi
mkdir -p "$OUT/cwd"

# Record environment + binary identity BEFORE running
{
    echo "=== c0_verify run metadata ==="
    echo "date: $(date -Is)"
    echo "seq: office2"
    echo "config: PHOTO_SLAM_LINE_MODE=2 PHOTO_SLAM_PO_LINE=1 PHOTO_SLAM_LBA_LINE=1"
    echo "       PHOTO_SLAM_DEBUG_MONO_INIT=1 PHOTO_SLAM_DEBUG_LINE_EDGES=1"
    echo "libORB_SLAM3.so: $(md5sum "$SO" | awk '{print $1}')"
    echo "bin/replica_mono: $(md5sum "$BIN" | awk '{print $1}')"
    echo "SL config: $SL"
    echo "Gaussian config: $GS"
} > "$OUT/metadata.txt"

T0=$(date +%s)
EC=0
( cd "$OUT/cwd" && env \
    PHOTO_SLAM_LINE_MODE=2 PHOTO_SLAM_PO_LINE=1 PHOTO_SLAM_LBA_LINE=1 \
    PHOTO_SLAM_DEBUG_MONO_INIT=1 PHOTO_SLAM_DEBUG_LINE_EDGES=1 \
    "$BIN" "$VOC" "$SL" "$GS" "$DS" "$OUT/" no_viewer ) > "$OUT/run.log" 2>&1 || EC=$?
T1=$(date +%s)
echo "exit=$EC wall=$((T1-T0))s" >> "$OUT/run.log"

# Preserve CWD-relative debug exports (written by LocalMapping at KF id 2)
for f in 2_Keyframe_Camera_before.txt 2_Keyframe_Camera_after.txt; do
    [ -f "$OUT/cwd/$f" ] && cp "$OUT/cwd/$f" "$OUT/$f"
done

# Tracking-time stats (per-frame times written by replica_mono)
if [ -f "$OUT/TrackingTime.txt" ]; then
    python3 - "$OUT/TrackingTime.txt" >> "$OUT/run.log" <<'PY'
import sys, numpy as np
t = np.loadtxt(sys.argv[1])
print(f"[TrackingTime] n={len(t)} mean={t.mean():.4f}s median={np.median(t):.4f}s "
      f"p99={np.percentile(t,99):.4f}s max={t.max():.4f}s sum={t.sum():.2f}s")
PY
fi

# ATE / RPE
if [ -f "$OUT/CameraTrajectory_TUM.txt" ]; then
    python3 "$EV" "$GT" "$OUT/CameraTrajectory_TUM.txt" \
        "$OUT/ate.txt" "$OUT/rpe.txt" "$OUT/ape.txt" >> "$OUT/run.log" 2>&1 || \
        echo "EVAL_FAILED" >> "$OUT/run.log"
fi

# Reset / lost / reloc scan
{
    echo "--- reset/lost/reloc scan ---"
    grep -cE "Reseting|reset|LOST|Lost|RELOCALIZ|New Map created" "$OUT/run.log" || true
    echo "--- LineMode startup line ---"
    grep -m1 "\[LineMode\]" "$OUT/run.log" || true
} >> "$OUT/metadata.txt"

echo "c0_verify done: exit=$EC (see $OUT/run.log)"
