#!/usr/bin/env bash
# Reproduce the room1 LOOP<->Gaussian coordinate-sync fix validation.
#
# The fix is in:
#   src/gaussian_mapper_line.cpp  (LoopClosingBA handler)
#   src/gaussian_model_line.cpp   (applyGbaCorrections + allocateCoordEventId)
#   include/gaussian_mapper_line.h
#   include/gaussian_model_line.h
#
# Set SRC to the isolated source tree and OUT to a new output root. The script
# never touches production results.
set -euo pipefail

SRC_FIX="${SRC_FIX:-/tmp/psl-room1-control/src-fix}"
SRC_AUDIT="${SRC_AUDIT:-/tmp/psl-room1-control/src-fix-audit}"
CTRL="${CTRL:-/tmp/psl-room1-control}"
OUT="${OUT:-$CTRL/repro-loop-fix}"
DATASET="${DATASET:-/workspace/code/SEGS-SLAM/datasets/replica/room1}"
VOCAB="${VOCAB:-/workspace/code/PSL-perf/ORB-SLAM3/Vocabulary/ORBvoc.txt}"
ORB_CFG="${ORB_CFG:-$CTRL/config/room1_orb.yaml}"
GAUSS_CFG="${GAUSS_CFG:-$CTRL/config/room1_gaussian.yaml}"

mkdir -p "$OUT"
echo "[repro] src_fix=$SRC_FIX src_audit=$SRC_AUDIT"
sha256sum "$ORB_CFG" "$GAUSS_CFG"

# 1. Build the isolated trees (top-level only; ORB-SLAM3 headers are unchanged).
cmake -S "$SRC_FIX" -B "$SRC_FIX/build-fix" \
  -DCMAKE_BUILD_TYPE=Release \
  -DTorch_DIR=/usr/local/lib/python3.10/dist-packages/torch/share/cmake/Torch \
  -DOpenCV_DIR=/usr/local/lib/cmake/opencv4
cmake --build "$SRC_FIX/build-fix" -j8 --target replica_rgbd test_loop_coord_sync
cmake -S "$SRC_AUDIT" -B "$SRC_AUDIT/build-fix-audit" \
  -DCMAKE_BUILD_TYPE=Release \
  -DTorch_DIR=/usr/local/lib/python3.10/dist-packages/torch/share/cmake/Torch \
  -DOpenCV_DIR=/usr/local/lib/cmake/opencv4
cmake --build "$SRC_AUDIT/build-fix-audit" -j8 --target replica_rgbd

# 2. Pure-CPU rigid/Sim(3) contract: identity, known rigid, scale, covariance,
#    line anchor/direction/cache, unselected rows, count, duplicate, order.
"$SRC_FIX/bin/test_loop_coord_sync" 2>&1 | tee "$OUT/contract.log"

# 3. Same-event quality audit. The LOOP event is not guaranteed on every run:
#    repeat until a non-identity event appears. Event arms are:
#      before camera + before model
#      after camera  + before model   (legacy gate-skipped map)
#      after camera  + corrected map  (fixed path)
for r in 1 2 3 4 5 6; do
  mkdir -p "$OUT/r$r"
  env -u CUDA_LAUNCH_BLOCKING \
      -u PHOTO_SLAM_RASTER_AGG_SELFCHECK \
      -u PHOTO_SLAM_RASTER_PROF \
      -u PHOTO_SLAM_GPU_PROF \
      -u PHOTO_SLAM_FROZEN_ABLATION \
    PHOTO_SLAM_LINE_MODE=2 \
    PHOTO_SLAM_GBA_APPLY_CORRECTION=0 \
    PHOTO_SLAM_APPEND_STEPS=0 \
    PHOTO_SLAM_SEED=0 \
    PHOTO_SLAM_PSNR_DIAG=1 \
    PHOTO_SLAM_PSNR_DIAG_FID=0 \
    PHOTO_SLAM_RASTER_BACKEND=C \
    PHOTO_SLAM_EVENT_QUALITY_DIR="$OUT/r$r/event_quality" \
    PHOTO_SLAM_EVENT_QUALITY_VIEWS=24 \
    timeout --signal=TERM --kill-after=30s 1800 \
      "$SRC_AUDIT/bin/replica_rgbd" "$VOCAB" "$ORB_CFG" "$GAUSS_CFG" "$DATASET" \
      "$OUT/r$r/out" no_viewer > "$OUT/r$r/run.log" 2>&1 || true

  if grep -q '\[LoopSync\]' "$OUT/r$r/run.log"; then
    echo "[repro] LOOP found in run $r"
    grep -E '\[LineLoop-Gaussian\]|\[LoopSync\]|\[EventQuality\] loop' \
      "$OUT/r$r/run.log" | tee "$OUT/r$r/event_summary.log"
    cat "$OUT/r$r"/event_quality/loop_evt1/summary.csv | tee -a "$OUT/event_summary.csv"
    break
  fi
done

# 4. Clean production-like regression (NO event diagnostics: those diagnostics
#    intentionally run counterfactual arms and make the final PSNR unusable).
env -u CUDA_LAUNCH_BLOCKING \
    -u PHOTO_SLAM_RASTER_AGG_SELFCHECK \
    -u PHOTO_SLAM_RASTER_PROF \
    -u PHOTO_SLAM_GPU_PROF \
    -u PHOTO_SLAM_FROZEN_ABLATION \
  PHOTO_SLAM_LINE_MODE=2 \
  PHOTO_SLAM_GBA_APPLY_CORRECTION=0 \
  PHOTO_SLAM_APPEND_STEPS=0 \
  PHOTO_SLAM_SEED=0 \
  PHOTO_SLAM_PSNR_DIAG=1 \
  PHOTO_SLAM_PSNR_DIAG_FID=0 \
  PHOTO_SLAM_RASTER_BACKEND=C \
  timeout --signal=TERM --kill-after=30s 1800 \
    "$SRC_AUDIT/bin/replica_rgbd" "$VOCAB" "$ORB_CFG" "$GAUSS_CFG" "$DATASET" \
    "$OUT/regression" no_viewer > "$OUT/regression.log" 2>&1 || true

echo "[repro] done. Check $OUT"
echo "[repro] native PSNR:"
find "$OUT/regression" -name psnr_gaussian_splatting.txt -exec sh -c \
  'echo $1; python3 - "$1" <<"PY"
import re,sys,statistics
v=[float(m.group(1)) for l in open(sys.argv[1],errors="ignore") if (m:=re.match(r"\\s*\\d+\\s+([0-9.]+)",l))]
print("n=%d mean=%.4f last=%.4f" % (len(v), statistics.mean(v), v[-1]))
PY' _ {} \;
