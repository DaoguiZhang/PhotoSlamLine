#!/usr/bin/env bash
# ============================================================================
# raster-perf-v6 / 四: normal online office0 run with the rasteriser backward
# backend as the ONLY variable.
#
# usage: run_online_ab.sh <A|B|C> <out_dir> [seq]
#
# Normal online entry point: full 2000-frame sequence, densification / clone /
# split / prune and the append-point schedule all active exactly as in a normal
# run. Nothing about the front end, keyframe policy or densify schedule changes.
# The only deliberate difference between arms is PHOTO_SLAM_RASTER_BACKEND.
#
# Line STRUCTURAL loss stays off for this round (PHOTO_SLAM_LINE_LOSS_EVERY=0),
# same as the frozen conditions, so the round measures the renderer and not the
# loss ablation.
#
# No audit / profiler / self-check switch is set: formal performance runs keep
# every diagnostic off.
# ============================================================================
set -euo pipefail

BACKEND="${1:?usage: run_online_ab.sh <A|B|C> <out_dir> [seq]}"
OUT="${2:?usage: run_online_ab.sh <A|B|C> <out_dir> [seq]}"
SEQ="${3:-office0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-/workspace/code/SEGS-SLAM/datasets}"
DS="${DATASET_ROOT}/replica/${SEQ}"
VOCAB="${ROOT}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
ORB_CFG="${ROOT}/cfg/ORB_SLAM3/RGB-D/Replica/${SEQ}.yaml"
GAU_CFG="${ROOT}/cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml"

[[ -d "${DS}/results" ]] || { echo "missing dataset ${DS}/results" >&2; exit 1; }
rm -rf "${OUT}"
mkdir -p "${OUT}/snapshots"
cp -- "${ORB_CFG}" "${OUT}/orb_settings.used.yaml"
cp -- "${GAU_CFG}" "${OUT}/gaussian_settings.used.yaml"
git -C "${ROOT}" rev-parse HEAD > "${OUT}/git_commit.txt"
sha256sum "${ROOT}/bin/replica_rgbd" "${ROOT}/ORB-SLAM3/lib/libORB_SLAM3.so" \
    > "${OUT}/binaries.sha256"

export PHOTO_SLAM_RASTER_BACKEND="${BACKEND}"
# line structural loss OFF for this round
export PHOTO_SLAM_LINE_LOSS_EVERY=0
export PHOTO_SLAM_LINE_MODE="${PHOTO_SLAM_LINE_MODE:-2}"
export PHOTO_SLAM_LINE_LOOP="${PHOTO_SLAM_LINE_LOOP:-0}"
# snapshot capture (a 193k-state checkpoint for the common-state experiment)
export PHOTO_SLAM_SNAPSHOT_DIR="${OUT}/snapshots"
export PHOTO_SLAM_SNAPSHOT_AT="${SNAPSHOT_AT:-9000,20000}"
# diagnostics deliberately OFF
unset PHOTO_SLAM_RASTER_PROF || true
unset PHOTO_SLAM_RASTER_AGG_SELFCHECK || true
unset PHOTO_SLAM_RASTER_CALLTRACE || true
unset PHOTO_SLAM_CREATION_AUDIT || true
unset PHOTO_SLAM_CREATION_LOG || true
unset PHOTO_SLAM_LINE_POP_AUDIT || true
unset PHOTO_SLAM_LINE_SAMPLE_AUDIT || true
unset PHOTO_SLAM_GPU_PROF || true

echo "[online] seq=${SEQ} backend=${BACKEND} out=${OUT}"
cd "${ROOT}"
"${ROOT}/bin/replica_rgbd" \
    "${VOCAB}" "${ORB_CFG}" "${GAU_CFG}" "${DS}" "${OUT}" no_viewer \
    2>&1 | tee "${OUT}/run.log"
echo "[online] done backend=${BACKEND}"
