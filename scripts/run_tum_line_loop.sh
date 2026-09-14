#!/usr/bin/env bash
# Line loop-closing A/B/C experiment harness (TUM monocular).
#
#   A (LINE_LOOP=0): C0 baseline  -- point-only loop closing.
#   B (LINE_LOOP=1): point loop detection + line map/Gaussian sync (no line
#                    Sim3 residuals, point-only GBA with line writeback).
#   C (LINE_LOOP=2): B + line residuals in Sim(3) refinement + point-line GBA.
#
# Usage:
#   scripts/run_tum_line_loop.sh <MODE 0|1|2> <RUN_INDEX> [DATASET_DIR]
#
# Default dataset: TUM freiburg3_long_office_household (contains a revisit).
# Requires groundtruth.txt in the dataset directory for ATE evaluation.
set -u

MODE="${1:?usage: <MODE 0|1|2> <RUN_INDEX> [DATASET_DIR]}"
RUN="${2:?usage: <MODE 0|1|2> <RUN_INDEX> [DATASET_DIR]}"
DATASET="${3:-/workspace/code/SEGS-SLAM/datasets/tum/rgbd_dataset_freiburg3_long_office_household}"

case "${MODE}" in
  0) GROUP="A_c0_point_loop";;
  1) GROUP="B_line_sync";;
  2) GROUP="C_line_sim3_gba";;
  *) echo "bad MODE (0=A,1=B,2=C)"; exit 2;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BIN="${PROJECT_ROOT}/bin/tum_mono"
VOCAB="${PROJECT_ROOT}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
SLAM_CFG="${PROJECT_ROOT}/cfg/ORB_SLAM3/Monocular/TUM/tum_freiburg3_long_office_household.yaml"
GS_CFG="${PROJECT_ROOT}/cfg/gaussian_mapper/Monocular/TUM/tum_freiburg3_long_office_household.yaml"
GT="${DATASET}/groundtruth.txt"

OUT="${PROJECT_ROOT}/results/line_loop/${GROUP}/run${RUN}"
CWD="${OUT}/cwd"
mkdir -p "${CWD}"

# C0 line pipeline config (full point-line) -- do NOT touch FAST/init/PO/LBA.
export PHOTO_SLAM_LINE_MODE=2
export PHOTO_SLAM_PO_LINE=1
export PHOTO_SLAM_LBA_LINE=1
export PHOTO_SLAM_LINE_LOOP="${MODE}"
export PHOTO_SLAM_DEBUG_LINE_LOOP=1

LOG="${OUT}/run.log"

{
  echo "=== command ==="
  echo "PHOTO_SLAM_LINE_MODE=2 PO_LINE=1 LBA_LINE=1 PHOTO_SLAM_LINE_LOOP=${MODE} PHOTO_SLAM_DEBUG_LINE_LOOP=1 \\"
  echo "  ${BIN} ${VOCAB} ${SLAM_CFG} ${GS_CFG} ${DATASET} ${OUT}/ no_viewer"
  echo "=== git ==="
  git -C "${PROJECT_ROOT}" rev-parse HEAD
  git -C "${PROJECT_ROOT}" status --short | head -40
  echo "=== binary checksums ==="
  md5sum "${BIN}" "${PROJECT_ROOT}/ORB-SLAM3/lib/libORB_SLAM3.so" \
           "${PROJECT_ROOT}/lib/libgaussian_mapper.so" 2>/dev/null
} | tee "${LOG}"

cp "${SLAM_CFG}" "${OUT}/slam.yaml.used"
cp "${GS_CFG}" "${OUT}/gs.yaml.used"

echo "=== run start: ${GROUP}/run${RUN} (MODE=${MODE}) ===" | tee -a "${LOG}"

export LD_LIBRARY_PATH="${PROJECT_ROOT}/ORB-SLAM3/lib:${PROJECT_ROOT}/ORB-SLAM3/Thirdparty/g2o/lib:${PROJECT_ROOT}/ORB-SLAM3/Thirdparty/DBoW2/lib:${PROJECT_ROOT}/lib"

( cd "${CWD}" && \
  "${BIN}" "${VOCAB}" "${SLAM_CFG}" "${GS_CFG}" "${DATASET}" "${OUT}/" no_viewer \
) 2>&1 | tee -a "${LOG}"

echo "=== run end: exit=${PIPESTATUS[0]} ===" | tee -a "${LOG}"

{
  echo "=== metrics ==="
  echo "trajectory_frames: $(wc -l < "${OUT}/CameraTrajectory_TUM.txt" 2>/dev/null)"
  echo "keyframes: $(wc -l < "${OUT}/KeyFrameTrajectory_TUM.txt" 2>/dev/null)"
  echo "loop_accepted: $(grep -c 'candidate accepted' "${LOG}" 2>/dev/null)"
  echo "loop_corrections: $(grep -c 'CorrectLoopWithLine' "${LOG}" 2>/dev/null)"
  echo "gba_triggers: $(grep -c 'GBA triggered' "${LOG}" 2>/dev/null)"
  echo "gba_writebacks: $(grep -c 'GBA writeback' "${LOG}" 2>/dev/null)"
  grep -E "New Map created with" "${LOG}" | tail -1
} | tee -a "${LOG}" > "${OUT}/metrics.txt"

if [ -f "${GT}" ] && [ -f "${OUT}/CameraTrajectory_TUM.txt" ]; then
  python3 "${SCRIPT_DIR}/eval_tum_sim3.py" "${GT}" "${OUT}/CameraTrajectory_TUM.txt" \
      "${OUT}/ate_sim3.txt" 2>&1 | tee -a "${LOG}"
fi

echo "=== done ${GROUP}/run${RUN} ===" | tee -a "${LOG}"
