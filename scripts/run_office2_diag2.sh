#!/usr/bin/env bash
# Run one office2 B/P/L/C ablation run (independent PO/LBA line-edge toggles).
#
# Usage:
#   scripts/run_office2_diag2.sh <PO_LINE(0|1)> <LBA_LINE(0|1)> <RUN_INDEX>
#
# All runs use PHOTO_SLAM_LINE_MODE=2 (line frontend active). The two toggles
# independently enable/disable line edges in Pose Optimization and Local BA:
#   0/0 = B   1/0 = P   0/1 = L   1/1 = C
set -u

PO="${1:?usage: <PO_LINE 0|1> <LBA_LINE 0|1> <RUN>}"
LBA="${2:?usage: <PO_LINE 0|1> <LBA_LINE 0|1> <RUN>}"
RUN="${3:?usage: <PO_LINE 0|1> <LBA_LINE 0|1> <RUN>}"

case "${PO}${LBA}" in
  00) GROUP="B_po0_lba0";;
  10) GROUP="P_po1_lba0";;
  01) GROUP="L_po0_lba1";;
  11) GROUP="C_po1_lba1";;
  *) echo "bad PO/LBA"; exit 2;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BIN="${PROJECT_ROOT}/bin/replica_mono"
VOCAB="${PROJECT_ROOT}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
SLAM_CFG="${PROJECT_ROOT}/cfg/ORB_SLAM3/Monocular/Replica/office2.yaml"
GS_CFG="${PROJECT_ROOT}/cfg/gaussian_mapper/Monocular/Replica/replica_mono_diag.yaml"
DATASET="/workspace/code/SEGS-SLAM/datasets/replica/office2"
GT="${DATASET}/pose_TUM.txt"

OUT="${PROJECT_ROOT}/results/diagnosis/office2/exp2/${GROUP}/run${RUN}"
CWD="${OUT}/cwd"
mkdir -p "${CWD}"

export PHOTO_SLAM_LINE_MODE=2
export PHOTO_SLAM_PO_LINE="${PO}"
export PHOTO_SLAM_LBA_LINE="${LBA}"
export PHOTO_SLAM_DEBUG_MONO_INIT=1
export PHOTO_SLAM_DEBUG_LINE_EDGES=1

LOG="${OUT}/run.log"

{
  echo "=== command ==="
  echo "PHOTO_SLAM_LINE_MODE=2 PHOTO_SLAM_PO_LINE=${PO} PHOTO_SLAM_LBA_LINE=${LBA} \\"
  echo "  PHOTO_SLAM_DEBUG_MONO_INIT=1 PHOTO_SLAM_DEBUG_LINE_EDGES=1 \\"
  echo "  ${BIN} ${VOCAB} ${SLAM_CFG} ${GS_CFG} ${DATASET} ${OUT}/ no_viewer"
  echo "=== git ==="
  git -C "${PROJECT_ROOT}" rev-parse HEAD
  git -C "${PROJECT_ROOT}" status --short | head -40
  echo "=== binary checksums ==="
  md5sum "${BIN}" "${PROJECT_ROOT}/ORB-SLAM3/lib/libORB_SLAM3.so" 2>/dev/null
  echo "=== gpu before ==="
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"
} | tee "${LOG}"

cp "${SLAM_CFG}" "${OUT}/office2.yaml.used"
cp "${GS_CFG}" "${OUT}/replica_mono_diag.yaml.used"

echo "=== run start: ${GROUP}/run${RUN} (PO=${PO} LBA=${LBA}) ===" | tee -a "${LOG}"

( cd "${CWD}" && \
  "${BIN}" "${VOCAB}" "${SLAM_CFG}" "${GS_CFG}" "${DATASET}" "${OUT}/" no_viewer \
) 2>&1 | tee -a "${LOG}"

echo "=== run end: exit=${PIPESTATUS[0]} ===" | tee -a "${LOG}"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tee -a "${LOG}"

# BA before/after debug exports (KF id 2)
for f in 2_Keyframe_Camera_before.txt 2_Keyframe_Camera_after.txt \
         2_Keyframe_MapPoints_before.obj 2_Keyframe_MapPoints_after.obj \
         2_Keyframe_Maplines_before.obj 2_Keyframe_Maplines_after.obj; do
  [ -f "${CWD}/${f}" ] && cp "${CWD}/${f}" "${OUT}/${f}"
done

{
  echo "=== metrics ==="
  echo "init_frame_ts: $(awk 'NR==1{print $1}' "${OUT}/CameraTrajectory_TUM.txt" 2>/dev/null)"
  echo "trajectory_frames: $(wc -l < "${OUT}/CameraTrajectory_TUM.txt" 2>/dev/null)"
  echo "keyframes: $(wc -l < "${OUT}/KeyFrameTrajectory_TUM.txt" 2>/dev/null)"
  grep -E "New Map created with" "${LOG}" | tail -1
  grep -E "Map .* has .* KFs" "${LOG}" | tail -3
} | tee -a "${LOG}" > "${OUT}/metrics.txt"

if [ -f "${OUT}/CameraTrajectory_TUM.txt" ]; then
  python3 "${SCRIPT_DIR}/eval_ate_rpe.py" "${GT}" "${OUT}/CameraTrajectory_TUM.txt" \
      "${OUT}/ate.txt" "${OUT}/rpe.txt" "${OUT}/ape.txt" 2>&1 | tee -a "${LOG}"
fi

echo "=== done ${GROUP}/run${RUN} ===" | tee -a "${LOG}"
