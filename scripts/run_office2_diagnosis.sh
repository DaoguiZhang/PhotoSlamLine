#!/usr/bin/env bash
# Run one office2 Mono ablation run (A/B/C).
#
# Usage:
#   scripts/run_office2_diagnosis.sh <MODE> <RUN_INDEX>
#
#   MODE: 0 = A point-only
#         1 = B line-frontend-only (MapLines created, no line edges in optimization)
#         2 = C full point-line (current behavior)
#
# Everything (full log, trajectories, BA before/after debug exports, ATE, APE)
# is written to results/diagnosis/office2/<GROUP>/run<RUN>/.

set -u

MODE="${1:?usage: run_office2_diagnosis.sh <MODE(0|1|2)> <RUN_INDEX>}"
RUN="${2:?usage: run_office2_diagnosis.sh <MODE(0|1|2)> <RUN_INDEX>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

case "$MODE" in
  0) GROUP="A_point_only";;
  1) GROUP="B_line_frontend";;
  2) GROUP="C_point_line_current";;
  *) echo "ERROR: bad MODE $MODE (must be 0/1/2)" >&2; exit 2;;
esac

BIN="${PROJECT_ROOT}/bin/replica_mono"
VOCAB="${PROJECT_ROOT}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
SLAM_CFG="${PROJECT_ROOT}/cfg/ORB_SLAM3/Monocular/Replica/office2.yaml"
GS_CFG="${PROJECT_ROOT}/cfg/gaussian_mapper/Monocular/Replica/replica_mono_diag.yaml"
DATASET="/workspace/code/SEGS-SLAM/datasets/replica/office2"
GT="${DATASET}/pose_TUM.txt"

OUT="${PROJECT_ROOT}/results/diagnosis/office2/${GROUP}/run${RUN}"
# The LocalMapping debug block (mnId==2) writes 2_Keyframe_* to the CWD of the
# process, so we run the binary from a per-run scratch directory to avoid
# overwriting anything in the repo or in scripts/.
CWD="${OUT}/cwd"
mkdir -p "${CWD}"

export PHOTO_SLAM_LINE_MODE="${MODE}"

LOG="${OUT}/run.log"

# Save execution command + git commit + actual YAML for reproducibility.
{
  echo "=== command ==="
  echo "PHOTO_SLAM_LINE_MODE=${MODE} \\"
  echo "  ${BIN} ${VOCAB} ${SLAM_CFG} ${GS_CFG} ${DATASET} ${OUT}/ no_viewer"
  echo "=== git ==="
  git -C "${PROJECT_ROOT}" rev-parse HEAD
  git -C "${PROJECT_ROOT}" status --short | head -40
} | tee "${LOG}"

cp "${SLAM_CFG}" "${OUT}/office2.yaml.used"
cp "${GS_CFG}" "${OUT}/replica_mono_diag.yaml.used"

echo "=== run start: ${GROUP}/run${RUN} (MODE=${MODE}) ===" | tee -a "${LOG}"

( cd "${CWD}" && \
  "${BIN}" "${VOCAB}" "${SLAM_CFG}" "${GS_CFG}" "${DATASET}" "${OUT}/" no_viewer \
) 2>&1 | tee -a "${LOG}"

echo "=== run end: exit=${PIPESTATUS[0]} ===" | tee -a "${LOG}"

# Save BA before/after debug exports produced by LocalMapping at KF id 2.
for f in 2_Keyframe_Camera_before.txt 2_Keyframe_Camera_after.txt \
         2_Keyframe_MapPoints_before.obj 2_Keyframe_MapPoints_after.obj \
         2_Keyframe_Maplines_before.obj 2_Keyframe_Maplines_after.obj; do
  if [ -f "${CWD}/${f}" ]; then cp "${CWD}/${f}" "${OUT}/${f}"; fi
done

# Extract coarse metrics that the current code already prints.
{
  echo "=== metrics ==="
  echo "init_frame_ts: $(awk 'NR==1{print $1}' "${OUT}/CameraTrajectory_TUM.txt" 2>/dev/null)"
  echo "trajectory_frames: $(wc -l < "${OUT}/CameraTrajectory_TUM.txt" 2>/dev/null)"
  echo "keyframes: $(wc -l < "${OUT}/KeyFrameTrajectory_TUM.txt" 2>/dev/null)"
  grep -E "New Map created with" "${LOG}" | tail -1
  grep -E "Map .* has .* KFs" "${LOG}" | tail -3
  grep -E "Total_Iterations" "${OUT}/total_mapping_time.txt" 2>/dev/null
} | tee -a "${LOG}" > "${OUT}/metrics.txt"

# ATE (Sim3 / scale-corrected) + per-frame APE.
if [ -f "${OUT}/CameraTrajectory_TUM.txt" ]; then
  python3 "${SCRIPT_DIR}/eval_ate_sim3.py" "${GT}" "${OUT}/CameraTrajectory_TUM.txt" \
      "${OUT}/ate.txt" "${OUT}/ape.txt" 2>&1 | tee -a "${LOG}"
fi

echo "=== done ${GROUP}/run${RUN} ===" | tee -a "${LOG}"
