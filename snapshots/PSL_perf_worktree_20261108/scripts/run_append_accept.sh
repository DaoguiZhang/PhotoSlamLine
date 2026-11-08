#!/usr/bin/env bash
# ============================================================================
# run_append_accept.sh — P/L append-refinement acceptance (100 real updates)
#
# Runs STRICTLY SERIALLY:
#   1) P_ref : bin/replica_rgbd_point, LINE_MODE=0, raster backend A, office0
#   2) L_ref : bin/replica_rgbd,       LINE_MODE=2, raster backend C, office0
#              structural line-loss weights forced to 0.0
#
# Each run enables PHOTO_SLAM_APPEND_STEPS (default 100) and writes evidence to
#   <out>/append/append_status.txt
#   <out>/append/append_summary.txt
#   <out>/append/append_optimizer_boundary.csv
#   <out>/append/append_step_audit.csv
#   <out>/append/append_param_diff.csv
#   <out>/append/append_views_planned.csv
#   <out>/append/append_views_actual.csv
#
# Exit code: 0 only when BOTH processes exit 0 AND both append_status.txt files
# contain ok=1 AND append_completed_steps equals the requested count. Any
# failure propagates a nonzero exit code; success is never printed on failure.
#
# Usage:
#   scripts/run_append_accept.sh
#   STEPS=100 SEED=12345 scripts/run_append_accept.sh
#   OUTROOT=/path/to/out scripts/run_append_accept.sh
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

STEPS="${STEPS:-100}"
SEED="${SEED:-12345}"
SEQ="${SEQ:-office0}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/code/SEGS-SLAM/datasets}"
DS="${DATASET_ROOT}/replica/${SEQ}"
OUTROOT="${OUTROOT:-${ROOT}/abl_out/append_accept}"

VOCAB="${ROOT}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
ORB_CFG="${ROOT}/cfg/ORB_SLAM3/RGB-D/Replica/${SEQ}.yaml"
P_CFG="${ROOT}/cfg/gaussian_mapper/RGB-D/Replica/${SEQ}.yaml"
L_CFG="${OUTROOT}/config/gaussian_append_L.yaml"

[[ -x "${ROOT}/bin/replica_rgbd_point" ]] || {
    echo "[append-accept] missing bin/replica_rgbd_point (run scripts/build_two_level.sh)" >&2
    exit 2
}
[[ -x "${ROOT}/bin/replica_rgbd" ]] || {
    echo "[append-accept] missing bin/replica_rgbd (run scripts/build_two_level.sh)" >&2
    exit 2
}
[[ -f "${ORB_CFG}" && -f "${P_CFG}" ]] || {
    echo "[append-accept] missing config for ${SEQ}" >&2
    exit 2
}
[[ -d "${DS}/results" ]] || {
    echo "[append-accept] missing dataset ${DS}" >&2
    exit 2
}

mkdir -p "${OUTROOT}/config" "${OUTROOT}/logs"

# L natural training is the verified L_ref condition: same config, structural
# line-loss weights set to zero. (The append phase ALSO hard-disables the
# structural block independently of these weights.)
python3 - "${P_CFG}" "${L_CFG}" <<'PY'
import re, sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
for k in ("weight_line_coherence", "weight_line_shape_ecc", "weight_line_shape_ori"):
    s, n = re.subn(r"(Optimization\." + k + r":\s*)\S+", r"\g<1>0.0", s)
    if n == 0:
        s += "\nOptimization." + k + ": 0.0\n"
open(dst, "w").write(s)
PY

COMMIT="$(git -C "${ROOT}" rev-parse HEAD)"
DIRTY="$(git -C "${ROOT}" status --porcelain | wc -l)"
{
    echo "commit=${COMMIT}"
    echo "dirty_files=${DIRTY}"
    echo "steps=${STEPS}"
    echo "seed=${SEED}"
    echo "sequence=${SEQ}"
    echo "dataset=${DS}"
    echo "orb_cfg=${ORB_CFG} sha256=$(sha256sum "${ORB_CFG}" | cut -d' ' -f1)"
    echo "p_cfg=${P_CFG} sha256=$(sha256sum "${P_CFG}" | cut -d' ' -f1)"
    echo "l_cfg=${L_CFG} sha256=$(sha256sum "${L_CFG}" | cut -d' ' -f1)"
    sha256sum "${ROOT}/bin/replica_rgbd_point" "${ROOT}/bin/replica_rgbd" \
              "${ROOT}/lib/libgaussian_mapper.so" \
              "${ROOT}/ORB-SLAM3/lib/libORB_SLAM3.so"
    echo "started=$(date -Is)"
} > "${OUTROOT}/provenance.txt"

# Clear every optional diagnostic switch so the two runs differ ONLY in the
# documented P/L factors.
append_env() {
    env -u PHOTO_SLAM_GBA_APPLY_CORRECTION \
        -u PHOTO_SLAM_GBA_RECOVERY \
        -u PHOTO_SLAM_GBA_RECOVERY_DIAG \
        -u PHOTO_SLAM_GBA_UPDATE_AUDIT \
        -u PHOTO_SLAM_LINE_LOSS_EVERY \
        -u PHOTO_SLAM_FROZEN_ABLATION \
        -u PHOTO_SLAM_SNAPSHOT_DIR \
        -u PHOTO_SLAM_SNAPSHOT_AT \
        -u PHOTO_SLAM_GPU_PROF \
        "$@"
}

run_case() {
    local tag="$1" bin="$2" mode="$3" backend="$4" cfg="$5"
    local out="${OUTROOT}/${tag}"
    local log="${OUTROOT}/logs/${tag}.log"
    rm -rf "${out}"
    mkdir -p "${out}"

    echo "[append-accept] ${tag}: bin=${bin} line_mode=${mode} backend=${backend} steps=${STEPS}"
    append_env \
        PHOTO_SLAM_RASTER_BACKEND="${backend}" \
        PHOTO_SLAM_LINE_MODE="${mode}" \
        PHOTO_SLAM_APPEND_STEPS="${STEPS}" \
        PHOTO_SLAM_APPEND_SEED="${SEED}" \
        PHOTO_SLAM_APPEND_DIAG=1 \
        "${ROOT}/bin/${bin}" "${VOCAB}" "${ORB_CFG}" "${cfg}" "${DS}" "${out}" no_viewer \
        > "${log}" 2>&1
    local ec=$?
    LAST_PROCESS_EXIT="${ec}"
    echo "${ec}" > "${out}/process_exit_code.txt"

    local status="${out}/append/append_status.txt"
    local status_ok=0 steps_done=-1
    if [[ -f "${status}" ]]; then
        grep -qx 'ok=1' "${status}" && status_ok=1
        steps_done="$(grep '^append_completed_steps=' "${status}" | head -1 | cut -d= -f2)"
    fi
    local steps_ok=0
    [[ "${steps_done}" == "${STEPS}" ]] && steps_ok=1
    local guard_ok=0
    [[ -f "${out}/eval_guard_shutdown.txt" ]] && grep -qx 'ok=1' "${out}/eval_guard_shutdown.txt" && guard_ok=1

    if [[ "${ec}" -eq 0 && "${status_ok}" -eq 1 && "${steps_ok}" -eq 1 && "${guard_ok}" -eq 1 ]]; then
        echo "[append-accept] ${tag}: PASS (exit=${ec}, append_steps=${steps_done}, eval_guard=1)"
        return 0
    fi
    echo "[append-accept] ${tag}: FAIL (exit=${ec}, status_ok=${status_ok}, append_steps=${steps_done}, eval_guard=${guard_ok})" >&2
    [[ -f "${status}" ]] && sed -n '1,40p' "${status}" | sed 's/^/    /' >&2
    echo "    log: ${log}" >&2
    return 1
}

P_RC=0
L_RC=0
LAST_PROCESS_EXIT=0
run_case "P_ref" "replica_rgbd_point" "0" "A" "${P_CFG}" || P_RC=$?
# P must pass before L is attempted: never continue the chain with a broken P.
if [[ "${P_RC}" -ne 0 ]]; then
    {
        echo "append_acceptance"
        echo "p_ref_exit=${LAST_PROCESS_EXIT}"
        echo "l_ref_exit=not_run_p_failed"
        echo "p_ref_status=$(grep -s '^ok=' "${OUTROOT}/P_ref/append/append_status.txt" || echo 'ok=missing')"
        echo "finished=$(date -Is)"
    } > "${OUTROOT}/summary.txt"
    echo "[append-accept] RESULT: FAIL at P; L NOT run (see ${OUTROOT}/summary.txt)" >&2
    # Propagate the REAL process exit code when there is one, else generic 1.
    [[ "${LAST_PROCESS_EXIT}" -ne 0 ]] && exit "${LAST_PROCESS_EXIT}"
    exit 1
fi
# Strictly serial: L starts only after P has passed.
run_case "L_ref" "replica_rgbd" "2" "C" "${L_CFG}" || L_RC=$?
if [[ "${L_RC}" -ne 0 ]]; then
    {
        echo "append_acceptance"
        echo "p_ref_exit=${P_RC}"
        echo "l_ref_exit=${LAST_PROCESS_EXIT}"
        echo "p_ref_status=$(grep -s '^ok=' "${OUTROOT}/P_ref/append/append_status.txt" || echo 'ok=missing')"
        echo "l_ref_status=$(grep -s '^ok=' "${OUTROOT}/L_ref/append/append_status.txt" || echo 'ok=missing')"
        echo "finished=$(date -Is)"
    } > "${OUTROOT}/summary.txt"
    echo "[append-accept] RESULT: FAIL at L (see ${OUTROOT}/summary.txt)" >&2
    [[ "${LAST_PROCESS_EXIT}" -ne 0 ]] && exit "${LAST_PROCESS_EXIT}"
    exit 1
fi

{
    echo "append_acceptance"
    echo "p_ref_exit=${P_RC}"
    echo "l_ref_exit=${L_RC}"
    echo "p_ref_status=$(grep -s '^ok=' "${OUTROOT}/P_ref/append/append_status.txt" || echo 'ok=missing')"
    echo "l_ref_status=$(grep -s '^ok=' "${OUTROOT}/L_ref/append/append_status.txt" || echo 'ok=missing')"
    echo "p_ref_append_steps=$(grep -s '^append_completed_steps=' "${OUTROOT}/P_ref/append/append_status.txt" | cut -d= -f2 || true)"
    echo "l_ref_append_steps=$(grep -s '^append_completed_steps=' "${OUTROOT}/L_ref/append/append_status.txt" | cut -d= -f2 || true)"
    echo "finished=$(date -Is)"
} > "${OUTROOT}/summary.txt"

if [[ "${P_RC}" -ne 0 || "${L_RC}" -ne 0 ]]; then
    echo "[append-accept] RESULT: FAIL (P=${P_RC}, L=${L_RC}); see ${OUTROOT}/summary.txt" >&2
    exit 1
fi

echo "[append-accept] RESULT: PASS (P and L each completed ${STEPS} real optimizer updates)"
echo "[append-accept] evidence: ${OUTROOT}/P_ref/append, ${OUTROOT}/L_ref/append"
exit 0
