#!/usr/bin/env bash
# ============================================================================
# run_three_budget_round.sh — one diagnostic round, 120 s cumulative training
#
# Serial only (never concurrent compile/train/LPIPS):
#   P_ref : point entry, LINE_MODE=0, backend A
#   P_fast: point entry, LINE_MODE=0, backend C
#   L_fast: line entry,   LINE_MODE=2, backend C, structural weights 0.0
#
# Budget mode:
#   PHOTO_SLAM_TRAIN_TIME_BUDGET=120
#   PHOTO_SLAM_APPEND_STEPS=0   (budget is primary; no step cap)
#
# The budget counts TRAINING CALLS only (natural online + original final
# refinement + append), with a trailing CUDA synchronize. Queue draining,
# setup, evaluation, boundary checks, model save and export are outside it.
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SEQ="${SEQ:-office0}"
BUDGET="${BUDGET:-120}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/code/SEGS-SLAM/datasets}"
DS="${DATASET_ROOT}/replica/${SEQ}"
OUTROOT="${OUTROOT:-${ROOT}/results/append_budget120}"
RUNS="${RUNS:-P_ref,P_fast,L_fast}"

VOCAB="${ROOT}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
ORB_CFG="${ROOT}/cfg/ORB_SLAM3/RGB-D/Replica/${SEQ}.yaml"
P_CFG="${ROOT}/cfg/gaussian_mapper/RGB-D/Replica/${SEQ}.yaml"
L_CFG="${OUTROOT}/config/gaussian_L_fast.yaml"

for b in bin/replica_rgbd_point bin/replica_rgbd; do
    [[ -x "${ROOT}/${b}" ]] || {
        echo "[budget-round] missing ${b}; run scripts/build_two_level.sh" >&2
        exit 2
    }
done
[[ -d "${DS}/results" ]] || { echo "[budget-round] missing ${DS}" >&2; exit 2; }

mkdir -p "${OUTROOT}/runs" "${OUTROOT}/logs" "${OUTROOT}/config"
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

{
    echo "commit=$(git -C "${ROOT}" rev-parse HEAD)"
    echo "branch=$(git -C "${ROOT}" rev-parse --abbrev-ref HEAD)"
    echo "dirty_files=$(git -C "${ROOT}" status --porcelain | wc -l)"
    echo "budget_seconds=${BUDGET}"
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

clean_env() {
    env -u PHOTO_SLAM_APPEND_STEPS \
        -u PHOTO_SLAM_APPEND_SEED \
        -u PHOTO_SLAM_APPEND_DIAG \
        -u PHOTO_SLAM_GBA_APPLY_CORRECTION \
        -u PHOTO_SLAM_GBA_RECOVERY \
        -u PHOTO_SLAM_GBA_RECOVERY_DIAG \
        -u PHOTO_SLAM_GBA_UPDATE_AUDIT \
        -u PHOTO_SLAM_LINE_LOSS_EVERY \
        -u PHOTO_SLAM_FROZEN_ABLATION \
        -u PHOTO_SLAM_SNAPSHOT_DIR \
        -u PHOTO_SLAM_SNAPSHOT_AT \
        -u PHOTO_SLAM_GPU_PROF \
        -u PHOTO_SLAM_RASTER_PROF \
        -u PHOTO_SLAM_CREATION_AUDIT \
        -u PHOTO_SLAM_LINE_POP_AUDIT \
        "$@"
}

run_case() {
    local tag="$1" bin="$2" mode="$3" backend="$4" cfg="$5"
    local out="${OUTROOT}/runs/${tag}"
    local log="${OUTROOT}/logs/${tag}.log"
    rm -rf "${out}"; mkdir -p "${out}"
    echo "[budget-round] ${tag}: bin=${bin} line_mode=${mode} backend=${backend} budget=${BUDGET}s"
    local t0 t1 wall ec
    t0=$(date +%s%N)
    clean_env \
        PHOTO_SLAM_RASTER_BACKEND="${backend}" \
        PHOTO_SLAM_LINE_MODE="${mode}" \
        PHOTO_SLAM_TRAIN_TIME_BUDGET="${BUDGET}" \
        "${ROOT}/bin/${bin}" "${VOCAB}" "${ORB_CFG}" "${cfg}" "${DS}" "${out}" no_viewer \
        > "${log}" 2>&1
    ec=$?
    t1=$(date +%s%N)
    wall=$(python3 -c "print(f'{(${t1}-${t0})/1e9:.3f}')")
    {
        echo "tag=${tag}"
        echo "binary=${bin}"
        echo "line_mode=${mode}"
        echo "raster_backend=${backend}"
        echo "budget_seconds=${BUDGET}"
        echo "exit_code=${ec}"
        echo "wall_seconds=${wall}"
        echo "gaussian_cfg=${cfg}"
    } > "${out}/run_meta.txt"

    local status="${out}/append/append_status.txt"
    local ok=0 bstat="" bmis=""
    if [[ -f "${status}" ]]; then
        grep -qx 'ok=1' "${status}" && ok=1
        bstat="$(grep '^budget_status=' "${status}" | head -1 | cut -d= -f2)"
        bmis="$(grep '^budget_unmatched=' "${status}" | head -1 | cut -d= -f2)"
    fi
    local plys=0
    [[ -n "$(find "${out}/append/natural_end/point_cloud" -name point_cloud.ply 2>/dev/null | head -1)" ]] && plys=$((plys+1))
    [[ -n "$(find "${out}/append/budget_end/point_cloud" -name point_cloud.ply 2>/dev/null | head -1)" ]] && plys=$((plys+1))
    local guard_ok=0
    [[ -f "${out}/eval_guard_shutdown.txt" ]] && grep -qx 'ok=1' "${out}/eval_guard_shutdown.txt" && guard_ok=1

    if [[ "${ec}" -eq 0 && "${ok}" -eq 1 && "${plys}" -eq 2 && "${guard_ok}" -eq 1 && "${bstat}" != "pending" && "${bstat}" != "" ]]; then
        echo "[budget-round] ${tag}: PASS exit=${ec} wall=${wall}s budget_status=${bstat} unmatched=${bmis:-?} eval_guard=1"
        return 0
    fi
    echo "[budget-round] ${tag}: FAIL exit=${ec} status_ok=${ok} plys=${plys} eval_guard=${guard_ok} budget_status=${bstat}" >&2
    [[ -f "${status}" ]] && sed -n '1,60p' "${status}" | sed 's/^/    /' >&2
    echo "    log: ${log}" >&2
    if [[ "${ec}" -ne 0 ]]; then
        exit "${ec}"
    fi
    exit 1
}

for tag in ${RUNS//,/ }; do
    case "${tag}" in
        P_ref)  run_case P_ref  replica_rgbd_point 0 A "${P_CFG}" ;;
        P_fast) run_case P_fast replica_rgbd_point 0 C "${P_CFG}" ;;
        L_fast) run_case L_fast replica_rgbd       2 C "${L_CFG}" ;;
        *) echo "[budget-round] unknown run ${tag}" >&2; exit 2 ;;
    esac
done

# Offline evaluation, strictly after ALL training. Renders then metrics + LPIPS
# run sequentially inside the script.
python3 "${SCRIPT_DIR}/eval_common_views_budget.py" \
    "${OUTROOT}" "${DS}/pose_TUM.txt" "${ORB_CFG}" "${L_CFG}" "${DS}" \
    --lpips --runs "${RUNS}" || exit $?

python3 "${SCRIPT_DIR}/audit_append_protocol.py" \
    "${OUTROOT}" "${ORB_CFG}" --runs "${RUNS}" || exit $?

python3 "${SCRIPT_DIR}/summarize_budget_round.py" "${OUTROOT}" || exit $?

echo "[budget-round] DONE -> ${OUTROOT}/summary.md"
