#!/usr/bin/env bash
# Pre-registered, isolated stability runs for Photo-SLAM-L (line rescue).
#
# WHY THIS EXISTS
#   An earlier ad-hoc loop exported PHOTO_SLAM_EXPORT_DIAG / _POSE_TRACE /
#   PHOTO_SLAM_AUDIT once and never unset them, so later runs appended their rows
#   into an EARLIER run's files and one whole directory (g3) became a mix of three
#   runs. Any conclusion drawn from those files was invalid. This driver exists to
#   make that impossible:
#     * every run gets a fresh directory and an explicit, complete environment;
#     * diagnostic outputs are set for that run only and unset afterwards;
#     * the plan (conditions x rounds x modes) is written to PRE_REGISTRATION.txt
#       BEFORE anything runs, and re-running with a different plan is refused;
#     * EVERY round is appended to TABLE.csv, including failed and crashed ones -
#       no round may be dropped because it failed.
#
# USAGE
#   scripts/stability_runs.sh --tag <name> --modes "mono rgbd" --rounds 3 \
#                             [--outdir /tmp/lr_stab] [--conds "P B C"]
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG=""; MODES="mono rgbd"; ROUNDS=3; OUTDIR="/tmp/lr_stab"; CONDS="P B C"
while [ $# -gt 0 ]; do
    case "$1" in
        --tag) TAG="$2"; shift 2;;
        --modes) MODES="$2"; shift 2;;
        --rounds) ROUNDS="$2"; shift 2;;
        --conds) CONDS="$2"; shift 2;;
        --outdir) OUTDIR="$2"; shift 2;;
        *) echo "unknown arg: $1" >&2; exit 2;;
    esac
done
[ -n "${TAG}" ] || { echo "--tag is required" >&2; exit 2; }

SEQ="${SEQ:-/workspace/code/Photo-SLAM/datasets/eth3d/desk_3}"
VOC="${ROOT}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
CFG_MONO_ORB="${ROOT}/cfg/ORB_SLAM3/Monocular/ETH3D/eth3d_desk_3.yaml"
CFG_MONO_GAU="${ROOT}/cfg/gaussian_mapper/Monocular/ETH3D/eth3d_desk_3.yaml"
CFG_RGBD_ORB="${ROOT}/cfg/ORB_SLAM3/RGB-D/ETH3D/eth3d_desk_3.yaml"
CFG_RGBD_GAU="${ROOT}/cfg/gaussian_mapper/RGB-D/ETH3D/eth3d_desk_3.yaml"

D="${OUTDIR}/${TAG}"
mkdir -p "${D}"
REG="${D}/PRE_REGISTRATION.txt"
PLAN="tag=${TAG}
modes=${MODES}
conditions=${CONDS}
rounds_per_condition=${ROUNDS}
scene=desk_3  sequence=${SEQ}
stress=drop 30% of points from frame 30 (PHOTO_SLAM_STRESS_DROP_POINT_RATIO=0.30)
P = LINE_MODE=1 rescue=0 ; B = LINE_MODE=2 rescue=0 ; C = LINE_MODE=2 rescue=1
PHOTO_SLAM_SLAM_ONLY=1 (SLAM only, no gaussian training)
diagnostics: per-run online trace + export diag (+chain) + audit, always fresh dirs
gate: scripts/export_acceptance.py (15 checks) on every completed round
table: every round, including failures and crashes, appended to TABLE.csv
git=$(cd "${ROOT}" && git rev-parse HEAD)
lib=$(sha256sum "${ROOT}/ORB-SLAM3/lib/libORB_SLAM3.so" | cut -d' ' -f1)
bin_mono=$(sha256sum "${ROOT}/bin/eth3d_mono" | cut -d' ' -f1)
bin_rgbd=$(sha256sum "${ROOT}/bin/eth3d_rgbd" | cut -d' ' -f1)
cfg_mono_orb=$(sha256sum "${CFG_MONO_ORB}" | cut -d' ' -f1)
cfg_rgbd_orb=$(sha256sum "${CFG_RGBD_ORB}" | cut -d' ' -f1)
dataset_rgb=$(sha256sum "${SEQ}/rgb.txt" | cut -d' ' -f1)
dataset_asso=$(sha256sum "${SEQ}/associated.txt" | cut -d' ' -f1)"
if [ -f "${REG}" ]; then
    if [ "$(cat "${REG}")" != "${PLAN}" ]; then
        echo "[stab] WARNING: ${REG} exists with a DIFFERENT plan; refusing to run." >&2
        diff <(echo "${PLAN}") "${REG}" >&2 || true
        exit 3
    fi
else
    echo "${PLAN}" > "${REG}"
fi
cat "${REG}"

TABLE="${D}/TABLE.csv"
if [ ! -f "${TABLE}" ]; then
    echo "tag,mode,cond,round,exit,gate,checks_ok,complete,wall_s,written,input_frames,set_frames,coverage_pct,longest_set_run,n_lost,n_recently_lost,n_relocalized,n_new_map,n_rescue,n_rescue_productive,merges,carries,a6_violations,a6_max_step" > "${TABLE}"
fi

export LD_LIBRARY_PATH="${ROOT}/ORB-SLAM3/lib:${ROOT}/ORB-SLAM3/Thirdparty/g2o/lib:${ROOT}/ORB-SLAM3/Thirdparty/DBoW2/lib:${ROOT}/lib"
export PHOTO_SLAM_SLAM_ONLY=1
export PHOTO_SLAM_DEBUG_MERGE=1 PHOTO_SLAM_DEBUG_LINE_GATE=1 PHOTO_SLAM_DEBUG_LINE_RESCUE=1
export PHOTO_SLAM_DEBUG_LINE_SCAN=1 PHOTO_SLAM_DEBUG_LINE_EDGES=1
export PHOTO_SLAM_STRESS_START_FRAME=30

run_one() {   # mode cond round
    local mode="$1" cond="$2" r="$3"
    local d="${D}/${mode}_${cond}_r${r}"
    rm -rf "${d}"; mkdir -p "${d}"
    # ---- explicit, complete environment for THIS run (no leakage) ----
    unset PHOTO_SLAM_POSE_TRACE PHOTO_SLAM_EXPORT_DIAG PHOTO_SLAM_EXPORT_DIAG_CHAIN
    unset PHOTO_SLAM_AUDIT PHOTO_SLAM_AUDIT_RANGE PHOTO_SLAM_AUDIT_IDS
    unset PHOTO_SLAM_STRESS_DROP_POINT_RATIO
    export PHOTO_SLAM_STRESS_DROP_POINT_RATIO=0.30
    export PHOTO_SLAM_POSE_TRACE="${d}/online_trace.txt"
    export PHOTO_SLAM_EXPORT_DIAG="${d}/export_diag.txt"
    export PHOTO_SLAM_EXPORT_DIAG_CHAIN=1
    export PHOTO_SLAM_AUDIT="${d}/audit.txt"
    export PHOTO_SLAM_AUDIT_RANGE=250:420
    case "${cond}" in
        P) export PHOTO_SLAM_LINE_MODE=1; export PHOTO_SLAM_LINE_RESCUE=0;;
        B) export PHOTO_SLAM_LINE_MODE=2; export PHOTO_SLAM_LINE_RESCUE=0;;
        C) export PHOTO_SLAM_LINE_MODE=2; export PHOTO_SLAM_LINE_RESCUE=1;;
    esac
    local bin cfg_orb cfg_gau asso=""
    if [ "${mode}" = "mono" ]; then
        bin="${ROOT}/bin/eth3d_mono"; cfg_orb="${CFG_MONO_ORB}"; cfg_gau="${CFG_MONO_GAU}"
    else
        bin="${ROOT}/bin/eth3d_rgbd"; cfg_orb="${CFG_RGBD_ORB}"; cfg_gau="${CFG_RGBD_GAU}"
        asso="${SEQ}/associated.txt"
    fi
    {
        echo "scene=desk_3"; echo "mode=${mode}"; echo "cond=${cond}"; echo "round=${r}"; echo "tag=${TAG}"
        echo "line_mode=${PHOTO_SLAM_LINE_MODE}"; echo "rescue=${PHOTO_SLAM_LINE_RESCUE}"
        echo "orb_cfg=${cfg_orb}"; echo "gauss_cfg=${cfg_gau}"; echo "association=${asso}"
        echo "git=$(cd "${ROOT}" && git rev-parse HEAD)"
        echo "lib=$(sha256sum "${ROOT}/ORB-SLAM3/lib/libORB_SLAM3.so" | cut -d' ' -f1)"
    } > "${d}/run.meta"

    local t0 t1
    t0=$(date +%s)
    if [ "${mode}" = "mono" ]; then
        timeout --signal=TERM --kill-after=60 900 "${bin}" "${VOC}" "${cfg_orb}" "${cfg_gau}" "${SEQ}" "${d}" no_viewer > "${d}/log.txt" 2>&1
    else
        timeout --signal=TERM --kill-after=60 900 "${bin}" "${VOC}" "${cfg_orb}" "${cfg_gau}" "${SEQ}" "${asso}" "${d}" no_viewer > "${d}/log.txt" 2>&1
    fi
    local rc=$?
    t1=$(date +%s)
    unset PHOTO_SLAM_POSE_TRACE PHOTO_SLAM_EXPORT_DIAG PHOTO_SLAM_EXPORT_DIAG_CHAIN
    unset PHOTO_SLAM_AUDIT PHOTO_SLAM_AUDIT_RANGE PHOTO_SLAM_AUDIT_IDS
    echo "exit=${rc}" >> "${d}/run.meta"; echo "wall_s=$((t1-t0))" >> "${d}/run.meta"

    python3 "${ROOT}/scripts/export_acceptance.py" "${d}" --seq "${SEQ}" > "${d}/acceptance_run.txt" 2>&1
    local gate=$?
    python3 "${ROOT}/scripts/stability_table.py" --dir "${d}" --seq "${SEQ}" --mode "${mode}" \
        --tag "${TAG}" --cond "${cond}" --round "${r}" --exit-code "${rc}" --gate "${gate}" \
        >> "${TABLE}" 2>> "${d}/table_error.txt" || echo "  [stab] table row failed for ${d}" >&2
    tail -1 "${TABLE}"
}

for r in $(seq 1 "${ROUNDS}"); do
    for mode in ${MODES}; do
        for cond in ${CONDS}; do
            run_one "${mode}" "${cond}" "${r}"
        done
    done
done
echo "[stab] table: ${TABLE}"
