#!/usr/bin/env bash
# ============================================================================
# SMALL-BATCH validation: 2 interleaved rounds per planned condition.
#
#   scene desk_3 x mode {rgbd, mono} x stress {30, 50} x cond {P, B, C} x 2 rounds
#
# Rules enforced here (no exceptions):
#   * unique output directory per round: <base>/<tag>/r<r>/<mode>_L<level>_<cond>
#   * the 15-check export acceptance gate runs IMMEDIATELY after every round
#   * FAIL FAST: a non-zero exit, an incomplete round or a failed gate stops the
#     whole batch; log/run.meta/online trace/export diag are kept for the failure
#   * the freeze manifest (commit, configs, binaries, dataset hashes, env) is
#     written before the first round, so the basis is fixed for the whole batch
#
# usage: smallbatch_2x.sh <tag> <base_dir> [rounds]
# ============================================================================
set -uo pipefail

TAG="${1:?usage: smallbatch_2x.sh <tag> <base_dir> [rounds]}"
BASE="${2:?missing base_dir}"
ROUNDS="${3:-2}"

ROOT=/workspace/code/Photo-SLAM-L
SEQ=/workspace/code/Photo-SLAM/datasets/eth3d/desk_3
VOC="${ROOT}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
GATE="${ROOT}/scripts/export_acceptance.py"
STATS="${ROOT}/scripts/round_stats.py"

CFG_RGBD_ORB="${ROOT}/cfg/ORB_SLAM3/RGB-D/ETH3D/eth3d_desk_3.yaml"
CFG_RGBD_GAU="${ROOT}/cfg/gaussian_mapper/RGB-D/ETH3D/eth3d_desk_3.yaml"
CFG_MONO_ORB="${ROOT}/cfg/ORB_SLAM3/Monocular/ETH3D/eth3d_desk_3.yaml"
CFG_MONO_GAU="${ROOT}/cfg/gaussian_mapper/Monocular/ETH3D/eth3d_desk_3.yaml"

export RUN_TIMEOUT=600
export PHOTO_SLAM_SLAM_ONLY=1
export PHOTO_SLAM_DEBUG_MERGE=1
export PHOTO_SLAM_DEBUG_LINE_GATE=1
export PHOTO_SLAM_DEBUG_LINE_RESCUE=1
export PHOTO_SLAM_DEBUG_LINE_SCAN=1
export PHOTO_SLAM_DEBUG_LINE_EDGES=1
export LD_LIBRARY_PATH="${ROOT}/ORB-SLAM3/lib:${ROOT}/ORB-SLAM3/Thirdparty/g2o/lib:${ROOT}/ORB-SLAM3/Thirdparty/DBoW2/lib:${ROOT}/lib"

for f in "${VOC}" "${GATE}" "${STATS}" "${CFG_RGBD_ORB}" "${CFG_RGBD_GAU}" \
         "${CFG_MONO_ORB}" "${CFG_MONO_GAU}" "${SEQ}/rgb.txt" "${SEQ}/associated.txt" \
         "${SEQ}/groundtruth.txt"; do
    [ -e "${f}" ] || { echo "MISSING: ${f}" >&2; exit 3; }
done

if [ -d "${BASE}/${TAG}" ] && [ -n "$(ls -A "${BASE}/${TAG}" 2>/dev/null)" ]; then
    echo "REFUSING to reuse non-empty batch dir: ${BASE}/${TAG}" >&2
    exit 2
fi
OUTDIR="${BASE}/${TAG}"
mkdir -p "${OUTDIR}"

# ---- freeze manifest -------------------------------------------------------
FREEZE="${OUTDIR}/FREEZE.txt"
{
    echo "tag=${TAG}  rounds=${ROUNDS}  start=$(date -Is)"
    echo "--- git ---"
    git -C "${ROOT}" rev-parse HEAD 2>/dev/null
    echo "dirty files: $(git -C "${ROOT}" status --porcelain 2>/dev/null | wc -l)"
    echo "--- binaries (must not change during the batch) ---"
    sha256sum "${ROOT}/bin/eth3d_rgbd" "${ROOT}/bin/eth3d_mono" \
              "${ROOT}/ORB-SLAM3/lib/libORB_SLAM3.so" 2>/dev/null
    echo "--- configs ---"
    sha256sum "${CFG_RGBD_ORB}" "${CFG_RGBD_GAU}" "${CFG_MONO_ORB}" "${CFG_MONO_GAU}" \
              "${VOC}" 2>/dev/null
    echo "--- dataset ---"
    sha256sum "${SEQ}/rgb.txt" "${SEQ}/associated.txt" "${SEQ}/groundtruth.txt" 2>/dev/null
    echo "dataset_frames=$(grep -vc '^#' "${SEQ}/rgb.txt")"
    echo "--- environment ---"
    echo "PHOTO_SLAM_SLAM_ONLY=${PHOTO_SLAM_SLAM_ONLY} RUN_TIMEOUT=${RUN_TIMEOUT}"
    echo "PHOTO_SLAM_DEBUG_MERGE=${PHOTO_SLAM_DEBUG_MERGE}"
    echo "PHOTO_SLAM_DEBUG_LINE_GATE=${PHOTO_SLAM_DEBUG_LINE_GATE}"
    echo "PHOTO_SLAM_DEBUG_LINE_RESCUE=${PHOTO_SLAM_DEBUG_LINE_RESCUE}"
    echo "PHOTO_SLAM_DEBUG_LINE_SCAN=${PHOTO_SLAM_DEBUG_LINE_SCAN}"
    echo "PHOTO_SLAM_DEBUG_LINE_EDGES=${PHOTO_SLAM_DEBUG_LINE_EDGES}"
    echo "PHOTO_SLAM_STRESS_DROP_POINT_RATIO/START_FRAME set per round (0.30/0.50, start 30)"
    echo "PHOTO_SLAM_LINE_MODE/RESCUE set per condition (P=1/0, B=2/0, C=2/1)"
    echo "PHOTO_SLAM_POSE_TRACE and PHOTO_SLAM_EXPORT_DIAG = per-round files (default off)"
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
} > "${FREEZE}"
cat "${FREEZE}"

TABLE="${OUTDIR}/ROUND_TABLE.csv"
echo "round,mode,level,cond,exit,gate,complete,wall_s,input_frames,online_pose_frames,exported_frames,maps,excluded_frames,max_adj_online_step,max_online_to_export,max_refkf_change" > "${TABLE}"
FAILS="${OUTDIR}/FAILURES.txt"
: > "${FAILS}"

for r in $(seq 1 "${ROUNDS}"); do
    for mode in rgbd mono; do
        if [ "${mode}" = "rgbd" ]; then
            BIN="${ROOT}/bin/eth3d_rgbd"
            ORB_CFG="${CFG_RGBD_ORB}"; GAUSS_CFG="${CFG_RGBD_GAU}"; ASSO="${SEQ}/associated.txt"
        else
            BIN="${ROOT}/bin/eth3d_mono"
            ORB_CFG="${CFG_MONO_ORB}"; GAUSS_CFG="${CFG_MONO_GAU}"; ASSO=""
        fi
        for level in 30 50; do
            export PHOTO_SLAM_STRESS_DROP_POINT_RATIO="0.${level}"
            export PHOTO_SLAM_STRESS_START_FRAME=30
            for cond in P B C; do
                case "${cond}" in
                    P) export PHOTO_SLAM_LINE_MODE=1; export PHOTO_SLAM_LINE_RESCUE=0 ;;
                    B) export PHOTO_SLAM_LINE_MODE=2; export PHOTO_SLAM_LINE_RESCUE=0 ;;
                    C) export PHOTO_SLAM_LINE_MODE=2; export PHOTO_SLAM_LINE_RESCUE=1 ;;
                esac

                d="${OUTDIR}/r${r}/${mode}_L${level}_${cond}"
                mkdir -p "${d}"
                export PHOTO_SLAM_POSE_TRACE="${d}/online_trace.txt"
                export PHOTO_SLAM_EXPORT_DIAG="${d}/export_diag.txt"

                {
                    echo "scene=desk_3"; echo "mode=${mode}"; echo "cond=${cond}"
                    echo "run_idx=${r}"; echo "level=${level}"; echo "tag=${TAG}"
                    echo "line_mode=${PHOTO_SLAM_LINE_MODE}"; echo "rescue=${PHOTO_SLAM_LINE_RESCUE}"
                    echo "orb_cfg=${ORB_CFG}"; echo "gauss_cfg=${GAUSS_CFG}"
                    echo "sequence=${SEQ}"; echo "association=${ASSO}"
                    echo "stress_drop_point_ratio=${PHOTO_SLAM_STRESS_DROP_POINT_RATIO}"
                    echo "stress_start_frame=${PHOTO_SLAM_STRESS_START_FRAME}"
                    echo "freeze=$(sha256sum "${FREEZE}" | cut -c1-16)"
                } > "${d}/run.meta"

                START=$(date +%s)
                if [ "${mode}" = "rgbd" ]; then
                    timeout --signal=TERM --kill-after=60 "${RUN_TIMEOUT}" \
                        "${BIN}" "${VOC}" "${ORB_CFG}" "${GAUSS_CFG}" "${SEQ}" "${ASSO}" "${d}" no_viewer \
                        > "${d}/log.txt" 2>&1
                else
                    timeout --signal=TERM --kill-after=60 "${RUN_TIMEOUT}" \
                        "${BIN}" "${VOC}" "${ORB_CFG}" "${GAUSS_CFG}" "${SEQ}" "${d}" no_viewer \
                        > "${d}/log.txt" 2>&1
                fi
                RC=$?
                END=$(date +%s)
                SAVED=$(grep -c "Saving keyframe trajectory" "${d}/log.txt" 2>/dev/null)
                SAVED=${SAVED:-0}
                COMPLETE=0
                [ "${RC}" = "0" ] && [ "${SAVED}" -ge 1 ] && COMPLETE=1
                { echo "exit=${RC}"; echo "wall_s=$((END-START))"; echo "complete=${COMPLETE}"; } >> "${d}/run.meta"

                GATE_RC=999
                if [ "${COMPLETE}" = "1" ]; then
                    python3 "${GATE}" "${d}" --seq "${SEQ}" > "${d}/acceptance_run.txt" 2>&1
                    GATE_RC=$?
                fi
                cat "${d}/acceptance_run.txt" 2>/dev/null | tail -3

                # extra per-round quantities (also prints the jump window)
                python3 "${STATS}" --trace "${d}/online_trace.txt" --diag "${d}/export_diag.txt" \
                    --tum "${d}/CameraTrajectory_TUM.txt" --seq "${SEQ}" --log "${d}/log.txt" \
                    --out "${d}/round_stats.json" > "${d}/round_stats.txt" 2>&1
                head -1 "${d}/round_stats.txt"
                sed -n '2,20p' "${d}/round_stats.txt"

                python3 - "${d}/round_stats.json" "${r}" "${mode}" "${level}" "${cond}" \
                        "${RC}" "${GATE_RC}" "${COMPLETE}" "$((END-START))" <<'PY' >> "${TABLE}" 2>/dev/null || true
import json, sys
p, r, mode, lvl, cond, rc, grc, comp, wall = sys.argv[1:10]
try:
    d = json.load(open(p))
except Exception:
    d = {}
print(",".join(str(x) for x in [r, mode, lvl, cond, rc, grc, comp, wall,
      d.get("input_frames"), d.get("online_pose_frames"), d.get("exported_frames"),
      d.get("maps"), d.get("excluded_frames"), d.get("max_adjacent_online_step"),
      d.get("max_online_to_export"), d.get("max_refKF_pose_change")]))
PY

                if [ "${COMPLETE}" != "1" ] || [ "${GATE_RC}" != "0" ]; then
                    echo "FAIL r=${r} ${mode} L${level} ${cond}: exit=${RC} complete=${COMPLETE} gate_rc=${GATE_RC}" | tee -a "${FAILS}"
                    echo "BATCH=FAIL (stopped at the first failure; artefacts kept in ${d})"
                    exit 1
                fi
                echo "ok r=${r} ${mode} L${level} ${cond}"
            done
        done
    done
done

echo "BATCH=PASS tag=${TAG} rounds=${ROUNDS} table=${TABLE}"
echo "end=$(date -Is)"
