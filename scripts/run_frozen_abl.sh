#!/usr/bin/env bash
# =============================================================================
# Fixed-geometry (frozen) mapper convergence experiment -- runner.
#
# A/B pair: the SAME binary, the SAME frozen scene/initialisation checkpoint and
# the SAME pre-generated training schedule; the ONLY difference is the structural
# (line) loss, switched at runtime:
#     A = structural loss OFF   (existing condition "B": line seeds, no line loss)
#     B = structural loss ON    (existing condition "C": line seeds + line loss)
# Both share the line-sampled seeds, the densification rule and the KF sequence,
# so `input_hash`, `image_hash` and `kf_seq_hash` must be identical across them.
#
# usage: run_frozen_abl.sh <label> <steps> [scene] [rotate]
# env: CONDS (default "B,C"), REPEAT (rotation index), SENSOR (default rgbd)
# Outputs go to a dedicated tree: /workspace/code/PSL-fgm-ablation/abl_out/<label>
# =============================================================================
set -uo pipefail

LABEL="${1:?usage: run_frozen_abl.sh <label> <steps> [scene] [rotate]}"
STEPS="${2:?usage: run_frozen_abl.sh <label> <steps> [scene] [rotate]}"
SCENE="${3:-office0}"
ROTATE="${4:-0}"

W="${W:-/workspace/code/PSL-fgm-ablation}"
CONDS="${CONDS:-B,C}"
SENSOR="${SENSOR:-rgbd}"
PROTOCOL_VERSION="${PROTOCOL_VERSION:-fgm-conv-v2}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/code/SEGS-SLAM/datasets}"
DATASET="${DATASET_ROOT}/replica/${SCENE}"
VOCAB="${W}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
OUT_DIR="${W}/abl_out/${LABEL}"
# Shared frozen directory. In PER_PROCESS mode several processes point at the
# same value so they share one on-disk initialisation checkpoint, while each
# process keeps its own OUT_DIR (and therefore its own scene/run artifacts).
FROZEN_DIR="${FROZEN_DIR:-${OUT_DIR}/frozen}"

if [ "$SENSOR" = "mono" ]; then
    ORB_CFG="${W}/cfg/ORB_SLAM3/Monocular/Replica/${SCENE}.yaml"
    GAU_CFG="${W}/cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml"
    EXE="${W}/bin/replica_mono"
else
    ORB_CFG="${W}/cfg/ORB_SLAM3/RGB-D/Replica/${SCENE}.yaml"
    GAU_CFG="${W}/cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml"
    EXE="${W}/bin/replica_rgbd"
fi

for f in "$EXE" "$VOCAB" "$ORB_CFG" "$GAU_CFG"; do
    [ -e "$f" ] || { echo "MISSING: $f" >&2; exit 3; }
done
if [ -d "${OUT_DIR}" ] && [ -n "$(find "${OUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
    echo "REFUSING to overwrite non-empty ${OUT_DIR}" >&2; exit 4
fi
mkdir -p "${OUT_DIR}"

# SLAM side unchanged from the accepted configuration; the ablation itself is
# driven entirely by the mapper-side switches below.
export PHOTO_SLAM_LINE_MODE=2
export PHOTO_SLAM_LINE_LOOP=0
export PHOTO_SLAM_SHADOW_LINE=0
export PHOTO_SLAM_SLAM_ONLY=0
# --- frozen-geometry ablation switches ---
export PHOTO_SLAM_FROZEN_ABLATION="${FROZEN_DIR}"
export PHOTO_SLAM_FROZEN_STEPS="${STEPS}"
export PHOTO_SLAM_FROZEN_CONDS="${CONDS}"
export PHOTO_SLAM_FROZEN_ROTATE="${ROTATE}"
# Fixed Gaussian count: no densify/clone/split/prune, no opacity reset, no
# inactive-geo densify -> total and line Gaussian counts must stay constant.
export PHOTO_SLAM_FROZEN_FIXED_COUNT="${FIXED_COUNT:-1}"
# Evaluation points in COMPLETED OPTIMIZER STEPS.
export PHOTO_SLAM_FROZEN_EVAL_STEPS="${EVAL_STEPS:-0,100,250,500,1000,2000,3000,5000}"
# Number of fixed, shared evaluation views (evenly spaced keyframes).
export PHOTO_SLAM_FROZEN_EVAL_VIEWS="${EVAL_VIEWS:-24}"
# Training-schedule seed: the SAME value is given to both variants within a run.
export PHOTO_SLAM_FROZEN_SEQ_SEED="${SEQ_SEED:-12345}"

echo "note: any global seed switches must be set identically for both variants"

{
  echo "protocol_version=${PROTOCOL_VERSION}"
  echo "label=${LABEL}"; echo "steps=${STEPS}"; echo "conds=${CONDS}"
  echo "rotate=${ROTATE}"; echo "sensor=${SENSOR}"; echo "scene=${SCENE}"
  echo "frozen_dir=${FROZEN_DIR}"
  echo "fixed_count=${PHOTO_SLAM_FROZEN_FIXED_COUNT}"
  echo "eval_steps=${PHOTO_SLAM_FROZEN_EVAL_STEPS}"
  echo "eval_views=${PHOTO_SLAM_FROZEN_EVAL_VIEWS}"
  echo "seq_seed=${PHOTO_SLAM_FROZEN_SEQ_SEED}"
  echo "worktree=${W}"; echo "git_commit=$(git -C "$W" rev-parse HEAD)"
  echo "branch=$(git -C "$W" rev-parse --abbrev-ref HEAD)"
  echo "dirty=$(git -C "$W" status --porcelain | wc -l)"
  echo "orb_config=${ORB_CFG}"; echo "gaussian_config=${GAU_CFG}"
  echo "exe_sha256=$(sha256sum "$EXE" | cut -d" " -f1)"
  echo "slam_lib_sha256=$(sha256sum "${W}/ORB-SLAM3/lib/libORB_SLAM3.so" | cut -d" " -f1)"
} > "${OUT_DIR}/run.meta"

echo "---- frozen-abl ${LABEL} steps=${STEPS} conds=${CONDS} rot=${ROTATE} scene=${SCENE} ----"
start=$(date +%s.%N)
timeout -s INT "${TIMEOUT_S:-7200}" \
    "${EXE}" "${VOCAB}" "${ORB_CFG}" "${GAU_CFG}" "${DATASET}" "${OUT_DIR}" "no_viewer" \
    > "${OUT_DIR}/run.log" 2>&1
rc=$?
end=$(date +%s.%N)
wall=$(awk -v a="$start" -v b="$end" 'BEGIN{printf "%.3f", b-a}')
echo "EXIT=${rc}" >> "${OUT_DIR}/run.log"
{ echo "exit=${rc}"; echo "wall_s=${wall}"; } > "${OUT_DIR}/run.result"

echo "--- summary ---"
for f in "${FROZEN_DIR}"/summary*.csv; do
    [ -f "$f" ] || continue
    echo "## $f"
    awk -F, 'NR==1{n=NF; for(i=1;i<=n;i++) w[i]=length($i)}
             {for(i=1;i<=n;i++) if(length($i)>w[i]) w[i]=length($i)}
             {for(i=1;i<=n;i++) printf "%-*s%s", w[i], $i, (i<n?"  ":"\n")}' "$f"
done
echo "--- init_fingerprint.txt ---"
[ -f "${FROZEN_DIR}/init_fingerprint.txt" ] && cat "${FROZEN_DIR}/init_fingerprint.txt"
for t in ${CONDS//,/ }; do
    if [ -f "${FROZEN_DIR}/${t}_metrics.csv" ]; then
        echo "--- ${t}_metrics.csv ---"
        cat "${FROZEN_DIR}/${t}_metrics.csv"
    fi
done
echo "---- ${LABEL}: exit=${rc} wall=${wall}s ----"
