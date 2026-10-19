#!/usr/bin/env bash
# ============================================================================
# quad_office0.sh — Photo-SLAM / Photo-SLAM-L four-condition online comparison
#
# Conditions (exactly one factor per pair):
#   P_ref : point-only build   + raster backend A
#   P_fast: point-only build   + raster backend C
#   L_ref : Photo-SLAM-L build + raster backend A
#   L_fast: Photo-SLAM-L build + raster backend C
#
# "point-only build" = the SAME repository compiled with USE_LINE_GAUSSIAN=0:
#   * entry calls System::TrackRGBD (no LSD line extraction in the frame)
#   * PointLocalMapping / PointLoopClosing via PHOTO_SLAM_LINE_MODE=0
#   * GaussianMapper (no line seed / line densify / line Gaussian)
# It is NOT byte-identical to upstream Photo-SLAM; see
# reports/quad_office0_report.md for the identity evidence and the differences.
#
# L keeps the line front end, line map seeding/densify and line-aware mapping.
# Structural line losses are disabled through their ACTUAL weights
# (Optimization.weight_line_coherence/shape_ecc/shape_ori = 0.0), never by
# LINE_MODE (which would also change the front end).
# GBA Gaussian deformation stays at its default (OFF) and is never enabled.
#
# Both conditions share ONE Gaussian config file; only the image recording of
# ground truth is turned on (needed for offline LPIPS; it does not train).
#
# Usage:
#   scripts/quad_office0.sh                     # 3 rounds, full office0
#   ROUNDS=1 scripts/quad_office0.sh
#   FRAMES=300 scripts/quad_office0.sh          # smoke on a fixed subset
# Order is pre-planned and interleaved; runs are strictly serial.
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SEQ="${SEQ:-office0}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/code/SEGS-SLAM/datasets}"
DS="${DATASET_ROOT}/replica/${SEQ}"
FRAMES="${FRAMES:-0}"            # 0 = full sequence
ROUNDS="${ROUNDS:-3}"
OUTROOT="${OUTROOT:-${ROOT}/results/quad_${SEQ}}"
VOCAB="${ROOT}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
ORB_CFG="${ROOT}/cfg/ORB_SLAM3/RGB-D/Replica/${SEQ}.yaml"
CFG_SRC="${ROOT}/cfg/gaussian_mapper/RGB-D/Replica/${SEQ}.yaml"
GT="${DS}/pose_TUM.txt"

[[ -d "${DS}/results" ]] || { echo "missing dataset ${DS}" >&2; exit 1; }
[[ -x "${ROOT}/bin/replica_rgbd" && -x "${ROOT}/bin/replica_rgbd_point" ]] || {
    echo "build both targets first (scripts/build_two_level.sh)" >&2; exit 1; }

mkdir -p "${OUTROOT}/config" "${OUTROOT}/logs" "${OUTROOT}/runs"
CFG="${OUTROOT}/config/gaussian_quad.yaml"
python3 - "${CFG_SRC}" "${CFG}" <<'PY'
import re, sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
for k in ("weight_line_coherence", "weight_line_shape_ecc", "weight_line_shape_ori"):
    s, n = re.subn(r"(Optimization\." + k + r":\s*)\S+", r"\g<1>0.0", s)
    if n == 0:
        s += "\nOptimization." + k + ": 0.0\n"
s, n = re.subn(r"(Record\.record_ground_truth_image:\s*)\S+", r"\g<1>1", s)
if n == 0:
    s += "\nRecord.record_ground_truth_image: 1\n"
open(dst, "w").write(s)
print("[quad] wrote " + dst + " (line loss weights = 0.0, GT recording on)")
PY

# ---- fixed input view (identical bytes for every condition) ----------------
INPUT_DIR="${DS}"
if [[ "${FRAMES}" != "0" ]]; then
    VIEW="${OUTROOT}/input_${FRAMES}f/results"
    rm -rf "${OUTROOT}/input_${FRAMES}f"; mkdir -p "${VIEW}"
    ls "${DS}/results"/frame*.jpg 2>/dev/null | sort | head -"${FRAMES}" | while read -r f; do
        ln -sf "${f}" "${VIEW}/$(basename "${f}")"; done
    ls "${DS}/results"/depth*.png 2>/dev/null | sort | head -"${FRAMES}" | while read -r f; do
        ln -sf "${f}" "${VIEW}/$(basename "${f}")"; done
    INPUT_DIR="${OUTROOT}/input_${FRAMES}f"
fi
N_INPUT=$(ls "${INPUT_DIR}/results"/frame*.jpg 2>/dev/null | wc -l)

# ---- provenance (recorded under results/, not committed) ------------------
COMMIT="$(git -C "${ROOT}" rev-parse HEAD)"
{
    echo "commit=${COMMIT}"
    echo "dirty_files=$(git -C "${ROOT}" status --porcelain | wc -l)"
    echo "sequence=${SEQ}"
    echo "input_dir=${INPUT_DIR}"
    echo "input_frames=${N_INPUT}"
    echo "orb_cfg=${ORB_CFG} sha256=$(sha256sum "${ORB_CFG}" | cut -d' ' -f1)"
    echo "gaussian_cfg=${CFG} sha256=$(sha256sum "${CFG}" | cut -d' ' -f1)"
    sha256sum "${ROOT}/bin/replica_rgbd" "${ROOT}/bin/replica_rgbd_point" \
              "${ROOT}/ORB-SLAM3/lib/libORB_SLAM3.so" \
              "${ROOT}/lib/libgaussian_mapper.so"
    grep -E "^(CMAKE_BUILD_TYPE|CMAKE_CXX_FLAGS|CMAKE_CUDA_ARCHITECTURES|CMAKE_CUDA_FLAGS):" \
        "${ROOT}/build/CMakeCache.txt" 2>/dev/null
} > "${OUTROOT}/provenance.txt"

orders=("P_ref P_fast L_ref L_fast" "L_fast L_ref P_fast P_ref" "P_fast P_ref L_fast L_ref")

peak_vram() {  # $1 pid -> max MiB seen for that pid (0 if not sampled)
    local pid="$1" max=0
    while kill -0 "${pid}" 2>/dev/null; do
        local v
        v=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
            | awk -F, -v p="${pid}" '$1+0==p+0 {gsub(/ /,"",$2); print $2}' | head -1)
        if [[ -n "${v:-}" ]] && [[ "${v}" -gt "${max}" ]]; then max="${v}"; fi
        sleep 0.2
    done
    echo "${max}"
}

run_one() {  # cond round
    local cond="$1" round="$2"
    local bin mode backend
    case "${cond}" in
        P_ref)  bin=replica_rgbd_point; mode=0; backend=A ;;
        P_fast) bin=replica_rgbd_point; mode=0; backend=C ;;
        L_ref)  bin=replica_rgbd;       mode=2; backend=A ;;
        L_fast) bin=replica_rgbd;       mode=2; backend=C ;;
        *) echo "unknown condition ${cond}" >&2; return 2 ;;
    esac
    local out="${OUTROOT}/runs/${cond}_r${round}"
    local log="${OUTROOT}/logs/${cond}_r${round}.log"
    rm -rf "${out}"; mkdir -p "${out}"

    # Exactly one factor changes per pair; nothing else is exported.
    unset PHOTO_SLAM_GBA_APPLY_CORRECTION PHOTO_SLAM_GBA_RECOVERY PHOTO_SLAM_GBA_RECOVERY_DIAG \
          PHOTO_SLAM_GBA_UPDATE_AUDIT PHOTO_SLAM_LINE_LOSS_EVERY PHOTO_SLAM_LINE_LOOP \
          PHOTO_SLAM_SNAPSHOT_DIR PHOTO_SLAM_SNAPSHOT_AT PHOTO_SLAM_RASTER_PROF \
          PHOTO_SLAM_GPU_PROF PHOTO_SLAM_CREATION_AUDIT PHOTO_SLAM_LINE_POP_AUDIT
    export PHOTO_SLAM_RASTER_BACKEND="${backend}"
    export PHOTO_SLAM_LINE_MODE="${mode}"

    local t0 t1 wall ec peak peak_file="${out}/peak_vram.txt"
    t0=$(date +%s%N)
    "${ROOT}/bin/${bin}" "${VOCAB}" "${ORB_CFG}" "${CFG}" "${INPUT_DIR}" "${out}" no_viewer \
        > "${log}" 2>&1 &
    local pid=$!
    ( peak_vram "${pid}" > "${peak_file}" ) &
    local vram_job=$!
    wait "${pid}"; ec=$?
    wait "${vram_job}" 2>/dev/null
    peak=$(cat "${peak_file}" 2>/dev/null || echo 0)
    t1=$(date +%s%N)
    wall=$(python3 -c "print(f'{(${t1}-${t0})/1e9:.3f}')")

    {
        echo "condition=${cond}"
        echo "round=${round}"
        echo "binary=${bin}"
        echo "line_mode=${mode}"
        echo "raster_backend=${backend}"
        echo "exit_code=${ec}"
        echo "wall_seconds=${wall}"
        echo "peak_vram_mib=${peak:-0}"
        echo "gaussian_cfg=${CFG}"
    } > "${out}/run_meta.txt"
    echo "[quad] round=${round} ${cond} exit=${ec} wall=${wall}s peakVRAM=${peak:-0}MiB"
}

for ((r = 1; r <= ROUNDS; r++)); do
    read -r -a order <<< "${orders[$(((r - 1) % 3))]}"
    echo "[quad] round ${r} order: ${order[*]}"
    for cond in "${order[@]}"; do
        run_one "${cond}" "${r}"
    done
done

echo "[quad] all runs finished; aggregating"
python3 "${SCRIPT_DIR}/quad_office0_summary.py" "${OUTROOT}" "${GT}" "${N_INPUT}"
