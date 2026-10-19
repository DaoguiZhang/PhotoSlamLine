#!/usr/bin/env bash
# Tanks and Temples (tandt_db) monocular runner for Photo-SLAM-L.
#
# Each scene must contain rgb/ plus rgb.txt (timestamp rgb/<file>). Scenes that
# ship only rgb/ get their rgb.txt auto-generated (index/FPS, see readme.txt).
#
# Scenes: train family francis temple truck   (all 1920x1080; one shared config)
#
# Examples (from anywhere):
#   SEQS="train" bash scripts/tandt_mono.sh
#   RUNS=1 SEQS="train family" RESULT_SET=tandt_mono PHOTO_SLAM_LINE_MODE=2 bash scripts/tandt_mono.sh
#   PHOTO_SLAM_NO_PACING=1 SEQS="train" bash scripts/tandt_mono.sh      # offline throughput
#   SEQS="train" PHOTO_SLAM_SLAM_ONLY=1 bash scripts/tandt_mono.sh      # SLAM-only (no 3DGS)
#   SEQS="train" TANDT_FULL=1 bash scripts/tandt_mono.sh                # use rgb_2000.txt (all frames)
#   SEQS="train" PHOTO_SLAM_IMAGE_LIST=rgb_2000.txt bash scripts/tandt_mono.sh   # explicit list
#
# Switches (same semantics as the other datasets):
#   PHOTO_SLAM_LINE_MODE 0=point-only 1=line-frontend 2=full point-line (default 2)
#   PHOTO_SLAM_NO_PACING 1=ignore the 5 fps real-time pacing (large scenes run faster)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BIN="${PROJECT_ROOT}/bin/tandt_mono"
VOCAB="${PROJECT_ROOT}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
RESULTS_ROOT="${PROJECT_ROOT}/results"
DATASET_ROOT="${DATASET_ROOT:-/workspace/code/Photo-SLAM/datasets/tandt_db}"

# T&T scenes are all 1920x1080; the generic tandt_train.yaml is used for every scene.
# Override ORB_CFG / GAUS_CFG to use a per-scene config.
ORB_CFG="${ORB_CFG:-${PROJECT_ROOT}/cfg/ORB_SLAM3/Monocular/tandt/tandt_train.yaml}"
GAUS_CFG="${GAUS_CFG:-${PROJECT_ROOT}/cfg/gaussian_mapper/Monocular/tandt/tandt_train.yaml}"

RUNS="${RUNS:-1}"
SEQS="${SEQS:-train}"
RESULT_SET="${RESULT_SET:-tandt_mono}"

# Full-length frame list: TANDT_FULL=1 selects <seq>/rgb_2000.txt when present
# (e.g. train ships rgb.txt=500 frames but rgb_2000.txt=2107 frames).
TANDT_FULL="${TANDT_FULL:-0}"
# Explicit override, wins over TANDT_FULL (absolute path, or name relative to the scene dir).
IMAGE_LIST_ENV="${PHOTO_SLAM_IMAGE_LIST:-}"

PHOTO_SLAM_LINE_MODE="${PHOTO_SLAM_LINE_MODE:-2}"
PHOTO_SLAM_LINE_LOOP="${PHOTO_SLAM_LINE_LOOP:-0}"
PHOTO_SLAM_SHADOW_LINE="${PHOTO_SLAM_SHADOW_LINE:-0}"
PHOTO_SLAM_SLAM_ONLY="${PHOTO_SLAM_SLAM_ONLY:-0}"
PHOTO_SLAM_NO_PACING="${PHOTO_SLAM_NO_PACING:-0}"
export PHOTO_SLAM_LINE_MODE PHOTO_SLAM_LINE_LOOP PHOTO_SLAM_SHADOW_LINE \
       PHOTO_SLAM_SLAM_ONLY PHOTO_SLAM_NO_PACING

[ -x "${BIN}" ] || {
    echo "ERROR: ${BIN} not found. Build it first: cmake --build build --target tandt_mono" >&2
    exit 1
}
[ -f "${VOCAB}" ] || { echo "ERROR: vocabulary not found: ${VOCAB}" >&2; exit 1; }

# Auto-generate missing rgb.txt (index / FPS, FPS via TANDT_FPS, default 5.0).
missing=""
for seq in ${SEQS}; do
    [ -f "${DATASET_ROOT}/${seq}/rgb.txt" ] || missing="${missing} ${seq}"
done
if [ -n "${missing}" ]; then
    echo "rgb.txt missing for:${missing} -> generating (FPS=${TANDT_FPS:-5.0})"
    SEQS="${missing# }" FPS="${TANDT_FPS:-5.0}" bash "${SCRIPT_DIR}/make_tandt_rgb_txt.sh"
fi

mkdir -p "${RESULTS_ROOT}"
for ((run = 1; run <= RUNS; ++run)); do
    for seq in ${SEQS}; do
        dataset_dir="${DATASET_ROOT}/${seq}"
        [ -d "${dataset_dir}" ] || { echo "ERROR: no such scene: ${dataset_dir}" >&2; exit 1; }

        # Choose the image list: explicit override > TANDT_FULL (rgb_2000.txt) > rgb.txt
        if [ -n "${IMAGE_LIST_ENV}" ]; then
            export PHOTO_SLAM_IMAGE_LIST="${IMAGE_LIST_ENV}"
            image_list="${IMAGE_LIST_ENV}"
        elif [ "${TANDT_FULL}" = "1" ] && [ -f "${dataset_dir}/rgb_2000.txt" ]; then
            export PHOTO_SLAM_IMAGE_LIST="rgb_2000.txt"
            image_list="rgb_2000.txt"
        else
            unset PHOTO_SLAM_IMAGE_LIST 2>/dev/null || true
            image_list="rgb.txt"
            if [ "${TANDT_FULL}" != "1" ] && [ -f "${dataset_dir}/rgb_2000.txt" ]; then
                echo "NOTE: ${seq} also ships rgb_2000.txt (full frame set); set TANDT_FULL=1 to use it."
            fi
        fi

        if [ "${RUNS}" -gt 1 ]; then
            out_dir="${RESULTS_ROOT}/${RESULT_SET}/run_${run}/${seq}"
        else
            out_dir="${RESULTS_ROOT}/${RESULT_SET}/${seq}"
        fi
        mkdir -p "${out_dir}"

        echo "==================== ${seq} (run ${run}/${RUNS}) ===================="
        {
            echo "scene=${seq}  run=${run}"
            echo "dataset=${dataset_dir}"
            echo "orb_config=${ORB_CFG}"
            echo "gaussian_config=${GAUS_CFG}"
            echo "line_mode=${PHOTO_SLAM_LINE_MODE} slam_only=${PHOTO_SLAM_SLAM_ONLY} no_pacing=${PHOTO_SLAM_NO_PACING}"
            echo "image_list=${image_list}"
            printf '[command]\n'
            printf '%q ' "${BIN}" "${VOCAB}" "${ORB_CFG}" "${GAUS_CFG}" "${dataset_dir}" "${out_dir}" no_viewer
            printf '\n'
        } > "${out_dir}/version.txt"

        set +e
        "${BIN}" "${VOCAB}" "${ORB_CFG}" "${GAUS_CFG}" "${dataset_dir}" "${out_dir}" no_viewer \
            2>&1 | tee "${out_dir}/run.log"
        ec=${PIPESTATUS[0]}
        set -e

        echo "EXIT=${ec}" | tee -a "${out_dir}/run.log"
        echo "[${seq}] EXIT=${ec} files: $(ls "${out_dir}" | tr '\n' ' ')"
    done
done
