#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Replica RGB-D batch runner for Photo-SLAM-L
#
# Examples:
#
# 1) Smoke test with full line loop:
#    RUNS=1 \
#    SEQS="office0" \
#    RESULT_SET=replica_rgbd_lc_smoke \
#    PHOTO_SLAM_LINE_LOOP=2 \
#    bash scripts/replica_rgbd.sh
#
# 2) Full experiment:
#    RUNS=5 \
#    RESULT_SET=replica_rgbd_lc \
#    PHOTO_SLAM_LINE_LOOP=2 \
#    bash scripts/replica_rgbd.sh
#
# 3) Override dataset root:
#    DATASET_ROOT=/path/to/datasets bash scripts/replica_rgbd.sh
# ============================================================================

# Resolve project root from this script's location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BIN_DIR="${PROJECT_ROOT}/bin"
VOCAB="${PROJECT_ROOT}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
RESULTS_ROOT="${PROJECT_ROOT}/results"

# Dataset root must contain:
#   replica/office0
#   replica/office1
#   ...
DATASET_ROOT="${DATASET_ROOT:-/workspace/code/SEGS-SLAM/datasets}"
REPLICA_ROOT="${DATASET_ROOT}/replica"

# Batch settings.
RUNS="${RUNS:-5}"
SEQS="${SEQS:-office0 office1 office2 office3 office4 room0 room1 room2}"
RESULT_SET="${RESULT_SET:-replica_rgbd}"

# Runtime modes.
#
# PHOTO_SLAM_LINE_MODE:
#   0 = point-only
#   1 = line frontend, without line optimization edges
#   2 = complete point-line tracking and local mapping
#
# PHOTO_SLAM_LINE_LOOP:
#   0 = standard point loop closing
#       Corrected KF poses are still propagated to associated Gaussians.
#   1 = Stage LB
#       Point loop + MapLine correction/fusion + Gaussian synchronization.
#   2 = Stage LC
#       Stage LB + point-line Sim3 refinement + line Global BA.
#       Automatically falls back to point Sim3 when no valid line pairs exist.
#
# PHOTO_SLAM_SHADOW_LINE:
#   0 = enable normal MapLine backend
#   1 = skip MapLine creation/culling/fusion for isolation
#
# PHOTO_SLAM_SLAM_ONLY:
#   0 = enable Gaussian mapping
#   1 = SLAM-only, if supported by replica_rgbd executable
PHOTO_SLAM_LINE_MODE="${PHOTO_SLAM_LINE_MODE:-2}"
PHOTO_SLAM_LINE_LOOP="${PHOTO_SLAM_LINE_LOOP:-0}"
PHOTO_SLAM_SHADOW_LINE="${PHOTO_SLAM_SHADOW_LINE:-0}"
PHOTO_SLAM_SLAM_ONLY="${PHOTO_SLAM_SLAM_ONLY:-0}"

# Gaussian loop-correction magnitude gate.
# These explicitly match the currently accepted defaults.
PHOTO_SLAM_GAUSS_LOOP_TRANS_M="${PHOTO_SLAM_GAUSS_LOOP_TRANS_M:-0.10}"
PHOTO_SLAM_GAUSS_LOOP_ROT_DEG="${PHOTO_SLAM_GAUSS_LOOP_ROT_DEG:-2.0}"

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_file() {
    [ -f "$1" ] || die "file not found: $1"
}

require_executable() {
    [ -x "$1" ] || die "executable not found or not executable: $1"
}

require_directory() {
    [ -d "$1" ] || die "directory not found: $1"
}

validate_integer() {
    local name="$1"
    local value="$2"

    case "${value}" in
        ''|*[!0-9]*)
            die "${name} must be a positive integer, got '${value}'"
            ;;
    esac

    [ "${value}" -ge 1 ] ||
        die "${name} must be >= 1, got '${value}'"
}

validate_choice() {
    local name="$1"
    local value="$2"
    shift 2

    local allowed
    for allowed in "$@"; do
        if [ "${value}" = "${allowed}" ]; then
            return 0
        fi
    done

    die "${name} has invalid value '${value}'"
}

directory_is_nonempty() {
    [ -d "$1" ] &&
        [ -n "$(find "$1" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]
}

# ---------------------------------------------------------------------------
# Validate arguments and runtime files
# ---------------------------------------------------------------------------

validate_integer "RUNS" "${RUNS}"
validate_choice "PHOTO_SLAM_LINE_MODE" "${PHOTO_SLAM_LINE_MODE}" 0 1 2
validate_choice "PHOTO_SLAM_LINE_LOOP" "${PHOTO_SLAM_LINE_LOOP}" 0 1 2
validate_choice "PHOTO_SLAM_SHADOW_LINE" "${PHOTO_SLAM_SHADOW_LINE}" 0 1
validate_choice "PHOTO_SLAM_SLAM_ONLY" "${PHOTO_SLAM_SLAM_ONLY}" 0 1

require_executable "${BIN_DIR}/replica_rgbd"
require_file "${VOCAB}"
require_directory "${REPLICA_ROOT}"

mkdir -p "${RESULTS_ROOT}"

# Validate every sequence and configuration before starting the batch.
for seq in ${SEQS}; do
    require_directory "${REPLICA_ROOT}/${seq}"
    require_file "${PROJECT_ROOT}/cfg/ORB_SLAM3/RGB-D/Replica/${seq}.yaml"
done

GAUS_CFG="${PROJECT_ROOT}/cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml"
require_file "${GAUS_CFG}"

# ---------------------------------------------------------------------------
# Record version information
# ---------------------------------------------------------------------------

GIT_COMMIT="$(
    git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown
)"
GIT_COMMIT_SHORT="$(
    git -C "${PROJECT_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown
)"
GIT_BRANCH="$(
    git -C "${PROJECT_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null ||
        echo unknown
)"

echo "============================================================"
echo "Replica RGB-D runner"
echo "  project       : ${PROJECT_ROOT}"
echo "  git           : ${GIT_BRANCH} @ ${GIT_COMMIT_SHORT}"
echo "  dataset root  : ${REPLICA_ROOT}"
echo "  runs          : ${RUNS}"
echo "  sequences     : ${SEQS}"
echo "  result set    : ${RESULT_SET}"
echo "  LINE_MODE     : ${PHOTO_SLAM_LINE_MODE}"
echo "  LINE_LOOP     : ${PHOTO_SLAM_LINE_LOOP}"
echo "  SHADOW_LINE   : ${PHOTO_SLAM_SHADOW_LINE}"
echo "  SLAM_ONLY     : ${PHOTO_SLAM_SLAM_ONLY}"
echo "  GAUSS_TRANS_M : ${PHOTO_SLAM_GAUSS_LOOP_TRANS_M}"
echo "  GAUSS_ROT_DEG : ${PHOTO_SLAM_GAUSS_LOOP_ROT_DEG}"
echo "============================================================"

export PHOTO_SLAM_LINE_MODE
export PHOTO_SLAM_LINE_LOOP
export PHOTO_SLAM_SHADOW_LINE
export PHOTO_SLAM_SLAM_ONLY
export PHOTO_SLAM_GAUSS_LOOP_TRANS_M
export PHOTO_SLAM_GAUSS_LOOP_ROT_DEG

completed_runs=0
skipped_runs=0

# ---------------------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------------------

for ((run = 1; run <= RUNS; ++run)); do
    for seq in ${SEQS}; do
        dataset_dir="${REPLICA_ROOT}/${seq}"
        orb_cfg="${PROJECT_ROOT}/cfg/ORB_SLAM3/RGB-D/Replica/${seq}.yaml"
        gau_cfg="${GAUS_CFG}"

        if [ "${RUNS}" -gt 1 ]; then
            out_dir="${RESULTS_ROOT}/${RESULT_SET}/run_${run}/${seq}"
        else
            out_dir="${RESULTS_ROOT}/${RESULT_SET}/${seq}"
        fi

        run_log="${out_dir}/run.log"

        # Safely resume a completed batch.
        if [ -f "${run_log}" ] &&
           grep -qx "EXIT=0" "${run_log}"; then
            echo "SKIP: completed result exists: ${out_dir}"
            skipped_runs=$((skipped_runs + 1))
            continue
        fi

        # Never overwrite incomplete or unrecognized results.
        if directory_is_nonempty "${out_dir}"; then
            die "non-empty incomplete result directory: ${out_dir}
Move or rename it before retrying. Existing files were not overwritten."
        fi

        mkdir -p "${out_dir}"

        # Save exact configurations used by this run.
        cp -- "${orb_cfg}" "${out_dir}/orb_settings.used.yaml"
        cp -- "${gau_cfg}" "${out_dir}/gaussian_settings.used.yaml"

        # Construct the command as an array to preserve paths safely.
        command=(
            "${BIN_DIR}/replica_rgbd"
            "${VOCAB}"
            "${orb_cfg}"
            "${gau_cfg}"
            "${dataset_dir}"
            "${out_dir}"
            "no_viewer"
        )

        # Save reproducibility metadata.
        {
            echo "git_branch=${GIT_BRANCH}"
            echo "git_commit=${GIT_COMMIT}"
            echo "run=${run}"
            echo "sequence=${seq}"
            echo "dataset=${dataset_dir}"
            echo "orb_config=${orb_cfg}"
            echo "gaussian_config=${gau_cfg}"
            echo "line_mode=${PHOTO_SLAM_LINE_MODE}"
            echo "line_loop=${PHOTO_SLAM_LINE_LOOP}"
            echo "shadow_line=${PHOTO_SLAM_SHADOW_LINE}"
            echo "slam_only=${PHOTO_SLAM_SLAM_ONLY}"
            echo "gauss_loop_trans_m=${PHOTO_SLAM_GAUSS_LOOP_TRANS_M}"
            echo "gauss_loop_rot_deg=${PHOTO_SLAM_GAUSS_LOOP_ROT_DEG}"
            echo
            echo "[git status --short]"
            git -C "${PROJECT_ROOT}" status --short 2>/dev/null || true
            echo
            echo "[command]"
            printf '%q ' "${command[@]}"
            printf '\n'
        } > "${out_dir}/version.txt"

        # Record binary checksums.
        {
            sha256sum "${BIN_DIR}/replica_rgbd"

            orb_library="${PROJECT_ROOT}/ORB-SLAM3/lib/libORB_SLAM3.so"
            if [ -f "${orb_library}" ]; then
                sha256sum "${orb_library}"
            fi
        } > "${out_dir}/checksums.sha256"

        echo "------------------------------------------------------------"
        echo "[${run}/${RUNS}] sequence=${seq}"
        echo "output=${out_dir}"
        echo "------------------------------------------------------------"

        set +e
        "${command[@]}" > "${run_log}" 2>&1
        rc=$?
        set -e

        echo "EXIT=${rc}" >> "${run_log}"

        if [ "${rc}" -ne 0 ]; then
            echo "ERROR: replica_rgbd exited with ${rc}" >&2
            echo "sequence : ${seq}" >&2
            echo "run      : ${run}" >&2
            echo "log      : ${run_log}" >&2
            exit "${rc}"
        fi

        completed_runs=$((completed_runs + 1))
        echo "DONE: run=${run}, sequence=${seq}, EXIT=0"
    done
done

echo "============================================================"
echo "Replica RGB-D batch completed"
echo "  newly completed : ${completed_runs}"
echo "  skipped         : ${skipped_runs}"
echo "  results         : ${RESULTS_ROOT}/${RESULT_SET}"
echo "============================================================"