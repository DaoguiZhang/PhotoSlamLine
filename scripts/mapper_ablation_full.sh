#!/usr/bin/env bash
# Frozen-geometry mapper ablation across Replica RGB-D and RGB-only (Mono).
#
# For every scene x sensor the SLAM geometry (keyframe poses, sparse point
# cloud incl. line samples, images, camera parameters) is produced ONCE and
# frozen. A/B/C/D are then replayed on that identical snapshot inside the same
# process, so every within-run difference is attributable to the mapper
# configuration (paired design). No SLAM/LBA/GBA touches the geometry during
# replay.
#
#   A  Point-only mapper   : point init, no line structure loss, ordinary densify
#   B  + line init         : line-sampled init, no line structure loss, ordinary densify
#   C  + L_str             : B + full structure regularisation, ordinary densify
#   D  + Dir. dens.        : C + directional (line-aware) densification
#
# Fixed: snapshot per scene x sensor, exactly STEPS real optimizer steps,
# resolution/background/LR/SH/densification/opacity-reset schedule, evaluation
# views, and a pre-generated KF sampling sequence shared by all groups/repeats.
#
# Usage:
#   bash scripts/mapper_ablation_full.sh                       # full grid
#   MODES="rgbd" SCENES="office0" SEEDS="0" bash scripts/mapper_ablation_full.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODES="${MODES:-rgbd mono}"
SCENES="${SCENES:-office0 office1 office2 office3 office4 room0 room1 room2}"
SEEDS="${SEEDS:-0 1 2}"
FRAMES="${FRAMES:-500}"
STEPS="${STEPS:-10000}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/code/SEGS-SLAM/datasets/replica}"
OUT_ROOT="${PROJECT_ROOT}/results/mapper_ablation_full"
VOC="${PROJECT_ROOT}/ORB-SLAM3/Vocabulary/ORBvoc.txt"

export LD_LIBRARY_PATH="${PROJECT_ROOT}/ORB-SLAM3/lib:${PROJECT_ROOT}/ORB-SLAM3/Thirdparty/g2o/lib:${PROJECT_ROOT}/ORB-SLAM3/Thirdparty/DBoW2/lib:${PROJECT_ROOT}/lib"

# Geometry must carry lines so that A/B/C/D can select; loop closure and shadow
# lines off so the frozen geometry is free of those sources of variation.
export PHOTO_SLAM_LINE_MODE=2
export PHOTO_SLAM_LINE_LOOP=0
export PHOTO_SLAM_SHADOW_LINE=0
export PHOTO_SLAM_FROZEN_STEPS="${STEPS}"
unset PHOTO_SLAM_IMAGE_LIST || true

MASTER="${OUT_ROOT}/runs.csv"
mkdir -p "${OUT_ROOT}"

git_commit="$(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"

# ---- frozen snapshot identification of the run environment ----
cat > "${OUT_ROOT}/run_env.json" <<JSON
{
  "git_commit": "${git_commit}",
  "branch": "$(git -C "${PROJECT_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)",
  "gpu": "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)",
  "torch": "$(python3 -c 'import torch;print(torch.__version__)' 2>/dev/null)",
  "steps": ${STEPS},
  "frames_per_scene": ${FRAMES},
  "seeds": "${SEEDS}",
  "scenes": "${SCENES}",
  "modes": "${MODES}",
  "date": "$(date -Is)"
}
JSON
cat "${OUT_ROOT}/run_env.json"

echo "mode,scene,seed,condition,use_line_gaussians,line_loss,directional_densify,psnr,ssim,gaussians_init,gaussians_final,gaussians_point,gaussians_line,opt_steps,refine_seconds,nan,grad_finite,exit,input_hash,image_hash,kf_seq_hash" > "${MASTER}"

for mode in ${MODES}; do
    case "${mode}" in
        rgbd) BIN="${PROJECT_ROOT}/bin/replica_rgbd"
              ORB_DIR="RGB-D"; GAUS_CFG="${PROJECT_ROOT}/cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml"
              WITH_DEPTH=1 ;;
        mono) BIN="${PROJECT_ROOT}/bin/replica_mono"
              ORB_DIR="Monocular"; GAUS_CFG="${PROJECT_ROOT}/cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml"
              WITH_DEPTH=0 ;;
        *) echo "unknown mode ${mode}" >&2; continue ;;
    esac
    [ -x "${BIN}" ] || { echo "missing binary ${BIN}" >&2; continue; }

    # derived config: enable GT image recording only (no training effect)
    GAUS_CFG_RUN="${OUT_ROOT}/gaussian_record_gt_${mode}.yaml"
    python3 - "${GAUS_CFG}" "${GAUS_CFG_RUN}" <<'PY'
import re, sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
s2 = re.sub(r"(Record\.record_ground_truth_image:\s*)\S+", r"\g<1>1", s)
if s2 == s:
    sys.exit("ERROR: Record.record_ground_truth_image missing in " + src)
open(dst, "w").write(s2)
PY

    for scene in ${SCENES}; do
        seq="${DATASET_ROOT}/${scene}"
        orb_cfg="${PROJECT_ROOT}/cfg/ORB_SLAM3/${ORB_DIR}/Replica/${scene}.yaml"
        if [ ! -d "${seq}/results" ] || [ ! -f "${orb_cfg}" ]; then
            echo "SKIP ${mode}/${scene}: missing ${seq}/results or ${orb_cfg}" >&2
            continue
        fi

        # ---- fixed truncated input view (identical for every run) ----
        # Mono deliberately gets NO depth symlinks: the RGB-only algorithm must
        # not be able to consume depth.
        view="${OUT_ROOT}/views/${mode}/${scene}_${FRAMES}/results"
        rm -rf "$(dirname "${view}")"; mkdir -p "${view}"
        ls "${seq}/results"/frame* 2>/dev/null | head -"${FRAMES}" | while read -r f; do
            ln -sf "${f}" "${view}/$(basename "${f}")"
        done
        if [ "${WITH_DEPTH}" = "1" ]; then
            ls "${seq}/results"/depth* 2>/dev/null | head -"${FRAMES}" | while read -r f; do
                ln -sf "${f}" "${view}/$(basename "${f}")"
            done
        fi
        n_rgb=$(ls "${view}"/frame* 2>/dev/null | wc -l)
        n_depth=$(ls "${view}"/depth* 2>/dev/null | wc -l)
        echo "[view] ${mode}/${scene}: ${n_rgb} rgb, ${n_depth} depth"

        for seed in ${SEEDS}; do
            run_dir="${OUT_ROOT}/runs/${mode}/${scene}/seed${seed}"
            # ---- reuse an already-complete run instead of repeating it ----
            # A complete run has a summary.csv with all 4 conditions AND a
            # non-zero exit recorded for each. Incomplete/failed runs are redone.
            reuse=0
            if [ -f "${run_dir}/summary.csv" ] \
               && [ "$(awk -F, 'NR>1 && $NF==0' "${run_dir}/summary.csv" | wc -l)" -eq 4 ] \
               && [ -f "${run_dir}/snapshot.txt" ] && [ -f "${run_dir}/kf_sequence.txt" ]; then
                reuse=1
            fi
            if [ "${reuse}" = "1" ]; then
                echo "===== ${mode}/${scene} seed=${seed} : REUSING existing complete run ====="
            else
            rm -rf "${run_dir}"; mkdir -p "${run_dir}"
            # interleaved condition order (rotated by repeat) to avoid order bias
            export PHOTO_SLAM_FROZEN_ABLATION="${run_dir}"
            export PHOTO_SLAM_FROZEN_ROTATE="${seed}"
            export PHOTO_SLAM_SEED="${seed}"
            echo "===== ${mode}/${scene} seed=${seed} order_rot=${seed} ====="
            t0=$SECONDS
            timeout 10800 "${BIN}" "${VOC}" "${orb_cfg}" "${GAUS_CFG_RUN}" "$(dirname "${view}")" \
                "${run_dir}" no_viewer > "${run_dir}/run.log" 2>&1
            ec=$?; dt=$((SECONDS - t0))
            echo "  exit=${ec} wall=${dt}s"
            grep -E "^\[FrozenABL\]" "${run_dir}/run.log" | sed 's/^/  /' | head -20 || true
            fi
            if [ -f "${run_dir}/summary.csv" ]; then
                awk -v m="${mode}" -v s="${scene}" -v k="${seed}" -F, 'NR>1{print m","s","k","$0}' \
                    "${run_dir}/summary.csv" >> "${MASTER}"
            else
                echo "WARNING: no summary.csv for ${mode}/${scene}/seed${seed} (exit ${ec})" >&2
                echo "${mode},${scene},${seed},RUN_FAILED,,,,,,,,,,,,,,,,${ec},,,," >> "${MASTER}"
            fi
        done
    done
done

echo
echo "master runs table: ${MASTER}"
