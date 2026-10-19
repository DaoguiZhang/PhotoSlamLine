#!/usr/bin/env bash
# Repeated A/B/C/D line-condition ablation on a fixed RGB-D sequence.
#
# Fixed: sequence, first FRAMES frames, ORB config, Gaussian config, eval frames,
#        and the RNG seed (PHOTO_SLAM_SEED) per repetition.
# Varied: exactly one factor per condition pair:
#   A_full   : PHOTO_SLAM_LINE_MODE=2, Gaussian line loss ON
#   B_gauss0 : PHOTO_SLAM_LINE_MODE=2, Gaussian line loss OFF (weight_line_* = 0)
#   C_lm1    : PHOTO_SLAM_LINE_MODE=1, Gaussian line loss ON   (line FRONTEND only,
#              no line edges in Pose-Opt / Local-BA  -> A vs C = line GEOMETRIC
#              OPTIMIZATION condition comparison)
#   D_pt0    : PHOTO_SLAM_LINE_MODE=0, point-only reference (no line frontend)
#   A vs B = Gaussian LINE-LOSS condition comparison.
#
# Metrics: mean PSNR over keyframes, SE(3) ATE (primary) and Sim(3) ATE
# (supplementary), keyframe count, Gaussian iterations, wall time, exit code.
#
# Usage (from anywhere):
#   bash scripts/ablation_replica_rgbd.sh
#   FRAMES=500 SEEDS="0 1 2" bash scripts/ablation_replica_rgbd.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BIN="${PROJECT_ROOT}/bin/replica_rgbd"
VOC="${PROJECT_ROOT}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
DATASET="${DATASET:-/workspace/code/SEGS-SLAM/datasets/replica/office0}"
FRAMES="${FRAMES:-500}"
SEEDS="${SEEDS:-0 1 2}"
ORB_CFG="${PROJECT_ROOT}/cfg/ORB_SLAM3/RGB-D/Replica/office0.yaml"
GAUS_CFG="${PROJECT_ROOT}/cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml"
GT="${DATASET}/pose_TUM.txt"

OUT_ROOT="${PROJECT_ROOT}/results/ablation_replica_rgbd_${FRAMES}f"
SEQ_VIEW="${OUT_ROOT}/input_${FRAMES}"

export PHOTO_SLAM_LINE_LOOP="${PHOTO_SLAM_LINE_LOOP:-0}"
export PHOTO_SLAM_SHADOW_LINE="${PHOTO_SLAM_SHADOW_LINE:-0}"

[ -x "${BIN}" ] || { echo "ERROR: build replica_rgbd first" >&2; exit 1; }
[ -f "${GT}" ] || { echo "ERROR: groundtruth not found: ${GT}" >&2; exit 1; }

# ---- fixed truncated input view (symlinks; identical for every run) ----
mkdir -p "${SEQ_VIEW}/results"
ls "${DATASET}/results"/frame*.jpg 2>/dev/null | head -"${FRAMES}" | while read -r f; do
    ln -sf "${f}" "${SEQ_VIEW}/results/$(basename "${f}")"
done
ls "${DATASET}/results"/depth*.png 2>/dev/null | head -"${FRAMES}" | while read -r f; do
    ln -sf "${f}" "${SEQ_VIEW}/results/$(basename "${f}")"
done
n_rgb=$(ls "${SEQ_VIEW}/results"/frame*.jpg 2>/dev/null | wc -l)
echo "fixed input: ${SEQ_VIEW}/results  (${n_rgb} frames)"

# ---- fixed "no Gaussian line loss" config (only the 3 weights differ) ----
NOLL_CFG="${OUT_ROOT}/gaussian_no_lineloss.yaml"
python3 - "${GAUS_CFG}" "${NOLL_CFG}" <<'PY'
import re, sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
for k in ("weight_line_coherence", "weight_line_shape_ecc", "weight_line_shape_ori"):
    s = re.sub(rf"(Optimization\.{k}:\s*)\S+", r"\g<1>0.0", s)
open(dst, "w").write(s)
print(f"wrote {dst} (weight_line_* = 0)")
PY

CSV="${OUT_ROOT}/summary.csv"
echo "condition,seed,line_mode,psnr,se3_rmse,sim3_rmse,kfs,gauss_iters,seconds,exit" > "${CSV}"

mean_psnr() {  # $1 = out dir
    local p; p=$(find "$1" -name psnr.txt | head -1)
    awk 'NR>1{s+=$2;n++} END{if(n)printf "%.4f",s/n; else printf "NA"}' "$p" 2>/dev/null
}

run_case() {  # cond mode cfg seed
    local cond="$1" mode="$2" cfg="$3" seed="$4"
    local out="${OUT_ROOT}/${cond}_seed${seed}"
    rm -rf "${out}"; mkdir -p "${out}"

    export PHOTO_SLAM_LINE_MODE="${mode}" PHOTO_SLAM_SEED="${seed}"
    local t0=$SECONDS
    timeout 3600 "${BIN}" "${VOC}" "${ORB_CFG}" "${cfg}" "${SEQ_VIEW}" "${out}" no_viewer \
        > "${OUT_ROOT}/${cond}_seed${seed}.log" 2>&1
    local ec=$? dt=$((SECONDS - t0))

    local psnr se3 sim3 kfs it
    psnr=$(mean_psnr "${out}")
    se3=$(python3 "${PROJECT_ROOT}/scripts/eval_tum_se3.py" "${GT}" "${out}/CameraTrajectory_TUM.txt" "${out}/ate_se3.txt" 2>/dev/null | grep -oE 'rmse=[0-9.]+' | cut -d= -f2)
    sim3=$(python3 "${PROJECT_ROOT}/scripts/eval_tum_sim3.py" "${GT}" "${out}/CameraTrajectory_TUM.txt" "${out}/ate_sim3.txt" 2>/dev/null | grep -oE 'rmse=[0-9.]+' | cut -d= -f2)
    kfs=$(wc -l < "${out}/KeyFrameTrajectory_TUM.txt" 2>/dev/null || echo 0)
    it=$(grep -oE "Total Iterations: [0-9]+" "${OUT_ROOT}/${cond}_seed${seed}.log" | grep -oE "[0-9]+$" | tail -1)

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${cond}" "${seed}" "${mode}" "${psnr:-NA}" "${se3:-NA}" "${sim3:-NA}" \
        "${kfs:-0}" "${it:-NA}" "${dt}" "${ec}" >> "${CSV}"
    echo "[${cond} seed=${seed}] EXIT=${ec} ${dt}s PSNR=${psnr:-NA} SE3=${se3:-NA} Sim3=${sim3:-NA} KFs=${kfs:-0} iters=${it:-NA}"
}

for seed in ${SEEDS}; do
    run_case A_full   2 "${GAUS_CFG}" "${seed}"
    run_case B_gauss0 2 "${NOLL_CFG}" "${seed}"
    run_case C_lm1    1 "${GAUS_CFG}" "${seed}"
    run_case D_pt0    0 "${GAUS_CFG}" "${seed}"
done

echo
echo "==== summary (mean +/- std over seeds) ===="
python3 - "${CSV}" <<'PY'
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1])))
def f(v):
    try: return float(v)
    except Exception: return None
cols = ["psnr", "se3_rmse", "sim3_rmse", "kfs", "gauss_iters", "seconds"]
order = ["A_full", "B_gauss0", "C_lm1", "D_pt0"]
print(f"{'condition':10s} " + " ".join(f"{c:>18s}" for c in cols))
for cond in order:
    sub = [r for r in rows if r["condition"] == cond]
    if not sub: continue
    cells = []
    for c in cols:
        vals = [f(r[c]) for r in sub if f(r[c]) is not None]
        cells.append(f"{statistics.mean(vals):.4f}±{statistics.stdev(vals):.4f}" if len(vals) > 1
                     else (f"{vals[0]:.4f}" if vals else "NA"))
    print(f"{cond:10s} " + " ".join(f"{x:>18s}" for x in cells))
print()
print("A vs C: line GEOMETRIC OPTIMIZATION condition (line edges in PO/LBA on vs off)")
print("A vs B: Gaussian LINE-LOSS condition (weight_line_* on vs 0)")
print("D      : point-only reference")
PY
