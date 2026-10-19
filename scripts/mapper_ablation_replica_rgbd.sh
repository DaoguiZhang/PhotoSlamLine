#!/usr/bin/env bash
# Frozen-geometry MAPPER ablation on Replica office0 (RGB-D).
#
# The SLAM side (keyframe poses, images, sparse point cloud, line samples) is
# produced once per run and then FROZEN. Four mapper conditions are replayed on
# that identical input inside the same process (see
# GaussianMapperLine::runFrozenAblation), so every A/B/C/D difference within a
# run is attributable to the mapper configuration alone (paired design).
#
#   A  point baseline      : no line-seeded Gaussians, no line loss, ordinary densify
#   B  line sampling       : line seeds, no structural regularisation, ordinary densify
#   C  B + structural reg  : line seeds, line loss ON, ordinary densify
#   D  C + directional     : line seeds, line loss ON, directional (line-aware) densify
#
# B/C/D share an identical initialisation and pruning strategy.
#
# Fixed: sequence, first FRAMES frames, ORB/Gaussian config, 10000 optimizer
#        steps, keyframe sampling budget, resolution, LR schedule, eval views.
# Not touched: the old 10K result directories (output goes to a new directory).
#
# Usage:
#   bash scripts/mapper_ablation_replica_rgbd.sh
#   SEEDS="0" FRAMES=500 bash scripts/mapper_ablation_replica_rgbd.sh     # 1x validation
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BIN="${PROJECT_ROOT}/bin/replica_rgbd"
VOC="${PROJECT_ROOT}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
DATASET="${DATASET:-/workspace/code/SEGS-SLAM/datasets/replica/office0}"
FRAMES="${FRAMES:-500}"
SEEDS="${SEEDS:-0 1 2}"
STEPS="${STEPS:-10000}"
CONDS="${CONDS:-A,B,C,D}"
ORB_CFG="${PROJECT_ROOT}/cfg/ORB_SLAM3/RGB-D/Replica/office0.yaml"
GAUS_CFG="${PROJECT_ROOT}/cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml"
OUT_ROOT="${PROJECT_ROOT}/results/mapper_ablation_office0_${FRAMES}f"
SEQ_VIEW="${OUT_ROOT}/input_${FRAMES}"

export LD_LIBRARY_PATH="${PROJECT_ROOT}/ORB-SLAM3/lib:${PROJECT_ROOT}/ORB-SLAM3/Thirdparty/g2o/lib:${PROJECT_ROOT}/ORB-SLAM3/Thirdparty/DBoW2/lib:${PROJECT_ROOT}/lib"

[ -x "${BIN}" ] || { echo "ERROR: build replica_rgbd first" >&2; exit 1; }
[ -d "${DATASET}/results" ] || { echo "ERROR: dataset not found: ${DATASET}/results" >&2; exit 1; }

# ---- fixed truncated input view (symlinks; identical for every run) ----
mkdir -p "${SEQ_VIEW}/results"
ls "${DATASET}/results"/frame*.jpg 2>/dev/null | head -"${FRAMES}" | while read -r f; do
    ln -sf "${f}" "${SEQ_VIEW}/results/$(basename "${f}")"
done
ls "${DATASET}/results"/depth*.png 2>/dev/null | head -"${FRAMES}" | while read -r f; do
    ln -sf "${f}" "${SEQ_VIEW}/results/$(basename "${f}")"
done
n_rgb=$(ls "${SEQ_VIEW}/results"/frame*.jpg 2>/dev/null | wc -l)
echo "fixed input : ${SEQ_VIEW}/results (${n_rgb} frames)"
echo "steps       : ${STEPS}  conditions: ${CONDS}  seeds: ${SEEDS}"
echo "output root : ${OUT_ROOT}"

# ---- derived Gaussian config: enable ground-truth image recording ONLY ----
# Recording has no effect on training/optimisation; it exists purely so that
# LPIPS can be computed post-hoc from the saved renders. This one key is the
# only difference from the base config (verified below).
GAUS_CFG_RUN="${OUT_ROOT}/gaussian_record_gt.yaml"
python3 - "${GAUS_CFG}" "${GAUS_CFG_RUN}" <<'PY'
import re, sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
s2 = re.sub(r"(Record\.record_ground_truth_image:\s*)\S+", r"\g<1>1", s)
if s2 == s:
    sys.exit("ERROR: Record.record_ground_truth_image not found in " + src)
open(dst, "w").write(s2)
print("derived config:", dst, "-> Record.record_ground_truth_image: 1 (recording only)")
PY

# Geometry must carry lines so that A/B/C/D can select; loop/shadow off to keep
# the SLAM side identical and free of loop-closure variation.
export PHOTO_SLAM_LINE_MODE=2
export PHOTO_SLAM_LINE_LOOP=0
export PHOTO_SLAM_SHADOW_LINE=0
export PHOTO_SLAM_FROZEN_STEPS="${STEPS}"
export PHOTO_SLAM_FROZEN_CONDS="${CONDS}"

MASTER="${OUT_ROOT}/summary_all.csv"
echo "seed,condition,use_line_gaussians,line_loss,directional_densify,psnr,ssim,gaussians,iterations,refine_seconds,nan,exit" > "${MASTER}"

for seed in ${SEEDS}; do
    run_dir="${OUT_ROOT}/run${seed}"
    rm -rf "${run_dir}"; mkdir -p "${run_dir}"
    export PHOTO_SLAM_FROZEN_ABLATION="${run_dir}"
    export PHOTO_SLAM_SEED="${seed}"

    echo
    echo "===== SLAM run seed=${seed} (freeze + 4 replays) ====="
    t0=$SECONDS
    timeout 7200 "${BIN}" "${VOC}" "${ORB_CFG}" "${GAUS_CFG_RUN}" "${SEQ_VIEW}" "${run_dir}" no_viewer \
        > "${run_dir}/run.log" 2>&1
    ec=$?; dt=$((SECONDS - t0))
    echo "[run seed=${seed}] exit=${ec} wall=${dt}s"

    grep -E "^\[FrozenABL\]" "${run_dir}/run.log" | sed "s/^/  /" || true

    if [ -f "${run_dir}/summary.csv" ]; then
        awk -v s="${seed}" 'NR>1{print s","$0}' "${run_dir}/summary.csv" >> "${MASTER}"
    else
        echo "WARNING: ${run_dir}/summary.csv missing (run failed?)" >&2
    fi
done

echo
echo "===== per-condition mean +/- std over seeds ====="
python3 - "${MASTER}" <<'PY'
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    print("no rows"); raise SystemExit
def f(v):
    try: return float(v)
    except Exception: return None
order = ["A", "B", "C", "D"]
cols = [("psnr", "PSNR"), ("ssim", "SSIM"), ("gaussians", "Gaussians"),
        ("refine_seconds", "refine_s")]
hdr = f"{'cond':6s}" + "".join(f"{t:>22s}" for _, t in cols)
print(hdr)
for cond in order:
    sub = [r for r in rows if r["condition"] == cond]
    if not sub: continue
    cells = []
    for key, _ in cols:
        vals = [f(r[key]) for r in sub if f(r[key]) is not None]
        if len(vals) > 1:
            cells.append(f"{statistics.mean(vals):.4f}+/-{statistics.stdev(vals):.4f}")
        elif vals:
            cells.append(f"{vals[0]:.4f}")
        else:
            cells.append("NA")
    print(f"{cond:6s}" + "".join(f"{c:>22s}" for c in cells))
print()
print("set A :", next((f"{r['use_line_gaussians']}/{r['line_loss']}/{r['directional_densify']}"
                      for r in rows if r["condition"] == "A"), "NA"),
      "(line_gauss/line_loss/directional_densify)")
for c in ["B", "C", "D"]:
    print(f"set {c} :", next((f"{r['use_line_gaussians']}/{r['line_loss']}/{r['directional_densify']}"
                             for r in rows if r["condition"] == c), "NA"))
print()
print("exit codes:", sorted({r["exit"] for r in rows}))
print("nan flags :", sorted({r["nan"] for r in rows}))
PY

echo
echo "===== per-condition deltas within each seed (paired) ====="
python3 - "${MASTER}" <<'PY'
import csv, statistics, sys
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1])))
bysd = defaultdict(dict)
for r in rows:
    try: bysd[r["seed"]][r["condition"]] = float(r["psnr"])
    except Exception: pass
diffs = defaultdict(list)
for s, d in bysd.items():
    if len(d) < 2: continue
    base = d.get("A")
    for c in ["B", "C", "D"]:
        if base is not None and c in d:
            diffs[f"A-{c}"].append(base - d[c])
for k in sorted(diffs):
    v = diffs[k]
    if len(v) > 1:
        print(f"{k}: {statistics.mean(v):+.4f} dB (sd {statistics.stdev(v):.4f}, n={len(v)})")
    else:
        print(f"{k}: {v[0]:+.4f} dB (n={len(v)})")
PY

# ---- optional LPIPS over the recorded shutdown renders ----
echo
echo "===== LPIPS (post-hoc, from saved renders) ====="
python3 - "${OUT_ROOT}" "${SEEDS}" <<'PY'
import os, sys, glob
root = sys.argv[1]
try:
    import cv2, numpy as np, torch, lpips
except Exception as e:
    print("LPIPS unavailable:", e); raise SystemExit
try:
    net = lpips.LPIPS(net="alex")
    net.eval()
except Exception as e:
    print("LPIPS weights unavailable (offline?):", e); raise SystemExit
for rd in sorted(glob.glob(os.path.join(root, "run*"))):
    for cond in ["A", "B", "C", "D"]:
        gts = sorted(glob.glob(os.path.join(rd, cond, "*_shutdown", "image_gt", "*")))
        if not gts:
            continue
        vals = []
        for g in gts:
            rp = os.path.join(os.path.dirname(os.path.dirname(g)), "image", os.path.basename(g))
            if not os.path.exists(rp):
                continue
            a = cv2.imread(rp); b = cv2.imread(g)
            if a is None or b is None or a.shape != b.shape:
                continue
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB); b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
            ta = torch.from_numpy(a).float().permute(2, 0, 1)[None] / 127.5 - 1.0
            tb = torch.from_numpy(b).float().permute(2, 0, 1)[None] / 127.5 - 1.0
            with torch.no_grad():
                vals.append(float(net(ta, tb).item()))
        if vals:
            print(f"{os.path.basename(rd)}/{cond}: LPIPS={sum(vals)/len(vals):.4f} n={len(vals)}")
PY

echo
echo "master table: ${MASTER}"
