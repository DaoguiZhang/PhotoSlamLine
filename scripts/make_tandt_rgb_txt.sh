#!/usr/bin/env bash
# Generate rgb.txt for Tanks-and-Temples (tandt_db) scenes that only ship a raw rgb/ folder.
# Mirrors the recipe in datasets/tandt_db/train/readme.txt: timestamp = frame_index / FPS.
#
# Usage (from anywhere):
#   bash scripts/make_tandt_rgb_txt.sh                       # all scenes, FPS=5, only if missing
#   SEQS="truck temple" FORCE=1 bash scripts/make_tandt_rgb_txt.sh
#   DATASET_ROOT=/path/to/tandt_db FPS=5.0 bash scripts/make_tandt_rgb_txt.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET_ROOT="${DATASET_ROOT:-/workspace/code/Photo-SLAM/datasets/tandt_db}"
SEQS="${SEQS:-train family francis temple truck}"
FPS="${FPS:-5.0}"
FORCE="${FORCE:-0}"

for seq in ${SEQS}; do
    rgb_dir="${DATASET_ROOT}/${seq}/rgb"
    rgb_txt="${DATASET_ROOT}/${seq}/rgb.txt"

    if [ ! -d "${rgb_dir}" ]; then
        echo "SKIP ${seq}: no rgb/ dir at ${rgb_dir}"
        continue
    fi
    if [ -f "${rgb_txt}" ] && [ "${FORCE}" != "1" ]; then
        echo "SKIP ${seq}: rgb.txt already exists (set FORCE=1 to overwrite)"
        continue
    fi

    python3 - "${rgb_dir}" "${rgb_txt}" "${FPS}" <<'PY'
import os, sys
rgb_dir, rgb_txt, fps = sys.argv[1], sys.argv[2], float(sys.argv[3])
imgs = sorted(f for f in os.listdir(rgb_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
with open(rgb_txt, "w") as f:
    for i, img in enumerate(imgs):
        f.write(f"{i / fps:.6f} rgb/{img}\n")
print(f"wrote {rgb_txt}: {len(imgs)} frames @ {fps} fps")
PY
done
