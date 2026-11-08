#!/usr/bin/env python3
"""Build the formal full-GPU-mapping scene manifest and per-cell OFF configs.

Full and OFF share one binary. For every cell the OFF config is generated from
that cell's official Gaussian YAML by zeroing only the three effective
structural-loss weights; this script verifies the diff is exactly those three
lines.
"""
import csv
import os
import re
import subprocess
import sys

EXP = "/workspace/code/PSL-fullmap-exp-v1"
SRC = "/workspace/code/PSL-fullmap-lock-v1"
REPLICA = "/workspace/code/SEGS-SLAM/datasets/replica"
TUM = "/workspace/code/SEGS-SLAM/datasets/tum"
ETH = "/workspace/code/Photo-SLAM/datasets/eth3d"
KEYS = ("weight_line_coherence", "weight_line_shape_ecc", "weight_line_shape_ori")

rows = []


def add(cell, family, dataset, modality, binary, orb, gauss, seq, assoc, frames):
    rows.append(dict(cell=cell, family=family, dataset=dataset, modality=modality,
                     binary=binary, orb_cfg=orb, gauss_cfg=gauss,
                     seq_dir=seq, assoc=assoc, frames=frames))


for s in ("office0", "office1", "office2", "office3", "office4",
          "room0", "room1", "room2"):
    add(f"replica_{s}_mono", "Replica", "Replica", "mono", "replica_mono",
        f"{SRC}/cfg/ORB_SLAM3/Monocular/Replica/{s}.yaml",
        f"{SRC}/cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml",
        f"{REPLICA}/{s}", "", 2000)

for s in ("freiburg1_desk", "freiburg2_xyz", "freiburg3_long_office_household"):
    add(f"tum_{s}_rgbd", "TUM", "TUM", "rgbd", "tum_rgbd",
        f"{SRC}/cfg/ORB_SLAM3/RGB-D/TUM/tum_{s}.yaml",
        f"{SRC}/cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml",
        f"{TUM}/rgbd_dataset_{s}",
        f"{SRC}/cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_{s}.txt", "")

for s in ("freiburg1_desk", "freiburg2_xyz", "freiburg3_long_office_household"):
    add(f"tum_{s}_mono", "TUM", "TUM", "mono", "tum_mono",
        f"{SRC}/cfg/ORB_SLAM3/Monocular/TUM/tum_{s}.yaml",
        f"{SRC}/cfg/gaussian_mapper/Monocular/TUM/tum_mono.yaml",
        f"{TUM}/rgbd_dataset_{s}", "", "")

ETH_RGBD = ("desk_3", "mannequin_1", "mannequin_3", "planar_2", "planar_3",
            "plant_scene_1", "sfm_bench", "sfm_garden", "sfm_house_loop",
            "sfm_lab_room_1", "table_7")
for s in ETH_RGBD:
    cell = "eth3d_" + re.sub(r"[^a-z0-9]+", "", s.lower()) + "_rgbd"
    add(cell, "ETH3D", "ETH3D", "rgbd", "eth3d_rgbd",
        f"{SRC}/cfg/ORB_SLAM3/RGB-D/ETH3D/eth3d_{s}.yaml",
        f"{SRC}/cfg/gaussian_mapper/RGB-D/ETH3D/eth3d_{s}.yaml",
        f"{ETH}/{s}", f"{ETH}/{s}/associated.txt", "")

ETH_MONO = ("desk_3", "planar_2", "planar_3", "sfm_garden", "table_7")
for s in ETH_MONO:
    cell = "eth3d_" + re.sub(r"[^a-z0-9]+", "", s.lower()) + "_mono"
    add(cell, "ETH3D", "ETH3D", "mono", "eth3d_mono",
        f"{SRC}/cfg/ORB_SLAM3/Monocular/ETH3D/eth3d_{s}.yaml",
        f"{SRC}/cfg/gaussian_mapper/Monocular/ETH3D/eth3d_{s}.yaml",
        f"{ETH}/{s}", "", "")


def main() -> int:
    out_csv = os.path.join(EXP, "manifest.csv")
    os.makedirs(os.path.join(EXP, "configs", "off", "formal"), exist_ok=True)
    offdir = os.path.join(EXP, "configs", "off", "formal")
    for r in rows:
        for k in ("orb_cfg", "gauss_cfg"):
            if not os.path.isfile(r[k]):
                print(f"MISSING {k}: {r[k]}", file=sys.stderr)
                return 1
        if r["assoc"] and not os.path.isfile(r["assoc"]):
            print(f"MISSING assoc: {r['assoc']}", file=sys.stderr)
            return 1
        off = os.path.join(offdir, r["cell"] + ".off.yaml")
        text = open(r["gauss_cfg"], encoding="utf-8").read()
        for key in KEYS:
            pat = re.compile(rf"^(\s*Optimization\.{key}:\s*)\S+", re.MULTILINE)
            if len(pat.findall(text)) != 1:
                print(f"ERROR: {r['gauss_cfg']} has no unique {key}", file=sys.stderr)
                return 1
            text = pat.sub(r"\g<1>0.0", text)
        open(off, "w", encoding="utf-8").write(text)
        # Verify exactly the three weight lines changed.
        d = subprocess.run(["diff", "-u", r["gauss_cfg"], off],
                           capture_output=True, text=True).stdout.splitlines()
        changed = [ln for ln in d if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---"))]
        if len(changed) != 6 or not all(any(k in ln for k in KEYS) for ln in changed):
            print(f"ERROR: OFF diff for {r['cell']} is not exactly 3 weights", file=sys.stderr)
            print("\n".join(d), file=sys.stderr)
            return 1
        r["off_cfg"] = off

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv} ({len(rows)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
