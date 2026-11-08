#!/usr/bin/env python3
"""Join A/C/H offline native GS PSNR by image name and stage counters."""
import argparse
import csv
import os
import statistics
from pathlib import Path

CKPTS = ["frame_500", "frame_1000", "frame_1500", "frame_end", "natural_end"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tmp/psl-room1-control/h-round3")
    ap.add_argument("--fixed", default="/workspace/code/PSL-perf/evidence/online_checkpoints_r0/fixed_early_frames.txt")
    ap.add_argument("--out", default="/workspace/code/PSL-perf/evidence/checkpoint_followup_r3")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fixed = [os.path.basename(x.strip()) for x in open(args.fixed) if x.strip()]

    quality = []
    fixed_curve = []
    common_frames = []
    for ck in CKPTS:
        vals, names = {}, {}
        for arm in "ACH":
            cams = {int(r["fid"]): os.path.basename(r["img_name"])
                    for r in csv.DictReader(open(root / arm / ck / "cameras.csv"))}
            ps = {int(r["fid"]): float(r["psnr_gs"])
                  for r in csv.DictReader(open(root / "offline" / f"{arm}_{ck}" /
                                               "offline_psnr.csv"))}
            vals[arm] = {cams[f]: v for f, v in ps.items() if f in cams}
            names[arm] = set(vals[arm])
        common = sorted(set.intersection(*names.values()))
        fixed_common = [x for x in fixed if all(x in vals[a] for a in "ACH")]
        for arm in "ACH":
            quality.append([ck, arm, len(names[arm]),
                            len(common),
                            statistics.mean(vals[arm][x] for x in common),
                            len(fixed_common),
                            statistics.mean(vals[arm][x] for x in fixed_common)
                            if fixed_common else float("nan"),
                            vals[arm].get("frame000000.jpg", float("nan"))])
        fixed_curve.append([ck, len(fixed_common),
                            *[statistics.mean(vals[a][x] for x in fixed_common)
                              for a in "ACH"]])
        for x in common:
            common_frames.append([ck, x, *[vals[a][x] for a in "ACH"]])

    with open(out / "quality_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint", "arm", "kfs", "all_common_n",
                    "all_common_psnr_gs", "fixed_common_n",
                    "fixed_common_psnr_gs", "frame0_psnr_gs"])
        w.writerows(quality)
    with open(out / "fixed_early_curve.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint", "n", "A", "C", "H"])
        w.writerows(fixed_curve)
    with open(out / "common_frames.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint", "image", "A", "C", "H"])
        w.writerows(common_frames)

    stage = []
    for arm in "ACH":
        for ck in ("frame_end", "natural_end"):
            d = root / arm / ck
            meta = {}
            for line in open(d / "meta.txt"):
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    meta[k] = v
            levels, sh, backends = {}, {}, {}
            for line in open(d / "stage_updates.csv"):
                k = line.strip().split(",")
                if len(k) != 3:
                    continue
                kind, key, val = k
                if kind == "pyramid_level":
                    levels[key] = int(val)
                elif kind == "active_sh":
                    sh[key] = int(val)
                elif kind == "backend":
                    backends[key] = int(val)
                elif kind == "training_pixels":
                    pixels = int(val)
            b = {}
            for line in open(d / "raster_h_counts.csv"):
                if line.startswith("mode"):
                    continue
                v = line.strip().split(",")
                b = {"mode": v[0], "a": int(v[1]), "c": int(v[2]),
                     "switches": int(v[3]), "last": v[4], "last_sh": v[5]}
            adam = list(csv.DictReader(open(d / "adam_boundary.csv")))
            step = adam[0]["step"] if adam else "NA"
            host = adam[0]["optimizer_steps"] if adam else "NA"
            stage.append([arm, ck, meta.get("optimizer_steps"),
                          meta.get("gaussians"), meta.get("line_gaussians"),
                          meta.get("mapper_kfs"), pixels,
                          levels.get("0"), levels.get("1"), levels.get("2"),
                          levels.get("3"), sh.get("0"), sh.get("1"),
                          sh.get("2"), sh.get("3"),
                          backends.get("A", 0), backends.get("C", 0),
                          b.get("a"), b.get("c"), b.get("switches"),
                          b.get("last"), b.get("last_sh"), step, host])
    with open(out / "stage_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "checkpoint", "optimizer_steps", "gaussians",
                    "line_gaussians", "kfs", "training_pixels",
                    "level0", "level1", "level2", "level3",
                    "sh0", "sh1", "sh2", "sh3", "stage_A", "stage_C",
                    "h_actual_A", "h_actual_C", "h_switches", "h_last",
                    "h_last_sh", "adam_step", "host_steps"])
        w.writerows(stage)

    print(f"wrote {out}/quality_summary.csv")
    print(f"wrote {out}/fixed_early_curve.csv")
    print(f"wrote {out}/common_frames.csv")
    print(f"wrote {out}/stage_summary.csv")


if __name__ == "__main__":
    main()
