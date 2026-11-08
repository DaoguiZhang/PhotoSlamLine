#!/usr/bin/env python3
"""Summarize A/B/C office2 Mono ablation results into a comparison table."""
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results",
                    "diagnosis", "office2")

GROUPS = [
    ("A_point_only", "A point-only", 0),
    ("B_line_frontend", "B line-frontend-only", 1),
    ("C_point_line_current", "C full point-line", 2),
]


def parse_ate(path):
    if not os.path.exists(path):
        return None
    d = {}
    for line in open(path):
        line = line.strip()
        if line.startswith("rmse"):
            d["rmse"] = float(line.split()[1])
        elif line.startswith("mean"):
            d["mean"] = float(line.split()[1])
        elif line.startswith("median"):
            d["median"] = float(line.split()[1])
        elif line.startswith("std"):
            d["std"] = float(line.split()[1])
        elif line.startswith("max"):
            d["max"] = float(line.split()[1])
        elif line.startswith("scale"):
            d["scale"] = float(line.split()[1])
        elif line.startswith("matched_pairs"):
            d["matched"] = int(line.split()[1])
        elif line.startswith("max_ape_frame"):
            d["max_ape_frame"] = int(float(line.split()[1]))
    return d


def parse_metrics(path):
    d = {}
    if not os.path.exists(path):
        return d
    for line in open(path):
        line = line.strip()
        if ":" in line:
            k, v = line.split(":", 1)
            d[k.strip()] = v.strip()
    return d


print(f"{'group':<22} {'run':>4} {'rmse_m':>9} {'mean_m':>9} {'med_m':>9} "
      f"{'std_m':>9} {'scale':>8} {'frames':>7} {'kfs':>5} {'init_frame':>10} "
      f"{'init_pts':>8} {'init_lines':>10}")
print("-" * 120)

table = {}
for key, label, mode in GROUPS:
    rmse_list = []
    row_group = []
    for run in (1, 2, 3):
        d = os.path.join(ROOT, key, f"run{run}")
        ate = parse_ate(os.path.join(d, "ate.txt"))
        met = parse_metrics(os.path.join(d, "metrics.txt"))
        init_line = met.get("New Map created", "")
        pts = "-"
        lns = "-"
        # "New Map created with 464 points and 76 lines." or "... with 464 points"
        import re
        m = re.search(r"with (\d+) points", init_line)
        if m:
            pts = m.group(1)
        m = re.search(r"and (\d+) lines", init_line)
        if m:
            lns = m.group(1)
        kfs = met.get("keyframes", "-")
        init_ts = met.get("init_frame_ts", "-")
        frames = met.get("trajectory_frames", "-")
        if ate:
            rmse_list.append(ate["rmse"])
            print(f"{key:<22} {run:>4} {ate['rmse']:>9.6f} {ate['mean']:>9.6f} "
                  f"{ate['median']:>9.6f} {ate['std']:>9.6f} {ate['scale']:>8.4f} "
                  f"{frames:>7} {kfs:>5} {init_ts:>10} {pts:>8} {lns:>10}")
            row_group.append(ate)
        else:
            print(f"{key:<22} {run:>4}   MISSING ate.txt")
    if rmse_list:
        arr = np.array(rmse_list)
        table[key] = (label, arr)
        print(f"{key:<22} {'mean':>4} {arr.mean():>9.6f} "
              f"{'sd':>6} {arr.std(ddof=1):>9.6f}" if len(arr) > 1 else
              f"{key:<22} {'mean':>4} {arr.mean():>9.6f}")
    print()

print("=" * 120)
print("Summary (RMSE mean ± std over runs):")
for key, label, mode in GROUPS:
    if key in table:
        _, arr = table[key]
        sd = arr.std(ddof=1) if len(arr) > 1 else 0.0
        print(f"  {label:<26} rmse = {arr.mean():.6f} ± {sd:.6f} m   "
              f"(runs: {', '.join(f'{v:.6f}' for v in arr)})")
    else:
        print(f"  {label:<26} NO DATA")
