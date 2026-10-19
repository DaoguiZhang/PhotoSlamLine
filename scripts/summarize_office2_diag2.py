#!/usr/bin/env python3
"""Summarize exp2 B/P/L/C office2 results (independent PO/LBA line toggles)."""
import os
import re
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                    "results", "diagnosis", "office2", "exp2")

GROUPS = [
    ("B_po0_lba0", "B (PO off, LBA off)", "00"),
    ("P_po1_lba0", "P (PO on, LBA off)", "10"),
    ("L_po0_lba1", "L (PO off, LBA on)", "01"),
    ("C_po1_lba1", "C (PO on, LBA on)", "11"),
]


def read_ate(path):
    if not os.path.exists(path):
        return {}
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
    return d


def read_rpe(path):
    if not os.path.exists(path):
        return {}
    d = {}
    for line in open(path):
        m = re.match(r"interval=(\d+): trans_mean=([\d.eE+-]+) trans_rmse=([\d.eE+-]+) "
                     r"rot_mean_deg=([\d.eE+-]+) rot_rmse_deg=([\d.eE+-]+)", line)
        if m:
            d[int(m.group(1))] = {
                "trans_rmse": float(m.group(3)),
                "rot_rmse_deg": float(m.group(5)),
            }
    return d


print(f"{'group':<20} {'run':>3} {'rmse_m':>9} {'median_m':>9} {'worst_m':>9} "
      f"{'scale':>7} {'cov%':>5} {'kfs':>5} {'med_depth':>9} "
      f"{'RPE1_t':>7} {'RPE1_r':>7} {'RPE10_t':>8} {'RPE10_r':>7}")
print("-" * 118)

summary = {}
for key, label, _ in GROUPS:
    rmse_list = []
    for run in (1, 2, 3):
        d = os.path.join(ROOT, key, f"run{run}")
        ate = read_ate(os.path.join(d, "ate.txt"))
        rpe = read_rpe(os.path.join(d, "rpe.txt"))
        log = os.path.join(d, "run.log")
        md = "-"
        if os.path.exists(log):
            m = re.search(r"medianDepth=([\d.eE+-]+)", open(log).read())
            if m:
                md = f"{float(m.group(1)):.3f}"
        kfs = "-"
        if os.path.exists(os.path.join(d, "metrics.txt")):
            for line in open(os.path.join(d, "metrics.txt")):
                if line.startswith("keyframes:"):
                    kfs = line.split(":")[1].strip()
        cov = "-"
        if ate and ate.get("matched"):
            cov = f"{100.0 * ate['matched'] / 2000.0:.2f}"
        if ate and rpe:
            r1, r10 = rpe.get(1, {}), rpe.get(10, {})
            print(f"{key:<20} {run:>3} {ate['rmse']:>9.6f} {ate['median']:>9.6f} "
                  f"{ate['max']:>9.6f} {ate['scale']:>7.3f} {cov:>5} {kfs:>5} {md:>9} "
                  f"{r1.get('trans_rmse', -1):>7.4f} {r1.get('rot_rmse_deg', -1):>7.3f} "
                  f"{r10.get('trans_rmse', -1):>8.4f} {r10.get('rot_rmse_deg', -1):>7.3f}")
            rmse_list.append(ate["rmse"])
        else:
            print(f"{key:<20} {run:>3}   MISSING ate/rpe")
    if rmse_list:
        a = np.array(rmse_list)
        summary[key] = (label, a)
        sd = a.std(ddof=1) if len(a) > 1 else 0.0
        print(f"{key:<20} {'avg':>3} {a.mean():>9.6f} {np.median(a):>9.6f} {a.max():>9.6f} "
              f"sd={sd:.6f}")
    print()

print("=" * 118)
print("Summary (n=3):")
for key, label, _ in GROUPS:
    if key in summary:
        _, a = summary[key]
        sd = a.std(ddof=1) if len(a) > 1 else 0.0
        print(f"  {label:<26} rmse = {a.mean():.6f} +/- {sd:.6f} m  "
              f"(runs: {', '.join(f'{v:.6f}' for v in a)})")
