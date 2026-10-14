#!/usr/bin/env python3
"""Report the pre-fixed point-drop ladder (30/50/70/90 %) per level x condition.

For each run it reports, from the raw log and the saved trajectory:
  * actual point inliers after pose optimisation  (ptAfterPO quantiles)
  * unique valid MapLines                         (lnUniq quantiles)
  * direction spread of those lines               (lnSpread quantiles)
  * rescue attempts and DEDUPLICATED rescued frames
  * frames fed / trajectory frames / coverage / longest continuous segment
  * offline GT error when a compatible ground-truth file is available

No level is singled out as "best"; all levels are reported as measured.

Usage: python3 scripts/ladder_report.py <ladder_root> [--gt-dir DIR] [--fed FILE]
"""
import os
import re
import sys
import csv
import glob
import json
import statistics as st
import subprocess

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(ROOT_DIR)
EVAL = os.path.join(ROOT_DIR, "eval_tum_se3_v2.py")

RX_SCAN = re.compile(
    r"\[LineScan\] stage=(\S+) map=(-?\d+) frame=(\d+)"
    r" ptBeforePO=(\d+) lnBeforePO=(\d+) ptAfterPO=(\d+) lnAfterPO=(\d+)"
    r" oldPtVerdict=(\S+) lnUniq=(\d+) lnEdges=(\d+) lnSpread=(\S+)"
    r" lnOk=(\d+) lnReason=(-?\d+)")
RX_RESCUE = re.compile(r"\[LineRescue\] stage=(\S+) map=(-?\d+) frame=(\d+)"
                       r".*oldPtVerdict=(\S+) new=(\S+)")
RX_META = re.compile(r"(\w+)=(.*)")


def q(v, p):
    if not v:
        return float("nan")
    v = sorted(v)
    return v[min(len(v) - 1, int(p / 100.0 * (len(v) - 1)))]


def traj_ts(path):
    ts = []
    if os.path.exists(path):
        for line in open(path, errors="replace"):
            line = line.strip()
            if line and not line.startswith("#"):
                p = line.split()
                if len(p) >= 8:
                    ts.append(float(p[0]))
    return ts


def fed_ts(path):
    ts = []
    if path and os.path.exists(path):
        for line in open(path, errors="replace"):
            line = line.strip()
            if line and not line.startswith("#"):
                ts.append(float(line.split()[0]))
    return ts


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    root = sys.argv[1]
    fed_file = None
    gt_dir = None
    if "--fed" in sys.argv:
        fed_file = sys.argv[sys.argv.index("--fed") + 1]
    if "--gt-dir" in sys.argv:
        gt_dir = sys.argv[sys.argv.index("--gt-dir") + 1]
    fed = fed_ts(fed_file)

    rows = []
    for d in sorted(glob.glob(os.path.join(root, "drop*", "*"))):
        m = re.search(r"drop(\d+)", d)
        cond = os.path.basename(d).rsplit("_", 1)[-1]
        if not m or cond not in ("A", "P", "B", "C"):
            continue
        level = int(m.group(1))
        log = os.path.join(d, "log.txt")
        meta = {}
        mp = os.path.join(d, "run.meta")
        if os.path.exists(mp):
            for line in open(mp, errors="replace"):
                mm = RX_META.match(line.strip())
                if mm:
                    meta[mm.group(1)] = mm.group(2)

        scans, attempts, rescued_frames = [], 0, set()
        pt, uniq, spread = [], [], []
        if os.path.exists(log):
            for line in open(log, errors="replace"):
                ms = RX_SCAN.search(line)
                if ms:
                    g = ms.groups()
                    scans.append(g)
                    pt.append(int(g[5]))
                    uniq.append(int(g[8]))
                    try:
                        spread.append(float(g[10]))
                    except ValueError:
                        spread.append(0.0)
                    continue
                mr = RX_RESCUE.search(line)
                if mr:
                    g = mr.groups()
                    if g[3] == "FAIL":
                        attempts += 1
                    if g[3] == "FAIL" and g[4] == "OK":
                        rescued_frames.add(int(g[2]))

        ts = traj_ts(os.path.join(d, "CameraTrajectory_TUM.txt"))
        cov = longest = 0
        if fed and ts:
            import bisect
            present = []
            t = sorted(ts)
            for e in fed:
                i = bisect.bisect_left(t, e)
                ok = False
                for j in (i - 1, i):
                    if 0 <= j < len(t) and abs(t[j] - e) <= 0.02:
                        ok = True
                present.append(ok)
            cur = 0
            for ok in present:
                cur = cur + 1 if ok else 0
                longest = max(longest, cur)
            cov = sum(present) / len(fed)

        row = dict(level=level, cond=cond, scenemode=os.path.basename(d),
                   exit=meta.get("exit", ""), wall_s=meta.get("wall_s", ""),
                   drop=meta.get("stress_drop_point_ratio", ""),
                   start=meta.get("stress_start_frame", ""),
                   decisions=len(scans),
                   ptAfterPO_p50=q(pt, 50), ptAfterPO_min=min(pt) if pt else "",
                   lnUniq_p50=q(uniq, 50), lnUniq_max=max(uniq) if uniq else "",
                   lnUniq_ge8=sum(1 for u in uniq if u >= 8),
                   spread_p50=round(q(spread, 50), 1),
                   spread_max=round(max(spread), 1) if spread else "",
                   rescue_attempts=attempts,
                   rescued_frames_dedup=len(rescued_frames),
                   traj_frames=len(ts), fed=len(fed), coverage=round(cov, 4),
                   longest_seg=longest,
                   ate="", sigma="", rpe1="")
        if gt_dir and len(ts) >= 3:
            gt = None
            for cand in sorted(glob.glob(os.path.join(gt_dir, "*groundtruth*"))) or \
                        sorted(glob.glob(os.path.join(gt_dir, "gt*.txt"))):
                gt = cand
                break
            if gt:
                prefix = os.path.join(d, "ate_gt")
                subprocess.run([sys.executable, EVAL, gt,
                                os.path.join(d, "CameraTrajectory_TUM.txt"),
                                prefix, "--mode", "se3", "--rpe-deltas", "1"],
                               check=False, capture_output=True)
                if os.path.exists(prefix + ".json"):
                    with open(prefix + ".json") as f:
                        r = json.load(f)
                    if r.get("status") == "OK":
                        row["ate"] = round(r["ate_se3"]["rmse"], 6)
                        row["sigma"] = round(r["ate_se3"]["std"], 6)
                        v = r["rpe"].get("1")
                        if v:
                            row["rpe1"] = round(v["trans_rms_m"], 6)
        rows.append(row)

    if not rows:
        print("no ladder runs found under", root)
        return 1
    keys = list(rows[0].keys())
    out = os.path.join(root, "ladder.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("=" * 132)
    print("PRE-FIXED POINT-DROP LADDER (all levels reported; none selected as best)")
    print("=" * 132)
    hdr = (f"{'drop%':<6}{'cond':<5}{'dec':<6}{'ptAfter p50':<12}{'ptAfter min':<12}"
           f"{'uniq p50':<9}{'uniq max':<9}{'uniq>=8':<9}{'spr p50':<9}"
           f"{'att':<7}{'resc#':<7}{'traj':<6}{'cov':<7}{'longest':<9}{'ATE':<11}")
    print(hdr)
    for r in sorted(rows, key=lambda x: (x["level"], x["cond"])):
        print(f"{r['level']:<6}{r['cond']:<5}{r['decisions']:<6}"
              f"{r['ptAfterPO_p50']:<12}{r['ptAfterPO_min']:<12}"
              f"{r['lnUniq_p50']:<9}{r['lnUniq_max']:<9}{r['lnUniq_ge8']:<9}"
              f"{r['spread_p50']:<9}{r['rescue_attempts']:<7}"
              f"{r['rescued_frames_dedup']:<7}{r['traj_frames']:<6}"
              f"{r['coverage']:<7}{r['longest_seg']:<9}"
              f"{(r['ate'] if r['ate'] != '' else '-'):<11}")
    print("\nresc# = DEDUPLICATED rescued frames (unique frame ids with old=FAIL -> new=OK)")
    print("CSV:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
