#!/usr/bin/env python3
"""Fair A/B/C comparison over a COMMON time window, with effect sizes.

Why: conditions can differ in tracking coverage (e.g. Mono tracked 869 frames in
A but only 432 in B). Comparing ATE computed over different time spans is not
valid, because drift accumulates with elapsed time. This script therefore:

  1. finds, per (scene, mode), the time window common to every run that produced
     a trajectory:  [max(first_ts), min(last_ts)];
  2. re-evaluates EVERY run restricted to that window (--tmin/--tmax) with the
     corrected evaluator (SE(3) for RGB-D, Sim(3) for Mono);
  3. reports per-condition mean/std, the A-B and B-C differences, their standard
     errors and Welch two-sided p-values, plus coverage/KF/line-edge summaries.

Usage: python3 scripts/ablation_common_window.py <root> [--out DIR]
"""
import os
import re
import sys
import csv
import json
import glob
import math
import subprocess
import statistics as st

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(ROOT_DIR)
EVAL = os.path.join(ROOT_DIR, "eval_tum_se3_v2.py")
DATASETS = "/workspace/code/SEGS-SLAM/datasets/tum"
RUN_RX = re.compile(r"^(near|far)_(rgbd|mono)_([ABCP])$")


# --------------------------------------------------------------- statistics --
def betacf(a, b, x):
    MAXIT, EPS, FPMIN = 200, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    bt = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * betacf(a, b, x) / a
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b


def t_sf2(t, df):
    """Two-sided p-value of Student t."""
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    return betai(df / 2.0, 0.5, x)


def welch(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return (st.mean(a) - st.mean(b) if a and b else float("nan"),
                float("nan"), float("nan"))
    ma, mb = st.mean(a), st.mean(b)
    va, vb = st.variance(a), st.variance(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return ma - mb, 0.0, 1.0
    t = (ma - mb) / se
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return ma - mb, t, t_sf2(abs(t), df)


# ------------------------------------------------------------------- helpers --
def traj_ts(path):
    ts = []
    if not os.path.exists(path):
        return ts
    for line in open(path, errors="replace"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) >= 8:
            ts.append(float(p[0]))
    return ts


def collect(root):
    runs = {}
    for d in sorted(glob.glob(os.path.join(root, "**", "*_*_*"), recursive=True)):
        if not os.path.isdir(d):
            continue
        m = RUN_RX.match(os.path.basename(d))
        if not m:
            continue
        scene, mode, cond = m.groups()
        rm = re.search(r"r(\d+)", os.path.basename(os.path.dirname(d)))
        runs.setdefault((scene, mode), {}).setdefault(cond, []).append(
            {"run": int(rm.group(1)) if rm else 0, "dir": d,
             "ts": traj_ts(os.path.join(d, "CameraTrajectory_TUM.txt"))})
    return runs


def evaluate(d, scene, mode, tmin, tmax, tag):
    traj = os.path.join(d, "CameraTrajectory_TUM.txt")
    if not os.path.exists(traj) or not traj_ts(traj):
        return None
    gt = f"{DATASETS}/rgbd_dataset_freiburg3_structure_notexture_{scene}/groundtruth.txt"
    prefix = os.path.join(d, f"ate_{tag}")
    cmd = [sys.executable, EVAL, gt, traj, prefix,
           "--mode", "sim3" if mode == "mono" else "se3",
           "--rpe-deltas", "1,10,30"]
    if tmin is not None:
        cmd += ["--tmin", repr(tmin)]
    if tmax is not None:
        cmd += ["--tmax", repr(tmax)]
    subprocess.run(cmd, check=False, capture_output=True)
    p = prefix + ".json"
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    root = sys.argv[1]
    runs_all = collect(root)
    outdir = root
    if "--out" in sys.argv:
        outdir = sys.argv[sys.argv.index("--out") + 1]
    os.makedirs(outdir, exist_ok=True)

    rows = []
    for (scene, mode), conds in sorted(runs_all.items()):
        valid = [r for c in conds.values() for r in c if len(r["ts"]) >= 10]
        if not valid:
            print(f"[{scene}/{mode}] no valid trajectories")
            continue
        tmin = max(r["ts"][0] for r in valid)
        tmax = min(r["ts"][-1] for r in valid)
        print(f"\n### {scene}/{mode}: common window {tmin:.3f}..{tmax:.3f} "
              f"({tmax - tmin:.2f} s) over {len(valid)} valid runs "
              f"(of {sum(len(c) for c in conds.values())} total)")

        for cond in "ABCP":
            for r in conds.get(cond, []):
                res = evaluate(r["dir"], scene, mode, tmin, tmax, "cw")
                row = dict(scene=scene, mode=mode, cond=cond, run=r["run"],
                           traj_frames=len(r["ts"]), valid=bool(len(r["ts"]) >= 10))
                if res and res.get("status") == "OK":
                    a = res["ate_se3"]
                    row.update(pairs=res["n_pairs"], ate=round(a["rmse"], 6),
                               sigma=round(a["std"], 6))
                    for dd in (1, 10, 30):
                        v = res["rpe"].get(str(dd))
                        if v:
                            row[f"rpe{dd}_m"] = round(v["trans_rms_m"], 6)
                            row[f"rpe{dd}_deg"] = round(v["rot_rms_deg"], 4)
                else:
                    row.update(pairs="", ate="", sigma="", eval_status="NO_DATA")
                rows.append(row)

    print("\n" + "=" * 100)
    print("COMMON-WINDOW RESULTS")
    print("=" * 100)
    hdr = (f"{'scene':<6}{'mode':<6}{'cond':<5}{'nVal':<6}{'nTot':<6}"
           f"{'ATE mean':<11}{'ATE std':<11}{'rpe1_m':<10}{'rpe1_deg':<10}"
           f"{'rpe30_m':<10}{'rpe30_deg':<10}")
    print(hdr)
    metrics = {}
    for (scene, mode) in sorted(runs_all):
        for cond in "ABCP":
            sub = [r for r in rows if r["scene"] == scene and r["mode"] == mode
                   and r["cond"] == cond]
            if not sub:
                continue
            good = [r for r in sub if isinstance(r["ate"], float)]
            vals = {k: [r[k] for r in good if isinstance(r.get(k), float)]
                    for k in ("ate", "sigma", "rpe1_m", "rpe1_deg",
                              "rpe10_m", "rpe30_m", "rpe30_deg")}
            metrics[(scene, mode, cond)] = vals
            fmean = lambda v: (f"{st.mean(v):.6f}" if v else "-")
            ate_v = vals["ate"]
            sd = (f"{st.stdev(ate_v):.6f}" if len(ate_v) > 1
                  else ("0.000000" if ate_v else "-"))
            print(f"{scene:<6}{mode:<6}{cond:<5}{len(good):<6}{len(sub):<6}"
                  f"{fmean(ate_v):<11}{sd:<11}"
                  f"{fmean(vals['rpe1_m']):<10}{fmean(vals['rpe1_deg']):<10}"
                  f"{fmean(vals['rpe30_m']):<10}{fmean(vals['rpe30_deg']):<10}")

    print("\n" + "=" * 100)
    print("EFFECT SIZES")
    print("  P-B : adds line edges to the optimizers, SAME entry point (clean test)")
    print("  A-P : switches to the other monocular entry point (A vs A-prime)")
    print("  B-C : null control (identical logic; the rescue never fires)")
    print("=" * 100)
    print(f"{'scene':<6}{'mode':<6}{'metric':<12}"
          f"{'P-B':<13}{'t':<9}{'p':<11}"
          f"{'A-P':<13}{'t':<9}{'p':<11}"
          f"{'B-C':<13}{'t':<9}{'p':<11}")
    for (scene, mode) in sorted(runs_all):
        for key in ("ate", "rpe1_m", "rpe1_deg", "rpe30_m"):
            A = metrics.get((scene, mode, "A"), {}).get(key, [])
            P = metrics.get((scene, mode, "P"), {}).get(key, [])
            B = metrics.get((scene, mode, "B"), {}).get(key, [])
            C = metrics.get((scene, mode, "C"), {}).get(key, [])
            d0, t0, p0 = welch(P, B)
            d1, t1, p1 = welch(A, P)
            d2, t2, p2 = welch(B, C)
            pf = lambda x: ("-" if x != x else f"{x:.4f}")
            print(f"{scene:<6}{mode:<6}{key:<12}"
                  f"{d0:<13.6g}{t0:<9.3f}{pf(p0):<11}"
                  f"{d1:<13.6g}{t1:<9.3f}{pf(p1):<11}"
                  f"{d2:<13.6g}{t2:<9.3f}{pf(p2):<11}")

    out = os.path.join(outdir, "common_window.csv")
    if rows:
        keys = sorted({k for r in rows for k in r})
        order = ["scene", "mode", "cond", "run", "valid", "traj_frames",
                 "pairs", "ate", "sigma", "rpe1_m", "rpe1_deg",
                 "rpe10_m", "rpe30_m", "rpe30_deg", "eval_status"]
        keys = order + [k for k in keys if k not in order]
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print("\nCSV:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
