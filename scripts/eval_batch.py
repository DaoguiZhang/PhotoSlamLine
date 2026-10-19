#!/usr/bin/env python3
"""Batch re-evaluation of office1 runs: full pose/map/process metrics table.

For each run directory, computes:
  - Sim(3) ATE (rmse/mean/median/max), scale, SE(3) ATE
  - RPE 1/10/30, matched pairs, coverage, max jump, NaN/Inf
  - map count, reset flag, lost flag, relocalization flag
  - KF count, MP/ML count (from log tail summary)

Usage: eval_batch.py <out_dir> <gt_tum.txt> <eval_script> [--prefix name]
"""
import sys, os, json, subprocess, glob, re

def parse_eval_json(line):
    try:
        return json.loads(line)
    except Exception:
        return {}

def parse_log(log_path):
    txt = ""
    try:
        with open(log_path, errors="ignore") as f:
            txt = f.read()
    except OSError:
        pass
    resets = len(re.findall(r"ResetActiveMap", txt))
    # count atlas maps: look for "There are N maps in the atlas" (max N)
    atlas = re.findall(r"There are (\d+) maps in the atlas", txt)
    n_maps = max((int(x) for x in atlas), default=0)
    lost = len(re.findall(r"Track Lost|mState == LOST|State LOST", txt))
    reloc = len(re.findall(r"Relocalization", txt))
    newmap = len(re.findall(r"New Map created with", txt))
    # KF count: from KeyFrameTrajectory file lines (fallback)
    return {
        "reset_count": resets,
        "atlas_maps": n_maps,
        "lost_markers": lost,
        "reloc_markers": reloc,
        "new_map_markers": newmap,
    }

def parse_kf_mp_ml(run_dir):
    # KeyFrameTrajectory_TUM.txt line count = KF count
    kf = 0
    try:
        with open(os.path.join(run_dir, "KeyFrameTrajectory_TUM.txt")) as f:
            kf = sum(1 for ln in f if ln.strip() and not ln.startswith("#"))
    except OSError:
        pass
    return kf

def main():
    out_dir = sys.argv[1]
    gt = sys.argv[2]
    evalscript = sys.argv[3]
    runs = sorted(glob.glob(os.path.join(out_dir, "run_*")))
    rows = []
    for run_dir in runs:
        name = os.path.basename(run_dir)
        est = os.path.join(run_dir, "CameraTrajectory_TUM.txt")
        if not os.path.exists(est):
            continue
        out_prefix = os.path.join(run_dir, "eval")
        proc = subprocess.run(
            [sys.executable, evalscript, gt, est, out_prefix],
            capture_output=True, text=True)
        m = {}
        if proc.stdout.strip():
            m = parse_eval_json(proc.stdout.strip().splitlines()[-1])
        log_info = parse_log(os.path.join(run_dir, "run.log"))
        kf = parse_kf_mp_ml(run_dir)
        row = {
            "name": name,
            "ate_rmse": m.get("ate_rmse_m"),
            "ate_mean": m.get("ate_mean_m"),
            "ate_median": m.get("ate_median_m"),
            "ate_max": m.get("ate_max_m"),
            "ate_se3": m.get("ate_rmse_se3_m"),
            "scale": m.get("scale"),
            "rpe1": (m.get("rpe") or {}).get("rpe_1"),
            "rpe10": (m.get("rpe") or {}).get("rpe_10"),
            "rpe30": (m.get("rpe") or {}).get("rpe_30"),
            "n_est": m.get("n_est"),
            "matched_gt": m.get("matched_gt_frames"),
            "coverage": m.get("coverage_gt"),
            "max_jump": m.get("max_jump_m"),
            "nan": m.get("nan_count"),
            "inf": m.get("inf_count"),
            "atlas_maps": log_info["atlas_maps"],
            "reset": log_info["reset_count"],
            "lost": log_info["lost_markers"],
            "reloc": log_info["reloc_markers"],
            "new_map": log_info["new_map_markers"],
            "kf": kf,
        }
        # failure flags
        fail = []
        if log_info["reset_count"] > 0: fail.append("RESET")
        if log_info["atlas_maps"] > 1: fail.append("2MAPS")
        if m.get("ate_rmse_m") is not None and m["ate_rmse_m"] > 0.02: fail.append("ATE>0.02")
        if (m.get("nan_count", 0) or 0) > 0 or (m.get("inf_count", 0) or 0) > 0: fail.append("NANINF")
        if m.get("coverage_gt") is not None and m["coverage_gt"] < 0.5: fail.append("LOWCOV")
        row["FAIL"] = ",".join(fail) if fail else "-"
        rows.append(row)

    # print table
    hdr = ["name","ate_rmse","ate_mean","ate_med","ate_max","ate_se3","scale",
           "rpe1","rpe10","rpe30","n_est","cov","maxjump","nan","inf",
           "maps","reset","lost","reloc","kf","FAIL"]
    print("\t".join(hdr))
    for r in rows:
        vals = [str(r.get(k, "")) for k in hdr]
        print("\t".join(vals))
    # summary
    rmses = [r["ate_rmse"] for r in rows if r["ate_rmse"] is not None]
    scales = [r["scale"] for r in rows if r["scale"] is not None]
    fails = [r for r in rows if r["FAIL"] != "-"]
    print("\n# summary: n=%d  failures=%d  ates=%s  scales=%s" % (
        len(rows), len(fails),
        ",".join("%.4f" % x for x in rmses) if rmses else "-",
        ",".join("%.3f" % x for x in scales) if scales else "-"))
    if fails:
        print("# FAILING RUNS:")
        for r in fails:
            print("#   %s ate=%.4f scale=%.3f maps=%d reset=%d FAIL=%s" % (
                r["name"], r["ate_rmse"] or -1, r["scale"] or -1,
                r["atlas_maps"], r["reset"], r["FAIL"]))

if __name__ == "__main__":
    main()
