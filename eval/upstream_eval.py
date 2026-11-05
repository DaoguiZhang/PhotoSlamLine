#!/usr/bin/env python3
"""Evaluate the 6 upstream control rows with the same evaluator as everything else.

Upstream Photo-SLAM runs the full pipeline (SLAM + Gaussian mapper), so its
accuracy is comparable to the internal cells (same input, same ground truth, same
evaluator, same association) but its COST is not: the internal table is SLAM-only.
The two cost tables are therefore kept apart and never averaged together.

Usage: python3 eval/upstream_eval.py --root /workspace/eval_runs/<root>
"""
import argparse
import csv
import json
import os
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_inputs  # noqa: E402

EVAL = "/workspace/code/PSL-perf/scripts/eval_tum_se3_v2.py"
DS = "/workspace/code/SEGS-SLAM/datasets"


def run_eval(est, gt, prefix, max_dt):
    cmd = [sys.executable, EVAL, gt, est, prefix, "--mode", "sim3",
           "--method", "interp", "--max-dt", str(max_dt),
           "--rpe-deltas", "1,30"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not os.path.exists(prefix + ".json"):
        return None
    with open(prefix + ".json") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--max-dt", type=float, default=0.02)
    a = ap.parse_args()
    root = a.root
    out_dir = os.path.join(root, "metrics_v4")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    up = os.path.join(root, "UPSTREAM")
    for scene in sorted(os.listdir(up)) if os.path.isdir(up) else []:
        for rep in sorted(os.listdir(os.path.join(up, scene, "mono", "UP"))):
            d = os.path.join(up, scene, "mono", "UP", rep)
            est = os.path.join(d, "CameraTrajectory_TUM.txt")
            gt = os.path.join(DS, "tum", "rgbd_dataset_freiburg3_" + scene,
                              "groundtruth.txt")
            rec = {"scene": scene, "repeat": rep, "dir": d,
                   "has_traj": int(os.path.exists(est)),
                   "has_gpu_file": int(os.path.exists(
                       os.path.join(d, "GpuPeakUsageMB.txt")))}
            st = os.path.join(d, "status.txt")
            rec["exit_code"] = ""
            if os.path.exists(st):
                for line in open(st):
                    if line.startswith("exit_code="):
                        rec["exit_code"] = int(line.split("=")[1])
                    if line.startswith("wall_s="):
                        rec["wall_s"] = float(line.split("=")[1])
            mt = os.path.join(d, "meta.txt")
            if os.path.exists(mt):
                for line in open(mt):
                    if line.startswith("upstream_commit="):
                        rec["upstream_commit"] = line.split("=", 1)[1].strip()
            if rec["has_traj"] and rec["exit_code"] == 0:
                meta = run_eval(est, gt, os.path.join(d, "eval_up"), a.max_dt)
                if meta and meta.get("status") == "OK":
                    rec.update({
                        "status": "OK", "n_pairs": meta["n_pairs"],
                        "ate_rmse_m": meta["ate_se3"]["rmse"],
                        "align_scale": meta["align_scale"],
                        "rot_rmse_deg": (meta.get("rot_deg") or {}).get("rmse"),
                        "rpe1_trans_rms_m": (meta["rpe"].get("1") or {}).get(
                            "trans_rms_m"),
                        "rpe1_dt_s": (meta["rpe"].get("1") or {}).get(
                            "dt_mean_s"),
                        "rpe30_trans_rms_m": (meta["rpe"].get("30") or {}).get(
                            "trans_rms_m"),
                    })
                else:
                    rec["status"] = "EVALUATOR_FAILED"
            else:
                rec["status"] = "NOT_RUN_OR_FAILED"
            rows.append(rec)

    outp = os.path.join(out_dir, "UPSTREAM_PER_RUN.csv")
    keys = sorted({k for r in rows for k in r}) if rows else ["scene"]
    with open(outp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # per scene: median over the three repeats
    summ = []
    for scene in sorted({r["scene"] for r in rows}):
        g = [r for r in rows if r["scene"] == scene and r["status"] == "OK"]
        rec = {"scene": scene, "n_reps": sum(1 for r in rows if r["scene"] == scene),
               "n_ok": len(g), "scope": "UPSTREAM_FULL_GAUSSIAN"}
        for m in ("ate_rmse_m", "rot_rmse_deg", "rpe1_trans_rms_m",
                  "rpe30_trans_rms_m", "wall_s"):
            vals = [r[m] for r in g if isinstance(r.get(m), (int, float))]
            if vals:
                rec[m + "_values"] = "|".join("%.6g" % v for v in vals)
                rec[m + "_median"] = "%.6g" % statistics.median(vals)
            else:
                rec[m + "_values"] = rec[m + "_median"] = "UNDETERMINED"
        summ.append(rec)
    sp = os.path.join(out_dir, "UPSTREAM_SUMMARY.csv")
    with open(sp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in summ for k in r}),
                           extrasaction="ignore")
        w.writeheader()
        for r in summ:
            w.writerow(r)

    for r in summ:
        print("%-14s ok=%d/%d ate_med=%s wall_med=%s"
              % (r["scene"], r["n_ok"], r["n_reps"],
                 r.get("ate_rmse_m_median", "NA"),
                 r.get("wall_s_median", "NA")))
    print("wrote %s and %s" % (outp, sp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
