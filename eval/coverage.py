#!/usr/bin/env python3
"""Correct-pose coverage and longest continuous correct tracking.

Design rule: this script does NOT re-implement association, interpolation or
alignment. It shells out to the project's validated evaluator
`eval_tum_se3_v2.py` (which is estimate-centric: one error per *estimate* frame,
GT interpolated with slerp, alignment fitted est -> GT) and consumes the
per-estimate-frame dump it writes. The only thing added here is the join to the
run's own input identity (`final_state.csv`) and the interval logic.

Join key: the estimate frame's OWN timestamp, i.e. the input frame timestamp that
the entry passed to Track*. It is never a GT sample timestamp and never a GT
sample index; a 100 Hz GT is not a 30 Hz video.

Denominators (fixed by eval/PROTOCOL.md):
  n_input          all frames the entry fed to Track*
  n_gt_evaluable   input frames that have a GT association within max_dt
                   -> denominator of correct coverage
  correct[tol]     #(exported AND accepted AND err<=tol AND rot<=rot_tol) / n_gt_evaluable
  lower_bound[tol] the same count divided by n_input: every input frame that
                   could not be judged counts against the system. Reported so a
                   reader can see the worst case permitted by the evidence.
  undetermined     (n_input - n_gt_evaluable) / n_input

A missing estimate is NEVER silently dropped: it is absent from the numerator,
present in n_input, and it breaks a continuous interval.

Continuous-interval break conditions (all pre-declared):
  1. the frame is not exported (no pose written)
  2. the frame state is not OK
  3. translation error > 0.20 m or rotation error > 5 deg
  4. a map reset occurs (`state == NO_IMAGES_YET` at frame index > 0)
  5. the timestamp gap to the next input frame exceeds --gap-max
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys

EVAL = "/workspace/code/PSL-perf/scripts/eval_tum_se3_v2.py"
TOLS = [0.05, 0.10, 0.20, 0.50]
ROT_TOL = 5.0
LONG_TOL = 0.20
OK = 2
NO_IMAGES_YET = 0


def nearest(sorted_ts, t, max_dt):
    """Index of the closest GT sample within max_dt, else None."""
    lo, hi = 0, len(sorted_ts) - 1
    if hi < 0:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_ts[mid] < t:
            lo = mid + 1
        else:
            hi = mid
    best = None
    for i in (lo - 1, lo, lo + 1):
        if 0 <= i < len(sorted_ts):
            d = abs(sorted_ts[i] - t)
            if best is None or d < best[0]:
                best = (d, i)
    if best is None or best[0] > max_dt:
        return None
    return best[1]


def read_gt_timestamps(path):
    ts = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            ts.append(float(s.split()[0]))
    ts.sort()
    return ts


def read_final_state(path):
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for d in r:
            rows.append((int(d["frame"]), float(d["timestamp"]), int(d["state"])))
    rows.sort(key=lambda x: x[0])
    return rows


def read_perframe(path):
    """Per-estimate-frame errors, read by COLUMN NAME.

    Positional parsing is unsafe: an older dump had no rot_deg column, so index
    4 was the translation component ex and would have been consumed as a
    rotation angle. The header is parsed and the required names are resolved
    explicitly.
    """
    out = {}
    cols = None
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                if cols is None:
                    names = line.lstrip("#").split()
                    if "err_m" in names:
                        cols = (names.index("err_m"), names.index("ts_est"),
                                names.index("rot_deg") if "rot_deg" in names
                                else None)
                continue
            if cols is None:
                raise SystemExit("perframe dump has no header: %s" % path)
            p = line.split()
            ie, it, ir = cols
            if ir is None:
                raise SystemExit("perframe dump lacks rot_deg, refusing to "
                                 "read a translation component as rotation: %s"
                                 % path)
            out[round(float(p[it]), 6)] = (float(p[ie]), float(p[ir]))
    return out


def run_evaluator(est, gt, prefix, mode, max_dt):
    # NOTE the argument order: eval_tum_se3_v2.py takes <gt> <est> (see its
    # __doc__ "/ Usage:"), and is estimate-centric: it iterates the ESTIMATE.
    cmd = [sys.executable, EVAL, gt, est, prefix,
           "--mode", mode, "--method", "interp", "--max-dt", str(max_dt),
           "--rpe-deltas", "1,30"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return None, p.stdout + p.stderr
    with open(prefix + ".json") as f:
        return json.load(f), p.stdout


def sanitize_est(est_path, out_path):
    """Drop non-finite poses from the exported trajectory.

    Returns (path_to_use, n_dropped).  Dropped rows are counted and reported;
    they are also absent from the per-frame join, so they break a continuous
    interval instead of being silently ignored.
    """
    kept, dropped = [], 0
    for line in open(est_path):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = s.split()
        if len(p) < 8:
            continue
        try:
            vals = [float(x) for x in p[1:8]]
        except ValueError:
            dropped += 1
            continue
        if not all(math.isfinite(v) for v in vals):
            dropped += 1
            continue
        kept.append(s)
    with open(out_path, "w") as f:
        for s in kept:
            f.write(s + "\n")
    return out_path, dropped


def evaluate_run(run_dir, gt, mode, max_dt, gap_max, rot_tol, verbose=False):
    res = {"run": run_dir, "mode": mode, "max_dt": max_dt,
           "gap_max_s": gap_max, "rot_tol_deg": rot_tol, "status": "UNKNOWN"}
    st = os.path.join(run_dir, "status.txt")
    if os.path.exists(st):
        for line in open(st):
            if line.startswith("exit_code="):
                res["exit_code"] = int(line.split("=")[1])
    est = os.path.join(run_dir, "CameraTrajectory_TUM.txt")
    fs = os.path.join(run_dir, "final_state.csv")
    if not (os.path.exists(est) and os.path.exists(fs)):
        res["status"] = "MISSING_ARTEFACT"
        return res

    os.makedirs(os.path.join(run_dir, "eval"), exist_ok=True)
    prefix = os.path.join(run_dir, "eval", "perframe")
    est_use, n_bad = sanitize_est(est, prefix + ".est_finite.txt")
    res["n_dropped_nonfinite"] = n_bad
    meta, log = run_evaluator(est_use, gt, prefix, mode, max_dt)
    res["evaluator_stdout"] = log.strip().splitlines()[:4]
    if meta is None:
        res["status"] = "EVALUATOR_FAILED"
        return res
    res["ate_rmse_m"] = meta["ate_se3"]["rmse"]
    res["align_scale"] = meta["align_scale"]
    res["align_R"] = meta.get("align_R")
    res["align_t"] = meta.get("align_t")
    res["rot_rmse_deg"] = meta.get("rot_deg", {}).get("rmse")
    res["n_pairs"] = meta["n_pairs"]

    states = read_final_state(fs)
    pf = read_perframe(prefix + ".perframe.txt")

    n_input = len(states)
    n_ok = sum(1 for s in states if s[2] == OK)
    n_exported = sum(1 for s in states if round(s[1], 6) in pf)
    # Denominator of correct coverage = every INPUT frame that has a GT sample
    # within max_dt, whether or not the system exported a pose for it. Using the
    # exported/matched count instead would silently drop the frames the system
    # failed on and inflate the ratio (10 inputs, 4 correct would read 1.0).
    gt_ts_all = read_gt_timestamps(gt)
    n_gt_inputs = sum(1 for _, ts, _ in states
                      if nearest(gt_ts_all, ts, max_dt) is not None)
    n_eval = n_gt_inputs
    res.update({
        "n_input": n_input, "n_accepted": n_ok,
        "n_exported_and_matched": n_exported,
        "n_gt_evaluable": n_eval,
        "n_gt_evaluable_with_pose": len(pf),
        "accepted_cov": n_ok / n_input if n_input else 0.0,
        "gt_evaluable_cov": n_eval / n_input if n_input else 0.0,
        "undetermined_frac": (n_input - n_eval) / n_input if n_input else 1.0,
    })
    if n_eval == 0:
        res["status"] = "NO_GT_EVALUABLE_FRAME"
        return res

    for tol in TOLS:
        good = 0
        good_t = 0
        for _, ts, state in states:
            e = pf.get(round(ts, 6))
            if e is None:
                continue
            err, rot = e
            if state != OK:
                continue
            if err <= tol:
                good_t += 1
                if rot <= rot_tol:
                    good += 1
        res["correct_cov_%.2f" % tol] = good / n_eval
        res["correct_lower_bound_%.2f" % tol] = good / n_input
        res["correct_trans_only_%.2f" % tol] = good_t / n_eval

    # ---- longest continuous correctly-tracked interval -------------------
    good_flags = []
    prev_ts = None
    for idx, (_, ts, state) in enumerate(states):
        e = pf.get(round(ts, 6))
        ok = (e is not None and state == OK and e[0] <= LONG_TOL
              and e[1] <= rot_tol)
        if idx > 0 and state == NO_IMAGES_YET:
            ok = False           # map reset boundary
        if prev_ts is not None and (ts - prev_ts) > gap_max:
            ok = False           # time gap
        prev_ts = ts
        good_flags.append((ok, ts))

    best_s = 0.0
    best_n = 0
    cur_start = None
    cur_n = 0
    for ok, ts in good_flags:
        if ok:
            if cur_n == 0:
                cur_start = ts
            cur_n += 1
            span = ts - cur_start
            if span > best_s or (span == best_s and cur_n > best_n):
                best_s, best_n = span, cur_n
        else:
            cur_n = 0
    res["longest_continuous_s"] = best_s
    res["longest_continuous_frames"] = best_n

    # Same interval logic with the rotation gate removed. Reported because the
    # position-only Sim(3) alignment leaves a systematic rotation residual on
    # these near-planar sequences, which makes the joint gate unattainable and
    # would otherwise report 0 s for every run.
    flags_t = []
    prev_ts = None
    for idx, (_, ts, state) in enumerate(states):
        e = pf.get(round(ts, 6))
        ok = (e is not None and state == OK and e[0] <= LONG_TOL)
        if idx > 0 and state == NO_IMAGES_YET:
            ok = False
        if prev_ts is not None and (ts - prev_ts) > gap_max:
            ok = False
        prev_ts = ts
        flags_t.append((ok, ts))
    best_t = 0.0
    cur_start = None
    cur_n = 0
    for ok, ts in flags_t:
        if ok:
            if cur_n == 0:
                cur_start = ts
            cur_n += 1
            if (ts - cur_start) > best_t:
                best_t = ts - cur_start
        else:
            cur_n = 0
    res["longest_continuous_trans_only_s"] = best_t

    # ---- automatic consistency checks (fail loudly, never silently) ------
    problems = []
    for tol in TOLS:
        j = res.get("correct_cov_%.2f" % tol)
        t = res.get("correct_trans_only_%.2f" % tol)
        if j is not None and t is not None and j > t + 1e-12:
            problems.append("joint>translation at tol %.2f" % tol)
        if j is not None and j > res["accepted_cov"] + 1e-12:
            problems.append("correct>accepted at tol %.2f" % tol)
    if res.get("longest_continuous_frames", 0) == 0 and best_s > 1e-9:
        problems.append("joint longest>0 while joint frames==0")
    if res.get("longest_continuous_trans_only_s", 0) > 0 and best_t == 0:
        problems.append("translation-only longest inconsistent")
    res["consistency"] = "OK" if not problems else "; ".join(problems)
    res["status"] = "OK" if res.get("exit_code", 0) == 0 else "NONZERO_EXIT"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--data", default="/workspace/code/SEGS-SLAM/datasets/tum")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gap-max", type=float, default=0.5)
    ap.add_argument("--rot-tol", type=float, default=ROT_TOL)
    ap.add_argument("--max-dt", type=float, default=0.02)
    a = ap.parse_args()

    DS = "/workspace/code/SEGS-SLAM/datasets"
    TUM_DIRS = {"freiburg1_desk": "rgbd_dataset_freiburg1_desk",
                "freiburg1_room": "rgbd_dataset_freiburg1_room",
                "freiburg2_xyz": "rgbd_dataset_freiburg2_xyz"}

    def gt_for(dataset, scene):
        if dataset == "TUM":
            d = TUM_DIRS.get(scene, "rgbd_dataset_freiburg3_" + scene)
            return os.path.join(DS, "tum", d, "groundtruth.txt")
        if dataset == "ETH3D":
            return os.path.join(DS, "eth3d", scene, "groundtruth.txt")
        if dataset == "Replica":
            # pose_TUM.txt timestamps are frame indices; the run's time base is
            # i/30. The cached copy maps index k -> k/30 (content unchanged).
            c = "/workspace/eval_runs/gt_cache/replica_%s.txt" % scene
            return c if os.path.exists(c) else os.path.join(DS, "replica", scene,
                                                            "pose_TUM.txt")
        raise SystemExit("unknown dataset " + dataset)
    rows = []
    for dataset in sorted(os.listdir(a.root)):
        droot = os.path.join(a.root, dataset)
        if not os.path.isdir(droot) or dataset == "UPSTREAM":
            continue
        for scene in sorted(os.listdir(droot)):
            sdir = os.path.join(droot, scene)
            if not os.path.isdir(sdir):
                continue
            gt = gt_for(dataset, scene)
            if not os.path.exists(gt):
                continue
            for mod in sorted(os.listdir(sdir)):
                pmod = os.path.join(sdir, mod)
                if not os.path.isdir(pmod):
                    continue
                for arm in sorted(os.listdir(pmod)):
                    adir = os.path.join(pmod, arm)
                    if not os.path.isdir(adir):
                        continue
                    for rep in sorted(os.listdir(adir)):
                        rd = os.path.join(adir, rep)
                        if not os.path.isdir(rd):
                            continue
                        r = evaluate_run(rd, gt,
                                         "sim3" if mod == "mono" else "se3",
                                         a.max_dt, a.gap_max, a.rot_tol)
                        r.update({"dataset": dataset, "scene": scene,
                                  "modality": mod, "arm": arm, "repeat": rep})
                        rows.append(r)

    os.makedirs(a.out, exist_ok=True)
    keys = ["dataset", "scene", "modality", "arm", "repeat", "status", "exit_code",
            "n_input", "n_accepted", "n_gt_evaluable", "accepted_cov",
            "gt_evaluable_cov", "undetermined_frac",
            "correct_cov_0.05", "correct_cov_0.10", "correct_cov_0.20",
            "correct_cov_0.50", "correct_lower_bound_0.20",
            "correct_trans_only_0.05", "correct_trans_only_0.10",
            "correct_trans_only_0.20", "correct_trans_only_0.50",
            "longest_continuous_s", "longest_continuous_frames",
            "longest_continuous_trans_only_s",
            "ate_rmse_m", "rot_rmse_deg", "align_scale"]
    with open(os.path.join(a.out, "coverage_per_run.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(os.path.join(a.out, "coverage.json"), "w") as f:
        json.dump(rows, f, indent=1)
    for r in rows:
        print("%-8s %-18s %-5s %-3s %-5s %-6s cov=%s correct0.20=%s longest=%ss ate=%s"
              % (r.get("dataset",""), r["scene"], r["modality"], r["arm"], r["repeat"], r["status"],
                 ("%.3f" % r["accepted_cov"]) if "accepted_cov" in r else "NA",
                 ("%.3f" % r.get("correct_cov_0.20", float("nan"))),
                 ("%.1f" % r.get("longest_continuous_s", float("nan"))),
                 ("%.4f" % r["ate_rmse_m"]) if "ate_rmse_m" in r else "NA"))


if __name__ == "__main__":
    main()
