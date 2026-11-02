#!/usr/bin/env python3
"""Metrics for the frozen cross-dataset front-end protocol (see eval/PROTOCOL.md).

Everything is computed from artefacts written by the run itself; nothing is
re-run and no failed cell is replaced.

Per run directory:
  final_state.csv            frame,timestamp,state,state_name
  CameraTrajectory_TUM.txt   ts tx ty tz qx qy qz qw   (world<-camera, TUM)
  status.txt / meta.txt      exit code, provenance

Definitions implemented exactly as frozen:
  accepted coverage        = #(state==OK) / frames_processed
  finite-exported coverage = #(finite pose in trajectory) / frames_processed
  gt-matched coverage      = #(pose with GT within max_dt) / gt-evaluable inputs
  correct-pose coverage    = #(accepted AND err<=tol) / gt-evaluable inputs,
                             for tol in the PRE-FIXED set, all reported
  longest continuous       = longest run of consecutive frames that are
                             accepted AND inside the 0.20 m / 5 deg tolerance,
                             measured in seconds using real timestamps
Alignment: ONE SE(3) (no scale) Umeyama fit per run over the whole trajectory.
"""
import argparse
import glob
import json
import math
import os
import statistics as stats
import sys

import numpy as np

TOLS = [0.05, 0.10, 0.20, 0.50]
ROT_TOL_DEG = 5.0
LONG_TOL = 0.20


def quat_to_R(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def read_tum(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            if len(p) < 8:
                continue
            try:
                ts = float(p[0])
                t = [float(p[1]), float(p[2]), float(p[3])]
                q = [float(p[7]), float(p[4]), float(p[5]), float(p[6])]  # w,x,y,z
            except ValueError:
                continue
            out.append((ts, np.array(t), q))
    out.sort(key=lambda r: r[0])
    return out


def align_se3(src, dst, with_scale=False):
    """Umeyama alignment: dst ~ s*R*src + t. s=1 unless with_scale."""
    if len(src) < 3:
        return np.eye(3), np.zeros(3), 1.0
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    S = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, Dv, Vt = np.linalg.svd(S)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1.0
    R = U @ D @ Vt
    s = 1.0
    if with_scale:
        var_s = ((src - mu_s) ** 2).sum() / len(src)
        if var_s > 1e-18:
            s = float(np.trace(np.diag(Dv) @ D) / var_s)
    t = mu_d - s * (R @ mu_s)
    return R, t, s


def rot_deg(R):
    c = (np.trace(R) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def nearest(gt_ts, t, max_dt):
    lo, hi = 0, len(gt_ts) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if gt_ts[mid] < t:
            lo = mid + 1
        else:
            hi = mid
    best = None
    for i in (lo - 1, lo, lo + 1):
        if 0 <= i < len(gt_ts):
            d = abs(gt_ts[i] - t)
            if best is None or d < best[0]:
                best = (d, i)
    if best is None or best[0] > max_dt:
        return None
    return best[1]


def evaluate(run_dir, gt_path, max_dt=0.02, with_scale=False):
    res = {"run": run_dir, "valid": False}
    st = os.path.join(run_dir, "status.txt")
    if not os.path.exists(st):
        res["error"] = "no status.txt"
        return res
    txt = open(st).read()
    exit_code = None
    for line in txt.splitlines():
        if line.startswith("exit_code="):
            exit_code = int(line.split("=")[1])
    res["exit_code"] = exit_code
    if exit_code != 0:
        res["error"] = "non-zero exit"
        return res
    res["valid"] = True

    states = []
    with open(os.path.join(run_dir, "final_state.csv")) as f:
        next(f)
        for line in f:
            p = line.strip().split(",")
            if len(p) < 4:
                continue
            states.append((int(p[0]), float(p[1]), int(p[2])))
    n_frames = len(states)
    res["frames_processed"] = n_frames
    n_ok = sum(1 for s in states if s[2] == 2)
    res["accepted"] = n_ok
    res["accepted_cov"] = n_ok / n_frames if n_frames else 0.0

    traj = read_tum(os.path.join(run_dir, "CameraTrajectory_TUM.txt"))
    finite = [(ts, t, q) for (ts, t, q) in traj if np.all(np.isfinite(t))]
    res["finite_exported"] = len(finite)
    res["finite_cov"] = len(finite) / n_frames if n_frames else 0.0

    gt = read_tum(gt_path)
    if not gt or not finite:
        res["error"] = "empty trajectory or GT"
        res["gt_evaluable"] = 0
        return res
    gt_ts = [g[0] for g in gt]
    est_ts = [f[0] for f in finite]

    # Pairs are indexed by GROUND TRUTH, with the estimate interpolated at the GT
    # timestamp (the frozen `--method interp` convention, and the convention of
    # the reference evaluator). Alignment is therefore fitted in the GT->est
    # direction and residuals are converted back to GT metres by dividing by the
    # fitted scale.
    pairs = []
    for g in gt:
        i = nearest(est_ts, g[0], max_dt)
        if i is None:
            continue
        if i + 1 < len(finite) and abs(finite[i + 1][0] - finite[i][0]) > 1e-9:
            a = (g[0] - finite[i][0]) / (finite[i + 1][0] - finite[i][0])
            if 0.0 <= a <= 1.0:
                t = finite[i][1] + a * (finite[i + 1][1] - finite[i][1])
            else:
                t = finite[i][1]
        else:
            t = finite[i][1]
        pairs.append((g[0], t, finite[i][2], g))
    res["gt_evaluable"] = len(pairs)
    res["gt_cov"] = len(pairs) / n_frames if n_frames else 0.0
    if len(pairs) < 3:
        res["error"] = "too few GT matches"
        return res

    src = np.array([p[3][1] for p in pairs])   # gt
    dst = np.array([p[1] for p in pairs])      # est
    R, tvec, s = align_se3(src, dst, with_scale)
    res["align_scale"] = s
    if not (s > 1e-12):
        res["error"] = "degenerate scale"
        return res
    err = np.linalg.norm(s * (R @ src.T).T + tvec - dst, axis=1) / s

    state_by_ts = {round(s_[1], 6): s_[2] for s_ in states}
    accepted_mask = np.array(
        [state_by_ts.get(round(p[0], 6), -1) == 2 for p in pairs])
    rot = []
    for p in pairs:
        Rw = R @ quat_to_R(p[3][2])   # expected est rotation
        Re = quat_to_R(p[2])
        rot.append(rot_deg(Rw.T @ Re))
    rot = np.array(rot)

    res["ate_rmse_se3"] = float(np.sqrt(np.mean(err ** 2)))
    for tol in TOLS:
        ok = (err <= tol) & (rot <= ROT_TOL_DEG)
        res["correct_cov_%.2f" % tol] = float(
            np.sum(ok & accepted_mask) / len(pairs))

    # longest continuous correctly-tracked run (accepted AND within LONG_TOL)
    good = (err <= LONG_TOL) & (rot <= ROT_TOL_DEG) & accepted_mask
    best = cur = 0
    best_span = (0.0, 0.0)
    start = None
    for i, g in enumerate(good):
        if g:
            cur += 1
            if start is None:
                start = pairs[i][0]
            if cur > best:
                best = cur
                best_span = (start, pairs[i][0])
        else:
            cur = 0
            start = None
    res["longest_continuous_frames"] = best
    res["longest_continuous_s"] = float(best_span[1] - best_span[0])

    # lifecycle from stdout
    log = os.path.join(run_dir, "stdout.log")
    if os.path.exists(log):
        txt = open(log, errors="ignore").read()
        res["maps_final"] = txt.count("SaveKeyFrameTrajectoryTUM:   map")
        res["inits"] = txt.count("[Tracking] init")
    return res


def scan(root, gt_map, with_scale_for):
    rows = []
    for st in sorted(glob.glob(os.path.join(root, "*", "*", "*", "rep*"))):
        parts = st.split(os.sep)
        scene, mod, arm = parts[-4], parts[-3], parts[-2]
        rep = parts[-1]
        gt = gt_map[scene]
        r = evaluate(st, gt, with_scale=with_scale_for(mod))
        r.update({"scene": scene, "modality": mod, "arm": arm,
                  "repeat": rep.replace("rep", "")})
        rows.append(r)
    return rows


def agg(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    if not vals:
        return None, None
    return stats.median(vals), (min(vals), max(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/eval_runs/cabinet")
    ap.add_argument("--data", default="/workspace/code/SEGS-SLAM/datasets/tum")
    ap.add_argument("--out", default="/workspace/eval_runs/cabinet_metrics")
    a = ap.parse_args()

    # Monocular scale is unobservable in this pipeline (measured scale factor
    # ~0 with SE(3)), so monocular runs are aligned with Sim(3); RGB-D is metric
    # and keeps the rigid SE(3) alignment. Recorded in DEVIATIONS.md (D7).
    def with_scale_for(mod):
        return mod == "mono"

    gt_map = {}
    for scene in ("cabinet", "large_cabinet"):
        gt_map[scene] = os.path.join(
            a.data, "rgbd_dataset_freiburg3_%s" % scene, "groundtruth.txt")

    rows = scan(a.root, gt_map, with_scale_for)
    os.makedirs(a.out, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(a.out, "per_run.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

    cells = {}
    for r in rows:
        cells.setdefault((r["scene"], r["modality"], r["arm"]), []).append(r)

    lines = []
    lines.append("| scene | mod | arm | valid | accepted cov (med[min,max]) | "
                 "correct@0.05 | correct@0.10 | correct@0.20 | correct@0.50 | "
                 "longest cont. (s) | ATE RMSE (m) | scale | maps |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(cells):
        rs = cells[key]
        nvalid = sum(1 for r in rs if r.get("valid"))
        def cell(k, fmt="%.3f"):
            m, rng = agg(rs, k)
            if m is None:
                return "N/A"
            return (fmt + " [" + fmt + "," + fmt + "]") % (m, rng[0], rng[1])
        lines.append("| %s | %s | %s | %d/%d | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            key[0], key[1], key[2], nvalid, len(rs),
            cell("accepted_cov"), cell("correct_cov_0.05"), cell("correct_cov_0.10"),
            cell("correct_cov_0.20"), cell("correct_cov_0.50"),
            cell("longest_continuous_s", "%.1f"),
            cell("ate_rmse_se3"), cell("align_scale", "%.4f"),
            cell("maps_final", "%.0f")))

    md = "\n".join(lines)
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write(md + "\n")
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(rows, f, indent=1, default=str)
    print(md)


if __name__ == "__main__":
    main()
