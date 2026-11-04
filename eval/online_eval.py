#!/usr/bin/env python3
"""Evaluate the v_final_2 pose products: ONLINE per frame, FINAL per map.

Two different questions, deliberately kept apart:

**ONLINE** - "what did the system believe at the time it processed each frame?"
  Input: `online_poses.csv` (one row per input frame, `pose_valid` flag).
  The stream is split into epochs by `eval/mapmeta.epochs` (invalid pose, map-id
  change, reset, time gap, or a large backward reference-keyframe jump). Each
  epoch is aligned and scored SEPARATELY: no alignment, RPE or continuity run
  ever spans two epochs, because a different map is a different world frame (and
  because a merge can leave `map_id` unchanged while the frame changes - see
  `eval/POSE_PRODUCTS.md`).
  Denominator: every INPUT frame that has a GT sample within `--max-dt`. A frame
  with no pose, or a pose that is flagged invalid, counts against the system; it
  is never turned into a success and never removed from the denominator.

**FINAL** - "after optimisation, how good is each map's keyframe trajectory?"
  Input: `traj_map<id>.txt`. These are KEYFRAME trajectories, so the result is
  CONDITIONAL on the frames that became keyframes. It is reported as ATE/RPE/rot
  with its own n_kf, and is NEVER used as full-frame coverage.

Outputs (in --out):
  ONLINE_PER_RUN.csv     one row per run: coverage counts at several tolerances,
                         longest continuous interval, frame-weighted ATE/RPE,
                         epoch count, and the consistency verdict
  ONLINE_PER_EPOCH.csv   one row per (run, epoch): map id, n frames, ATE, RPE(d=1)
  FINAL_PER_MAP.csv      one row per (run, map): n_kf, ATE, RPE(d=1, d=30), rot

Only executes when a run has status.txt with exit_code=0? No: failing runs are
included with their status, because dropping them would flatter the system.

Usage:
  python3 eval/online_eval.py --root DIR --out DIR [--max-dt 0.02]
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mapmeta            # noqa: E402
import map_history_check  # noqa: E402
import scene_inputs       # noqa: E402

EVAL = "/workspace/code/PSL-perf/scripts/eval_tum_se3_v2.py"
TOLS = [0.05, 0.10, 0.20, 0.50]
LONG_TOL = 0.20
OK = 2

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
        # Pre-declared rule: ORB-SLAM3's reported camera axes match the RAW
        # rotation stored in traj.txt; the repository's converter applies
        # diag(1,-1,-1) once more, which leaves a constant 180.0 deg residual
        # (measured on office0/room0: 179.9 deg constant vs converted, 0.1 deg
        # median vs raw). Positions are identical in both variants, so only the
        # rotation-based criteria are affected. See eval/REPLICA_GT.md s.6.
        p = "/workspace/eval_runs/gt_replica/%s_gt_tum_s_raw.txt" % scene
        if not os.path.exists(p):
            raise SystemExit("derived Replica GT missing: %s" % p)
        return p
    raise SystemExit("unknown dataset " + dataset)


def gt_timestamps(path):
    ts = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            ts.append(float(s.split()[0]))
    ts.sort()
    return ts


def nearest(ts_sorted, t, max_dt):
    lo, hi = 0, len(ts_sorted) - 1
    if hi < 0:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if ts_sorted[mid] < t:
            lo = mid + 1
        else:
            hi = mid
    best = None
    for i in (lo - 1, lo, lo + 1):
        if 0 <= i < len(ts_sorted):
            d = abs(ts_sorted[i] - t)
            if best is None or d < best[0]:
                best = (d, i)
    if best is None or best[0] > max_dt:
        return None
    return best[1]


def write_est(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write("%.9f %.9f %.9f %.9f %.12f %.12f %.12f %.12f\n"
                    % (r["ts"], r["t"][0], r["t"][1], r["t"][2],
                       r["q"][0], r["q"][1], r["q"][2], r["q"][3]))


def read_perframe(path):
    """ts -> (err_m, rot_deg); columns resolved by name, never by position."""
    out = {}
    idx = None
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                names = line.lstrip("#").split()
                idx = (names.index("ts_est"), names.index("err_m"),
                       names.index("rot_deg"))
                continue
            if idx is None:
                raise SystemExit("perframe dump without header: %s" % path)
            p = line.split()
            out[round(float(p[idx[0]]), 6)] = (float(p[idx[1]]),
                                               float(p[idx[2]]))
    return out


def run_eval(est, gt, prefix, mode, max_dt):
    cmd = [sys.executable, EVAL, gt, est, prefix, "--mode", mode,
           "--method", "interp", "--max-dt", str(max_dt),
           "--rpe-deltas", "1,30"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not os.path.exists(prefix + ".json"):
        return None
    with open(prefix + ".json") as f:
        return json.load(f)


def eval_run(run_dir, dataset, scene, modality, arm, rep, max_dt, gap_max,
             rot_tol, out_epochs, out_maps):
    gt = gt_for(dataset, scene)
    mode = "sim3" if modality == "mono" else "se3"
    res = {"dataset": dataset, "scene": scene, "modality": modality,
           "arm": arm, "repeat": rep, "mode": mode, "status": "NO_ARTEFACT"}

    st = os.path.join(run_dir, "status.txt")
    exit_code = None
    if os.path.exists(st):
        for line in open(st):
            if line.startswith("exit_code="):
                exit_code = int(line.split("=")[1])
    res["exit_code"] = exit_code

    online = mapmeta.read_online(run_dir)
    fs = os.path.join(run_dir, "final_state.csv")
    if online is None or not os.path.exists(fs):
        return res
    states = []
    with open(fs, newline="") as f:
        for d in csv.DictReader(f):
            states.append((int(d["frame"]), float(d["timestamp"]),
                           int(d["state"])))
    states.sort(key=lambda x: x[0])
    res["status"] = "OK" if exit_code == 0 else "NONZERO_EXIT"

    gts = gt_timestamps(gt)
    n_input = len(states)
    n_gt = sum(1 for _, ts, _ in states if nearest(gts, ts, max_dt) is not None)
    n_valid = sum(1 for r in online if r["pose_valid"] == 1)

    # The full input count comes from the frozen input, not from the rows a run
    # managed to write before it died: a crashed run must not shrink the
    # denominator it is judged on.
    n_expected = scene_inputs.expected_inputs(dataset, scene, modality)
    res.update({"n_input": n_input, "n_input_expected": n_expected,
                "input_complete": int(n_input == n_expected),
                "n_gt_evaluable": n_gt, "n_pose_valid": n_valid,
                "n_pose_invalid": n_input - n_valid,
                "n_undetermined_pose": n_input - n_valid,
                "n_unprocessed": max(0, (n_expected or 0) - n_input),
                "pose_valid_frac": (n_valid / n_input) if n_input else 0.0})

    # ---- map history decides whether an alignment can be trusted ----------
    events = mapmeta.read_events(run_dir) or []
    summary_ids = set(mapmeta.read_maps(run_dir).keys())
    verdict = map_history_check.assess(online, events, summary_ids)
    res["map_history_verdict"] = verdict["verdict"]
    res["map_history_note"] = "; ".join(verdict["problems"] + verdict["notes"])
    n_merge = verdict["n_merge"]
    res["n_merge_events"] = n_merge
    res["n_reset_events"] = sum(1 for e in events if e["event"] == "MAP_RESET")

    # ---- ONLINE: per-epoch evaluation ------------------------------------
    # ---- ONLINE: per align-epoch evaluation ------------------------------
    eps = mapmeta.align_epochs(online, n_merge=n_merge, gap_max=gap_max)
    res["n_epochs"] = len(eps)
    ev_dir = os.path.join(run_dir, "eval")
    os.makedirs(ev_dir, exist_ok=True)

    frame_err = {}          # round(ts,6) -> (err, rot)
    frame_epoch = {}        # round(ts,6) -> epoch index
    ate_w = []              # (n, ate_rmse) for the frame-weighted rollup
    rpe1_w = []
    rpe30_w = []
    epoch_rows = []
    bad_epochs = 0
    for k, e in enumerate(eps):
        est = os.path.join(ev_dir, "online_ep%02d.txt" % k)
        write_est(est, e["rows"])
        meta = run_eval(est, gt, est[:-4], mode, max_dt)
        if meta is None or meta.get("status") != "OK":
            bad_epochs += 1
            epoch_rows.append({"dataset": dataset, "scene": scene,
                               "modality": modality, "arm": arm,
                               "repeat": rep, "epoch": k,
                               "map_id": e["map_id"], "n_frames": len(e["rows"]),
                               "status": "EVALUATOR_FAILED"})
            continue
        pf = read_perframe(est[:-4] + ".perframe.txt")
        for ts, v in pf.items():
            frame_err[ts] = v
            frame_epoch[ts] = k
        ate = meta["ate_se3"]["rmse"]
        n_pairs = meta["n_pairs"]
        r1 = meta["rpe"].get("1", {}).get("trans_rms_m")
        r30 = meta["rpe"].get("30", {}).get("trans_rms_m")
        ate_w.append((n_pairs, ate))
        if r1 is not None:
            rpe1_w.append((meta["rpe"]["1"]["n"], r1))
        if r30 is not None:
            rpe30_w.append((meta["rpe"]["30"]["n"], r30))
        epoch_rows.append({
            "dataset": dataset, "scene": scene, "modality": modality,
            "arm": arm, "repeat": rep, "epoch": k, "map_id": e["map_id"],
            "n_frames": len(e["rows"]), "n_pairs": n_pairs,
            "status": "OK",
            "ate_rmse_m": ate, "align_scale": meta["align_scale"],
            "rot_rmse_deg": (meta.get("rot_deg") or {}).get("rmse"),
            "rpe1_dt_s": (meta["rpe"].get("1") or {}).get("dt_mean_s"),
            "rpe1_trans_rms_m": r1,
            "rpe1_rot_rms_deg": (meta["rpe"].get("1") or {}).get("rot_rms_deg"),
            "rpe30_dt_s": (meta["rpe"].get("30") or {}).get("dt_mean_s"),
            "rpe30_trans_rms_m": r30,
            "rpe30_rot_rms_deg": (meta["rpe"].get("30") or {}).get("rot_rms_deg"),
        })
    res["n_epochs_evaluated"] = len(eps) - bad_epochs
    out_epochs.extend(epoch_rows)

    # Combined RMSE across epochs must be pooled by sample count:
    #   rmse = sqrt( sum(n_i * rmse_i^2) / sum(n_i) )
    # Averaging the per-epoch RMSEs would weight a 5-frame epoch like a
    # 500-frame one. The unpooled variant is kept but named for what it is.
    def pooled(pairs):
        tot = sum(n for n, _ in pairs)
        if tot == 0:
            return None
        return math.sqrt(sum(n * v * v for n, v in pairs) / tot)

    def mean_of_epoch_rmse(pairs):
        if not pairs:
            return None
        return sum(v for _, v in pairs) / len(pairs)

    res["ate_rmse_pooled_over_epochs"] = pooled(ate_w)
    res["ate_rmse_mean_of_epoch_rmse"] = mean_of_epoch_rmse(ate_w)
    res["rpe1_trans_rms_pooled"] = pooled(rpe1_w)
    res["rpe30_trans_rms_pooled"] = pooled(rpe30_w)

    # ---- coverage on the input denominator -------------------------------
    if n_gt == 0:
        res["status"] = "NO_GT_EVALUABLE_FRAME"
        return res
    for tol in TOLS:
        good = 0
        good_t = 0
        for _, ts, state in states:
            v = frame_err.get(round(ts, 6))
            if v is None or state != OK:
                continue
            err, rot = v
            if err <= tol:
                good_t += 1
                if rot <= rot_tol:
                    good += 1
        res["online_correct_cov_%.2f" % tol] = good / n_gt
        res["online_correct_trans_only_%.2f" % tol] = good_t / n_gt
    res["online_correct_lower_bound_0.20"] = res.get(
        "online_correct_cov_0.20", 0.0) * n_gt / n_input if n_input else 0.0

    # ---- longest continuous run, inside each align epoch ------------------
    states_by_frame = {}
    for f, ts, st in states:
        states_by_frame[f] = st
        states_by_frame[("ts", f)] = ts
    best_s, best_n = 0.0, 0
    best_t = 0.0
    for e in eps:
        s, n = mapmeta.continuity_intervals(
            e, states_by_frame, frame_err, LONG_TOL, rot_tol, gap_max)
        if s > best_s or (s == best_s and n > best_n):
            best_s, best_n = s, n
        st_, _ = mapmeta.continuity_intervals(
            e, states_by_frame, frame_err, LONG_TOL, rot_tol, gap_max,
            rot_gate=False)
        best_t = max(best_t, st_)
    res["online_longest_continuous_s"] = best_s
    res["online_longest_continuous_frames"] = best_n
    res["online_longest_continuous_trans_only_s"] = best_t

    # ---- FINAL: conditional per-map keyframe evaluation -------------------
    maps = mapmeta.read_maps(run_dir)
    kf_rows = []
    for mid, info in sorted(maps.items()):
        if not info["exists"]:
            continue
        pref = os.path.join(ev_dir, "final_map%02d" % mid)
        meta = run_eval(info["file"], gt, pref, mode, max_dt)
        row = {"dataset": dataset, "scene": scene, "modality": modality,
               "arm": arm, "repeat": rep, "map_id": mid,
               "n_kf": info["n_kf"], "scope": "KEYFRAME_CONDITIONAL"}
        if meta is None or meta.get("status") != "OK":
            row["status"] = "EVALUATOR_FAILED"
        else:
            row.update({
                "status": "OK", "n_pairs": meta["n_pairs"],
                "ate_rmse_m": meta["ate_se3"]["rmse"],
                "align_scale": meta["align_scale"],
                "rot_rmse_deg": (meta.get("rot_deg") or {}).get("rmse"),
                "rpe1_dt_s": (meta["rpe"].get("1") or {}).get("dt_mean_s"),
                "rpe1_trans_rms_m": (meta["rpe"].get("1") or {}).get(
                    "trans_rms_m"),
                "rpe30_dt_s": (meta["rpe"].get("30") or {}).get("dt_mean_s"),
                "rpe30_trans_rms_m": (meta["rpe"].get("30") or {}).get(
                    "trans_rms_m"),
            })
        kf_rows.append(row)
    out_maps.extend(kf_rows)
    # ---- metric status ---------------------------------------------------
    # A run whose map history cannot be reconstructed gets numbers but they are
    # labelled UNDETERMINED and excluded from every headline aggregate: an
    # alignment that may span two world frames is not a measurement.
    if res.get("map_history_verdict") == "AMBIGUOUS":
        res["metric_status"] = "UNDETERMINED_MAP_HISTORY"
    elif res.get("exit_code") not in (0, None):
        res["metric_status"] = "CRASHED_PARTIAL"
    elif n_gt == 0:
        res["metric_status"] = "NO_GT_EVALUABLE"
    else:
        res["metric_status"] = "OK"
    # Convention guard: if the axis rule were wrong for this estimator the
    # rotation error would sit near 180 deg rather than near zero.
    # Only epochs with enough pairs are informative: a 5-frame epoch can be
    # aligned with almost any rotation, so its residual says nothing about the
    # convention. 30 is fixed before looking at any run.
    worst_rot = max([e.get("rot_rmse_deg") or 0.0 for e in epoch_rows
                     if e.get("status") == "OK"
                     and int(e.get("n_pairs") or 0) >= 30] or [0.0])
    res["rot_rmse_max_epoch_deg"] = worst_rot
    res["convention_ok"] = int(worst_rot < 90.0)
    res["n_maps"] = len(maps)
    res["final_kf_ate_rmse_m_map_weighted"] = (
        sum(r.get("ate_rmse_m", 0.0) for r in kf_rows) / len(kf_rows)
        if kf_rows and all("ate_rmse_m" in r for r in kf_rows) else None)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-dt", type=float, default=0.02)
    ap.add_argument("--gap-max", type=float, default=0.5)
    ap.add_argument("--rot-tol", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    runs = []
    for dataset in sorted(os.listdir(a.root)):
        droot = os.path.join(a.root, dataset)
        if not os.path.isdir(droot) or dataset in ("metrics", "UPSTREAM"):
            continue
        for scene in sorted(os.listdir(droot)):
            sd = os.path.join(droot, scene)
            if not os.path.isdir(sd):
                continue
            gt = gt_for(dataset, scene)
            if not os.path.exists(gt):
                continue
            for mod in sorted(os.listdir(sd)):
                md = os.path.join(sd, mod)
                if not os.path.isdir(md):
                    continue
                for arm in sorted(os.listdir(md)):
                    ad = os.path.join(md, arm)
                    if not os.path.isdir(ad):
                        continue
                    for rep in sorted(os.listdir(ad)):
                        rd = os.path.join(ad, rep)
                        if os.path.isdir(rd) and os.path.exists(
                                os.path.join(rd, "status.txt")):
                            runs.append((rd, dataset, scene, mod, arm, rep))
    if a.limit:
        runs = runs[:a.limit]

    online_rows, epoch_rows, map_rows = [], [], []
    for rd, dataset, scene, mod, arm, rep in runs:
        online_rows.append(eval_run(rd, dataset, scene, mod, arm, rep,
                                    a.max_dt, a.gap_max, a.rot_tol,
                                    epoch_rows, map_rows))

    os.makedirs(a.out, exist_ok=True)
    def dump(name, rows):
        if not rows:
            return
        keys = sorted({k for r in rows for k in r})
        with open(os.path.join(a.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    dump("ONLINE_PER_RUN.csv", online_rows)
    dump("ONLINE_PER_EPOCH.csv", epoch_rows)
    dump("FINAL_PER_MAP.csv", map_rows)
    print("runs=%d epochs=%d maps=%d -> %s"
          % (len(online_rows), len(epoch_rows), len(map_rows), a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
