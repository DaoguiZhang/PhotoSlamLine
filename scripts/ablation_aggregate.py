#!/usr/bin/env python3
"""Aggregate A/B/C line-ablation runs into per-(scene, mode) CSV + summary.

Layout expected under the root (run index optional):

    <root>/[r<K>/]{near,far}_{rgbd,mono}_{A,B,C}/
        log.txt  CameraTrajectory_TUM.txt  run.meta

For every run it re-evaluates the saved trajectory with the corrected evaluator
(`scripts/eval_tum_se3_v2.py`, estimate-centric association, SE(3) for RGB-D and
Sim(3) for Monocular) and combines that with the tracking/diagnostic quantities
parsed from the log.

Reported per run:
  pairs, ATE RMSE (m), per-frame translation sigma (m),
  RPE d=1/10/30: translation rms (m and m/s) and rotation rms (deg),
  frames fed / trajectory frames / coverage / longest continuous segment /
  frames not localised, wall time (s), KF count, map points, map lines,
  PoseOptimizationWithLine calls, calls carrying line edges, total line edges,
  line inliers, rescued frames (old FAIL -> new OK), gate old-path failures.

Usage: python3 scripts/ablation_aggregate.py <root> [--out DIR] [--eval/--no-eval]
"""
import os
import re
import sys
import csv
import json
import glob
import subprocess
import statistics as st

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(ROOT_DIR)
DATASETS = "/workspace/code/SEGS-SLAM/datasets/tum"
EVAL = os.path.join(ROOT_DIR, "eval_tum_se3_v2.py")

RUN_RX = re.compile(r"^(near|far)_(rgbd|mono)_([ABCP])$")
RX_EDGE = re.compile(r"\[LineEdgeDiag\] PO frame=(\d+) NL=(\d+) lnTrackedNonNull=(\d+)"
                     r" ptEdges=(\d+) ptIn=(\d+) ptOut=(\d+) ptChi2=(\S+)"
                     r" lnEdges=(\d+) lnEdgesMono=(\d+) lnEdgesStereo=(\d+)"
                     r" lnIn=(\d+) lnOut=(\d+) lnUniq=(\d+)"
                     r"(?: lnSpread=(\S+))? lnChi2=(\S+)")
RX_RESCUE = re.compile(r"\[LineRescue\].*oldPtVerdict=(\S+) new=(\S+)")
RX_GATE = re.compile(r"\[LineGate\].*old_would_fail=(\d+)")
RX_KF = re.compile(r"has (\d+) KFs")
RX_MAP = re.compile(r"([\d]+) .*?, ([\d]+) ")


def read_meta(path):
    d = {}
    if os.path.exists(path):
        for line in open(path, errors="replace"):
            if "=" in line:
                k, v = line.strip().split("=", 1)
                d[k] = v
    return d


def expected_timestamps(scene, mode):
    """The timestamps the tracker is actually fed, in order."""
    seq = f"{DATASETS}/rgbd_dataset_freiburg3_structure_notexture_{scene}"
    out = []
    if mode == "rgbd":
        p = os.path.join(PROJ, "cfg/ORB_SLAM3/RGB-D/TUM/associations",
                         f"fr3_structure_notexture_{scene}.txt")
        for line in open(p):
            line = line.strip()
            if line:
                out.append(float(line.split()[0]))
    else:
        for line in open(os.path.join(seq, "rgb.txt")):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(float(line.split()[0]))
    return out


def traj_timestamps(path):
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


def coverage(exp, traj, tol=0.02):
    if not exp:
        return dict(fed=0, traj_n=len(traj), coverage=0.0, longest=0, not_localised=0)
    import bisect
    t = sorted(traj)
    present = []
    for e in exp:
        i = bisect.bisect_left(t, e)
        ok = False
        for j in (i - 1, i):
            if 0 <= j < len(t) and abs(t[j] - e) <= tol:
                ok = True
        present.append(ok)
    longest = cur = 0
    for ok in present:
        cur = cur + 1 if ok else 0
        longest = max(longest, cur)
    n = sum(present)
    return dict(fed=len(exp), traj_n=len(traj), coverage=n / len(exp),
                longest=longest, not_localised=len(exp) - n)


def parse_log(log):
    d = dict(kf=0, map_points=0, map_lines=0, po_calls=0, po_with_lines=0,
             line_edges=0, line_inliers=0, rescued=0, gate_fail=0)
    if not os.path.exists(log):
        return d
    for line in open(log, errors="replace"):
        m = RX_EDGE.search(line)
        if m:
            d["po_calls"] += 1
            ne = int(m.group(8))
            d["line_edges"] += ne
            d["line_inliers"] += int(m.group(11))
            if ne > 0:
                d["po_with_lines"] += 1
            continue
        m = RX_RESCUE.search(line)
        if m:
            if m.group(1) == "FAIL" and m.group(2) == "OK":
                d["rescued"] += 1
            continue
        m = RX_GATE.search(line)
        if m and int(m.group(1)) == 1:
            d["gate_fail"] += 1
            continue
        m = RX_KF.search(line)
        if m:
            d["kf"] = max(d["kf"], int(m.group(1)))
            continue
        if "个地图点" in line:
            nums = re.findall(r"(\d+)\s*个地图点.*?(\d+)\s*条地图线", line)
            if nums:
                d["map_points"] = int(nums[0][0])
                d["map_lines"] = int(nums[0][1])
    return d


def run_eval(gt, traj, mode, out_prefix):
    subprocess.run([sys.executable, EVAL, gt, traj, out_prefix,
                    "--mode", mode, "--rpe-deltas", "1,10,30"],
                   check=False, capture_output=True)
    p = out_prefix + ".json"
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    root = sys.argv[1]
    do_eval = "--no-eval" not in sys.argv
    outdir = root
    if "--out" in sys.argv:
        outdir = sys.argv[sys.argv.index("--out") + 1]
    os.makedirs(outdir, exist_ok=True)

    rows = []
    for d in sorted(glob.glob(os.path.join(root, "**", "*_*_*"), recursive=True)):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        m = RUN_RX.match(name)
        if not m:
            continue
        scene, mode, cond = m.groups()
        rm = re.search(r"r(\d+)", os.path.basename(os.path.dirname(d)))
        run_idx = int(rm.group(1)) if rm else 0
        meta = read_meta(os.path.join(d, "run.meta"))
        traj = os.path.join(d, "CameraTrajectory_TUM.txt")
        gt = f"{DATASETS}/rgbd_dataset_freiburg3_structure_notexture_{scene}/groundtruth.txt"
        exp = expected_timestamps(scene, mode)
        tts = traj_timestamps(traj)
        cov = coverage(exp, tts)
        lg = parse_log(os.path.join(d, "log.txt"))

        row = dict(scene=scene, mode=mode, cond=cond, run=run_idx,
                   exit=meta.get("exit", ""), wall_s=meta.get("wall_s", ""),
                   complete=meta.get("complete", ""),
                   last_frame=meta.get("last_frame", ""),
                   timeout_s=meta.get("timeout_s", ""),
                   fed=cov["fed"], traj_frames=cov["traj_n"],
                   coverage=round(cov["coverage"], 4),
                   longest_seg=cov["longest"],
                   not_localised=cov["not_localised"],
                   kf=lg["kf"], map_points=lg["map_points"], map_lines=lg["map_lines"],
                   po_calls=lg["po_calls"], po_with_lines=lg["po_with_lines"],
                   line_edges=lg["line_edges"], line_inliers=lg["line_inliers"],
                   rescued=lg["rescued"], gate_old_fail=lg["gate_fail"])

        res = None
        if do_eval and len(tts) >= 3:
            prefix = os.path.join(d, "ate")
            res = run_eval(gt, traj, "sim3" if mode == "mono" else "se3", prefix)
        if res and res.get("status") == "OK":
            a = res["ate_se3"]
            row.update(pairs=res["n_pairs"], unmatched=res["n_unmatched_est_frames"],
                       align_scale=round(res["align_scale"], 6),
                       ate_rmse=round(a["rmse"], 6), ate_mean=round(a["mean"], 6),
                       ate_median=round(a["median"], 6), ate_sigma=round(a["std"], 6),
                       ate_max=round(a["max"], 6))
            for dd in (1, 10, 30):
                v = res["rpe"].get(str(dd)) or res["rpe"].get(dd)
                if v:
                    row[f"rpe{dd}_trans_m"] = round(v["trans_rms_m"], 6)
                    row[f"rpe{dd}_trans_ms"] = round(v["trans_rms_m_per_s"], 5)
                    row[f"rpe{dd}_rot_deg"] = round(v["rot_rms_deg"], 4)
        else:
            row.update(pairs="", unmatched="", align_scale="", ate_rmse="",
                       ate_mean="", ate_median="", ate_sigma="", ate_max="")
            row["eval_status"] = (res or {}).get("status", "NO_TRAJECTORY")
            for dd in (1, 10, 30):
                row[f"rpe{dd}_trans_m"] = ""
                row[f"rpe{dd}_trans_ms"] = ""
                row[f"rpe{dd}_rot_deg"] = ""
        rows.append(row)

    if not rows:
        print("no run directories found under", root)
        return 1

    fields = ["scene", "mode", "cond", "run", "exit", "wall_s",
              "complete", "last_frame", "timeout_s",
              "pairs", "unmatched", "align_scale", "ate_rmse", "ate_mean",
              "ate_median", "ate_sigma", "ate_max",
              "rpe1_trans_m", "rpe1_trans_ms", "rpe1_rot_deg",
              "rpe10_trans_m", "rpe10_trans_ms", "rpe10_rot_deg",
              "rpe30_trans_m", "rpe30_trans_ms", "rpe30_rot_deg",
              "fed", "traj_frames", "coverage", "longest_seg", "not_localised",
              "kf", "map_points", "map_lines",
              "po_calls", "po_with_lines", "line_edges", "line_inliers",
              "rescued", "gate_old_fail"]
    for r in rows:
        for k in fields:
            r.setdefault(k, "")

    # combined CSV
    all_csv = os.path.join(outdir, "all_runs.csv")
    with open(all_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields + ["eval_status"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # per (scene, mode) CSV
    groups = sorted({(r["scene"], r["mode"]) for r in rows})
    for scene, mode in groups:
        sub = [r for r in rows if r["scene"] == scene and r["mode"] == mode]
        p = os.path.join(outdir, f"{scene}_{mode}.csv")
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields + ["eval_status"])
            w.writeheader()
            for r in sub:
                w.writerow(r)

    # summary
    print("=" * 104)
    print(f"{'scene':<6}{'mode':<6}{'cond':<5}{'n':<4}{'ATE mean':<11}{'ATE std':<11}"
          f"{'cov':<8}{'kf':<6}{'POwLn':<8}{'lnEdges':<9}{'resc':<6}{'gateFail':<9}"
          f"{'lost':<6}{'wall':<6}")
    print("=" * 104)
    for scene, mode in groups:
        for cond in "ABCP":
            sub = [r for r in rows if r["scene"] == scene and r["mode"] == mode
                   and r["cond"] == cond]
            if not sub:
                continue
            # A group is only usable if it finished cleanly AND wrote its
            # trajectory. Timed-out / crashed groups stay in the CSV flagged with
            # complete=0 but never contribute to the numeric aggregates.
            n_tot = len(sub)
            # A group is usable only if it finished cleanly AND wrote its
            # trajectory. Runs flagged complete=0, or (legacy runs without the
            # flag) with a non-zero/absent exit code, are kept in the CSV but
            # never contribute to the numeric aggregates.
            def _usable(r):
                if r.get("complete", "") == "0":
                    return False
                if r.get("complete", "") == "" and str(r.get("exit", "")) not in ("0",):
                    return False
                return True
            sub = [r for r in sub if _usable(r)]
            if not sub:
                print(f"{scene:<6}{mode:<6}{cond:<5}{'0/' + str(n_tot):<10}"
                      f"ALL INCOMPLETE (excluded)")
                continue
            ate = [r["ate_rmse"] for r in sub if isinstance(r["ate_rmse"], float)]
            cov = [r["coverage"] for r in sub if isinstance(r["coverage"], float)]
            kf = [r["kf"] for r in sub if isinstance(r["kf"], int)]
            pol = [r["po_with_lines"] for r in sub]
            le = [r["line_edges"] for r in sub]
            resc = [r["rescued"] for r in sub]
            gf = [r["gate_old_fail"] for r in sub]
            nl = [r["not_localised"] for r in sub]
            wl = [float(r["wall_s"]) for r in sub if r["wall_s"] != ""]
            fm = f"{st.mean(ate):.6f}" if ate else "n/a"
            fs = f"{st.stdev(ate):.6f}" if len(ate) > 1 else ("0.000000" if ate else "n/a")
            print(f"{scene:<6}{mode:<6}{cond:<5}{len(sub):<4}{fm:<11}{fs:<11}"
                  f"{(st.mean(cov) if cov else 0):<8.3f}"
                  f"{(int(st.mean(kf)) if kf else 0):<6}"
                  f"{(int(st.mean(pol)) if pol else 0):<8}"
                  f"{(int(st.mean(le)) if le else 0):<9}"
                  f"{sum(resc):<6}{sum(gf):<9}"
                  f"{(int(st.mean(nl)) if nl else 0):<6}"
                  f"{(st.mean(wl) if wl else 0):<6.0f}"
                  + (f"  INCOMPLETE={n_tot - len(sub)}" if n_tot != len(sub) else ""))
    print()
    print("CSV:", all_csv)
    for scene, mode in groups:
        print("CSV:", os.path.join(outdir, f"{scene}_{mode}.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
