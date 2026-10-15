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
                       r".*oldPtVerdict=(\S+) new=(\S+) reason=(-?\d+)"
                       r" rescueLnUniq=(\d+) rescueLnEdges=(\d+)"
                       r" gateMmOldFail=(-?\d+) gateTrkRefOldFail=(-?\d+) rescue=(\d+)")
RX_META = re.compile(r"(\w+)=(.*)")
RX_FRAME = re.compile(r"frame=(\d+)")


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
    outdir = None
    if "--fed" in sys.argv:
        fed_file = sys.argv[sys.argv.index("--fed") + 1]
    if "--gt-dir" in sys.argv:
        gt_dir = sys.argv[sys.argv.index("--gt-dir") + 1]
    if "--outdir" in sys.argv:
        outdir = sys.argv[sys.argv.index("--outdir") + 1]
    if outdir is None:
        outdir = os.path.join(root, "report_fixed")
    os.makedirs(outdir, exist_ok=True)
    fed = fed_ts(fed_file)

    def first_meta_value(key):
        for d0 in sorted(glob.glob(os.path.join(root, "drop*", "*"))):
            mp0 = os.path.join(d0, "run.meta")
            if os.path.exists(mp0):
                for line in open(mp0, errors="replace"):
                    mm = RX_META.match(line.strip())
                    if mm and mm.group(1) == key:
                        return mm.group(2)
        return ""

    # Defaults, so the caller cannot silently get empty coverage/ATE columns by
    # forgetting an option.
    seq0 = first_meta_value("sequence")
    if not fed and seq0:
        for cand in (os.path.join(seq0, "rgb.txt"), os.path.join(seq0, "associated.txt")):
            if os.path.exists(cand):
                fed = fed_ts(cand)
                break
    if not gt_dir and seq0 and os.path.isdir(seq0):
        gt_dir = seq0
    print(f"# fed frames: {len(fed)} (unique {len(set(fed))})")
    print(f"# ground truth dir: {gt_dir}")
    print(f"# output dir: {outdir}")

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

        scans = []
        pt, uniq, spread = [], [], []
        frames_seen = set()
        old_fail_frames = set()
        rescue_on_records = 0
        rescue_off_records = 0
        rescue_attempt_frames = set()
        rescue_pass_frames = set()
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
                    f = int(g[2])
                    frames_seen.add(f)
                    if g[7] == "FAIL":
                        old_fail_frames.add(f)
                    continue
                mr = RX_RESCUE.search(line)
                if mr:
                    g = mr.groups()
                    f = int(g[2])
                    armed = (g[10] == "1")
                    if armed:
                        rescue_on_records += 1
                    else:
                        rescue_off_records += 1
                    # Three separate notions, deliberately not collapsed:
                    #   old_fail_frames        old point-only verdict was FAIL
                    #   rescue_attempt_frames   FAIL *and* the switch was armed
                    #   rescue_pass_frames      FAIL but the new verdict was OK
                    if g[3] == "FAIL" and armed:
                        rescue_attempt_frames.add(f)
                    if g[3] == "FAIL" and g[4] == "OK":
                        rescue_pass_frames.add(f)

        traj_path = os.path.join(d, "CameraTrajectory_TUM.txt")
        ts_raw = traj_ts(traj_path)
        # System::SaveTrajectoryTUM appends a COPY of the previous entry -- the
        # timestamp included -- whenever mCurrentFrame.isSet() is false (the
        # "This can happen if tracking is lost" branch in Tracking::Track and
        # Tracking::TrackWithLine). Consecutive such frames therefore repeat one
        # timestamp many times (observed: 82 identical lines), so the raw line
        # count is NOT the number of localized frames. The honest per-frame
        # count is the number of DISTINCT timestamps; the first occurrence of a
        # timestamp is the real entry and the rest are stale duplicates.
        seen_ts = set()
        ts = []
        for t_ in ts_raw:
            if t_ not in seen_ts:
                seen_ts.add(t_)
                ts.append(t_)
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

        last_frame_log = 0
        if os.path.exists(log):
            for line in open(log, errors="replace"):
                for mt in RX_FRAME.finditer(line):
                    v = int(mt.group(1))
                    if v > last_frame_log:
                        last_frame_log = v

        row = dict(level=level, cond=cond, scenemode=os.path.basename(d),
                   line_mode=meta.get("line_mode", ""),
                   rescue_switch=meta.get("rescue", ""),
                   exit=meta.get("exit", ""), wall_s=meta.get("wall_s", ""),
                   complete=meta.get("complete", ""),
                   drop=meta.get("stress_drop_point_ratio", ""),
                   start=meta.get("stress_start_frame", ""),
                   fed_total=len(fed), fed_unique=len(set(fed)),
                   frames_seen=len(frames_seen),
                   last_frame_log=last_frame_log,
                   last_frame_meta=meta.get("last_frame", ""),
                   decisions=len(scans),
                   ptAfterPO_p50=q(pt, 50), ptAfterPO_min=min(pt) if pt else "",
                   lnUniq_p50=q(uniq, 50), lnUniq_max=max(uniq) if uniq else "",
                   lnUniq_ge8=sum(1 for u in uniq if u >= 8),
                   spread_p50=round(q(spread, 50), 1),
                   spread_max=round(max(spread), 1) if spread else "",
                   old_fail_frames=len(old_fail_frames),
                   rescue_attempt_frames=len(rescue_attempt_frames),
                   rescue_pass_frames=len(rescue_pass_frames),
                   rescue_records_on=rescue_on_records,
                   rescue_records_off=rescue_off_records,
                   traj_lines=len(ts_raw),
                   traj_unique_ts=len(ts),
                   traj_dup_lines=len(ts_raw) - len(ts),
                   coverage=round(cov, 4),
                   longest_seg=longest,
                   ate="", sigma="", rpe1="", gt_matches="")
        if gt_dir and len(ts) >= 3:
            gt = None
            for cand in sorted(glob.glob(os.path.join(gt_dir, "*groundtruth*"))) or \
                        sorted(glob.glob(os.path.join(gt_dir, "gt*.txt"))):
                gt = cand
                break
            if gt:
                # Evaluate the DE-DUPLICATED trajectory. The raw file repeats the
                # previous (timestamp, pose) once per unset frame, which would
                # otherwise over-weight those instants in the association.
                #
                # The artefact name must contain EVERY distinguishing key. The
                # run-directory basename alone is not unique across stress
                # levels (drop30/50/70/90 all contain e.g. desk_3_rgbd_B), so
                # level-less names let later levels overwrite earlier ones in
                # the same outdir. Prefix the level directory.
                tag = os.path.basename(os.path.dirname(d)) + "_" + os.path.basename(d)
                dedup_path = os.path.join(outdir, tag + "_traj_dedup.txt")
                seen_w = set()
                with open(traj_path) as fin, open(dedup_path, "w") as fout:
                    for line in fin:
                        s = line.strip()
                        if not s or s.startswith("#"):
                            continue
                        p = s.split()
                        if len(p) < 8:
                            continue
                        k = float(p[0])
                        if k in seen_w:
                            continue
                        seen_w.add(k)
                        fout.write(" ".join(p) + "\n")
                prefix = os.path.join(outdir, tag + "_ate")
                subprocess.run([sys.executable, EVAL, gt, dedup_path,
                                prefix, "--mode", "se3", "--rpe-deltas", "1"],
                               check=False, capture_output=True)
                if os.path.exists(prefix + ".json"):
                    with open(prefix + ".json") as f:
                        r = json.load(f)
                    if r.get("status") == "OK":
                        row["ate"] = round(r["ate_se3"]["rmse"], 6)
                        row["sigma"] = round(r["ate_se3"]["std"], 6)
                        row["gt_matches"] = r.get("n_pairs", "")
                        v = r["rpe"].get("1")
                        if v:
                            row["rpe1"] = round(v["trans_rms_m"], 6)
        rows.append(row)

    if not rows:
        print("no ladder runs found under", root)
        return 1
    keys = list(rows[0].keys())
    out = os.path.join(outdir, "ladder_fixed.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("=" * 150)
    print("POINT-DROP LADDER - per run, all levels reported, none selected as best")
    print("=" * 150)
    hdr = (f"{'drop':<5}{'cond':<5}{'lm':<3}{'rs':<3}{'exit':<5}{'cmp':<4}"
           f"{'fed':<6}{'last':<6}{'seen':<6}{'trajln':<7}{'uniqts':<7}"
           f"{'cov':<7}{'longest':<8}{'oldFAIL':<8}{'resAtt':<7}{'resPass':<8}"
           f"{'ATE':<10}{'RPE1':<9}{'gtN':<6}")
    print(hdr)
    for r in sorted(rows, key=lambda x: (x["level"], x["cond"])):
        print(f"{str(r['level'])+'%':<5}{r['cond']:<5}{str(r['line_mode']):<3}"
              f"{str(r['rescue_switch']):<3}{str(r['exit']):<5}{str(r['complete']):<4}"
              f"{r['fed_total']:<6}{r['last_frame_log']:<6}{r['frames_seen']:<6}"
              f"{r['traj_lines']:<7}{r['traj_unique_ts']:<7}"
              f"{r['coverage']:<7}{r['longest_seg']:<8}{r['old_fail_frames']:<8}"
              f"{r['rescue_attempt_frames']:<7}{r['rescue_pass_frames']:<8}"
              f"{(r['ate'] if r['ate'] != '' else '-'):<10}"
              f"{(r['rpe1'] if r['rpe1'] != '' else '-'):<9}{r['gt_matches']:<6}")
    print()
    print("drop/cond = level and line condition (P: lm=1 no line edges; B: lm=2 rescue off;")
    print("            C: lm=2 rescue on). lm/rs = run.meta line_mode / rescue switch.")
    print("fed    = input frames in the fed list (unique).  last = max frame= in the log.")
    print("seen   = distinct frame ids appearing in [LineScan]/[LineRescue] records.")
    print("trajln = raw lines in CameraTrajectory_TUM.txt; uniqts = DISTINCT timestamps.")
    print("         SaveTrajectoryTUM appends a copy of the previous (timestamp, pose) for")
    print("         every frame with mCurrentFrame unset, so trajln over-states coverage;")
    print("         coverage and longest_seg are computed on DISTINCT timestamps only.")
    print("oldFAIL= distinct frames whose old point-only verdict was FAIL.")
    print("resAtt = distinct frames with FAIL and the rescue switch ARMED (attempted).")
    print("resPass= distinct frames with FAIL but new verdict OK (passed). These are NOT")
    print("         yet validated: see the per-frame audit before calling them rescues.")
    print("ATE/RPE1 = SE(3) ATE RMSE on the de-duplicated trajectory; gtN = associated pairs.")
    print("CSV:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
