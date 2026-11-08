#!/usr/bin/env python3
"""One CSV row of stability evidence for one run directory.

Deliberately prefers quantities that do NOT depend on the exported file being
good: tracking state, coverage, longest continuous tracked segment, loss/reset
events and line-rescue activity. The exported-file checks (A6) are reported
separately so that a smoothed export can never stand in for stability.

Usage:
  stability_table.py --dir DIR --seq SEQ --mode mono|rgbd --tag T --cond C \
                     --round R --exit-code N --gate N
"""

import argparse
import math
import os
import re
import sys


def read_trace(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            v = line.split()
            if len(v) != 29:
                continue
            rows.append({"frame": int(v[1]), "state": int(v[2]), "is_set": int(v[3]),
                         "refkf": int(v[5]), "refmap": int(v[6])})
    return rows


def longest_run(rows, pred):
    best = cur = 0
    for r in rows:
        if pred(r):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cond", required=True)
    ap.add_argument("--round", required=True)
    ap.add_argument("--exit-code", required=True)
    ap.add_argument("--gate", required=True)
    a = ap.parse_args()

    d = a.dir
    log = open(os.path.join(d, "log.txt"), errors="ignore").read() if os.path.exists(os.path.join(d, "log.txt")) else ""
    trace = read_trace(os.path.join(d, "online_trace.txt"))
    tum = os.path.join(d, "CameraTrajectory_TUM.txt")
    written = sum(1 for L in open(tum) if L.strip() and not L.startswith("#")) if os.path.exists(tum) else 0
    n_input = 0
    rgb = os.path.join(a.seq, "rgb.txt")
    if os.path.exists(rgb):
        n_input = sum(1 for L in open(rgb) if L.strip() and not L.startswith("#"))

    m = re.search(r"SaveTrajectoryTUM: exported map=(-?\d+) refKFs=\d+ written=(\d+)"
                  r" skipped_lost=(\d+) skipped_other_map=(\d+) skipped_no_refkf=(\d+)"
                  r" \(of (\d+) frame entries\)", log)
    n_entries = int(m.group(6)) if m else 0

    checks_ok = n_checks = 0
    acc_txt = ""
    acc = os.path.join(d, "acceptance_run.txt")
    if os.path.exists(acc):
        acc_txt = open(acc, errors="ignore").read()
        mc = re.search(r"(\d+)/(\d+) checks", acc_txt)
        if mc:
            checks_ok, n_checks = int(mc.group(1)), int(mc.group(2))
    a6 = re.findall(r"violation \{[^}]*'frame_id0': (\d+), 'frame_id1': (\d+),[^}]*'est_disp': ([0-9.]+)", acc_txt)
    a6_max = max((float(x[2]) for x in a6), default=0.0)
    wall = "?"
    meta = os.path.join(d, "run.meta")
    if os.path.exists(meta):
        mw = re.search(r"wall_s=(\d+)", open(meta, errors="ignore").read())
        if mw:
            wall = mw.group(1)

    set_frames = sum(1 for r in trace if r["is_set"] == 1)
    n_lost = sum(1 for r in trace if r["state"] == 4)
    n_recent = sum(1 for r in trace if r["state"] == 3)
    longest = longest_run(trace, lambda r: r["is_set"] == 1)

    n_reloc = len(re.findall(r"Relocalized!!", log))
    n_newmap = len(re.findall(r"Creation of new map with id:", log))
    maps_exported = 0
    me = re.search(r"exported map=(-?\d+)", log)
    if me:
        maps_exported = len(re.findall(r"SaveTrajectoryTUM:\s+map \d+ has", log))

    n_rescue = len(re.findall(r"\[LineRescue\][^\n]*rescue=1", log))
    # A rescue matters only if it intervened while tracking was NOT healthy, and it
    # is "productive" only if a sustained tracked segment follows it. Routine
    # rescues on already-healthy frames are counted but not credited.
    crisis = re.findall(r"\[LineRescue\][^\n]*frame=(\d+)[^\n]*rescue=1[^\n]*state=(\d+)", log)
    n_rescue_crisis = sum(1 for _, st in crisis if int(st) != 2)
    # "productive" rescue: the frame id of the rescue is followed by >=30 consecutive
    # frames that entered the trajectory lists.
    idx = {r["frame"]: i for i, r in enumerate(trace)}
    productive = 0
    for f, st in crisis:
        if int(st) == 2:
            continue
        i = idx.get(int(f))
        if i is None:
            continue
        if longest_run(trace[i:i + 60], lambda r: r["is_set"] == 1) >= 30:
            productive += 1

    audit = os.path.join(d, "audit.txt")
    carries = merges = 0
    if os.path.exists(audit):
        carries = sum(1 for L in open(audit, errors="ignore") if " CARRY " in L)
        merges = sum(1 for L in open(audit, errors="ignore") if " MERGE_EVENT " in L)

    coverage = (100.0 * set_frames / n_input) if n_input else float("nan")
    print("%s,%s,%s,%s,%s,%s,%d/%d,%d,%s,%d,%d,%d,%.1f,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.3f" % (
        a.tag, a.mode, a.cond, a.round, a.exit_code, "PASS" if int(a.gate) == 0 else "FAIL",
        checks_ok, n_checks, 1 if (int(a.exit_code) == 0 and n_entries > 0 and written > 0) else 0,
        wall, written, n_input, set_frames, coverage, longest,
        n_lost, n_recent, n_reloc, n_newmap, n_rescue_crisis, productive, merges, carries,
        len(a6), a6_max))
    return 0


if __name__ == "__main__":
    sys.exit(main())
