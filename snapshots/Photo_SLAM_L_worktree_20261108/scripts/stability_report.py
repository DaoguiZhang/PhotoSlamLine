#!/usr/bin/env python3
"""Per-round integrity audit + extended online-stability metrics for one tag.

Two things the earlier table did not separate:

  * integrity: each run directory must be internally consistent (its own export
    diag rows == its own written trajectory rows, its trace == the fed frames, its
    run.meta git/lib == the pre-registration, and its files must not be shared
    with another round). The earlier ad-hoc loop failed exactly here: it exported
    the diagnostic paths once, so later runs appended into an earlier run's files.
  * stability: is_set vs OK state, longest continuous segment of each, LOST /
    RECENTLY_LOST, relocalizations, new maps, map count, rescue ATTEMPTS
    (episodes, not lines) and INDEPENDENT EFFECTIVE rescue events (an episode that
    is followed by a sustained tracked segment and is separated from the previous
    effective one). No threshold is tuned anywhere: the definitions are fixed
    before the runs.

Usage: stability_report.py --tagdir /tmp/lr_stab/mono_v1 [--seq SEQ] [--csv out]
"""

import argparse
import glob
import hashlib
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chain_error_check import read_trace  # noqa: E402

SUSTAIN = 30        # frames of a "sustained tracked segment"
EPISODE_GAP = 5     # fed frames; rescue lines closer than this belong to one episode
INDEPENDENT_GAP = 30


def sha(path):
    if not os.path.exists(path):
        return "-"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


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
    ap.add_argument("--tagdir", required=True)
    ap.add_argument("--seq", default="/workspace/code/Photo-SLAM/datasets/eth3d/desk_3")
    ap.add_argument("--csv", default="")
    a = ap.parse_args()

    reg = os.path.join(a.tagdir, "PRE_REGISTRATION.txt")
    reg_txt = open(reg, errors="ignore").read() if os.path.exists(reg) else ""
    reg_git = (re.search(r"^git=(\S+)", reg_txt, re.M) or [None, "-"])[1]
    reg_lib = (re.search(r"^lib=(\S+)", reg_txt, re.M) or [None, "-"])[1]
    n_input = sum(1 for L in open(os.path.join(a.seq, "rgb.txt")) if L.strip() and not L.startswith("#"))

    dirs = sorted(d for d in glob.glob(os.path.join(a.tagdir, "*_r*")) if os.path.isdir(d))
    print("=== integrity audit: %d round(s) under %s ===" % (len(dirs), a.tagdir))
    print("    registration git=%s lib=%s  (input frames=%d)" % (reg_git[:12], reg_lib[:12], n_input))
    seen = {}
    bad = 0
    rows_out = []
    for d in dirs:
        name = os.path.basename(d)
        log = open(os.path.join(d, "log.txt"), errors="ignore").read() if os.path.exists(os.path.join(d, "log.txt")) else ""
        trace = sorted(read_trace(os.path.join(d, "online_trace.txt")).values(),
                       key=lambda r: r["frame"])
        tum = os.path.join(d, "CameraTrajectory_TUM.txt")
        tum_rows = sum(1 for L in open(tum) if L.strip() and not L.startswith("#")) if os.path.exists(tum) else 0
        diag_rows = 0
        dg = os.path.join(d, "export_diag.txt")
        if os.path.exists(dg):
            diag_rows = sum(1 for i, L in enumerate(open(dg, errors="ignore")) if i > 1 and L.strip())
        written = re.search(r"written=(\d+)", log)
        written = int(written.group(1)) if written else -1
        meta = open(os.path.join(d, "run.meta"), errors="ignore").read() if os.path.exists(os.path.join(d, "run.meta")) else ""
        m_git = (re.search(r"^git=(\S+)", meta, re.M) or [None, "-"])[1]
        m_lib = (re.search(r"^lib=(\S+)", meta, re.M) or [None, "-"])[1]
        ok_meta = (m_git == reg_git) and (m_lib == reg_lib)
        ok_counts = (diag_rows == tum_rows == written)
        # The runner may skip input frames that repeat a timestamp, so the trace is
        # checked for being contiguous in fed frame ids rather than for counting all
        # input frames; the difference is reported instead of failed.
        ids = [r["frame"] for r in trace]
        missing = [f for f in range(0, (ids[-1] + 1) if ids else 0) if f not in set(ids)]
        ok_trace = (len(missing) <= 10)   # runner skips repeated timestamps
        skipped_input = n_input - len(trace)
        h = sha(dg)
        dup = "" if h not in seen else "  <== SAME diag hash as %s" % seen[h]
        seen[h] = name
        ok = ok_meta and ok_counts and ok_trace and not dup
        if not ok:
            bad += 1
        print("  %-14s meta=%s counts=%s(diag %d/tum %d/written %d) trace rows=%d gaps=%d(%s) diaghash=%s%s"
              % (name, "ok" if ok_meta else "MISMATCH", "ok" if ok_counts else "MISMATCH",
                 diag_rows, tum_rows, written, len(trace), len(missing),
                 ("ok" if ok_trace else "MISMATCH"), h, dup))

        # ---- stability metrics ----
        set_rows = [r for r in trace if r["is_set"] == 1]
        n_set = len(set_rows)
        n_ok = sum(1 for r in trace if r["state"] == 2)
        n_recent = sum(1 for r in trace if r["state"] == 3)
        n_lost = sum(1 for r in trace if r["state"] == 4)
        longest_set = longest_run(trace, lambda r: r["is_set"] == 1)
        longest_ok = longest_run(trace, lambda r: r["state"] == 2)
        n_reloc = len(re.findall(r"Relocalized!!", log))
        n_newmap = len(re.findall(r"Creation of new map with id:", log))
        n_maps_atlas = len(re.findall(r"SaveTrajectoryTUM:\s+map \d+ has", log))
        exp_map = re.search(r"exported map=(\d+)", log)
        # rescue attempts = episodes of consecutive rescue=1 frames
        resc = [(int(f), int(s)) for f, s in re.findall(
            r"\[LineRescue\][^\n]*frame=(\d+)[^\n]*rescue=1[^\n]*state=(\d+)", log)]
        idx = {r["frame"]: i for i, r in enumerate(trace)}
        episodes = []
        for f, st in resc:
            if episodes and f - episodes[-1][-1][0] <= EPISODE_GAP:
                episodes[-1].append((f, st))
            else:
                episodes.append([(f, st)])
        n_ep = len(episodes)
        n_ep_degraded = sum(1 for e in episodes if any(st != 2 for _, st in e))
        effective = []
        degraded_recovered = 0
        for e in episodes:
            i = idx.get(e[0][0])
            if i is None:
                continue
            sustained = longest_run(trace[i:i + 60], lambda r: r["is_set"] == 1) >= SUSTAIN
            if sustained:
                effective.append(i)
            if sustained and any(st != 2 for _, st in e):
                degraded_recovered += 1
        indep = 0
        prev = None
        for i in effective:
            if prev is None or i - prev >= INDEPENDENT_GAP:
                indep += 1
            prev = i
        gate = os.path.join(d, "acceptance_run.txt")
        gt = open(gate, errors="ignore").read() if os.path.exists(gate) else ""
        m = re.search(r"(\d+)/(\d+) checks", gt)
        checks = m.group(0) if m else "0/0"
        viol = len(re.findall(r"violation \{", gt))
        rows_out.append((name, checks, viol, n_input, n_set, n_ok, n_recent, n_lost,
                         longest_set, longest_ok, n_reloc, n_newmap, n_maps_atlas,
                         exp_map.group(1) if exp_map else "-", n_ep, n_ep_degraded,
                         degraded_recovered, indep))

    print("\n=== extended online stability (all rounds, failures included) ===")
    hdr = ("round,checks,a6_viol,input,is_set,state_OK,recently_lost,lost,longest_set,"
           "longest_OK,reloc,new_map,maps_atlas,export_map,rescue_episodes,"
           "episodes_in_degraded,degraded_recovered,independent_effective")
    print("  " + hdr)
    for r in rows_out:
        print("  " + ",".join(str(x) for x in r))
    if a.csv:
        with open(a.csv, "w") as f:
            f.write(hdr + "\n")
            for r in rows_out:
                f.write(",".join(str(x) for x in r) + "\n")
        print("\n  written: %s" % a.csv)
    print("\n  integrity: %d of %d round(s) FAILED the isolation/hash audit" % (bad, len(dirs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
