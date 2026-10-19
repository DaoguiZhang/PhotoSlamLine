#!/usr/bin/env python3
"""Rebuild one run's per-frame truth from its raw log and trajectory, and verify
structurally that the coverage numerator really is localized frames.

What can and cannot be checked, stated first because it bounds every claim below:

  * mlbLost itself is NOT written to the log, so it cannot be read back. What is
    checkable is its documented consequence in System::SaveTrajectoryTUM: a line
    is written only when lbL is false, and Tracking appends a COPY of the
    previous (timestamp, pose) for every frame with mCurrentFrame.isSet() false.
    So the file is [localized frames] interleaved with [stale duplicates], and the
    number of DISTINCT timestamps equals the number of frames that produced a new
    pose. The script verifies the properties that this must imply:
       - every written timestamp is a member of the fed frame list
       - distinct timestamps appear in strictly increasing order
       - every duplicate line is an exact copy of its immediate predecessor
    A violation of any of these would invalidate the reading and is reported.

Outputs (into --outdir, never over the run's own files):
  <run>_frames.csv    frame_id, timestamp, localized, dup_lines, log_records
  <run>_traj_dedup.txt
"""
import os
import re
import sys

RX_FRAME = re.compile(r"frame=(\d+)")
RX_SCAN = re.compile(r"\[LineScan\] stage=(\S+) map=(-?\d+) frame=(\d+)"
                     r" ptBeforePO=(\d+) lnBeforePO=(\d+) ptAfterPO=(\d+)")
RX_LINEDIAG = re.compile(r"\[LineEdgeDiag\] PO frame=(\d+) NL=(\d+)")
RX_FOOTRACK = re.compile(r"Fail to track local map!|Fail to track local map")


def fed_list(path):
    ts = []
    for line in open(path, errors="replace"):
        s = line.strip()
        if s and not s.startswith("#"):
            ts.append(float(s.split()[0]))
    seen, out = set(), []
    for t in ts:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def main():
    rundir = sys.argv[1] if len(sys.argv) > 1 else "."
    fed_file = None
    outdir = os.path.join(rundir, "frame_table")
    for i, a in enumerate(sys.argv):
        if a == "--fed":
            fed_file = sys.argv[i + 1]
        if a == "--outdir":
            outdir = sys.argv[i + 1]
    meta = {}
    mp = os.path.join(rundir, "run.meta")
    if os.path.exists(mp):
        for line in open(mp, errors="replace"):
            if "=" in line:
                k, v = line.strip().split("=", 1)
                meta[k] = v
    seq = meta.get("sequence", "")
    if not fed_file:
        for cand in (os.path.join(seq, "rgb.txt"), meta.get("association", "")):
            if cand and os.path.exists(cand):
                fed_file = cand
                break
    fed = fed_list(fed_file)
    os.makedirs(outdir, exist_ok=True)
    name = os.path.basename(rundir.rstrip("/"))

    # ---- trajectory: raw lines and the implied duplicate structure ---------
    tpath = os.path.join(rundir, "CameraTrajectory_TUM.txt")
    raw_lines, raw_ts = [], []
    for line in open(tpath, errors="replace"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = s.split()
        if len(p) < 8:
            continue
        raw_lines.append(" ".join(p))
        raw_ts.append(float(p[0]))

    dup_counts = {}
    for t in raw_ts:
        dup_counts[t] = dup_counts.get(t, 0) + 1
    distinct = []
    for t in raw_ts:
        if t not in distinct:
            distinct.append(t)
    monotone = all(distinct[i] < distinct[i + 1] for i in range(len(distinct) - 1))
    nonadjacent_dup = 0
    for i in range(1, len(raw_lines)):
        if raw_ts[i] == raw_ts[i - 1] and raw_lines[i] != raw_lines[i - 1]:
            nonadjacent_dup += 1
    in_fed = sum(1 for t in distinct if any(abs(t - e) <= 0.01 for e in fed))

    dedup_path = os.path.join(outdir, name + "_traj_dedup.txt")
    seen = set()
    with open(dedup_path, "w") as f:
        for ln, t in zip(raw_lines, raw_ts):
            if t in seen:
                continue
            seen.add(t)
            f.write(ln + "\n")

    # ---- per-frame log evidence -------------------------------------------
    recs, ptmax = {}, {}
    n_footrack = 0
    log = os.path.join(rundir, "log.txt")
    if os.path.exists(log):
        for line in open(log, errors="replace"):
            if RX_FOOTRACK.search(line):
                n_footrack += 1
                continue
            m = RX_SCAN.search(line)
            if m:
                f = int(m.group(3))
                recs[f] = recs.get(f, 0) + 1
                v = int(m.group(6))
                if v > ptmax.get(f, -1):
                    ptmax[f] = v
                continue
            m = RX_LINEDIAG.search(line)
            if m:
                f = int(m.group(1))
                recs[f] = recs.get(f, 0) + 1

    # ---- per-frame table ---------------------------------------------------
    localized_set = set()
    for t in distinct:
        for i, e in enumerate(fed):
            if abs(t - e) <= 0.01:
                localized_set.add(i)
                break
    csv_path = os.path.join(outdir, name + "_frames.csv")
    with open(csv_path, "w") as f:
        f.write("frame_id,timestamp,localized,dup_lines,log_records,ptAfterPO_max\n")
        for i, e in enumerate(fed):
            loc = 1 if i in localized_set else 0
            f.write(f"{i},{e:.9f},{loc},0,{recs.get(i, 0)},{ptmax.get(i, -1)}\n")

    seg = cur = 0
    for i in range(len(fed)):
        cur = cur + 1 if i in localized_set else 0
        seg = max(seg, cur)

    print(f"RUN {name}  fed={len(fed)}  traj_lines={len(raw_lines)}  "
          f"distinct_ts={len(distinct)}  dup_lines={len(raw_lines) - len(distinct)}")
    print(f"  coverage = {len(distinct)}/{len(fed)} = {len(distinct)/max(1,len(fed)):.4f}"
          f"   longest_segment = {seg} frames")
    print("  mlbLost structural verification (consequences, not the flag itself):")
    print(f"    all distinct timestamps are fed frames : {in_fed}/{len(distinct)}"
          f"  {'OK' if in_fed == len(distinct) else 'VIOLATION'}")
    print(f"    distinct timestamps strictly increasing: {monotone}"
          f"  {'OK' if monotone else 'VIOLATION'}")
    print(f"    duplicate lines equal predecessor      : "
          f"{len(raw_lines)-len(distinct)-nonadjacent_dup}/{len(raw_lines)-len(distinct)}"
          f"  {'OK' if nonadjacent_dup == 0 else 'VIOLATION'}")
    print(f"    frames with a log record={len(recs)}  localized={len(distinct)}  "
          f"fed-lacking-pose={len(fed)-len(distinct)}")
    print(f"    (\"Fail to track local map!\" occurrences in log = {n_footrack}; that "
          f"message carries no frame id, so it cannot be attributed per frame)")
    print(f"  wrote {csv_path}")
    print(f"  wrote {dedup_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
