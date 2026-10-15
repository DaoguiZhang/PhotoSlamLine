#!/usr/bin/env python3
"""Classify every A6 (adjacent-input-frame jump) violation of a run directory.

Built after the mono_v1 matrix showed A6 still failing in P and B as well as C,
i.e. the residual class is not rescue-specific. For each violating pair this
prints, from that run's OWN artifacts:

  export step, the ONLINE step at the same timestamps, hops/liveKF/refKF of both
  rows, and which map the reference keyframe belonged to.

Classes:
  online          the estimator's own trajectory jumps there (export follows it)
  same-anchor     both rows use the SAME live anchor with hops=0 - the culling
                  chain cannot explain it at all
  anchor-switch   the live anchor or the hop count changes between the rows
  no-diag-row     the frame is not in the export diag (filtered out)

Usage: classify_a6.py --dir DIR
"""

import argparse
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chain_error_check import read_diag, read_trace, dist  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    d = a.dir

    diag_path = os.path.join(d, "export_diag.txt")
    trace_path = os.path.join(d, "online_trace.txt")
    tum_path = os.path.join(d, "CameraTrajectory_TUM.txt")
    diag = {}
    if os.path.exists(diag_path):
        _, rows = read_diag(diag_path)
        diag = {round(r["ts"], 6): r for r in rows}
    tr = read_trace(trace_path) if os.path.exists(trace_path) else {}
    if not os.path.exists(tum_path):
        print("%s: no trajectory" % d)
        return 0

    seq = []
    for L in open(tum_path):
        if L.startswith("#") or not L.strip():
            continue
        v = L.split()
        seq.append((float(v[0]), (float(v[1]), float(v[2]), float(v[3]))))

    counts = {}
    details = []
    for (t0, p0), (t1, p1) in zip(seq, seq[1:]):
        if t1 - t0 > 0.05:
            continue
        de = dist(p0, p1)
        if de <= a.threshold:
            continue
        r0, r1 = diag.get(round(t0, 6)), diag.get(round(t1, 6))
        o0, o1 = tr.get(round(t0, 6)), tr.get(round(t1, 6))
        on = dist(o0["Tcw"][0], o1["Tcw"][0]) if (o0 and o1) else float("nan")
        if not (r0 and r1):
            cls = "no-diag-row"
        elif on == on and on > a.threshold:
            cls = "online"
        elif r0["hops"] == 0 and r1["hops"] == 0 and r0["livekf"] == r1["livekf"]:
            cls = "same-anchor"
        elif (r0["livekf"] != r1["livekf"]) or (r0["hops"] != r1["hops"]):
            cls = "anchor-switch"
        else:
            cls = "other"
        counts[cls] = counts.get(cls, 0) + 1
        details.append((t0, t1, de, on, r0, r1, cls, o0, o1))

    name = os.path.basename(d.rstrip("/"))
    if not details:
        print("%-22s A6 clean" % name)
        return 0
    print("%-22s %d violation(s): %s" % (name, len(details),
          ", ".join("%s=%d" % kv for kv in sorted(counts.items()))))
    if a.quiet:
        return 0
    for t0, t1, de, on, r0, r1, cls, o0, o1 in details[:6]:
        print("    %.6f->%.6f export=%.3f online=%.3f [%s]" % (t0, t1, de, on, cls))
        for tag, r, o in (("A", r0, o0), ("B", r1, o1)):
            if r:
                print("       %s refKF=%-4d(refmap=%d) liveKF=%-4d hops=%d state=%s set=%s"
                      % (tag, r["refkf"], r["refmap"], r["livekf"], r["hops"],
                         o["state"] if o else "?", o["is_set"] if o else "?"))
            else:
                print("       %s (not exported / not in trace)" % tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
