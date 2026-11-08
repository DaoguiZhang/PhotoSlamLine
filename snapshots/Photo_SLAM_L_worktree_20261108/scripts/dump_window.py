#!/usr/bin/env python3
"""Dump a frame window of one round: online step vs exported step, plus the
reference keyframe identity and the final keyframe position the exporter used.

Steps are used rather than absolute positions because the exported trajectory is
the online one shifted by the constant origin Two, so raw positions are not
comparable while consecutive differences are.

usage: dump_window.py <run_dir> <seq_dir> <frame_lo> <frame_hi>
"""
import math
import sys


def main():
    run, seq, lo, hi = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    fed = []
    for line in open(seq + "/rgb.txt", errors="replace"):
        s = line.strip()
        if s and not s.startswith("#"):
            fed.append(float(s.split()[0]))
    idx = {round(t, 6): i for i, t in enumerate(fed)}

    tr, dg = {}, {}
    for line in open(run + "/online_trace.txt", errors="replace"):
        if line.startswith("#"):
            continue
        q = line.split()
        if len(q) < 29:
            continue
        k = idx.get(round(float(q[0]), 6))
        if k is not None:
            tr[k] = dict(ref=int(q[5]), mapid=int(q[6]), is_set=int(q[3]),
                         on=[float(q[8]), float(q[9]), float(q[10])])
    for line in open(run + "/export_diag.txt", errors="replace"):
        if line.startswith("#"):
            continue
        q = line.split()
        if len(q) < 15:
            continue
        k = idx.get(round(float(q[0]), 6))
        if k is not None:
            dg[k] = dict(ref0=int(q[1]), live=int(q[3]), hops=int(q[4]),
                         ex=[float(q[5]), float(q[6]), float(q[7])],
                         kffinal=[float(q[8]), float(q[9]), float(q[10])])

    def step(d, k, key):
        if k in d and (k - 1) in d:
            return math.dist(d[k][key], d[k - 1][key])
        return float("nan")

    print(f"window {lo}..{hi} of {run}")
    print(f"{'frame':>6} {'refKF':>6} {'liveKF':>6} {'hops':>4} {'online_step':>12} "
          f"{'export_step':>12} {'KFfinal_moved':>14}")
    for k in range(lo, hi + 1):
        if k not in tr or k not in dg:
            print(f"{k:>6}   {'no trace' if k not in tr else 'no diag (excluded from export)'}")
            continue
        kfm = (math.dist(dg[k]["kffinal"], dg[k - 1]["kffinal"])
               if (k - 1) in dg else float("nan"))
        print(f"{k:>6} {tr[k]['ref']:>6} {dg[k]['live']:>6} {dg[k]['hops']:>4} "
              f"{step(tr, k, 'on'):>12.4f} {step(dg, k, 'ex'):>12.4f} {kfm:>14.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
