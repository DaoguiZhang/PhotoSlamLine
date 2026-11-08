#!/usr/bin/env python3
"""Summarise the line-rescue / line-gate / line-edge diagnostic streams.

Parses stdout/stderr captured from a Photo-SLAM-L run and answers, with
PER-FRAME evidence:

  1. [LineGate]      at the ORIGINAL early-return sites, how many frames would
                     the OLD point-only control flow have failed on, and were
                     they bypassed to the line search?
  2. [LineRescue]    of the frames that reached a joint point+line verdict, how
                     many had oldPtVerdict=FAIL and were kept alive (new=OK)?
  3. [LineEdgeDiag]  emitted by Optimizer::PoseOptimizationWithLine for EVERY
                     call.  How many calls actually carried line edges, and how
                     many unique MapLines / line inliers were involved?  This is
                     the ONLY honest source for "did lines participate in pose
                     optimisation" -- never the rescue-only counters.

Usage:
    python3 scripts/line_diag_report.py <log1> [log2 ...] [--label NAME]
"""
import sys
import re
from collections import defaultdict, Counter

RE_GATE = re.compile(
    r"\[LineGate\] site=(\S+) frame=(\d+) ptMatches=(\d+)"
    r"(?: lnMatches=(\d+))? old_would_fail=(\d+)(?: (?:bypass|rescueEnabled)=(\d+))?")
RE_RESCUE = re.compile(
    r"\[LineRescue\] stage=(\S+) map=(-?\d+) frame=(\d+)"
    r" ptBeforePO=(\d+) lnBeforePO=(\d+) ptAfterPO=(\d+) lnAfterPO=(\d+)"
    r" oldPtVerdict=(\S+) new=(\S+) reason=(-?\d+)"
    r" rescueLnUniq=(\d+) rescueLnEdges=(\d+)"
    r" gateMmOldFail=(-?\d+) gateTrkRefOldFail=(-?\d+) rescue=(\d+) state=(-?\d+)")
RE_EDGE = re.compile(
    r"\[LineEdgeDiag\] PO frame=(\d+) NL=(\d+) lnTrackedNonNull=(\d+)"
    r" ptEdges=(\d+) ptIn=(\d+) ptOut=(\d+) ptChi2=(\S+)"
    r" lnEdges=(\d+) lnEdgesMono=(\d+) lnEdgesStereo=(\d+)"
    r" lnIn=(\d+) lnOut=(\d+) lnUniq=(\d+)"
    r"(?: lnSpread=(\S+))? lnChi2=(\S+)")


def pct(vals, p):
    if not vals:
        return 0
    vals = sorted(vals)
    k = int(p / 100.0 * (len(vals) - 1))
    return vals[k]


def summarise(path, label):
    gates = defaultdict(list)      # site -> [(frame, ptMatches, oldfail, bypass)]
    rescues = []
    edges = []
    with open(path, errors="replace") as f:
        for line in f:
            m = RE_GATE.search(line)
            if m:
                site, frame, pt = m.group(1), int(m.group(2)), int(m.group(3))
                ln = int(m.group(4)) if m.group(4) else -1
                of = int(m.group(5))
                by = int(m.group(6)) if m.group(6) else -1
                gates[site].append((frame, pt, ln, of, by))
                continue
            m = RE_RESCUE.search(line)
            if m:
                g = m.groups()
                rescues.append(dict(stage=g[0], frame=int(g[2]),
                                    ptBefore=int(g[3]), lnBefore=int(g[4]),
                                    ptAfter=int(g[5]), lnAfter=int(g[6]),
                                    old=g[7], new=g[8], reason=int(g[9]),
                                    rescueUniq=int(g[10]), rescueEdges=int(g[11]),
                                    gateMm=int(g[12]), gateTrkRef=int(g[13]),
                                    state=int(g[15])))
                continue
            m = RE_EDGE.search(line)
            if m:
                g = m.groups()
                edges.append(dict(frame=int(g[0]), NL=int(g[1]),
                                  nonNull=int(g[2]), ptEdges=int(g[3]),
                                  ptIn=int(g[4]), ptOut=int(g[5]),
                                  lnEdges=int(g[7]), lnMono=int(g[8]),
                                  lnStereo=int(g[9]), lnIn=int(g[10]),
                                  lnOut=int(g[11]), lnUniq=int(g[12])))

    print("=" * 78)
    print(f"LOG: {path}   ({label})")
    print("=" * 78)

    print("\n[1] LineGate (evaluated BEFORE any bypass)")
    if not gates:
        print("    no [LineGate] lines (was PHOTO_SLAM_DEBUG_LINE_GATE=1 set?)")
    for site, rows in sorted(gates.items()):
        fail = [r for r in rows if r[3] == 1]
        # NOTE: fields[4] is the *enable* flag (rescueEnabled), NOT an actual
        # bypass. A real bypass needs old_would_fail==1 AND rescueEnabled==1.
        bypassed = [r for r in fail if r[4] == 1]
        print(f"    site={site}: evaluations={len(rows)}  "
              f"old_would_fail={len(fail)}  of_those_bypassed_to_line_search={len(bypassed)}")
        if fail:
            pts = [r[1] for r in fail]
            print(f"        ptMatches on old-fail frames: min={min(pts)} "
                  f"median={pct(pts,50)} max={max(pts)}")
            print(f"        frames: {[r[0] for r in fail][:20]}"
                  f"{' ...' if len(fail) > 20 else ''}")

    print("\n[2] LineRescue (post-optimisation joint verdict)")
    if not rescues:
        print("    no [LineRescue] lines")
    else:
        c = Counter((r["stage"], r["old"]) for r in rescues)
        for (stage, old), n in sorted(c.items()):
            print(f"    stage={stage:<26} oldPtVerdict={old:<5} n={n}")
        kept = [r for r in rescues if r["old"] == "FAIL" and r["new"] == "OK"]
        lost = [r for r in rescues if r["old"] == "OK" and r["new"] == "FAIL"]
        print(f"    RESCUED (old=FAIL -> new=OK): {len(kept)}   "
              f"BROKEN (old=OK -> new=FAIL): {len(lost)}   total={len(rescues)}")
        for r in kept[:20]:
            print(f"      rescued frame={r['frame']} stage={r['stage']} "
                  f"ptAfter={r['ptAfter']} lnAfter={r['lnAfter']} "
                  f"rescueUniq={r['rescueUniq']} reason={r['reason']}")
        reasons = Counter(r["reason"] for r in rescues)
        print(f"    reason histogram: {dict(sorted(reasons.items()))}")

    print("\n[3] LineEdgeDiag (EVERY PoseOptimizationWithLine call)")
    if not edges:
        print("    no [LineEdgeDiag] lines "
              "(was PHOTO_SLAM_DEBUG_LINE_EDGES=1 set?)")
    else:
        ne = [e["lnEdges"] for e in edges]
        nz = [e for e in edges if e["lnEdges"] > 0]
        print(f"    calls={len(edges)}  calls_with_line_edges={len(nz)} "
              f"({100.0*len(nz)/len(edges):.1f}%)")
        print(f"    lnEdges  min={min(ne)} median={pct(ne,50)} "
              f"p90={pct(ne,90)} max={max(ne)}")
        print(f"    lnUniq   min={min(e['lnUniq'] for e in edges)} "
              f"median={pct([e['lnUniq'] for e in edges],50)} "
              f"max={max(e['lnUniq'] for e in edges)}")
        if nz:
            print(f"    frames WITH line edges: "
                  f"{[e['frame'] for e in nz][:30]}{' ...' if len(nz) > 30 else ''}")
            tot_in = sum(e["lnIn"] for e in nz)
            tot_ed = sum(e["lnEdges"] for e in nz)
            print(f"    aggregate line inliers/edges over those frames: "
                  f"{tot_in}/{tot_ed}")
        else:
            print("    >>> NO frame handed a line edge to pose optimisation <<<")
    print()


def main():
    args = [a for a in sys.argv[1:]]
    label = ""
    if "--label" in args:
        i = args.index("--label")
        label = args[i + 1]
        del args[i:i + 2]
    if not args:
        print(__doc__)
        return 1
    for p in args:
        summarise(p, label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
