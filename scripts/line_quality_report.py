#!/usr/bin/env python3
"""Line-quality report from the [LineEdgeDiag] stream.

Answers the "are the lines entering pose optimisation trustworthy?" question with
measurements rather than assumptions:

  * lnSpread  -- directional coverage (deg) of the line edges actually used in
                 each PoseOptimizationWithLine call.  Low spread => near-parallel
                 / degenerate set.  Reported so degeneracy is MEASURED.
  * lnOut     -- line edges the robust kernel rejected as outliers
                 (chi2 > chi2LineMono = 25 for mono, 35.345 for stereo).
  * all-outlier calls -- calls where every line edge was rejected; these are the
                 direct evidence of ERRONEOUS lines actually used.
  * lnChi2    -- summed robust chi2 of the ACCEPTED line edges.
  * lnEdges == 0 -- calls with no line edge at all (point-only fallback inside
                 the motion model when fewer than 10 lines matched).

Usage: python3 scripts/line_quality_report.py <log> [<log> ...]
"""
import sys
import re
import statistics as st

RE = re.compile(
    r"\[LineEdgeDiag\] PO frame=(\d+) NL=(\d+) lnTrackedNonNull=(\d+)"
    r" ptEdges=(\d+) ptIn=(\d+) ptOut=(\d+) ptChi2=(\S+)"
    r" lnEdges=(\d+) lnEdgesMono=(\d+) lnEdgesStereo=(\d+)"
    r" lnIn=(\d+) lnOut=(\d+) lnUniq=(\d+)"
    r"(?: lnSpread=(\S+))? lnChi2=(\S+)")


def summarise(path):
    rows = []
    for line in open(path, errors="replace"):
        m = RE.search(line)
        if not m:
            continue
        g = m.groups()
        rows.append(dict(frame=int(g[0]), NL=int(g[1]), nonNull=int(g[2]),
                         ptEdges=int(g[3]),
                         lnEdges=int(g[7]), lnMono=int(g[8]), lnStereo=int(g[9]),
                         lnIn=int(g[10]), lnOut=int(g[11]), lnUniq=int(g[12]),
                         spread=float(g[13]) if g[13] else None,
                         chi2=float(g[14])))
    print("=" * 78)
    print(f"LINE QUALITY: {path}")
    print("=" * 78)
    print(f"PO calls parsed: {len(rows)}")
    if not rows:
        return
    used = [r for r in rows if r["lnEdges"] > 0]
    print(f"calls WITH line edges: {len(used)} "
          f"({100.0*len(used)/len(rows):.1f}%)")
    print(f"calls with NO line edge (point-only fallback): {len(rows)-len(used)}")
    if not used:
        return

    nz = [r for r in used if r["lnUniq"] > 0]
    print(f"\ncalls with >=1 unique MapLine: {len(nz)}")

    # --- directional spread -------------------------------------------------
    sp = [r["spread"] for r in used if r["spread"] is not None]
    if sp:
        print("\nlnSpread (deg) over line-edge calls:")
        print(f"   min={min(sp):.1f} p25={st.quantiles(sp, n=4)[0]:.1f} "
              f"median={st.median(sp):.1f} p75={st.quantiles(sp, n=4)[2]:.1f} "
              f"max={max(sp):.1f}")
        for thr in (10.0, 25.0, 45.0):
            n = sum(1 for v in sp if v < thr)
            print(f"   spread < {thr:>4.0f} deg : {n} calls "
                  f"({100.0*n/len(sp):.1f}%)  <- near-parallel/degenerate")
        # degenerate among multi-line calls only (1 line cannot span)
        multi = [r for r in used if r["lnEdges"] >= 2]
        if multi:
            deg = [r for r in multi if r["spread"] is not None
                   and r["spread"] < 25.0]
            print(f"   of the {len(multi)} calls with >=2 line edges, "
                  f"{len(deg)} had spread < 25 deg")

    # --- outliers -----------------------------------------------------------
    print("\nline-edge outlier behaviour:")
    with_out = [r for r in used if r["lnOut"] > 0]
    all_out = [r for r in used if r["lnOut"] > 0 and r["lnIn"] == 0]
    print(f"   calls with >=1 rejected line edge : {len(with_out)} "
          f"({100.0*len(with_out)/len(used):.1f}% of line-edge calls)")
    print(f"   calls where ALL line edges rejected: {len(all_out)}")
    if all_out:
        print(f"      frames: {[r['frame'] for r in all_out][:25]}"
              f"{' ...' if len(all_out) > 25 else ''}")
    tot_e = sum(r["lnEdges"] for r in used)
    tot_o = sum(r["lnOut"] for r in used)
    print(f"   aggregate rejected/total line edges: {tot_o}/{tot_e} "
          f"({100.0*tot_o/max(tot_e,1):.1f}%)")

    ch = [r["chi2"] for r in nz]
    if ch:
        print(f"\nsummed accepted-line chi2 over calls with >=1 unique MapLine: "
              f"median={st.median(ch):.4f} max={max(ch):.4f}")
    print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    for p in sys.argv[1:]:
        summarise(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
