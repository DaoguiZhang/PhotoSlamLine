#!/usr/bin/env python3
"""Find NATURAL line-rescue opportunities from the [LineScan] stream.

An "opportunity" is a decision point where, on the SAME frame:
  * the ORIGINAL point-only verdict failed  (oldPtVerdict=FAIL), and
  * the line evidence alone was sufficient  (lnOk=1: enough unique reliable
    MapLines, inlier ratio OK, and directional spread >= the configured angle).

Nothing here is selected using ground truth, and no threshold is tuned: the scan
runs inside condition B (rescue DISABLED), so it reports what the rescue *would*
have done without perturbing the trajectory.

Usage: python3 scripts/rescue_opportunity_scan.py <log> [<log> ...] [--csv F]
"""
import sys
import re
import csv
from collections import Counter

RX = re.compile(
    r"\[LineScan\] stage=(\S+) map=(-?\d+) frame=(\d+)"
    r" ptBeforePO=(\d+) lnBeforePO=(\d+) ptAfterPO=(\d+) lnAfterPO=(\d+)"
    r" oldPtVerdict=(\S+) lnUniq=(\d+) lnEdges=(\d+) lnSpread=(\S+)"
    r" lnOk=(\d+) lnReason=(-?\d+) gateMmOldFail=(-?\d+)"
    r" gateTrkRefOldFail=(-?\d+) state=(-?\d+)")


def parse(path):
    rows = []
    for line in open(path, errors="replace"):
        m = RX.search(line)
        if not m:
            continue
        g = m.groups()
        rows.append(dict(
            stage=g[0], frame=int(g[2]),
            ptBeforePO=int(g[3]), lnBeforePO=int(g[4]),
            ptAfterPO=int(g[5]), lnAfterPO=int(g[6]),
            old=g[7], uniq=int(g[8]), edges=int(g[9]),
            spread=float(g[10]) if g[10] not in ("nan",) else float("nan"),
            lnOk=int(g[11]), lnReason=int(g[12]),
            gateMm=int(g[13]), gateTrkRef=int(g[14]), state=int(g[15])))
    return rows


def quant(v, p):
    if not v:
        return 0
    v = sorted(v)
    return v[min(len(v) - 1, int(p / 100.0 * len(v)))]


def report(path, csvw=None):
    rows = parse(path)
    print("=" * 84)
    print(f"RESCUE OPPORTUNITY SCAN: {path}")
    print("=" * 84)
    if not rows:
        print("  no [LineScan] lines "
              "(was PHOTO_SLAM_DEBUG_LINE_SCAN=1 set, and LINE_MODE=2?)")
        return
    print(f"  decisions: {len(rows)}   frames: {len({r['frame'] for r in rows})}")
    st = Counter(r["stage"] for r in rows)
    print(f"  stages    : {dict(st)}")

    fail = [r for r in rows if r["old"] == "FAIL"]
    opp = [r for r in fail if r["lnOk"] == 1]
    print(f"\n  oldPtVerdict=FAIL (rescue would be attempted) : {len(fail)}")
    print(f"  ... AND lnOk=1  (NATURAL OPPORTUNITY)          : {len(opp)}")
    # also the stricter variant: enough points to satisfy the rescue point floor
    opp10 = [r for r in opp if r["ptAfterPO"] >= 10]
    print(f"  ... AND ptAfterPO >= 10 (point floor)          : {len(opp10)}")

    print("\n  line evidence over ALL decisions:")
    uq = [r["uniq"] for r in rows]
    sp = [r["spread"] for r in rows if r["spread"] == r["spread"]]
    eo = [r["edges"] for r in rows]
    print(f"    lnUniq   min={min(uq)} p50={quant(uq,50)} p90={quant(uq,90)} max={max(uq)}"
          f"   lnOk=1 in {sum(1 for r in rows if r['lnOk']==1)} decisions")
    print(f"    lnEdges  min={min(eo)} p50={quant(eo,50)} max={max(eo)}")
    if sp:
        print(f"    lnSpread min={min(sp):.1f} p50={quant(sp,50):.1f} "
              f"p90={quant(sp,90):.1f} max={max(sp):.1f} deg")

    print("\n  rejection reasons when oldPtVerdict=FAIL:")
    print(f"    {dict(Counter(r['lnReason'] for r in fail))}")
    print("  (lnReason 1=too few unique lines, 2=parallel/degenerate, "
          "3=bad inlier ratio; 0/OK means the line test passed)")

    print("\n  VO gate flags recorded on the same frames:")
    print(f"    gateMmOldFail=1     : {sum(1 for r in rows if r['gateMm']==1)}")
    print(f"    gateTrkRefOldFail=1 : {sum(1 for r in rows if r['gateTrkRef']==1)}")

    if opp:
        print("\n  OPPORTUNITY FRAMES (oldPtVerdict=FAIL and lnOk=1):")
        for r in opp[:40]:
            print(f"    frame={r['frame']:<6} stage={r['stage']:<22} "
                  f"ptBefore={r['ptBeforePO']:<6} ptAfter={r['ptAfterPO']:<5} "
                  f"uniq={r['uniq']:<4} edges={r['edges']:<4} "
                  f"spread={r['spread']:.1f}")
        if len(opp) > 40:
            print(f"    ... and {len(opp)-40} more")

    if csvw is not None:
        for r in rows:
            csvw.writerow([path, r["stage"], r["frame"], r["ptBeforePO"],
                           r["lnBeforePO"], r["ptAfterPO"], r["lnAfterPO"],
                           r["old"], r["uniq"], r["edges"], r["spread"],
                           r["lnOk"], r["lnReason"], r["gateMm"],
                           r["gateTrkRef"], r["state"]])
    print()


def main():
    args = [a for a in sys.argv[1:]]
    csv_path = None
    if "--csv" in args:
        i = args.index("--csv")
        csv_path = args[i + 1]
        del args[i:i + 2]
    if not args:
        print(__doc__)
        return 1
    f = None
    w = None
    if csv_path:
        f = open(csv_path, "w", newline="")
        w = csv.writer(f)
        w.writerow(["log", "stage", "frame", "ptBeforePO", "lnBeforePO",
                    "ptAfterPO", "lnAfterPO", "oldPtVerdict", "lnUniq",
                    "lnEdges", "lnSpread", "lnOk", "lnReason",
                    "gateMmOldFail", "gateTrkRefOldFail", "state"])
    for p in args:
        report(p, w)
    if f:
        f.close()
        print("CSV:", csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
