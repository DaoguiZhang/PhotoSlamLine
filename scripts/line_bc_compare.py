#!/usr/bin/env python3
"""Compare two line-ablation logs (B vs C) frame by frame.

Answers exactly the question that must be settled BEFORE any "the rescue helps"
claim:  are there frames where the ORIGINAL point-only control flow fails (so
condition B loses the frame) but condition C bypasses the gate and keeps it?

It also reports how many frames actually handed a LINE edge to
PoseOptimizationWithLine in each condition -- the only valid evidence that
lines participated in pose optimisation.

Usage:
    python3 scripts/line_bc_compare.py <B_log> <C_log> [--csv out.csv]
"""
import sys
import re
import csv
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from line_diag_report import RE_GATE, RE_RESCUE, RE_EDGE  # noqa: E402


def parse(path):
    gate, rescue, edge = [], {}, {}
    with open(path, errors="replace") as f:
        for line in f:
            m = RE_GATE.search(line)
            if m:
                gate.append({"site": m.group(1), "frame": int(m.group(2)),
                             "pt": int(m.group(3)),
                             "ln": int(m.group(4)) if m.group(4) else -1,
                             "old_fail": int(m.group(5)),
                             "bypass": int(m.group(6)) if m.group(6) else -1})
                continue
            m = RE_RESCUE.search(line)
            if m:
                g = m.groups()
                rescue[int(g[2])] = {"stage": g[0], "old": g[7], "new": g[8],
                                     "ptAfter": int(g[5]), "lnAfter": int(g[6]),
                                     "reason": int(g[9])}
                continue
            m = RE_EDGE.search(line)
            if m:
                g = m.groups()
                fr = int(g[0])
                e = edge.setdefault(fr, {"calls": 0, "lnEdges": 0, "lnUniq": 0,
                                         "ptEdges": 0, "NL": int(g[1]),
                                         "nonNull": int(g[2])})
                e["calls"] += 1
                e["lnEdges"] += int(g[7])
                e["lnUniq"] = max(e["lnUniq"], int(g[12]))
                e["ptEdges"] += int(g[3])
    return gate, rescue, edge


def main():
    a = [x for x in sys.argv[1:]]
    csv_out = None
    if "--csv" in a:
        i = a.index("--csv")
        csv_out = a[i + 1]
        del a[i:i + 2]
    if len(a) < 2:
        print(__doc__)
        return 1
    b_log, c_log = a[0], a[1]

    gB, rB, eB = parse(b_log)
    gC, rC, eC = parse(c_log)

    b_fail = {g["frame"]: g for g in gB
              if g["site"] == "MotionModel" and g["old_fail"] == 1}
    b_fail_tr = {g["frame"]: g for g in gB
                 if g["site"] == "TrackReferenceKeyFrame" and g["old_fail"] == 1}
    c_bypass = {g["frame"]: g for g in gC if g["bypass"] == 1 and g["old_fail"] == 1}
    c_fail = {g["frame"]: g for g in gC
              if g["site"] == "MotionModel" and g["old_fail"] == 1}

    print("=" * 78)
    print(f"B (rescue OFF) log : {b_log}")
    print(f"C (rescue ON ) log : {c_log}")
    print("=" * 78)

    print(f"\n[B] MotionModel old-path failures : {len(b_fail)}")
    print(f"[C] MotionModel old-path failures : {len(c_fail)}")
    print(f"[C] ACTUAL gate bypasses (old_would_fail=1 AND rescueEnabled=1): "
          f"{len(c_bypass)}")
    print(f"[B] TrackReferenceKF old failures: {len(b_fail_tr)}")
    print(f"[B] frames evaluated (PO calls)  : {sum(e['calls'] for e in eB.values())}"
          f" over {len(eB)} frames")
    print(f"[C] frames evaluated (PO calls)  : {sum(e['calls'] for e in eC.values())}"
          f" over {len(eC)} frames")

    print(f"\n[B] frames WITH line edges in PO  : "
          f"{sum(1 for e in eB.values() if e['lnEdges'] > 0)}")
    print(f"[C] frames WITH line edges in PO  : "
          f"{sum(1 for e in eC.values() if e['lnEdges'] > 0)}")
    for tag, e_ in (("B", eB), ("C", eC)):
        tot = sum(v["lnEdges"] for v in e_.values())
        frm = [k for k, v in e_.items() if v["lnEdges"] > 0]
        print(f"    {tag}: total line edges handed to PO = {tot}"
              f"   example frames = {sorted(frm)[:15]}")

    # The decisive set: old path fails, C bypasses
    both = sorted(set(c_bypass) & (set(c_fail) | set(b_fail) | set(b_fail_tr)))
    print(f"\n[DECISIVE] frames where the OLD point-only flow failed and C "
          f"bypassed the gate: {len(both)}")
    for fr in both[:40]:
        eb = eB.get(fr, {})
        ec = eC.get(fr, {})
        print(f"   frame={fr:<6} C_bypass=1 C_lnEdges={ec.get('lnEdges','-'):>4} "
              f"C_lnUniq={ec.get('lnUniq','-'):>3} B_lnEdges={eb.get('lnEdges','-'):>4} "
              f"C_rescue={rC.get(fr,{}).get('new','-')}")

    if csv_out:
        frames = sorted(set(list(eB) + list(eC) + list(rB) + list(rC)))
        with open(csv_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame",
                        "B_lnEdges", "B_lnUniq", "B_ptEdges",
                        "C_lnEdges", "C_lnUniq", "C_ptEdges",
                        "B_oldfail", "C_oldfail", "C_bypass",
                        "B_rescue_new", "C_rescue_new", "C_reason"])
            for fr in frames:
                eb = eB.get(fr, {})
                ec = eC.get(fr, {})
                w.writerow([fr,
                            eb.get("lnEdges", ""), eb.get("lnUniq", ""),
                            eb.get("ptEdges", ""),
                            ec.get("lnEdges", ""), ec.get("lnUniq", ""),
                            ec.get("ptEdges", ""),
                            1 if fr in b_fail or fr in b_fail_tr else 0,
                            1 if fr in c_fail else 0,
                            1 if fr in c_bypass else 0,
                            rB.get(fr, {}).get("new", ""),
                            rC.get(fr, {}).get("new", ""),
                            rC.get(fr, {}).get("reason", "")])
        print(f"\nper-frame CSV written: {csv_out}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
