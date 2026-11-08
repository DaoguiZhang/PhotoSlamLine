#!/usr/bin/env python3
"""Aggregate interleaved A/B/C runs emitted by fr3_abc_interleave.sh.

Reads every `<OUTBASE>/r<k>_<C>/ate_se3.json` plus the run's diagnostic counts
and prints, per condition: n, mean, sample std, min, max of the SE(3) ATE RMSE,
the per-frame sigma and the RPE, together with the run-to-run SPREAD. The point
is to compare the between-condition difference against the within-condition
spread -- not to declare a winner from one run.

Usage: python3 scripts/aggregate_abc.py <OUTBASE>
"""
import sys
import os
import re
import json
import glob
import statistics as st


def counts(log):
    d = {"po_calls": 0, "po_with_lines": 0, "frames_with_lines": 0,
         "ln_edges_total": 0, "rescued": 0, "gate_oldfail": 0}
    if not os.path.exists(log):
        return d
    rx_edge = re.compile(r"\[LineEdgeDiag\] PO frame=(\d+) NL=(\d+) "
                         r"lnTrackedNonNull=(\d+) ptEdges=(\d+) ptIn=(\d+) "
                         r"ptOut=(\d+) ptChi2=(\S+) lnEdges=(\d+)")
    rx_res = re.compile(r"\[LineRescue\].*oldPtVerdict=(\S+) new=(\S+)")
    rx_gate = re.compile(r"\[LineGate\].*old_would_fail=(\d+)")
    fr = set()
    with open(log, errors="replace") as f:
        for line in f:
            m = rx_edge.search(line)
            if m:
                d["po_calls"] += 1
                n = int(m.group(8))          # lnEdges is group 8 in rx_edge
                d["ln_edges_total"] += n
                if n > 0:
                    d["po_with_lines"] += 1
                    fr.add(int(m.group(1)))
                continue
            m = rx_res.search(line)
            if m:
                if m.group(1) == "FAIL" and m.group(2) == "OK":
                    d["rescued"] += 1
                continue
            m = rx_gate.search(line)
            if m and int(m.group(1)) == 1:
                d["gate_oldfail"] += 1
    d["frames_with_lines"] = len(fr)
    return d


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    base = sys.argv[1]
    runs = {}
    for d in sorted(glob.glob(os.path.join(base, "r*_*"))):
        m = re.search(r"r(\d+)_([ABC])$", d)
        if not m:
            continue
        j = os.path.join(d, "ate_se3.json")
        if not os.path.exists(j):
            continue
        with open(j) as f:
            r = json.load(f)
        if r.get("status") != "OK":
            print(f"  skip {d}: status={r.get('status')}")
            continue
        runs.setdefault(m.group(2), []).append({
            "run": int(m.group(1)),
            "rmse": r["ate_se3"]["rmse"],
            "sigma": r["ate_se3"]["std"],
            "pairs": r["n_pairs"],
            "rpe1": r["rpe"].get("1", {}).get("trans_drift_rms_m_per_s"),
            "counts": counts(os.path.join(d, "log.txt")),
        })

    if not runs:
        print("no completed runs found")
        return 1

    print("=" * 80)
    print("PER-RUN")
    print("=" * 80)
    hdr = (f"{'cond':<5}{'run':<4}{'pairs':<7}{'ATE_RMSE':<11}{'sigma':<10}"
           f"{'RPE_d1':<10}{'POcalls':<9}{'POwLines':<10}{'lnEdges':<9}"
           f"{'framesWLn':<10}{'rescued':<8}{'gateFail':<9}")
    print(hdr)
    for c in ("A", "B", "C"):
        for r in sorted(runs.get(c, []), key=lambda x: x["run"]):
            k = r["counts"]
            print(f"{c:<5}{r['run']:<4}{r['pairs']:<7}{r['rmse']:<11.6f}"
                  f"{r['sigma']:<10.6f}"
                  f"{(r['rpe1'] if r['rpe1'] is not None else float('nan')):<10.4f}"
                  f"{k['po_calls']:<9}{k['po_with_lines']:<10}"
                  f"{k['ln_edges_total']:<9}{k['frames_with_lines']:<10}"
                  f"{k['rescued']:<8}{k['gate_oldfail']:<9}")

    print()
    print("=" * 80)
    print("AGGREGATE")
    print("=" * 80)
    print(f"{'cond':<5}{'n':<4}{'ATE mean':<12}{'ATE std':<12}{'min':<12}"
          f"{'max':<12}{'spread':<12}")
    summary = {}
    for c in ("A", "B", "C"):
        v = [r["rmse"] for r in runs.get(c, [])]
        if not v:
            continue
        mean = st.mean(v)
        sd = st.stdev(v) if len(v) > 1 else 0.0
        summary[c] = (mean, sd, min(v), max(v))
        print(f"{c:<5}{len(v):<4}{mean:<12.6f}{sd:<12.6f}{min(v):<12.6f}"
              f"{max(v):<12.6f}{max(v)-min(v):<12.6f}")

    if len(summary) >= 2:
        worst_within = max(s for _, s, _, _ in summary.values())
        vals = [m for m, _, _, _ in summary.values()]
        between = max(vals) - min(vals)
        print(f"\nmax within-condition std = {worst_within:.6f} m")
        print(f"between-condition range  = {between:.6f} m")
        if between <= 2 * worst_within:
            print("VERDICT: between-condition difference is NOT separable from "
                  "run-to-run noise at this n -> cannot claim line benefit/harm.")
        else:
            print("VERDICT: between-condition difference exceeds 2x the "
                  "within-condition std; still confirm with more repeats before "
                  "claiming benefit/harm.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
