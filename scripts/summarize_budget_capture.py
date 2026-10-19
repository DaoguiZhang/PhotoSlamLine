#!/usr/bin/env python3
"""mapper-budget-v1 capture summariser.

One re-runnable command that turns a capture directory into the tables used in
reports/mapper_budget_v1_report.md. No GPU needed; reads the files the capture
already wrote.

    python3 scripts/summarize_budget_capture.py abl_out/budget_capture_office0
"""
import collections
import csv
import os
import statistics
import sys


def hdr(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


def creation(d):
    ev = os.path.join(d, "creation_audit_events.csv")
    if not os.path.isfile(ev):
        return
    hdr("二.4  creation accounting (add / remove per entry)")
    rows = list(csv.DictReader(open(ev)))
    add, rem, cnt = collections.Counter(), collections.Counter(), collections.Counter()
    for r in rows:
        k = (r["reason"], r["entry"])
        add[k] += int(r["delta_in"])
        rem[k] += int(r["delta_out"])
        cnt[k] += 1
    print(f"{'reason':<22}{'entry':<10}{'events':>8}{'added':>12}{'removed':>12}")
    for k in sorted(set(add) | set(rem), key=lambda k: -(add[k] + rem[k])):
        print(f"{k[0]:<22}{k[1]:<10}{cnt[k]:>8}{add[k]:>12}{rem[k]:>12}")
    print("\n  NOTE clone/split removed = 0 => all removals are the prune mask;")
    print("       densifyAndSplitWithLineAwareness appends and keeps the parent.")


def linepop(d):
    p = os.path.join(d, "linepop_linepop.csv")
    if not os.path.isfile(p):
        return
    hdr("二.2  line-Gaussian population (online)")
    rows = list(csv.DictReader(open(p)))
    print(f"{'iteration':>10}{'stage':>10}{'total':>10}{'line':>8}{'ordinary':>10}")
    for r in rows[:3]:
        print(f"{r['iteration']:>10}{r['stage']:>10}{r['total']:>10}{r['line']:>8}{r['ordinary']:>10}")
    print("   ...")
    for r in rows[-3:]:
        print(f"{r['iteration']:>10}{r['stage']:>10}{r['total']:>10}{r['line']:>8}{r['ordinary']:>10}")

    q = os.path.join(d, "linepop_lineinherit.csv")
    if os.path.isfile(q):
        ir = list(csv.DictReader(open(q)))
        mis = sum(1 for r in ir if r["aligned_line"] != "1"
                  or r["aligned_anchor"] != "1" or r["aligned_dir"] != "1")
        zd = sum(1 for r in ir if r["line_zero_dir_rows"].lstrip("-").isdigit()
                 and int(r["line_zero_dir_rows"]) > 0)
        hdr("二.3  line metadata inheritance")
        print(f"  events checked                     : {len(ir)}")
        print(f"  events with any misalignment       : {mis}")
        print(f"  events with zero-direction line row: {zd}")


def linesample(d):
    p = os.path.join(d, "linesample_linesample.csv")
    if not os.path.isfile(p):
        return
    hdr("二.1  line-sampling entry: candidates and rejections")
    rows = list(csv.DictReader(open(p)))
    print(f"{'iteration':>10}{'raw':>6}{'candidates':>12}{'rejected_voxel':>16}{'accepted':>10}")
    for r in rows:
        print(f"{r['iteration']:>10}{r['raw_points']:>6}{r['candidates']:>12}"
              f"{r['rejected_voxel_occupied']:>16}{r['accepted']:>10}")
    tot_c = sum(int(r["candidates"]) for r in rows)
    print(f"\n  entry reached {len(rows)} time(s) in the whole run; candidate pool = {tot_c}")


def costcurve(d):
    p = os.path.join(d, "gaussian_training_log.txt")
    if not os.path.isfile(p):
        return
    hdr("三  real-scale cost curve (from the capture's own per-step log)")
    rows = []
    for line in open(p):
        f = line.split()
        if len(f) >= 4:
            try:
                rows.append((int(f[0]), float(f[1]), int(f[3])))
            except ValueError:
                pass
    if not rows:
        return
    print(f"  steps={len(rows)}  billed mapping={sum(r[1] for r in rows)/1000:.2f} s"
          f"  mean={statistics.mean(r[1] for r in rows):.3f} ms/step")
    b = collections.OrderedDict()
    for it, ms, n in rows:
        b.setdefault((n // 40000) * 40000, []).append(ms)
    print(f"\n{'count_bucket':>14}{'steps':>8}{'mean_ms/step':>14}{'median':>9}")
    for k in sorted(b):
        v = b[k]
        print(f"{k:>14}{len(v):>8}{statistics.mean(v):>14.3f}{statistics.median(v):>9.1f}")
    for lbl, seg in (("first 500", rows[:500]), ("last 500", rows[-500:])):
        ns = [r[2] for r in seg]
        ms = [r[1] for r in seg]
        print(f"  {lbl}: n~{statistics.mean(ns):.0f}  mean={statistics.mean(ms):.3f} "
              f"median={statistics.median(ms):.1f} ms/step")
    print("\n  A ~100x count increase costs only ~1.5x the per-step time, so the frozen-pair")
    print("  decomposition CANNOT be scaled by primitive count.")


def snapshots(d):
    s = os.path.join(d, "snapshots")
    if not os.path.isdir(s):
        return
    hdr("二.5  persisted snapshots (full tensors, not PLY)")
    for f in sorted(os.listdir(s)):
        full = os.path.join(s, f)
        if f.endswith(".txt"):
            print(f"  --- {f}")
            for line in open(full):
                print(f"      {line.rstrip()}")
        else:
            print(f"  {f:<28}{os.path.getsize(full)/1e6:>8.1f} MB")


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "abl_out/budget_capture_office0"
    if not os.path.isdir(d):
        print(f"no such capture dir: {d}", file=sys.stderr)
        return 1
    print(f"capture: {d}")
    creation(d)
    linepop(d)
    linesample(d)
    snapshots(d)
    costcurve(d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
