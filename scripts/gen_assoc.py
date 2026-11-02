#!/usr/bin/env python3
"""Deterministic RGB/depth association with the same semantics as TUM
evaluation/associate.py --max_difference D:

  1. read both timestamp lists (ignoring '#' comments and blank lines)
  2. for every timestamp t1 in list1, collect every t2 in list2 with
     |t2-t1| < max_difference as a candidate pair
  3. sort all candidate pairs by |t2-t1| ascending
  4. greedily accept a pair when NEITHER timestamp has been used yet

Note: unlike the reference script this reports how many RGB frames found no
depth partner, and never silently drops them from the caller's point of view --
the caller decides. For the mono arm the association file is irrelevant: mono
reads rgb.txt directly so every RGB frame is processed.
"""
import argparse
import sys


def read_list(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            out.append((float(parts[0]), parts[1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("first")
    ap.add_argument("second")
    ap.add_argument("--max_difference", type=float, default=0.02)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    l1 = read_list(a.first)
    l2 = read_list(a.second)
    d2 = {}
    for t, f in l2:
        d2.setdefault(t, []).append(f)

    cand = []
    for t1, f1 in l1:
        for t2, files in d2.items():
            diff = abs(t2 - t1)
            if diff < a.max_difference:
                for f2 in files:
                    cand.append((diff, t1, f1, t2, f2))
    cand.sort(key=lambda x: x[0])

    used1, used2 = set(), set()
    lines = []
    for diff, t1, f1, t2, f2 in cand:
        if t1 in used1 or t2 in used2:
            continue
        used1.add(t1)
        used2.add(t2)
        lines.append((t1, f1, t2, f2))

    with open(a.out, "w") as f:
        for t1, f1, t2, f2 in lines:
            f.write("%.6f %s %.6f %s\n" % (t1, f1, t2, f2))

    sys.stderr.write(
        "first=%d second=%d matched=%d unmatched_first=%d max_difference=%.6f -> %s\n"
        % (len(l1), len(l2), len(lines), len(l1) - len(lines), a.max_difference, a.out)
    )


if __name__ == "__main__":
    main()
