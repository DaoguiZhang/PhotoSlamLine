#!/usr/bin/env python3
"""perf-stage3 audit summariser.

One re-runnable command that turns the raw audit artifacts into the tables used in
reports/perf_stage3_report.md.

  # 二: online creation-source audit
  python3 scripts/summarize_creation_audit.py --creation-dir abl_out/online_audit_r0

  # 三: backward decomposition
  python3 scripts/summarize_creation_audit.py --decomp-glob 'abl_out/*/frozen/B_bwd_decomp_step*.csv'

  # 一: density-subset nesting / hashes
  python3 scripts/summarize_creation_audit.py --subset-run q_acc100 q_formal_r0

No GPU is needed: everything is read from files that the audit runs already wrote.
"""
import argparse
import collections
import csv
import glob
import hashlib
import os
import sys


def fmt(x, nd=6):
    return f"{x:.{nd}f}"


# ---------------------------------------------------------------- 二: creation
def summarize_creation(directory):
    events = os.path.join(directory, "creation_audit_events.csv")
    summary = os.path.join(directory, "creation_audit_summary.txt")
    if not os.path.isfile(events):
        print(f"!! missing {events}", file=sys.stderr)
        return 1

    rows = list(csv.DictReader(open(events)))
    tot = {}
    for k in ("delta_in", "delta_out"):
        tot[k] = sum(int(r[k]) for r in rows)

    added = collections.Counter()
    removed = collections.Counter()
    ev = collections.Counter()
    for r in rows:
        key = (r["reason"], r["entry"])
        added[key] += int(r["delta_in"])
        removed[key] += int(r["delta_out"])
        ev[key] += 1

    print("=" * 78)
    print("二  ONLINE CREATION-SOURCE AUDIT")
    print("=" * 78)
    print(f"events: {len(rows)}   total delta_in={tot['delta_in']}   "
          f"total delta_out={tot['delta_out']}")
    print()
    print(f"{'reason':<22}{'entry':<14}{'events':>8}{'added':>12}{'removed':>12}{'share':>9}")
    grand = tot["delta_in"] - tot["delta_out"]
    for key in sorted(set(added) | set(removed),
                      key=lambda k: -(added[k] + removed[k])):
        a, r_, n = added[key], removed[key], ev[key]
        share = a / tot["delta_in"] if tot["delta_in"] else 0.0
        print(f"{key[0]:<22}{key[1]:<14}{n:>8}{a:>12}{r_:>12}{share:>8.2%}")
    print()
    print(f"net from audit (in - out) = {grand}")

    if os.path.isfile(summary):
        s = dict(l.split(None, 1) for l in open(summary).read().strip().split("\n") if l.strip())
        print()
        for k in ("current_total", "peak_total", "net_from_audit",
                  "add_ordinary", "add_line", "clone", "split", "prune_removed"):
            if k in s:
                print(f"  {k:<18}{s[k].strip()}")

        # conservation: initial + inserted - removed == current
        cur = int(s.get("current_total", 0))
        net = (int(s.get("add_ordinary", 0)) + int(s.get("add_line", 0))
               + int(s.get("clone", 0)) + int(s.get("split", 0))
               - int(s.get("prune_removed", 0)))
        initial = cur - net
        print()
        print(f"  conservation: initial={initial} + inserted - removed = {cur} "
              f"(current_total)")
        print(f"  NOTE initial is the Gaussians present before the first audit "
              f"event (createFromPcd), so it is not itself an audit event.")
    return 0


# ------------------------------------------------------------ 二b: duplicate
def audit_duplicate_guards(root):
    """Report what the codebase does and does not have as an identity guard."""
    print()
    print("-" * 78)
    print("二b  DUPLICATE-SEEDING GUARD CHECK (static)")
    print("-" * 78)
    checks = [
        ("point_ids_ used as a dedup key",
         "point_ids_", ("index", "nonzero")),
        ("voxel occupancy queried before seeding",
         "isVoxelOccupied", ()),
        ("voxel grid written", "addPointToVoxel", ()),
    ]
    for label, needle, exclude in checks:
        hits = collections.Counter()
        for path in glob.glob(os.path.join(root, "src", "*.cpp")) + \
                    glob.glob(os.path.join(root, "include", "*.h")):
            try:
                text = open(path, errors="ignore").read()
            except OSError:
                continue
            for i, line in enumerate(text.split("\n"), 1):
                if needle in line and not any(e in line for e in exclude):
                    hits[f"{os.path.relpath(path, root)}:{i}"] += 1
        print(f"  {label}: {len(hits)} site(s)")
        for k in sorted(hits)[:6]:
            print(f"      {k}")
    print("  => recorded as a KNOWN GAP: no MapPointID / MapLineID association is")
    print("     stored per Gaussian, so 'the same source re-seeded' cannot be decided")
    print("     by identity. Direction similarity / voxel proximity are NOT identity.")
    return 0


# ----------------------------------------------------------------- 三: decomp
def summarize_decomp(patterns):
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(p)))
    if not files:
        print(f"!! no decomposition files matched {patterns}", file=sys.stderr)
        return 1
    print()
    print("=" * 78)
    print("三  BACKWARD DECOMPOSITION (host wall clock vs CUDA device time)")
    print("=" * 78)
    for f in files:
        rows = list(csv.DictReader(open(f)))
        print(f"\n--- {f}")
        print(f"{'stage':<30}{'host ms':>12}{'dev ms':>12}{'cpu ms':>12}")
        for r in rows:
            h, d = r["host_ms_per_iter"], r["dev_ms_per_iter"]
            if h == "":
                print(f"{r['stage']:<30}{'':>12}{d:>12}")
                continue
            try:
                hf, df = float(h), float(d)
            except ValueError:
                continue
            print(f"{r['stage']:<30}{fmt(hf):>12}{fmt(df):>12}{fmt(hf - df):>12}")
    return 0


# ------------------------------------------------------------- 一: subset check
def summarize_subset(root, runs):
    print()
    print("=" * 78)
    print("一  DENSITY-SUBSET RECONSTRUCTION / NESTING / HASHES")
    print("=" * 78)
    try:
        import torch
    except ImportError:
        print("!! torch not importable; skipping subset reconstruction", file=sys.stderr)
        return 1
    rc = 0
    for run in runs:
        ckpt = os.path.join(root, "abl_out", run, "frozen", "init_checkpoint.pt")
        if not os.path.isfile(ckpt):
            print(f"!! missing {ckpt}", file=sys.stderr)
            rc = 1
            continue
        sd = torch.jit.load(ckpt, map_location="cpu").state_dict()
        is_line = sd["6"].to(torch.bool).flatten()
        d = sd["7"]
        idx = [int(i) for i in torch.nonzero(is_line).flatten()]
        n_line_total, n_ord = len(idx), int((~is_line).sum())
        print(f"\n--- {run}: total={n_ord + n_line_total} line={n_line_total}")
        sets = {}
        for frac in (0.0, 0.25, 0.5, 1.0):
            if frac <= 0.0:
                kept = []
            elif frac >= 1.0:
                kept = list(idx)
            else:
                k = round(1.0 / frac)
                groups = collections.defaultdict(list)
                for row in idx:
                    x, y, z = (float(v) for v in d[row])
                    if x < 0 or (x == 0 and (y < 0 or (y == 0 and z < 0))):
                        x, y, z = -x, -y, -z
                    groups[(round(x * 100), round(y * 100), round(z * 100))].append(row)
                kept = []
                for key in sorted(groups):
                    v = sorted(groups[key])
                    kept.extend(v[::k])
                kept.sort()
                print(f"    frac={frac} stride={k} groups={len(groups)}")
            sets[frac] = set(kept)
            h = hashlib.sha256(",".join(map(str, kept)).encode()).hexdigest()[:16]
            print(f"    keep={frac:<5} n_kept={len(kept):<5} total={n_ord + len(kept):<6} "
                  f"idx_sha256={h}")
        nested = (sets[0.0] <= sets[0.25] <= sets[0.5] <= sets[1.0])
        print(f"    NESTED Q0<=Q25<=Q50<=Q100: {nested}")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    ap.add_argument("--creation-dir")
    ap.add_argument("--decomp-glob", action="append", default=[])
    ap.add_argument("--subset-run", nargs="*", default=[])
    ap.add_argument("--guards", action="store_true")
    a = ap.parse_args()

    rc = 0
    if a.subset_run:
        rc |= summarize_subset(a.root, a.subset_run)
    if a.creation_dir:
        rc |= summarize_creation(a.creation_dir)
        if a.guards or True:
            rc |= audit_duplicate_guards(a.root)
    if a.decomp_glob:
        rc |= summarize_decomp(a.decomp_glob)
    if not (a.subset_run or a.creation_dir or a.decomp_glob):
        ap.print_help()
    return rc


if __name__ == "__main__":
    sys.exit(main())
