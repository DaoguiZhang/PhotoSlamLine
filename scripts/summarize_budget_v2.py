#!/usr/bin/env python3
"""mapper-budget-v2-hash summariser.

Reads a paired R100/R50 replay directory and prints the stream symmetry, the
paired quality/time table and the verdict against the pre-registered gate.

    python3 scripts/summarize_budget_v2.py abl_out/bv2_formal_r0

Gate (one-sided, delta = R50 - R100), fixed before any R50 measurement:
    PSNR >= -0.05 dB, SSIM >= -0.001, LPIPS <= +0.002
"""
import csv
import os
import sys

GATE = {"psnr": -0.05, "ssim": -0.001, "lpips": 0.002}


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "abl_out/bv2_formal_r0"
    arms = sys.argv[2].split(",") if len(sys.argv) > 2 else ["R100", "R50"]
    if not os.path.isdir(d):
        print(f"no such dir: {d}", file=sys.stderr)
        return 1

    print("=" * 78)
    print("stream symmetry (both arms must see the SAME candidate pool)")
    print("=" * 78)
    pools = {}
    for a in arms:
        p = os.path.join(d, a, "inactive_batches.csv")
        if not os.path.isfile(p):
            print(f"  {a}: missing {p}")
            continue
        rows = read(p)
        cand = sum(int(r["candidates"]) for r in rows)
        kept = sum(int(r["kept"]) for r in rows)
        pools[a] = cand
        print(f"  {a}: batches={len(rows):>3} candidates={cand:>8} kept={kept:>8} "
              f"ratio={kept / cand if cand else 0:.3f}")
    if len(pools) == 2 and len(set(pools.values())) == 1:
        print("  -> SAME pool in both arms: OK")
    elif len(pools) == 2:
        print("  -> WARNING: pools differ; the comparison is NOT valid")

    print()
    print("=" * 78)
    print("paired result")
    print("=" * 78)
    s = os.path.join(d, "frozen", "summary_" + "".join(arms) + ".csv")
    if not os.path.isfile(s):
        print(f"missing {s}", file=sys.stderr)
        return 1
    rows = {r["condition"]: r for r in read(s)}
    keys = [("psnr", "PSNR", "%.4f"), ("ssim", "SSIM", "%.6f"),
            ("gaussians_final", "final Gaussians", "%s"),
            ("gaussians_line", "line Gaussians", "%s"),
            ("mapping_seconds", "mapping s", "%.4f"),
            ("refine_seconds", "refine s", "%.4f")]
    a0, a1 = arms
    print(f"{'metric':<18}{a0:>14}{a1:>14}{'delta':>14}")
    deltas = {}
    for k, lbl, fmt in keys:
        v0, v1 = float(rows[a0][k]), float(rows[a1][k])
        deltas[k] = v1 - v0
        print(f"{lbl:<18}{fmt % v0:>14}{fmt % v1:>14}{v1 - v0:>+14.6g}")

    print()
    print("=" * 78)
    print("verdict against the pre-registered one-sided gate")
    print("=" * 78)
    ok = True
    for k, lbl, thr in (("psnr", "PSNR (dB)", GATE["psnr"]),
                        ("ssim", "SSIM", GATE["ssim"])):
        v = deltas[k]
        good = v >= thr
        ok &= good
        print(f"  {lbl:<10} delta={v:+.6f}  threshold>={thr:+.4g}  "
              f"{'PASS' if good else 'FAIL'}")
    print(f"  {'LPIPS':<10} not measured unless an lpips_fgm.py pass was run for this pair")
    dt = deltas["mapping_seconds"]
    print(f"  mapping time delta {dt:+.4f} s ({100 * dt / float(rows[a0]['mapping_seconds']):+.2f} %) "
          f"-> {'faster' if dt < 0 else 'NOT faster'}")
    print()
    print("  NOTE: a time delta of a few percent is inside the measured same-scene")
    print("        wall-clock drift (0.7-6.1 %); do not call it a regression.")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
