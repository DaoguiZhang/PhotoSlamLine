#!/usr/bin/env python3
"""Aggregate the formal batch without pooling scenes across repeats.

Per cell/condition: mean+-sample-std over accepted repeats.
Per family/modality: first average the cells within each repeat (equal weight),
then mean+-sample-std over repeats. This mirrors the pre-registered protocol and
never uses a scene-pooled std across repeats.
Also reports paired Full-OFF deltas matched by repeat.
"""
import csv
import os
import statistics
import sys

EXP = "/workspace/code/PSL-fullmap-exp-v1"
SRC = os.path.join(EXP, "runs", "formal-v1", "summary.csv")
OUT = os.path.join(EXP, "report")
METRICS = ("mean_psnr", "total_iterations", "ply_vertices", "wall_s")


def fnum(v):
    try:
        return float(v)
    except Exception:
        return None


def load():
    rows = []
    with open(SRC, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["rep"] = int(r["rep"])
            r["accept_b"] = str(r.get("accept", "")).lower() in ("true", "1")
            for m in METRICS:
                r[m] = fnum(r.get(m))
            rows.append(r)
    return rows


def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    if len(vals) == 1:
        return vals[0], 0.0, 1
    return statistics.mean(vals), statistics.stdev(vals), len(vals)


def main() -> int:
    if not os.path.exists(SRC):
        print(f"missing {SRC}", file=sys.stderr)
        return 2
    os.makedirs(OUT, exist_ok=True)
    rows = load()

    # Per cell/condition.
    cell_rows = []
    keys = sorted({(r["family"], r["dataset"], r["modality"], r["cell"], r["condition"])
                   for r in rows})
    for fam, ds, mod, cell, cond in keys:
        sub = [r for r in rows if r["cell"] == cell and r["condition"] == cond and r["accept_b"]]
        rec = dict(family=fam, dataset=ds, modality=mod, cell=cell, condition=cond,
                   n_accepted=len(sub))
        for m in METRICS:
            mean, sd, n = ms([r[m] for r in sub])
            rec[m + "_mean"] = mean
            rec[m + "_std"] = sd
        cell_rows.append(rec)
    with open(os.path.join(OUT, "formal_summary_cell.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cell_rows[0].keys()))
        w.writeheader(); w.writerows(cell_rows)

    # Per family/modality: repeat -> equal-weight cell mean.
    group_rows = []
    groups = sorted({(r["family"], r["modality"]) for r in rows})
    for fam, mod in groups:
        rec = dict(family=fam, modality=mod)
        cells = sorted({r["cell"] for r in rows if r["family"] == fam and r["modality"] == mod})
        for cond in ("full", "off"):
            per_rep = []
            for rep in sorted({r["rep"] for r in rows}):
                vals = [r["mean_psnr"] for r in rows
                        if r["family"] == fam and r["modality"] == mod
                        and r["condition"] == cond and r["rep"] == rep
                        and r["accept_b"] and r["mean_psnr"] is not None]
                # Only use a repeat when every cell of the group is accepted.
                if len(vals) == len(cells):
                    per_rep.append(statistics.mean(vals))
            mean, sd, n = ms(per_rep)
            rec[f"{cond}_psnr_mean"] = mean
            rec[f"{cond}_psnr_std"] = sd
            rec[f"{cond}_repeats_used"] = n
            rec[f"{cond}_cells"] = len(cells)
        group_rows.append(rec)

    # Paired deltas per cell (Full - OFF, same repeat).
    pair_rows = []
    for fam, ds, mod, cell, _ in keys:
        deltas = []
        for rep in sorted({r["rep"] for r in rows}):
            a = [r for r in rows if r["cell"] == cell and r["rep"] == rep and r["condition"] == "full" and r["accept_b"]]
            b = [r for r in rows if r["cell"] == cell and r["rep"] == rep and r["condition"] == "off" and r["accept_b"]]
            if a and b and a[0]["mean_psnr"] is not None and b[0]["mean_psnr"] is not None:
                deltas.append(a[0]["mean_psnr"] - b[0]["mean_psnr"])
        mean, sd, n = ms(deltas)
        pair_rows.append(dict(family=fam, dataset=ds, modality=mod, cell=cell,
                              n_pairs=n, d_psnr_mean=mean, d_psnr_std=sd))
    with open(os.path.join(OUT, "formal_paired_delta.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        w.writeheader(); w.writerows(pair_rows)
    with open(os.path.join(OUT, "formal_summary_group.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(group_rows[0].keys()))
        w.writeheader(); w.writerows(group_rows)

    invalid = [r for r in rows if not r["accept_b"]]
    print(f"rows={len(rows)} accepted={len(rows)-len(invalid)} invalid={len(invalid)}")
    print(f"wrote {OUT}/formal_summary_cell.csv")
    print(f"wrote {OUT}/formal_summary_group.csv")
    print(f"wrote {OUT}/formal_paired_delta.csv")
    for r in group_rows:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
