#!/usr/bin/env python3
"""Aggregate and report the frozen-geometry mapper ablation.

Reads <root>/runs.csv (written by scripts/mapper_ablation_full.sh), computes
LPIPS from the saved shutdown renders (the C++ pipeline does not compute it),
and emits:

  <root>/report/runs_full.csv      full per-run table (one row per scene x sensor x seed x variant)
  <root>/report/per_scene.csv      per-scene mean+/-std (supplementary)
  <root>/report/main_ablation.tex  booktabs/multirow main table (LaTeX)
  <root>/report/summary.json       aggregated numbers + coverage / failure info

Averaging rule (as specified): for every metric, average the 8 scenes with
equal weight WITHIN each repeat to get m_r, then report mean +/- sample-std
(n-1) over the 3 repeats. Runs are never pooled across scenes for the std.
Missing/failed scene-runs are reported, never silently dropped from the mean.
"""

import argparse
import csv
import glob
import json
import math
import os
import statistics
import sys

CONDS = ["A", "B", "C", "D"]
COND_LABEL = {
    "A": "Point-only mapper",
    "B": "+ line init",
    "C": "+ $L_{str}$",
    "D": "+ Dir. dens.",
}
MODE_LABEL = {"rgbd": "RGB-D", "mono": "Mono"}


def load_runs(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("condition") not in CONDS:
                rows.append({**r, "_failed": True})
                continue
            rows.append({**r, "_failed": False})
    return rows


def fnum(v):
    try:
        return float(v)
    except Exception:
        return None


def find_shutdown(run_dir, cond):
    d = glob.glob(os.path.join(run_dir, cond, "*_shutdown"))
    return d[0] if d else None


def lpips_for(run_dir, cond):
    """Mean LPIPS over the shutdown renders of one condition run."""
    d = find_shutdown(run_dir, cond)
    if not d:
        return None
    gts = sorted(glob.glob(os.path.join(d, "image_gt", "*_gt.*")))
    if not gts:
        return None
    vals = []
    for g in gts:
        base = os.path.basename(g)
        stem, ext = os.path.splitext(base)
        stem = stem[:-3] if stem.endswith("_gt") else stem
        r = os.path.join(d, "image", stem + ext)
        if not os.path.exists(r):
            continue
        a = _cv2.imread(r)
        b = _cv2.imread(g)
        if a is None or b is None or a.shape != b.shape:
            continue
        ta = _torch.from_numpy(a[:, :, ::-1].copy()).float().permute(2, 0, 1)[None] / 127.5 - 1.0
        tb = _torch.from_numpy(b[:, :, ::-1].copy()).float().permute(2, 0, 1)[None] / 127.5 - 1.0
        with _torch.no_grad():
            vals.append(float(_net(ta, tb).item()))
    if not vals:
        return None
    return sum(vals) / len(vals)


# ---- lazy LPIPS import so the script still runs without the package ----
_cv2 = _torch = _net = None
try:
    import cv2 as _cv2
    import torch as _torch
    import lpips as _lpips
    _net = _lpips.LPIPS(net="alex")
    _net.eval()
except Exception as e:  # pragma: no cover
    print(f"[warn] LPIPS unavailable: {e}", file=sys.stderr)


def mean_sd(vals):
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)  # sample std (n-1)


def macro(rows, mode, cond, key):
    """Equal-weight mean over scenes within each repeat, then mean/stdev over
    repeats. Returns (mean, sd, n_repeats, scene_counts)."""
    per_rep, counts = [], []
    for seed in sorted({r["seed"] for r in rows}):
        v = [fnum(r[key]) for r in rows
             if r["mode"] == mode and r["condition"] == cond and r["seed"] == seed
             and not r["_failed"]]
        v = [x for x in v if x is not None]
        counts.append(len(v))
        if v:
            per_rep.append(statistics.mean(v))   # equal weight over scenes
    m, sd = mean_sd(per_rep)
    return m, sd, len(per_rep), counts


def fmt(v, nd):
    return "--" if v is None else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/mapper_ablation_full")
    ap.add_argument("--scenes", default="office0 office1 office2 office3 office4 room0 room1 room2")
    ap.add_argument("--no-lpips", action="store_true")
    args = ap.parse_args()

    root = args.root
    rep_dir = os.path.join(root, "report")
    os.makedirs(rep_dir, exist_ok=True)
    scenes = args.scenes.split()

    rows = load_runs(os.path.join(root, "runs.csv"))

    # ---- attach LPIPS per run ----
    for r in rows:
        r["lpips"] = None
        if r["_failed"] or args.no_lpips or _net is None:
            continue
        run_dir = os.path.join(root, "runs", r["mode"], r["scene"], f"seed{r['seed']}")
        r["lpips"] = lpips_for(run_dir, r["condition"])

    # ---- full per-run CSV ----
    full_path = os.path.join(rep_dir, "runs_full.csv")
    fields = ["mode", "scene", "seed", "condition", "use_line_gaussians", "line_loss",
              "directional_densify", "psnr", "ssim", "lpips", "gaussians_init",
              "gaussians_final", "gaussians_point", "gaussians_line", "opt_steps",
              "refine_seconds", "nan", "grad_finite", "exit",
              "input_hash", "image_hash", "kf_seq_hash"]
    with open(full_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- per-scene supplementary table ----
    per_scene_path = os.path.join(rep_dir, "per_scene.csv")
    with open(per_scene_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "variant", "scene", "metric", "mean", "std", "n_repeats"])
        for mode in ["rgbd", "mono"]:
            for cond in CONDS:
                for scene in scenes:
                    for key, nd in (("psnr", 3), ("ssim", 4), ("lpips", 4),
                                    ("refine_seconds", 1), ("gaussians_final", 1)):
                        v = [fnum(r[key]) if key != "lpips" else r["lpips"] for r in rows
                             if r["mode"] == mode and r["condition"] == cond
                             and r["scene"] == scene and not r["_failed"]]
                        v = [x for x in v if x is not None]
                        if not v:
                            continue
                        m, sd = mean_sd(v)
                        w.writerow([mode, cond, scene, key, f"{m:.6f}",
                                    f"{sd:.6f}", len(v)])

    # ---- main table ----
    metrics = [("psnr", 3), ("ssim", 4), ("lpips", 4), ("refine_seconds", 1),
               ("gaussians_final", 1)]  # gaussians divided by 1000 below
    agg = {}
    coverage = {}
    for mode in ["rgbd", "mono"]:
        for cond in CONDS:
            for key, nd in metrics:
                m, sd, nrep, counts = macro(rows, mode, cond, key)
                if key == "gaussians_final" and m is not None:
                    m, sd = m / 1000.0, (sd / 1000.0 if sd else 0.0)
                agg[(mode, cond, key)] = (m, sd, nrep)
                coverage[(mode, cond, key)] = counts

    def best(mode, key, lower=False):
        vals = [(agg[(mode, c, key)][0], c) for c in CONDS]
        vals = [(v, c) for v, c in vals if v is not None]
        if not vals:
            return None
        return (min if lower else max)(vals)[1]

    mode_rows = {"rgbd": [], "mono": []}
    for mode in ["rgbd", "mono"]:
        b_psnr, b_ssim = best(mode, "psnr"), best(mode, "ssim")
        b_lp, b_t = best(mode, "lpips", True), best(mode, "refine_seconds", True)

        def cell(key, nd, want_best, suf=""):
            m, sd, _ = agg[(mode, CONDS[ci], key)]
            if m is None:
                return "--"
            s = f"{m:.{nd}f}" + (f"\\,$\\pm$\\,{sd:.{nd}f}" if sd else "")
            s += suf
            return f"\\textbf{{{s}}}" if want_best else s

        for ci, cond in enumerate(CONDS):
            sw = [
                "$\\checkmark$" if cond != "A" else "--",                       # line init
                "$\\checkmark$" if cond in ("C", "D") else "--",                # L_str
                "$\\checkmark$" if cond == "D" else "--",                       # dir. dens.
            ]
            mode_rows[mode].append(
                " & ".join([
                    COND_LABEL[cond],
                    sw[0], sw[1], sw[2],
                    cell("psnr", 3, cond == b_psnr),
                    cell("ssim", 4, cond == b_ssim),
                    cell("lpips", 4, cond == b_lp),
                    cell("refine_seconds", 1, cond == b_t),
                    cell("gaussians_final", 1, False),
                ])
            )

    tex = [
        "% Generated by scripts/mapper_ablation_report.py -- do not edit by hand.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Frozen-geometry mapper ablation on Replica (8 scenes: office0--4, room0--2; "
        "500 frames per scene). Every scene$\\times$sensor geometry (keyframe poses, sparse "
        "point cloud incl.\\ line samples, images, camera parameters) is frozen once and the "
        "four mapper variants replay that identical snapshot, so all differences are "
        "attributable to the mapper configuration. Each run performs exactly 10\\,000 real "
        "optimizer steps on one shared, pre-generated keyframe sampling sequence. "
        "Values are macro-averages over the 3 paired repeats of the per-repeat mean over the "
        "8 scenes (mean$\\pm$sample std, $n{-}1$); scenes are never pooled across repeats for "
        "the std. Training time is the per-scene wall time of the 10\\,000 steps including "
        "densification/pruning, excluding data loading and evaluation. "
        "A is a point-only \\emph{mapper}, not a point-only SLAM run. Best quality and shortest "
        "time per camera mode in bold; the Gaussian count is reported as capacity and is not "
        "assumed to be better when smaller.}",
        "\\label{tab:main_ablation}",
        "\\begin{tabular}{l l c c c r r r r r}",
        "\\toprule",
        "Cam & Variant & Line Init & $L_{str}$ & Dir.\\ Dens. & PSNR$\\uparrow$ & SSIM$\\uparrow$ "
        "& LPIPS$\\downarrow$ & Train Time(s)$\\downarrow$ & Final Gaussians($\\times10^{3}$) \\\\",
        "\\midrule",
    ]
    for mode in ["rgbd", "mono"]:
        for i, line in enumerate(mode_rows[mode]):
            cam = f"\\multirow{{4}}{{*}}{{{MODE_LABEL[mode]}}}" if i == 0 else ""
            tex.append(f"{cam} & {line} \\\\")
        if mode == "rgbd":
            tex.append("\\midrule")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    tex_path = os.path.join(rep_dir, "main_ablation.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(tex))

    # ---- coverage / failure report ----
    failed = [r for r in rows if r["_failed"]]
    short = {}
    for (mode, cond, key), counts in coverage.items():
        if key != "psnr":
            continue
        for i, c in enumerate(counts):
            if c != len(scenes):
                short.setdefault(f"{mode}/{cond}/repeat{i}", c)
    summary = {
        "n_rows": len(rows),
        "n_failed_rows": len(failed),
        "scenes_expected": len(scenes),
        "scenes_found": sorted({r["scene"] for r in rows if not r["_failed"]}),
        "partial_coverage": short,
        "aggregated": {f"{m}/{c}/{k}": {"mean": v[0], "sd": v[1], "n_repeats": v[2]}
                       for (m, c, k), v in agg.items()},
    }
    json_path = os.path.join(rep_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"per-run CSV      : {full_path}")
    print(f"per-scene CSV    : {per_scene_path}")
    print(f"LaTeX table      : {tex_path}")
    print(f"aggregate JSON   : {json_path}")
    if failed:
        print(f"[warn] {len(failed)} failed run row(s) recorded")
    if short:
        print(f"[warn] partial scene coverage: {short}")


if __name__ == "__main__":
    main()
