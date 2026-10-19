#!/usr/bin/env python3
"""Aggregate the four-condition office0 comparison.

usage: quad_office0_summary.py <out_root> <gt_traj> <n_input_frames>

Reads <out_root>/runs/<cond>_r<round>/ and writes per_run.csv + summary.md.
Every number comes from the run's own artifacts:
  * [QuadMetrics] line in the log  -> optimizer_steps, gaussians, kfs, mapping_seconds
  * latest <iter>_shutdown snapshot -> psnr.txt, dssim.txt, image/ vs image_gt/
  * CameraTrajectory_TUM.txt vs GT  -> SE(3) ATE (scripts/eval_tum_se3_v2.py)
  * run_meta.txt                    -> exit code, wall seconds, peak VRAM
"""
import csv
import os
import re
import statistics
import subprocess
import sys

CONDS = ["P_ref", "P_fast", "L_ref", "L_fast"]
PAIRS = [("P_fast", "P_ref"), ("L_fast", "L_ref"),
         ("L_fast", "P_fast"), ("L_fast", "P_ref")]
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_meta(path):
    d = {}
    if not os.path.exists(path):
        return d
    for line in open(path):
        if "=" in line:
            k, v = line.strip().split("=", 1)
            d[k] = v
    return d


def parse_metrics(log):
    d = {}
    if not os.path.exists(log):
        return d
    pat = re.compile(r"\[QuadMetrics\] optimizer_steps=(\d+) gaussians=(\d+) "
                     r"line_gaussians=(\d+) kfs=(\d+) mapping_seconds=([0-9.eE+-]+)")
    for line in open(log, errors="replace"):
        m = pat.search(line)
        if m:
            d = dict(optimizer_steps=int(m.group(1)), gaussians=int(m.group(2)),
                     line_gaussians=int(m.group(3)), kfs=int(m.group(4)),
                     mapping_seconds=float(m.group(5)))
    return d


def latest_snapshot(run_dir):
    best, best_it = None, -1
    for name in os.listdir(run_dir):
        p = os.path.join(run_dir, name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "psnr.txt")):
            m = re.match(r"(\d+)", name)
            it = int(m.group(1)) if m else -1
            if it > best_it:
                best, best_it = p, it
    return best


def mean_column(path):
    if not path or not os.path.exists(path):
        return None
    vals = []
    for line in open(path):
        if line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) >= 2:
            try:
                vals.append(float(parts[1]))
            except ValueError:
                pass
    return statistics.mean(vals) if vals else None


def ate(gt, est):
    if not (os.path.exists(gt) and os.path.exists(est)):
        return None
    try:
        out = subprocess.run([sys.executable, os.path.join(ROOT_DIR, "scripts", "eval_tum_se3_v2.py"),
                              gt, est], capture_output=True, text=True, timeout=300).stdout
    except Exception:
        return None
    m = re.search(r"rmse=([0-9.]+)", out)
    return float(m.group(1)) if m else None


def traj_lines(path):
    if not os.path.exists(path):
        return 0
    return sum(1 for line in open(path) if line.strip() and not line.startswith("#"))


def collect(out_root, gt, n_input, do_lpips=True):
    rows = []
    for round_dir in sorted(os.listdir(os.path.join(out_root, "runs"))):
        cond, r = round_dir.rsplit("_r", 1)
        run = os.path.join(out_root, "runs", round_dir)
        meta = parse_meta(os.path.join(run, "run_meta.txt"))
        met = parse_metrics(os.path.join(out_root, "logs", round_dir + ".log"))
        snap = latest_snapshot(run)
        est = os.path.join(run, "CameraTrajectory_TUM.txt")
        row = dict(condition=cond, round=int(r), snap=snap,
                   exit_code=meta.get("exit_code"),
                   wall_seconds=float(meta["wall_seconds"]) if "wall_seconds" in meta else None,
                   peak_vram_mib=int(meta["peak_vram_mib"]) if meta.get("peak_vram_mib", "NA") != "NA" else None,
                   psnr=mean_column(os.path.join(snap, "psnr.txt")) if snap else None,
                   ssim=mean_column(os.path.join(snap, "dssim.txt")) if snap else None,
                   ate=ate(gt, est),
                   coverage=(traj_lines(est) / n_input) if n_input else None,
                   lpips=None)
        row.update(met)
        if do_lpips and snap:
            try:
                out = subprocess.run([sys.executable, os.path.join(ROOT_DIR, "scripts", "eval_snapshot_lpips.py"),
                                      snap, "--device", "cpu"],
                                     capture_output=True, text=True, timeout=3600).stdout
                m = re.search(r"lpips=([0-9.]+)", out)
                row["lpips"] = float(m.group(1)) if m else None
            except Exception:
                row["lpips"] = None
        rows.append(row)
    return rows


def fmt(v, nd=4):
    return "NA" if v is None else f"{v:.{nd}f}"


def main(argv):
    out_root = argv[0] if argv else os.path.join("results", "quad_office0")
    gt = argv[1] if len(argv) > 1 else None
    n_input = int(argv[2]) if len(argv) > 2 else 0
    rows = collect(out_root, gt, n_input)
    cols = ["condition", "round", "exit_code", "wall_seconds", "mapping_seconds",
            "optimizer_steps", "kfs", "gaussians", "line_gaussians", "psnr",
            "ssim", "lpips", "ate", "coverage", "peak_vram_mib"]
    with open(os.path.join(out_root, "per_run.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    lines = ["# office0 four-condition comparison", ""]
    lines.append("Protocol: full-sequence online runs; the evaluation snapshot is the final "
                 "`<iter>_shutdown` render of every keyframe (training-view pose and colour), "
                 "PSNR/SSIM from the binary's own `psnr.txt`/`dssim.txt` (same loss_utils code in "
                 "both builds), LPIPS offline on the same saved pairs (CPU, not in mapping time), "
                 "SE(3) ATE from `CameraTrajectory_TUM.txt` vs `pose_TUM.txt`, coverage = written "
                 "poses / input frames. Conditions are cross-process runs of an online front end: "
                 "they do NOT share frozen geometry, keyframes or update counts.")
    lines.append("")
    lines.append("## Per-run")
    lines.append("")
    lines.append("| cond | r | exit | wall s | map s | opt steps | KF | Gauss | line G | PSNR | SSIM | LPIPS | ATE | cov | VRAM MiB |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            row["condition"], row["round"], row.get("exit_code", "NA"),
            fmt(row.get("wall_seconds"), 1), fmt(row.get("mapping_seconds"), 1),
            row.get("optimizer_steps", "NA"), row.get("kfs", "NA"),
            row.get("gaussians", "NA"), row.get("line_gaussians", "NA"),
            fmt(row.get("psnr")), fmt(row.get("ssim")), fmt(row.get("lpips")),
            fmt(row.get("ate")), fmt(row.get("coverage"), 3),
            row.get("peak_vram_mib", "NA")))

    def agg(cond, key):
        vals = [r[key] for r in rows if r["condition"] == cond and r.get(key) is not None]
        if not vals:
            return None, None
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    lines.append("")
    lines.append("## Mean ± sd over rounds")
    lines.append("")
    keys = ["psnr", "ssim", "lpips", "ate", "coverage", "wall_seconds",
            "mapping_seconds", "optimizer_steps", "kfs", "gaussians", "peak_vram_mib"]
    lines.append("| cond | " + " | ".join(keys) + " |")
    lines.append("|---" * (len(keys) + 1) + "|")
    for cond in CONDS:
        cells = []
        for k in keys:
            m, s = agg(cond, k)
            cells.append("NA" if m is None else f"{m:.4g} ± {s:.3g}")
        lines.append(f"| {cond} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## Requested comparisons (mean of per-round differences)")
    lines.append("")
    lines.append("| comparison | ΔPSNR | ΔSSIM | ΔLPIPS | ΔATE | Δwall s | Δmapping s |")
    lines.append("|---|---|---|---|---|---|---|")
    for a, b in PAIRS:
        def dscore(k):
            diffs = []
            for r in sorted({x["round"] for x in rows}):
                va = next((x.get(k) for x in rows if x["condition"] == a and x["round"] == r), None)
                vb = next((x.get(k) for x in rows if x["condition"] == b and x["round"] == r), None)
                if va is not None and vb is not None:
                    diffs.append(va - vb)
            if not diffs:
                return "NA"
            sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
            return f"{statistics.mean(diffs):+.4g} ± {sd:.3g}"
        lines.append("| {} − {} | {} | {} | {} | {} | {} | {} |".format(
            a, b, dscore("psnr"), dscore("ssim"), dscore("lpips"), dscore("ate"),
            dscore("wall_seconds"), dscore("mapping_seconds")))

    text = "\n".join(lines) + "\n"
    with open(os.path.join(out_root, "summary.md"), "w") as f:
        f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
