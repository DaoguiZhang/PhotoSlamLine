#!/usr/bin/env python3
"""Summarise the one-round three-condition budget experiment.

Reads:
  <root>/runs/<tag>/append/append_status.txt
  <root>/runs/<tag>/run_meta.txt
  <root>/common_view_summary_budget.csv
  <root>/pose_audit.csv
Writes:
  <root>/summary.csv
  <root>/summary.md
"""
import csv
import os
import sys


def read_kv(path):
    d = {}
    if not os.path.exists(path):
        return d
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        d[k.strip()] = v.strip()
    return d


def main(root):
    tags = ["P_ref", "P_fast", "L_fast"]
    meta = {}
    for tag in tags:
        rd = os.path.join(root, "runs", tag)
        meta[tag] = read_kv(os.path.join(rd, "append", "append_status.txt"))
        meta[tag].update(read_kv(os.path.join(rd, "run_meta.txt")))

    metrics = {}
    sc = os.path.join(root, "common_view_summary_budget.csv")
    if os.path.exists(sc):
        with open(sc) as f:
            for r in csv.DictReader(f):
                metrics[(r["run"], r["state"])] = r

    cols = ["run", "state", "budget_seconds", "budget_status", "budget_unmatched",
            "budget_overshoot_seconds", "cumulative_train_seconds",
            "natural_completed_steps", "append_completed_steps",
            "append_iterations", "natural_cumulative_train_seconds",
            "append_train_seconds", "wall_seconds",
            "gaussians_at_entry", "gaussians",
            "line_gaussians_at_entry", "line_gaussians",
            "psnr", "ssim", "lpips", "align_rmse"]
    rows = []
    for tag in tags:
        m = meta.get(tag, {})
        for state in ("natural_end", "budget_end"):
            mm = metrics.get((tag, state), {})
            if state == "natural_end":
                cum = m.get("natural_cumulative_train_seconds", "")
                app_steps = "0"
                app_iter = "0"
                app_train = "0"
                over = "0"
                gauss = m.get("gaussians_at_entry", "")
                lgauss = m.get("line_gaussians_at_entry", "")
            else:
                cum = m.get("cumulative_train_seconds", "")
                app_steps = m.get("append_completed_steps", "")
                app_iter = m.get("append_iterations", "")
                app_train = m.get("append_train_seconds", "")
                over = m.get("budget_overshoot_seconds", "")
                gauss = m.get("gaussians", "")
                lgauss = m.get("line_gaussians", "")
            rows.append(dict(
                run=tag, state=state,
                budget_seconds=m.get("budget_seconds", ""),
                budget_status=m.get("budget_status", ""),
                budget_unmatched=m.get("budget_unmatched", ""),
                budget_overshoot_seconds=over,
                cumulative_train_seconds=cum,
                natural_completed_steps=m.get("natural_completed_steps", ""),
                append_completed_steps=app_steps,
                append_iterations=app_iter,
                natural_cumulative_train_seconds=m.get(
                    "natural_cumulative_train_seconds", ""),
                append_train_seconds=app_train,
                wall_seconds=m.get("wall_seconds", ""),
                gaussians_at_entry=m.get("gaussians_at_entry", ""),
                gaussians=gauss,
                line_gaussians_at_entry=m.get("line_gaussians_at_entry", ""),
                line_gaussians=lgauss,
                psnr=mm.get("psnr", ""), ssim=mm.get("ssim", ""),
                lpips=mm.get("lpips", ""), align_rmse=mm.get("align_rmse", "")))

    with open(os.path.join(root, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)

    def fnum(x):
        try:
            return float(x)
        except Exception:
            return None

    lines = ["# Append / cumulative-training-budget round", ""]
    lines.append("| run | state | updates natural+append | cumulative train s | "
                 "overshoot s | wall s | gaussians | line | PSNR | SSIM | LPIPS | budget |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append("| %s | %s | %s+%s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["run"], r["state"], r["natural_completed_steps"],
            r["append_completed_steps"], r["cumulative_train_seconds"],
            r["budget_overshoot_seconds"], r["wall_seconds"],
            r["gaussians"], r["line_gaussians"],
            r["psnr"], r["ssim"], r["lpips"],
            ("pre_append" if r["state"] == "natural_end"
             else r["budget_status"] +
                  ("" if r["budget_unmatched"] != "1" else " (unmatched)"))))

    lines.append("")
    lines.append("## Same-budget paired differences (positive = first better)")
    lines.append("")
    lines.append("| pair | state | dPSNR | dSSIM | dLPIPS (first-second) |")
    lines.append("|---|---|---|---|---|")
    for a, b in [("P_fast", "P_ref"), ("L_fast", "P_ref"), ("L_fast", "P_fast")]:
        for state in ("natural_end", "budget_end"):
            ka = (a, state); kb = (b, state)
            if ka not in metrics or kb not in metrics:
                continue
            dp = fnum(metrics[ka]["psnr"]) - fnum(metrics[kb]["psnr"])
            ds = fnum(metrics[ka]["ssim"]) - fnum(metrics[kb]["ssim"])
            la, lb = fnum(metrics[ka].get("lpips")), fnum(metrics[kb].get("lpips"))
            dl = (la - lb) if (la is not None and lb is not None) else None
            dl_s = "n/a" if dl is None else "%+.5f" % dl
            lines.append("| %s-%s | %s | %+.4f | %+.5f | %s |" %
                         (a, b, state, dp, ds, dl_s))

    pa = os.path.join(root, "pose_audit.csv")
    if os.path.exists(pa):
        lines.append("")
        lines.append("## Pose / geometry audit")
        lines.append("")
        lines.append("See `pose_audit.csv` and `three_keyframe_geometry.csv`.")
        with open(pa) as f:
            for r in csv.DictReader(f):
                lines.append("- %s: natural-vs-budget pose delta %.3g, JSON-vs-TUM "
                             "delta %.3g, three-KF relative delta %.3g" % (
                                 r["run"], float(r["natural_vs_budget_max_pos"]),
                                 float(r["json_vs_tum_max_pos"]),
                                 float(r["three_rel_max_pos"])))
    lines.append("")
    lines.append("Common views: 24 GT frame ids; per-run SE(3) alignment; camera "
                 "basis C=diag(1,-1,-1); SSIM=11x11 uniform window; LPIPS=AlexNet "
                 "RGB in [-1,1]. The 24 views are NOT a strict held-out set.")
    with open(os.path.join(root, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
