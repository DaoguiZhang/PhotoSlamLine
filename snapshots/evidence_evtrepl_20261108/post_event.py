#!/usr/bin/env python3
"""Post-event rendering change: does the event make the map better or worse?

For every geometry event the harness evaluated the held-out views immediately
before and immediately after the consumer ran (event_log.csv columns
psnr_eval_before / psnr_eval_after). This script reports, per (scene, seed,
condition), the mean (after - before) and the same for the training views, plus
the mechanism counts that must explain it.

The only defensible claim is a *pairwise* one: within one run the two numbers
come from the same model and the same views, so the delta is not contaminated by
run-to-run trajectory differences.
"""
import csv
import os
import statistics as st

ROOT = os.path.dirname(os.path.abspath(__file__))
REPLAY = os.path.join(ROOT, "replay")
REPORT = os.path.join(ROOT, "report")
os.makedirs(REPORT, exist_ok=True)

rows = []
for scene in ["office0_mono", "desk3_rgbd"]:
    for seed in [0, 1, 2]:
        for cond in ["A", "A_repeat", "V", "N", "VN"]:
            p = os.path.join(REPLAY, f"{scene}_s{seed}", cond, "event_log.csv")
            if not os.path.isfile(p):
                continue
            with open(p) as f:
                ev = [r for r in csv.DictReader(f)]
            geo = [r for r in ev if r["have_eval"] == "1"]
            if not geo:
                rows.append(dict(scene=scene, seed=seed, condition=cond,
                                 n_geometry_events=0, note="no geometry event"))
                continue
            d = [float(r["psnr_eval_after"]) - float(r["psnr_eval_before"])
                 for r in geo]
            ds = [float(r["ssim_eval_after"]) - float(r["ssim_eval_before"])
                  for r in geo]
            pts = sum(int(r["points_transformed"]) for r in geo)
            line = sum(int(r["line_transformed"]) for r in geo)
            kfc = sum(int(r["kfs_corrected"]) for r in geo)
            rows.append(dict(
                scene=scene, seed=seed, condition=cond,
                n_geometry_events=len(geo),
                types="|".join(r["type"] for r in geo),
                steps="|".join(r["replay_step"] for r in geo),
                mean_d_psnr=round(st.mean(d), 4),
                min_d_psnr=round(min(d), 4),
                max_d_psnr=round(max(d), 4),
                mean_d_ssim=round(st.mean(ds), 5),
                kfs_corrected=kfc, points_transformed=pts,
                line_transformed=line,
                per_event="|".join(f"{x:+.3f}" for x in d),
            ))

fields = ["scene", "seed", "condition", "n_geometry_events", "types", "steps",
          "mean_d_psnr", "min_d_psnr", "max_d_psnr", "mean_d_ssim",
          "kfs_corrected", "points_transformed", "line_transformed",
          "per_event", "note"]
with open(os.path.join(REPORT, "post_event.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)

for r in rows:
    print(r["scene"], r["seed"], r["condition"],
          r.get("n_geometry_events"), r.get("mean_d_psnr"),
          "pts=" + str(r.get("points_transformed")),
          "line=" + str(r.get("line_transformed")))
