#!/usr/bin/env python3
"""Turn the frozen matrix into the report tables, from one version of per-run data.

Rules that this script implements, all fixed before the runs finished:

* **Every table comes from the per-run rows.** No number is typed in by hand and
  no table is assembled from a different pass over the data. The chain is
  run directory -> eval_tum_se3_v2.py -> eval/coverage.py -> coverage_per_run.csv
  -> this script.
* **Repeats are process repeats, not seeds.** Each cell has 3 independent
  processes. Nothing in the frozen source consumes a seed for the SLAM run, so
  "seed" is never claimed. Three values are reported raw (min/median/max), not
  collapsed into a mean with an invented error bar.
* **Frames are not replicates.** A run contributes one number per metric; the
  thousands of frames inside it are the sample over which that number is defined,
  not independent repetitions.
* **Same repeat index is not a pairing.** rep1 of one arm has no special
  correspondence with rep1 of another; comparisons use the medians over repeats,
  and the ranges are shown, so overlapping ranges are visible instead of being
  declared equivalent.
* **Scenes are weighted equally** inside a dataset, regardless of how many frames
  or how long a scene is.
* **Failures are kept.** Every terminal state other than COMPLETED appears in the
  per-run table with its exit code; a scene whose runs failed contributes no
  metric and is counted as failed rather than silently dropped.
* **Upstream is a separate table.** Different pipeline (full Gaussian), so no
  internal number is mixed into it and none of its cost numbers enter the
  SLAM-only table.

Usage:
  python3 eval/aggregate.py [--root /workspace/eval_runs/final_runs]
"""
import argparse
import csv
import json
import os
import statistics
import subprocess
import sys

METRICS = [
    # name, column in coverage_per_run.csv
    ("accepted_cov", "accepted_cov"),
    ("correct_cov_0.05", "correct_cov_0.05"),
    ("correct_cov_0.10", "correct_cov_0.10"),
    ("correct_cov_0.20", "correct_cov_0.20"),
    ("correct_trans_only_0.20", "correct_trans_only_0.20"),
    ("correct_lower_bound_0.20", "correct_lower_bound_0.20"),
    ("longest_continuous_s", "longest_continuous_s"),
    ("ate_rmse_m", "ate_rmse_m"),
    ("rot_rmse_deg", "rot_rmse_deg"),
    ("align_scale", "align_scale"),
]


def sh(cmd):
    return subprocess.call(cmd)




def extra_tables(out):
    """ONLINE / FINAL tables, built from the files online_eval.py wrote.

    ONLINE  : per input frame, per-epoch alignment (eval/online_eval.py)
    FINAL   : per map, keyframe-conditional
    Both keep the raw per-run rows; the summaries use the median over the three
    process repeats and weight scenes equally, exactly like PER_DATASET.csv.
    """
    import statistics
    def read(name):
        p = os.path.join(out, name)
        if not os.path.exists(p):
            return []
        with open(p, newline="") as f:
            return list(csv.DictReader(f))

    def fnum(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    online = read("ONLINE_PER_RUN.csv")
    if online:
        keys = [k for k in ("online_correct_cov_0.05", "online_correct_cov_0.10",
                            "online_correct_cov_0.20", "online_correct_cov_0.50",
                            "online_correct_trans_only_0.20",
                            "online_longest_continuous_s",
                            "ate_rmse_pooled_over_epochs",
                            "ate_rmse_mean_of_epoch_rmse",
                            "rpe1_trans_rms_pooled", "rpe30_trans_rms_pooled",
                            "pose_valid_frac", "n_epochs") if k in online[0]]
        # per scene x arm: median over repeats
        # Only runs whose map history is trustworthy enter the headline table;
        # the others are counted and named instead of being averaged in.
        usable = [r for r in online if r.get("metric_status") == "OK"]
        excluded = len(online) - len(usable)
        g = {}
        for r in usable:
            g.setdefault((r["dataset"], r["scene"], r["modality"], r["arm"]),
                         []).append(r)
        per_scene = []
        for k in sorted(g):
            rows = g[k]
            rec = {"dataset": k[0], "scene": k[1], "modality": k[2],
                   "arm": k[3], "n_reps": len(rows)}
            for m in keys:
                vals = [fnum(r.get(m)) for r in rows]
                vals = [v for v in vals if v is not None]
                rec[m] = ("%.6g" % statistics.median(vals)) if vals else "UNDETERMINED"
            per_scene.append(rec)
        with open(os.path.join(out, "ONLINE_PER_SCENE.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_scene[0].keys()))
            w.writeheader()
            for r in per_scene:
                w.writerow(r)
        print("ONLINE headline uses %d runs; %d excluded (map history / crash / "
              "no GT)" % (len(usable), excluded))
        d = {}
        for r in per_scene:
            d.setdefault((r["dataset"], r["modality"], r["arm"]), []).append(r)
        per_ds = []
        for k in sorted(d):
            rec = {"dataset": k[0], "modality": k[1], "arm": k[2],
                   "n_scenes": len(d[k])}
            for m in keys:
                vals = [fnum(r[m]) for r in d[k]]
                vals = [v for v in vals if v is not None]
                rec[m + "_scene_equal"] = ("%.6g" % statistics.fmean(vals)) if vals else "UNDETERMINED"
            per_ds.append(rec)
        with open(os.path.join(out, "ONLINE_PER_DATASET.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_ds[0].keys()))
            w.writeheader()
            for r in per_ds:
                w.writerow(r)

    fin = read("FINAL_PER_MAP.csv")
    if fin:
        keys = [k for k in ("n_kf", "n_pairs", "ate_rmse_m", "rot_rmse_deg",
                            "align_scale", "rpe1_trans_rms_m", "rpe30_trans_rms_m",
                            "rpe1_dt_s", "rpe30_dt_s")
                if k in fin[0]]
        # Step 1: ONE row per run, summarising that run's own maps. Maps inside a
        # run are not independent repeats, so they are collapsed here (median over
        # the run's maps, with the map count kept) rather than pooled across
        # repeats.
        per_run = {}
        for r in fin:
            per_run.setdefault((r["dataset"], r["scene"], r["modality"],
                                r["arm"], r["repeat"]), []).append(r)
        run_rows = []
        for k in sorted(per_run):
            maps = per_run[k]
            okm = [m for m in maps if m.get("status") == "OK"]
            rec = {"dataset": k[0], "scene": k[1], "modality": k[2], "arm": k[3],
                   "repeat": k[4], "scope": "KEYFRAME_CONDITIONAL",
                   "n_maps": len(maps), "n_maps_evaluable": len(okm),
                   "n_maps_failed": len(maps) - len(okm)}
            for m in keys:
                vals = [fnum(x.get(m)) for x in okm]
                vals = [v for v in vals if v is not None]
                rec[m + "_median_over_maps_in_run"] = (
                    "%.6g" % statistics.median(vals)) if vals else "UNDETERMINED"
            run_rows.append(rec)
        with open(os.path.join(out, "FINAL_PER_RUN.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(run_rows[0].keys()))
            w.writeheader()
            for r in run_rows:
                w.writerow(r)
        # Step 2: median/min/max over the three process repeats of that per-run
        # value.
        g = {}
        for r in run_rows:
            g.setdefault((r["dataset"], r["scene"], r["modality"], r["arm"]),
                         []).append(r)
        rows = []
        for k in sorted(g):
            rec = {"dataset": k[0], "scene": k[1], "modality": k[2], "arm": k[3],
                   "scope": "KEYFRAME_CONDITIONAL", "n_reps": len(g[k])}
            for m in keys:
                col = m + "_median_over_maps_in_run"
                vals = [fnum(r.get(col)) for r in g[k]]
                vals = [v for v in vals if v is not None]
                if vals:
                    rec[m + "_values"] = "|".join("%.6g" % v for v in vals)
                    rec[m + "_median"] = "%.6g" % statistics.median(vals)
                    rec[m + "_min"] = "%.6g" % min(vals)
                    rec[m + "_max"] = "%.6g" % max(vals)
                else:
                    for suffix in ("_values", "_median", "_min", "_max"):
                        rec[m + suffix] = "UNDETERMINED"
            rows.append(rec)
        with open(os.path.join(out, "FINAL_CONDITIONAL.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    return len(online), len(fin)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/eval_runs/final_runs_v2")
    ap.add_argument("--out", default=None,
                    help="table directory; defaults to <root>/metrics")
    a = ap.parse_args()
    root = a.root
    out = a.out or os.path.join(root, "metrics")
    os.makedirs(out, exist_ok=True)

    # 1. per-run metrics, recomputed from the run directories themselves
    cov_csv = os.path.join(out, "coverage_per_run.csv")
    rc = sh([sys.executable, "eval/coverage.py", "--root", root, "--out", out])
    if rc != 0 or not os.path.exists(cov_csv):
        print("coverage.py failed (rc=%d); no tables written" % rc)
        return 1
    cov = list(csv.DictReader(open(cov_csv)))

    # 2. per-run table, failures included
    per_run = []
    for r in cov:
        row = {"dataset": r["dataset"], "scene": r["scene"],
               "modality": r["modality"], "arm": r["arm"], "repeat": r["repeat"],
               "status": r["status"], "exit_code": r["exit_code"],
               "n_input": r["n_input"], "n_accepted": r["n_accepted"],
               "n_gt_evaluable": r["n_gt_evaluable"],
               "consistency": r["consistency"]}
        for name, col in METRICS:
            row[name] = r.get(col, "")
        per_run.append(row)
    with open(os.path.join(out, "PER_RUN.csv"), "w", newline="") as f:
        keys = ["dataset", "scene", "modality", "arm", "repeat", "status",
                "exit_code", "n_input", "n_accepted", "n_gt_evaluable",
                "consistency"] + [n for n, _ in METRICS]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in per_run:
            w.writerow({k: r[k] for k in keys})

    # 3. per-scene x arm: median and range over the repeats
    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    groups = {}
    for r in per_run:
        groups.setdefault((r["dataset"], r["scene"], r["modality"], r["arm"]),
                          []).append(r)
    per_scene = []
    for k in sorted(groups):
        g = sorted(groups[k], key=lambda r: int(str(r["repeat"]).replace("rep", "")))
        ok = [r for r in g if r["status"] == "OK"]
        rec = {"dataset": k[0], "scene": k[1], "modality": k[2], "arm": k[3],
               "n_reps": len(g), "n_ok": len(ok),
               "statuses": "|".join(r["status"] for r in g)}
        for name, _ in METRICS:
            vals = [num(r[name]) for r in ok]
            vals = [v for v in vals if v is not None]
            if vals:
                rec[name + "_values"] = "|".join("%.6g" % v for v in vals)
                rec[name + "_median"] = "%.6g" % statistics.median(vals)
                rec[name + "_min"] = "%.6g" % min(vals)
                rec[name + "_max"] = "%.6g" % max(vals)
            else:
                rec[name + "_values"] = rec[name + "_median"] = "UNDETERMINED"
                rec[name + "_min"] = rec[name + "_max"] = "UNDETERMINED"
        per_scene.append(rec)
    keys2 = (["dataset", "scene", "modality", "arm", "n_reps", "n_ok",
              "statuses"] +
             [x for n, _ in METRICS for x in (n + "_values", n + "_median",
                                              n + "_min", n + "_max")])
    with open(os.path.join(out, "PER_SCENE.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys2)
        w.writeheader()
        for r in per_scene:
            w.writerow(r)

    # 4. dataset x modality x arm, scenes weighted equally
    dg = {}
    for r in per_scene:
        dg.setdefault((r["dataset"], r["modality"], r["arm"]), []).append(r)
    per_dataset = []
    for k in sorted(dg):
        g = dg[k]
        rec = {"dataset": k[0], "modality": k[1], "arm": k[2],
               "n_scenes": len(g),
               "n_scenes_with_metrics": sum(1 for r in g
                                            if r["n_ok"] > 0)}
        for name, _ in METRICS:
            vals = []
            for r in g:
                v = num(r[name + "_median"])
                if v is not None:
                    vals.append(v)
            if vals:
                rec[name + "_scene_equal_median_of_medians"] = "%.6g" % (
                    statistics.fmean(vals))
                rec[name + "_scene_min_of_medians"] = "%.6g" % min(vals)
                rec[name + "_scene_max_of_medians"] = "%.6g" % max(vals)
            else:
                rec[name + "_scene_equal_median_of_medians"] = "UNDETERMINED"
                rec[name + "_scene_min_of_medians"] = "UNDETERMINED"
                rec[name + "_scene_max_of_medians"] = "UNDETERMINED"
        per_dataset.append(rec)
    keys3 = (["dataset", "modality", "arm", "n_scenes",
              "n_scenes_with_metrics"] +
             [n + s for n, _ in METRICS
              for s in ("_scene_equal_median_of_medians", "_scene_min_of_medians",
                        "_scene_max_of_medians")])
    with open(os.path.join(out, "PER_DATASET.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys3)
        w.writeheader()
        for r in per_dataset:
            w.writerow(r)

    # 5. attempt ledger, so cost/exit status travel with the accuracy tables
    sh([sys.executable, "scripts/matrix.py", "ledger"])

    n_on, n_fin = extra_tables(out)
    print("wrote %s/{PER_RUN,PER_SCENE,PER_DATASET}.csv" % out)
    print("wrote %s/{ONLINE_PER_SCENE,ONLINE_PER_DATASET,FINAL_CONDITIONAL}.csv "
          "from online=%d final=%d rows" % (out, n_on, n_fin))
    print("rows: per_run=%d per_scene=%d per_dataset=%d"
          % (len(per_run), len(per_scene), len(per_dataset)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
