#!/usr/bin/env python3
"""Stage report: previous results next to the improvements already visible.

Reads ONLY finished runs that have a status.txt (a run still writing has none),
evaluates the finished v_final_4 runs with the same protocol as everything else,
and puts them side by side with the v_final_2 attempt of the same plan id.
Nothing here is a final table and nothing overwrites a formal artefact: the
output goes to an independent preview directory.

Classification per plan id, comparing the old attempt with the new one:

  crash_recovered      old exit != 0, new exit == 0
  history_evaluable    old metric_status was UNDETERMINED_MAP_HISTORY, new is OK
  metric_changed       both evaluable, coverage or ATE differs beyond rounding
  unchanged            both evaluable and equal within tolerance
  still_failing        new exit != 0
  pending              no new attempt yet (still queued)

Usage: python3 eval/preview.py --out /workspace/eval_runs/preview_v4
"""
import argparse
import collections
import csv
import json
import os
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import online_eval  # noqa: E402

BASE = "/workspace/eval_runs/final_runs_v2"
SUPP = "/workspace/eval_runs/supplement_v4"
EVAL = "/workspace/code/PSL-perf/scripts/eval_tum_se3_v2.py"


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(v, d=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def run_tracking_ms(run_dir):
    """Mean per-frame tracking time from the entry's own bucket report."""
    p = os.path.join(run_dir, "stdout.log")
    if not os.path.exists(p):
        return None, None
    last = None
    for line in open(p, errors="replace"):
        if "[EVAL-TIMING]" in line:
            last = line
    if not last:
        return None, None
    kv = {}
    for tok in last.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            kv[k] = v
    try:
        frames = float(kv.get("frames", 0))
        track = float(kv.get("track_ms", 0))
        pace = float(kv.get("pace_ms", 0))
        tot = float(kv.get("sum_ms", 0))
    except ValueError:
        return None, None
    return (track / frames if frames else None,
            {"track_s": track / 1000.0, "pace_s": pace / 1000.0,
             "sum_s": tot / 1000.0, "frames": frames})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/eval_runs/preview_v4")
    a = ap.parse_args()
    out = a.out
    os.makedirs(out, exist_ok=True)

    # ---- 1. evaluate the FINISHED v4 runs only --------------------------
    # online_eval skips directories without status.txt, which is exactly the
    # rule "do not score a run that is still being written".
    rc = subprocess.call([sys.executable, "eval/online_eval.py", "--root", SUPP,
                          "--out", os.path.join(out, "v4_metrics")])
    v4 = read_csv(os.path.join(out, "v4_metrics", "ONLINE_PER_RUN.csv"))
    v4_by_id = {("%s|%s|%s|%s|%s" % (r["dataset"], r["scene"], r["modality"],
                                     r["arm"], r["repeat"])): r for r in v4}
    v2 = {r["dataset"] + "|" + r["scene"] + "|" + r["modality"] + "|" + r["arm"]
          + "|" + r["repeat"]: r
          for r in read_csv(os.path.join(BASE, "metrics_v4", "ONLINE_PER_RUN.csv"))}
    old_settle = {r["id"]: r for r in read_csv(os.path.join(BASE, "SETTLEMENT.csv"))}

    cls = collections.Counter()
    rows = []
    for pid in sorted(set(v2) | set(v4_by_id)):
        o, n = v2.get(pid), v4_by_id.get(pid)
        rec = {"id": pid,
               "old_exit": (o or {}).get("exit_code", ""),
               "new_exit": (n or {}).get("exit_code", ""),
               "old_metric": (o or {}).get("metric_status", "NOT_RUN"),
               "new_metric": (n or {}).get("metric_status", "PENDING"),
               "old_cov20": (o or {}).get("online_correct_cov_0.20", ""),
               "new_cov20": (n or {}).get("online_correct_cov_0.20", ""),
               "old_ate": (o or {}).get("ate_rmse_pooled_over_epochs", ""),
               "new_ate": (n or {}).get("ate_rmse_pooled_over_epochs", ""),
               "old_rpe1": (o or {}).get("rpe1_trans_rms_pooled", ""),
               "new_rpe1": (n or {}).get("rpe1_trans_rms_pooled", ""),
               "old_rot": (o or {}).get("rot_rmse_max_epoch_deg", ""),
               "new_rot": (n or {}).get("rot_rmse_max_epoch_deg", ""),
               }
        if n is None:
            rec["class"] = "pending"
        elif str(n.get("exit_code")) != "0":
            rec["class"] = "still_failing"
        elif str(o.get("exit_code", "0")) != "0":
            rec["class"] = "crash_recovered"
        elif (o.get("metric_status") != "OK") and n.get("metric_status") == "OK":
            rec["class"] = "history_evaluable"
        elif o.get("metric_status") == "OK" and n.get("metric_status") == "OK":
            co, cn = fnum(o.get("online_correct_cov_0.20")), fnum(n.get("online_correct_cov_0.20"))
            ao, an = fnum(o.get("ate_rmse_pooled_over_epochs")), fnum(n.get("ate_rmse_pooled_over_epochs"))
            changed = ((co is not None and cn is not None and abs(co - cn) > 1e-6)
                       or (ao is not None and an is not None and abs(ao - an) > 1e-9))
            rec["class"] = "metric_changed" if changed else "unchanged"
        else:
            rec["class"] = "other"
        cls[rec["class"]] += 1
        rows.append(rec)

    with open(os.path.join(out, "PREVIEW_BY_ID.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["id"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- 2. old batch per scene, per arm (every repeat kept) -------------
    per_run = []
    for pid, r in sorted(v2.items()):
        d = os.path.join(BASE, r["dataset"], r["scene"], r["modality"], r["arm"],
                         r["repeat"])
        per_frame_ms, timing = run_tracking_ms(d)
        st = old_settle.get(pid, {})
        per_run.append({
            "id": pid, "dataset": r["dataset"], "scene": r["scene"],
            "modality": r["modality"], "arm": r["arm"], "repeat": r["repeat"],
            "exit_code": r["exit_code"], "metric_status": r["metric_status"],
            "history": st.get("history", ""),
            "cov_0.20": r.get("online_correct_cov_0.20", ""),
            "cov_trans_0.20": r.get("online_correct_trans_only_0.20", ""),
            "ate_pooled_m": r.get("ate_rmse_pooled_over_epochs", ""),
            "rpe1_trans_m": r.get("rpe1_trans_rms_pooled", ""),
            "rpe30_trans_m": r.get("rpe30_trans_rms_pooled", ""),
            "rot_max_epoch_deg": r.get("rot_rmse_max_epoch_deg", ""),
            "track_ms_per_frame": ("%.2f" % per_frame_ms) if per_frame_ms else "",
            "track_s": (timing or {}).get("track_s", ""),
            "pace_s": (timing or {}).get("pace_s", ""),
            "sum_s": (timing or {}).get("sum_s", ""),
            "n_epochs": r.get("n_epochs", ""),
            "convention_ok": r.get("convention_ok", ""),
        })
    with open(os.path.join(out, "V2_PER_RUN.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_run[0].keys()))
        w.writeheader()
        for r in per_run:
            w.writerow(r)

    # per scene x arm summary with counts
    g = collections.defaultdict(list)
    for r in per_run:
        g[(r["dataset"], r["scene"], r["modality"], r["arm"])].append(r)
    summ = []
    for k in sorted(g):
        rs = g[k]
        ok = [r for r in rs if r["metric_status"] == "OK"]
        rec = {"dataset": k[0], "scene": k[1], "modality": k[2], "arm": k[3],
               "n_reps": len(rs), "n_ok": len(ok),
               "n_crash": sum(1 for r in rs if r["metric_status"] == "CRASHED_PARTIAL"),
               "n_undetermined": sum(1 for r in rs if r["metric_status"] == "UNDETERMINED_MAP_HISTORY"),
               }
        for m in ("cov_0.20", "cov_trans_0.20", "ate_pooled_m", "rpe1_trans_m",
                  "rpe30_trans_m", "rot_max_epoch_deg"):
            vals = [fnum(r[m]) for r in ok]
            vals = [v for v in vals if v is not None]
            if vals:
                rec[m + "_values"] = "|".join("%.4g" % v for v in vals)
                rec[m + "_median"] = "%.4g" % statistics.median(vals)
                rec[m + "_range"] = "%.4g..%.4g" % (min(vals), max(vals))
            else:
                rec[m + "_values"] = rec[m + "_median"] = rec[m + "_range"] = "UNDETERMINED"
        tr = [fnum(r["track_ms_per_frame"]) for r in rs]
        tr = [v for v in tr if v is not None]
        rec["track_ms_per_frame_median"] = "%.1f" % statistics.median(tr) if tr else ""
        summ.append(rec)
    with open(os.path.join(out, "V2_PER_SCENE.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader()
        for r in summ:
            w.writerow(r)

    json.dump({"classes": dict(cls), "n_v4_finished": len(v4),
               "n_v2": len(v2)},
              open(os.path.join(out, "PREVIEW_SUMMARY.json"), "w"), indent=1)
    print("preview written to", out)
    print("  v4 finished runs evaluated: %d" % len(v4))
    print("  id classes:", dict(cls))
    return 0


if __name__ == "__main__":
    sys.exit(main())
