#!/usr/bin/env python3
"""Per-item settlement of the frozen manifest: every row, every state, no gaps.

Each row is reported with the five independent facts, never merged into one:

  exit        the process exit code recorded in status.txt (or MISSING)
  input       frames processed vs the frozen input count
  artefacts   A-E of eval/acceptance.py for that directory
  history     map-history verdict (OK / AMBIGUOUS / BROKEN / NOT_CHECKED)
  metrics     metric_status from eval/online_eval.py (or NOT_EVALUATED)

and a final `valid` column that is 1 only when everything above holds. The
reason for a 0 is written out, so a reader never has to infer why a cell is
missing from a table.

Usage: python3 eval/settle.py --root /workspace/eval_runs/<root> [--out FILE]
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acceptance       # noqa: E402
import mapmeta          # noqa: E402
import map_history_check as mhc  # noqa: E402
import scene_inputs     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = a.root
    man = os.path.join(root, "MASTER_MANIFEST.csv")
    rows = list(csv.DictReader(open(man)))

    # metric_status / history from the evaluation pass, if it ran
    metrics = {}
    mp = os.path.join(root, "metrics_v4", "ONLINE_PER_RUN.csv")
    if not os.path.exists(mp):
        mp = os.path.join(root, "metrics_v3", "ONLINE_PER_RUN.csv")
    if os.path.exists(mp):
        for r in csv.DictReader(open(mp)):
            metrics[(r["dataset"], r["scene"], r["modality"], r["arm"],
                     r["repeat"])] = r

    out = []
    up_metrics = {}
    up_path = os.path.join(root, "metrics_v4", "UPSTREAM_PER_RUN.csv")
    if os.path.exists(up_path):
        for r in csv.DictReader(open(up_path)):
            rep = str(r.get("repeat", ""))
            if not rep.startswith("rep"):
                rep = "rep" + rep
            up_metrics[(r.get("scene"), rep)] = r

    for r in rows:
        d = r["out_dir"]
        rec = {"id": r["id"], "dataset": r["dataset"], "scene": r["scene"],
               "modality": r["modality"], "arm": r["system_arm"],
               "repeat": r["repeat"], "dir_exists": int(os.path.isdir(d))}
        rec["source"] = r.get("source_root", "")
        rec["role"] = r.get("role", "")
        rec["attempt_state"] = r.get("attempt_state", "")

        # --- upstream control rows ---------------------------------------
        # They run the upstream Photo-SLAM binary, which writes
        # CameraTrajectory_TUM.txt and status.txt but none of this project's
        # internal CSVs. Absence of the internal CSVs is not a crash, so these
        # rows are judged by their own products and their own evaluator.
        if r["dataset"] == "UPSTREAM" or r["system_arm"] == "UPSTREAM":
            st = os.path.join(d, "status.txt")
            code = None
            if os.path.exists(st):
                for line in open(st):
                    if line.startswith("exit_code="):
                        code = int(line.split("=", 1)[1])
            traj = os.path.exists(os.path.join(d, "CameraTrajectory_TUM.txt"))
            repk = str(r["repeat"])
            if not repk.startswith("rep"):
                repk = "rep" + repk
            m = up_metrics.get((r["scene"], repk))
            mstat = (m or {}).get("status") or (m or {}).get("metric_status") or "NOT_EVALUATED"
            rec.update({
                "exit_code": "MISSING" if code is None else code,
                "n_processed": ((m or {}).get("n_pairs", "")),
                "n_expected": "",
                "input_complete": "NA",
                "artefacts": "UPSTREAM_PRODUCTS" if traj else "NO_TRAJECTORY",
                "history": "NOT_APPLICABLE",
                "metric_status": mstat,
                "convention_ok": (m or {}).get("convention_ok", ""),
                "n_gt_evaluable": (m or {}).get("n_gt_evaluable", ""),
                "online_cov_0.20": (m or {}).get("online_correct_cov_0.20", ""),
                "ate_rmse_m": (m or {}).get("ate_rmse_m", ""),
                "align_scale": (m or {}).get("align_scale", ""),
                "rpe1_trans_rms_m": (m or {}).get("rpe1_trans_rms_m", ""),
                "upstream_commit": (m or {}).get("upstream_commit", ""),
                "history_note": "upstream binary: no map event log by design",
            })
            ok = (rec["exit_code"] == 0 and traj and mstat == "OK")
            rec["valid"] = int(bool(ok))
            why = []
            if rec["exit_code"] != 0:
                why.append("exit=%s" % rec["exit_code"])
            if not traj:
                why.append("no CameraTrajectory_TUM.txt")
            if mstat != "OK":
                why.append("metrics=%s" % mstat)
            rec["reason"] = "; ".join(why)
            out.append(rec)
            continue

        # --- exit -------------------------------------------------------
        st = os.path.join(d, "status.txt")
        code = None
        if os.path.exists(st):
            for line in open(st):
                if line.startswith("exit_code="):
                    code = int(line.split("=")[1])
        rec["exit_code"] = "MISSING" if code is None else code

        # --- input ------------------------------------------------------
        fs = os.path.join(d, "final_state.csv")
        n_proc = 0
        if os.path.exists(fs):
            with open(fs) as f:
                n_proc = max(0, sum(1 for _ in f) - 1)
        if r["dataset"] == "UPSTREAM":
            n_exp = None
            rec["input_note"] = "upstream control row: no internal input rule"
        else:
            try:
                n_exp = scene_inputs.expected_inputs(r["dataset"], r["scene"],
                                                     r["modality"])
            except Exception as e:                  # dataset not reachable
                n_exp = None
                rec["input_note"] = "expected count unavailable: %s" % e
        rec["n_processed"] = n_proc
        rec["n_expected"] = "" if n_exp is None else n_exp
        rec["input_complete"] = ("" if n_exp is None
                                 else int(n_proc == n_exp))

        # --- artefacts --------------------------------------------------
        if os.path.isdir(d) and os.path.exists(st):
            art = acceptance.check_run(d)
            bad = [k for k in "ABCDE" if art[k] != "PASS"]
            rec["artefacts"] = "PASS" if not bad else "FAIL_" + "".join(bad)
        else:
            rec["artefacts"] = "NO_RUN"

        # --- history ----------------------------------------------------
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "online_poses.csv")):
            h = mhc.check_run(d)
            rec["history"] = h["verdict"]
            rec["history_note"] = "; ".join(h["problems"] + h["notes"])[:120]
        else:
            rec["history"] = "NOT_CHECKED"
            rec["history_note"] = "online_poses.csv absent"

        # --- metrics ----------------------------------------------------
        # the per-run tables key the repeat as "repN", the manifest as "N"
        rep = r["repeat"] if str(r["repeat"]).startswith("rep") else "rep" + str(r["repeat"])
        key = (r["dataset"], r["scene"], r["modality"], r["system_arm"], rep)
        if key in metrics:
            rec["metric_status"] = metrics[key]["metric_status"]
            rec["convention_ok"] = metrics[key].get("convention_ok", "")
            rec["n_gt_evaluable"] = metrics[key].get("n_gt_evaluable", "")
            rec["online_cov_0.20"] = metrics[key].get(
                "online_correct_cov_0.20", "")
        else:
            rec["metric_status"] = "NOT_EVALUATED"
            rec["convention_ok"] = rec["n_gt_evaluable"] = ""
            rec["online_cov_0.20"] = ""

        # --- verdict ----------------------------------------------------
        ok = (rec["exit_code"] == 0
              and rec["input_complete"] == 1
              and rec["artefacts"] == "PASS"
              and rec["history"] == "OK"
              and rec["metric_status"] == "OK")
        rec["valid"] = int(bool(ok))
        reasons = []
        if rec["exit_code"] != 0:
            reasons.append("exit=%s" % rec["exit_code"])
        if rec["input_complete"] != 1:
            reasons.append("input %s/%s" % (rec["n_processed"], rec["n_expected"]))
        if rec["artefacts"] not in ("PASS",):
            reasons.append("artefacts=%s" % rec["artefacts"])
        if rec["history"] != "OK":
            reasons.append("history=%s" % rec["history"])
        if rec["metric_status"] != "OK":
            reasons.append("metrics=%s" % rec["metric_status"])
        rec["reason"] = "; ".join(reasons)
        out.append(rec)

    outp = a.out or os.path.join(root, "SETTLEMENT.csv")
    keys = ["id", "dataset", "scene", "modality", "arm", "repeat", "dir_exists",
            "source", "role", "attempt_state",
            "exit_code", "n_processed", "n_expected", "input_complete",
            "artefacts", "history", "metric_status", "convention_ok",
            "n_gt_evaluable", "online_cov_0.20", "ate_rmse_m", "align_scale",
            "rpe1_trans_rms_m", "upstream_commit", "valid", "reason",
            "history_note"]
    with open(outp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in out:
            w.writerow(r)

    import collections
    print("rows=%d valid=%d" % (len(out), sum(int(r["valid"]) for r in out)))
    for k in ("role", "exit_code", "input_complete", "artefacts", "history",
              "metric_status"):
        print("  %-14s %s" % (k, dict(collections.Counter(str(r.get(k, "")) for r in out))))
    inv = [r for r in out if not int(r["valid"])]
    print("invalid rows: %d" % len(inv))
    for r in inv[:40]:
        print("   %-52s %s" % (r["id"], r["reason"]))
    print("table=%s" % outp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
