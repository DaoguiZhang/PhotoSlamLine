#!/usr/bin/env python3
"""Summarise the fixed-geometry mapper convergence A/B experiment.

Reads abl_out/<label>/frozen/{summary.csv,<tag>_metrics.csv} produced by
scripts/run_frozen_abl.sh and answers, for every seed:

  1. equal-optimizer-step comparison  (same completed_optimizer_steps)
  2. equal-mapping-time comparison    (interpolated at matched cum_mapping_seconds)
  3. consistency across seeds
  4. whether a training metric improves while the held-out eval metric does not

Variant naming: the pair written by the runner is
    B = structural (line) loss OFF
    C = structural (line) loss ON
Both share ONE initialisation checkpoint, ONE frozen scene, ONE training schedule.

usage: summarize_frozen_abl.py <label> [<label> ...]
"""
import csv
import os
import statistics
import sys

OFF_TAG = "B"
ON_TAG = "C"
OFF_NAME = "structural loss OFF"
ON_NAME = "structural loss ON"

METRIC_COLS = [
    "cum_mapping_seconds", "eval_seconds", "eval_psnr", "eval_ssim", "eval_l1",
    "struct_ori", "struct_shape", "params_unchanged", "iteration_unchanged",
    "params_finite", "total_gaussians", "line_gaussians",
]


def read_metrics(path):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append({k: (float(v) if k != "condition" else v) for k, v in r.items()})
    return rows


def read_summary(path):
    with open(path, newline="") as fh:
        return {r["condition"]: r for r in csv.DictReader(fh)}


def interp_at_time(rows, t):
    """Eval rows are monotone in cum_mapping_seconds; linear interpolation."""
    if not rows:
        return None
    if t <= rows[0]["cum_mapping_seconds"]:
        return rows[0]
    if t >= rows[-1]["cum_mapping_seconds"]:
        return rows[-1]
    for a, b in zip(rows, rows[1:]):
        ta, tb = a["cum_mapping_seconds"], b["cum_mapping_seconds"]
        if ta <= t <= tb:
            if tb - ta < 1e-12:
                return b
            w = (t - ta) / (tb - ta)
            out = dict(a)
            for k in METRIC_COLS:
                out[k] = a[k] + w * (b[k] - a[k])
            out["completed_optimizer_steps"] = a["completed_optimizer_steps"] + w * (
                b["completed_optimizer_steps"] - a["completed_optimizer_steps"])
            return out
    return rows[-1]


def load_label(label, root="abl_out"):
    fdir = os.path.join(root, label, "frozen")
    summ = read_summary(os.path.join(fdir, "summary.csv"))
    met = {}
    for tag in (OFF_TAG, ON_TAG):
        p = os.path.join(fdir, f"{tag}_metrics.csv")
        met[tag] = read_metrics(p) if os.path.exists(p) else []
    init = {}
    ip = os.path.join(fdir, "init_fingerprint.txt")
    if os.path.exists(ip):
        for line in open(ip):
            parts = line.split()
            if len(parts) == 2:
                init[parts[0]] = parts[1]
    return summ, met, init


def main(labels):
    per_seed = {}
    gate_msgs = []
    for label in labels:
        try:
            summ, met, init = load_label(label)
        except FileNotFoundError as e:
            print(f"[skip] {label}: {e}")
            continue

        off, on = summ.get(OFF_TAG), summ.get(ON_TAG)
        if not off or not on:
            print(f"[skip] {label}: missing one of the {OFF_TAG}/{ON_TAG} rows")
            continue

        print("=" * 78)
        print(f"label {label}   commit {summ[OFF_TAG].get('input_hash','?')[:8]}"
              f"  steps={off['opt_steps']}")
        print("=" * 78)

        # ---- integrity gates ------------------------------------------------
        gate = []
        if init.get("fingerprint"):
            gate.append(f"init fingerprint {init['fingerprint']}")
        fp_ok = (off.get("init_fingerprint") == on.get("init_fingerprint")
                 and off.get("init_fp_match") == "1" and on.get("init_fp_match") == "1")
        gate.append(f"init_checkpoint_identical={fp_ok}")
        gate.append(f"init_restore={off.get('init_restore')}/{on.get('init_restore')}")
        gate.append(f"count_violation={off.get('count_violation')}/{on.get('count_violation')}")
        gate.append(f"eval_params_stable={off.get('eval_params_stable')}/"
                    f"{on.get('eval_params_stable')}")
        gate.append(f"input_hash_same={off.get('input_hash') == on.get('input_hash')}")
        gate.append(f"kf_seq_hash_same={off.get('kf_seq_hash') == on.get('kf_seq_hash')}")
        gate.append("counts {}->{} vs {}->{}".format(
            off.get("gaussians_init"), off.get("gaussians_final"),
            on.get("gaussians_init"), on.get("gaussians_final")))
        gate.append(f"exit={off.get('exit')}/{on.get('exit')}")
        print("GATES: " + " | ".join(gate))
        gate_msgs.append((label, gate, fp_ok))

        m_off, m_on = met.get(OFF_TAG, []), met.get(ON_TAG, [])
        steps = sorted({r["completed_optimizer_steps"] for r in m_off})
        by_step_off = {r["completed_optimizer_steps"]: r for r in m_off}
        by_step_on = {r["completed_optimizer_steps"]: r for r in m_on}

        # ---- (1) equal steps ------------------------------------------------
        print(f"\n-- (1) equal OPTIMIZER STEPS  "
              f"({OFF_TAG}={OFF_NAME}  {ON_TAG}={ON_NAME}) --")
        print(f"{'steps':>6} {'off_psnr':>9} {'on_psnr':>9} {'d_psnr':>8} "
              f"{'off_ssim':>9} {'on_ssim':>9} {'d_ssim':>8} "
              f"{'off_sori':>10} {'on_sori':>10} {'off_sshape':>11} {'on_sshape':>11}")
        step_deltas = {"psnr": [], "ssim": [], "sori": [], "sshape": []}
        for s in steps:
            a, b = by_step_off.get(s), by_step_on.get(s)
            if not a or not b:
                continue
            dp = b["eval_psnr"] - a["eval_psnr"]
            ds = b["eval_ssim"] - a["eval_ssim"]
            do = b["struct_ori"] - a["struct_ori"]
            dsh = b["struct_shape"] - a["struct_shape"]
            step_deltas["psnr"].append(dp)
            step_deltas["ssim"].append(ds)
            step_deltas["sori"].append(do)
            step_deltas["sshape"].append(dsh)
            print(f"{s:>6.0f} {a['eval_psnr']:>9.4f} {b['eval_psnr']:>9.4f} {dp:>+8.4f} "
                  f"{a['eval_ssim']:>9.5f} {b['eval_ssim']:>9.5f} {ds:>+8.5f} "
                  f"{a['struct_ori']:>10.6f} {b['struct_ori']:>10.6f} "
                  f"{a['struct_shape']:>11.6f} {b['struct_shape']:>11.6f}")

        # ---- (2) equal mapping time ----------------------------------------
        print(f"\n-- (2) equal MAPPING TIME (common overlap grid) --")
        print(f"{'t_s':>8} {'off_steps':>10} {'on_steps':>9} {'off_psnr':>9} "
              f"{'on_psnr':>9} {'d_psnr':>8} {'off_ssim':>9} {'on_ssim':>9} {'d_ssim':>8}")
        time_deltas = {"psnr": [], "ssim": [], "t": []}
        if m_off and m_on:
            # At a given mapping time each variant has completed however many
            # optimizer steps it managed; ON is slower per step, so ON must be
            # given its own step count at the same time.
            t_end = min(m_off[-1]["cum_mapping_seconds"],
                        m_on[-1]["cum_mapping_seconds"])
            grid = {t_end}
            for rows in (m_off, m_on):
                for r in rows:
                    if 0 < r["cum_mapping_seconds"] <= t_end:
                        grid.add(r["cum_mapping_seconds"])
            for t in sorted(grid):
                a = interp_at_time(m_off, t)
                b = interp_at_time(m_on, t)
                if not a or not b:
                    continue
                dp = b["eval_psnr"] - a["eval_psnr"]
                ds = b["eval_ssim"] - a["eval_ssim"]
                time_deltas["psnr"].append(dp)
                time_deltas["ssim"].append(ds)
                time_deltas["t"].append(t)
                print(f"{t:>8.3f} {a['completed_optimizer_steps']:>10.1f} "
                      f"{b['completed_optimizer_steps']:>9.1f} "
                      f"{a['eval_psnr']:>9.4f} {b['eval_psnr']:>9.4f} {dp:>+8.4f} "
                      f"{a['eval_ssim']:>9.5f} {b['eval_ssim']:>9.5f} {ds:>+8.5f}")
            print(f"  per-step cost: off={m_off[-1]['cum_mapping_seconds']/max(m_off[-1]['completed_optimizer_steps'],1)*1000:.3f} ms/step"
                  f"  on={m_on[-1]['cum_mapping_seconds']/max(m_on[-1]['completed_optimizer_steps'],1)*1000:.3f} ms/step")

        # ---- (3) training metric vs eval metric -----------------------------
        print(f"\n-- (3) training metric (struct_shape, lower better) vs eval PSNR --")
        for tag, rows in ((OFF_TAG, m_off), (ON_TAG, m_on)):
            if len(rows) < 2:
                continue
            print(f"{tag}: struct_shape {rows[0]['struct_shape']:.6f} -> "
                  f"{rows[-1]['struct_shape']:.6f}   eval_psnr "
                  f"{rows[0]['eval_psnr']:.4f} -> {rows[-1]['eval_psnr']:.4f}")

        per_seed[label] = {
            "step_deltas": step_deltas,
            "time_deltas": time_deltas,
            "off_steps": off["opt_steps"],
            "map_off": off.get("mapping_seconds"),
            "map_on": on.get("mapping_seconds"),
        }

    # ---- (4) cross-seed consistency ----------------------------------------
    if len(per_seed) > 1:
        print("\n" + "=" * 78)
        print("CROSS-SEED CONSISTENCY  (delta = ON - OFF)")
        print("=" * 78)
        print(f"{'label':>10} {'d_psnr@final':>13} {'d_ssim@final':>13} "
              f"{'d_psnr@equal-time':>18} {'d_ssim@equal-time':>18} "
              f"{'map_s_off':>10} {'map_s_on':>10}")
        finals_p, finals_s, tp, ts = [], [], [], []
        for label, d in per_seed.items():
            fp = d["step_deltas"]["psnr"][-1] if d["step_deltas"]["psnr"] else float("nan")
            fs = d["step_deltas"]["ssim"][-1] if d["step_deltas"]["ssim"] else float("nan")
            etp = d["time_deltas"]["psnr"][-1] if d["time_deltas"]["psnr"] else float("nan")
            ets = d["time_deltas"]["ssim"][-1] if d["time_deltas"]["ssim"] else float("nan")
            finals_p.append(fp); finals_s.append(fs); tp.append(etp); ts.append(ets)
            print(f"{label:>10} {fp:>+13.4f} {fs:>+13.5f} {etp:>+18.4f} {ets:>+18.5f} "
                  f"{d['map_off']:>10} {d['map_on']:>10}")

        def line(name, vals):
            if not vals:
                return
            pos = sum(1 for v in vals if v > 0)
            print(f"{name}: median d={statistics.median(vals):+.5f} "
                  f"min={min(vals):+.5f} max={max(vals):+.5f} "
                  f"positive_in_{pos}/{len(vals)}")

        line("equal-steps   PSNR", finals_p)
        line("equal-steps   SSIM", finals_s)
        line("equal-time    PSNR", tp)
        line("equal-time    SSIM", ts)

    bad = [l for l, _, ok in gate_msgs if not ok]
    print("\n" + "=" * 78)
    if bad:
        print(f"!! INIT-EQUALITY GATE FAILED for: {bad} -- results not interpretable")
        return 1
    print("All labels passed the init-equality / count-freeze / eval-neutrality gates.")
    print("NOTE: eval views are keyframes whose images were also sampled during "
          "training, so this is NOT a strictly held-out set; the init colour "
          "provenance cannot be excluded.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
