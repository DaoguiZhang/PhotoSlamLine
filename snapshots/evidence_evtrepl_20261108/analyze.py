#!/usr/bin/env python3
"""Aggregate the common-input replay runs.

Reads  replay/<scene>_s<seed>/<condition>/{steps.csv,event_log.csv,summary.txt}
and writes:
  report/all_steps.csv      one row per (scene, seed, condition, checkpoint)
  report/all_events.csv     one row per (scene, seed, condition, event)
  report/input_identity.csv proof that all conditions of a seed used the same input
  report/mechanism.csv      per (scene, seed, condition) mechanism coverage counts
  report/deltas.csv         V-A, N-A, VN-A, VN-V, VN-N, A_repeat-A per seed
  report/deltas_summary.csv mean / sd / direction per contrast
No numbers are typed by hand here; everything is derived from the run outputs.
"""
import csv
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
REPLAY = os.path.join(ROOT, "replay")
REPORT = os.path.join(ROOT, "report")
os.makedirs(REPORT, exist_ok=True)

SCENES = ["office0_mono", "desk3_rgbd"]
SEEDS = [0, 1, 2]
CONDS = ["A", "A_repeat", "V", "N", "VN"]
CHECKPOINTS = [0, 100, 500, 1000, 3000, 5000]

steps_rows, event_rows, mech_rows, ident_rows = [], [], [], []

for scene in SCENES:
    for seed in SEEDS:
        run = os.path.join(REPLAY, f"{scene}_s{seed}")
        if not os.path.isdir(run):
            print(f"missing run dir {run}", file=sys.stderr)
            continue
        for cond in CONDS:
            cdir = os.path.join(run, cond)
            sp = os.path.join(cdir, "steps.csv")
            if not os.path.isfile(sp):
                print(f"missing {sp}", file=sys.stderr)
                continue
            with open(sp) as f:
                for r in csv.DictReader(f):
                    row = {k: v for k, v in r.items() if k not in ("condition", "v", "n")}
                    row.update(scene=scene, seed=seed, condition=cond,
                               v=r.get("v", ""), n=r.get("n", ""))
                    steps_rows.append(row)

            summ = {}
            sump = os.path.join(cdir, "summary.txt")
            if os.path.isfile(sump):
                for line in open(sump):
                    if "=" in line:
                        k, v = line.strip().split("=", 1)
                        summ[k] = v
            ident_rows.append(dict(
                scene=scene, seed=seed, condition=cond,
                input_snapshot_hash=summ.get("input_snapshot_hash", ""),
                kf_sequence_hash=summ.get("kf_sequence_hash", ""),
                v=summ.get("v", ""), n=summ.get("n", ""),
                v_effective=summ.get("v_effective", ""),
                n_effective=summ.get("n_effective", ""),
                opt_steps=summ.get("opt_steps", ""),
                requested_steps=summ.get("requested_steps", ""),
                events_applied=summ.get("events_applied", ""),
                events_in_stream=summ.get("events_in_stream", ""),
                gaussians_init=summ.get("gaussians_init", ""),
                gaussians_final=summ.get("gaussians_final", ""),
                gaussians_final_line=summ.get("gaussians_final_line", ""),
                finite=summ.get("finite", ""),
                mapping_ms=summ.get("mapping_ms_incl_events_and_voxel", ""),
                voxel_rebuilds=summ.get("voxel_rebuilds", ""),
                voxel_invalidations=summ.get("voxel_invalidations", ""),
                voxel_rebuild_points=summ.get("voxel_rebuild_points", ""),
                voxel_rebuild_ms=summ.get("voxel_rebuild_ms", ""),
            ))

            ep = os.path.join(cdir, "event_log.csv")
            geo = {"LoopClosingBA": 0, "GlobalBAWriteback": 0, "MapMerge": 0,
                   "ScaleRefinement": 0}
            pts_tf = line_tf = kf_corr = kf_skip = 0
            vox_reb = vox_pts = 0
            vox_ms = ev_ms = 0.0
            n_ev = n_nonfinite = 0
            if os.path.isfile(ep):
                with open(ep) as f:
                    for r in csv.DictReader(f):
                        n_ev += 1
                        r.update(scene=scene, seed=seed, condition=cond)
                        event_rows.append(r)
                        t = r["type"]
                        if t in geo:
                            geo[t] += 1
                        pts_tf += int(r["points_transformed"])
                        line_tf += int(r["line_transformed"])
                        kf_corr += int(r["kfs_corrected"])
                        kf_skip += int(r["kfs_skipped"])
                        vox_reb += int(r["voxel_rebuilds"])
                        vox_pts += int(r["voxel_points"])
                        vox_ms += float(r["voxel_ms"])
                        ev_ms += float(r["event_ms"])
                        n_nonfinite += int(r["non_finite"])
            mech_rows.append(dict(
                scene=scene, seed=seed, condition=cond,
                events=n_ev,
                loop=geo["LoopClosingBA"], gba=geo["GlobalBAWriteback"],
                merge=geo["MapMerge"], scale=geo["ScaleRefinement"],
                kfs_corrected=kf_corr, kfs_skipped=kf_skip,
                points_transformed=pts_tf, line_gaussians_transformed=line_tf,
                voxel_rebuilds=vox_reb, voxel_points=vox_pts,
                voxel_ms=round(vox_ms, 2), event_ms=round(ev_ms, 2),
                non_finite=n_nonfinite,
            ))


def dump(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


dump(os.path.join(REPORT, "all_steps.csv"), steps_rows,
     ["scene", "seed", "condition", "v", "n", "step", "psnr_heldout",
      "ssim_heldout", "psnr_train", "ssim_train", "gaussians",
      "gaussians_point", "gaussians_line", "cum_mapping_ms", "eval_ms",
      "is_key_event", "event_seq"])
dump(os.path.join(REPORT, "all_events.csv"), event_rows,
     ["scene", "seed", "condition", "seq", "type", "replay_step", "v", "n",
      "ev_kfs", "ev_points", "ev_line_samples", "gauss_before", "gauss_after",
      "gauss_delta", "line_before", "line_after", "kfs_created",
      "kfs_corrected", "kfs_skipped", "points_transformed", "line_transformed",
      "non_finite", "voxel_rebuilds", "voxel_invalidations", "voxel_points",
      "voxel_ms", "event_ms", "finite_ok", "have_eval", "psnr_eval_before",
      "psnr_eval_after", "ssim_eval_before", "ssim_eval_after"])
dump(os.path.join(REPORT, "input_identity.csv"), ident_rows,
     ["scene", "seed", "condition", "v", "n", "v_effective", "n_effective",
      "input_snapshot_hash", "kf_sequence_hash", "opt_steps",
      "requested_steps", "events_applied", "events_in_stream",
      "gaussians_init", "gaussians_final", "gaussians_final_line", "finite",
      "mapping_ms", "voxel_rebuilds", "voxel_invalidations",
      "voxel_rebuild_points", "voxel_rebuild_ms"])
dump(os.path.join(REPORT, "mechanism.csv"), mech_rows,
     ["scene", "seed", "condition", "events", "loop", "gba", "merge", "scale",
      "kfs_corrected", "kfs_skipped", "points_transformed",
      "line_gaussians_transformed", "voxel_rebuilds", "voxel_points",
      "voxel_ms", "event_ms", "non_finite"])

# ---------------- contrasts ------------------------------------------------
# value lookup: (scene, seed, cond, step) -> metric
val = {}
gauss = {}
cumms = {}
for r in steps_rows:
    key = (r["scene"], int(r["seed"]), r["condition"], int(r["step"]))
    val[key] = (float(r["psnr_heldout"]), float(r["ssim_heldout"]),
                float(r["psnr_train"]))
    gauss[(r["scene"], int(r["seed"]), r["condition"])] = (
        int(r["gaussians"]), int(r["gaussians_line"]))
    cumms[(r["scene"], int(r["seed"]), r["condition"], int(r["step"]))] = float(
        r["cum_mapping_ms"])

CONTRASTS = [("V", "A"), ("N", "A"), ("VN", "A"), ("VN", "V"), ("VN", "N"),
             ("A_repeat", "A")]
delta_rows = []
for scene in SCENES:
    for seed in SEEDS:
        for step in CHECKPOINTS:
            base_ok = (scene, seed, "A", step) in val
            for a, b in CONTRASTS:
                ka, kb = (scene, seed, a, step), (scene, seed, b, step)
                if ka not in val or kb not in val:
                    continue
                delta_rows.append(dict(
                    scene=scene, seed=seed, step=step, contrast=f"{a}-{b}",
                    d_psnr_heldout=round(val[ka][0] - val[kb][0], 4),
                    d_ssim_heldout=round(val[ka][1] - val[kb][1], 5),
                    d_psnr_train=round(val[ka][2] - val[kb][2], 4),
                    d_cum_mapping_ms=round(cumms[ka] - cumms[kb], 1),
                ))
dump(os.path.join(REPORT, "deltas.csv"), delta_rows,
     ["scene", "seed", "step", "contrast", "d_psnr_heldout", "d_ssim_heldout",
      "d_psnr_train", "d_cum_mapping_ms"])

summary_rows = []
for scene in SCENES:
    for step in CHECKPOINTS:
        for a, b in CONTRASTS:
            sel = [d for d in delta_rows
                   if d["scene"] == scene and d["step"] == step
                   and d["contrast"] == f"{a}-{b}"]
            if not sel:
                continue
            ps = [d["d_psnr_heldout"] for d in sel]
            pt = [d["d_psnr_train"] for d in sel]
            ds = [d["d_ssim_heldout"] for d in sel]
            tm = [d["d_cum_mapping_ms"] for d in sel]
            pos = sum(1 for x in ps if x > 0)
            neg = sum(1 for x in ps if x < 0)
            summary_rows.append(dict(
                scene=scene, step=step, contrast=f"{a}-{b}", n=len(sel),
                mean_psnr_heldout=round(st.mean(ps), 4),
                sd_psnr_heldout=round(st.stdev(ps), 4) if len(ps) > 1 else 0.0,
                mean_ssim_heldout=round(st.mean(ds), 5),
                mean_psnr_train=round(st.mean(pt), 4),
                seeds_pos=pos, seeds_neg=neg,
                mean_d_mapping_ms=round(st.mean(tm), 1),
                per_seed="|".join(f"{x:+.4f}" for x in ps),
            ))
dump(os.path.join(REPORT, "deltas_summary.csv"), summary_rows,
     ["scene", "step", "contrast", "n", "mean_psnr_heldout", "sd_psnr_heldout",
      "mean_ssim_heldout", "mean_psnr_train", "seeds_pos", "seeds_neg",
      "mean_d_mapping_ms", "per_seed"])

# ---------------- gaussian counts ------------------------------------------
gc_rows = []
for scene in SCENES:
    for seed in SEEDS:
        for cond in CONDS:
            if (scene, seed, cond) in gauss:
                g, l = gauss[(scene, seed, cond)]
                gc_rows.append(dict(scene=scene, seed=seed, condition=cond,
                                    gaussians_final=g, line_final=l))
dump(os.path.join(REPORT, "gaussian_counts.csv"), gc_rows,
     ["scene", "seed", "condition", "gaussians_final", "line_final"])

print(f"steps rows   {len(steps_rows)}")
print(f"event rows   {len(event_rows)}")
print(f"identity     {len(ident_rows)}")
print(f"mechanism    {len(mech_rows)}")
print(f"delta rows   {len(delta_rows)}")
print(f"summary rows {len(summary_rows)}")
