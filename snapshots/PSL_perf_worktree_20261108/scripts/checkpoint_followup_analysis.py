#!/usr/bin/env python3
"""Verify offline checkpoint rescoring and summarize 1000->1500 state changes.

Inputs are the existing r0 online checkpoints and the offline scores produced by
eval_checkpoint_snapshot.  No quality metric is computed here; the C++ tool is
the only scorer.  This script only joins files and computes pose/statistics.
"""
import argparse
import csv
import math
import os
import statistics
from pathlib import Path

import numpy as np
import torch


def load_psnr(path):
    out = {}
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            fid, value = line.split()
            out[int(fid)] = float(value)
    return out


def load_cameras(path):
    return {int(r["fid"]): r for r in csv.DictReader(open(path))}


def load_fixed(path):
    return [os.path.basename(x.strip()) for x in open(path) if x.strip()]


def quat_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def relative_pose(r0, r1):
    q0 = [float(r0[k]) for k in ("qw", "qx", "qy", "qz")]
    q1 = [float(r1[k]) for k in ("qw", "qx", "qy", "qz")]
    t0 = np.array([float(r0[k]) for k in ("tx", "ty", "tz")])
    t1 = np.array([float(r1[k]) for k in ("tx", "ty", "tz")])
    R0, R1 = quat_R(q0), quat_R(q1)
    R = R0.T @ R1
    t = R0.T @ (t1 - t0)
    angle = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))
    return angle, float(np.linalg.norm(t))


def tensor_stats(sd):
    xyz = sd["0"].float()
    scaling = sd["3"].float().exp()
    opacity = torch.sigmoid(sd["4"].float()).squeeze(-1)
    is_line = sd["6"].bool()
    lo = torch.quantile(xyz, 0.01, dim=0)
    hi = torch.quantile(xyz, 0.99, dim=0)
    ext = hi - lo
    centroid = xyz.mean(0)
    dist = torch.linalg.norm(xyz - centroid, dim=1)

    def q(v, p):
        return float(torch.quantile(v, p))

    row = {
        "n": int(xyz.shape[0]),
        "line": int(is_line.sum()),
        "ext_x": float(ext[0]),
        "ext_y": float(ext[1]),
        "ext_z": float(ext[2]),
        "ext_diag": float(ext.norm()),
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "centroid_z": float(centroid[2]),
        "dist_med": q(dist, 0.5),
        "dist_p95": q(dist, 0.95),
        "scale_med_x": q(scaling[:, 0], 0.5),
        "scale_med_y": q(scaling[:, 1], 0.5),
        "scale_med_z": q(scaling[:, 2], 0.5),
        "scale_p95_x": q(scaling[:, 0], 0.95),
        "scale_p95_y": q(scaling[:, 1], 0.95),
        "scale_p95_z": q(scaling[:, 2], 0.95),
        "scale_max": float(scaling.max()),
        "opacity_mean": float(opacity.mean()),
        "opacity_med": q(opacity, 0.5),
        "opacity_p05": q(opacity, 0.05),
        "opacity_p95": q(opacity, 0.95),
        "opacity_frac_gt_0_5": float((opacity > 0.5).float().mean()),
        "line_scale_med": q(scaling[is_line][:, 0], 0.5) if is_line.any() else float("nan"),
        "line_opacity_med": q(opacity[is_line], 0.5) if is_line.any() else float("nan"),
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tmp/psl-room1-control")
    ap.add_argument("--evidence", default="/workspace/code/PSL-perf/evidence/online_checkpoints_r0")
    ap.add_argument("--out", default="/workspace/code/PSL-perf/evidence/checkpoint_followup_r1")
    args = ap.parse_args()
    root, ev, out = Path(args.root), Path(args.evidence), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fixed = set(load_fixed(ev / "fixed_early_frames.txt"))
    checkpoints = ["frame_500", "frame_1000", "frame_1500", "frame_end"]

    # 1) Per-frame offline vs checkpoint-time cross-check.
    with open(out / "offline_crosscheck.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["side", "checkpoint", "n", "max_abs_diff", "mean_diff",
                    "offline_mean", "checkpoint_mean"])
        for side in "AC":
            for ck in checkpoints:
                off = {int(r["fid"]): float(r["psnr_gs"])
                       for r in csv.DictReader(open(root / "offline-score-r0" /
                                                    f"{side}_{ck}" / "offline_psnr.csv"))}
                base = load_psnr(ev / side / ck / "psnr_gaussian_splatting.txt")
                common = sorted(set(off) & set(base))
                diffs = [off[k] - base[k] for k in common]
                w.writerow([side, ck, len(common), max(abs(x) for x in diffs),
                            statistics.mean(diffs), statistics.mean(off[k] for k in common),
                            statistics.mean(base[k] for k in common)])

    # 2) Fixed-12 native GS PSNR from offline files.
    with open(out / "fixed12_offline.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint", "n", "A_mean", "C_mean", "A_minus_C_mean",
                    "A_minus_C_median"])
        for ck in checkpoints:
            vals, names = {}, {}
            for side in "AC":
                off = {int(r["fid"]): float(r["psnr_gs"])
                       for r in csv.DictReader(open(root / "offline-score-r0" /
                                                    f"{side}_{ck}" / "offline_psnr.csv"))}
                cam = load_cameras(root / f"online-diag-{side}" / "ckpt" / ck / "cameras.csv")
                vals[side] = {os.path.basename(cam[f]["img_name"]): v
                              for f, v in off.items() if f in cam and
                              os.path.basename(cam[f]["img_name"]) in fixed}
                names[side] = set(vals[side])
            common = sorted(names["A"] & names["C"])
            d = [vals["A"][x] - vals["C"][x] for x in common]
            w.writerow([ck, len(common),
                        statistics.mean(vals["A"][x] for x in common),
                        statistics.mean(vals["C"][x] for x in common),
                        statistics.mean(d), statistics.median(d)])

    # 3) Camera pose changes 1000 -> 1500 for fixed views, plus their quality.
    camera_rows = []
    summary_rows = []
    for side in "AC":
        c0 = load_cameras(root / f"online-diag-{side}" / "ckpt" / "frame_1000" / "cameras.csv")
        c1 = load_cameras(root / f"online-diag-{side}" / "ckpt" / "frame_1500" / "cameras.csv")
        p0 = {int(r["fid"]): float(r["psnr_gs"])
              for r in csv.DictReader(open(root / "offline-score-r0" /
                                           f"{side}_frame_1000" / "offline_psnr.csv"))}
        p1 = {int(r["fid"]): float(r["psnr_gs"])
              for r in csv.DictReader(open(root / "offline-score-r0" /
                                           f"{side}_frame_1500" / "offline_psnr.csv"))}
        for f in sorted(set(c0) & set(c1)):
            name = os.path.basename(c0[f]["img_name"])
            if name not in fixed or f not in p0 or f not in p1:
                continue
            rot, trans = relative_pose(c0[f], c1[f])
            camera_rows.append([side, name, c0[f]["fid"], c1[f]["fid"], rot, trans,
                                p0[f], p1[f], p1[f] - p0[f]])
        if camera_rows:
            rr = [r for r in camera_rows if r[0] == side]
            ds = np.array([r[8] for r in rr])
            ro = np.array([r[4] for r in rr])
            tr = np.array([r[5] for r in rr])
            summary_rows.append([side, len(rr), max(ro), statistics.mean(ro),
                                 max(tr), statistics.mean(tr), ds.mean(),
                                 np.median(ds), ds.min(), ds.max(),
                                 np.corrcoef(ro, ds)[0, 1],
                                 np.corrcoef(tr, ds)[0, 1]])
    with open(out / "camera_fixed12_1000_1500.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["side", "image", "fid_1000", "fid_1500", "rot_deg", "trans",
                    "psnr_1000", "psnr_1500", "d_psnr"])
        w.writerows(camera_rows)
    with open(out / "camera_fixed12_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["side", "n", "rot_max", "rot_mean", "trans_max", "trans_mean",
                    "dpsnr_mean", "dpsnr_median", "dpsnr_min", "dpsnr_max",
                    "corr_rot_dpsnr", "corr_trans_dpsnr"])
        w.writerows(summary_rows)

    # 4) Model scale/opacity statistics from the checkpoint tensors.
    model_rows = []
    for side in "AC":
        for ck in checkpoints:
            sd = torch.jit.load(str(root / f"online-diag-{side}" / "ckpt" / ck /
                                    "snapshot.pt"), map_location="cpu").state_dict()
            model_rows.append([side, ck, *tensor_stats(sd).values()])
    cols = ["side", "checkpoint", "n", "line", "ext_x", "ext_y", "ext_z",
            "ext_diag", "centroid_x", "centroid_y", "centroid_z", "dist_med",
            "dist_p95", "scale_med_x", "scale_med_y", "scale_med_z",
            "scale_p95_x", "scale_p95_y", "scale_p95_z", "scale_max",
            "opacity_mean", "opacity_med", "opacity_p05", "opacity_p95",
            "opacity_frac_gt_0_5", "line_scale_med", "line_opacity_med"]
    with open(out / "model_stats.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(model_rows)

    print(f"wrote {out}/offline_crosscheck.csv")
    print(f"wrote {out}/fixed12_offline.csv")
    print(f"wrote {out}/camera_fixed12_1000_1500.csv")
    print(f"wrote {out}/model_stats.csv")


if __name__ == "__main__":
    main()
