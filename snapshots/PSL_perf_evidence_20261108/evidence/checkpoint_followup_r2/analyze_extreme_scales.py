#!/usr/bin/env python3
"""Fixed-rule extreme-scale audit for the existing A/C 1000/1500 snapshots.

The anomaly set is fixed before any counterfactual render:
  top 20 rows by max(exp(scaling_)) within each class (line / ordinary),
  deterministic tie break by ascending row index.
No PSNR is read here.  The same rule is used for A and C and for 1000/1500.
"""
import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np
import torch

TOP_K = 20


def load_snapshot(path):
    return torch.jit.load(str(path), map_location="cpu").state_dict()


def load_cams(path):
    rows = {}
    for r in csv.DictReader(open(path)):
        rows[int(r["fid"])] = r
    return rows


def q(v, p):
    return float(torch.quantile(v, p)) if len(v) else float("nan")


def project_visibility(xyz, ref_kf, cams, masks, mask_ids):
    """Count KFs where the Gaussian center projects inside the image and mask."""
    R = None
    visible = 0
    ref_visible = 0
    ref_uv = None
    per_kf = []
    for fid, r in cams.items():
        qw, qx, qy, qz = [float(r[k]) for k in ("qw", "qx", "qy", "qz")]
        tx, ty, tz = [float(r[k]) for k in ("tx", "ty", "tz")]
        # Tcw: Xc = R Xw + t
        xx, yy, zz = 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy), 1 - 2 * (qx * qx + qy * qy)
        yx, yz = 2.0 * (qy * qz + qw * qx), 2.0 * (qy * qz - qw * qx)
        zx, zz2 = 2.0 * (qx * qz - qw * qy), 1 - 2 * (qx * qx + qz * qz)
        # Full quaternion rotation matrix, Tcw.
        Rm = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ])
        Xc = Rm @ xyz + np.array([tx, ty, tz])
        if Xc[2] <= 0.01:
            continue
        fx, fy = float(r["fx"]), float(r["fy"])
        cx, cy = float(r["cx"]), float(r["cy"])
        u = fx * Xc[0] / Xc[2] + cx
        v = fy * Xc[1] / Xc[2] + cy
        W, H = int(r["width"]), int(r["height"])
        inside = 0 <= u < W and 0 <= v < H
        mask_ok = False
        if inside and int(fid) in mask_ids:
            mi = mask_ids[int(fid)]
            if mi < len(masks):
                m = masks[mi][0]  # [H,W]
                mask_ok = float(m[int(v), int(u)]) > 0.5
        ok = inside and mask_ok
        if ok:
            visible += 1
            per_kf.append(fid)
        if fid == ref_kf:
            ref_visible = int(ok)
            ref_uv = [float(u), float(v), float(Xc[2])]
    return visible, ref_visible, ref_uv, per_kf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tmp/psl-room1-control")
    ap.add_argument("--out", default="/workspace/code/PSL-perf/evidence/checkpoint_followup_r2")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    (out / "anomaly_rule.txt").write_text(
        "Fixed before the counterfactual render:\n"
        f"primary anomaly set = top {TOP_K} rows by max(exp(scaling_)) within each\n"
        "class (is_line true/false) and snapshot; ties by ascending row index.\n"
        "No PSNR or render outcome is used to select rows.  The same rule is used\n"
        "for A/C and frame_1000/frame_1500.\n"
    )

    stats_rows, anomaly_rows, line_term_rows = [], [], []
    for side in "AC":
        for ck in ("frame_1000", "frame_1500"):
            ckdir = root / f"online-diag-{side}" / "ckpt" / ck
            sd = load_snapshot(ckdir / "snapshot.pt")
            s = sd["3"].float().exp()
            maxs = s.max(1).values
            mins = s.min(1).values
            ratio = maxs / (mins + 1e-12)
            opacity = torch.sigmoid(sd["4"].float()).squeeze(-1)
            is_line = sd["6"].bool()
            cams = load_cams(ckdir / "cameras.csv")
            mask_ids = {}
            with open(ckdir / "mask_cameras.txt") as f:
                for i, line in enumerate(f):
                    mask_ids[int(line.strip())] = i
            masks = torch.jit.load(str(ckdir / "masks.pt"), map_location="cpu").state_dict()
            mask_list = [masks[k] for k in sorted(masks.keys(), key=int)]
            xyz = sd["0"].float().numpy()

            for cls, cmask in (("line", is_line), ("ordinary", ~is_line)):
                idx_all = torch.nonzero(cmask).flatten()
                if len(idx_all) == 0:
                    continue
                vals = maxs[idx_all].numpy()
                # deterministic: descending max scale, ascending row index
                order = np.lexsort((idx_all.numpy(), -vals))
                top = idx_all.numpy()[order[:TOP_K]]
                stats_rows.append([
                    side, ck, cls, len(idx_all),
                    q(maxs[cmask], 0.50), q(maxs[cmask], 0.95),
                    q(maxs[cmask], 0.99), float(maxs[cmask].max()),
                    q(ratio[cmask], 0.50), q(ratio[cmask], 0.95),
                    q(ratio[cmask], 0.99), float(ratio[cmask].max()),
                    q(opacity[cmask], 0.50), q(opacity[cmask], 0.95),
                ])
                for rank, i in enumerate(top):
                    i = int(i)
                    ref = int(sd["14"][i])
                    vis, ref_vis, ref_uv, _ = project_visibility(
                        xyz[i], ref, cams, mask_list, mask_ids)
                    anomaly_rows.append([
                        side, ck, cls, rank, i, int(sd["13"][i]), ref,
                        *[float(x) for x in s[i]],
                        float(maxs[i]), float(ratio[i]), float(opacity[i]),
                        int(sd["12"][i]), vis, ref_vis,
                        "" if ref_uv is None else ref_uv[0],
                        "" if ref_uv is None else ref_uv[1],
                        "" if ref_uv is None else ref_uv[2],
                    ])
            # Line eccentricity term: s_perp / s_parallel, the exact code form.
            if is_line.any():
                ss = s[is_line]
                sp = ss[:, 0]
                sperp = ss[:, 1:3].mean(1)
                ecc = sperp / (sp + 1e-6)
                line_term_rows.append([
                    side, ck, int(is_line.sum()),
                    q(sp, 0.01), q(sp, 0.50), q(sp, 0.95),
                    q(sperp, 0.01), q(sperp, 0.50), q(sperp, 0.95),
                    q(ecc, 0.50), q(ecc, 0.95), q(ecc, 0.99), float(ecc.max()),
                    int((sp < 1e-5).sum()), int((sperp > sp).sum()),
                ])

    with open(out / "scale_stats.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["side", "checkpoint", "class", "n", "max_axis_p50",
                    "max_axis_p95", "max_axis_p99", "max_axis_max",
                    "ratio_p50", "ratio_p95", "ratio_p99", "ratio_max",
                    "opacity_p50", "opacity_p95"])
        w.writerows(stats_rows)
    with open(out / "anomaly_rows.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["side", "checkpoint", "class", "rank", "index", "point_id",
                    "ref_kf", "s0", "s1", "s2", "max_axis", "ratio", "opacity",
                    "exist_since_iter", "visible_kfs", "ref_visible",
                    "ref_u", "ref_v", "ref_depth"])
        w.writerows(anomaly_rows)
    with open(out / "line_eccentricity.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["side", "checkpoint", "n_line", "s_parallel_p01",
                    "s_parallel_p50", "s_parallel_p95", "s_perp_p01",
                    "s_perp_p50", "s_perp_p95", "ecc_p50", "ecc_p95",
                    "ecc_p99", "ecc_max", "n_s_parallel_lt_1e-5",
                    "n_s_perp_gt_parallel"])
        w.writerows(line_term_rows)

    print(f"wrote {out}/scale_stats.csv")
    print(f"wrote {out}/anomaly_rows.csv")
    print(f"wrote {out}/line_eccentricity.csv")


if __name__ == "__main__":
    main()
