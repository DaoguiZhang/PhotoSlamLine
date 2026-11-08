#!/usr/bin/env python3
"""Common-view evaluation for the append / cumulative-training-budget round.

Layout expected:
  <run_root>/runs/<run>/CameraTrajectory_TUM.txt
  <run_root>/runs/<run>/append/natural_end/point_cloud/*/point_cloud.ply
  <run_root>/runs/<run>/append/budget_end/point_cloud/*/point_cloud.ply

For EACH run and EACH state:
  - estimate a fresh SE(3) Umeyama alignment from that run's own trajectory to
    the GT trajectory (never reused across runs),
  - transform the same 24 GT cameras into that map frame with the confirmed
    camera basis conversion C = diag(1,-1,-1),
  - render with bin/eval_common_views (no training, no optimizer),
  - compute PSNR (float [0,1], peak=1), standard 11x11 uniform-window SSIM and
    LPIPS (AlexNet, RGB, inputs in [-1,1]) sequentially.

Metrics are computed offline AFTER training. No concurrency.
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES = list(range(0, 2000, 83))[:24]
BASIS_C = np.diag([1.0, -1.0, -1.0])


def read_tum(path):
    ts, P, Q = [], [], []
    for line in open(path):
        s = line.split()
        if len(s) < 8:
            continue
        ts.append(float(s[0]))
        P.append([float(x) for x in s[1:4]])
        Q.append([float(x) for x in s[4:8]])
    return np.array(ts), np.array(P), np.array(Q)


def q2R(q):
    x, y, z, w = q
    n = np.linalg.norm(q)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R2q(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s; x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    return np.array([x, y, z, w])


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, D, Vt = np.linalg.svd(S)
    E = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    t = mu_d - R @ mu_s
    res = np.linalg.norm((R @ src.T).T + t - dst, axis=1)
    return R, t, float(np.sqrt((res ** 2).mean()))


def to_f01(path):
    return np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse == 0 else float(10.0 * np.log10(1.0 / mse))


def ssim_standard(a, b, win=11):
    ta = torch.from_numpy(a).permute(2, 0, 1)[None]
    tb = torch.from_numpy(b).permute(2, 0, 1)[None]
    pad = win // 2
    mu1 = F.avg_pool2d(ta, win, 1, pad)
    mu2 = F.avg_pool2d(tb, win, 1, pad)
    s11 = F.avg_pool2d(ta * ta, win, 1, pad) - mu1 * mu1
    s22 = F.avg_pool2d(tb * tb, win, 1, pad) - mu2 * mu2
    s12 = F.avg_pool2d(ta * tb, win, 1, pad) - mu1 * mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / (
        (mu1 ** 2 + mu2 ** 2 + C1) * (s11 + s22 + C2))
    return float(m.mean())


def find_ply(run_dir, state):
    pat = os.path.join(run_dir, "append", state, "point_cloud", "*", "point_cloud.ply")
    c = glob.glob(pat)
    if not c:
        raise SystemExit(f"[FATAL] missing {state} PLY: {pat}")
    return max(c, key=lambda p: int(os.path.basename(os.path.dirname(p)).split("_")[-1]))


def write_cameras(run_dir, R, t, gtP, gtQ):
    camfile = os.path.join(run_dir, "eval_cameras_budget.txt")
    with open(camfile, "w") as f:
        f.write("# fid qw qx qy qz tx ty tz  (Tcw in this map's frame)\n")
        for fid in FRAMES:
            Rwc = q2R(gtQ[fid])
            C = gtP[fid]
            Rwc_e = R.T @ Rwc @ BASIS_C
            C_e = R.T @ (C - t)
            Rcw = Rwc_e.T
            tcw = -Rcw @ C_e
            q = R2q(Rcw)
            f.write(f"{fid} {q[3]:.9f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} "
                    f"{tcw[0]:.9f} {tcw[1]:.9f} {tcw[2]:.9f}\n")
    return camfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root")
    ap.add_argument("gt")
    ap.add_argument("orb_cfg")
    ap.add_argument("gauss_cfg")
    ap.add_argument("dataset")
    ap.add_argument("--lpips", action="store_true")
    ap.add_argument("--runs", default="P_ref,P_fast,L_fast")
    args = ap.parse_args()

    # Metric sanity checks (same implementation used for every run).
    z = np.zeros((64, 64, 3), np.float32)
    o = np.ones_like(z)
    assert np.isinf(psnr(z, z))
    assert abs(ssim_standard(z, z) - 1.0) < 1e-6
    assert abs(psnr(z, o) - 0.0) < 1e-6
    print("[sanity] PSNR same=inf black/white=0 SSIM same=1")

    gts, gtP, gtQ = read_tum(args.gt)
    lp = None
    if args.lpips:
        import lpips
        lp = lpips.LPIPS(net="alex", verbose=False).eval()
        # LPIPS sanity: identical image -> ~0 (inputs are RGB in [-1,1]).
        ta = torch.from_numpy(z).permute(2, 0, 1)[None] * 2 - 1
        print("[sanity] LPIPS identical=%.8f" % float(lp(ta, ta).item()))

    rows, summaries, align_rows = [], [], []
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    for run in runs:
        rd = os.path.join(args.run_root, "runs", run)
        est = os.path.join(rd, "CameraTrajectory_TUM.txt")
        if not os.path.exists(est):
            raise SystemExit(f"[FATAL] missing trajectory {est}")
        es, eP, eQ = read_tum(est)
        idx = np.clip(np.searchsorted(gts, es), 1, len(gts) - 1)
        idx = np.where(np.abs(gts[idx] - es) < np.abs(gts[idx - 1] - es),
                       idx, idx - 1)
        R, t, rmse = umeyama(eP, gtP[idx])
        align_rows.append(dict(run=run, align_rmse=rmse, n_assoc=len(es)))
        camfile = write_cameras(rd, R, t, gtP, gtQ)
        print("[align] %-8s rmse=%.6f n=%d" % (run, rmse, len(es)))

        for state in ("natural_end", "budget_end"):
            ply = find_ply(rd, state)
            out = os.path.join(rd, f"eval_{state}")
            subprocess.run([os.path.join(ROOT, "bin", "eval_common_views"),
                            args.gauss_cfg, ply, args.orb_cfg, camfile, out],
                           check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.STDOUT)
            vals = []
            for fid in FRAMES:
                p1 = os.path.join(out, "image", f"{fid:06d}.png")
                ref = os.path.join(args.dataset, "results", f"frame{fid:06d}.jpg")
                if not (os.path.exists(p1) and os.path.exists(ref)):
                    raise SystemExit(f"[FATAL] missing pair {p1} / {ref}")
                A, B = to_f01(p1), to_f01(ref)
                if A.shape != B.shape:
                    raise SystemExit(f"[FATAL] shape {A.shape} vs {B.shape}")
                p = psnr(A, B)
                s = ssim_standard(A, B)
                lpv = None
                if lp is not None:
                    ta = torch.from_numpy(A).permute(2, 0, 1)[None] * 2 - 1
                    tb = torch.from_numpy(B).permute(2, 0, 1)[None] * 2 - 1
                    lpv = float(lp(ta, tb).item())
                vals.append((p, s, lpv))
                rows.append(dict(run=run, state=state, fid=fid, psnr=p,
                                 ssim=s, lpips=lpv, render=p1, ref=ref))
            ps = [v[0] for v in vals]
            ss = [v[1] for v in vals]
            ls = [v[2] for v in vals if v[2] is not None]
            summaries.append(dict(
                run=run, state=state, n=len(vals),
                psnr=float(np.mean(ps)), ssim=float(np.mean(ss)),
                lpips=(float(np.mean(ls)) if ls else None),
                align_rmse=rmse))
            print("[eval] %-8s %-11s PSNR=%.4f SSIM=%.5f LPIPS=%s" %
                  (run, state, np.mean(ps), np.mean(ss),
                   ("%.5f" % np.mean(ls)) if ls else "na"))

    with open(os.path.join(args.run_root, "common_view_per_frame_budget.csv"),
              "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(args.run_root, "common_view_summary_budget.csv"),
              "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader(); w.writerows(summaries)
    with open(os.path.join(args.run_root, "common_view_alignment_budget.csv"),
              "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(align_rows[0].keys()))
        w.writeheader(); w.writerows(align_rows)
    json.dump(dict(frames=FRAMES), open(os.path.join(
        args.run_root, "common_view_frames_budget.json"), "w"), indent=1)

    # Paired same-frame differences (positive = first run better).
    by = {}
    for r in rows:
        by.setdefault((r["run"], r["state"]), {})[r["fid"]] = r
    pairs = [("P_fast", "P_ref"), ("L_fast", "P_ref"), ("L_fast", "P_fast")]
    print("\n[paired deltas] positive means first run better")
    for a, b in pairs:
        for state in ("natural_end", "budget_end"):
            if (a, state) not in by or (b, state) not in by:
                continue
            d = [(by[(a, state)][f]["psnr"] - by[(b, state)][f]["psnr"],
                  by[(a, state)][f]["ssim"] - by[(b, state)][f]["ssim"])
                 for f in FRAMES]
            dp = float(np.mean([x[0] for x in d]))
            ds = float(np.mean([x[1] for x in d]))
            print("  %-7s - %-7s %-11s dPSNR=%+.4f dSSIM=%+.5f" %
                  (a, b, state, dp, ds))
    print("\n[done] wrote", args.run_root)


if __name__ == "__main__":
    main()
