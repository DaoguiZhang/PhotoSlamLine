#!/usr/bin/env python3
"""Common-view evaluation of the 12 final maps (office0).

- 24 deterministic frame ids covering the sequence (every 83rd of 2000).
- GT poses (pose_TUM.txt, T_wc) are mapped into each run's map frame with a
  SE(3) Umeyama alignment (no scale) estimated from that run's own
  CameraTrajectory_TUM.txt (T_wc) against GT.
- The evaluation camera (Tcw) is transformed, never the Gaussians.
- Renders with bin/eval_common_views, then computes PSNR/SSIM (one numpy
  implementation for all runs) and LPIPS on CPU.

usage: eval_common_views.py <out_root> <gt> <orb_cfg> <gauss_cfg> <dataset>
"""
import csv, hashlib, json, os, subprocess, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES = list(range(0, 2000, 83))[:24]


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
        w = 0.25 * s; x = (R[2, 1] - R[1, 2]) / s; y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s; y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s; y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s; y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
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


def psnr(a, b):
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
    return 99.0 if mse == 0 else 10 * np.log10(1.0 / mse)


def ssim_global(a, b):
    a = a.astype(np.float64) / 255.0; b = b.astype(np.float64) / 255.0
    mu1, mu2 = a.mean(), b.mean()
    v1, v2 = a.var(), b.var()
    c = (a - mu1) * (b - mu2)
    cov = c.mean()
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    return float(((2 * mu1 * mu2 + C1) * (2 * cov + C2)) /
                 ((mu1 ** 2 + mu2 ** 2 + C1) * (v1 + v2 + C2)))


def main(argv):
    out_root, gt_path, orb_cfg, gcfg, ds = argv[:5]
    from PIL import Image
    gts, gtP, gtQ = read_tum(gt_path)
    ids = np.array(FRAMES)
    run_dirs = sorted(os.listdir(os.path.join(out_root, "runs")))
    lp = None
    try:
        import torch, lpips
        lp = lpips.LPIPS(net="alex", verbose=False).eval()
        def lpips_pair(p1, p2):
            A = np.asarray(Image.open(p1).convert("RGB"), np.float32) / 255.0
            B = np.asarray(Image.open(p2).convert("RGB"), np.float32) / 255.0
            ta = torch.from_numpy(A).permute(2, 0, 1)[None] * 2 - 1
            tb = torch.from_numpy(B).permute(2, 0, 1)[None] * 2 - 1
            return float(lp(ta, tb).item())
    except Exception as e:
        print("LPIPS unavailable:", e)
        lpips_pair = None

    rows, ev = [], []
    for run in run_dirs:
        rd = os.path.join(out_root, "runs", run)
        est = os.path.join(rd, "CameraTrajectory_TUM.txt")
        sd = [n for n in os.listdir(rd) if n.endswith('_shutdown') and n.split('_')[0].isdigit()]
        import glob as _glob
        _snap = max(sd, key=lambda n: int(n.split('_')[0])) if sd else None
        _c = _glob.glob(os.path.join(rd, _snap, "ply", "point_cloud", "*", "point_cloud.ply")) if _snap else []
        if not _c: raise SystemExit(f"[FATAL] no full Gaussian PLY for {run}")
        ply = _c[0]
        if not (os.path.exists(est) and os.path.exists(ply)):
            raise SystemExit(f"[FATAL] {run}: traj={os.path.exists(est)} ply={ply} {os.path.exists(ply)}")
        es, eP, eQ = read_tum(est)
        # associate est<->gt by nearest timestamp
        idx = np.clip(np.searchsorted(gts, es), 1, len(gts) - 1)
        idx = np.where(np.abs(gts[idx] - es) < np.abs(gts[idx - 1] - es), idx, idx - 1)
        R, t, rmse = umeyama(eP, gtP[idx])
        # GT eval cameras -> map frame
        camfile = os.path.join(rd, "eval_cameras_v2.txt")
        with open(camfile, "w") as f:
            f.write("# fid qw qx qy qz tx ty tz  (Tcw in this map's frame)\n")
            for fid in FRAMES:
                Rwc = q2R(gtQ[fid]); C = gtP[fid]
                Rwc_e = R.T @ Rwc @ np.diag([1.0, -1.0, -1.0])
                C_e = R.T @ (C - t)
                Rcw = Rwc_e.T
                tcw = -Rcw @ C_e
                q = R2q(Rcw)
                f.write(f"{fid} {q[3]:.9f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} "
                        f"{tcw[0]:.9f} {tcw[1]:.9f} {tcw[2]:.9f}\n")
        out = os.path.join(rd, "eval_common_v2")
        subprocess.run([os.path.join(ROOT, "bin", "eval_common_views"), gcfg, ply,
                        orb_cfg, camfile, out], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        ev.append(dict(run=run, align_rmse=rmse, n_assoc=len(es)))
        for fid in FRAMES:
            p1 = os.path.join(out, "image", f"{fid:06d}.png")
            ref = os.path.join(ds, "results", f"frame{fid:06d}.jpg")
            if not (os.path.exists(p1) and os.path.exists(ref)):
                continue
            A = np.asarray(Image.open(p1).convert("RGB"), np.uint8)
            B = np.asarray(Image.open(ref).convert("RGB"), np.uint8)
            if A.shape != B.shape:
                print("shape mismatch", run, fid, A.shape, B.shape); continue
            rows.append(dict(run=run, cond=run.rsplit("_r", 1)[0], round=int(run.rsplit("_r", 1)[1]),
                             fid=fid, psnr=psnr(A, B), ssim=ssim_global(A, B),
                             lpips=lpips_pair(p1, ref) if lpips_pair else None))
    with open(os.path.join(out_root, "common_view_per_frame_v2.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(out_root, "common_view_alignment_v2.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ev[0].keys())); w.writeheader(); w.writerows(ev)
    ids_hash = hashlib.sha256(",".join(map(str, FRAMES)).encode()).hexdigest()
    json.dump(dict(frames=FRAMES, sha256=ids_hash), open(os.path.join(out_root, "common_view_frames.json"), "w"), indent=1)
    print("frames:", len(FRAMES), "sha256:", ids_hash)
    print("wrote", os.path.join(out_root, "common_view_per_frame_v2.csv"))


if __name__ == "__main__":
    main(sys.argv[1:])
