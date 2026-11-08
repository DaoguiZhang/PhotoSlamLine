#!/usr/bin/env python3
"""Re-score existing common-view PNG/JPG pairs with corrected metrics.

Bugs in the original scorer:
  1) PSNR used uint8 data with peak=1.0  -> constant -48.13 dB offset.
  2) "SSIM" was a single global mean/var over the whole image, not windowed.
Here: inputs are converted once to float32 in [0,1]; PSNR uses peak=1;
SSIM is the standard 11x11 uniform-window SSIM; LPIPS unchanged (already [0,1]).
usage: rescore_common_views.py <out_root> <gt_traj> <dataset> [--lpips]
"""
import csv, glob, json, os, statistics as st, sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def to_f01(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    return a


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def ssim(a, b, win=11):
    ta = torch.from_numpy(a).permute(2, 0, 1)[None]
    tb = torch.from_numpy(b).permute(2, 0, 1)[None]
    pad = win // 2
    mu1 = F.avg_pool2d(ta, win, 1, pad); mu2 = F.avg_pool2d(tb, win, 1, pad)
    s11 = F.avg_pool2d(ta * ta, win, 1, pad) - mu1 * mu1
    s22 = F.avg_pool2d(tb * tb, win, 1, pad) - mu2 * mu2
    s12 = F.avg_pool2d(ta * tb, win, 1, pad) - mu1 * mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s11 + s22 + C2))
    return float(m.mean())


def sanity():
    z = np.zeros((64, 64, 3), np.float32); o = np.ones_like(z)
    h = np.full_like(z, 0.5)
    print("[sanity] same-image PSNR=%s SSIM=%.6f" % (psnr(z, z), ssim(z, z)))
    print("[sanity] black/white PSNR=%.4f (expect 0)" % psnr(z, o))
    print("[sanity] 0 vs 0.5 PSNR=%.4f (expect 6.0206)" % psnr(z, h))
    print("[sanity] scale-invariance PSNR(255*)=%.4f" % psnr(z * 255, o * 255))


def main(argv):
    root, gt, ds = argv[:3]
    do_lpips = "--lpips" in argv
    sanity()
    net = None
    if do_lpips:
        import lpips
        net = lpips.LPIPS(net="alex", verbose=False).eval()
    rows = []
    for run in sorted(os.listdir(os.path.join(root, "runs"))):
        img_dir = os.path.join(root, "runs", run, "eval_common_v2", "image")
        for p1 in sorted(glob.glob(img_dir + "/*.png")):
            fid = int(os.path.basename(p1)[:-4])
            ref = os.path.join(ds, "results", f"frame{fid:06d}.jpg")
            if not os.path.exists(ref):
                raise SystemExit(f"[FATAL] missing reference {ref}")
            A, B = to_f01(p1), to_f01(ref)
            if A.shape != B.shape:
                raise SystemExit(f"[FATAL] shape {A.shape} vs {B.shape} ({p1})")
            s = ssim(A, B); p = psnr(A, B)
            lp = None
            if net is not None:
                ta = torch.from_numpy(A).permute(2, 0, 1)[None] * 2 - 1
                tb = torch.from_numpy(B).permute(2, 0, 1)[None] * 2 - 1
                lp = float(net(ta, tb).item())
            if not (np.isfinite(p) or p == float("inf")):
                raise SystemExit(f"[FATAL] non-finite PSNR {p} ({p1})")
            rows.append(dict(run=run, cond=run.rsplit("_r", 1)[0], round=int(run.rsplit("_r", 1)[1]),
                             fid=fid, psnr=p, ssim=s, lpips=lp,
                             render=p1, ref=ref))
    out = os.path.join(root, "common_view_per_frame_v2_fixed.csv")
    old = os.path.join(root, "common_view_per_frame.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("[done]", len(rows), "rows ->", out)
    print("[note] superseded scorer output kept as", os.path.basename(old),
          "(peak=1 on uint8 -> -48.13 dB offset; global SSIM)")
    # per-run and comparisons
    runs = sorted({r["run"] for r in rows}); frames = sorted({r["fid"] for r in rows})
    per = {}
    for r in rows:
        per.setdefault(r["run"], []).append(r)
    for run in runs:
        v = per[run]
        assert len(v) == 24 and len({x["fid"] for x in v}) == 24, run
        print("%-12s PSNR=%.3f SSIM=%.4f%s" % (run, st.mean(x["psnr"] for x in v),
              st.mean(x["ssim"] for x in v),
              "" if not do_lpips else " LPIPS=%.4f" % st.mean(x["lpips"] for x in v)))
    cond = {}
    for r in rows:
        cond.setdefault(r["cond"], {}).setdefault(r["round"], {})[r["fid"]] = r
    for a, b in [("P_fast", "P_ref"), ("L_fast", "L_ref"), ("L_fast", "P_fast"), ("L_fast", "P_ref")]:
        line = []
        for key in ("psnr", "ssim") + (("lpips",) if do_lpips else ()):
            d = [st.mean(cond[a][k][f][key] - cond[b][k][f][key] for f in frames) for k in (1, 2, 3)]
            line.append("%s %s mean=%+.3f sd=%.3f" % (key, ["%+.3f" % x for x in d], st.mean(d), st.stdev(d)))
        print("%s-%s: %s" % (a, b, " | ".join(line)))


if __name__ == "__main__":
    main(sys.argv[1:])
