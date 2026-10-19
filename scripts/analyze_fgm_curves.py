#!/usr/bin/env python3
"""Full-curve analysis for the fixed-geometry mapper convergence experiment.

protocol_version: fgm-conv-v2

Reads, for one or more labels:
  abl_out/<label>/frozen/<tag>_metrics.csv        per eval point summary
  abl_out/<label>/<tag>/eval_step<N>/per_frame.csv per-view PSNR/SSIM/L1
  abl_out/<label>/<tag>/eval_step<N>/image/*.png   rendered evaluation views
  abl_out/<label>/<tag>/eval_step<N>/image_gt/*.png ground truth for those views

and reports
  * PSNR / SSIM / LPIPS against completed optimizer steps
  * PSNR / SSIM / LPIPS against cumulative mapping seconds
  * per-eval-point paired ON-OFF differences
  * the same-scene OFF/OFF control delta as the pipeline tolerance
  * steps and time to reach a common quality threshold (same threshold for all)
  * structural-residual ratios recomputed from the raw CSV, with the exact rows

LPIPS is computed offline on the saved renders; it is NOT part of the measured
mapping time. If the renders are missing, LPIPS is reported as MISSING.

usage: analyze_fgm_curves.py <label> [<label> ...]
       analyze_fgm_curves.py --off <label> --on <label> [--ctrl <label>]
"""
import csv
import os
import statistics
import sys

ROOT = "abl_out"
OFF_TAG = "B"
ON_TAG = "C"
OFF_NAME = "LineSeed + Photo"
ON_NAME = "LineSeed + Photo + Struct"

_MEAN_COLS = ["cum_mapping_seconds", "eval_seconds", "eval_psnr", "eval_ssim",
              "eval_l1", "struct_ori", "struct_shape"]


# ---------------------------------------------------------------- IO helpers
def read_metrics(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        d = {}
        for k, v in r.items():
            if k == "condition":
                d[k] = v
            else:
                try:
                    d[k] = float(v)
                except ValueError:
                    d[k] = float("nan")
        out.append(d)
    return out


def read_per_frame(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------- LPIPS
def lpips_available():
    try:
        import lpips  # noqa
        import torch  # noqa
        return True
    except Exception:
        return False


class LpipsScorer:
    """Offline LPIPS over saved renders. Never touches the mapping timer."""

    def __init__(self):
        import lpips
        import torch
        self.torch = torch
        self.device = "cpu"
        try:
            if torch.cuda.is_available():
                self.device = "cuda"
        except Exception:
            self.device = "cpu"
        self.net = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval()
        self._dev_failed = False

    def __call__(self, p1, p2):
        import numpy as np
        from PIL import Image
        torch = self.torch
        a = np.asarray(Image.open(p1).convert("RGB"), dtype="float32") / 255.0
        b = np.asarray(Image.open(p2).convert("RGB"), dtype="float32") / 255.0
        ta = torch.from_numpy(a).permute(2, 0, 1)[None] * 2 - 1
        tb = torch.from_numpy(b).permute(2, 0, 1)[None] * 2 - 1
        try:
            with torch.no_grad():
                v = self.net(ta.to(self.device), tb.to(self.device))
        except Exception:
            if self.device == "cpu":
                raise
            # Fall back once if the CUDA LPIPS build is broken.
            self.device = "cpu"
            self.net = self.net.to("cpu")
            with torch.no_grad():
                v = self.net(ta, tb)
        return float(v.item())


def lpips_at(root_tag_dir, step, scorer, cache):
    img_dir = os.path.join(root_tag_dir, f"eval_step{int(step)}", "image")
    gt_dir = os.path.join(root_tag_dir, f"eval_step{int(step)}", "image_gt")
    if not (os.path.isdir(img_dir) and os.path.isdir(gt_dir)):
        return None
    names = sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))
    if not names:
        return None
    vals = []
    for nm in names:
        g = os.path.join(gt_dir, nm)
        if not os.path.exists(g):
            continue
        key = (img_dir, nm)
        if key in cache:
            vals.append(cache[key])
            continue
        v = scorer(os.path.join(img_dir, nm), g)
        cache[key] = v
        vals.append(v)
    return statistics.fmean(vals) if vals else None


# ---------------------------------------------------------------- comparison
def interp(rows, key, x):
    """Linear interpolation of `key` at cumulative-mapping-time x."""
    if not rows:
        return None
    if x <= rows[0]["cum_mapping_seconds"]:
        return rows[0]
    if x >= rows[-1]["cum_mapping_seconds"]:
        return rows[-1]
    for a, b in zip(rows, rows[1:]):
        ta, tb = a["cum_mapping_seconds"], b["cum_mapping_seconds"]
        if ta <= x <= tb:
            if tb - ta < 1e-12:
                return b
            w = (x - ta) / (tb - ta)
            d = dict(a)
            for k in _MEAN_COLS:
                d[k] = a[k] + w * (b[k] - a[k])
            d["completed_optimizer_steps"] = a["completed_optimizer_steps"] + w * (
                b["completed_optimizer_steps"] - a["completed_optimizer_steps"])
            return d
    return rows[-1]


def steps_to_threshold(rows, metric, thr, higher_better=True):
    """First completed_optimizer_steps at which `metric` crosses `thr`, linearly
    interpolated between eval points. Same threshold for every group."""
    for a, b in zip(rows, rows[1:]):
        va, vb = a[metric], b[metric]
        ok_a = va >= thr if higher_better else va <= thr
        ok_b = vb >= thr if higher_better else vb <= thr
        if ok_b and not ok_a:
            if abs(vb - va) < 1e-12:
                return b["completed_optimizer_steps"]
            w = (thr - va) / (vb - va)
            return a["completed_optimizer_steps"] + w * (
                b["completed_optimizer_steps"] - a["completed_optimizer_steps"])
    first = rows[0]
    ok0 = first[metric] >= thr if higher_better else first[metric] <= thr
    return first["completed_optimizer_steps"] if ok0 else None


def load_label(label):
    fdir = os.path.join(ROOT, label, "frozen")
    out = {}
    for tag in (OFF_TAG, ON_TAG):
        p = os.path.join(fdir, f"{tag}_metrics.csv")
        if os.path.exists(p):
            out[tag] = read_metrics(p)
    return out


def main(argv):
    if "--off" in argv:
        off_lbl = argv[argv.index("--off") + 1]
        on_lbl = argv[argv.index("--on") + 1]
        ctrl_lbl = argv[argv.index("--ctrl") + 1] if "--ctrl" in argv else None
        labels = [(off_lbl, on_lbl, ctrl_lbl)]
    else:
        labels = [(a, a, None) for a in argv if not a.startswith("--")]

    have_lpips = lpips_available()
    print(f"LPIPS available: {have_lpips}"
          f"{'' if have_lpips else '  -> LPIPS will be reported as MISSING'}")
    scorer = None
    cache = {}
    if have_lpips:
        try:
            scorer = LpipsScorer()
            print(f"LPIPS device: {scorer.device}")
        except Exception as e:  # pragma: no cover
            print(f"LPIPS init failed ({e}) -> MISSING")
            have_lpips = False

    summary_rows = []
    for off_lbl, on_lbl, ctrl_lbl in labels:
        data = load_label(off_lbl)
        if OFF_TAG not in data or ON_TAG not in data:
            print(f"[skip] {off_lbl}: needs both {OFF_TAG}_metrics.csv and {ON_TAG}_metrics.csv")
            continue
        r_off, r_on = data[OFF_TAG], data[ON_TAG]

        def base_of(lbl, tag):
            return os.path.join(ROOT, lbl, tag)

        print("\n" + "=" * 92)
        print(f"A/B  {off_lbl}   {OFF_NAME} (B)  vs  {ON_NAME} (C)")
        print("=" * 92)

        # ---- evaluate LPIPS on both sides ---------------------------------
        lp_off, lp_on = {}, {}
        if have_lpips and scorer is not None:
            for row in r_off:
                s = row["completed_optimizer_steps"]
                lp_off[s] = lpips_at(base_of(off_lbl, OFF_TAG), s, scorer, cache)
            for row in r_on:
                s = row["completed_optimizer_steps"]
                lp_on[s] = lpips_at(base_of(on_lbl, ON_TAG), s, scorer, cache)

        hdr = (f"{'steps':>6} {'map_s':>8} | {'PSNR off':>9} {'PSNR on':>9} {'d':>8} |"
               f" {'SSIM off':>9} {'SSIM on':>9} {'d':>9} |"
               f" {'LPIPS off':>9} {'LPIPS on':>9} {'d':>9}")
        print("\n-- (1) vs completed optimizer steps --")
        print(hdr)
        d_psnr, d_ssim, d_lpips, steps_list = [], [], [], []
        by_off = {r["completed_optimizer_steps"]: r for r in r_off}
        by_on = {r["completed_optimizer_steps"]: r for r in r_on}
        for s in sorted(by_off):
            a, b = by_off[s], by_on.get(s)
            if not b:
                continue
            dp = b["eval_psnr"] - a["eval_psnr"]
            ds = b["eval_ssim"] - a["eval_ssim"]
            lo, ln = lp_off.get(s), lp_on.get(s)
            dl = (ln - lo) if (lo is not None and ln is not None) else float("nan")
            d_psnr.append(dp); d_ssim.append(ds); steps_list.append(s)
            if dl == dl:
                d_lpips.append(dl)
            f = lambda v: ("%9.5f" % v) if v == v else "   MISSING"
            print(f"{s:>6.0f} {a['cum_mapping_seconds']:>8.3f} | "
                  f"{a['eval_psnr']:>9.4f} {b['eval_psnr']:>9.4f} {dp:>+8.4f} | "
                  f"{a['eval_ssim']:>9.5f} {b['eval_ssim']:>9.5f} {ds:>+9.5f} | "
                  f"{f(lo)} {f(ln)} "
                  f"{('%9.5f' % dl) if dl == dl else '  MISSING'}")

        print("\n-- (2) vs cumulative mapping seconds (common overlap) --")
        t_end = min(r_off[-1]["cum_mapping_seconds"], r_on[-1]["cum_mapping_seconds"])
        grid = {t_end}
        for rows in (r_off, r_on):
            for r in rows:
                if 0 < r["cum_mapping_seconds"] <= t_end:
                    grid.add(r["cum_mapping_seconds"])
        print(f"{'t_s':>8} {'off_steps':>10} {'on_steps':>9} | {'dPSNR':>8} {'dSSIM':>9}")
        td = []
        for t in sorted(grid):
            a = interp(r_off, "x", t)
            b = interp(r_on, "x", t)
            dp = b["eval_psnr"] - a["eval_psnr"]
            ds = b["eval_ssim"] - a["eval_ssim"]
            td.append((t, a["completed_optimizer_steps"], b["completed_optimizer_steps"], dp, ds))
            print(f"{t:>8.3f} {a['completed_optimizer_steps']:>10.1f} "
                  f"{b['completed_optimizer_steps']:>9.1f} | {dp:>+8.4f} {ds:>+9.5f}")
        print("  (linear interpolation between adjacent eval points; nothing is "
              "extrapolated beyond the common time range)")

        ms_off = r_off[-1]["cum_mapping_seconds"] / max(r_off[-1]["completed_optimizer_steps"], 1) * 1000
        ms_on = r_on[-1]["cum_mapping_seconds"] / max(r_on[-1]["completed_optimizer_steps"], 1) * 1000
        print(f"  per-step cost: {OFF_NAME} {ms_off:.3f} ms   {ON_NAME} {ms_on:.3f} ms "
              f"({(ms_on/ms_off - 1) * 100:+.1f} %)")

        # ---- threshold analysis, one common threshold ---------------------
        thr = 21.0
        s_off = steps_to_threshold(r_off, "eval_psnr", thr)
        s_on = steps_to_threshold(r_on, "eval_psnr", thr)
        print(f"\n-- (3) steps to reach PSNR >= {thr} (same threshold, exploratory choice) --")
        print(f"  {OFF_NAME}: {s_off if s_off is not None else 'not reached'}")
        print(f"  {ON_NAME}: {s_on if s_on is not None else 'not reached'}")
        if s_off is not None and s_on is not None:
            print(f"  difference: {s_on - s_off:+.0f} steps "
                  f"({(s_on/s_off - 1) * 100:+.2f} %)")

        # ---- structural residual, recomputed from the raw rows ------------
        print("\n-- (4) structural residual (raw rows, fixed reference weights) --")
        for tag, rows, nm in ((OFF_TAG, r_off, OFF_NAME), (ON_TAG, r_on, ON_NAME)):
            first, last = rows[0], rows[-1]
            s0, s1 = first["struct_shape"], last["struct_shape"]
            o0, o1 = first["struct_ori"], last["struct_ori"]
            ratio = (s0 / s1) if s1 > 0 else float("inf")
            print(f"  {nm}: struct_shape {s0:.6f} (step {first['completed_optimizer_steps']:.0f})"
                  f" -> {s1:.6f} (step {last['completed_optimizer_steps']:.0f})"
                  f"  = x{ratio:.1f}")
            print(f"  {nm}: struct_ori   {o0:.6f} -> {o1:.6f}"
                  f"{'' if o0 > 0 else '   (starts at exactly 0)'}")

        summary_rows.append(dict(label=off_lbl, d_psnr=d_psnr, d_ssim=d_ssim,
                                 d_lpips=d_lpips, steps=steps_list,
                                 ms_off=ms_off, ms_on=ms_on, td=td,
                                 s_off=s_off, s_on=s_on, thr=thr))

    # ---- control ---------------------------------------------------------
    if ctrl_lbl:
        print("\n" + "=" * 92)
        print(f"OFF/OFF CONTROL (same frozen scene, one process): {ctrl_lbl}")
        print("=" * 92)
        import glob
        files = sorted(glob.glob(os.path.join(ROOT, ctrl_lbl, "frozen", "*_metrics.csv")))
        if len(files) >= 2:
            a, b = read_metrics(files[0]), read_metrics(files[1])
            na = os.path.basename(files[0]).split("_")[0]
            nb = os.path.basename(files[1]).split("_")[0]
            print(f"{'steps':>6} {'PSNR ' + na:>14} {'PSNR ' + nb:>14} {'d':>9} "
                  f"{'dSSIM':>10}")
            wr, ws = [], []
            for x, y in zip(a, b):
                dp = y["eval_psnr"] - x["eval_psnr"]
                ds = y["eval_ssim"] - x["eval_ssim"]
                wr.append(abs(dp)); ws.append(abs(ds))
                print(f"{x['completed_optimizer_steps']:>6.0f} {x['eval_psnr']:>14.5f} "
                      f"{y['eval_psnr']:>14.5f} {dp:>+9.5f} {ds:>+10.6f}")
            print(f"  |dPSNR| max = {max(wr):.5f} dB   |dSSIM| max = {max(ws):.6f}")
            last_dp = b[-1]["eval_psnr"] - a[-1]["eval_psnr"]
            print(f"  final-point tolerance: |dPSNR| = {abs(last_dp):.5f} dB")
            if summary_rows:
                s = summary_rows[0]
                fin = s["d_psnr"][-1]
                print(f"\n  observed ON-OFF final dPSNR = {fin:+.5f} dB "
                      f"= {abs(fin)/abs(last_dp):.1f}x the OFF/OFF tolerance")

    print("\nNOTE: evaluation views are keyframes whose ground-truth images were also "
          "sampled during training, so this is NOT a strictly held-out set; the "
          "initialisation colour provenance cannot be excluded.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
