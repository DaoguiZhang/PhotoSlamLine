#!/usr/bin/env python3
"""Quality-step and quality-total-recovery-time curves for gba-recovery-v1.

Standard-library CSV + PIL only (matplotlib is not installed in this image).
Reads <root>/summary.csv and <root>/lpips.csv, writes PNGs to <root>/curves/.
"""
import csv
import os
import sys
from PIL import Image, ImageDraw

COLORS = {
    "PoseOnly": (30, 110, 200),
    "PoseOnly_r2": (30, 110, 200),
    "PoseAndMap": (200, 80, 30),
    "PoseAndMap_r2": (200, 80, 30),
}
DASHED = {"PoseOnly_r2", "PoseAndMap_r2"}
ARMS = ["PoseOnly", "PoseAndMap", "PoseAndMap_r2", "PoseOnly_r2"]


def load_summary(root):
    data = {a: [] for a in ARMS}
    with open(os.path.join(root, "summary.csv")) as f:
        for r in csv.DictReader(f):
            data.setdefault(r["arm"], []).append(
                dict(step=int(r["step"]), psnr=float(r["psnr"]),
                     ssim=float(r["ssim"]),
                     total=float(r["total_seconds"]),
                     train=float(r["train_seconds"]),
                     sync=float(r["sync_seconds"])))
    for a in data:
        data[a].sort(key=lambda x: x["step"])
    return data


def load_lpips(root):
    path = os.path.join(root, "lpips.csv")
    data = {}
    if not os.path.exists(path):
        return data
    with open(path) as f:
        for r in csv.DictReader(f):
            data.setdefault(r["arm"], []).append(
                (int(r["step"]), float(r["lpips"])))
    for a in data:
        data[a].sort()
    return data


def chart(path, series, title, xlabel, ylabel, x_is_time=False):
    W, H, M = 900, 560, 70
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    xs = [x for s in series for x, _ in s["pts"]]
    ys = [y for s in series for _, y in s["pts"]]
    if not xs or not ys:
        img.save(path)
        return
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin:
        xmax = xmin + 1
    pad = (ymax - ymin) * 0.08 or 0.01
    ymin -= pad
    ymax += pad
    def X(v): return M + (v - xmin) / (xmax - xmin) * (W - 2 * M)
    def Y(v): return H - M - (v - ymin) / (ymax - ymin) * (H - 2 * M)
    d.rectangle([M, M, W - M, H - M], outline=(0, 0, 0))
    for i in range(6):
        yv = ymin + (ymax - ymin) * i / 5
        d.line([M, Y(yv), W - M, Y(yv)], fill=(225, 225, 225))
        d.text((5, Y(yv) - 6), f"{yv:.4f}", fill=(0, 0, 0))
    for i in range(6):
        xv = xmin + (xmax - xmin) * i / 5
        d.line([X(xv), H - M, X(xv), H - M + 5], fill=(0, 0, 0))
        d.text((X(xv) - 14, H - M + 8), f"{xv:.3g}", fill=(0, 0, 0))
    d.text((W // 2 - len(title) * 3, 20), title, fill=(0, 0, 0))
    d.text((W // 2 - 60, H - 35), xlabel, fill=(0, 0, 0))
    d.text((8, 40), ylabel, fill=(0, 0, 0))
    ly = M + 6
    for s in series:
        pts = [(X(x), Y(y)) for x, y in s["pts"]]
        color = COLORS.get(s["name"], (0, 0, 0))
        for a, b in zip(pts, pts[1:]):
            if s["name"] in DASHED:
                # simple dashed segment
                steps = max(1, int(((b[0]-a[0])**2 + (b[1]-a[1])**2) ** 0.5 / 8))
                for k in range(steps):
                    if k % 2 == 0:
                        p = (a[0] + (b[0]-a[0])*k/steps, a[1] + (b[1]-a[1])*k/steps)
                        q = (a[0] + (b[0]-a[0])*(k+1)/steps, a[1] + (b[1]-a[1])*(k+1)/steps)
                        d.line([p, q], fill=color, width=2)
            else:
                d.line([a, b], fill=color, width=3)
        for x, y in pts:
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)
        d.text((W - M + 4, ly), s["name"], fill=color)
        ly += 16
    img.save(path)


def main(argv):
    root = argv[0] if argv else "abl_out/gba_recovery_v1"
    out = os.path.join(root, "curves")
    os.makedirs(out, exist_ok=True)
    s = load_summary(root)
    lp = load_lpips(root)
    chart(os.path.join(out, "psnr_vs_step.png"),
          [{"name": a, "pts": [(r["step"], r["psnr"]) for r in s.get(a, [])]}
           for a in ARMS if s.get(a)],
          "PSNR vs optimizer step", "step", "PSNR", False)
    chart(os.path.join(out, "ssim_vs_step.png"),
          [{"name": a, "pts": [(r["step"], r["ssim"]) for r in s.get(a, [])]}
           for a in ARMS if s.get(a)],
          "SSIM vs optimizer step", "step", "SSIM", False)
    chart(os.path.join(out, "psnr_vs_total_time.png"),
          [{"name": a, "pts": [(r["total"], r["psnr"]) for r in s.get(a, [])]}
           for a in ARMS if s.get(a)],
          "PSNR vs recovery time (sync + training)", "seconds", "PSNR", True)
    if lp:
        chart(os.path.join(out, "lpips_vs_step.png"),
              [{"name": a, "pts": lp.get(a, [])} for a in ARMS if lp.get(a)],
              "LPIPS vs optimizer step (lower is better)", "step", "LPIPS", False)
    print("wrote curves to " + out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
