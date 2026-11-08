#!/usr/bin/env python3
"""Acceptance summary for one full-mapping run directory."""
import glob
import os
import re
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_run.py <run_dir>", file=sys.stderr)
        return 2
    d = sys.argv[1]
    log = os.path.join(d, "stdout.log")
    text = open(log, encoding="utf-8", errors="replace").read() if os.path.exists(log) else ""
    meta = open(os.path.join(d, "run.meta"), encoding="utf-8", errors="replace").read() if os.path.exists(os.path.join(d, "run.meta")) else ""

    exit_code = re.findall(r"^exit=(-?\d+)", meta, re.M)
    iters = re.findall(r"Total Iterations:\s*(\d+)", text)
    psnrs = glob.glob(os.path.join(d, "**", "psnr.txt"), recursive=True)
    mean_psnr = None
    n_psnr = 0
    if psnrs:
        vals = []
        for line in open(psnrs[0], encoding="utf-8", errors="replace"):
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                vals.append(float(parts[1]))
        n_psnr = len(vals)
        mean_psnr = sum(vals) / len(vals) if vals else None

    plies = glob.glob(os.path.join(d, "**", "point_cloud.ply"), recursive=True)
    ply = plies[0] if plies else None
    n_vert = None
    ply_size = None
    if ply:
        ply_size = os.path.getsize(ply)
        with open(ply, "rb") as fh:
            head = fh.read(1024).decode("ascii", errors="replace")
        m = re.search(r"element vertex\s+(\d+)", head)
        n_vert = int(m.group(1)) if m else None

    line_loss = len(re.findall(r"\[LINE-LOSS\]", text))
    gauss_empty = "Gaussian model is empty" in text

    print(f"dir={d}")
    print(f"  exit={exit_code[-1] if exit_code else 'NA'}")
    print(f"  total_iterations={iters[-1] if iters else 'NA'}")
    print(f"  psnr_entries={n_psnr} mean_psnr={mean_psnr if mean_psnr is None else round(mean_psnr, 4)}")
    print(f"  ply={ply}")
    print(f"  ply_size={ply_size} vertices={n_vert}")
    print(f"  line_loss_diag_lines={line_loss} model_empty={gauss_empty}")
    ok = bool(exit_code and exit_code[-1] == "0") and iters and int(iters[-1]) > 0 and n_psnr > 0 and n_vert
    print(f"  ACCEPT={'YES' if ok else 'NO'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
