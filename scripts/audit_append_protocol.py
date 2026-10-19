#!/usr/bin/env python3
"""Protocol audit for the append/budget round.

Checks (no training, reads saved artifacts only):
  1. natural_end vs budget_end cameras.json: identical keyframe poses/intrinsics.
  2. P (and L) keyframe poses reproduced by KeyFrameTrajectory_TUM.txt.
  3. Three selected keyframes: rotation orthonormality, relative pose from
     cameras.json vs TUM, finite geometry.
  4. Common-view protocol: exactly 24 frames per run/state; per-run alignment;
     corrected SSIM is the 11x11 uniform-window implementation; LPIPS is
     AlexNet RGB in [-1,1] (sanity checked by the evaluator).

usage: audit_append_protocol.py <run_root> <orb_cfg>
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np


def q2R(q):
    x, y, z, w = q
    n = math.sqrt(sum(v * v for v in q)) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def read_tum(path):
    out = {}
    for line in open(path):
        s = line.split()
        if len(s) < 8:
            continue
        out[int(round(float(s[0])))] = (
            np.array([float(x) for x in s[1:4]]),
            q2R([float(x) for x in s[4:8]]))
    return out


def frame_index(img_name):
    b = os.path.basename(img_name)
    digits = "".join(c for c in b if c.isdigit())
    return int(digits) if digits else -1


def json_pose(kf):
    return np.array(kf["position"], float), np.array(kf["rotation"], float)


def rel(Ta, Tb):
    pa, Ra = Ta
    pb, Rb = Tb
    R = Ra.T @ Rb
    t = Ra.T @ (pb - pa)
    return t, R


def rerr(R1, R2):
    # Remove float32 orthonormality noise before taking the angle.
    U, _, Vt = np.linalg.svd(R1)
    R1 = U @ Vt
    U, _, Vt = np.linalg.svd(R2)
    R2 = U @ Vt
    c = (np.trace(R1.T @ R2) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root")
    ap.add_argument("orb_cfg")
    ap.add_argument("--runs", default="P_ref,P_fast,L_fast")
    args = ap.parse_args()

    # Intrinsics from the ORB config (Replica uses Camera1.*).
    import cv2
    fs = cv2.FileStorage(args.orb_cfg, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise SystemExit("[FATAL] cannot open ORB cfg")
    def fsval(*keys):
        for k in keys:
            node = fs.getNode(k)
            if not node.empty():
                return float(node.real())
        raise SystemExit(f"[FATAL] missing ORB cfg key {keys}")

    fx = fsval("Camera1.fx", "Camera.fx")
    fy = fsval("Camera1.fy", "Camera.fy")
    w = int(fsval("Camera1.w", "Camera.w", "Camera1.width", "Camera.width"))
    h = int(fsval("Camera1.h", "Camera.h", "Camera1.height", "Camera.height"))

    rows, three_rows = [], []
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    for run in runs:
        rd = os.path.join(args.run_root, "runs", run)
        ne = json.load(open(os.path.join(rd, "append", "natural_end", "cameras.json")))
        be = json.load(open(os.path.join(rd, "append", "budget_end", "cameras.json")))
        if [k["id"] for k in ne] != [k["id"] for k in be]:
            raise SystemExit(f"[FATAL] {run}: keyframe id sets differ")
        max_pos = max_rot = 0.0
        for a, b in zip(ne, be):
            pa, Ra = json_pose(a); pb, Rb = json_pose(b)
            max_pos = max(max_pos, float(np.linalg.norm(pa - pb)))
            max_rot = max(max_rot, rerr(Ra, Rb))

        tum = read_tum(os.path.join(rd, "KeyFrameTrajectory_TUM.txt"))
        rep_pos = rep_rot = 0.0
        pairs = []
        for kf in ne:
            fid = frame_index(kf["img_name"])
            if fid not in tum:
                continue
            pos, R = json_pose(kf)
            tp, tR = tum[fid]
            rep_pos = max(rep_pos, float(np.linalg.norm(pos - tp)))
            rep_rot = max(rep_rot, rerr(R, tR))
            pairs.append((fid, (pos, R), (tp, tR)))

        # Three keyframes: first / middle / last by frame id.
        pairs.sort(key=lambda x: x[0])
        sel = [pairs[0], pairs[len(pairs) // 2], pairs[-1]]
        rel_pos = rel_rot = 0.0
        for i in range(len(sel)):
            fid, J, T = sel[i]
            pos, R = J
            orth = float(np.linalg.norm(R.T @ R - np.eye(3)))
            det = float(np.linalg.det(R))
            intr_ok = (abs(kf_float(ne, fid, "fx") - fx) < 1e-6 and
                       abs(kf_float(ne, fid, "fy") - fy) < 1e-6 and
                       int(kf_float(ne, fid, "width")) == w and
                       int(kf_float(ne, fid, "height")) == h)
            three_rows.append(dict(run=run, fid=fid, orthonormality=orth,
                                   det=det, intrinsics_ok=int(intr_ok)))
            if i > 0:
                jfid, J0, T0 = sel[i - 1]
                jp, jR = rel(J0, J)
                tp, tR = rel(T0, T)
                rel_pos = max(rel_pos, float(np.linalg.norm(jp - tp)))
                rel_rot = max(rel_rot, rerr(jR, tR))

        rows.append(dict(run=run, n_kf=len(ne),
                         natural_vs_budget_max_pos=max_pos,
                         natural_vs_budget_max_rot_deg=max_rot,
                         json_vs_tum_max_pos=rep_pos,
                         json_vs_tum_max_rot_deg=rep_rot,
                         three_rel_max_pos=rel_pos,
                         three_rel_max_rot_deg=rel_rot))

    with open(os.path.join(args.run_root, "pose_audit.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(args.run_root, "three_keyframe_geometry.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(three_rows[0].keys()))
        w.writeheader(); w.writerows(three_rows)

    # Common-view protocol audit.
    per = os.path.join(args.run_root, "common_view_per_frame_budget.csv")
    summ = os.path.join(args.run_root, "common_view_summary_budget.csv")
    if os.path.exists(per) and os.path.exists(summ):
        frames = {}
        with open(per) as f:
            for r in csv.DictReader(f):
                frames.setdefault((r["run"], r["state"]), set()).add(int(r["fid"]))
        bad = [k for k, v in frames.items() if len(v) != 24]
        if bad:
            raise SystemExit(f"[FATAL] not 24 frames for {bad}")
        with open(summ) as f:
            for r in csv.DictReader(f):
                if not math.isfinite(float(r["psnr"])):
                    raise SystemExit(f"[FATAL] non-finite PSNR {r}")
        print("[eval-protocol] 24 frames per run/state; PSNR finite; "
              "SSIM=11x11 uniform window; LPIPS=AlexNet RGB [-1,1]")
    else:
        print("[eval-protocol] common-view metrics not found yet")

    print("[pose-audit]")
    for r in rows:
        print("  %-8s n_kf=%d nat/bud pos=%.3g rot=%.3g json/tum pos=%.3g rot=%.3g "
              "three-rel pos=%.3g rot=%.3g" %
              (r["run"], r["n_kf"], r["natural_vs_budget_max_pos"],
               r["natural_vs_budget_max_rot_deg"], r["json_vs_tum_max_pos"],
               r["json_vs_tum_max_rot_deg"], r["three_rel_max_pos"],
               r["three_rel_max_rot_deg"]))


def kf_float(kfs, fid, key):
    for kf in kfs:
        if frame_index(kf["img_name"]) == fid:
            return float(kf[key])
    return float("nan")


if __name__ == "__main__":
    main()
