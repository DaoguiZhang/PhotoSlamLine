#!/usr/bin/env python3
"""Read-only check of the frames whose epoch stamp lands on the merge frame.

In three finished runs the epoch advances *at* the frame the merge event names
(gap = 0) instead of at the next one. The evaluator claims to exclude such a
frame from every era. This script verifies that claim from the artefacts, and
measures what the exclusion is worth, without re-running anything and without
using max_dt as evidence about coordinate frames.

For each case it prints:

  anchor        the frame index named by the MAP_MERGE event, its epoch stamp,
                its map id, and whether it has a valid pose
  excluded      whether its timestamp appears in any eval/online_ep*.txt (the
                only place era frames come from). Absent means it contributes to
                neither the alignment, nor RPE, nor the per-frame error table
                that feeds the correct-coverage numerator.
  next_era      the era the next posed frame opens, with its frame range,
                n_pairs and ATE
  anchor_error  the anchor frame's own error if it were scored in that era,
                computed with the era's own align_R / align_scale / align_t

Usage: python3 eval/boundary_cases.py --root <run root> [--out FILE]
"""
import argparse
import csv
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mapmeta  # noqa: E402

GT = {
    ("ETH3D", "desk_3"): "/workspace/code/SEGS-SLAM/datasets/eth3d/desk_3/groundtruth.txt",
}


def quat_to_R(q):
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0:
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    x, y, z, w = x / n, y / n, z / n, w / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def mv(R, p):
    return [sum(R[i][k] * p[k] for k in range(3)) for i in range(3)]


def matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)]


def load_gt(path):
    rows = []
    for line in open(path):
        s = line.split()
        if len(s) < 8:
            continue
        try:
            v = [float(x) for x in s[:8]]
        except ValueError:
            continue
        rows.append(v)
    rows.sort(key=lambda r: r[0])
    return rows


def gt_at(rows, ts):
    lo, hi = 0, len(rows) - 1
    if ts < rows[0][0] or ts > rows[-1][0]:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if rows[mid][0] < ts:
            lo = mid + 1
        else:
            hi = mid
    b = rows[lo]
    a = rows[max(0, lo - 1)]
    if b[0] == a[0]:
        return b[1:4], b[4:8]
    w = (ts - a[0]) / (b[0] - a[0])
    p = [a[1 + i] * (1 - w) + b[1 + i] * w for i in range(3)]
    q = [a[4 + i] * (1 - w) + b[4 + i] * w for i in range(4)]
    return p, q


def era_rows(d, events, online):
    """The eras the CURRENT record produces, recomputed here.

    Stale `online_ep*.txt` files from an earlier pass with a different split can
    remain in `eval/`, so the file names are never used to decide membership.
    """
    exact, _ = mapmeta.merge_split_frames(events)
    eps = mapmeta.align_epochs(online, split_frames=exact)
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/eval_runs/supplement_v4")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cases = []
    for ds in sorted(os.listdir(a.root)):
        p = os.path.join(a.root, ds)
        if not os.path.isdir(p):
            continue
        for scene in sorted(os.listdir(p)):
            for mod in sorted(os.listdir(os.path.join(p, scene))):
                for arm in sorted(os.listdir(os.path.join(p, scene, mod))):
                    ad = os.path.join(p, scene, mod, arm)
                    for rep in sorted(os.listdir(ad)):
                        d = os.path.join(ad, rep)
                        if not os.path.exists(os.path.join(d, "status.txt")):
                            continue
                        cases.append((ds, scene, mod, arm, rep, d))

    rows = []
    for ds, scene, mod, arm, rep, d in cases:
        events = mapmeta.read_events(d) or []
        merges = [e for e in events
                  if e.get("event") == "MAP_MERGE" and e.get("frame") is not None
                  and e["frame"] >= 0]
        if not merges:
            continue
        online = mapmeta.read_online(d)
        posed = [r for r in online if str(r.get("pose_valid")) == "1"]
        if not posed:
            continue
        first_epoch_frame = {}
        for r in posed:
            first_epoch_frame.setdefault(int(r["epoch"]), r)
        for m in merges:
            F = int(m["frame"])
            nxt = next((r for r in posed if r["frame"] > F), None)
            if nxt is None:
                continue
            f_row = next((r for r in posed if r["frame"] == F), None)
            # gap = 0 only when this frame itself carries the new epoch
            if f_row is None or int(f_row["epoch"]) != int(nxt["epoch"]):
                continue

            # which era owns the next posed frame, and is F in any era?
            eps = era_rows(d, events, online)
            era_idx = None
            era_meta = {}
            anchor_in_any_era = False
            for k, e in enumerate(eps):
                tss = {round(float(r["ts"]), 6) for r in e["rows"]}
                if round(float(nxt["ts"]), 6) in tss:
                    era_idx = k
                    j = os.path.join(d, "eval", "online_ep%02d.json" % k)
                    if os.path.exists(j):
                        era_meta = json.load(open(j))
                    era_meta["_n_frames"] = len(e["rows"])
                if round(float(f_row["ts"]), 6) in tss:
                    anchor_in_any_era = True

            err_f = rot_f = None
            gtp = GT.get((ds, scene))
            if gtp and os.path.exists(gtp) and era_meta.get("align_R"):
                g = gt_at(load_gt(gtp), float(f_row["ts"]))
                if g:
                    R = era_meta["align_R"]
                    t = era_meta["align_t"]
                    s = era_meta.get("align_scale", 1.0) or 1.0
                    p = mv(R, f_row["t"])
                    p = [s * p[i] + t[i] for i in range(3)]
                    err_f = math.sqrt(sum((p[i] - g[0][i]) ** 2 for i in range(3)))
                    Rf = matmul(R, quat_to_R(f_row["q"]))
                    Rg = quat_to_R(g[1])
                    tr = sum(Rf[i][k] * Rg[i][k] for i in range(3) for k in range(3))
                    c = max(-1.0, min(1.0, (tr - 1) / 2))
                    rot_f = math.degrees(math.acos(c))

            era_ate = (era_meta.get("ate_se3") or {}).get("rmse")
            n_era = era_meta.get("n_pairs") or era_meta.get("_n_frames")
            ate_with = None
            if era_ate is not None and err_f is not None and n_era:
                ate_with = math.sqrt((n_era * era_ate ** 2 + err_f ** 2) / (n_era + 1))
            rows.append({
                "dataset": ds, "scene": scene, "modality": mod, "arm": arm,
                "repeat": rep, "dir": d,
                "anchor_frame": F, "anchor_ts": f_row["ts"],
                "anchor_epoch_stamp": f_row["epoch"],
                "anchor_map_id": f_row["map_id"],
                "merge_event_epoch": m.get("epoch", ""),
                "next_posed_frame": nxt["frame"], "next_posed_epoch": nxt["epoch"],
                "next_posed_era_index": era_idx,
                "anchor_in_any_era_file": int(anchor_in_any_era),
                "era_n_pairs": era_meta.get("n_pairs", ""),
                "era_n_frames": era_meta.get("_n_frames", ""),
                "era_ate_m": "" if era_ate is None else round(era_ate, 5),
                "era_ate_if_anchor_counted_m": ("" if ate_with is None
                                                else round(ate_with, 5)),
                "anchor_error_if_scored_m": "" if err_f is None else round(err_f, 4),
                "anchor_rot_if_scored_deg": "" if rot_f is None else round(rot_f, 2),
            })

    keys = ["dataset", "scene", "modality", "arm", "repeat", "anchor_frame",
            "anchor_ts", "anchor_epoch_stamp", "anchor_map_id", "merge_event_epoch",
            "next_posed_frame", "next_posed_epoch", "next_posed_era_index",
            "anchor_in_any_era_file", "era_n_pairs", "era_n_frames", "era_ate_m",
            "era_ate_if_anchor_counted_m",
            "anchor_error_if_scored_m", "anchor_rot_if_scored_deg", "dir"]
    print("gap0_cases=%d" % len(rows))
    for r in rows:
        print("%-28s F=%-5s epoch=%-2s map=%-2s | next=%-5s epoch=%-2s era=%-2s | "
              "anchor_in_era_file=%s | era_n_pairs=%-5s era_ATE=%-8s | "
              "anchor_if_scored=%s m / %s deg | era_ATE_if_counted=%s" % (
                  "%s/%s/%s" % (r["scene"], r["modality"], r["arm"]),
                  r["anchor_frame"], r["anchor_epoch_stamp"], r["anchor_map_id"],
                  r["next_posed_frame"], r["next_posed_epoch"],
                  r["next_posed_era_index"], r["anchor_in_any_era_file"],
                  r["era_n_pairs"], r["era_ate_m"],
                  r["anchor_error_if_scored_m"], r["anchor_rot_if_scored_deg"],
                  r["era_ate_if_anchor_counted_m"]))
    bad = [r for r in rows if r["anchor_in_any_era_file"]]
    print("anchor_frames_found_inside_an_era=%d" % len(bad))
    if a.out:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
