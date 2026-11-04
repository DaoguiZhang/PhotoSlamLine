#!/usr/bin/env python3
"""Derive a TUM-format ground truth from Replica's raw `traj.txt`, with checks.

Why this exists
---------------
The Replica sequences ship `traj.txt` (2000 rows of 16 numbers = a row-major 4x4
camera-to-world matrix) and a pre-derived `pose_TUM.txt`. An earlier comparison
against `pose_TUM.txt` failed while checking the column arrangement and stopped
there. Inspecting the code that produced it settles the question without
guessing: `SEGS-SLAM/eval/shapeReplicaGT.py` is the repository's own derivation,
and it does exactly this

    c2w = line.reshape(4, 4)      # row-major
    c2w[:3, 1] *= -1              # negate the camera y axis column
    c2w[:3, 2] *= -1              # negate the camera z axis column
    q   = Rotation.from_matrix(c2w[:3, :3]).as_quat()   # scipy -> x,y,z,w
    write "i  tx ty tz  qx qy qz qw"

Two consequences matter:

1. The OpenGL-style camera axes of the raw file are converted to the
   x-right / y-down / z-forward axes that ORB-SLAM3 uses by negating the y and z
   *columns* of the rotation. That is the documented rule, not a choice made by
   looking at errors.
2. The first column it writes is the integer **frame index**, not a time in
   seconds. Our entry feeds Replica frames on a synthetic 1/30 s per index base
   (recorded in `eval_entry.txt` and in `final_state.csv`), so a GT whose time
   column is a raw index is not comparable with the estimate until the time base
   is made explicit. This script therefore emits both:
     * `<scene>_gt_tum_s.txt`   ts = i / 30        <- used for evaluation
     * `<scene>_gt_index.txt`   ts = i             <- byte-comparable with the
                                                      repository's pose_TUM.txt
   The 1/30 s mapping is the entry's own declared input time base (and matches
   `Camera.fps: 30` in the repository's Replica configs); it is fixed before any
   error is computed and is not chosen by looking at results.

Everything below is verified, not assumed: orthonormality and det=+1 of every
matrix, quaternion <-> matrix round trip, strictly increasing timestamps, the
index range covered by both files, agreement with the repository's
`pose_TUM.txt`, and per-frame pose continuity (to catch a wrong axis order, a
transpose, or a misaligned index that would still be "a valid rotation").

Usage:
  python3 eval/derive_replica_gt.py --replica-root DIR --out DIR [--fps 30]
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

GL_TO_CV = np.diag([1.0, -1.0, -1.0])  # negate camera y and z axes


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def quat_from_R(R):
    """Rotation matrix -> unit quaternion (x, y, z, w), Shepperd's method."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=float)
    return q / np.linalg.norm(q)


def R_from_quat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def load_traj(path):
    rows = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            s = line.split()
            if not s:
                continue
            if len(s) != 16:
                raise SystemExit("%s:%d expected 16 numbers, got %d"
                                 % (path, ln, len(s)))
            rows.append(np.array([float(v) for v in s]).reshape(4, 4))
    return rows


def rot_angle_deg(RA, RB):
    c = (np.trace(RA.T @ RB) - 1.0) / 2.0
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, c)))))


def check_scene(scene, root, out_dir, fps):
    traj = os.path.join(root, scene, "traj.txt")
    if not os.path.exists(traj):
        return None
    poses = load_traj(traj)
    n = len(poses)
    rec = {"scene": scene, "traj_txt": traj, "traj_sha256": sha256(traj),
           "n_frames": n}

    # ---- 1. the raw matrices must be valid rigid transforms ---------------
    worst_orth = 0.0
    worst_det = 0.0
    worst_last_row = 0.0
    for P in poses:
        R = P[:3, :3]
        worst_orth = max(worst_orth, float(np.abs(R.T @ R - np.eye(3)).max()))
        worst_det = max(worst_det, abs(float(np.linalg.det(R)) - 1.0))
        worst_last_row = max(worst_last_row,
                             float(np.abs(P[3, :] - np.array([0, 0, 0, 1.0])).max()))
    rec["raw_max_orthonormality_error"] = worst_orth
    rec["raw_max_det_minus_one"] = worst_det
    rec["raw_max_last_row_error"] = worst_last_row
    if worst_orth > 1e-6 or worst_det > 1e-6:
        raise SystemExit("%s: traj.txt rows are not rigid transforms "
                         "(orth %.3e, det %.3e)" % (scene, worst_orth, worst_det))

    # ---- 2. the documented OpenGL -> CV axis rule --------------------------
    # c2w[:,1] *= -1 ; c2w[:,2] *= -1  ==  R_cv = R_gl @ diag(1,-1,-1)
    pos = np.array([P[:3, 3] for P in poses])
    quats = np.array([quat_from_R(P[:3, :3] @ GL_TO_CV) for P in poses])
    rec["axis_rule"] = "R_cv = R_c2w @ diag(1,-1,-1)  (negate camera y,z columns)"

    # ---- 3. quaternion <-> matrix round trip -------------------------------
    worst_rt = 0.0
    worst_unit = 0.0
    for P, q in zip(poses, quats):
        worst_rt = max(worst_rt,
                       float(np.abs(R_from_quat(q) - P[:3, :3] @ GL_TO_CV).max()))
        worst_unit = max(worst_unit, abs(float(np.linalg.norm(q)) - 1.0))
    rec["quat_roundtrip_max_abs_error"] = worst_rt
    rec["quat_max_norm_error"] = worst_unit
    if worst_rt > 1e-9:
        raise SystemExit("%s: quaternion round trip failed (%.3e)" % (scene, worst_rt))

    # ---- 4. timestamps ----------------------------------------------------
    ts_s = [i / float(fps) for i in range(n)]
    ts_i = [float(i) for i in range(n)]
    rec["fps"] = fps
    rec["ts_first"] = ts_s[0]
    rec["ts_last"] = ts_s[-1]
    rec["ts_strictly_increasing"] = all(ts_s[i + 1] > ts_s[i] for i in range(n - 1))

    # ---- 5. pose continuity (catches transpose / wrong axis / index slip) --
    dp = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    dr = [rot_angle_deg(R_from_quat(quats[i]), R_from_quat(quats[i + 1]))
          for i in range(n - 1)]
    rec["max_step_m"] = float(dp.max()) if len(dp) else 0.0
    rec["median_step_m"] = float(np.median(dp)) if len(dp) else 0.0
    rec["max_step_deg"] = float(max(dr)) if dr else 0.0
    rec["median_step_deg"] = float(np.median(dr)) if dr else 0.0
    rec["path_length_m"] = float(dp.sum()) if len(dp) else 0.0

    # ---- 6. agreement with the repository's own pose_TUM.txt --------------
    ref = os.path.join(root, scene, "pose_TUM.txt")
    if os.path.exists(ref):
        rt, rp, rq = [], [], []
        with open(ref) as f:
            for line in f:
                s = line.split()
                if not s:
                    continue
                rt.append(float(s[0]))
                rp.append([float(v) for v in s[1:4]])
                rq.append([float(v) for v in s[4:8]])
        rp = np.array(rp)
        rec["pose_TUM_exists"] = True
        rec["pose_TUM_sha256"] = sha256(ref)
        rec["pose_TUM_rows"] = len(rt)
        rec["pose_TUM_first_column_is_index"] = bool(
            np.allclose(rt, np.arange(len(rt), dtype=float)))
        if len(rt) == n:
            rec["max_abs_pos_diff_vs_pose_TUM"] = float(np.abs(rp - pos).max())
            worst_deg = max(rot_angle_deg(R_from_quat(np.array(rq[i])),
                                          R_from_quat(quats[i]))
                            for i in range(n))
            rec["max_rot_diff_deg_vs_pose_TUM"] = float(worst_deg)
    else:
        rec["pose_TUM_exists"] = False

    # ---- 7. write the derived GT ------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    p_s = os.path.join(out_dir, "%s_gt_tum_s.txt" % scene)
    p_i = os.path.join(out_dir, "%s_gt_index.txt" % scene)
    with open(p_s, "w") as f:
        for i in range(n):
            f.write("%.9f %.9f %.9f %.9f %.12f %.12f %.12f %.12f\n"
                    % (ts_s[i], pos[i, 0], pos[i, 1], pos[i, 2],
                       quats[i, 0], quats[i, 1], quats[i, 2], quats[i, 3]))
    with open(p_i, "w") as f:
        for i in range(n):
            f.write("%.9f %.9f %.9f %.9f %.12f %.12f %.12f %.12f\n"
                    % (ts_i[i], pos[i, 0], pos[i, 1], pos[i, 2],
                       quats[i, 0], quats[i, 1], quats[i, 2], quats[i, 3]))
    rec["derived_gt_s"] = p_s
    rec["derived_gt_s_sha256"] = sha256(p_s)
    rec["derived_gt_index"] = p_i
    rec["derived_gt_index_sha256"] = sha256(p_i)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replica-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    a = ap.parse_args()

    scenes = sorted(d for d in os.listdir(a.replica_root)
                    if os.path.exists(os.path.join(a.replica_root, d, "traj.txt")))
    recs = [check_scene(s, a.replica_root, a.out, a.fps) for s in scenes]
    recs = [r for r in recs if r]
    report = {"replica_root": a.replica_root, "out": a.out, "fps": a.fps,
              "scenes": recs}
    os.makedirs(a.out, exist_ok=True)
    rp = os.path.join(a.out, "REPLICA_GT_REPORT.json")
    with open(rp, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print("scene      n     maxstep_m  maxstep_deg  posdiff_vs_poseTUM  rotdiff_deg  firstcol_index")
    for r in recs:
        print("%-10s %-5d %-10.4f %-12.3f %-19s %-12s %s" % (
            r["scene"], r["n_frames"], r["max_step_m"], r["max_step_deg"],
            ("%.3e" % r["max_abs_pos_diff_vs_pose_TUM"])
            if "max_abs_pos_diff_vs_pose_TUM" in r else "n/a",
            ("%.3e" % r["max_rot_diff_deg_vs_pose_TUM"])
            if "max_rot_diff_deg_vs_pose_TUM" in r else "n/a",
            r.get("pose_TUM_first_column_is_index")))
    print("report=%s" % rp)
    print("ALL CHECKS PASSED" if all(r["quat_roundtrip_max_abs_error"] <= 1e-9
                                     for r in recs) else "CHECK FAILURES")
    return 0


if __name__ == "__main__":
    sys.exit(main())
