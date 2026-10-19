#!/usr/bin/env python3
"""Continuity audit for an exported trajectory.

Motivation
----------
System::SaveTrajectoryTUM writes every frame in mlRelativeFramePoses but pins a
single origin Two = (first keyframe of the *current* map)^-1. System::
SaveTrajectoryEuRoC instead selects the map with the most keyframes and skips
frames whose reference keyframe lives in another map, i.e. it emits a coherent
fragment of one map.

Therefore: if a run ends with more than one populated map, the TUM file mixes
frames whose reference keyframes belong to different world frames while forcing
one origin. That is invisible in an aggregate ATE number, but it shows up as
large, isolated jumps between consecutive frames.

This script measures exactly that, without any ground truth and without
re-running SLAM: consecutive-frame translation steps over the (timestamp
deduplicated) trajectory. A frame change appears as a jump of order metres,
while genuine handheld motion between frames is a few centimetres.

usage: traj_continuity_audit.py FILE [FILE ...]
       traj_continuity_audit.py --jump-threshold 0.5 FILE ...
"""
import sys
import numpy as np


def load_dedup(path):
    """Return (t, xyz) with duplicate timestamps collapsed (first occurrence).

    Duplicate (timestamp, pose) lines are emitted by the Tracking fix for
    frames that were never fed to the pose estimator, so they carry no new
    information and would show as zero-length steps.
    """
    t, xyz = [], []
    seen = set()
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 4:
                continue
            ts = float(p[0])
            if ts in seen:
                continue
            seen.add(ts)
            t.append(ts)
            xyz.append([float(p[1]), float(p[2]), float(p[3])])
    return np.asarray(t), np.asarray(xyz)


def audit(path, jump_threshold=0.5):
    t, xyz = load_dedup(path)
    if len(t) < 3:
        return None
    order = np.argsort(t)
    t, xyz = t[order], xyz[order]
    d = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    dt = np.diff(t)
    bad = np.where(d > jump_threshold)[0]
    return {
        "file": path,
        "n": len(t),
        "step_med": float(np.median(d)),
        "step_p95": float(np.percentile(d, 95)),
        "step_max": float(d.max()),
        "n_jumps": int(len(bad)),
        "jump_times": [(float(t[i]), float(d[i])) for i in bad[:4]],
        "median_dt": float(np.median(dt)),
    }


def main():
    args = [a for a in sys.argv[1:]]
    thr = 0.5
    if "--jump-threshold" in args:
        i = args.index("--jump-threshold")
        thr = float(args[i + 1])
        del args[i:i + 2]
    if not args:
        print(__doc__)
        return 1

    print(f"jump_threshold = {thr} m per consecutive frame")
    print(f"{'file':<52}{'n':>6}{'med':>9}{'p95':>9}{'max':>10}{'jumps':>7}")
    flagged = 0
    for p in args:
        r = audit(p, thr)
        if r is None:
            print(f"{p:<52}{'-- too short --':>30}")
            continue
        name = r["file"].split("/")[-1]
        print(f"{name:<52}{r['n']:>6}{r['step_med']:>9.4f}{r['step_p95']:>9.4f}"
              f"{r['step_max']:>10.4f}{r['n_jumps']:>7}")
        if r["n_jumps"]:
            flagged += 1
            for ts, step in r["jump_times"]:
                print(f"    jump at t={ts:.6f} step={step:.4f} m")
    print(f"\nfiles with a discontinuity > {thr} m: {flagged}/{len(args)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
