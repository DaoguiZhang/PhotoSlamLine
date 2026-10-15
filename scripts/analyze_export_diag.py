#!/usr/bin/env python3
"""Attribute frame-to-frame position steps to the export's post-hoc reconstruction.

The exporters rebuild every frame pose as
    T = (relative pose recorded at tracking time) x (FINAL pose of the frame's
        reference keyframe, or of the ancestor reached by the culling walk)
so a frame's exported pose depends on which keyframe it references and on how
many culling hops were needed. This script joins the per-frame export diagnosis
(PHOTO_SLAM_EXPORT_DIAG) with the input frame order and reports whether large
steps coincide with a CHANGE of reference keyframe or with culling hops.

usage: analyze_export_diag.py <diag_file> <trajectory_tum> <seq_dir> [lo] [hi]
"""
import math
import sys


def rgb_index(seq, tol=0.02):
    """Map a timestamp to its INPUT FRAME INDEX (feeding order).

    Nearest value within tol rather than an exact key, because a text stream
    with a limited precision setting can round the timestamp (the exported
    trajectory uses 6 decimals, a diagnostic stream may use fewer).
    """
    import bisect
    order = []
    for line in open(seq + "/rgb.txt", errors="replace"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        order.append(float(s.split()[0]))
    sorted_ts = sorted(order)
    pos_of = {t: i for i, t in enumerate(order)}

    def lookup(t):
        i = bisect.bisect_left(sorted_ts, t)
        best = None
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(sorted_ts) and abs(sorted_ts[j] - t) <= tol:
                if best is None or abs(sorted_ts[j] - t) < abs(sorted_ts[best] - t):
                    best = j
        return pos_of[sorted_ts[best]] if best is not None else None

    return lookup


def main():
    diag, traj, seq = sys.argv[1], sys.argv[2], sys.argv[3]
    lo = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    hi = int(sys.argv[5]) if len(sys.argv) > 5 else 10 ** 9
    idx = rgb_index(seq)

    pos = {}
    for line in open(traj, errors="replace"):
        p = line.split()
        if len(p) < 8:
            continue
        k = idx(float(p[0]))
        if k is not None:
            pos[k] = (float(p[1]), float(p[2]), float(p[3]))

    rows = []
    for line in open(diag, errors="replace"):
        if line.startswith("#"):
            continue
        p = line.split()
        # ts refKF_id refKF_map liveKF_id cull_hops x y z
        if len(p) < 8:
            continue
        k = idx(float(p[0]))
        if k is None:
            continue
        rows.append(dict(f=k, ts=float(p[0]), ref=int(p[1]), refmap=p[2],
                         live=int(p[3]), hops=int(p[4])))
    rows.sort(key=lambda r: r["f"])

    print(f"{'frame':>6} {'refKF':>7} {'refmap':>6} {'liveKF':>7} {'hops':>4} "
          f"{'step':>9}  refKF_changed")
    same, changed, hopped = [], [], []
    step_rows = []
    prev = None
    for r in rows:
        step = float("nan")
        ch = "-"
        if prev is not None:
            a, b = pos.get(prev["f"]), pos.get(r["f"])
            if a and b:
                step = math.dist(a, b)
            ch = "yes" if (r["ref"] != prev["ref"] or r["live"] != prev["live"]) else "no"
            if step == step:
                (changed if ch == "yes" else same).append(step)
                step_rows.append((r["f"], step, ch))
        if r["hops"]:
            hopped.append((r["f"], r["hops"]))
        if lo <= r["f"] <= hi:
            print(f"{r['f']:>6} {r['ref']:>7} {r['refmap']:>6} {r['live']:>7} "
                  f"{r['hops']:>4} {step:>9.4f}  {ch}")
        prev = r

    def stats(v):
        if not v:
            return "n=0"
        v = sorted(v)
        return (f"n={len(v)} median={v[len(v)//2]:.4f} p95={v[int(0.95*(len(v)-1))]:.4f} "
                f"max={v[-1]:.4f}")

    print("\nstep size when the reference/live keyframe is UNCHANGED: " + stats(same))
    print("step size when the reference/live keyframe CHANGED : " + stats(changed))
    print(f"frames needing culling hops: {len(hopped)}"
          + (f" (e.g. {hopped[:8]})" if hopped else ""))

    # Largest steps and whether the reconstruction identity changed with them.
    top = sorted(step_rows, key=lambda x: -x[1])[:8]
    print("\nlargest steps (frame, step, refKF changed):")
    for f, st, ch in top:
        print(f"  frame={f} step={st:.4f} refKF_changed={ch}")


if __name__ == "__main__":
    main()
