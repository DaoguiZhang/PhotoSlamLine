#!/usr/bin/env python3
"""Decide whether a frame-to-frame discontinuity is already in the ONLINE pose or
only appears in the EXPORTED pose.

Inputs (all produced by one run):
  --trace  PHOTO_SLAM_POSE_TRACE file: per fed frame, ts/frame id/state/is_set/
           n_frame_entries/refKF id/refKF map/parent KF id/online Tcw
  --tum    exported TUM trajectory of the same run
  --diag   PHOTO_SLAM_EXPORT_DIAG file (also carries the export origin Two)
  --seq    dataset directory (rgb.txt defines the INPUT FRAME ORDER)

The exporter writes, for a frame whose reference keyframe is P and whose relative
pose is Tcr,  Tcw_export = Tcr * P * Two, while the estimator produced
Tcw_online = Tcr * P. Hence Twc_export = Two^-1 * Twc_online: the exported
trajectory is the online trajectory moved by ONE rigid transform. Comparing the
two therefore needs Two, and a discontinuity is invariant under it, so a jump
visible online cannot be produced by the export and vice versa.

usage: pose_trace_compare.py --trace F --tum F --diag F --seq DIR [--window 6]
"""
import argparse
import bisect
import math


def q_to_R(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def apply(T, p):
    """T = (R, t) applied to p: R p + t."""
    R, t = T
    return [R[i][0] * p[0] + R[i][1] * p[1] + R[i][2] * p[2] + t[i] for i in range(3)]


def inv(T):
    R, t = T
    Rt = [[R[j][i] for j in range(3)] for i in range(3)]
    tt = [-sum(Rt[i][j] * t[j] for j in range(3)) for i in range(3)]
    return (Rt, tt)


def rgb_order(seq):
    ts = []
    for line in open(seq + "/rgb.txt", errors="replace"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        ts.append(float(s.split()[0]))
    return ts


class Assoc:
    def __init__(self, ts, tol=0.02):
        self.sorted_ts = sorted(ts)
        self.pos_of = {t: i for i, t in enumerate(ts)}
        self.tol = tol

    def __call__(self, t):
        i = bisect.bisect_left(self.sorted_ts, t)
        best = None
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(self.sorted_ts) and abs(self.sorted_ts[j] - t) <= self.tol:
                if best is None or abs(self.sorted_ts[j] - t) < abs(self.sorted_ts[best] - t):
                    best = j
        return self.pos_of[self.sorted_ts[best]] if best is not None else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--tum", required=True)
    ap.add_argument("--diag", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--window", type=int, default=6)
    a = ap.parse_args()

    idx = Assoc(rgb_order(a.seq))

    # export origin + exported positions
    Two, diag = None, {}
    for line in open(a.diag, errors="replace"):
        p = line.split()
        if not p:
            continue
        if p[0] == "#" and len(p) > 2 and p[1] == "Two":
            v = [float(x) for x in p[2:9]]
            Two = (q_to_R(*v[3:7]), v[0:3])
        elif not p[0].startswith("#") and len(p) >= 8:
            k = idx(float(p[0]))
            if k is not None:
                diag[k] = dict(ts=float(p[0]), ref=int(p[1]), refmap=p[2],
                               live=int(p[3]), hops=int(p[4]),
                               p=[float(p[5]), float(p[6]), float(p[7])])

    online = {}
    for line in open(a.trace, errors="replace"):
        if line.startswith("#"):
            continue
        p = line.split()
        # ts frame_id state is_set n_entries refKF_id refKF_map parentKF_id
        # Tcw(tx ty tz qx qy qz qw)  == 15 fields
        if len(p) < 15:
            continue
        k = idx(float(p[0]))
        if k is None:
            continue
        online[k] = dict(ts=float(p[0]), fid=int(p[1]), state=int(p[2]), is_set=int(p[3]),
                         n_entries=int(p[4]), ref=int(p[5]), refmap=int(p[6]),
                         parent=int(p[7]),
                         # Tcw is stored as translation + quaternion (qx qy qz qw)
                         Tcw=(q_to_R(*[float(x) for x in p[11:15]]),
                              [float(x) for x in p[8:11]]))

    tum = {}
    for line in open(a.tum, errors="replace"):
        p = line.split()
        if len(p) < 8:
            continue
        k = idx(float(p[0]))
        if k is not None:
            tum[k] = [float(x) for x in p[1:4]]

    print(f"frames: online_trace={len(online)} export_diag={len(diag)} tum={len(tum)}")
    if Two is None:
        print("no '# Two' line in the diag file: cannot map the online poses into the "
              "exported frame exactly")
        return 1

    # 1) faithfulness of the export: online pose mapped by Two^-1 must equal exported
    Two_inv = inv(Two)
    dev = []
    for k in sorted(set(online) & set(diag)):
        # camera centre = translation of Twc, derived from Twc = Tcw^-1
        Twc = inv(online[k]["Tcw"])
        centre = Twc[1]
        p_mapped = apply(Two_inv, centre)
        dev.append((k, math.dist(p_mapped, diag[k]["p"])))
    if dev:
        d = sorted(x[1] for x in dev)
        print(f"online->export consistency over {len(d)} frames: "
              f"median={d[len(d)//2]:.6f} max={d[-1]:.6f} m")

    # 2) step series. The verdict must compare LIKE WITH LIKE: the exported file
    # contains only the frames whose reference keyframe belongs to the exported
    # map, so the online series is evaluated BOTH over every fed frame and over
    # exactly the frames the export contains.
    def steps(posmap, allowed=None):
        ks = sorted(posmap)
        out = {}
        for k0, k1 in zip(ks, ks[1:]):
            if k1 - k0 != 1:
                continue
            if allowed is not None and (k0 not in allowed or k1 not in allowed):
                continue
            out[k1] = math.dist(posmap[k0], posmap[k1])
        return out

    online_pos = {k: inv(v["Tcw"])[1] for k, v in online.items() if v["is_set"]}
    export_pos = {k: v["p"] for k, v in diag.items()}
    exp_frames = set(export_pos)

    def flag(s):
        if not s:
            return [], 0.0
        med = sorted(s.values())[len(s) // 2]
        thr = max(20.0 * med, 0.05)
        return [k for k, v in s.items() if v > thr], med

    s_on_all = steps(online_pos)
    s_on_cmp = steps(online_pos, exp_frames)
    s_ex = steps(export_pos)
    f_on_all, med_on_all = flag(s_on_all)
    f_on, med_on = flag(s_on_cmp)
    f_ex, med_ex = flag(s_ex)
    print(f"adjacent-frame steps: online(all) n={len(s_on_all)} median={med_on_all:.4f} "
          f"flagged={len(f_on_all)}")
    print(f"                      online(on the exported frames) n={len(s_on_cmp)} "
          f"median={med_on:.4f} flagged={len(f_on)}")
    print(f"                      export n={len(s_ex)} median={med_ex:.4f} flagged={len(f_ex)}")

    # 3) where the export pose deviates from the online pose, worst first
    dev.sort(key=lambda x: -x[1])
    print("largest online->export deviations (frame, deviation_m, refKF0, refKF, "
          "liveKF, hops):")
    for k, d in dev[:5]:
        e = diag[k]
        o = online.get(k, {})
        print(f"  frame={k} dev={d:.4f} refKF0={e.get('ref','-')} refKF={o.get('ref','-')}"
              f" liveKF={e.get('live','-')} hops={e.get('hops','-')}")

    if not f_on and not f_ex:
        print("VERDICT: on the exported frames, neither the online pose series nor the "
              "exported pose series has an adjacent-frame discontinuity")
    elif f_on and f_ex:
        print("VERDICT: the discontinuity is ALREADY IN THE ONLINE POSE and survives the "
              "export -> the estimate, not the export; the export only re-expresses it")
    elif f_on and not f_ex:
        print("VERDICT: the discontinuity is IN THE ONLINE POSE and the export does NOT "
              "contain it -> the post-hoc reconstruction changed those poses "
              "(check the deviation list above for those frames)")
    elif f_ex and not f_on:
        print("VERDICT: the discontinuity appears ONLY in the EXPORTED pose -> introduced "
              "by the post-hoc reconstruction (reference keyframe / culling walk)")
    if f_on_all and not f_on:
        print(f"note: {len(f_on_all)} online jump(s) fall on frames the export excludes, so "
              f"they cannot appear in the file")

    # 3) windows around the first flagged frame, online and export side by side
    firsts = sorted(set(f_on) | set(f_ex))
    if firsts:
        c = firsts[0]
        lo, hi = c - a.window, c + a.window
        print(f"\nwindow around frame {c} (frame, online_step, export_step, "
              f"refKF, liveKF, hops, online_entries):")
        for k in sorted(set(s_on_cmp) | set(s_ex) | set(online) | set(diag)):
            if not (lo <= k <= hi):
                continue
            on = f"{s_on_all[k]:.4f}" if k in s_on_all else "     -"
            ex = f"{s_ex[k]:.4f}" if k in s_ex else "     -"
            o = online.get(k, {})
            e = diag.get(k, {})
            print(f"  {k:>6} on={on:>9} ex={ex:>9} refKF={o.get('ref','-')}/{e.get('ref','-')}"
                  f" liveKF={e.get('live','-')} hops={e.get('hops','-')}"
                  f" n_entries={o.get('n_entries','-')} state={o.get('state','-')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
