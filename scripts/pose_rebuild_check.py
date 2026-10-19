#!/usr/bin/env python3
"""Rebuild the exported pose from the SOURCE FORMULA and compare it frame by frame
with the exported file, restricted to jump windows.

Window selection follows the requirement exactly: same run, ADJACENT INPUT FRAMES,
online tracking state valid, and both frames belonging to the SAME map.

Per frame it checks, with no fit and no tolerance fudging:

  1  online pose           Tcw recorded by the tracker
  2  Tcr                   the object that entered mlRelativeFramePoses
  3  reference KF pose     as read at trace time (may already differ from the pose
                           used when Tcr was computed, which is reported as drift)
  4  live KF pose          the chained keyframe pose the exporter used
  5  Two                   the export origin
  6  rebuilt pose          Tcw_rebuilt = Tcr * TrwKF * Two   (src: System.cc)
  7  residual              rebuilt vs the exported file, translation and rotation

usage: pose_rebuild_check.py --trace F --diag F --tum F --seq DIR [--window 6]
"""
import argparse
import bisect
import math

STATE_VALID = (2, 3)          # 2 = OK, 3 = RECENTLY_LOST


def q_to_R(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def mul(A, B):
    """(R,t) composition A * B."""
    Ra, ta = A
    Rb, tb = B
    R = [[sum(Ra[i][k] * Rb[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    t = [Ra[i][0] * tb[0] + Ra[i][1] * tb[1] + Ra[i][2] * tb[2] + ta[i] for i in range(3)]
    return (R, t)


def inv(T):
    R, t = T
    Rt = [[R[j][i] for j in range(3)] for i in range(3)]
    return (Rt, [-sum(Rt[i][j] * t[j] for j in range(3)) for i in range(3)])


def dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def rot_angle(Ra, Rb):
    tr = sum(Rb[k][i] * Ra[k][i] for i in range(3) for k in range(3))
    return math.degrees(math.acos(max(-1.0, min(1.0, (tr - 1.0) / 2.0))))


def rgb_order(seq):
    ts = []
    for line in open(seq + "/rgb.txt", errors="replace"):
        s = line.strip()
        if s and not s.startswith("#"):
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
    ap.add_argument("--diag", required=True)
    ap.add_argument("--tum", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--window", type=int, default=6)
    a = ap.parse_args()
    idx = Assoc(rgb_order(a.seq))

    Two = None
    diag = {}
    for line in open(a.diag, errors="replace"):
        p = line.split()
        if not p:
            continue
        if p[0] == "#" and len(p) > 2 and p[1] == "Two":
            v = [float(x) for x in p[2:9]]
            Two = (q_to_R(*v[3:7]), v[0:3])
        elif not p[0].startswith("#"):
            if len(p) < 15:
                continue
            k = idx(float(p[0]))
            if k is None:
                continue
            diag[k] = dict(ref0=int(p[1]), refmap=p[2], live=int(p[3]), hops=int(p[4]),
                           exp_xyz=[float(p[5]), float(p[6]), float(p[7])],
                           exp_T=(q_to_R(*[float(x) for x in p[11:15]]),
                                  [float(x) for x in p[8:11]]),
                           TrwKF=(q_to_R(*[float(x) for x in p[11:15]]),
                                  [float(x) for x in p[8:11]]))

    online = {}
    n_old = 0
    for line in open(a.trace, errors="replace"):
        if line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 29:            # 8 ids + Tcw7 + Tcr7 + refKFpose7
            n_old += 1
            continue
        k = idx(float(p[0]))
        if k is None:
            continue
        online[k] = dict(ts=float(p[0]), state=int(p[2]), is_set=int(p[3]), ref=int(p[5]),
                         refmap=int(p[6]),
                         Tcw=(q_to_R(*[float(x) for x in p[11:15]]), [float(x) for x in p[8:11]]),
                         Tcr=(q_to_R(*[float(x) for x in p[18:22]]), [float(x) for x in p[15:18]]),
                         refnow=(q_to_R(*[float(x) for x in p[25:29]]), [float(x) for x in p[22:25]]))

    tum = {}
    for line in open(a.tum, errors="replace"):
        p = line.split()
        if len(p) < 8:
            continue
        k = idx(float(p[0]))
        if k is not None:
            tum[k] = (q_to_R(*[float(x) for x in p[4:8]]), [float(x) for x in p[1:4]])

    print(f"frames: trace={len(online)} diag={len(diag)} tum={len(tum)}"
          + (f"  [{n_old} trace rows are from the OLD 15-field format and were skipped]" if n_old else ""))
    if Two is None:
        print("no origin Two in the diag file")
        return 1

    # --- eligible windows: adjacent input frames, valid state, same map --------
    ks = sorted(online)
    steps, med_src = {}, []
    pairs = []
    for k0, k1 in zip(ks, ks[1:]):
        if k1 - k0 != 1:
            continue
        o0, o1 = online[k0], online[k1]
        if o0["state"] not in STATE_VALID or o1["state"] not in STATE_VALID:
            continue
        if not (o0["is_set"] and o1["is_set"]):
            continue
        if o0["refmap"] != o1["refmap"]:
            continue
        p0 = inv(o0["Tcw"])[1]
        p1 = inv(o1["Tcw"])[1]
        d = dist(p0, p1)
        steps[k1] = d
        med_src.append(d)
        pairs.append((k0, k1, d))
    med = sorted(med_src)[len(med_src) // 2] if med_src else 0.0
    thr = max(20.0 * med, 0.05)
    flagged = [(k0, k1, d) for k0, k1, d in pairs if d > thr]
    print(f"eligible adjacent pairs (same map, valid state): {len(pairs)}  "
          f"median step={med:.4f}  flagged={len(flagged)}")

    # --- per-frame rebuild residual over ALL eligible frames -------------------
    res_t, res_a, ids = [], [], []
    for k in ks:
        if k not in diag or k not in online:
            continue
        o, e = online[k], diag[k]
        Tcw_rebuilt = mul(mul(o["Tcr"], e["TrwKF"]), Two)
        Twc_rebuilt = inv(Tcw_rebuilt)
        if k in tum:
            res_t.append(dist(Twc_rebuilt[1], tum[k][1]))
            res_a.append(rot_angle(Twc_rebuilt[0], tum[k][0]))
            ids.append(k)
    if res_t:
        st, sa = sorted(res_t), sorted(res_a)
        print(f"REBUILD vs exported file over {len(res_t)} frames: "
              f"translation median={st[len(st)//2]:.3e} max={st[-1]:.3e} m | "
              f"rotation median={sa[len(sa)//2]:.3e} max={sa[-1]:.3e} deg")

    # --- windows -----------------------------------------------------------------
    for k0, k1, d in flagged[:3]:
        lo, hi = k1 - a.window, k1 + a.window
        print(f"\nwindow around the flagged adjacent pair {k0}->{k1} (online step {d:.4f} m):")
        print(f"  {'frame':>6} {'state':>5} {'refKF':>6} {'map':>4} {'live':>6} {'hops':>4} "
              f"{'online_step':>11} {'rebuild_res_t':>13} {'rebuild_res_a':>13} "
              f"{'online_vs_export':>16} {'KFdrift_t':>10}")
        for k in range(lo, hi + 1):
            if k not in online:
                continue
            o = online[k]
            e = diag.get(k)
            st = f"{steps[k]:.4f}" if k in steps else "     -"
            r_t = r_a = dev = drift = float("nan")
            if e:
                Tcw_rebuilt = mul(mul(o["Tcr"], e["TrwKF"]), Two)
                Twc_reb = inv(Tcw_rebuilt)
                if k in tum:
                    r_t = dist(Twc_reb[1], tum[k][1])
                    r_a = rot_angle(Twc_reb[0], tum[k][0])
                dev = dist(inv(o["Tcw"])[1], e["exp_xyz"])
                # keyframe pose drift since tracking: Tcr^-1 * Tcw_online vs refnow
                T_used_then = mul(inv(o["Tcr"]), o["Tcw"])
                drift = dist(inv(T_used_then)[1], inv(o["refnow"])[1])
            print(f"  {k:>6} {o['state']:>5} {o['ref']:>6} {o['refmap']:>4} "
                  f"{e['live'] if e else -1:>6} {e['hops'] if e else -1:>4} {st:>11} "
                  f"{r_t:>13.3e} {r_a:>13.3e} {dev:>16.4f} {drift:>10.4f}")

    if not flagged:
        print("\nVERDICT: no eligible jump window in this run -> label it 'not reproduced'.")
    else:
        print("\nVERDICT: the exported pose is reproduced by the source formula to the "
              "residuals above at the flagged frames; the online-vs-export column shows "
              "whether the change happened in the estimate or in the reconstruction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
