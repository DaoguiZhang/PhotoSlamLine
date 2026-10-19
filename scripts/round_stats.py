#!/usr/bin/env python3
"""Per-round quantities required in addition to the 15-check export gate.

Reported for every round:
  input frames                     frames the sequence can feed (rgb.txt)
  valid online pose frames         trace rows with a pose (is_set == 1)
  largest-map-fragment frames      lines in the exported TUM file
  maps                             maps present in the atlas census
  excluded frames                  skipped_lost + skipped_other_map + skipped_no_refkf
  max adjacent valid online step   max translation between ADJACENT input frames
                                   that both have a valid online pose in the same map
  max online -> final export pose  max |exported position - (online pose mapped by Two)|
  max reference-KF pose change     max distance between the reference keyframe pose
                                   implied at tracking time (Tcr^-1 * Tcw) and the
                                   chained FINAL keyframe pose the exporter used
                                   (only frames with no culling hop)

A large jump also prints a per-frame window. Online anomalies are NEVER dropped
just because the final export is smooth.

usage: round_stats.py --trace F --diag F --tum F --seq DIR --log F [--out F] [--window 4]
"""
import argparse
import bisect
import json
import math
import os
import re

STATE_VALID = (2, 3)


def q_to_R(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def mul(A, B):
    Ra, ta = A
    Rb, tb = B
    R = [[sum(Ra[i][k] * Rb[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    t = [Ra[i][0] * tb[0] + Ra[i][1] * tb[1] + Ra[i][2] * tb[2] + ta[i] for i in range(3)]
    return (R, t)


def inv(T):
    R, t = T
    Rt = [[R[j][i] for j in range(3)] for i in range(3)]
    return (Rt, [-sum(Rt[i][j] * t[j] for j in range(3)) for i in range(3)])


def apply(T, p):
    R, t = T
    return [R[i][0] * p[0] + R[i][1] * p[1] + R[i][2] * p[2] + t[i] for i in range(3)]


def dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


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
    ap.add_argument("--log", required=True)
    ap.add_argument("--out")
    ap.add_argument("--window", type=int, default=4)
    a = ap.parse_args()

    fed = []
    for line in open(os.path.join(a.seq, "rgb.txt"), errors="replace"):
        s = line.strip()
        if s and not s.startswith("#"):
            fed.append(float(s.split()[0]))
    idx = Assoc(fed)

    Two, diag = None, {}
    for line in open(a.diag, errors="replace"):
        p = line.split()
        if not p:
            continue
        if p[0] == "#" and len(p) > 2 and p[1] == "Two":
            v = [float(x) for x in p[2:9]]
            Two = (q_to_R(*v[3:7]), v[0:3])
        elif not p[0].startswith("#") and len(p) >= 15:
            k = idx(float(p[0]))
            if k is None:
                continue
            diag[k] = dict(ref0=int(p[1]), hops=int(p[4]),
                           exp_xyz=[float(p[5]), float(p[6]), float(p[7])],
                           TrwKF=(q_to_R(*[float(x) for x in p[11:15]]),
                                  [float(x) for x in p[8:11]]))

    online = {}
    for line in open(a.trace, errors="replace"):
        if line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 29:
            continue
        k = idx(float(p[0]))
        if k is None:
            continue
        online[k] = dict(state=int(p[2]), is_set=int(p[3]), ref=int(p[5]), refmap=int(p[6]),
                         Tcw=(q_to_R(*[float(x) for x in p[11:15]]), [float(x) for x in p[8:11]]),
                         Tcr=(q_to_R(*[float(x) for x in p[18:22]]), [float(x) for x in p[15:18]]))

    tum = {}
    for line in open(a.tum, errors="replace"):
        p = line.split()
        if len(p) < 8:
            continue
        k = idx(float(p[0]))
        if k is not None:
            tum[k] = [float(x) for x in p[1:4]]

    log = open(a.log, errors="replace").read()
    m = re.search(r"atlas holds (\d+) map\(s\)", log)
    maps = int(m.group(1)) if m else None
    m = re.search(r"written=(\d+) skipped_lost=(\d+) skipped_other_map=(\d+)"
                  r" skipped_no_refkf=(\d+) \(of (\d+) frame entries\)", log)
    if m:
        written, lost, other, noref, entries = (int(g) for g in m.groups())
    else:
        written = len(tum)
        lost = other = noref = entries = None

    # adjacent valid online steps (same map, valid state)
    valid = {k: v for k, v in online.items() if v["is_set"] and v["state"] in STATE_VALID}
    steps = []
    for k0, k1 in zip(sorted(valid), sorted(valid)[1:]):
        if k1 - k0 != 1 or valid[k0]["refmap"] != valid[k1]["refmap"]:
            continue
        steps.append((k1, dist(inv(valid[k0]["Tcw"])[1], inv(valid[k1]["Tcw"])[1])))
    # NOTE: (frame, step) pairs -> the FIRST element is the frame index.
    arg_step, max_step = (max(steps, key=lambda x: x[1]) if steps else (None, 0.0))

    # online pose -> final export pose, and reference-KF pose change
    Two_inv = inv(Two) if Two else None
    dev, kfdiff = [], []
    for k in sorted(set(online) & set(diag)):
        o, e = online[k], diag[k]
        if o["is_set"]:
            dev.append((k, dist(apply(Two_inv, inv(o["Tcw"])[1]), e["exp_xyz"])))
        if o["is_set"] and e["hops"] == 0:
            used_then = mul(inv(o["Tcr"]), o["Tcw"])
            kfdiff.append((k, dist(inv(used_then)[1], inv(e["TrwKF"])[1])))
    max_dev, arg_dev = (max(dev, key=lambda x: x[1]) if dev else (None, 0.0))
    max_kf, arg_kf = (max(kfdiff, key=lambda x: x[1]) if kfdiff else (None, 0.0))
    arg_dev, max_dev = (arg_dev, max_dev) if isinstance(arg_dev, int) else (None, max_dev)
    arg_kf, max_kf = (arg_kf, max_kf) if isinstance(arg_kf, int) else (None, max_kf)
    if dev:
        arg_dev, max_dev = max(dev, key=lambda x: x[1])
    if kfdiff:
        arg_kf, max_kf = max(kfdiff, key=lambda x: x[1])

    res = dict(input_frames=len(fed), online_pose_frames=len(valid),
               exported_frames=written, maps=maps,
               excluded_frames=((lost or 0) + (other or 0) + (noref or 0)) if m else None,
               skipped_lost=lost, skipped_other_map=other, skipped_no_refkf=noref,
               frame_entries=entries,
               max_adjacent_online_step=max_step, max_adjacent_online_step_at=arg_step,
               max_online_to_export=round(max_dev, 6), max_online_to_export_at=arg_dev,
               max_refKF_pose_change=round(max_kf, 6), max_refKF_pose_change_at=arg_kf)
    print(json.dumps(res, sort_keys=True))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)

    if arg_step is not None:
        lo, hi = int(arg_step) - a.window, int(arg_step) + a.window
        print(f"window around the largest adjacent online step (frame {arg_step}, "
              f"step {max_step:.4f} m):")
        for k in range(lo, hi + 1):
            if k not in online:
                continue
            o = online.get(k, {})
            e = diag.get(k, {})
            st = next((s for kk, s in steps if kk == k), None)
            print(f"  frame={k} state={o.get('state','-')} is_set={o.get('is_set','-')} "
                  f"refKF={o.get('ref','-')} map={o.get('refmap','-')} "
                  f"liveKF={e.get('ref0','-')} hops={e.get('hops','-')} "
                  f"online_step={'%.4f' % st if st is not None else '-'} "
                  f"online2export={next((('%.4f' % d) for kk, d in dev if kk == k), '-')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
