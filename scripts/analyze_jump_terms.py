#!/usr/bin/env python3
"""Term-by-term attribution of the exported pose increment between consecutive
frames, for a jump window. Read-only: uses only the artefacts a round already
wrote (online trace, export diag with Two and the chained anchor pose, TUM file).

Source formula (System.cc):
    Tcw_export = Tcr * (Trw * pKF->GetPose()) * Two
where ``Trw * pKF->GetPose()`` is the chained anchor pose recorded in the diag as
``liveKFpose`` (for hops == 0 it IS that keyframe's own final pose; for hops > 0 it
is the product of the per-level mTcp from the culling walk).

For every consecutive pair (k-1, k) the exported increment is split into:
    d_Tcr    = |export(Tcr_k , anchor_{k-1}) - export_{k-1}|
               the part caused by the frame's own relative pose
    d_anchor = |export(Tcr_k , anchor_k ) - export(Tcr_k , anchor_{k-1})|
               the part caused by the anchor pose (and by switching anchor)
and the table marks whether the anchor KEYFRAME changed and whether the two frames
are hops==0 or hops>0, because a hops==0 frame has no mTcp product at all and
therefore cannot be explained by a broken parent chain.

usage: analyze_jump_terms.py <run_dir> <seq_dir> <lo> <hi>
"""
import math
import sys


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


def dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def main():
    run, seq, lo, hi = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    fed = []
    for line in open(seq + "/rgb.txt", errors="replace"):
        s = line.strip()
        if s and not s.startswith("#"):
            fed.append(float(s.split()[0]))
    idx = {round(t, 6): i for i, t in enumerate(fed)}

    Two = None
    diag = {}
    for line in open(run + "/export_diag.txt", errors="replace"):
        p = line.split()
        if not p:
            continue
        if p[0] == "#" and len(p) > 2 and p[1] == "Two":
            v = [float(x) for x in p[2:9]]
            Two = (q_to_R(*v[3:7]), v[0:3])
        elif not p[0].startswith("#") and len(p) >= 15:
            k = idx.get(round(float(p[0]), 6))
            if k is not None:
                diag[k] = dict(ref0=int(p[1]), live=int(p[3]), hops=int(p[4]),
                               kfpose=(q_to_R(*[float(x) for x in p[11:15]]),
                                       [float(x) for x in p[8:11]]))

    tr = {}
    for line in open(run + "/online_trace.txt", errors="replace"):
        if line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 29:
            continue
        k = idx.get(round(float(p[0]), 6))
        if k is None:
            continue
        tr[k] = dict(ref=int(p[5]), mapid=int(p[6]),
                     Tcr=(q_to_R(*[float(x) for x in p[18:22]]), [float(x) for x in p[15:18]]))

    exp = {}
    for line in open(run + "/CameraTrajectory_TUM.txt", errors="replace"):
        p = line.split()
        if len(p) < 8:
            continue
        k = idx.get(round(float(p[0]), 6))
        if k is not None:
            exp[k] = [float(p[1]), float(p[2]), float(p[3])]

    if Two is None:
        print("no Two in the diag: cannot rebuild")
        return 1
    print(f"Two translation = {['%.4f' % v for v in Two[1]]}   (constant for the whole file)")

    def export_pos(k, anchor):
        T = mul(mul(tr[k]["Tcr"], anchor), Two)
        return inv(T)[1]

    print(f"\n{'frame':>6} {'refKF0':>6} {'hops':>4} {'liveKF':>6} {'anchorChanged':>13} "
          f"{'exportStep':>10} {'d_Tcr':>9} {'d_anchor':>9} {'online_step':>11} "
          f"{'anchor_pose_moved':>17}")
    prev = None
    for k in range(lo, hi + 1):
        if k not in diag or k not in tr or k not in exp:
            print(f"{k:>6}   not in the exported set (excluded or no pose)")
            prev = None
            continue
        d = diag[k]
        online_step = (dist(inv(mul(inv(tr[k - 1]["Tcr"]) if False else tr[k - 1]["Tcr"], tr[k - 1]["Tcr"])[0])[1],
                            inv(tr[k]["Tcr"])[1]) if False else float("nan"))
        if prev is not None and (k - 1) in diag and (k - 1) in exp:
            p = diag[k - 1]
            step = dist(exp[k], exp[k - 1])
            A = export_pos(k, p["kfpose"])
            d_tcr = dist(A, exp[k - 1])
            d_anchor = dist(exp[k], A)
            ch = "yes" if d["live"] != p["live"] else "no"
            moved = dist(d["kfpose"][1], p["kfpose"][1])
        else:
            step = d_tcr = d_anchor = moved = float("nan")
            ch = "-"
        print(f"{k:>6} {d['ref0']:>6} {d['hops']:>4} {d['live']:>6} {ch:>13} "
              f"{step:>10.4f} {d_tcr:>9.4f} {d_anchor:>9.4f} {'-':>11} {moved:>17.4f}")
        prev = k

    print("\nanchors used in the window (keyframe id -> chained final position, hops in which it was used):")
    seen = {}
    for k in range(lo, hi + 1):
        if k in diag:
            d = diag[k]
            seen.setdefault(d["live"], []).append((k, d["hops"], d["kfpose"][1], tr.get(k, {}).get("mapid"),
                                                   d["ref0"]))
    for live, rows in sorted(seen.items()):
        hops = sorted({r[1] for r in rows})
        maps = sorted({r[3] for r in rows})
        print(f"  liveKF={live}  frames={[r[0] for r in rows]}")
        print(f"      hops={hops}  trace_mapid={maps}  chained_final_pos={['%.4f' % v for v in rows[0][2]]}"
              f"  original_refKF={sorted({r[4] for r in rows})}")
    ks = sorted(seen)
    for a, b in zip(ks, ks[1:]):
        print(f"  anchor pair {a} <-> {b}: chained final positions differ by "
              f"{dist(seen[a][0][2], seen[b][0][2]):.4f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
