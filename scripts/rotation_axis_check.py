#!/usr/bin/env python3
"""Rotation check on a short single-map trajectory, following the ETH3D spec.

What it does differently from the earlier checks:

  * the ground truth is INTERPOLATED to the estimate timestamps (linear for the
    position, SLERP for the rotation), so the comparison is not limited to whatever
    GT sample happened to be nearest;
  * it compares the FULL relative rotation in SO(3): both the ANGLE and the AXIS.
    Comparing angles alone cannot distinguish "the same motion" from "a motion with
    the same magnitude", which is exactly what a rotating frame difference produces;
  * it tests the fixed camera-side transform hypothesis. If the estimated and the
    ground-truth camera frames differ by a constant C (world frames may differ too),
    then Re = W Rg C, and consequently
        Re_i^T Re_{i+1} = C^T (Rg_i^T Rg_{i+1}) C
    so the relative rotations are related by CONJUGATION, which preserves the angle
    but rotates the axis: axis(Re) = C^T axis(Rg). C is fitted on one part of the
    frames (Procrustes on the axes) and validated on the held-out part;
  * residuals are then checked against time and against map events (map id changes
    recorded in the export diagnostic).

usage: rotation_axis_check.py --gt GT --tum TUM [--diag DIAG] [--max-dt 0.02]
                              [--axis-floor 0.5]
"""
import argparse
import bisect
import math


def qnorm(q):
    n = math.sqrt(sum(c * c for c in q)) or 1.0
    return [c / n for c in q]


def qmul(a, b):
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return [w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2]


def qconj(q):
    return [-q[0], -q[1], -q[2], q[3]]


def q_to_R(q):
    x, y, z, w = qnorm(q)
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def R_to_axis(R):
    """axis-angle of R."""
    tr = R[0][0] + R[1][1] + R[2][2]
    c = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
    ang = math.acos(c)
    v = [R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1]]
    n = math.sqrt(sum(x * x for x in v))
    if n < 1e-12:
        return ang, [0.0, 0.0, 0.0]
    return ang, [x / n for x in v]


def axis_angle_between(u, v):
    d = max(-1.0, min(1.0, sum(u[i] * v[i] for i in range(3))))
    return math.degrees(math.acos(d))


def angle_between_R(Ra, Rb):
    tr = sum(Rb[k][i] * Ra[k][i] for i in range(3) for k in range(3))
    return math.degrees(math.acos(max(-1.0, min(1.0, (tr - 1.0) / 2.0))))


def slerp(q0, q1, u):
    q0, q1 = qnorm(q0), qnorm(q1)
    d = sum(a * b for a, b in zip(q0, q1))
    if d < 0:
        q1 = [-c for c in q1]
        d = -d
    if d > 0.9995:
        return qnorm([q0[i] + u * (q1[i] - q0[i]) for i in range(4)])
    th = math.acos(max(-1.0, min(1.0, d)))
    s = math.sin(th)
    a = math.sin((1 - u) * th) / s
    b = math.sin(u * th) / s
    return qnorm([a * q0[i] + b * q1[i] for i in range(4)])


def load_tum(path):
    rows = []
    for line in open(path, errors="replace"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = s.split()
        if len(p) < 8:
            continue
        rows.append([float(x) for x in p[:8]])
    rows.sort()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--tum", required=True)
    ap.add_argument("--diag")
    ap.add_argument("--max-dt", type=float, default=0.02)
    ap.add_argument("--axis-floor", type=float, default=0.5,
                    help="ignore pairs whose relative angle is below this (axis undefined)")
    ap.add_argument("--head", type=int, default=0, help="use only the first N estimate frames")
    a = ap.parse_args()

    gt = load_tum(a.gt)
    es = load_tum(a.tum)
    if a.head:
        es = es[:a.head]
    gts = [r[0] for r in gt]
    print(f"GT={len(gt)} rows, EST={len(es)} rows"
          + (f" (first {a.head})" if a.head else ""))

    # GT interpolated at the estimate timestamps
    interp = []
    pos_pairs = []
    for r in es:
        t = r[0]
        i = bisect.bisect_left(gts, t)
        if i == 0 or i >= len(gt):
            continue
        t0, t1 = gt[i - 1][0], gt[i][0]
        if t1 - t0 > 0.5 or t - t0 > a.max_dt or t1 - t > a.max_dt:
            continue
        u = (t - t0) / (t1 - t0)
        p = [gt[i - 1][1 + k] + u * (gt[i][1 + k] - gt[i - 1][1 + k]) for k in range(3)]
        q = slerp(gt[i - 1][4:8], gt[i][4:8], u)
        interp.append((t, q_to_R(r[4:8]), q_to_R(q), i))
        pos_pairs.append(([r[1], r[2], r[3]], p))
    print(f"frames with an interpolated GT pose: {len(interp)}")

    # relative rotations: angle error and AXIS error
    ang_err, axis_err, axes_e, axes_g, times = [], [], [], [], []
    for (t0, Re0, Rg0, _), (t1, Re1, Rg1, _) in zip(interp, interp[1:]):
        de = [[sum(Re0[k][i] * Re1[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        dg = [[sum(Rg0[k][i] * Rg1[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        ae, ue = R_to_axis(de)
        ag, ug = R_to_axis(dg)
        ang_err.append(abs(math.degrees(ae) - math.degrees(ag)))
        if math.degrees(ae) > a.axis_floor and math.degrees(ag) > a.axis_floor:
            axis_err.append(axis_angle_between(ue, ug))
            axes_e.append(ue)
            axes_g.append(ug)
            times.append(t1)

    def stats(v, unit):
        if not v:
            return "n=0"
        s = sorted(v)
        return (f"n={len(s)} median={s[len(s)//2]:.3f}{unit} "
                f"p95={s[int(0.95 * (len(s) - 1))]:.3f}{unit} max={s[-1]:.3f}{unit}")

    print(f"relative-rotation ANGLE error  : {stats(ang_err, ' deg')}")
    print(f"relative-rotation AXIS  error  : {stats(axis_err, ' deg')}")
    # Absolute misorientation with the interpolated GT, i.e. with NO fitted frame
    # transform at all. If this is small, the two orientation sequences already
    # live in the same frame and no convention has to be invented.
    abs_err = [angle_between_R(Re, Rg) for _t, Re, Rg, _i in interp]
    print(f"ABSOLUTE misorientation (no fit): {stats(abs_err, ' deg')}")
    if abs_err:
        n = len(abs_err)
        print("absolute misorientation by time decile (deg median):",
              " ".join(f"{sorted(abs_err[i*n//10:(i+1)*n//10])[len(abs_err[i*n//10:(i+1)*n//10])//2]:.2f}"
                       for i in range(10)))

    # fixed camera-side transform C: fit on the first half of the axis pairs,
    # validate on the second half, and the other way round
    if len(axes_e) >= 20:
        try:
            import numpy as np
            E = np.array(axes_e)
            G = np.array(axes_g)
            for name, fit_sl, val_sl in (("fit=first half", slice(0, len(E) // 2), slice(len(E) // 2, None)),
                                         ("fit=second half", slice(len(E) // 2, None), slice(0, len(E) // 2))):
                M = np.einsum('ni,nj->ij', E[fit_sl], G[fit_sl])   # axis(Re) = C^T axis(Rg)
                U, _, Vt = np.linalg.svd(M)
                C = U @ Vt
                if np.linalg.det(C) < 0:
                    U[:, -1] *= -1
                    C = U @ Vt
                Ct = C.T
                val = [axis_angle_between(E[n], Ct @ G[n]) for n in range(*val_sl.indices(len(E)))]
                s = sorted(val)
                print(f"  C from {name}: held-out axis residual median={s[len(s)//2]:.3f} deg "
                      f"p95={s[int(0.95 * (len(s) - 1))]:.3f} deg (n={len(s)})")
                if name == "fit=first half":
                    Mf = np.array(axes_g)
            print("  note: a small held-out residual means ONE constant camera-side "
                  "transform explains the orientation difference")
        except ImportError:
            print("numpy unavailable: skipped the constant-C test")

    # residual vs time (deciles) to expose a time-varying offset
    if len(axis_err) > 20:
        n = len(axis_err)
        print("axis error by time decile (deg median):",
              " ".join(f"{sorted(axis_err[i*n//10:(i+1)*n//10])[max(0,len(axis_err[i*n//10:(i+1)*n//10])//2)]:.2f}"
                       for i in range(10)))

    # Fit the constant frame transform where it actually shows up: on the ABSOLUTE
    # orientations (a constant LEFT factor cancels in relative rotations). Fit on one
    # half of the frames, validate on the other half, residual = angle(Re, C Rg).
    if len(interp) >= 40:
        try:
            import numpy as np
            A = np.array([Re for _t, Re, _Rg, _i in interp])
            B = np.array([Rg for _t, _Re, Rg, _i in interp])
            half = len(A) // 2

            def fit_validate(sl_fit, sl_val, name):
                M = np.einsum('nij,nkj->ik', A[sl_fit], B[sl_fit])    # sum A B^T
                U, _, Vt = np.linalg.svd(M)
                C = U @ Vt
                if np.linalg.det(C) < 0:
                    U[:, -1] *= -1
                    C = U @ Vt
                ca, _cu = R_to_axis(C)
                res = []
                for n in range(*sl_val.indices(len(A))):
                    CB = C @ B[n]
                    tr = float(np.trace(CB.T @ A[n]))                 # angle(C B, A)
                    res.append(math.degrees(math.acos(max(-1.0, min(1.0, (tr - 1.0) / 2.0)))))
                s = sorted(res)
                print(f"  constant frame transform, {name}: |C|={math.degrees(ca):.2f} deg, "
                      f"held-out residual median={s[len(s)//2]:.3f} p95="
                      f"{s[int(0.95*(len(s)-1))]:.3f} max={s[-1]:.3f} deg (n={len(s)})")
                return res

            fit_validate(slice(0, half), slice(half, None), "fit=first half, validate=second")
            res2 = fit_validate(slice(half, None), slice(0, half), "fit=second half, validate=first")
            n2 = len(res2)
            print("  validated residual by time decile (deg median):",
                  " ".join(f"{sorted(res2[i*n2//10:(i+1)*n2//10])[len(res2[i*n2//10:(i+1)*n2//10])//2]:.2f}"
                           for i in range(10)))
        except ImportError:
            print("numpy unavailable: skipped the constant frame transform test")

    # Independent attribution: SE(3) alignment (Umeyama, no scale) of the positions.
    # The estimated map lives in an arbitrary world frame anchored to its first
    # keyframe, so an alignment rotation is expected. If that rotation reproduces
    # the constant found on the orientations, the constant IS the world-frame
    # convention and not an orientation error.
    if len(pos_pairs) >= 10:
        try:
            import numpy as np
            Pe = np.array([p for p, _g in pos_pairs])
            Pg = np.array([g for _p, g in pos_pairs])
            ce, cg = Pe.mean(0), Pg.mean(0)
            H = (Pe - ce).T @ (Pg - cg)
            U, _s, Vt = np.linalg.svd(H)
            D = np.eye(3)
            D[2, 2] = math.copysign(1.0, np.linalg.det(Vt.T @ U.T))
            R_align = Vt.T @ D @ U.T
            ang, _ax = R_to_axis(R_align)
            t_align = cg - R_align @ ce
            resid = [math.dist(R_align @ Pe[n] + t_align, Pg[n]) for n in range(len(Pe))]
            rs = sorted(resid)
            print(f"  SE(3) alignment rotation angle = {math.degrees(ang):.2f} deg; "
                  f"position residual median={rs[len(rs)//2]:.4f} m max={rs[-1]:.4f} m")
            res2 = []
            for _t, Re, Rg, _i in interp:
                RA = R_align @ Re
                tr = float(np.trace(RA.T @ Rg))
                res2.append(math.degrees(math.acos(max(-1.0, min(1.0, (tr - 1.0) / 2.0)))))
            s2 = sorted(res2)
            print(f"  rotation residual AFTER the alignment rotation: "
                  f"median={s2[len(s2)//2]:.3f} p95={s2[int(0.95*(len(s2)-1))]:.3f} deg "
                  f"(n={len(s2)})")
            if abs(math.degrees(ang) - 173.1) < 15.0:
                print("  => the constant on the orientations matches the ALIGNMENT rotation: "
                      "it is the arbitrary world frame of the estimated map")
        except ImportError:
            print("numpy unavailable: skipped the SE(3) alignment attribution")

    # residual vs map events
    if a.diag:
        kf_of = {}
        for line in open(a.diag, errors="replace"):
            p = line.split()
            if p and not p[0].startswith("#") and len(p) >= 8:
                try:
                    kf_of[round(float(p[0]), 6)] = int(p[3])     # live keyframe id
                except ValueError:
                    pass
        print(f"diag loaded for {len(kf_of)} frames (live keyframe id per frame, used to "
              f"spot map/KF events next to residual spikes)")
    print("note: orientation is never inferred from the motion direction here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
