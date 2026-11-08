#!/usr/bin/env python3
"""Rotation check that follows the ETH3D SLAM trajectory specification only.

Specification used (ETH3D SLAM benchmark, TUM-style file):
    # timestamp tx ty tz qx qy qz qw
    - one sample per line, 8 columns, quaternion in the order x, y, z, w
    - the pose is T_wc: it maps CAMERA coordinates to WORLD, i.e. it is the pose
      of the camera in the world frame (translation = camera centre in world)
    - 100 Hz ground truth; image/estimate timestamps must be associated in time

What this script verifies, per frame, without ever inferring orientation from the
motion direction:
  1  column count / parseability of the GT and of the estimate
  2  quaternion unit norm, and that the rotation matrix built from it is
     orthonormal with determinant +1
  3  cross-format consistency: the quaternion in TUM and the rotation matrix in
     the KITTI file of the SAME frame must describe the same rotation
  4  time association: for every estimate frame, the GT sample actually used and
     the residual dt, including whether the GT sample is before or after it
  5  a time-offset sweep: shift the GT association by delta and report the median
     relative-rotation residual. Relative rotations cancel any constant frame
     offset, so a clear minimum at delta != 0 is evidence of a TIME association
     error rather than of a frame convention.

usage: rotation_spec_check.py --gt GT --tum TUM [--kitti KITTI] [--max-dt 0.02]
"""
import argparse
import bisect
import math


def q_to_R(q):
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0:
        raise ValueError("zero quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def orthonormality(R):
    err = 0.0
    for i in range(3):
        for j in range(3):
            dot = sum(R[k][i] * R[k][j] for k in range(3))
            err = max(err, abs(dot - (1.0 if i == j else 0.0)))
    det = (R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1])
           - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0])
           + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]))
    return err, det


def rot_angle(Ra, Rb):
    tr = sum(Rb[k][i] * Ra[k][i] for i in range(3) for k in range(3))
    return math.degrees(math.acos(max(-1.0, min(1.0, (tr - 1.0) / 2.0))))


def load_rows(path, ncol):
    rows = []
    bad = 0
    for line in open(path, errors="replace"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = s.split()
        if len(p) < ncol:
            bad += 1
            continue
        try:
            rows.append([float(x) for x in p[:ncol]])
        except ValueError:
            bad += 1
    return rows, bad


def load_kitti(path):
    out = []
    for line in open(path, errors="replace"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = s.split()
        if len(p) < 12:
            continue
        v = [float(x) for x in p]
        out.append([[v[0], v[1], v[2]], [v[4], v[5], v[6]], [v[8], v[9], v[10]]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--tum", required=True)
    ap.add_argument("--kitti")
    ap.add_argument("--max-dt", type=float, default=0.02)
    ap.add_argument("--per-frame", type=int, default=0,
                    help="print a per-frame check table for the first N estimate frames")
    a = ap.parse_args()

    gt_rows, gt_bad = load_rows(a.gt, 8)
    es_rows, es_bad = load_rows(a.tum, 8)
    gt_rows.sort()
    es_rows.sort()
    gt_ts = [r[0] for r in gt_rows]
    print(f"GT: {len(gt_rows)} rows ({gt_bad} unparsable) | EST: {len(es_rows)} rows "
          f"({es_bad} unparsable)")

    # 1/2 quaternion + matrix validity
    def validity(rows, name):
        norms, onorm, dets = [], [], []
        for r in rows:
            q = r[4:8]
            norms.append(abs(math.sqrt(sum(c * c for c in q)) - 1.0))
            R = q_to_R(q)
            e, d = orthonormality(R)
            onorm.append(e)
            dets.append(d)
        print(f"  {name}: max|q|-1 = {max(norms):.3e}  max orthonormality error = "
              f"{max(onorm):.3e}  det range = [{min(dets):.6f}, {max(dets):.6f}]")
        return norms, onorm, dets

    print("spec check (quaternion order qx qy qz qw, pose T_wc):")
    validity(gt_rows, "GT ")
    validity(es_rows, "EST")

    # 6 per-frame table: quaternion validity, matrix validity, GT used, dt, and
    #    (when available) the deviation from the KITTI matrix of the same frame.
    if a.per_frame > 0 and a.kitti:
        ki = load_kitti(a.kitti)
        print(f"\nper-frame check (first {a.per_frame} estimate frames):")
        print(f"  {'ts':>14} {'|q|-1':>10} {'orth':>10} {'det':>9} {'gt_ts':>14} "
              f"{'dt':>9} {'vs_KITTI_deg':>12}")
        for n, r in enumerate(es_rows[:a.per_frame]):
            t = r[0]
            i = bisect.bisect_left(gt_ts, t)
            cand = [j for j in (i - 1, i, i + 1) if 0 <= j < len(gt_ts)]
            j = min(cand, key=lambda k: abs(gt_ts[k] - t)) if cand else None
            gt_ts_used = gt_ts[j] if j is not None else float('nan')
            R = q_to_R(r[4:8])
            e, d = orthonormality(R)
            qn = abs(math.sqrt(sum(c * c for c in r[4:8])) - 1.0)
            dev = rot_angle(R, ki[n]) if n < len(ki) else float('nan')
            print(f"  {t:>14.6f} {qn:>10.2e} {e:>10.2e} {d:>9.6f} {gt_ts_used:>14.6f} "
                  f"{gt_ts_used - t:>+9.6f} {dev:>12.4f}")

    # 3 cross-format agreement on the SAME frame (quaternion vs stored matrix)
    if a.kitti:
        ki = load_kitti(a.kitti)
        if len(ki) == len(es_rows):
            worst = 0.0
            for r, Rk in zip(es_rows, ki):
                worst = max(worst, rot_angle(q_to_R(r[4:8]), Rk))
            print(f"  TUM quaternion vs KITTI matrix, same frame index: max deviation "
                  f"{worst:.4f} deg")
        else:
            print(f"  KITTI line count {len(ki)} != TUM {len(es_rows)}: not comparable")

    # 4 time association: nearest GT sample and its residual
    dts, before, after, unmatched = [], 0, 0, 0
    for r in es_rows:
        t = r[0]
        i = bisect.bisect_left(gt_ts, t)
        cand = [j for j in (i - 1, i, i + 1) if 0 <= j < len(gt_ts)]
        if not cand:
            unmatched += 1
            continue
        j = min(cand, key=lambda k: abs(gt_ts[k] - t))
        d = gt_ts[j] - t
        if abs(d) > a.max_dt:
            unmatched += 1
            continue
        dts.append(d)
        before += 1 if d < 0 else 0
        after += 1 if d > 0 else 0
    if dts:
        s = sorted(dts)
        print(f"  association: matched={len(dts)} unmatched={unmatched} "
              f"median dt={s[len(s)//2]:+.6f}s  dt range=[{s[0]:+.6f},{s[-1]:+.6f}] "
              f"GT_before={before} GT_after={after}")
        print("  (a systematic shift of median dt away from 0 is a time association offset)")

    # 5 time-offset sweep on RELATIVE rotations (cancels any constant frame offset)
    def residual(shift):
        vals = []
        prev = None
        for r in es_rows:
            t = r[0] + shift
            i = bisect.bisect_left(gt_ts, t)
            cand = [j for j in (i - 1, i, i + 1) if 0 <= j < len(gt_ts)]
            if not cand:
                prev = None
                continue
            j = min(cand, key=lambda k: abs(gt_ts[k] - t))
            if abs(gt_ts[j] - t) > a.max_dt:
                prev = None
                continue
            cur = (q_to_R(r[4:8]), q_to_R(gt_rows[j][4:8]), j)
            # Pair each accepted estimate frame with the PREVIOUS accepted estimate
            # frame. The ground-truth samples they map to are k apart (100 Hz GT vs
            # 30 Hz frames), which is fine: Re(i)^T Re(i+1) and Rg(j)^T Rg(j+k)
            # describe the same time interval. Requiring adjacent GT indices would
            # throw away almost every pair.
            if prev is not None:
                re = [[sum(prev[0][k][i] * cur[0][k][j] for k in range(3)) for j in range(3)]
                      for i in range(3)]
                rg = [[sum(prev[1][k][i] * cur[1][k][j] for k in range(3)) for j in range(3)]
                      for i in range(3)]
                vals.append(rot_angle(re, rg))
            prev = cur
        if not vals:
            return None
        s = sorted(vals)
        return s[len(s) // 2], len(vals)

    r0 = residual(0.0)
    if r0:
        print(f"  relative-rotation residual at delta=0: median={r0[0]:.3f} deg over {r0[1]} pairs")
        best = None
        for k in range(-25, 26):
            d = k * 0.02
            rr = residual(d)
            if rr and (best is None or rr[0] < best[1]):
                best = (d, rr[0])
        print(f"  best delta over [-0.5,+0.5]s: delta={best[0]:+.2f}s "
              f"median residual={best[1]:.3f} deg")
        if abs(best[0]) > 0.02 and best[1] < 0.5 * r0[0]:
            print("  VERDICT: a time offset improves the agreement a lot -> investigate "
                  "the TIME ASSOCIATION before any frame convention")
        else:
            print("  VERDICT: no time offset is needed. CAVEAT: this statistic compares")
            print("  rotation MAGNITUDES only, and a slowly varying frame difference also")
            print("  preserves them, so it neither confirms nor refutes a frame convention.")
            print("  Section 6 (constant correction) and an AXIS comparison are what")
            print("  discriminate a constant offset from a time-varying one.")
    # 6 constant frame correction on the TIME-ASSOCIATED pairs.
    # If est and GT describe the same rotational motion (which section 5 tests
    # without assuming either side), then one constant rotation C must satisfy
    # Re ~ C Rg. This tests that directly with orthogonal Procrustes instead of
    # searching a table of hand-written conventions.
    A, B = [], []
    for r in es_rows:
        t = r[0]
        i = bisect.bisect_left(gt_ts, t)
        cand = [j for j in (i - 1, i, i + 1) if 0 <= j < len(gt_ts)]
        if not cand:
            continue
        j = min(cand, key=lambda k: abs(gt_ts[k] - t))
        if abs(gt_ts[j] - t) > a.max_dt:
            continue
        A.append(q_to_R(r[4:8]))
        B.append(q_to_R(gt_rows[j][4:8]))
    if len(A) > 10:
        try:
            import numpy as np
            A = np.array(A)
            B = np.array(B)
            M = np.einsum('nij,nkj->ik', A, B)     # sum_n A_n B_n^T
            U, _, Vt = np.linalg.svd(M)
            C = U @ Vt
            if np.linalg.det(C) < 0:
                U[:, -1] *= -1
                C = U @ Vt
            res = []
            for n in range(len(A)):
                tr = float(np.trace(C @ B[n].T @ A[n]))
                res.append(math.degrees(math.acos(max(-1.0, min(1.0, (tr - 1.0) / 2.0)))))
            s = sorted(res)
            print(f"constant frame correction over {len(A)} time-associated pairs: "
                  f"median residual={s[len(s)//2]:.3f} deg  p95={s[int(0.95*(len(s)-1))]:.3f}  "
                  f"max={s[-1]:.3f}")
            print("  (small residual => the two orientation sequences differ by ONE "
                  "constant rotation, i.e. a frame convention, not by per-frame error)")
        except ImportError:
            print("numpy unavailable: skipped the constant-frame-correction test")

    print("note: orientation is never inferred from the motion direction here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
