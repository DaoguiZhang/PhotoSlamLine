#!/usr/bin/env python3
"""Cross-validate the ~170 deg orientation offset between the estimated and GT
poses, without assuming which is wrong.

Everything is derived from the raw TUM records, so each hypothesis is a concrete
algebraic statement rather than a guess:

  * the raw quaternions are converted with the declared (qx,qy,qz,qw) order, and
    the alternative orderings are tested explicitly;
  * T_wc vs T_cw is tested by inverting one side (R -> R^T) in all four
    combinations;
  * coordinate-frame conventions are tested as constant rotations C applied on
    either side, for C in {I, 180 deg about x / y / z, +-90 deg about x, +-90 deg
    about z} (optical vs body axes);
  * the alignment itself is used as evidence: the rotation R_align that the
    positional Umeyama fit implies is reported, then applied to the orientations
    as well (R_align R_e and R_e R_align), because a world-frame difference must
    show up in the orientations too;
  * the residual is characterised by median AND spread: a genuine constant
    convention offset corrects to a small, stable angle, whereas a wrong
    hypothesis leaves either a large median or a large spread.

Usage: rot_convention_check.py <gt_tum> <est_tum> [<est_tum2> ...]
"""
import bisect
import math
import sys

import numpy as np


def q2R(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rot_angle(R):
    c = min(1.0, max(-1.0, (np.trace(R) - 1.0) / 2.0))
    return math.degrees(math.acos(c))


def load_tum(path):
    rows = []
    for line in open(path, errors="replace"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = s.split()
        if len(p) < 8:
            continue
        try:
            rows.append([float(v) for v in p[:8]])
        except ValueError:
            continue
    rows.sort(key=lambda r: r[0])
    return rows


def umeyama(src, dst):
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    S = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, _, Vt = np.linalg.svd(S)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    return R, mu_d - R @ mu_s


def Rx(a):
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)],
                     [0, math.sin(a), math.cos(a)]])


def Rz(a):
    return np.array([[math.cos(a), -math.sin(a), 0],
                     [math.sin(a), math.cos(a), 0], [0, 0, 1]])


CANDS = {
    "I": np.eye(3),
    "Rx(180)": Rx(math.pi),
    "Ry(180)": np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
    "Rz(180)": Rz(math.pi),
    "Rx(+90)": Rx(math.pi / 2),
    "Rx(-90)": Rx(-math.pi / 2),
    "Rz(+90)": Rz(math.pi / 2),
    "Rz(-90)": Rz(-math.pi / 2),
}


def check(gt_path, est_path, label):
    gt = load_tum(gt_path)
    est = load_tum(est_path)
    if not gt or len(est) < 3:
        print(f"[{label}] insufficient data")
        return
    gts = [r[0] for r in gt]

    pairs = []
    for row in est:
        t = row[0]
        k = bisect.bisect_left(gts, t)
        if k <= 0 or k >= len(gt) or abs(gts[k] - t) > 0.02:
            continue
        pairs.append((row, gt[k]))
    if len(pairs) < 3:
        print(f"[{label}] fewer than 3 associated poses")
        return

    Pe = np.array([p[0][1:4] for p in pairs])
    Pg = np.array([p[1][1:4] for p in pairs])
    R_align, t_align = umeyama(Pe, Pg)
    rmse = float(np.sqrt(np.mean(np.sum((Pe @ R_align.T + t_align - Pg) ** 2, axis=1))))

    print("=" * 100)
    print(f"[{label}]  pairs={len(pairs)}  est={est_path}")
    print(f"  position: SE(3) alignment RMSE = {rmse:.4f} m  (positions clearly belong to "
          f"the same path, or this would be metres)")
    print(f"  implied alignment rotation angle = {rot_angle(R_align):.2f} deg")
    best = min(CANDS.items(), key=lambda kv: abs(rot_angle(kv[1]) - rot_angle(R_align)))
    print(f"  closest candidate convention to it: {best[0]} "
          f"({rot_angle(best[1]):.2f} deg)")
    print()

    eps = 1e-9
    Rg = [q2R(*p[1][4:8]) for p in pairs]
    # orientation orderings (qxyzw as declared, and the main alternative)
    Re_decl = [q2R(*p[0][3:7]) for p in pairs]
    Re_alt = [q2R(p[0][6], p[0][3], p[0][4], p[0][5]) for p in pairs]

    def stats(name, fn):
        vals = [rot_angle(fn(Re, Rg_)) for Re, Rg_ in zip(Re_decl, Rg)]
        vals = [v for v in vals if v == v]
        if not vals:
            return
        print(f"  {name:<52} median={np.median(vals):8.2f}  "
              f"p10={np.percentile(vals,10):7.2f}  p90={np.percentile(vals,90):7.2f}  "
              f"std={np.std(vals):7.2f}")

    print("  hypothesis screen (median relative rotation angle in degrees):")
    stats("declared quat order, as-is        Re^T Rg", lambda Re, Rg_: Re.T @ Rg_)
    stats("declared order, GT inverted        Re^T Rg^T", lambda Re, Rg_: Re.T @ Rg_.T)
    stats("declared order, EST inverted       Re   Rg", lambda Re, Rg_: Re @ Rg_)
    stats("declared order, both inverted      Re   Rg^T", lambda Re, Rg_: Re @ Rg_.T)
    for nm, C in CANDS.items():
        stats(f"axis convention  ({nm}) Re C -> compare", lambda Re, Rg_, C=C: (Re @ C).T @ Rg_)
        stats(f"axis convention  ({nm}) C Re -> compare", lambda Re, Rg_, C=C: (C @ Re).T @ Rg_)
    stats("alignment applied left  R_align Re", lambda Re, Rg_: (R_align @ Re).T @ Rg_)
    stats("alignment applied right Re R_align", lambda Re, Rg_: (Re @ R_align).T @ Rg_)
    stats("alignment left, GT inverted", lambda Re, Rg_: (R_align @ Re).T @ Rg_.T)
    print()
    print("  alternative quaternion orderings on the ESTIMATE side, as-is comparison:")
    vals = [rot_angle(a.T @ b) for a, b in zip(Re_alt, Rg)]
    print(f"  {'est read as (w,x,y,z)':<52} median={np.median(vals):8.2f}  "
          f"p10={np.percentile(vals,10):7.2f}  p90={np.percentile(vals,90):7.2f}")
    print()

    # ---------------------------------------------------------------------
    # Decisive test: is the discrepancy ONE constant pair (L, Cr) with
    # Rg ~= L * Re * Cr, i.e. a world-frame rotation composed with a camera-axis
    # convention? The screen above tests those separately; only the combination
    # can explain a system whose POSITIONS are already centimetre-accurate.
    # Solved by alternating projection onto SO(3) (SVD of a matrix sum), which is
    # the standard way to average rotations.
    #   collapse  -> the offset is a pure convention, orientation is recoverable
    #   no collapse -> the orientations genuinely disagree per frame
    # ---------------------------------------------------------------------
    def so3_mean(Rs):
        A = np.zeros((3, 3))
        for R_ in Rs:
            A += R_
        U, _, Vt = np.linalg.svd(A)
        D = np.eye(3)
        if np.linalg.det(U @ Vt) < 0:
            D[2, 2] = -1
        return U @ D @ Vt

    L = R_align.copy()
    Cr = np.eye(3)
    pairsLR = list(zip(Re_decl, Rg))
    for _ in range(40):
        Cr = so3_mean([(L @ Re).T @ Rg_ for Re, Rg_ in pairsLR])
        L = so3_mean([Rg_ @ (Re @ Cr).T for Re, Rg_ in pairsLR])
    res = [rot_angle((L @ Re @ Cr).T @ Rg_) for Re, Rg_ in pairsLR]
    print("  CONSTANT (L, Cr) SOLVE   Rg ~= L Re Cr:")
    print(f"    median={np.median(res):8.2f}  p10={np.percentile(res,10):7.2f}  "
          f"p90={np.percentile(res,90):7.2f}  std={np.std(res):8.2f} deg")
    print(f"    |L| rotation angle = {rot_angle(L):.2f} deg   "
          f"Cr angle = {rot_angle(Cr):.2f} deg")
    verdict = ("CONVENTION: a single constant pair collapses the residual"
               if np.median(res) < 5.0 and np.std(res) < 5.0
               else "NOT a single constant convention: orientations still disagree "
                    "per frame after the best constant correction")
    print(f"    -> {verdict}")
    print()

    # ---------------------------------------------------------------------
    # Presumption-free check: compare RELATIVE rotations between consecutive
    # associated frames. A constant frame offset cancels in Re1^T Re2, so this
    # compares the rotational CONTENT without assuming either side is wrong.
    # Agreement here with disagreement above means a frame convention is missing
    # from the tested set; disagreement here means the orientation content itself
    # differs, so no convention can repair it.
    # ---------------------------------------------------------------------
    rel_e, rel_g, steps = [], [], []
    for i in range(1, len(pairsLR)):
        Re1, Rg1 = pairsLR[i - 1]
        Re2, Rg2 = pairsLR[i]
        rel_e.append(rot_angle(Re1.T @ Re2))
        rel_g.append(rot_angle(Rg1.T @ Rg2))
        steps.append(i)
    if len(rel_e) > 2:
        import statistics as st
        d = [abs(a - b) for a, b in zip(rel_e, rel_g)]
        m = float(np.corrcoef(rel_e, rel_g)[0, 1])
        print("  RELATIVE rotations (any constant offset cancels):")
        print(f"    est per step: median={st.median(rel_e):7.3f}  mean={st.mean(rel_e):7.3f}  "
              f"p95={np.percentile(rel_e, 95):7.3f} deg")
        print(f"    gt  per step: median={st.median(rel_g):7.3f}  mean={st.mean(rel_g):7.3f}  "
              f"p95={np.percentile(rel_g, 95):7.3f} deg")
        print(f"    |est-gt|    : median={st.median(d):7.3f}  p95={np.percentile(d, 95):7.3f} deg")
        print(f"    correlation ={m:7.4f}   (near 1 => same rotational content)")
        if m > 0.9 and st.median(d) < 5.0:
            print("    -> rotational CONTENT agrees; the gap is a frame convention "
                  "outside the tested set")
        else:
            print("    -> rotational content does NOT agree; the absolute mismatch is "
                  "not merely a convention")
        print()

    # ---------------------------------------------------------------------
    # Criterion for reading the tables above.
    # ---------------------------------------------------------------------
    print("  NOTE: no row above is accepted as a fix unless it collapses BOTH the median")
    print("  and the spread. A large median with small spread means a constant convention")
    print("  offset; a small median with large spread would mean the orientation is simply")
    print("  wrong per frame, which no fixed convention can repair.")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    gt = sys.argv[1]
    for i, est in enumerate(sys.argv[2:], 1):
        check(gt, est, f"case {i}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
