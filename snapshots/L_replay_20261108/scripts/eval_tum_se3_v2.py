#!/usr/bin/env python3
"""SE(3) ATE / RPE for TUM-format trajectories -- ESTIMATE-CENTRIC evaluation.

Why this exists
---------------
`eval_tum_se3.py` associated by iterating over its FIRST argument and used a
monotone pointer without de-duplication.  When called as
`eval_tum_se3.py <gt> <est>` the GT (dense, ~100 Hz) is the iteration set, so a
single estimate frame is emitted many times and the reported `n` is the number
of *GT* samples that found a neighbour -- NOT the number of evaluated estimate
frames.  On `fr3_structure_notexture_near` that produced `n=3551` for a
trajectory that only has 1054 poses.

Correct procedure (this script)
-------------------------------
1. Load GT and estimate, require the SAME time base (report overlap; abort
   WITH NO ATE if the windows do not overlap -- never shift/scale time).
2. Iterate over every *estimate* frame exactly once.  For each, either
   interpolate GT at that timestamp (linear on translation, slerp on rotation)
   or take the nearest GT sample; one estimate frame => at most one error.
3. Report the true pair count (<= number of estimate frames), the time-residual
   distribution, the ATE translation RMSE, the per-frame translation sigma, and
   the RPE (relative pose error) for several delta steps.

Alignment / mode
    --mode se3  (default, for RGB-D / stereo): rigid Umeyama, scale fixed to 1.
    --mode sim3 (for Monocular)              : Umeyama with scale (Sim(3)).
    The alignment is applied to the trajectory BEFORE the RPE is computed.

RPE definition (this is the one used everywhere below)
    For each estimate frame pair (k, k+d) that both have a GT association:
        dt        = t[k+d] - t[k]                                 [s]
        e_trans   = || (p_al[k+d]-p_al[k]) - (p_gt[k+d]-p_gt[k]) || [m]
        dR_est    = R_est[k]^T R_est[k+d]
        dR_gt     = R_gt[k]^T  R_gt[k+d]
        e_rot     = angle( dR_gt^T dR_est )                        [deg]
    where p_al is the ALIGNED estimate position. Using the raw, unaligned
    estimate would contaminate the translation difference with the global
    world-frame rotation and is therefore wrong. e_rot is invariant to the
    global alignment (a common rotation cancels in dR).
    Reported per delta: e_trans in metres AND in m/s (rms/mean/median/max/std),
    and e_rot in degrees (rms/mean/median/max).

Usage:
    python3 scripts/eval_tum_se3_v2.py <gt.txt> <est.txt> [out_prefix]
        [--max-dt 0.02] [--method interp|nearest] [--mode se3|sim3]
        [--rpe-deltas 1,10,30]
"""
import sys
import os
import json
import math
import numpy as np


# ---------------------------------------------------------------- loading ---
def load_tum(path):
    ts, T = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 8:
                continue
            ts.append(float(p[0]))
            T.append([float(x) for x in p[1:8]])
    return np.array(ts, dtype=np.float64), np.array(T, dtype=np.float64)


def quat_to_R(qx, qy, qz, qw):
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n == 0.0:
        return np.eye(3)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def slerp(q0, q1, w):
    """Spherical linear interpolation of unit quaternions [qx,qy,qz,qw]."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1 = -q1
        d = -d
    if d > 0.9995:
        r = q0 + w * (q1 - q0)
        return r / np.linalg.norm(r)
    th0 = math.acos(max(-1.0, min(1.0, d)))
    th = th0 * w
    q2 = q1 - q0 * d
    q2 = q2 / np.linalg.norm(q2)
    return q0 * math.cos(th) + q2 * math.sin(th)


# ------------------------------------------------------------ association ---
def build_pairs(ts_est, ts_gt, max_dt, method):
    """One association per estimate frame.  Returns
    (idx_est, pos_gt, quat_gt, dt) as arrays aligned per estimate frame."""
    order = np.argsort(ts_gt)
    ts_g = ts_gt[order]
    pos_g = None  # filled by caller through closure-free return below

    pairs_ie, pairs_ig, pairs_dt, pairs_w, pairs_i0, pairs_i1 = [], [], [], [], [], []
    n_unmatched = 0
    for ie, te in enumerate(ts_est):
        k = int(np.searchsorted(ts_g, te))
        i0 = min(max(k - 1, 0), len(ts_g) - 1)
        i1 = min(max(k, 0), len(ts_g) - 1)
        d0 = abs(ts_g[i0] - te)
        d1 = abs(ts_g[i1] - te)
        if method == "interp" and i0 != i1:
            dt_span = ts_g[i1] - ts_g[i0]
            if dt_span <= 2 * max_dt:
                w = (te - ts_g[i0]) / dt_span if dt_span > 0 else 0.0
                if 0.0 <= w <= 1.0:
                    pairs_ie.append(ie)
                    pairs_i0.append(i0)
                    pairs_i1.append(i1)
                    pairs_w.append(w)
                    pairs_dt.append(0.0)  # interpolated exactly at te
                    continue
        j = i0 if d0 <= d1 else i1
        if min(d0, d1) <= max_dt:
            pairs_ie.append(ie)
            pairs_i0.append(j)
            pairs_i1.append(j)
            pairs_w.append(0.0)
            pairs_dt.append(abs(ts_g[j] - te))
        else:
            n_unmatched += 1
    return (np.array(pairs_ie, dtype=int), np.array(pairs_i0, dtype=int),
            np.array(pairs_i1, dtype=int), np.array(pairs_w, dtype=np.float64),
            np.array(pairs_dt, dtype=np.float64), order, n_unmatched)


def umeyama(src, dst, with_scale=False):
    """dst ~= s * R @ src + t.  s is fixed to 1 unless with_scale (Sim(3))."""
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    cov = dst_c.T @ src_c / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    if with_scale:
        var_s = float(np.sum(src_c ** 2) / len(src))
        s = float(np.trace(np.diag(D) @ S) / var_s) if var_s > 0 else 1.0
    else:
        s = 1.0
    t = mu_d - s * (R @ mu_s)
    return R, t, s


def rot_angle_deg(Rm):
    """Rotation angle of a 3x3 rotation matrix, in degrees."""
    c = (float(np.trace(Rm)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


# ------------------------------------------------------------------- main ---
def main():
    argv = [a for a in sys.argv[1:]]
    max_dt = 0.02
    method = "interp"
    mode = "se3"
    tmin = None
    tmax = None
    only_ts = None
    rpe_deltas = [1, 10, 30]
    pos = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--max-dt":
            max_dt = float(argv[i + 1]); i += 2; continue
        if a == "--method":
            method = argv[i + 1]; i += 2; continue
        if a == "--mode":
            mode = argv[i + 1]; i += 2; continue
        if a == "--tmin":
            tmin = float(argv[i + 1]); i += 2; continue
        if a == "--tmax":
            tmax = float(argv[i + 1]); i += 2; continue
        if a == "--only-ts":
            only_ts = argv[i + 1]; i += 2; continue
        if a == "--rpe-deltas":
            rpe_deltas = [int(x) for x in argv[i + 1].split(",")]; i += 2; continue
        pos.append(a); i += 1
    if len(pos) < 2:
        print(__doc__)
        return 1
    gt_path, est_path = pos[0], pos[1]
    out_prefix = pos[2] if len(pos) > 2 else None

    ts_gt, T_gt = load_tum(gt_path)
    ts_est, T_est = load_tum(est_path)

    # Optional common-window restriction: only estimate frames inside [tmin,tmax]
    # are evaluated. Used to compare runs whose coverage differs, so that the
    # reported ATE/RPE refer to the SAME time span in every condition.
    if tmin is not None or tmax is not None:
        lo = -np.inf if tmin is None else tmin
        hi = np.inf if tmax is None else tmax
        keep = (ts_est >= lo) & (ts_est <= hi)
        ts_est, T_est = ts_est[keep], T_est[keep]

    # Restrict to an explicit set of common frame timestamps (one per line). This
    # is how runs with different coverage are compared on the SAME frame IDs
    # rather than merely the same time span.
    if only_ts is not None and os.path.exists(only_ts) and os.path.getsize(only_ts) > 0:
        allow = np.atleast_1d(np.loadtxt(only_ts))
        if allow.size:
            keep = np.array([float(np.min(np.abs(allow - t))) <= max_dt
                             for t in ts_est])
            ts_est, T_est = ts_est[keep], T_est[keep]

    res = {
        "gt_path": gt_path,
        "est_path": est_path,
        "method": method,
        "mode": mode,
        "max_dt": max_dt,
        "tmin": tmin,
        "tmax": tmax,
        "only_ts": only_ts,
        "gt_lines": int(len(ts_gt)),
        "gt_uniq_ts": int(len(np.unique(ts_gt))),
        "est_lines": int(len(ts_est)),
        "est_uniq_ts": int(len(np.unique(ts_est))),
        "gt_min_ts": float(ts_gt.min()), "gt_max_ts": float(ts_gt.max()),
        "est_min_ts": float(ts_est.min()), "est_max_ts": float(ts_est.max()),
    }

    # ---- same time base check -------------------------------------------
    lo = max(ts_gt.min(), ts_est.min())
    hi = min(ts_gt.max(), ts_est.max())
    overlap = hi - lo
    span_est = ts_est.max() - ts_est.min()
    frac = overlap / span_est if span_est > 0 else 0.0
    res["overlap_s"] = float(overlap)
    res["overlap_frac_of_est"] = float(frac)
    if overlap <= 0:
        print(f"[EVAL FAILED] GT and estimate time windows do not overlap "
              f"(gt=[{ts_gt.min():.3f},{ts_gt.max():.3f}] "
              f"est=[{ts_est.min():.3f},{ts_est.max():.3f}]). No ATE produced.")
        res["status"] = "FAILED_NO_TIME_OVERLAP"
        if out_prefix:
            with open(out_prefix + ".json", "w") as f:
                json.dump(res, f, indent=2)
        return 3

    ie, i0, i1, w, dt_res, order, n_unmatched = build_pairs(
        ts_est, ts_gt, max_dt, method)
    n_pairs = len(ie)
    res["n_pairs"] = int(n_pairs)
    res["n_est_frames"] = int(len(ts_est))
    res["n_unmatched_est_frames"] = int(n_unmatched)
    if n_pairs < 3:
        print(f"[EVAL FAILED] only {n_pairs} estimate frames associated "
              f"(unmatched={n_unmatched}); no ATE produced.")
        res["status"] = "FAILED_TOO_FEW_PAIRS"
        if out_prefix:
            with open(out_prefix + ".json", "w") as f:
                json.dump(res, f, indent=2)
        return 4

    # GT pose at the (interpolated) estimate timestamps
    ts_g = ts_gt[order]
    pos_g = T_gt[order, 0:3]
    quat_g = T_gt[order, 3:7]
    pos_gt = (1.0 - w)[:, None] * pos_g[i0] + w[:, None] * pos_g[i1]
    quat_gt = np.array([slerp(quat_g[a], quat_g[b], ww)
                        for a, b, ww in zip(i0, i1, w)])

    p_est = T_est[ie, 0:3]
    R, t, s = umeyama(p_est, pos_gt, with_scale=(mode == "sim3"))
    p_al = (s * (R @ p_est.T)).T + t
    err = np.linalg.norm(p_al - pos_gt, axis=1)
    res["align_scale"] = float(s)

    res["time_residual"] = {
        "mean": float(dt_res.mean()),
        "median": float(np.median(dt_res)),
        "p95": float(np.percentile(dt_res, 95)),
        "max": float(dt_res.max()),
    }
    res["ate_se3"] = {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mean": float(err.mean()),
        "median": float(np.median(err)),
        "std": float(err.std(ddof=1)) if len(err) > 1 else 0.0,
        "max": float(err.max()),
        "p95": float(np.percentile(err, 95)),
    }
    # per-frame translation sigma (deviation from the mean of the error vector)
    res["per_frame_sigma_m"] = float(np.linalg.norm(
        (p_al - pos_gt) - (p_al - pos_gt).mean(axis=0), axis=1).std(ddof=1)) \
        if len(err) > 1 else 0.0

    # ---- RPE (relative pose error), computed on the ALIGNED trajectory ----
    R_gt_m = np.array([quat_to_R(*q) for q in quat_gt])
    R_est_m = np.array([quat_to_R(*T_est[ie[k], 3:7]) for k in range(n_pairs)])
    rpe = {}
    for d in rpe_deltas:
        if d <= 0 or n_pairs <= d:
            continue
        idx = {int(v): k for k, v in enumerate(ie)}
        e_t, e_r, dts = [], [], []
        for k in range(len(ie)):
            if ie[k] + d > ie[-1]:
                continue
            k2 = idx.get(int(ie[k]) + d)
            if k2 is None:
                continue
            dt = ts_est[ie[k2]] - ts_est[ie[k]]
            if dt <= 0:
                continue
            d_est = p_al[k2] - p_al[k]
            d_gt = pos_gt[k2] - pos_gt[k]
            e_t.append(np.linalg.norm(d_est - d_gt))
            dR_est = R_est_m[k].T @ R_est_m[k2]
            dR_gt = R_gt_m[k].T @ R_gt_m[k2]
            e_r.append(rot_angle_deg(dR_gt.T @ dR_est))
            dts.append(dt)
        if not e_t:
            continue
        e_t = np.array(e_t); e_r = np.array(e_r); dts = np.array(dts)
        rpe[d] = {
            "n": int(len(e_t)),
            "dt_mean_s": float(dts.mean()),
            "trans_rms_m": float(np.sqrt(np.mean(e_t ** 2))),
            "trans_mean_m": float(e_t.mean()),
            "trans_median_m": float(np.median(e_t)),
            "trans_std_m": float(e_t.std(ddof=1)) if len(e_t) > 1 else 0.0,
            "trans_max_m": float(e_t.max()),
            "trans_rms_m_per_s": float(np.sqrt(np.mean(e_t ** 2)) / dts.mean()),
            "rot_rms_deg": float(np.sqrt(np.mean(e_r ** 2))),
            "rot_mean_deg": float(e_r.mean()),
            "rot_median_deg": float(np.median(e_r)),
            "rot_max_deg": float(e_r.max()),
        }
    res["rpe"] = rpe
    res["status"] = "OK"

    # ---- per-frame evidence --------------------------------------------
    if out_prefix:
        with open(out_prefix + ".perframe.txt", "w") as f:
            f.write("# est_idx ts_est ts_dt_res err_m ex ey ez gt_x gt_y gt_z\n")
            for k in range(n_pairs):
                f.write("%d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n" % (
                    ie[k], ts_est[ie[k]], dt_res[k], err[k],
                    (p_al - pos_gt)[k, 0], (p_al - pos_gt)[k, 1], (p_al - pos_gt)[k, 2],
                    pos_gt[k, 0], pos_gt[k, 1], pos_gt[k, 2]))
        with open(out_prefix + ".json", "w") as f:
            json.dump(res, f, indent=2)

    a = res["ate_se3"]
    print(f"[ATE v2/{mode}/{method}] pairs={n_pairs}/{len(ts_est)} "
          f"(unmatched={n_unmatched})  dt med={res['time_residual']['median']:.4f}s "
          f"max={res['time_residual']['max']:.4f}s  align_scale={res['align_scale']:.6f}")
    print(f"[ATE v2] rmse={a['rmse']:.6f} mean={a['mean']:.6f} "
          f"median={a['median']:.6f} std={a['std']:.6f} max={a['max']:.6f} "
          f"p95={a['p95']:.6f} m")
    for d, v in rpe.items():
        print(f"[RPE v2] d={d} n={v['n']} dt={v['dt_mean_s']:.4f}s "
              f"trans_rms={v['trans_rms_m']:.6f} m "
              f"({v['trans_rms_m_per_s']:.4f} m/s) "
              f"trans_mean={v['trans_mean_m']:.6f} m "
              f"rot_rms={v['rot_rms_deg']:.4f} deg "
              f"rot_mean={v['rot_mean_deg']:.4f} deg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
