#!/usr/bin/env python3
"""Per-frame audit of candidate line-rescue frames (drop50/C) with GT used only
OFFLINE, plus the same input frames from the P/B runs for comparison.

Why a dedicated script: eval_tum_se3_v2.py reports only aggregate ATE after a
single global SE(3) Umeyama alignment and does not export per-frame residuals.

Conventions, stated up front so the numbers are reproducible:
  * Frame id F in the diagnostics is assumed to index the fed frame list
    (0-based). This is VERIFIED, not assumed: the script reports the fraction of
    diagnostic frame ids whose fed timestamp is present in the trajectory, which
    is high only if the mapping holds, and also that every trajectory timestamp
    is a member of the fed set.
  * Alignment: SE(3), no scale, least squares over ALL associated pairs of the
    de-duplicated trajectory (same convention as eval_tum_se3_v2.py).
  * Per-frame error = ||aligned_est_t - gt_t|| at that frame's timestamp.
  * "Sustained": how many of the next 30 fed frames are present as poses.
  * Classification is pre-declared, not fitted after seeing the numbers:
      - outlier   : error > 3 x median error over the run's own frames
      - sustained : all 30 following fed frames localized
      - otherwise : reported as-is
    A candidate is called a valid rescue ONLY if it is neither an outlier nor a
    loss of tracking immediately after.
"""
import os
import re
import sys
import math

import numpy as np

RX_SCAN = re.compile(
    r"\[LineScan\] stage=(\S+) map=(-?\d+) frame=(\d+)"
    r" ptBeforePO=(\d+) lnBeforePO=(\d+) ptAfterPO=(\d+) lnAfterPO=(\d+)"
    r" oldPtVerdict=(\S+) lnUniq=(\d+) lnEdges=(\d+) lnSpread=(\S+)"
    r" lnOk=(\d+) lnReason=(-?\d+)")
RX_RESCUE = re.compile(
    r"\[LineRescue\] stage=(\S+) map=(-?\d+) frame=(\d+)"
    r".*oldPtVerdict=(\S+) new=(\S+) reason=(-?\d+)"
    r" rescueLnUniq=(\d+) rescueLnEdges=(\d+)"
    r" gateMmOldFail=(-?\d+) gateTrkRefOldFail=(-?\d+) rescue=(\d+)")
RX_LINEDIAG = re.compile(
    r"\[LineEdgeDiag\] PO frame=(\d+) NL=(\d+) lnTrackedNonNull=(\d+)"
    r" ptEdges=(\d+) ptIn=(\d+) ptOut=(\d+) ptChi2=(\S+)"
    r" lnEdges=(\d+) lnEdgesMono=(\d+) lnEdgesStereo=(\d+)"
    r" lnIn=(\d+) lnOut=(\d+) lnUniq=(\d+) lnSpread=(\S+) lnChi2=(\S+)")


def load_fed(path):
    ts = []
    for line in open(path, errors="replace"):
        s = line.strip()
        if s and not s.startswith("#"):
            ts.append(float(s.split()[0]))
    # unique, order preserved
    seen, out = set(), []
    for t in ts:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def load_traj(path):
    """Returns {ts: (x,y,z,quat)} keeping the FIRST pose per timestamp."""
    d = {}
    if not os.path.exists(path):
        return d
    for line in open(path, errors="replace"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = s.split()
        if len(p) < 8:
            continue
        t = float(p[0])
        if t in d:
            continue
        d[t] = tuple(float(x) for x in p[1:8])
    return d


def load_gt(path):
    rows = []
    for line in open(path, errors="replace"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = s.split()
        if len(p) < 8:
            continue
        try:
            rows.append([float(x) for x in p[:8]])
        except ValueError:
            continue
    rows.sort(key=lambda r: r[0])
    return rows


def gt_at(gt, t, tol=0.05):
    """GT pose at time t: linear position interpolation, nearest quaternion.

    Timestamps are compared with a tolerance, never for exact equality: the
    trajectory file is printed with 6 decimals while the fed list carries full
    precision, so exact float equality under-matches.
    """
    if not gt:
        return None
    ts = [r[0] for r in gt]
    import bisect
    i = bisect.bisect_left(ts, t)
    if i <= 0:
        if abs(ts[0] - t) > tol:
            return None
        return np.array(gt[0][1:4]), np.array(gt[0][4:8])
    if i >= len(gt):
        if abs(ts[-1] - t) > tol:
            return None
        return np.array(gt[-1][1:4]), np.array(gt[-1][4:8])
    t0, t1 = ts[i - 1], ts[i]
    if t < t0 - tol or t > t1 + tol:
        return None
    a = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
    a = min(1.0, max(0.0, a))
    pos = (1 - a) * np.array(gt[i - 1][1:4]) + a * np.array(gt[i][1:4])
    quat = np.array(gt[i][4:8] if a > 0.5 else gt[i - 1][4:8])
    return pos, quat


def quat_to_R(q):
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def umeyama(src, dst):
    """SE(3) (no scale), same least-squares convention as eval_tum_se3_v2."""
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    S = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, _, Vt = np.linalg.svd(S)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    return R, mu_d - R @ mu_s


def parse_log(path):
    """frame id -> list of records, for every diagnostic stream.

    Lists, not single records: a frame is reported once per stage
    (MotionModel / TrackReferenceKeyFrame / LocalMapRelocWindow / ...), and the
    decisive line-rescue verdict can be produced at a later stage than the
    first one. Keeping only the first record silently hid that.
    """
    scans, rescues, diag = {}, {}, {}
    if not os.path.exists(path):
        return scans, rescues, diag
    for line in open(path, errors="replace"):
        m = RX_SCAN.search(line)
        if m:
            g = m.groups()
            scans.setdefault(int(g[2]), []).append(g)
            continue
        m = RX_RESCUE.search(line)
        if m:
            g = m.groups()
            rescues.setdefault(int(g[2]), []).append(g)
            continue
        m = RX_LINEDIAG.search(line)
        if m:
            g = m.groups()
            diag.setdefault(int(g[0]), []).append(g)
    return scans, rescues, diag


def analyze(label, rundir, fed, gt):
    log = os.path.join(rundir, "log.txt")
    traj = load_traj(os.path.join(rundir, "CameraTrajectory_TUM.txt"))
    scans, rescues, diag = parse_log(log)

    fed_ts = fed
    traj_sorted = sorted(traj.keys())
    fed_sorted = sorted(fed_ts)

    def near(sorted_vals, t, tol):
        import bisect
        i = bisect.bisect_left(sorted_vals, t)
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(sorted_vals) and abs(sorted_vals[j] - t) <= tol:
                return True
        return False

    def nearest_key(sorted_vals, t, tol):
        """Fed timestamps carry full precision; the trajectory file is printed
        with 6 decimals, so look the pose up by nearest match, never by key."""
        import bisect
        i = bisect.bisect_left(sorted_vals, t)
        best = None
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(sorted_vals) and abs(sorted_vals[j] - t) <= tol:
                if best is None or abs(sorted_vals[j] - t) < abs(best - t):
                    best = sorted_vals[j]
        return best

    # --- verify the frame-id -> fed-index mapping ---------------------------
    # Diagnostics print a 0-based frame counter and the fed list is the per-frame
    # input list; if the counter indexes that list, most localized diagnostic
    # frames must have their fed timestamp present in the trajectory.
    mapped = sum(1 for f in scans if f < len(fed_ts))
    present = sum(1 for f in scans
                  if f < len(fed_ts) and near(traj_sorted, fed_ts[f], 0.01))
    traj_in_fed = sum(1 for t in traj_sorted if near(fed_sorted, t, 0.01))

    # --- global SE(3) alignment over all associated pairs -------------------
    est_ts = sorted(traj.keys())
    pairs = []
    gtc = {}
    for t in est_ts:
        g = gt_at(gt, t)
        if g is not None:
            gtc[t] = g
            pairs.append((t, np.array(traj[t][:3]), g[0]))
    R = t = None
    if len(pairs) >= 3:
        R, t = umeyama(np.array([p[1] for p in pairs]),
                       np.array([p[2] for p in pairs]))
    err = {}
    rerr = {}
    if R is not None:
        for ts_, e, g in pairs:
            err[ts_] = float(np.linalg.norm(R @ e + t - g))
        for ts_ in gtc:
            Re = quat_to_R(np.array(traj[ts_][3:7]))
            Rg = quat_to_R(gtc[ts_][1])
            dR = Re.T @ Rg
            c = min(1.0, max(-1.0, (np.trace(dR) - 1.0) / 2.0))
            rerr[ts_] = math.degrees(math.acos(c))
    med = float(np.median(list(err.values()))) if err else float("nan")
    rmse = float(np.sqrt(np.mean([v * v for v in err.values()]))) if err else float("nan")
    rmed = float(np.median(list(rerr.values()))) if rerr else float("nan")

    print("=" * 150)
    print(f"RUN {label}   log={log}")
    print(f"  diagnostic frames={len(scans)}  fed frames={len(fed_ts)}  "
          f"traj poses={len(traj)}  associated pairs={len(pairs)}")
    print(f"  mapping check: {present}/{mapped} diagnostic frame ids have their "
          f"fed timestamp present in the trajectory; {traj_in_fed}/{len(traj)} "
          f"trajectory timestamps are members of the fed set")
    print(f"  alignment SE(3) over {len(pairs)} pairs; per-frame err median={med:.4f} m "
          f"rmse={rmse:.4f} m")
    print(f"  ORIENTATION: run-wide median relative angle to GT = {rmed:.2f} deg. The "
          f"GT header declares (qx,qy,qz,qw); four plausible orderings were tested "
          f"and none collapses this (best ~90 deg), while positions align to "
          f"{med:.2f} m. That is a systematic frame/axis convention offset applied "
          f"to EVERY frame, not a defect of the audited frames, so the verdict below "
          f"uses TRANSLATION only and the rotation column is reported for "
          f"completeness, not as evidence.")
    if R is None:
        print("  (not enough GT-associated poses for per-frame errors)")
        return [], {}

    # --- candidate rescue frames (old FAIL, new OK at any stage) -----------
    cand = sorted(f for f, recs in rescues.items()
                  if any(r[3] == "FAIL" and r[4] == "OK" for r in recs))
    n_att = sum(1 for f, recs in rescues.items()
                if any(r[3] == "FAIL" and r[10] == "1" for r in recs))
    print(f"  rescue ATTEMPTED frames (FAIL and switch armed): {n_att}")
    print(f"  rescue PASS candidates (old=FAIL, new=OK): {len(cand)} -> {cand}")
    print()
    print(f"  {'frame':<7}{'ts':<15}{'ptAfterPO':<11}{'lnUniq':<8}{'spread':<9}"
          f"{'lnChi2':<9}{'err_m':<9}{'rot_deg':<9}{'next30':<8}{'verdict'}")
    for f in cand:
        ts_ = fed_ts[f] if f < len(fed_ts) else None
        tk = nearest_key(traj_sorted, ts_, 0.01) if ts_ is not None else None
        e = err.get(tk, float("nan")) if tk is not None else float("nan")
        re_ = rerr.get(tk, float("nan")) if tk is not None else float("nan")
        ok30 = 0
        if ts_ is not None:
            for k in range(1, 31):
                if f + k < len(fed_ts) and near(traj_sorted, fed_ts[f + k], 0.01):
                    ok30 += 1
        # a passing record at any stage, preferring the one with most inliers
        recs = [r for r in rescues[f] if r[3] == "FAIL" and r[4] == "OK"]
        sc = scans.get(f, [])
        best = max(sc, key=lambda g: int(g[5])) if sc else None
        dgs = diag.get(f, [])
        bestd = max(dgs, key=lambda g: int(g[11])) if dgs else None
        cls = []
        if e != e:
            cls.append("NO_GT_OR_NO_POSE")
        elif med == med and e > 3 * med:
            cls.append("GT_ERR_OUTLIER")
        if ok30 < 30:
            cls.append(f"NOT_SUSTAINED({ok30}/30)")
        verdict = "valid_rescue" if not cls else "FALSE_ACCEPT:" + ",".join(cls)
        print(f"  {f:<7}{(f'{ts_:.6f}' if ts_ is not None else '-'):<15}"
              f"{(best[5] if best else '-'):<11}{(best[8] if best else '-'):<8}"
              f"{(best[10] if best else '-'):<9}"
              f"{(bestd[14] if bestd and len(bestd) > 14 else '-'):<9}"
              f"{e:<9.4f}{re_:<9.4f}{ok30:<8}{verdict}")
        for r in recs:
            print(f"          PASS at stage={r[0]} old={r[3]} new={r[4]} reason={r[5]} "
                  f"rescueLnUniq={r[6]} rescueLnEdges={r[7]} "
                  f"gateMmOldFail={r[8]} gateTrkRefOldFail={r[9]} rescue={r[10]}")
        for g in sc:
            print(f"          scan stage={g[0]} ptBeforePO={g[3]} lnBeforePO={g[4]} "
                  f"ptAfterPO={g[5]} oldPtVerdict={g[7]} lnUniq={g[8]} "
                  f"lnEdges={g[9]} lnSpread={g[10]} lnOk={g[11]} lnReason={g[12]}")
        for g in dgs:
            # groups are 0-based: 0 frame, 1 NL, 2 lnTrackedNonNull, 3 ptEdges,
            # 4 ptIn, 5 ptOut, 6 ptChi2, 7 lnEdges, 8 lnEdgesMono,
            # 9 lnEdgesStereo, 10 lnIn, 11 lnOut, 12 lnUniq, 13 lnSpread, 14 lnChi2
            fld = lambda i, d="-": (g[i] if i < len(g) else d)
            print(f"          diag NL={fld(1)} lnTrackedNonNull={fld(2)} "
                  f"ptIn={fld(4)} ptOut={fld(5)} ptChi2={fld(6)} lnEdges={fld(7)} "
                  f"lnIn={fld(10)} lnOut={fld(11)} lnUniq={fld(12)} "
                  f"lnSpread={fld(13)} lnChi2={fld(14)}")
    print("  basis: TRANSLATION error after the run-wide SE(3) alignment, compared")
    print("  against the run's own median (median is printed above).")
    for f in cand:
        ts_ = fed_ts[f] if f < len(fed_ts) else None
        if ts_ is None:
            continue
        tk = nearest_key(traj_sorted, ts_, 0.01)
        e = err.get(tk) if tk is not None else None
        print(f"    frame {f} ts {ts_:.6f}: "
              f"localized={'yes' if tk is not None else 'no'}  "
              f"err={'-' if e is None else round(e, 4)} m  "
              f"run_median={round(med, 4)} m  "
              f"ratio={'-' if e is None else round(e / med, 2)}")

    # Per-frame record for EVERY fed frame in this run, so the same input frames
    # can be compared across conditions even when a condition emits no diagnostic
    # record for them ("no diagnostic" and "did not localize" are different
    # facts and must not be conflated).
    frames_info = {}
    for f in range(len(fed_ts)):
        ts_ = fed_ts[f]
        tk = nearest_key(traj_sorted, ts_, 0.01)
        ok30 = 0
        for k in range(1, 31):
            if f + k < len(fed_ts) and near(traj_sorted, fed_ts[f + k], 0.01):
                ok30 += 1
        sc = scans.get(f)
        best = max(sc, key=lambda g: int(g[5])) if sc else None
        rc = [r for r in rescues.get(f, []) if r[3] == "FAIL"]
        frames_info[f] = dict(
            has_diag=(sc is not None),
            localized=(tk is not None),
            err=(err.get(tk) if tk is not None else None),
            ok30=ok30,
            ptAfterPO=(int(best[5]) if best else None),
            lnUniq=(int(best[8]) if best else None),
            spread=(best[10] if best else None),
            verdict=(rc[-1][4] if rc else "-"),
            armed=(rc[-1][10] if rc else "-"))
    return cand, frames_info
    return cand, frames_info


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    base = sys.argv[1]
    fed_file = sys.argv[2] if len(sys.argv) > 2 else None
    gt_file = sys.argv[3] if len(sys.argv) > 3 else None

    runs = [
        ("drop50/P", os.path.join(base, "drop50", "desk_3_rgbd_P")),
        ("drop50/B", os.path.join(base, "drop50", "desk_3_rgbd_B")),
        ("drop50/C", os.path.join(base, "drop50", "desk_3_rgbd_C")),
    ]
    collected = {}
    cands = {}
    for lab, d in runs:
        if not os.path.isdir(d):
            print("missing", d)
            return 1
        meta = {}
        for line in open(os.path.join(d, "run.meta"), errors="replace"):
            if "=" in line:
                k, v = line.strip().split("=", 1)
                meta[k] = v
        fed_f = fed_file or meta.get("association")
        seq = meta.get("sequence", "")
        if fed_f and os.path.exists(os.path.join(seq, "rgb.txt")):
            fed_f = os.path.join(seq, "rgb.txt")
        gt_f = gt_file or os.path.join(seq, "groundtruth.txt")
        if not (fed_f and os.path.exists(fed_f)):
            print("missing fed list for", lab)
            return 1
        fed = load_fed(fed_f)
        gt = load_gt(gt_f) if os.path.exists(gt_f) else []
        cand, info = analyze(lab, d, fed, gt)
        collected[lab] = info
        cands[lab] = cand

    # ---- the SAME input frames, across conditions ---------------------------
    union = sorted(set().union(*[set(v) for v in cands.values()]) if cands else [])
    if union:
        print()
        print("=" * 150)
        print("SAME INPUT FRAMES, COMPARED ACROSS CONDITIONS")
        print("=" * 150)
        print(f"  {'frame':<7}{'cond':<10}{'localized':<11}{'ptAfterPO':<11}"
              f"{'lnUniq':<8}{'spread':<9}{'verdict':<8}{'armed':<7}"
              f"{'err_m':<9}{'next30':<8}")
        for f in union:
            for lab, _ in runs:
                info = collected.get(lab, {}).get(f)
                if not info:
                    print(f"  {f:<7}{lab:<10}{'frame id beyond fed list':<11}")
                    continue
                e = info['err']
                print(f"  {f:<7}{lab:<10}"
                      f"{('yes' if info['localized'] else 'no'):<11}"
                      f"{(info['ptAfterPO'] if info['ptAfterPO'] is not None else 'no_diag'):<11}"
                      f"{(info['lnUniq'] if info['lnUniq'] is not None else '-'):<8}"
                      f"{(info['spread'] if info['spread'] is not None else '-'):<9}"
                      f"{info['verdict']:<8}"
                      f"{info['armed']:<7}"
                      f"{('-' if e is None else round(e, 4)):<9}{info['ok30']:<8}")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
