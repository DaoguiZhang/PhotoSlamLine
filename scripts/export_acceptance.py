#!/usr/bin/env python3
"""Per-run EXPORT ACCEPTANCE GATE.

A run is not "accepted" because its exit code is 0. This gate recomputes every
claim from the artefacts the run wrote and FAILS CLOSED: the first violated
check makes the whole round fail, so a batch driver can stop before spending
more runs. Nothing here trusts a previous summary or a cache.

Checks, in order:

 A1 input frames          number of frames the sequence can feed (rgb.txt), and
                          the frame-entry count the pipeline accumulated.
 A2 atlas census          per-map keyframe count, which map was selected for
                          export, and how many frame entries reference each map
                          (the exporter prints this census).
 A3 conservation          frame entries == written + skipped_lost +
                          skipped_other_map + skipped_no_refkf, and the per-map
                          reference counts must sum to the non-lost entries.
 A4 format agreement      TUM vs EuRoC (nanosecond timestamps) vs KITTI
                          (rotation matrix, no timestamp, positionally aligned)
                          must describe the SAME poses after conversion.
 A5 timestamp order       the written timestamps are strictly increasing and
                          contain no duplicate.
 A6 continuity            translation between ADJACENT INPUT FRAME IDS only
                          (frames excluded from the export are not "adjacent"),
                          compared with the elapsed time and the ground-truth
                          displacement. Ground-truth COMPARISON IS NEVER USED TO
                          BUILD the trajectory - it is only an explanation for a
                          jump, and magnitudes are compared (frame-invariant
                          under the rigid alignment SE(3) allows, since RGB-D
                          scale is metric).
 A7 mono empty-current    if the current map holds no keyframe while other maps
                          do, the exported map must not be the current one and
                          the fragment must be non-empty.

usage: export_acceptance.py <run_dir> [--tol-trans 1e-4] [--tol-rot-deg 0.05]
Writes acceptance.txt and acceptance.json into <run_dir>; exit 0 iff all pass.
"""
import json
import math
import os
import re
import sys

# ---------------------------------------------------------------- parsing ----

RX_CENSUS_HEAD = re.compile(r"\[System\] (Save\w+): atlas holds (\d+) map\(s\)")
RX_CENSUS_MAP = re.compile(r"\[System\] Save\w+:   map (\d+) has (\d+) keyframe\(s\)(\s+\[current\])?")
RX_TUM_SUM = re.compile(
    r"\[System\] SaveTrajectoryTUM: exported map=(-?\d+) refKFs=(\d+) written=(\d+)"
    r" skipped_lost=(\d+) skipped_other_map=(\d+) skipped_no_refkf=(\d+)"
    r" \(of (\d+) frame entries\)")
RX_TUM_BYMAP = re.compile(
    r"^\[System\] SaveTrajectoryTUM: frame entries by reference-KF map:(.*)$",
    re.MULTILINE)
RX_KITTI_SUM = re.compile(
    r"\[System\] SaveTrajectoryKITTI: exported map=(-?\d+) refKFs=(\d+) written=(\d+)"
    r" skipped_other_map=(\d+) \(of (\d+) frame entries\)")


def load_tum(path):
    """TUM: ts tx ty tz qx qy qz qw"""
    out = []
    for line in open(path, errors="replace"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = s.split()
        if len(p) < 8:
            continue
        out.append((float(p[0]), [float(x) for x in p[1:4]], [float(x) for x in p[4:8]]))
    return out


def load_euroc(path):
    """EuRoC: 1e9*ts tx ty tz qx qy qz qw"""
    out = []
    for line in open(path, errors="replace"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = s.split()
        if len(p) < 8:
            continue
        out.append((float(p[0]) / 1e9, [float(x) for x in p[1:4]], [float(x) for x in p[4:8]]))
    return out


def load_kitti(path):
    """KITTI: r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz (no timestamp)"""
    out = []
    for line in open(path, errors="replace"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = s.split()
        if len(p) < 12:
            continue
        v = [float(x) for x in p]
        R = [v[0:3], v[4:7], v[8:11]]
        t = [v[3], v[7], v[11]]
        out.append((R, t))
    return out


def q_to_R(q):
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def rot_angle_deg(Ra, Rb):
    """Angle of Rb^T Ra, i.e. the rotation needed to take Rb onto Ra."""
    # trace(Rb^T Ra)
    tr = 0.0
    for i in range(3):
        for k in range(3):
            tr += Rb[k][i] * Ra[k][i]
    c = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
    return math.degrees(math.acos(c))


def read_meta(path):
    meta = {}
    if os.path.exists(path):
        for line in open(path, errors="replace"):
            if "=" in line:
                k, v = line.strip().split("=", 1)
                meta[k] = v
    return meta


def fed_frames(seq):
    """Input frame timestamps in feeding order (rgb.txt)."""
    ts = []
    for line in open(os.path.join(seq, "rgb.txt"), errors="replace"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        ts.append(float(s.split()[0]))
    return ts


def load_gt(seq):
    p = os.path.join(seq, "groundtruth.txt")
    if not os.path.exists(p):
        return []
    g = []
    for line in open(p, errors="replace"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        v = s.split()
        if len(v) < 4:
            continue
        g.append((float(v[0]), [float(v[1]), float(v[2]), float(v[3])]))
    g.sort()
    return g


def gt_pos(gt, t):
    """Linear interpolation of ground-truth position (explanation only)."""
    if not gt:
        return None
    lo, hi = 0, len(gt) - 1
    if t <= gt[0][0]:
        return gt[0][1]
    if t >= gt[hi][0]:
        return gt[hi][1]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if gt[mid][0] <= t:
            lo = mid
        else:
            hi = mid
    t0, p0 = gt[lo]
    t1, p1 = gt[hi]
    a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    return [p0[i] + a * (p1[i] - p0[i]) for i in range(3)]


def dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


# ---------------------------------------------------------------- the gate ----

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    run_dir = sys.argv[1]
    tol_trans = 1e-4
    tol_rot = 0.05
    seq_override = None
    if "--seq" in sys.argv:
        seq_override = sys.argv[sys.argv.index("--seq") + 1]
    if "--tol-trans" in sys.argv:
        tol_trans = float(sys.argv[sys.argv.index("--tol-trans") + 1])
    if "--tol-rot-deg" in sys.argv:
        tol_rot = float(sys.argv[sys.argv.index("--tol-rot-deg") + 1])

    meta = read_meta(os.path.join(run_dir, "run.meta"))
    # A run launched by hand may have no run.meta; then the sequence must be given
    # explicitly, otherwise the input-frame checks cannot be evaluated at all.
    seq = seq_override or meta.get("sequence", "")
    log_path = os.path.join(run_dir, "log.txt")
    log = open(log_path, errors="replace").read() if os.path.exists(log_path) else ""

    checks, fails = [], []

    def check(name, ok, detail):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})
        if not ok:
            fails.append(f"{name}: {detail}")

    # ---- A1 input frames -------------------------------------------------
    fed = fed_frames(seq) if seq and os.path.isdir(seq) else []
    tum_path = os.path.join(run_dir, "CameraTrajectory_TUM.txt")
    tum = load_tum(tum_path)
    check("A1 input_frames", len(fed) > 0,
          f"sequence feeds {len(fed)} frames; trajectory file has {len(tum)} entries")

    # ---- A2 atlas census -------------------------------------------------
    heads = list(RX_CENSUS_HEAD.finditer(log))
    census = []
    selected = None
    kfs = {}
    if heads:
        last = heads[-1]
        block = log[last.end():last.end() + 20000]
        for m in RX_CENSUS_MAP.finditer(block):
            census.append({"map": int(m.group(1)), "kfs": int(m.group(2)),
                           "current": bool(m.group(3))})
            kfs[int(m.group(1))] = int(m.group(2))
    msum = None
    for m in RX_TUM_SUM.finditer(log):
        msum = m
    bymap = {}
    for m in RX_TUM_BYMAP.finditer(log):
        for pair in m.group(1).split():
            mm = re.match(r"map(\d+)=(\d+)", pair)
            if mm:
                bymap[int(mm.group(1))] = int(mm.group(2))
                continue
            # Bucket for keyframes that are no longer in any atlas map, e.g.
            # because Map::clear() nulled their map pointer. Key -1.
            mm = re.match(r"no_live_map=(\d+)", pair)
            if mm:
                bymap[-1] = int(mm.group(1))
    check("A2 atlas_census", bool(heads) and bool(census) and msum is not None,
          f"{len(heads)} census block(s); selected map="
          f"{msum.group(1) if msum else 'NA'}; maps="
          f"{ {c['map']: c['kfs'] for c in census} }; entries-by-refmap={bymap}")
    cur = [c for c in census if c["current"]]
    check("A2b selected_map_has_kfs",
          (msum is not None) and (int(msum.group(1)) in kfs) and kfs.get(int(msum.group(1)), 0) > 0,
          f"selected map {msum.group(1) if msum else 'NA'} keys among census")

    # ---- A3 conservation -------------------------------------------------
    if msum:
        s_map, s_refkf, s_wr, s_lost, s_other, s_noref, s_total = [int(g) for g in msum.groups()]
        check("A3 conservation",
              s_wr + s_lost + s_other + s_noref == s_total,
              f"written({s_wr}) + lost({s_lost}) + other_map({s_other}) + "
              f"no_refkf({s_noref}) == entries({s_total})")
        check("A3b per_map_sums_to_entries",
              sum(bymap.values()) == s_total - s_lost - s_noref,
              f"sum(by_ref_map)={sum(bymap.values())} vs entries-lost-no_refkf="
              f"{s_total - s_lost - s_noref}")
        check("A3c written_eq_selected_map_refs",
              bymap.get(s_map, -1) == s_wr,
              f"written={s_wr} vs entries referencing selected map={bymap.get(s_map)}")
        check("A3d file_matches_counter",
              len(tum) == s_wr,
              f"TUM file lines={len(tum)} vs written={s_wr}")
    else:
        check("A3 conservation", False, "no SaveTrajectoryTUM summary line in log")

    # ---- A4 three-format agreement --------------------------------------
    euroc_path = os.path.join(run_dir, "CameraTrajectory_EuRoC.txt")
    kitti_path = os.path.join(run_dir, "CameraTrajectory_KITTI.txt")
    if os.path.exists(euroc_path):
        eu = load_euroc(euroc_path)
        check("A4a tum_vs_euroc_count", len(eu) == len(tum),
              f"TUM={len(tum)} EuRoC={len(eu)}")
        if eu and len(eu) == len(tum):
            dt = max(abs(a[0] - b[0]) for a, b in zip(tum, eu))
            dpos = max(dist(a[1], b[1]) for a, b in zip(tum, eu))
            drot = max(rot_angle_deg(q_to_R(a[2]), q_to_R(b[2])) for a, b in zip(tum, eu))
            check("A4b tum_vs_euroc_values",
                  dt < 1e-6 and dpos < tol_trans and drot < tol_rot,
                  f"max|dts|={dt:.3e}s max|dpos|={dpos:.3e}m max|drot|={drot:.4f}deg")
    else:
        check("A4a tum_vs_euroc_count", False, "EuRoC file missing")

    if os.path.exists(kitti_path):
        ki = load_kitti(kitti_path)
        check("A4c tum_vs_kitti_count", len(ki) == len(tum),
              f"TUM={len(tum)} KITTI={len(ki)}")
        if ki and len(ki) == len(tum):
            dpos = max(dist(a[1], b[1]) for a, b in zip(tum, ki))
            drot = max(rot_angle_deg(q_to_R(a[2]), b[0]) for a, b in zip(tum, ki))
            check("A4d tum_vs_kitti_values",
                  dpos < tol_trans and drot < tol_rot,
                  f"max|dpos|={dpos:.3e}m max|drot|={drot:.4f}deg")
    else:
        check("A4c tum_vs_kitti_count", False, "KITTI file missing")

    # ---- A5 timestamp order ---------------------------------------------
    ts = [e[0] for e in tum]
    check("A5 timestamps_unique", len(set(ts)) == len(ts),
          f"{len(ts)} timestamps, {len(set(ts))} unique")
    check("A5b strictly_increasing", all(ts[i] < ts[i + 1] for i in range(len(ts) - 1)),
          "monotone" if all(ts[i] < ts[i + 1] for i in range(len(ts) - 1)) else "not monotone")

    # ---- A6 continuity across ADJACENT INPUT FRAME IDS -------------------
    # The FAIL criterion applies only to pairs that are adjacent input frames
    # (frame_gap == 1). A pair that skips input frames is expected to move
    # further; such pairs are reported with the elapsed time and the
    # ground-truth displacement as an EXPLANATION, never as a failure.
    #
    # The threshold differs by sensor because scale does not mean the same
    # thing: RGB-D is metric, so the estimate may be compared with ground truth
    # in metres. Monocular SLAM has no metric scale (no depth, no IMU), so a
    # metre comparison would be meaningless; there the test is an internal
    # outlier test against the run's own median adjacent step.
    idx_of = {round(t, 6): i for i, t in enumerate(fed)}
    gt = load_gt(seq) if seq else []
    mode = meta.get("mode", "")
    present = []
    unmapped = 0
    for t, p, _q in tum:
        i = idx_of.get(round(t, 6))
        if i is None:
            unmapped += 1
            continue
        present.append((i, t, p))
    present.sort()

    adj = []
    for (i0, t0, p0), (i1, t1, p1) in zip(present, present[1:]):
        if i1 <= i0:
            continue
        g0, g1 = gt_pos(gt, t0), gt_pos(gt, t1)
        d_gt = dist(g0, g1) if (g0 and g1) else float("nan")
        adj.append({"i0": i0, "i1": i1, "t0": t0, "t1": t1,
                    "est": dist(p0, p1), "gt": d_gt, "gap": i1 - i0})
    adj_only = [a for a in adj if a["gap"] == 1]
    med_adj = sorted(a["est"] for a in adj_only)[len(adj_only) // 2] if adj_only else 0.0

    violations, gap_motion = [], []
    for a in adj:
        speed = a["est"] / (a["t1"] - a["t0"]) if a["t1"] > a["t0"] else float("inf")
        if a["gap"] == 1:
            if mode == "rgbd":
                bad = a["est"] > 0.5 or (a["gt"] == a["gt"] and a["est"] > 5 * a["gt"] + 0.2)
            else:
                # scale-free: adjacent frames must not move wildly more than the
                # run's own typical adjacent step
                bad = a["est"] > max(20.0 * med_adj, 0.05)
            if bad:
                violations.append({"frame_id0": a["i0"], "frame_id1": a["i1"],
                                   "gap": a["gap"], "dt": round(a["t1"] - a["t0"], 4),
                                   "est_disp": round(a["est"], 4),
                                   "gt_disp": None if a["gt"] != a["gt"] else round(a["gt"], 4),
                                   "est_speed": round(speed, 3)})
        else:
            # Not adjacent: informational, explained by dt and the GT motion.
            item = {"frame_id0": a["i0"], "frame_id1": a["i1"], "gap": a["gap"],
                    "dt": round(a["t1"] - a["t0"], 4), "est_disp": round(a["est"], 4),
                    "gt_disp": None if a["gt"] != a["gt"] else round(a["gt"], 4),
                    "est_speed": round(speed, 3)}
            if mode == "rgbd" and speed > 5.0:
                # physically impossible for a hand-held camera, so still a fail
                violations.append(item)
            elif a["est"] > 0.5:
                gap_motion.append(item)

    check("A6 continuity (adjacent input frames only)", len(violations) == 0,
          f"{len(present)} frames checked ({unmapped} unmapped to an input frame); "
          f"{len(adj_only)} adjacent-frame steps, median={med_adj:.4f}; "
          f"mode={mode}; {len(violations)} violation(s)")
    detail_jumps = violations
    detail_gaps = sorted(gap_motion, key=lambda x: -x["est_disp"])[:8]

    # ---- A7 mono empty-current-map rule ----------------------------------
    if cur and cur[0]["kfs"] == 0 and any(c["kfs"] > 0 for c in census):
        ok = (msum is not None) and int(msum.group(1)) != cur[0]["map"] and len(tum) > 0
        check("A7 mono_empty_current_map", ok,
              f"current map {cur[0]['map']} has 0 KFs; exported map="
              f"{msum.group(1) if msum else 'NA'} with {len(tum)} entries")
    else:
        check("A7 mono_empty_current_map", True, "current map non-empty (rule not triggered)")

    # ---- report ---------------------------------------------------------
    result = {
        "run_dir": run_dir,
        "meta": {k: meta.get(k) for k in ("mode", "cond", "run_idx", "stress_drop_point_ratio",
                                          "export_map", "export_written",
                                          "export_skipped_other_map")},
        "input_frames": len(fed),
        "trajectory_entries": len(tum),
        "atlas_maps": {str(c["map"]): c["kfs"] for c in census},
        "selected_map": int(msum.group(1)) if msum else None,
        "frames_by_ref_map": {str(k): v for k, v in bymap.items()},
        "checks": checks,
        "jumps": detail_jumps,
        "gap_motion": detail_gaps,
        "pass": len(fails) == 0,
        "failures": fails,
    }
    with open(os.path.join(run_dir, "acceptance.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(run_dir, "acceptance.txt"), "w") as f:
        f.write(f"EXPORT ACCEPTANCE  run={os.path.basename(run_dir)}  "
                f"mode={meta.get('mode')} cond={meta.get('cond')} "
                f"drop={meta.get('stress_drop_point_ratio')}\n")
        f.write(f"input frames={len(fed)}  trajectory entries={len(tum)}\n")
        f.write(f"atlas maps (id:kfs)={ {c['map']: c['kfs'] for c in census} }\n")
        f.write(f"selected map={result['selected_map']}  "
                f"frames by ref map={bymap}  (-1 = keyframe no longer in any atlas map)\n")
        for c in checks:
            f.write(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['check']}: {c['detail']}\n")
        if detail_jumps:
            f.write("  violations on adjacent input frames:\n")
            for j in detail_jumps:
                f.write(f"    {j}\n")
        if detail_gaps:
            f.write("  non-adjacent pairs (explained by dt and GT motion, not failures):\n")
            for j in detail_gaps:
                f.write(f"    {j}\n")
        f.write(f"RESULT: {'PASS' if result['pass'] else 'FAIL'}\n")

    print(f"[acceptance] {os.path.basename(run_dir)}: "
          f"{'PASS' if result['pass'] else 'FAIL'} "
          f"({sum(1 for c in checks if c['pass'])}/{len(checks)} checks)")
    for f_ in fails:
        print(f"    FAIL {f_}")
    for j in detail_jumps:
        print(f"    violation {j}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
