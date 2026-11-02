#!/usr/bin/env python3
"""Tests for eval/coverage.py and the per-frame dump of eval_tum_se3_v2.py.

Every test builds a synthetic run directory (final_state.csv +
CameraTrajectory_TUM.txt + a GT file) and drives the real code path, so the
evaluator, the join and the interval logic are all exercised together.

Required cases (from the task):
  T1 estimate == GT
  T2 known SE(3) / Sim(3) transform applied to the estimate
  T3 different sampling rates (100 Hz GT vs 10 Hz video)
  T4 missing estimates (input frame with no exported pose)
  T5 GT gap
  T6 rotation / quaternion handling
  T7 continuous-interval break on non-OK state, map reset and time gap
  T8 duplicate timestamps and invalid (non-finite) poses
  T9 per-frame error aggregation == validated ATE rmse
"""
import math
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coverage as C  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + ("  " + detail if detail else ""))
    if not cond:
        FAILS.append(name)


def R_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    return np.array([qx / n, qy / n, qz / n, qw / n])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def helix(t):
    """Smooth 6 s trajectory: position + rotation."""
    p = np.array([0.5 * math.cos(0.6 * t), 0.5 * math.sin(0.6 * t), 0.1 * t])
    R = rot_z(0.4 * t)
    return p, R


def write_tum(path, samples):
    with open(path, "w") as f:
        for ts, p, q in samples:
            f.write("%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n"
                    % (ts, p[0], p[1], p[2], q[0], q[1], q[2], q[3]))


def build_run(root, est_ts, gt_ts, T=(6.0, 6.4), est_fn=None, gt_fn=None,
              states=None, skip_est_idx=(), bad_pose_idx=(), dup_ts=False):
    """Create <root>/<scene>/<mod>/<arm>/rep1 with GT, est and final_state."""
    gt_fn = gt_fn or helix
    est_fn = est_fn or helix
    os.makedirs(root, exist_ok=True)
    gt_path = os.path.join(root, "gt.txt")
    gt = [(t,) + gt_fn(t) for t in gt_ts]
    write_tum(gt_path, [(t, p, R_to_quat(R)) for t, p, R in gt])

    run = os.path.join(root, "scene", "mono", "T0", "rep1")
    os.makedirs(run, exist_ok=True)
    lines = []
    fs = []
    for i, t in enumerate(est_ts):
        p, R = est_fn(t)
        if i in skip_est_idx:
            if i in bad_pose_idx:
                p = np.array([float("nan")] * 3)
                lines.append("%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n"
                             % (t, p[0], p[1], p[2], *R_to_quat(R)))
            # missing pose: present as input, absent from the trajectory export
        elif i in bad_pose_idx:
            lines.append("%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n"
                         % (t, float("nan"), p[1], p[2], *R_to_quat(R)))
        else:
            lines.append("%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n"
                         % (t, p[0], p[1], p[2], *R_to_quat(R)))
        if dup_ts and i in (5, 6):
            lines.append(lines[-1])
        st = states(i) if states else 2
        fs.append((i, t, st))
    with open(os.path.join(run, "CameraTrajectory_TUM.txt"), "w") as f:
        f.writelines(lines)
    with open(os.path.join(run, "final_state.csv"), "w") as f:
        f.write("frame,timestamp,state,state_name\n")
        for i, t, st in fs:
            f.write("%d,%.6f,%d,x\n" % (i, t, st))
    with open(os.path.join(run, "status.txt"), "w") as f:
        f.write("exit_code=0\n")
    return run, gt_path


def run_eval(run, gt, **kw):
    return C.evaluate_run(run, gt, "sim3",
                          kw.get("max_dt", 0.02), kw.get("gap_max", 0.5),
                          kw.get("rot_tol", 5.0))


def main():
    tmp = tempfile.mkdtemp(prefix="covtest_")
    gt_ts = np.arange(0.0, 6.001, 0.01)      # 100 Hz GT
    est_ts = np.arange(0.0, 6.001, 1.0 / 30)  # 30 Hz video

    print("T1 estimate == GT")
    run, gt = build_run(os.path.join(tmp, "t1"), est_ts, gt_ts)
    r = run_eval(run, gt)
    check("status OK", r["status"] == "OK", r["status"])
    check("ATE ~ 0", r["ate_rmse_m"] < 1e-3, "%.3e" % r["ate_rmse_m"])
    check("accepted cov == 1", abs(r["accepted_cov"] - 1.0) < 1e-9)
    check("gt-evaluable cov == 1", abs(r["gt_evaluable_cov"] - 1.0) < 1e-9)
    check("correct@0.05 == 1", abs(r["correct_cov_0.05"] - 1.0) < 1e-9)
    check("longest == full span",
          abs(r["longest_continuous_s"] - (est_ts[-1] - est_ts[0])) < 0.05,
          "%.3f" % r["longest_continuous_s"])

    print("T2a known SE(3) applied to the estimate")
    A = np.eye(3)
    A = rot_z(0.9).dot(np.array([[1, 0, 0], [0, math.cos(0.4), -math.sin(0.4)],
                                 [0, math.sin(0.4), math.cos(0.4)]]))
    tA = np.array([3.0, -2.0, 0.7])

    def est_se3(t):
        p, R = helix(t)
        return A.dot(p) + tA, A.dot(R)

    run, gt = build_run(os.path.join(tmp, "t2a"), est_ts, gt_ts, est_fn=est_se3)
    r = run_eval(run, gt)
    check("ATE ~ 0 under SE(3)", r["ate_rmse_m"] < 1e-3, "%.3e" % r["ate_rmse_m"])
    check("scale == 1 (se3 mode not used here, sim3 free)",
          abs(r["align_scale"] - 1.0) < 1e-3, "%.6f" % r["align_scale"])

    print("T2b known Sim(3): estimate is GT scaled by 2")
    def est_sim3(t):
        p, R = helix(t)
        return 2.0 * p, R

    run, gt = build_run(os.path.join(tmp, "t2b"), est_ts, gt_ts, est_fn=est_sim3)
    r = run_eval(run, gt)
    check("ATE ~ 0 under Sim(3)", r["ate_rmse_m"] < 1e-3, "%.3e" % r["ate_rmse_m"])
    check("recovered scale == 0.5", abs(r["align_scale"] - 0.5) < 1e-3,
          "%.6f" % r["align_scale"])
    check("align_R ~ identity", np.allclose(np.array(r["align_R"]), np.eye(3), atol=1e-3))

    print("T3 sampling rates: 100 Hz GT vs 10 Hz video")
    slow = np.arange(0.0, 6.001, 0.1)
    run, gt = build_run(os.path.join(tmp, "t3"), slow, gt_ts)
    r = run_eval(run, gt)
    check("ATE ~ 0 despite 10x rate difference", r["ate_rmse_m"] < 1e-3,
          "%.3e" % r["ate_rmse_m"])
    check("n_gt_evaluable == n_input", r["n_gt_evaluable"] == r["n_input"],
          "%d/%d" % (r["n_gt_evaluable"], r["n_input"]))
    check("no undetermined frames", r["undetermined_frac"] == 0.0)

    print("T4 missing estimates (input without exported pose)")
    skip = set(range(40, 60))
    hole_a, hole_b = est_ts[40], est_ts[59]
    run, gt = build_run(os.path.join(tmp, "t4"), est_ts, gt_ts, skip_est_idx=skip)
    r = run_eval(run, gt)
    check("inputs > gt-evaluable", r["n_input"] > r["n_gt_evaluable"],
          "%d vs %d" % (r["n_input"], r["n_gt_evaluable"]))
    check("undetermined_frac == 20/n", abs(r["undetermined_frac"] - 20.0 / len(est_ts)) < 1e-9,
          "%.4f" % r["undetermined_frac"])
    check("correct@0.05 == 1 on evaluated subset",
          abs(r["correct_cov_0.05"] - 1.0) < 1e-9)
    check("lower bound < full", r["correct_lower_bound_0.05"] < 1.0,
          "%.3f" % r["correct_lower_bound_0.05"])
    total = est_ts[-1] - est_ts[0]
    check("interval cannot span the hole",
          r["longest_continuous_s"] <= total - (hole_b - hole_a) + 0.1,
          "longest=%.2f total=%.2f hole=%.2f"
          % (r["longest_continuous_s"], total, hole_b - hole_a))

    print("T5 GT gap")
    gt_gap = np.array([t for t in gt_ts if not (2.0 <= t <= 2.5)])
    run, gt = build_run(os.path.join(tmp, "t5"), est_ts, gt_gap)
    r = run_eval(run, gt)
    check("some inputs lost GT", r["n_gt_evaluable"] < r["n_input"],
          "%d/%d" % (r["n_gt_evaluable"], r["n_input"]))
    check("undetermined_frac > 0", r["undetermined_frac"] > 0.0,
          "%.4f" % r["undetermined_frac"])
    check("gap splits the sequence",
          r["longest_continuous_s"] <= (est_ts[-1] - 2.5) + 0.1,
          "%.2f" % r["longest_continuous_s"])

    print("T6 rotation / quaternion handling")
    def est_rot(t):
        p, R = helix(t)
        return p, R.dot(rot_z(math.radians(20.0)))

    run, gt = build_run(os.path.join(tmp, "t6"), est_ts, gt_ts, est_fn=est_rot)
    r = run_eval(run, gt)
    check("ATE still ~ 0 (rotation only)", r["ate_rmse_m"] < 1e-3,
          "%.3e" % r["ate_rmse_m"])
    check("rot rmse ~ 20 deg", abs(r["rot_rmse_deg"] - 20.0) < 0.2,
          "%.3f" % r["rot_rmse_deg"])
    check("correct@0.05 == 0 because rot gate", r["correct_cov_0.05"] == 0.0)

    print("T7 interval breaks: non-OK state, map reset, time gap")
    def states_lost(i):
        return 4 if 30 <= i <= 40 else 2     # LOST for a stretch
    run, gt = build_run(os.path.join(tmp, "t7a"), est_ts, gt_ts, states=states_lost)
    r = run_eval(run, gt)
    check("LOST stretch splits the interval",
          r["longest_continuous_s"] <= (est_ts[-1] - est_ts[40]) + 0.1,
          "%.2f" % r["longest_continuous_s"])

    def states_reset(i):
        return 0 if i == 30 else 2           # NO_IMAGES_YET => map reset
    run, gt = build_run(os.path.join(tmp, "t7b"), est_ts, gt_ts, states=states_reset)
    r = run_eval(run, gt)
    check("map reset splits the interval at frame 30",
          r["longest_continuous_s"] <= (est_ts[-1] - est_ts[30]) + 0.1,
          "%.2f" % r["longest_continuous_s"])

    gap_ts = np.array([t for t in est_ts if not (2.0 <= t <= 3.5)])
    run, gt = build_run(os.path.join(tmp, "t7c"), gap_ts, gt_ts)
    r = run_eval(run, gt)
    check("time gap > gap_max splits the interval",
          r["longest_continuous_s"] <= (gap_ts[-1] - 3.5) + 0.1,
          "%.2f" % r["longest_continuous_s"])

    print("T8 duplicate timestamps and invalid poses")
    run, gt = build_run(os.path.join(tmp, "t8"), est_ts, gt_ts, dup_ts=True,
                        bad_pose_idx=(70,))
    r = run_eval(run, gt)
    check("survives duplicates + NaN", r["status"] == "OK", r["status"])
    check("ATE still ~ 0", r["ate_rmse_m"] < 1e-3, "%.3e" % r["ate_rmse_m"])

    print("T9 per-frame aggregation == validated ATE")
    # A case with a real (0.05 m) error, so the %.6f dump precision is not the
    # limiting factor. Consistency is asserted to 1e-3 relative; exact equality
    # is not claimed because the dump is printed with 6 decimals.
    def est_noisy(t):
        p, R = helix(t)
        return p + np.array([0.05 * math.sin(3.0 * t), 0.0, 0.0]), R
    run9, gt9 = build_run(os.path.join(tmp, "t9"), est_ts, gt_ts, est_fn=est_noisy)
    r9 = run_eval(run9, gt9)
    pf9 = os.path.join(run9, "eval", "perframe.perframe.txt")
    errs9 = [float(l.split()[3]) for l in open(pf9) if not l.startswith("#")]
    agg9 = float(np.sqrt(np.mean(np.array(errs9) ** 2)))
    check("rms(per-frame err) == ATE rmse (0.05 m case)",
          abs(agg9 - r9["ate_rmse_m"]) / r9["ate_rmse_m"] < 1e-3,
          "agg=%.6f ate=%.6f" % (agg9, r9["ate_rmse_m"]))

    ok = True
    for name in ("t1", "t2b", "t4", "t6"):
        run = os.path.join(tmp, name, "scene", "mono", "T0", "rep1")
        gt = os.path.join(tmp, name, "gt.txt")
        pf = os.path.join(run, "eval", "perframe.perframe.txt")
        errs = []
        for line in open(pf):
            if line.startswith("#"):
                continue
            errs.append(float(line.split()[3]))
        agg = float(np.sqrt(np.mean(np.array(errs) ** 2)))
        r = run_eval(run, gt)
        # tolerance = the 6-decimal print precision of the dump
        if abs(agg - r["ate_rmse_m"]) > 2e-6:
            ok = False
            print("    mismatch %s: agg=%.9f ate=%.9f"
                  % (name, agg, r["ate_rmse_m"]))
    check("clean cases agree within dump precision", ok)

    print()
    if FAILS:
        print("FAILED %d: %s" % (len(FAILS), ", ".join(FAILS)))
        return 1
    print("ALL CHECKS PASSED")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
