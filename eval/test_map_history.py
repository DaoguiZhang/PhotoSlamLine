#!/usr/bin/env python3
"""Counterexamples for the map-history reconstruction.

Each test builds a tiny but REAL run directory on disk (map_events.csv,
online_poses.csv, maps_summary.csv), runs the same code path the pipeline uses,
and asserts the verdict. The point is to pin the cases where a backward
reference-frame jump must NOT be treated as proof, and where a brief loss must
not create a new alignment.

Run: python3 eval/test_map_history.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mapmeta          # noqa: E402
import map_history_check as mhc  # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    print("%-58s %s %s" % (name, "PASS" if cond else "FAIL", detail))
    if not cond:
        FAILED.append(name)


def make_run(tmp, events, frames, summary_maps):
    """events: list of (event, map_id, kf_or_target); frames: list of dicts."""
    with open(os.path.join(tmp, "map_events.csv"), "w") as f:
        f.write("# synthetic\nseq,event,map_id,kf_id,reason\n")
        for i, (ev, mid, kf) in enumerate(events):
            f.write("%d,%s,%d,%d,synthetic\n" % (i, ev, mid, kf))
    with open(os.path.join(tmp, "online_poses.csv"), "w") as f:
        f.write("frame,timestamp,state,state_name,pose_valid,map_id,ref_kf_id,"
                "frame_id,tx,ty,tz,qx,qy,qz,qw\n")
        for r in frames:
            f.write("%d,%.6f,%d,%s,%d,%d,%d,%d,0,0,0,0,0,0,1\n"
                    % (r["frame"], r["ts"], r["state"], r["name"],
                       r["valid"], r["map_id"], r["ref_kf"], r["frame"]))
    with open(os.path.join(tmp, "maps_summary.csv"), "w") as f:
        f.write("map_id,is_in_use,is_bad,n_keyframes,n_mappoints,n_maplines,"
                "n_exported_kf,traj_file\n")
        for mid, nkf in summary_maps:
            p = os.path.join(tmp, "traj_map%d.txt" % mid)
            with open(p, "w") as g:
                for i in range(nkf):
                    g.write("%.6f 0 0 0 0 0 0 1\n" % (i * 0.033))
            f.write("%d,0,0,%d,0,0,%d,%s\n" % (mid, nkf, nkf, p))


def frames_uniform(n=60, map_id=1, kf0=10, state=2, valid=1):
    return [{"frame": i, "ts": i * 0.033, "state": state,
             "name": "OK" if state == 2 else "RECENTLY_LOST",
             "valid": valid, "map_id": map_id, "ref_kf": kf0 + i}
            for i in range(n)]


def scenario(tmp, events, frames, summary):
    d = tempfile.mkdtemp(dir=tmp)
    make_run(d, events, frames, summary)
    return mhc.check_run(d)


def main():
    tmp = tempfile.mkdtemp(prefix="maphist_")
    try:
        # T1: every frame valid, one merge, no discontinuity in the stream.
        # The merge instant cannot be located => AMBIGUOUS, never OK.
        r = scenario(tmp,
                     [("MAP_CREATE", 1, 0), ("MAP_MERGE", 1, 2)],
                     frames_uniform(),
                     [(1, 60)])
        check("T1 merge unlocatable, all frames valid -> AMBIGUOUS",
              r["verdict"] == "AMBIGUOUS", r["verdict"])

        # T2: big backward reference-frame jump inside one map and NO merge.
        # Must stay ONE epoch: a jump is not evidence of a coordinate change.
        fr = frames_uniform(60)
        for i in range(30, 60):
            fr[i]["ref_kf"] = 5 + i          # jump backwards by ~ (39-5)
        r = scenario(tmp, [("MAP_CREATE", 1, 0)], fr, [(1, 60)])
        check("T2 backward jump, no merge -> one epoch, OK",
              r["verdict"] == "OK" and r["n_epochs"] == 1,
              "%s epochs=%d" % (r["verdict"], r["n_epochs"]))

        # T3: merge with the SAME map id before and after and no jump.
        # Cannot place the merge => AMBIGUOUS.
        r = scenario(tmp,
                     [("MAP_CREATE", 4, 0), ("MAP_MERGE", 4, 9)],
                     frames_uniform(map_id=4), [(4, 60)])
        check("T3 merge, same map id, no jump -> AMBIGUOUS",
              r["verdict"] == "AMBIGUOUS", r["verdict"])

        # T4: brief loss, then back to the SAME map/world frame. One alignment
        # epoch (no free re-alignment) but continuity is broken.
        fr = frames_uniform(90)
        for i in range(40, 50):
            fr[i]["state"] = 3
            fr[i]["name"] = "RECENTLY_LOST"
            fr[i]["valid"] = 0
        r = scenario(tmp, [("MAP_CREATE", 1, 0)], fr, [(1, 90)])
        check("T4 brief loss, same world frame -> 1 epoch, OK",
              r["verdict"] == "OK" and r["n_epochs"] == 1,
              "%s epochs=%d" % (r["verdict"], r["n_epochs"]))
        on = mapmeta.read_online(r["run"])
        eps = mapmeta.align_epochs(on, n_merge=0)
        states = {}
        for f in fr:
            states[f["frame"]] = f["state"]
            states[("ts", f["frame"])] = f["ts"]
        errs = {round(f["ts"], 6): (0.01, 0.5) for f in fr}   # all correct
        best_s, best_n = mapmeta.continuity_intervals(
            eps[0], states, errs, tol=0.20, rot_tol=5.0, gap_max=0.5)
        check("T4b continuity broken by the loss (not 90 frames)",
              best_n < 90 and best_n >= 39,
              "longest=%d frames" % best_n)

        # T5: two merges with id reuse and locatable discontinuities each.
        # start high and step DOWN twice, so both merges have a backward
        # discontinuity available (a forward step is not a merge signature)
        fr = frames_uniform(90, map_id=7)
        for i in range(0, 30):
            fr[i]["ref_kf"] = 300 + i
        for i in range(30, 60):
            fr[i]["ref_kf"] = 150 + (i - 30)
        for i in range(60, 90):
            fr[i]["ref_kf"] = 40 + (i - 60)
        r = scenario(tmp,
                     [("MAP_CREATE", 7, 0), ("MAP_MERGE", 7, 3),
                      ("MAP_MERGE", 7, 5)],
                     fr, [(7, 90), (3, 10), (5, 10)])
        check("T5 two merges, id reuse, jumps available -> OK, 3 epochs",
              r["verdict"] == "OK" and r["n_epochs"] == 3,
              "%s epochs=%d" % (r["verdict"], r["n_epochs"]))

        # T6: structural break must win over everything else.
        fr = frames_uniform(10, map_id=9)
        r = scenario(tmp, [("MAP_CREATE", 9, 0)], fr, [(8, 10)])
        check("T6 map id absent from summary and no merge -> BROKEN",
              r["verdict"] == "BROKEN", r["verdict"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILED:
        print("FAILED: %s" % ", ".join(FAILED))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
