#!/usr/bin/env python3
"""Counterexamples for the map-history reconstruction.

Every test builds a REAL run directory on disk (map_events.csv,
online_poses.csv, maps_summary.csv) and goes through the same code the pipeline
uses. The purpose is to pin down the cases where an inference would be tempting
and wrong:

  * a merge that carries no frame anchor cannot be located, no matter how many
    reference-keyframe jumps exist ("as many jumps as merges" is not evidence);
  * a reference-keyframe jump inside one map is not a world-frame change;
  * a brief loss or a long time gap is not a new world frame, so it must not
    grant a fresh GT alignment (align) even though it breaks continuity.

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
    print("%-62s %s %s" % (name, "PASS" if cond else "FAIL", detail))
    if not cond:
        FAILED.append(name)


def make_run(tmp, events, frames, summary_maps, with_frame_column=True):
    """events: (event, map_id, kf_or_target[, frame]) tuples."""
    d = tempfile.mkdtemp(dir=tmp)
    with open(os.path.join(d, "map_events.csv"), "w") as f:
        f.write("# synthetic\n")
        f.write("seq,event,map_id,kf_id,reason%s\n"
                % (",frame" if with_frame_column else ""))
        for i, ev in enumerate(events):
            name, mid, kf = ev[0], ev[1], ev[2]
            if with_frame_column:
                fr = ev[3] if len(ev) > 3 else -1
                f.write("%d,%s,%d,%d,synthetic,%d\n" % (i, name, mid, kf, fr))
            else:
                f.write("%d,%s,%d,%d,synthetic\n" % (i, name, mid, kf))
    with open(os.path.join(d, "online_poses.csv"), "w") as f:
        f.write("frame,timestamp,state,state_name,pose_valid,map_id,ref_kf_id,"
                "frame_id,tx,ty,tz,qx,qy,qz,qw\n")
        for r in frames:
            f.write("%d,%.6f,%d,%s,%d,%d,%d,%d,0,0,0,0,0,0,1\n"
                    % (r["frame"], r["ts"], r["state"], r["name"],
                       r["valid"], r["map_id"], r["ref_kf"], r["frame"]))
    with open(os.path.join(d, "maps_summary.csv"), "w") as f:
        f.write("map_id,is_in_use,is_bad,n_keyframes,n_mappoints,n_maplines,"
                "n_exported_kf,traj_file\n")
        for mid, nkf in summary_maps:
            p = os.path.join(d, "traj_map%d.txt" % mid)
            with open(p, "w") as g:
                for i in range(nkf):
                    g.write("%.6f 0 0 0 0 0 0 1\n" % (i * 0.033))
            f.write("%d,0,0,%d,0,0,%d,%s\n" % (mid, nkf, nkf, p))
    return d


def frames_uniform(n=60, map_id=1, kf0=10, state=2, valid=1, dt=0.033):
    return [{"frame": i, "ts": i * dt, "state": state,
             "name": "OK" if state == 2 else "RECENTLY_LOST",
             "valid": valid, "map_id": map_id, "ref_kf": kf0 + i}
            for i in range(n)]


def run_case(tmp, events, frames, summary, with_frame_column=True):
    d = make_run(tmp, events, frames, summary, with_frame_column)
    return mhc.check_run(d)


def main():
    tmp = tempfile.mkdtemp(prefix="maphist_")
    try:
        # T1. Every frame valid, one merge, no anchor: unlocatable => AMBIGUOUS.
        r = run_case(tmp, [("MAP_CREATE", 1, 0), ("MAP_MERGE", 1, 2)],
                     frames_uniform(), [(1, 60)], with_frame_column=False)
        check("T1 unanchored merge, all frames valid -> AMBIGUOUS",
              r["verdict"] == "AMBIGUOUS", r["verdict"])

        # T2. A big backward reference-keyframe jump inside one map, no merge.
        # Not evidence of a coordinate change: one epoch.
        fr = frames_uniform(60)
        for i in range(30, 60):
            fr[i]["ref_kf"] = 5 + i
        r = run_case(tmp, [("MAP_CREATE", 1, 0)], fr, [(1, 60)], False)
        check("T2 backward jump, no merge -> one epoch, OK",
              r["verdict"] == "OK" and r["n_epochs"] == 1,
              "%s epochs=%d" % (r["verdict"], r["n_epochs"]))

        # T3. Merge with the same map id before and after, no jump, no anchor.
        r = run_case(tmp, [("MAP_CREATE", 4, 0), ("MAP_MERGE", 4, 9)],
                     frames_uniform(map_id=4), [(4, 60)], False)
        check("T3 merge, same map id, no anchor -> AMBIGUOUS",
              r["verdict"] == "AMBIGUOUS", r["verdict"])

        # T4. Brief loss then the same world frame: one alignment era, but the
        # continuity run is broken (checked in T4b).
        fr = frames_uniform(90)
        for i in range(40, 50):
            fr[i]["state"] = 3
            fr[i]["name"] = "RECENTLY_LOST"
            fr[i]["valid"] = 0
        r = run_case(tmp, [("MAP_CREATE", 1, 0)], fr, [(1, 90)], False)
        check("T4 brief loss, same world frame -> 1 epoch, OK",
              r["verdict"] == "OK" and r["n_epochs"] == 1,
              "%s epochs=%d" % (r["verdict"], r["n_epochs"]))
        on = mapmeta.read_online(r["run"])
        eps = mapmeta.align_epochs(on)
        states = {}
        for f in fr:
            states[f["frame"]] = f["state"]
            states[("ts", f["frame"])] = f["ts"]
        errs = {round(f["ts"], 6): (0.01, 0.5) for f in fr}
        _, best_n = mapmeta.continuity_intervals(
            eps[0], states, errs, tol=0.20, rot_tol=5.0, gap_max=0.5)
        check("T4b continuity broken by the loss (40 frames, not 90)",
              best_n == 40, "longest=%d" % best_n)

        # T5. Two merges WITH frame anchors: exact splits, no guessing.
        fr = frames_uniform(90, map_id=7)
        r = run_case(tmp,
                     [("MAP_CREATE", 7, 0, 0), ("MAP_MERGE", 7, 3, 30),
                      ("MAP_MERGE", 7, 5, 60)],
                     fr, [(7, 90), (3, 10), (5, 10)])
        check("T5 two anchored merges -> OK, 3 epochs",
              r["verdict"] == "OK" and r["n_epochs"] == 3,
              "%s epochs=%d" % (r["verdict"], r["n_epochs"]))

        # T6. A map id that nothing explains is a structural break.
        r = run_case(tmp, [("MAP_CREATE", 9, 0)], frames_uniform(10, map_id=9),
                     [(8, 10)], False)
        check("T6 unexplained map id -> BROKEN", r["verdict"] == "BROKEN",
              r["verdict"])

        # T7. THE REGRESSION THIS SUITE EXISTS FOR: one unanchored merge plus two
        # backward reference-keyframe jumps. A count-based rule ("two jumps for
        # one merge") would say OK; it must be AMBIGUOUS because no jump can be
        # tied to the merge instant.
        fr = frames_uniform(90, map_id=3)
        for i in range(30, 60):
            fr[i]["ref_kf"] = 200 + (i - 30)
        for i in range(60, 90):
            fr[i]["ref_kf"] = 40 + (i - 60)
        r = run_case(tmp, [("MAP_CREATE", 3, 0), ("MAP_MERGE", 3, 1)],
                     fr, [(3, 90), (1, 10)], with_frame_column=False)
        check("T7 unanchored merge + two jumps -> AMBIGUOUS (not OK)",
              r["verdict"] == "AMBIGUOUS",
              "%s epochs=%d" % (r["verdict"], r["n_epochs"]))

        # T8. A long time gap inside one map must NOT create a new alignment.
        fr = frames_uniform(60)
        for i in range(30, 60):
            fr[i]["ts"] = 30 * 0.033 + 5.0 + (i - 30) * 0.033   # 5 s gap
        r = run_case(tmp, [("MAP_CREATE", 1, 0)], fr, [(1, 60)], False)
        check("T8 long time gap, same map -> still one align epoch",
              r["verdict"] == "OK" and r["n_epochs"] == 1,
              "%s epochs=%d" % (r["verdict"], r["n_epochs"]))
        on = mapmeta.read_online(r["run"])
        eps = mapmeta.align_epochs(on)
        states = {}
        for f in fr:
            states[f["frame"]] = f["state"]
            states[("ts", f["frame"])] = f["ts"]
        errs = {round(f["ts"], 6): (0.01, 0.5) for f in fr}
        _, n_frames = mapmeta.continuity_intervals(
            eps[0], states, errs, tol=0.20, rot_tol=5.0, gap_max=0.5)
        check("T8b the gap breaks continuity instead (30 frames)",
              n_frames == 30, "longest=%d" % n_frames)
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
