#!/usr/bin/env python3
"""Assess whether a run's map history can be rebuilt, and say on what evidence.

The assessment is a pure function of the records (`assess`), so it can be tested
against synthetic counterexamples without a running system; `check_run` only adds
the file-system facts.

Evidence rules (each one is a decision, not a heuristic default)
---------------------------------------------------------------
* A merge is **locatable** only if the frame stream shows a discontinuity inside
  one map id that the splitter is allowed to cut (a backward reference-frame step
  beyond `kf_jump_max`), one cut at most per recorded merge.
* Reference-frame steps are **not** proof of a coordinate change: plain
  re-localisation inside one map produces them. They are used only as *cut
  candidates*, never as evidence that a change happened.
* If the log records more merges than the stream can locate, the run's ONLINE
  numbers are **AMBIGUOUS**: it is not known where the world frame changed, so no
  alignment can be trusted. "Enough cuts" is never used to declare OK.
* Every posed map id must be either in `maps_summary.csv` or explained by a merge
  rename (the absorbing map's id disappears because the survivor inherits the
  discarded map's id).

Verdicts: OK / AMBIGUOUS / BROKEN.
"""
import argparse
import collections
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mapmeta  # noqa: E402


def assess(online, events, summary_ids):
    """Pure assessment of whether the world-frame history can be rebuilt.

    The only accepted evidence that a merge happened at a given frame is the
    frame anchor the event itself carries. A reference-keyframe jump is not
    evidence (re-localisation inside one map produces the same jump) and neither
    is "there were at least as many jumps as merges": that argument cannot tie a
    particular merge to a particular frame, so it is never used to return OK.
    """
    out = {"problems": [], "notes": [], "verdict": "BROKEN"}
    if events is None:
        out["problems"].append("map_events.csv missing")
        return out
    if online is None:
        out["problems"].append("online_poses.csv missing")
        return out

    frames = [r["frame"] for r in online]
    if len(frames) != len(set(frames)):
        out["problems"].append("duplicate frame indices")
    posed = [r for r in online if r["pose_valid"] == 1]
    no_attr = [r["frame"] for r in posed if r["map_id"] < 0 or r["ref_kf"] < 0]
    if no_attr:
        out["problems"].append("posed frames without map/KF: %d" % len(no_attr))

    exact, unanchored = mapmeta.merge_split_frames(events)
    frame_set = set(frames)
    dangling = [f for f in exact if f not in frame_set]
    if dangling:
        out["problems"].append("merge anchored at frames that are not in the "
                               "per-frame record: %s" % dangling[:5])

    eps = mapmeta.align_epochs(online, split_frames=exact)
    out.update({
        "n_frames": len(online), "n_posed": len(posed),
        "n_merge": len(exact) + len(unanchored),
        "n_merge_anchored": len(exact),
        "n_merge_unanchored": len(unanchored),
        "n_epochs": len(eps),
        "events": dict(collections.Counter(e["event"] for e in events)),
        "epochs": [{"map_id": e["map_id"], "n": len(e["rows"]),
                    "frame_lo": e["frame_lo"], "frame_hi": e["frame_hi"],
                    "kf_lo": min(r["ref_kf"] for r in e["rows"]),
                    "kf_hi": max(r["ref_kf"] for r in e["rows"])}
                   for e in eps],
    })

    stream_ids = {e["map_id"] for e in eps}
    absorbing = {e["other_map_id"] for e in events if e["event"] == "MAP_MERGE"}
    unexplained = sorted(stream_ids - set(summary_ids) - absorbing)
    if unexplained:
        out["problems"].append("posed map ids not in maps_summary and not "
                               "explained by a merge rename: %s" % unexplained)
    out["map_ids_stream"] = sorted(stream_ids)
    out["map_ids_summary"] = sorted(summary_ids)

    if out["problems"]:
        out["verdict"] = "BROKEN"
        return out
    if unanchored:
        out["verdict"] = "AMBIGUOUS"
        out["notes"].append(
            "%d of %d merge(s) carry no frame anchor, so where the world frame "
            "changed is not in the record; reference-keyframe jumps are not "
            "accepted as evidence"
            % (len(unanchored), len(unanchored) + len(exact)))
        return out
    out["verdict"] = "OK"
    if exact:
        out["notes"].append("all %d merge(s) anchored at an input frame"
                            % len(exact))
    return out


def check_run(run_dir):
    ev = mapmeta.read_events(run_dir)
    on = mapmeta.read_online(run_dir)
    summary_ids = set()
    summ = os.path.join(run_dir, "maps_summary.csv")
    if os.path.exists(summ):
        with open(summ, newline="") as f:
            for r in csv.DictReader(f):
                summary_ids.add(int(r["map_id"]))
    out = assess(on, ev, summary_ids)
    out["run"] = run_dir
    if not os.path.exists(summ):
        out["problems"].append("maps_summary.csv missing")
        out["verdict"] = "BROKEN"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--root")
    ap.add_argument("--out")
    a = ap.parse_args()

    runs = list(a.run)
    if a.root:
        for d, _, files in os.walk(a.root):
            if "online_poses.csv" in files and "status.txt" in files:
                ok = False
                for line in open(os.path.join(d, "status.txt")):
                    if line.startswith("exit_code="):
                        ok = line.strip().endswith("0")
                if ok:
                    runs.append(d)
    res = [check_run(r) for r in sorted(runs)]
    base = os.path.commonpath([r["run"] for r in res]) if res else "."
    for r in res:
        print("%-52s %-9s epochs=%-3s merges=%-2s anchored=%-2s %s"
              % (os.path.relpath(r["run"], base), r["verdict"],
                 r.get("n_epochs"), r.get("n_merge"), r.get("n_merge_anchored"),
                 ("; ".join(r["problems"] + r["notes"]))[:80]))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        print("json=%s" % a.out)
    c = collections.Counter(r["verdict"] for r in res)
    print("verdicts:", dict(c))
    return 0 if c.get("BROKEN", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
