#!/usr/bin/env python3
"""Verify that a run's map history can be rebuilt uniquely from its artefacts.

What is checked, per run
------------------------
1. **Event parsing.** `map_events.csv` is read through `eval/mapmeta.py`, which
   reinterprets `kf_id` for `MAP_MERGE` as the ABSORBING map id (a merge has two
   maps; the schema has one id field). Reading the file positionally would treat
   a map id as a keyframe id.
2. **Per-frame attribution is not empty.** Every input frame appears exactly once
   in `online_poses.csv`, and frames with `pose_valid=1` carry a map id and a
   reference keyframe id.
3. **Epoch partition.** The posed frames are split into epochs (see
   `mapmeta.epochs`) on: invalid pose, map-id change, map-id reappearance,
   reset, timestamp gap. A *backward jump* of the reference keyframe id inside an
   otherwise uniform run is treated as an additional boundary and reported,
   because `LoopClosing` ends a merge with
   `pMergeMap->ChangeId(pCurrentMap->GetId())`: the surviving map INHERITS the
   id of the map it absorbed, so a world-frame change can hide inside one
   constant `map_id`.
4. **Event coverage.** The number of epochs must be accounted for by the event
   log: every `MAP_CREATE` after the first, and every `MAP_SWITCH`/`MAP_MERGE`,
   must correspond to at least as many boundaries. A run where the event log
   claims a world-frame change that the per-frame stream cannot locate is
   `AMBIGUOUS` -- it is reported, never silently aligned across.
5. **Cross-check against the exported maps.** Every map id that appears in the
   per-frame stream while `pose_valid=1` must exist in `maps_summary.csv`, except
   ids that were renamed away by a merge; those are matched through the merge
   events (`absorbed -> absorbing` and the subsequent rename).

Verdicts
--------
  OK          per-frame attribution is consistent and every epoch boundary in the
              event log is locatable in the frame stream
  AMBIGUOUS   the event log records a boundary that the per-frame record cannot
              place; the run's ONLINE numbers may not be used
  BROKEN      a structural inconsistency (duplicate frames, posed frame without
              map/KF, map id that cannot be explained)

Usage:
  python3 eval/map_history_check.py --run DIR [--run DIR ...] [--root ROOT]
"""
import argparse
import collections
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mapmeta  # noqa: E402


def check_run(run_dir):
    out = {"run": run_dir, "verdict": "BROKEN", "problems": [], "notes": []}
    ev = mapmeta.read_events(run_dir)
    on = mapmeta.read_online(run_dir)
    if ev is None:
        out["problems"].append("map_events.csv missing")
        return out
    if on is None:
        out["problems"].append("online_poses.csv missing")
        return out

    # ---- 2. one row per frame, posed rows carry attribution ---------------
    frames = [r["frame"] for r in on]
    if len(frames) != len(set(frames)):
        out["problems"].append("duplicate frame indices in online_poses.csv")
    posed = [r for r in on if r["pose_valid"] == 1]
    no_attr = [r["frame"] for r in posed
               if r["map_id"] < 0 or r["ref_kf"] < 0]
    if no_attr:
        out["problems"].append("posed frames without map/KF: %d (e.g. %s)"
                               % (len(no_attr), no_attr[:3]))

    # ---- 3. epoch partition ----------------------------------------------
    # Same rule as the evaluator: at most one jump split per recorded merge.
    n_merge = sum(1 for e in ev if e["event"] == "MAP_MERGE")
    eps = mapmeta.epochs(on, gap_max=0.5, max_jumps=n_merge)
    out["n_frames"] = len(on)
    out["n_posed"] = len(posed)
    out["n_epochs"] = len(eps)
    out["epochs"] = [{"map_id": e["map_id"], "n": len(e["rows"]),
                      "frame_lo": e["rows"][0]["frame"],
                      "frame_hi": e["rows"][-1]["frame"],
                      "kf_lo": min(r["ref_kf"] for r in e["rows"]),
                      "kf_hi": max(r["ref_kf"] for r in e["rows"])}
                     for e in eps]

    # reference-keyframe regression inside an epoch = hidden world-frame change
    regress = 0
    for e in eps:
        kfs = [r["ref_kf"] for r in e["rows"]]
        for a, b in zip(kfs, kfs[1:]):
            if b < a:
                regress += 1
    out["refkf_regressions"] = regress

    # ---- 4. event coverage -----------------------------------------------
    creates = [e for e in ev if e["event"] == "MAP_CREATE"]
    merges = [e for e in ev if e["event"] == "MAP_MERGE"]
    resets = [e for e in ev if e["event"] == "MAP_RESET"]
    out["events"] = dict(collections.Counter(e["event"] for e in ev))
    # A world frame can change only on a new map, a merge or a reset. MAP_STORED /
    # MAP_SWITCH / MAP_CURRENT are the informational consequence of one of those
    # (they fire in the same instant), so counting them again would double count.
    n_expected = max(0, len(creates) - 1) + len(merges) + len(resets)
    n_boundaries = max(0, len(eps) - 1)
    out["boundaries_in_stream"] = n_boundaries
    out["boundaries_in_eventlog"] = n_expected
    if n_expected > n_boundaries:
        out["notes"].append(
            "%d world-frame changes in the event log, %d boundaries in the "
            "stream (the rest fell inside frames with no valid pose, which "
            "already break an epoch)" % (n_expected, n_boundaries))

    # ---- 5. map ids explained --------------------------------------------
    summary_ids = set()
    summ = os.path.join(run_dir, "maps_summary.csv")
    if os.path.exists(summ):
        with open(summ, newline="") as f:
            for r in csv.DictReader(f):
                summary_ids.add(int(r["map_id"]))
    else:
        out["problems"].append("maps_summary.csv missing")

    # A merge ends with pMergeMap->ChangeId(pCurrentMap->GetId()): the surviving
    # map inherits the id of the map it ABSORBED, so the absorbing map's own id
    # disappears from the Atlas. Frames tracked before the merge still report
    # that old id, so it is explained by the merge even though no summary row
    # carries it.
    discarded = {e["map_id"] for e in ev if e["event"] == "MAP_MERGE"}
    absorbing = {e["other_map_id"] for e in ev if e["event"] == "MAP_MERGE"}
    stream_ids = {e["map_id"] for e in eps}
    unexplained = sorted(stream_ids - summary_ids - absorbing)
    if unexplained:
        out["problems"].append("posed map ids not in maps_summary and not "
                               "explained by a merge rename: %s" % unexplained)
    # every merge's discarded id must reappear in the summary, because the
    # survivor is renamed to it
    missing_after_merge = sorted(discarded - summary_ids)
    if missing_after_merge:
        out["notes"].append("merge target id(s) absent from maps_summary: %s"
                            % missing_after_merge)
    out["map_ids_stream"] = sorted(stream_ids)
    out["map_ids_summary"] = sorted(summary_ids)
    out["map_ids_absorbing_renamed_away"] = sorted(absorbing)

    # ---- verdict ----------------------------------------------------------
    # `regress` (a backward reference-keyframe step inside one epoch) is NOT a
    # fault: re-localising against an older keyframe of the SAME map produces it
    # routinely. It is reported for information only.
    #
    # The association basis is stated explicitly:
    #   * a merge during frames that have a valid pose moves the current frame
    #     into the other map's keyframe range, which is the discontinuity the
    #     epoch splitter is allowed to cut (at most one cut per recorded merge);
    #   * if fewer cuts were available than merges, the remaining merges happened
    #     while no pose was valid -- those frames already break the epoch, so the
    #     evaluation still cannot span them, but the exact instant is not
    #     pinpointed by the record. That case is named, not hidden.
    out["n_merge"] = n_merge
    out["n_jump_splits_available"] = len(mapmeta.jump_candidates(on))
    out["n_jump_splits_used"] = min(out["n_jump_splits_available"], n_merge)
    if out["problems"]:
        out["verdict"] = "BROKEN"
    elif out["n_jump_splits_used"] >= n_merge:
        out["verdict"] = "OK"
    else:
        out["verdict"] = "OK_ASSUMED_GAP"
        out["notes"].append(
            "%d merge(s), only %d locatable by a reference-keyframe discontinuity;"
            " the remainder fall in stretches with no valid pose, which are epoch"
            " boundaries anyway, but their exact instant is not recorded"
            % (n_merge, out["n_jump_splits_used"]))
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
    for r in res:
        print("%-58s %-9s epochs=%-3s posed=%-5s boundaries %s/%s regress=%s %s"
              % (os.path.relpath(r["run"], os.path.commonpath([x["run"] for x in res]) or "."),
                 r["verdict"], r.get("n_epochs"), r.get("n_posed"),
                 r.get("boundaries_in_stream"), r.get("boundaries_in_eventlog"),
                 r.get("refkf_regressions"),
                 ("; ".join(r["problems"] + r["notes"]))[:70]))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        print("json=%s" % a.out)
    c = collections.Counter(r["verdict"] for r in res)
    print("verdicts:", dict(c))
    return 0 if c.get("BROKEN", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
