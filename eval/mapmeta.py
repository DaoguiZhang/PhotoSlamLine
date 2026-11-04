#!/usr/bin/env python3
"""Shared reader for the v_final_2 pose products.

Two conventions are handled explicitly rather than inferred:

1. `map_events.csv` is `seq,event,map_id,kf_id,reason`. For **MAP_MERGE** the
   third column is the DISCARDED map and the fourth (`kf_id`) holds the
   ABSORBING map id -- a merge has two maps and the schema has one id field. The
   parser therefore exposes `map_id`, `other_map_id` (the absorbing map for
   MAP_MERGE), and `kf_id` (only meaningful for MAP_CREATE, where it is the
   initial keyframe id). Anything reading the raw file positionally would read a
   map id as a keyframe id.

2. Map ids are **reused**. `LoopClosing` finishes a merge with
   `pMergeMap->ChangeId(pCurrentMap->GetId())`, so the surviving map takes the id
   of the map that was absorbed. An `epoch` is therefore not "a run of equal
   map_id": it is a maximal run of consecutive frames for which the (map_id,
   reference keyframe) pair is consistent, additionally split when a map id
   reappears after being absent, and when the state drops back to
   NO_IMAGES_YET after the start. Epochs are the unit of alignment; nothing is
   ever aligned, RPE'd or searched for continuity across two epochs.
"""
import csv
import os

OK = 2
NO_IMAGES_YET = 0


def _csv_rows(path, comments=False):
    with open(path, newline="") as f:
        lines = [ln for ln in f
                 if not (comments and ln.lstrip().startswith("#"))]
    return list(csv.DictReader(lines))


def read_events(run_dir):
    p = os.path.join(run_dir, "map_events.csv")
    if not os.path.exists(p):
        return None
    out = []
    for r in _csv_rows(p, comments=True):
        ev = (r.get("event") or "").strip()
        if not ev:
            continue
        row = {"seq": int(r["seq"]), "event": ev,
               "map_id": int(r["map_id"]), "kf_or_target": int(r["kf_id"]),
               "reason": r.get("reason", "")}
        # explicit reinterpretation, see module docstring
        if ev == "MAP_MERGE":
            row["other_map_id"] = row["kf_or_target"]   # absorbing map
            row["kf_id"] = None
        else:
            row["other_map_id"] = None
            row["kf_id"] = (row["kf_or_target"]
                            if ev == "MAP_CREATE" else None)
        out.append(row)
    return out


def read_online(run_dir):
    """Per-input-frame online poses -> list of dicts, ordered by frame."""
    p = os.path.join(run_dir, "online_poses.csv")
    if not os.path.exists(p):
        return None
    rows = []
    for r in _csv_rows(p):
        rows.append({
            "frame": int(r["frame"]),
            "ts": float(r["timestamp"]),
            "state": int(r["state"]),
            "state_name": r["state_name"],
            "pose_valid": int(r["pose_valid"]),
            "map_id": int(r["map_id"]),
            "ref_kf": int(r["ref_kf_id"]),
            "frame_id": int(r["frame_id"]),
            "t": [float(r["tx"]), float(r["ty"]), float(r["tz"])],
            "q": [float(r["qx"]), float(r["qy"]), float(r["qz"]),
                  float(r["qw"])],
        })
    rows.sort(key=lambda x: x["frame"])
    return rows


def read_map_kfs(run_dir):
    """map_id -> trajectory file and row count, from maps_summary.csv."""
    out = {}
    summ = os.path.join(run_dir, "maps_summary.csv")
    if not os.path.exists(summ):
        return out
    for r in _csv_rows(summ):
        mid = int(r["map_id"])
        tf = r.get("traj_file", "")
        n = int(r.get("n_exported_kf", "0") or 0)
        out[mid] = {"file": tf, "n_kf": n,
                    "exists": bool(tf) and os.path.exists(tf)}
    return out


def jump_candidates(online, kf_jump_max=25):
    """Frames where the reference keyframe id steps backwards beyond kf_jump_max.

    Returned as (magnitude, index) pairs sorted by magnitude, so a caller can
    take the largest ones as candidate world-frame changes.
    """
    cands = []
    prev = None
    for i, r in enumerate(online):
        if r["pose_valid"] != 1:
            prev = None
            continue
        if prev is not None and prev["map_id"] == r["map_id"]:
            d = prev["ref_kf"] - r["ref_kf"]
            if d > kf_jump_max:
                cands.append((d, i))
        prev = r
    cands.sort(reverse=True)
    return cands


def epochs(online, gap_max=0.5, kf_jump_max=25, max_jumps=0):
    """Split the per-frame stream into alignable epochs.

    A new epoch starts when any of these holds:
      * the frame has no valid pose (invalid rows are excluded from every epoch
        and are never smoothed into the next one),
      * map_id differs from the previous posed frame,
      * the state returns to NO_IMAGES_YET after the start (map reset),
      * the timestamp gap exceeds gap_max,
      * one of the `max_jumps` largest BACKWARD reference-keyframe jumps (each
        beyond kf_jump_max) occurs.

The last rule exists because `LoopClosing` ends a merge with
    `pMergeMap->ChangeId(pCurrentMap->GetId())`: the survivor inherits the id of
the map it absorbed, so `map_id` can be constant across a world-frame change and
the merge would otherwise hide inside one epoch. Small backward moves are normal
inside one map (re-localisation against an older keyframe), and splitting on all
of them would fragment the run into dozens of epochs, each aligned on its own --
which inflates accuracy. The caller therefore passes the number of merges the
event log actually recorded, and only that many split points are taken (largest
first). With no merge, no jump split happens at all.
    """
    cands = jump_candidates(online, kf_jump_max)
    split_at = ({i for _, i in cands[:max_jumps]} if max_jumps > 0 else set())
    eps = []
    cur = None
    seen = set()
    prev_ts = None
    prev_state = None

    def close():
        nonlocal cur
        if cur is not None:
            eps.append(cur)
            seen.add(cur["map_id"])
            cur = None

    for i, r in enumerate(online):
        if r["pose_valid"] != 1:
            # an invalid frame closes the epoch: a pose is never interpolated
            # across a frame that had none
            close()
            prev_ts = r["ts"]
            prev_state = r["state"]
            continue
        start_new = (cur is None
                     or r["map_id"] != cur["map_id"]
                     or prev_state == NO_IMAGES_YET
                     or (prev_ts is not None
                         and r["ts"] - prev_ts > gap_max)
                     or i in split_at)
        if start_new:
            close()
            cur = {"map_id": r["map_id"], "rows": []}
        cur["rows"].append(r)
        prev_ts = r["ts"]
        prev_state = r["state"]
    close()
    return eps
