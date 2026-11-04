#!/usr/bin/env python3
"""Readers and interval definitions for the v_final_2 pose products.

Two different intervals, deliberately NOT the same thing
--------------------------------------------------------
**align_epoch** - a maximal run of frames that share ONE world coordinate
  frame. Only a change of world frame starts a new one: a different map, a reset
  (NO_IMAGES_YET after the start), an id that reappears (id reuse), or a merge
  that can be located in the frame stream. A brief tracking loss does NOT start a
  new align epoch: the system did not get a new world frame, so it must not get a
  fresh GT alignment either.

**continuity_interval** - a maximal run of consecutive input frames inside one
  align epoch that are accepted AND correct AND close in time. A brief loss
  breaks continuity but changes no alignment.

Conventions handled explicitly
------------------------------
* `map_events.csv` is `seq,event,map_id,kf_id,reason`. For **MAP_MERGE** the
  third column is the DISCARDED map and the fourth holds the ABSORBING map id.
  Exposed as `other_map_id`; nothing reads that column positionally.
* Map ids are reused: `LoopClosing` ends a merge with
  `pMergeMap->ChangeId(pCurrentMap->GetId())`, so the survivor takes the id of
  the map it absorbed. A constant `map_id` therefore does NOT prove a constant
  world frame.

What a backward reference-frame jump does and does not prove
------------------------------------------------------------
A large backward step in `ref_kf_id` inside one map id is *evidence* that the
frame started using other keyframes - which is what a merge looks like from the
frame stream - but it is not proof of a coordinate change: plain re-localisation
inside one map produces the same signature. So the splitter accepts at most one
such jump per merge that the event log actually recorded, and it never uses a
jump to *create* evidence. If the log records more merges than the stream can
locate, the run is reported AMBIGUOUS by `eval/map_history_check.py` and its
ONLINE numbers are labelled UNDETERMINED instead of being trusted.
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


def read_final_state(run_dir):
    p = os.path.join(run_dir, "final_state.csv")
    if not os.path.exists(p):
        return None
    rows = []
    for d in _csv_rows(p):
        rows.append((int(d["frame"]), float(d["timestamp"]), int(d["state"])))
    rows.sort(key=lambda x: x[0])
    return rows


def read_maps(run_dir):
    """map_id -> {file, n_kf, exists} from maps_summary.csv."""
    out = {}
    summ = os.path.join(run_dir, "maps_summary.csv")
    if not os.path.exists(summ):
        return out
    for r in _csv_rows(summ):
        mid = int(r["map_id"])
        tf = r.get("traj_file", "")
        out[mid] = {"file": tf,
                    "n_kf": int(r.get("n_exported_kf", "0") or 0),
                    "exists": bool(tf) and os.path.exists(tf)}
    return out


def jump_candidates(online, kf_jump_max=25):
    """(magnitude, index) of backward reference-frame steps inside one map id."""
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


def align_epochs(online, n_merge=0, gap_max=1.0, kf_jump_max=25):
    """World-frame eras (see module docstring).

    `n_merge` is the number of MAP_MERGE events recorded for this run; at most
    that many reference-frame discontinuities are accepted as world-frame
    changes. A brief invalid stretch does not split an epoch.
    """
    split_at = ({i for _, i in jump_candidates(online, kf_jump_max)[:n_merge]}
                if n_merge > 0 else set())
    eps = []
    cur = None
    prev_ts = None
    prev_state = None
    for i, r in enumerate(online):
        if r["pose_valid"] != 1:
            prev_state = r["state"]
            continue
        start_new = (cur is None
                     or r["map_id"] != cur["map_id"]
                     or prev_state == NO_IMAGES_YET
                     or (prev_ts is not None
                         and r["ts"] - prev_ts > gap_max)
                     or i in split_at)
        if start_new:
            if cur is not None:
                eps.append(cur)
            cur = {"map_id": r["map_id"], "rows": [],
                   "start_frame": r["frame"], "frame_lo": r["frame"],
                   "frame_hi": r["frame"]}
        cur["rows"].append(r)
        cur["frame_hi"] = r["frame"]
        prev_ts = r["ts"]
        prev_state = r["state"]
    if cur is not None:
        eps.append(cur)
    return eps


def continuity_intervals(epoch, states_by_frame, err_by_ts, tol, rot_tol,
                         gap_max, rot_gate=True):
    """Longest correct run inside ONE align epoch. Never crosses an epoch.

    The walk is over the INPUT FRAME SPAN of the epoch (frame_lo..frame_hi), not
    over the posed rows only: a frame that produced no valid pose is a break,
    even when the two posed frames around it are close in time. Otherwise a 10
    frame loss at 30 fps (0.33 s) would hide under a 0.5 s gap rule and the run
    would be reported as continuous.
    """
    best_s, best_n = 0.0, 0
    cur_start, cur_n = None, 0
    prev_ts = None
    for f in range(epoch["frame_lo"], epoch["frame_hi"] + 1):
        state = states_by_frame.get(f)
        ts = states_by_frame.get(("ts", f))
        e = err_by_ts.get(round(ts, 6)) if ts is not None else None
        ok = (e is not None and state == OK and e[0] <= tol
              and (e[1] <= rot_tol if rot_gate else True))
        if prev_ts is not None and ts is not None and ts - prev_ts > gap_max:
            ok = False
        if ok:
            if cur_n == 0:
                cur_start = ts
            cur_n += 1
            span = ts - cur_start
            if span > best_s or (span == best_s and cur_n > best_n):
                best_s, best_n = span, cur_n
        else:
            cur_n = 0
        prev_ts = ts if ts is not None else prev_ts
    return best_s, best_n
