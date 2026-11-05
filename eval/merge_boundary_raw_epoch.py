#!/usr/bin/env python3
"""Pair every merge event with the pose record by RAW epoch value.

The earlier audit (`merge_boundary_audit.py`) computed
`first frame of the new epoch - frame of the most recent anchor at or before it`.
That gap is non-negative by construction, so it can never fail: it is not
evidence. This script instead takes the epoch value written on the event row
itself and looks for the poses that carry exactly that value, so a stamp that
runs ahead of the event would show up as a negative gap.

For each MAP_MERGE event on a concrete input frame it reports, in this order:

  event_epoch        raw `epoch` column of the event row
  pose_epoch_at_F    raw `epoch` column of the pose recorded for frame F
  last_pre_frame     last posed frame whose raw epoch is < event_epoch
  first_post_frame   first posed frame whose raw epoch == event_epoch
  raw_gap            first_post_frame - F   (negative => the stamp ran ahead)
  era_index          the evaluator's era index that owns first_post_frame
                     (eras are recomputed here; stale online_ep*.txt are ignored)
  F_in_any_era       1 if the anchor frame's timestamp appears in any era table
  next_in_era        the era index that owns the next posed frame after F
  era_n_pairs, era_ate_m, era_ate_if_F_counted_m

Usage: python3 eval/merge_boundary_raw_epoch.py --root <run root> [--out FILE]
"""
import argparse
import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mapmeta  # noqa: E402


def runs_under(root):
    for ds in sorted(os.listdir(root)):
        p = os.path.join(root, ds)
        if not os.path.isdir(p) or ds.startswith("metrics"):
            continue
        for scene in sorted(os.listdir(p)):
            sd = os.path.join(p, scene)
            if not os.path.isdir(sd):
                continue
            for mod in sorted(os.listdir(sd)):
                md = os.path.join(sd, mod)
                if not os.path.isdir(md):
                    continue
                for arm in sorted(os.listdir(md)):
                    ad = os.path.join(md, arm)
                    if not os.path.isdir(ad):
                        continue
                    for rep in sorted(os.listdir(ad)):
                        d = os.path.join(ad, rep)
                        if os.path.isdir(d) and os.path.exists(
                                os.path.join(d, "status.txt")):
                            yield ds, scene, mod, arm, rep, d


def raw_events(d):
    """Event rows with the raw `epoch` column kept.

    `mapmeta.read_events` normalises the row and does not carry `epoch`, so the
    epoch value written by the event log is read here from the file itself.
    """
    p = os.path.join(d, "map_events.csv")
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p):
        if line.startswith("#") or not line.strip():
            continue
        out.append(line.rstrip("\n"))
    if not out:
        return []
    rd = csv.DictReader(out)
    rows = []
    for r in rd:
        def as_int(k):
            v = (r.get(k) or "").strip()
            try:
                return int(v)
            except ValueError:
                return None
        rows.append({"seq": as_int("seq"), "event": (r.get("event") or "").strip(),
                     "frame": as_int("frame"), "epoch": as_int("epoch"),
                     "map_id": as_int("map_id")})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/eval_runs/supplement_v4")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = []
    for ds, scene, mod, arm, rep, d in runs_under(a.root):
        events = mapmeta.read_events(d) or []
        raw = raw_events(d)
        merges = [e for e in raw if e["event"] == "MAP_MERGE"
                  and e.get("frame") is not None and e["frame"] >= 0]
        if not merges:
            continue
        online = mapmeta.read_online(d)
        posed = [r for r in online if str(r.get("pose_valid")) == "1"]
        if not posed:
            continue
        exact, _ = mapmeta.merge_split_frames(events)
        eps = mapmeta.align_epochs(online, split_frames=exact)
        era_of = {}
        for k, e in enumerate(eps):
            for r in e["rows"]:
                era_of[int(r["frame"])] = k

        for m in merges:
            E = int(m.get("epoch") or -1)
            F = int(m["frame"])
            if E < 0:
                continue
            pre = [r for r in posed if int(r["epoch"]) < E]
            post = [r for r in posed if int(r["epoch"]) == E]
            if not post:
                continue
            f_row = next((r for r in posed if int(r["frame"]) == F), None)
            first_post = int(post[0]["frame"])
            last_pre = int(pre[-1]["frame"]) if pre else -1
            nxt = next((r for r in posed if int(r["frame"]) > F), None)

            k = era_of.get(first_post)
            owner = first_post
            if k is None and nxt is not None:
                # the frame that carries the new epoch is the anchor frame and is
                # excluded from every era, so the era index for this epoch is the
                # era that owns the next posed frame
                k = era_of.get(int(nxt["frame"]))
                owner = int(nxt["frame"])
            era_epochs = sorted({int(r["epoch"]) for r in eps[k]["rows"]}) if k is not None else []
            era_epoch_ok = int(era_epochs == [E]) if era_epochs else 0
            j = os.path.join(d, "eval", "online_ep%02d.json" % k) if k is not None else ""
            meta = {}
            if j and os.path.exists(j):
                meta = json.load(open(j))
            n_rows = len(eps[k]["rows"]) if k is not None else None
            ate = (meta.get("ate_se3") or {}).get("rmse")
            n_pairs = meta.get("n_pairs")

            # would the anchor frame have been inside an era if not excluded?
            f_in_any = int(f_row is not None and int(f_row["frame"]) in era_of)
            rows.append({
                "dataset": ds, "scene": scene, "modality": mod, "arm": arm,
                "repeat": rep,
                "event_seq": m["seq"], "event_epoch": E, "event_frame": F,
                "pose_epoch_at_F": "" if f_row is None else f_row["epoch"],
                "pose_map_id_at_F": "" if f_row is None else f_row["map_id"],
                "last_pre_frame": last_pre, "first_post_frame": first_post,
                "raw_gap": first_post - F,
                "era_index": "" if k is None else k,
                "era_owner_frame": "" if k is None else owner,
                "era_epochs": " ".join(str(x) for x in era_epochs),
                "era_epoch_ok": era_epoch_ok,
                "era_first_frame": "" if k is None else int(eps[k]["rows"][0]["frame"]),
                "era_n_frames": "" if n_rows is None else n_rows,
                "era_n_pairs": "" if n_pairs is None else n_pairs,
                "era_ate_m": "" if ate is None else round(ate, 5),
                "F_in_any_era": f_in_any,
                "next_valid_frame": "" if nxt is None else nxt["frame"],
                "next_valid_era": "" if nxt is None else era_of.get(int(nxt["frame"]), ""),
                "dir": d,
            })

    zero = [r for r in rows if r["raw_gap"] == 0]
    print("merge_events_paired=%d  raw_gap_zero=%d  raw_gap_negative=%d"
          % (len(rows), len(zero), sum(1 for r in rows if r["raw_gap"] < 0)))
    print("%-34s %-6s %-6s %-6s %-6s %-8s %-5s %-6s %-5s %-6s %-5s"
          % ("run", "evt_eq", "at_F", "lastp", "first", "raw_gap", "era", "F_in",
             "nxt", "nxt_era", "F_cnt"))
    for r in sorted(rows, key=lambda x: (x["raw_gap"], x["dir"])):
        print("%-34s %-6s %-6s %-6s %-6s %-8s %-5s %-6s %-5s %-6s %-5s" % (
            "%s/%s/%s %s F=%s" % (r["scene"], r["modality"], r["arm"], r["repeat"],
                                  r["event_frame"]),
            r["event_epoch"], r["pose_epoch_at_F"], r["last_pre_frame"],
            r["first_post_frame"], r["raw_gap"], r["era_index"], r["F_in_any_era"],
            r["next_valid_frame"], r["next_valid_era"],
            r["era_ate_m"] if r["era_ate_m"] != "" else "—"))
    bad = [r for r in rows if r["raw_gap"] < 0 or r["F_in_any_era"]
           or r["next_valid_era"] != r["era_index"] or not r["era_epoch_ok"]]
    print("era_holds_exactly_one_raw_epoch=%d/%d"
          % (sum(1 for r in rows if r["era_epoch_ok"]), len(rows)))
    print("mismatched=%d" % len(bad))
    for r in bad:
        print("  MISMATCH", r["dir"], "raw_gap=%s F_in_any=%s era=%s next_era=%s"
              % (r["raw_gap"], r["F_in_any_era"], r["era_index"], r["next_valid_era"]))
    if a.out:
        keys = ["dataset", "scene", "modality", "arm", "repeat", "event_seq",
                "event_epoch", "event_frame", "pose_epoch_at_F", "pose_map_id_at_F",
                "last_pre_frame", "first_post_frame", "raw_gap", "era_index",
                "era_owner_frame", "era_epochs", "era_epoch_ok",
                "era_first_frame", "era_n_frames", "era_n_pairs", "era_ate_m",
                "F_in_any_era", "next_valid_frame", "next_valid_era", "dir"]
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
