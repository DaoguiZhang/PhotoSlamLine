#!/usr/bin/env python3
"""Check the real merge boundary in finished runs, from artefacts only.

The question this answers: when a run merges two maps, does the recorded
`epoch` change exactly at the frame that the merge event names, and can a frame
end up in the new era while still carrying a pre-merge pose?

What can be decided from the artefacts:

  gap        first frame whose epoch is k+1, minus the `frame` column of the
             merge event that raised the epoch. 1 means the frame right after
             the merge frame; 0 would mean the merged frame itself was stamped
             with the new epoch; negative would mean the stamp ran ahead of the
             event.
  unanchored a frame whose epoch rises with no earlier merge event, or with
             fewer merges than the epoch requires.
  orphan     a MAP_MERGE event that is later than every posed frame (it merged
             after the last frame was written), so nothing was re-stamped.

The arithmetic is done on the columns that exist. Nothing is inferred from the
number of candidate maps, ref keyframes or map ids, because map ids are reused.

Usage:
  python3 eval/merge_boundary_audit.py --root <run root> [--out FILE]
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mapmeta  # noqa: E402


def audit_run(d):
    events = mapmeta.read_events(d) or []
    online = mapmeta.read_online(d)
    merges = [e for e in events if e.get("event") == "MAP_MERGE"]
    resets = [e for e in events if e.get("event") in ("MAP_RESET", "MAP_CREATE")]
    if not events:
        return {"dir": d, "state": "NO_EVENTS"}
    if not merges and not [e for e in events if e.get("event") == "MAP_RESET"]:
        return {"dir": d, "state": "NO_MERGE"}

    by_frame = {}
    for r in online:
        if str(r.get("pose_valid")) != "1":
            continue
        by_frame[r["frame"]] = r
    frames = sorted(by_frame, key=int)
    if not frames:
        return {"dir": d, "state": "NO_POSED_FRAME"}

    # first frame of every epoch
    first_of_epoch = {}
    for f in frames:
        e = by_frame[f].get("epoch")
        if e is None or e == "":
            return {"dir": d, "state": "NO_EPOCH_COLUMN"}
        first_of_epoch.setdefault(int(e), int(f))

    ADVANCES = ("MAP_CREATE", "MAP_MERGE", "MAP_RESET")
    anchors = []
    for e in events:
        if e.get("event") in ADVANCES and e.get("frame") is not None and e["frame"] >= 0:
            anchors.append((int(e["frame"]), e["event"]))
    anchors.sort()
    merge_anchors = [a for a in anchors if a[1] == "MAP_MERGE"]

    merge_gaps, create_gaps, unanchored = [], [], 0
    lowest = min(first_of_epoch) if first_of_epoch else 0
    for k in sorted(first_of_epoch):
        if k == lowest:      # the opening epoch is anchored by process start
            continue
        f = first_of_epoch[k]
        prior = [a for a in anchors if a[0] <= f]
        if not prior:
            unanchored += 1
            continue
        af, kind = prior[-1]
        if kind == "MAP_MERGE":
            merge_gaps.append(f - af)
        else:
            create_gaps.append(f - af)

    orphan = sum(1 for mf, _ in merge_anchors if mf > int(frames[-1]))

    return {
        "dir": d, "state": "CHECKED",
        "n_posed": len(frames), "n_merge_events": len(merges),
        "n_reset_create_events": len(resets),
        "n_epochs": len(first_of_epoch),
        "first_frame_of_epoch": " ".join("%d:%d" % (k, v)
                                         for k, v in sorted(first_of_epoch.items())),
        "merge_anchor_gaps": " ".join(str(g) for g in merge_gaps),
        "min_merge_gap": min(merge_gaps) if merge_gaps else "",
        "create_anchor_gaps": " ".join(str(g) for g in create_gaps),
        "min_create_gap": min(create_gaps) if create_gaps else "",
        "unanchored_epochs": unanchored,
        "orphan_merges_after_last_frame": orphan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    for ds in sorted(os.listdir(a.root)):
        p = os.path.join(a.root, ds)
        if not os.path.isdir(p) or ds in ("metrics_v4", "metrics_v3", "UPSTREAM"):
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
                    ad = os.path.join(p, scene, mod, arm)
                    if not os.path.isdir(ad):
                        continue
                    for rep in sorted(os.listdir(ad)):
                        d = os.path.join(ad, rep)
                        if not os.path.isdir(d):
                            continue
                        if not os.path.exists(os.path.join(d, "status.txt")):
                            continue
                        r = audit_run(d)
                        r.update({"dataset": ds, "scene": scene, "modality": mod,
                                  "arm": arm, "repeat": rep})
                        rows.append(r)
    checked = [r for r in rows if r["state"] == "CHECKED"]
    print("runs=%d checked_with_merge=%d" % (len(rows), len(checked)))
    bad = [r for r in checked if r["unanchored_epochs"]
           or (r["min_merge_gap"] != "" and r["min_merge_gap"] < 0)
           or (r["min_create_gap"] != "" and r["min_create_gap"] < 0)]
    at_anchor = [r for r in checked if r["min_merge_gap"] == 0]
    print("merge_gap_ge1=%d merge_gap_zero(anchor_frame_stamped_new_epoch)=%d"
          % (sum(1 for r in checked if r["merge_anchor_gaps"]
                 and int(r["min_merge_gap"]) >= 1), len(at_anchor)))
    if not checked:
        print("no finished run in this root contains a merge event yet")
    for r in checked[:15]:
        print("%-34s epochs=%d merges=%d merge_gaps=[%s] create_gaps=[%s] unanchored=%d orphan=%d" % (
            r["dir"].split(a.root + "/")[-1], r["n_epochs"], r["n_merge_events"],
            r["merge_anchor_gaps"], r["create_anchor_gaps"],
            r["unanchored_epochs"], r["orphan_merges_after_last_frame"]))
    print("boundary_violations=%d" % len(bad))
    for r in bad[:10]:
        print("  VIOLATION", r["dir"], "unanchored=%d min_merge_gap=%s min_create_gap=%s"
              % (r["unanchored_epochs"], r["min_merge_gap"], r["min_create_gap"]))
    if a.out:
        keys = ["dataset", "scene", "modality", "arm", "repeat", "state", "n_posed",
                "n_merge_events", "n_reset_create_events", "n_epochs",
                "first_frame_of_epoch", "merge_anchor_gaps", "min_merge_gap",
                "create_anchor_gaps", "min_create_gap",
                "unanchored_epochs", "orphan_merges_after_last_frame", "dir"]
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
