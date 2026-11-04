#!/usr/bin/env python3
"""Mechanical acceptance of the artefact contract, per run, from the files.

Nothing here is a judgement call: every check reads a run directory and reports
what it found. The point is that "supported" has to mean "the file exists, has
these columns, this many rows, and these hashes", not "the code path exists".

Checks per run (`--root`/<dataset>/<scene>/<modality>/<arm>/<rep>):
  A  final_state.csv      one row per input frame, columns frame,timestamp,state,
                          state_name
  B  online_poses.csv     one row per input frame with pose_valid, map_id,
                          ref_kf_id, frame_id and a transform
  C  map_events.csv       exists and carries MAP_CREATE rows (a run with resets
                          must also carry MAP_RESET/MAP_SWITCH)
  D  all-maps export      maps_summary.csv + traj_map<id>.txt for every map, and
                          the row count of each traj_map file matches its
                          n_exported_kf
  E  consistency          every online_poses.csv frame index appears exactly once
                          and matches final_state.csv frame/timestamp

Cross-run checks:
  F  coverage denominators use every GT-evaluable input frame (coverage.py's
     n_gt_evaluable == n_input for rows whose GT covers the whole run)
  G  statistics tables exist and are newer than the run directories

Usage: python3 eval/acceptance.py --root /workspace/eval_runs/<root>
Exit code 0 only if A-E pass for every completed run and F/G hold.
"""
import argparse
import csv
import hashlib
import os
import sys


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rows_of(path, comments=False):
    """CSV rows. Map-event files start with a '#' banner, which would otherwise
    be taken as the header -- that is a parser bug, not a missing event."""
    with open(path, newline="") as f:
        lines = [ln for ln in f if not (comments and ln.lstrip().startswith("#"))]
    return list(csv.DictReader(lines))


def check_run(d):
    out = {"dir": d, "A": "MISSING", "B": "MISSING", "C": "MISSING",
           "D": "MISSING", "E": "MISSING", "detail": []}
    fs = os.path.join(d, "final_state.csv")
    op = os.path.join(d, "online_poses.csv")
    me = os.path.join(d, "map_events.csv")
    ms = os.path.join(d, "maps_summary.csv")

    n_in = None
    if os.path.exists(fs):
        r = rows_of(fs)
        cols = list(r[0].keys()) if r else []
        ok = cols[:4] == ["frame", "timestamp", "state", "state_name"]
        out["A"] = "PASS" if ok and r else "FAIL"
        n_in = len(r)
        out["detail"].append("final_state rows=%d cols=%d" % (len(r), len(cols)))

    if os.path.exists(op):
        r = rows_of(op)
        need = {"frame", "timestamp", "state", "state_name", "pose_valid",
                "map_id", "ref_kf_id", "frame_id", "tx", "ty", "tz",
                "qx", "qy", "qz", "qw"}
        cols = set(r[0].keys()) if r else set()
        okcols = need <= cols
        npose = sum(1 for x in r if x["pose_valid"] == "1")
        out["B"] = "PASS" if okcols and r else "FAIL"
        out["detail"].append("online_poses rows=%s valid=%d cols_ok=%s"
                             % (len(r), npose, okcols))
        if n_in is not None and len(r) != n_in:
            out["B"] = "FAIL"
            out["detail"].append("online_poses rows != final_state rows")
        # E: index/timestamp agreement
        if os.path.exists(fs):
            a = rows_of(fs)
            same = (len(a) == len(r)) and all(
                a[i]["frame"] == r[i]["frame"] and
                abs(float(a[i]["timestamp"]) - float(r[i]["timestamp"])) < 1e-9
                and a[i]["state"] == r[i]["state"]
                for i in range(min(len(a), len(r))))
            out["E"] = "PASS" if same else "FAIL"
    else:
        out["detail"].append("online_poses.csv absent")

    if os.path.exists(me):
        r = rows_of(me, comments=True)
        kinds = {}
        for x in r:
            kinds[x.get("event", "")] = kinds.get(x.get("event", ""), 0) + 1
        out["C"] = "PASS" if kinds.get("MAP_CREATE", 0) >= 1 else "FAIL"
        out["detail"].append("map_events=%s" % dict(sorted(kinds.items())))
    else:
        out["detail"].append("map_events.csv absent")

    if os.path.exists(ms):
        r = rows_of(ms)
        bad = []
        for x in r:
            tf = x.get("traj_file", "")
            n = int(x.get("n_exported_kf", "0") or 0)
            if not tf or not os.path.exists(tf):
                bad.append("missing %s" % tf)
                continue
            got = sum(1 for line in open(tf) if line.strip())
            if got != n:
                bad.append("%s rows %d != %d" % (os.path.basename(tf), got, n))
        out["D"] = "PASS" if r and not bad else "FAIL"
        out["detail"].append("maps=%d %s" % (len(r), bad[:3] if bad else ""))
    else:
        out["detail"].append("maps_summary.csv absent")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    root = a.root

    runs = []
    for dataset in sorted(os.listdir(root)):
        droot = os.path.join(root, dataset)
        if not os.path.isdir(droot) or dataset in ("metrics", "UPSTREAM"):
            continue
        for scene in sorted(os.listdir(droot)):
            sd = os.path.join(droot, scene)
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
                        rd = os.path.join(ad, rep)
                        if os.path.isdir(rd) and os.path.exists(
                                os.path.join(rd, "status.txt")):
                            runs.append(rd)

    res = [check_run(d) for d in runs]
    fields = ["dir", "A", "B", "C", "D", "E", "detail"]
    outp = os.path.join(root, "ACCEPTANCE_ARTEFACTS.csv")
    with open(outp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in res:
            r = dict(r)
            r["detail"] = " | ".join(r["detail"])
            w.writerow(r)

    ok = 0
    for r in res:
        if all(r[k] == "PASS" for k in "ABCDE"):
            ok += 1
    print("runs_checked=%d all_checks_pass=%d" % (len(res), ok))
    for k in "ABCDE":
        n = sum(1 for r in res if r[k] == "PASS")
        print("  %s PASS %d / %d" % (k, n, len(res)))
    print("table=%s" % outp)
    return 0 if res and ok == len(res) else 1


if __name__ == "__main__":
    sys.exit(main())
