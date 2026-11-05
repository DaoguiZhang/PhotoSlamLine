#!/usr/bin/env python3
"""Build one settled record per plan id from two attempt roots.

The main record (`final_runs_v2`) holds the v_final_2 attempts. The supplement
root holds the later attempts for the cells that v_final_2 did not deliver.
For every plan id exactly one directory wins, by this order and nothing else:

  1. the supplement attempt, if it exists and its status.txt exit code is 0 and
     both `online_poses.csv` and `final_state.csv` are present;
  2. otherwise the base attempt under the same rule;
  3. otherwise the supplement attempt if the directory exists (so a failing
     cell is settled as failing, not as missing);
  4. otherwise the base attempt.

Nothing is copied and nothing is overwritten: the output root contains only
symlinks, a manifest and a metric table. The origin of every id is written to
`source_root` and `reused` so a reader can tell a re-run cell from a reused one.

Usage:
  python3 eval/build_merged_root.py --base <v2> --supp <supp> --out <dir> \
      --metrics-supp <dir> --metrics-base <dir>
"""
import argparse
import csv
import os
import shutil
import sys

COLS_BASE = ["id", "dataset", "scene", "modality", "system_arm", "repeat",
             "config", "input", "assoc", "build_entry_sha256", "build_lib_sha256",
             "build_g2o_sha256", "evaluator_sha256", "gt_path"]


def exit_code(d):
    p = os.path.join(d, "status.txt")
    if not os.path.exists(p):
        return None
    for line in open(p):
        for k in ("exit_code=", "exit="):
            if line.startswith(k):
                try:
                    return int(line.split("=", 1)[1].strip())
                except ValueError:
                    return None
    return None


def complete(d):
    return (os.path.exists(os.path.join(d, "online_poses.csv"))
            and os.path.exists(os.path.join(d, "final_state.csv")))


def usable(d):
    return os.path.isdir(d) and exit_code(d) == 0 and complete(d)


def upstream_state(d):
    """The upstream control rows are judged by their own products.

    They come from the upstream Photo-SLAM binary, which does not write this
    project's `online_poses.csv` / `final_state.csv`; it writes
    `CameraTrajectory_TUM.txt` and `status.txt`. Missing internal CSVs are not
    a crash and must not be reported as one.
    """
    if not d or not os.path.isdir(d):
        return "MISSING"
    traj = os.path.exists(os.path.join(d, "CameraTrajectory_TUM.txt"))
    return "OK" if (exit_code(d) == 0 and traj) else "FAILED"


def read_manifest(path):
    if not os.path.exists(path):
        return []
    return list(csv.DictReader(open(path)))


def row_is_upstream(b, s):
    for r in (b, s):
        if r and (r.get("dataset") == "UPSTREAM"
                  or r.get("system_arm") == "UPSTREAM"):
            return True
    return False


def read_metrics(path):
    if not os.path.exists(path):
        return []
    return list(csv.DictReader(open(path)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--supp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics-supp", required=True)
    ap.add_argument("--metrics-base", required=True)
    ap.add_argument("--upstream-metrics", default=None)
    a = ap.parse_args()

    base = {r["id"]: r for r in read_manifest(os.path.join(a.base, "MASTER_MANIFEST.csv"))}
    supp = {r["id"]: r for r in read_manifest(os.path.join(a.supp, "SUPPLEMENT_MANIFEST.csv"))}
    if not base:
        sys.exit("no base manifest")
    out = a.out
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)

    ids = list(base.keys())
    for i in supp:
        if i not in base:
            ids.append(i)

    rows, chosen = [], {}
    stats = {"supp_ok": 0, "supp_failed": 0, "base_reused_ok": 0,
             "base_reused_failed": 0, "no_dir": 0}
    for i in ids:
        b = base.get(i)
        s = supp.get(i)
        bd = b["out_dir"] if b else None
        sd = s["out_dir"] if s else None
        # A cell that the supplement planned belongs to the supplement attempt,
        # whatever that attempt produced. A later failure must never be replaced
        # by the earlier success: the plan says this cell was re-run, and the
        # re-run's outcome is the record.
        if s is not None:
            src, d = "supplement", sd
        else:
            src, d = "base", bd
        state = (upstream_state(d) if row_is_upstream(b, s) else
                 ("OK" if (d and usable(d)) else
                  ("FAILED" if (d and os.path.isdir(d)) else "MISSING")))
        if src == "supplement":
            stats["supp_ok" if state == "OK" else "supp_failed"] += 1
        else:
            stats["base_reused_ok" if state == "OK" else "base_reused_failed"] += 1
        if state == "MISSING":
            stats["no_dir"] += 1
        row = dict(s or b)
        row["id"] = i
        row["source_root"] = src
        row["role"] = ("upstream" if row_is_upstream(b, s)
                       else ("reused" if src == "base" else "supplement"))
        row["reused"] = int(row["role"] == "reused" and state == "OK")
        row["attempt_state"] = state
        row["attempt_exit"] = "" if not d else exit_code(d)
        if not d:
            rows.append(row)
            continue
        # keep the canonical <dataset>/<scene>/<modality>/<arm>/rep<k> shape
        rep = "rep%s" % row["repeat"]
        rel = os.path.join(row["dataset"], row["scene"], row["modality"],
                           row["system_arm"], rep)
        link = os.path.join(out, rel)
        os.makedirs(os.path.dirname(link), exist_ok=True)
        if not os.path.lexists(link):
            os.symlink(d, link)
        row["out_dir"] = link
        chosen[i] = row
        rows.append(row)

    EXTRA = ["out_dir", "source_root", "role", "reused", "attempt_state",
             "attempt_exit"]
    with open(os.path.join(out, "MASTER_MANIFEST.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS_BASE + EXTRA)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLS_BASE + EXTRA})

    key = lambda r: (r["dataset"], r["scene"], r["modality"], r["arm"], r["repeat"])
    msupp = {key(r): r for r in read_metrics(os.path.join(a.metrics_supp, "ONLINE_PER_RUN.csv"))}
    mbase = {key(r): r for r in read_metrics(os.path.join(a.metrics_base, "ONLINE_PER_RUN.csv"))}
    mdir = os.path.join(out, "metrics_v4")
    os.makedirs(mdir, exist_ok=True)
    mrows, header = [], None
    for i, c in chosen.items():
        k = (c["dataset"], c["scene"], c["modality"], c["system_arm"], "rep%s" % c["repeat"])
        # only the winner's own evaluation pass may supply the numbers
        r = msupp.get(k) if c["source_root"] == "supplement" else mbase.get(k)
        if r is None:
            continue
        if header is None:
            header = list(r.keys())
        mrows.append(r)
    if header:
        with open(os.path.join(mdir, "ONLINE_PER_RUN.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in mrows:
                w.writerow(r)

    # per-era and per-map tables, again only from the winner's own pass
    for name, keycols in (("ONLINE_PER_EPOCH.csv", ("dataset", "scene", "modality",
                                                   "arm", "repeat")),
                          ("FINAL_PER_MAP.csv", ("dataset", "scene", "modality",
                                                 "arm", "repeat", "map_id"))):
        rows_s = read_metrics(os.path.join(a.metrics_supp, name))
        rows_b = read_metrics(os.path.join(a.metrics_base, name))
        grp = lambda rows: {k: [r for r in rows if tuple(r[c] for c in keycols[:5]) == k]
                            for k in {tuple(r[c] for c in keycols[:5]) for r in rows}}
        idx_s, idx_b = grp(rows_s), grp(rows_b)
        keep, hdr = [], None
        for i, c in chosen.items():
            base_key = (c["dataset"], c["scene"], c["modality"],
                        c["system_arm"], "rep%s" % c["repeat"])
            idx = idx_s if c["source_root"] == "supplement" else idx_b
            for r in idx.get(base_key, []):
                if hdr is None:
                    hdr = list(r.keys())
                keep.append(r)
        if hdr:
            with open(os.path.join(mdir, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=hdr, extrasaction="ignore")
                w.writeheader()
                for r in keep:
                    w.writerow(r)

    # upstream rows are evaluated by eval/upstream_eval.py, which writes its own
    # table; carry it into the merged root so settlement can read it here
    up_src = a.upstream_metrics or os.path.join(
        a.base, "metrics_v4", "UPSTREAM_PER_RUN.csv")
    up = read_metrics(up_src)
    if up:
        with open(os.path.join(mdir, "UPSTREAM_PER_RUN.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(up[0].keys()))
            w.writeheader()
            for r in up:
                w.writerow(r)

    print("ids=%d supplement_ok=%d supplement_failed=%d base_reused_ok=%d "
          "base_reused_failed=%d no_dir=%d metrics_rows=%d"
          % (len(ids), stats["supp_ok"], stats["supp_failed"],
             stats["base_reused_ok"], stats["base_reused_failed"],
             stats["no_dir"], len(mrows)))


if __name__ == "__main__":
    main()
