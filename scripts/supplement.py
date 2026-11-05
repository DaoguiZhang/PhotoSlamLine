#!/usr/bin/env python3
"""Supplementary run set: the cells that must be re-run under v_final_4.

Selection rule (frozen before any supplementary run)
---------------------------------------------------
The two crash fixes sit on
  A. `LocalBundleAdjustment -> MappingOperation::addKeyFrame -> GetKeyLineInfo`
     (reached only by the WithLine tracking path), and
  B. the RGB-D `Frame` constructor -> `ComputeLineStereoFromRGBD`
     (reached only by the WithLine RGB-D path).
`T0` uses `TrackMonocular` / `TrackRGBD`, enters neither, and the only shared
change outside those paths is `mvpMapLines.assign(NL, nullptr)` in the KeyFrame
constructor, which fills a vector T0 never reads. So T0 is reusable, with one
exception: T0 cells whose map history is UNDETERMINED only because the record
has no coordinate epoch must be re-run, because that is a property of the record,
not of the estimator path.

  supplement = { every P1/P2 cell }                    (324 = 2/3 of 486)
             U { T0 cells with UNDETERMINED history }   (from the settlement)

The seven crashed cells are P1/P2 and are therefore already inside the first
set; they are not added twice. Nothing is added outside the original 492 plan.

Usage:
  python3 scripts/supplement.py build|run|status|ledger
"""
import csv
import fcntl
import json
import os
import subprocess
import sys
import time
from collections import Counter

TREE = "/workspace/deliv/cabinet-eval"
BASE = "/workspace/eval_runs/final_runs_v2"          # frozen v_final_2 record
ROOT = "/workspace/eval_runs/supplement_v4"          # new root, never reused
MANIFEST = os.path.join(ROOT, "SUPPLEMENT_MANIFEST.csv")
LOCK = os.path.join(ROOT, ".supplement.lock")
PROGRESS = os.path.join(ROOT, "progress.log")
BIN = "/workspace/eval_builds/v1/entry_rel/eval_frontend"
LIB = "/workspace/eval_builds/v1/lib_rel/libORB_SLAM3.so"
G2O = "/workspace/eval_builds/v1/g2o_rel/lib/libg2o.so"
VOCAB = os.path.join(TREE, "ORB-SLAM3/Vocabulary/ORBvoc.txt")
CFG = os.path.join(TREE, "cfg/ORB_SLAM3")
LINE_MODE = {"T0": "0", "P1": "1", "P2": "2"}
TIMEOUT_S = 1800


def sha(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest() if os.path.exists(p) else ""


def load_settlement():
    p = os.path.join(BASE, "SETTLEMENT.csv")
    if not os.path.exists(p):
        raise SystemExit("run eval/settle.py first: %s missing" % p)
    return {r["id"]: r for r in csv.DictReader(open(p))}


def build():
    base_rows = list(csv.DictReader(open(os.path.join(BASE, "MASTER_MANIFEST.csv"))))
    settle = load_settlement()
    rows = []
    for r in base_rows:
        arm = r["system_arm"]
        if arm == "UPSTREAM":
            continue                     # separate table, already run
        st = settle.get(r["id"], {})
        reasons = []
        if arm in ("P1", "P2"):
            reasons.append("P1/P2: both crash fixes are on the WithLine path")
        if st.get("history") == "AMBIGUOUS" or st.get("metric_status") == \
                "UNDETERMINED_MAP_HISTORY":
            reasons.append("map history UNDETERMINED without a recorded epoch")
        if not reasons:
            continue
        out_dir = os.path.join(ROOT, r["dataset"], r["scene"], r["modality"],
                               arm, "rep%s" % r["repeat"])
        rows.append({
            "id": r["id"], "dataset": r["dataset"], "scene": r["scene"],
            "modality": r["modality"], "system_arm": arm,
            "repeat": r["repeat"], "config": r["config"], "input": r["input"],
            "assoc": r["assoc"], "out_dir": out_dir,
            "prior_status": st.get("metric_status", "NOT_EVALUATED"),
            "prior_exit": st.get("exit_code", ""),
            "prior_history": st.get("history", ""),
            "reason": "; ".join(reasons),
            "build_entry_sha256": sha(BIN), "build_lib_sha256": sha(LIB),
            "build_g2o_sha256": sha(G2O),
        })

    # sanity: no duplicates, nothing invented
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate ids in supplement"
    extra = [i for i in ids if i not in {r["id"] for r in base_rows}]
    assert not extra, "supplement contains ids outside the frozen plan: %s" % extra

    os.makedirs(ROOT, exist_ok=True)
    keys = list(rows[0].keys())
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    c = Counter(r["system_arm"] for r in rows)
    print("supplement manifest: %s" % MANIFEST)
    print("  rows=%d  by arm=%s" % (len(rows), dict(c)))
    print("  T0 kept from the v_final_2 record: %d"
          % sum(1 for r in base_rows if r["system_arm"] == "T0"
                and settle.get(r["id"], {}).get("metric_status") == "OK"))
    return 0


def run():
    lk = open(LOCK, "w")
    try:
        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another supplement scheduler holds %s" % LOCK)
        return 2
    rows = list(csv.DictReader(open(MANIFEST)))
    prog = open(PROGRESS, "a")
    todo = [r for r in rows if not os.path.exists(os.path.join(r["out_dir"],
                                                               "status.txt"))]
    print("to execute: %d of %d" % (len(todo), len(rows)), flush=True)
    # deterministic rotation inside each (dataset, scene, modality) group
    groups = {}
    for r in todo:
        groups.setdefault((r["dataset"], r["scene"], r["modality"]), []).append(r)
    ordered = []
    for gi, k in enumerate(sorted(groups)):
        g = sorted(groups[k], key=lambda r: (r["system_arm"], r["repeat"]))
        n = gi % len(g) if g else 0
        ordered += g[n:] + g[:n]
    n_done = 0
    for r in ordered:
        out = r["out_dir"]
        if os.path.exists(os.path.join(out, "status.txt")):
            continue
        os.makedirs(out, exist_ok=True)
        cfg = os.path.join(CFG, r["config"])
        env = dict(os.environ)
        env.update({"EVAL_ARM": r["system_arm"],
                    "PHOTO_SLAM_LINE_MODE": LINE_MODE[r["system_arm"]],
                    "PHOTO_SLAM_PO_LINE": "1", "PHOTO_SLAM_LBA_LINE": "1",
                    "PHOTO_SLAM_LINE_RESCUE": "0",
                    "PHOTO_SLAM_SLAM_ONLY": "1",
                    "PHOTO_SLAM_DEBUG_LINE_EDGES": "1"})
        if r["dataset"] == "Replica":
            args = [BIN, VOCAB, cfg, "/dev/null", r["input"], out,
                    "replica-" + r["modality"]]
        elif r["modality"] == "mono":
            args = [BIN, VOCAB, cfg, "/dev/null", r["input"], out, "mono"]
        else:
            args = [BIN, VOCAB, cfg, "/dev/null", r["input"], out, "rgbd",
                    r["assoc"]]
        with open(os.path.join(out, "meta.txt"), "w") as f:
            f.write("id=%s\nreason=%s\nprior_status=%s\nprior_exit=%s\n"
                    "cmd=%s\nstart=%s\n"
                    % (r["id"], r["reason"], r["prior_status"], r["prior_exit"],
                       " ".join(args), time.strftime("%Y-%m-%dT%H:%M:%S")))
        t0, note = time.time(), ""
        with open(os.path.join(out, "stdout.log"), "w") as lf:
            try:
                rc = subprocess.call(args, env=env, stdout=lf,
                                     stderr=subprocess.STDOUT, timeout=TIMEOUT_S)
            except subprocess.TimeoutExpired:
                rc, note = 124, "TIMEOUT after %ds" % TIMEOUT_S
        wall = time.time() - t0
        with open(os.path.join(out, "status.txt"), "w") as f:
            f.write("exit_code=%d\nwall_s=%.1f\nend=%s\nnote=%s\n"
                    % (rc, wall, time.strftime("%Y-%m-%dT%H:%M:%S"), note))
        line = "%s exit=%d wall=%.1fs %s\n" % (r["id"], rc, wall, note)
        prog.write(line)
        prog.flush()
        n_done += 1
        print(line.strip(), flush=True)
    print("run() finished, executed=%d" % n_done, flush=True)
    return 0


def status():
    rows = list(csv.DictReader(open(MANIFEST)))
    c = Counter()
    for r in rows:
        st = os.path.join(r["out_dir"], "status.txt")
        if os.path.exists(st):
            code = None
            for line in open(st):
                if line.startswith("exit_code="):
                    code = int(line.split("=")[1])
            c["COMPLETED" if code == 0 else "FAILED_%s" % code] += 1
        elif os.path.isdir(r["out_dir"]):
            c["PARTIAL"] += 1
        else:
            c["PENDING"] += 1
    print("rows=%d" % len(rows))
    for k, v in sorted(c.items()):
        print("  %-12s %d" % (k, v))
    return 0


def ledger():
    rows = list(csv.DictReader(open(MANIFEST)))
    out = os.path.join(ROOT, "ATTEMPT_LEDGER.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "dir", "exit_code", "wall_s", "start", "end", "note",
                    "reason", "build_lib_sha256"])
        for r in rows:
            d = r["out_dir"]
            if not os.path.isdir(d):
                continue
            rec = {"exit_code": "", "wall_s": "", "end": "", "note": "MISSING_STATUS",
                   "start": ""}
            st = os.path.join(d, "status.txt")
            if os.path.exists(st):
                for line in open(st):
                    k, _, v = line.strip().partition("=")
                    if k in rec:
                        rec[k] = v
            mt = os.path.join(d, "meta.txt")
            if os.path.exists(mt):
                for line in open(mt):
                    k, _, v = line.strip().partition("=")
                    if k == "start":
                        rec["start"] = v
            w.writerow([r["id"], d, rec["exit_code"], rec["wall_s"],
                        rec["start"], rec["end"], rec["note"], r["reason"],
                        r["build_lib_sha256"]])
    print("ledger=%s" % out)
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    sys.exit({"build": build, "run": run, "status": status,
              "ledger": ledger}[cmd]())
