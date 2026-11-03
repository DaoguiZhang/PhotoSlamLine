#!/usr/bin/env python3
"""Plan-item table + attempt table, derived only from run artefacts.

Two tables, two different things (they must not be conflated):

  PLAN_ITEMS.csv   one row per planned cell (dataset, scene, modality, arm,
                   repeat). 492 rows. This is the experiment count.
  ATTEMPTS.csv     one row per DIRECTORY that was actually created, i.e. per
                   execution attempt, including `_r1` retries. The number of
                   status.txt files is an attempt count, never an experiment
                   count.

Status resolution for a plan item:
  COMPLETED          the original attempt exited 0
  COMPLETED_RETRY    the original attempt failed and an `_r1` attempt exited 0
  FAILED             every attempt failed (max 1 retry by policy)
  PARTIAL            a directory exists with no status.txt (interrupted)
  BLOCKED/PENDING    as in the manifest

Usage: python3 eval/accounting.py [--root /workspace/eval_runs/v2]
"""
import argparse
import csv
import os
import re
from collections import Counter, defaultdict

ARMS = {"T0", "P1", "P2", "UP"}
REP_RE = re.compile(r"^rep(\d+)(_r(\d+))?$")


def read_status(d):
    st = os.path.join(d, "status.txt")
    out = {"dir": d, "has_status": False, "exit_code": "", "wall_s": "",
           "note": "", "end": ""}
    if not os.path.exists(st):
        return out
    out["has_status"] = True
    for line in open(st):
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if k in out:
            out[k] = v
    return out


def scan(root):
    """Walk <root>/<dataset>/<scene>/<modality>/<arm>/rep* and collect attempts."""
    attempts = []
    for ds in sorted(os.listdir(root)):
        pds = os.path.join(root, ds)
        if not os.path.isdir(pds):
            continue
        for scene in sorted(os.listdir(pds)):
            psc = os.path.join(pds, scene)
            if not os.path.isdir(psc):
                continue
            for mod in sorted(os.listdir(psc)):
                pmod = os.path.join(psc, mod)
                if not os.path.isdir(pmod):
                    continue
                for arm in sorted(os.listdir(pmod)):
                    parm = os.path.join(pmod, arm)
                    if not os.path.isdir(parm):
                        continue
                    for rep in sorted(os.listdir(parm)):
                        m = REP_RE.match(rep)
                        if not m:
                            continue
                        d = os.path.join(parm, rep)
                        if not os.path.isdir(d):
                            continue
                        st = read_status(d)
                        st.update({"dataset": ds, "scene": scene, "modality": mod,
                                   "arm": arm, "rep": int(m.group(1)),
                                   "attempt": int(m.group(3) or 0)})
                        attempts.append(st)
    return attempts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/eval_runs/v2")
    ap.add_argument("--manifest",
                    default="/workspace/eval_runs/MASTER_MANIFEST.csv")
    ap.add_argument("--out", default="/workspace/eval_runs/accounting")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    attempts = scan(a.root)
    with open(os.path.join(a.out, "ATTEMPTS.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "scene", "modality", "arm",
                                          "rep", "attempt", "has_status",
                                          "exit_code", "wall_s", "note", "end",
                                          "dir"])
        w.writeheader()
        for r in attempts:
            w.writerow(r)

    by_item = defaultdict(list)
    for r in attempts:
        by_item[(r["dataset"], r["scene"], r["modality"], r["arm"],
                 r["rep"])].append(r)

    rows = []
    if os.path.exists(a.manifest):
        with open(a.manifest) as f:
            for m in csv.DictReader(f):
                key = (m["dataset"], m["scene"], m["modality"], m["system_arm"],
                       int(m["repeat"]))
                atts = sorted(by_item.get(key, []), key=lambda x: x["attempt"])
                orig = [x for x in atts if x["attempt"] == 0]
                retr = [x for x in atts if x["attempt"] > 0]
                if m["status"] == "BLOCKED":
                    status, why = "BLOCKED", m["blocked_reason"]
                elif any(x["has_status"] and x["exit_code"] == "0" for x in orig):
                    status, why = "COMPLETED", ""
                elif any(x["has_status"] and x["exit_code"] == "0" for x in retr):
                    status, why = "COMPLETED_RETRY", ""
                elif orig and not orig[0]["has_status"]:
                    status, why = "PARTIAL", "no status.txt"
                elif atts:
                    codes = [x["exit_code"] for x in atts if x["has_status"]]
                    status = "FAILED"
                    why = "exit codes " + ",".join(codes) + (
                        " (retried once)" if retr else " (no retry)")
                else:
                    status, why = m["status"], ""
                rows.append({
                    "id": m["id"], "dataset": m["dataset"], "scene": m["scene"],
                    "modality": m["modality"], "arm": m["system_arm"],
                    "repeat": m["repeat"], "status": status, "reason": why,
                    "attempts": len(atts),
                    "config": m["config"], "input": m["input"],
                    "out_dir": m["out_dir"],
                    "entry_sha256": m["build_entry_sha256"],
                    "lib_sha256": m["build_lib_sha256"],
                })

    with open(os.path.join(a.out, "PLAN_ITEMS.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "dataset", "scene", "modality",
                                          "arm", "repeat", "status", "reason",
                                          "attempts", "config", "input",
                                          "out_dir", "entry_sha256",
                                          "lib_sha256"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("plan items: %d      directories (attempts): %d      status files: %d"
          % (len(rows), len(attempts), sum(1 for x in attempts if x["has_status"])))
    print("plan-item status:", dict(Counter(r["status"] for r in rows)))
    print("attempt exit codes:",
          dict(Counter(x["exit_code"] if x["has_status"] else "NO_STATUS"
                       for x in attempts)))
    print("retry attempts:", dict(Counter(x["exit_code"] if x["has_status"]
                                          else "NO_STATUS"
                                          for x in attempts if x["attempt"] > 0)))
    print()
    print("%-9s %-5s %-13s %s" % ("dataset", "mod", "status", "n"))
    for k, v in sorted(Counter((r["dataset"], r["modality"], r["status"])
                               for r in rows).items()):
        print("%-9s %-5s %-13s %d" % (k[0], k[1], k[2], v))
    print()
    print("FAILED items and their exit codes:")
    for r in rows:
        if r["status"] in ("FAILED", "PARTIAL"):
            print("  %-42s %-14s %s" % (r["id"], r["status"], r["reason"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
