#!/usr/bin/env python3
"""Compare variable-topology A0/A1/C arms from one room1 capture.

Reads the files written by PHOTO_SLAM_TOPO_REPLAY_DIR:
  summary.csv, common_state.txt, <arm>/metrics.csv,
  <arm>/topology_steps.csv, audit/topo_topology.csv, audit/topo_selected.csv

The comparison is descriptive. It reports the first step/stage at which the
topology counters differ and the first step at which the fixed-view GS PSNR
differs. It never pairs tensor rows after a topology change.
"""
import csv
import json
import os
import sys
from collections import defaultdict


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def num(x, default=-1.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def load_arm(root, arm):
    d = os.path.join(root, arm)
    metrics = {int(float(r["step"])): r for r in read_csv(os.path.join(d, "metrics.csv"))}
    steps = []
    for r in read_csv(os.path.join(d, "topology_steps.csv")):
        steps.append({
            "step": int(r["step"]),
            "iteration": int(r["iteration"]),
            "total": int(r["total"]),
            "line": int(r["line"]),
            "clone_cum": int(r["clone_cum"]),
            "split_cum": int(r["split_cum"]),
            "prune_cum": int(r["prune_cum"]),
            "view_index": int(r["view_index"]),
            "view_fid": int(r["view_fid"]),
        })
    topo = [r for r in read_csv(os.path.join(root, "audit", "topo_topology.csv"))
            if r.get("arm") == arm]
    selected = [r for r in read_csv(os.path.join(root, "audit", "topo_selected.csv"))
                if r.get("arm") == arm]
    return {"metrics": metrics, "steps": steps, "topology": topo,
            "selected": selected}


def norm_steps(steps):
    if not steps:
        return []
    base = steps[0]
    out = []
    for r in steps:
        x = dict(r)
        x["clone_arm"] = r["clone_cum"] - base["clone_cum"]
        x["split_arm"] = r["split_cum"] - base["split_cum"]
        x["prune_arm"] = r["prune_cum"] - base["prune_cum"]
        out.append(x)
    return out


def first_step_diff(a, b, key):
    for x, y in zip(a, b):
        if x["step"] != y["step"]:
            return {"step": min(x["step"], y["step"]), "reason": "step alignment"}
        if x[key] != y[key]:
            return {"step": x["step"], "a": x[key], "b": y[key], "key": key}
    return None


def topo_key(r):
    return (r.get("iteration"), r.get("stage"))


def first_topo_diff(a, b):
    da = {topo_key(r): r for r in a}
    db = {topo_key(r): r for r in b}
    keys = sorted(set(da) & set(db), key=lambda k: (int(k[0]), str(k[1])))
    fields = ["total_after", "line_after", "prune_opacity", "prune_lazy",
              "prune_big", "clone_delta", "split_delta"]
    for k in keys:
        x, y = da[k], db[k]
        for f in fields:
            if x.get(f) != y.get(f):
                return {"iteration": int(k[0]), "stage": k[1], "field": f,
                        "a": x.get(f), "b": y.get(f)}
    return None


def selected_hash(rows):
    # top-16 row indices and their gradient values at the pre-prune snapshot
    out = defaultdict(list)
    for r in rows:
        try:
            out[int(r["iteration"])].append((int(r["rank"]), int(r["row"]),
                                             float(r["grad"])))
        except (KeyError, ValueError):
            pass
    return {k: sorted(v) for k, v in out.items()}


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    arms = ["A0", "A1", "C"]
    data = {a: load_arm(root, a) for a in arms}
    for a in arms:
        data[a]["norm"] = norm_steps(data[a]["steps"])

    result = {"root": os.path.abspath(root), "arms": {}}
    for a in arms:
        m = data[a]["metrics"]
        result["arms"][a] = {
            "n_steps": len(data[a]["steps"]),
            "first": data[a]["steps"][0] if data[a]["steps"] else None,
            "last": data[a]["steps"][-1] if data[a]["steps"] else None,
            "psnr": {str(s): num(m[s]["psnr"]) for s in sorted(m)},
            "psnr_gs": {str(s): num(m[s].get("psnr_gs")) for s in sorted(m)},
        }

    for a, b in (("A0", "A1"), ("A0", "C")):
        result[f"{a}_vs_{b}"] = {
            "first_total": first_step_diff(data[a]["norm"], data[b]["norm"], "total"),
            "first_line": first_step_diff(data[a]["norm"], data[b]["norm"], "line"),
            "first_clone": first_step_diff(data[a]["norm"], data[b]["norm"], "clone_arm"),
            "first_split": first_step_diff(data[a]["norm"], data[b]["norm"], "split_arm"),
            "first_prune": first_step_diff(data[a]["norm"], data[b]["norm"], "prune_arm"),
            "first_topo_event": first_topo_diff(data[a]["topology"], data[b]["topology"]),
            "selected_hash_equal": selected_hash(data[a]["selected"]) ==
                                   selected_hash(data[b]["selected"]),
        }

    out = os.path.join(root, "analysis.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
