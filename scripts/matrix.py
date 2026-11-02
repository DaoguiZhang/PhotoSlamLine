#!/usr/bin/env python3
"""Master manifest + resumable serial executor for the cross-dataset matrix.

Usage:
  python3 scripts/matrix.py build     # (re)build MASTER_MANIFEST.csv
  python3 scripts/matrix.py run       # execute PENDING rows serially, resumable
  python3 scripts/matrix.py status    # counts by status / category

Rules enforced here (see eval/PROTOCOL.md):
  * one directory per run, never reused; a run with an existing status.txt is
    never re-executed
  * the real exit code is stored; failures are kept, never retried for a better
    number
  * every row carries its own identity: dataset, scene, modality, arm, repeat,
    config, input, association, build hashes, output dir
  * status vocabulary: PENDING PREPARING RUNNING COMPLETED FAILED BLOCKED INVALID
"""
import csv
import json
import os
import subprocess
import sys
import time

TREE = "/workspace/deliv/cabinet-eval"
RUNS = "/workspace/eval_runs/v2"
MANIFEST = "/workspace/eval_runs/MASTER_MANIFEST.csv"
DS = "/workspace/code/SEGS-SLAM/datasets"
BIN = os.path.join(TREE, "bin/eval_frontend")
LIB = os.path.join(TREE, "ORB-SLAM3/lib/libORB_SLAM3.so")
VOCAB = os.path.join(TREE, "ORB-SLAM3/Vocabulary/ORBvoc.txt")

ARMS = ["T0", "P1", "P2"]
REPS = [1, 2, 3]
LINE_MODE = {"T0": "0", "P1": "1", "P2": "2"}

TUM = [
    # scene, mono config, rgbd config, association
    ("cabinet", "Monocular/TUM/tum_freiburg3_cabinet.yaml",
     "RGB-D/TUM/tum_freiburg3_cabinet.yaml",
     "RGB-D/TUM/associations/fr3_cabinet.txt"),
    ("large_cabinet", "Monocular/TUM/tum_freiburg3_large_cabinet.yaml",
     "RGB-D/TUM/tum_freiburg3_large_cabinet.yaml",
     "RGB-D/TUM/associations/fr3_large_cabinet.txt"),
    ("long_office_household", "Monocular/TUM/tum_freiburg3_long_office_household.yaml",
     "RGB-D/TUM/tum_freiburg3_long_office_household.yaml",
     "RGB-D/TUM/associations/fr3_office.txt"),
    ("structure_notexture_near", "Monocular/TUM/tum_freiburg3_structure_notexture_near_lowfast.yaml",
     "RGB-D/TUM/tum_freiburg3_structure_notexture_near.yaml",
     "RGB-D/TUM/associations/fr3_structure_notexture_near.txt"),
    ("structure_notexture_far", "Monocular/TUM/tum_freiburg3_structure_notexture_far_lowfast.yaml",
     "RGB-D/TUM/tum_freiburg3_structure_notexture_far.yaml",
     "RGB-D/TUM/associations/fr3_structure_notexture_far.txt"),
    ("freiburg1_desk", "Monocular/TUM/tum_freiburg1_desk.yaml",
     "RGB-D/TUM/tum_freiburg1_desk.yaml", "RGB-D/TUM/associations/fr1_desk.txt"),
    ("freiburg1_room", "Monocular/TUM/tum_freiburg1_room.yaml",
     None, "RGB-D/TUM/associations/fr1_room.txt"),
    ("freiburg2_xyz", "Monocular/TUM/tum_freiburg2_xyz.yaml",
     "RGB-D/TUM/tum_freiburg2_xyz.yaml", "RGB-D/TUM/associations/fr2_xyz.txt"),
]
TUM_DIR = {"freiburg1_desk": "rgbd_dataset_freiburg1_desk",
           "freiburg1_room": "rgbd_dataset_freiburg1_room",
           "freiburg2_xyz": "rgbd_dataset_freiburg2_xyz"}


def sha(p):
    if not p or not os.path.exists(p):
        return ""
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rows_for(dataset, scene, modality, cfg, inp, assoc, blocked=""):
    out = []
    for arm in ARMS:
        for rep in REPS:
            d = os.path.join(RUNS, dataset, scene, modality, arm, "rep%d" % rep)
            out.append({
                "id": "%s|%s|%s|%s|rep%d" % (dataset, scene, modality, arm, rep),
                "dataset": dataset, "scene": scene, "modality": modality,
                "system_arm": arm, "repeat": rep,
                "config": cfg or "", "input": inp or "", "assoc": assoc or "",
                "out_dir": d, "blocked_reason": blocked,
            })
    return out


def build():
    rows = []
    # ---- TUM
    for scene, mcfg, rcfg, assoc in TUM:
        d = os.path.join(DS, "tum", TUM_DIR.get(scene, "rgbd_dataset_freiburg3_" + scene))
        inp = d if os.path.isdir(d) else ""
        rows += rows_for("TUM", scene, "mono", mcfg, inp, "")
        if rcfg:
            a = assoc if os.path.exists(os.path.join(TREE, "cfg/ORB_SLAM3", assoc)) else ""
            rows += rows_for("TUM", scene, "rgbd", rcfg, inp, a)
        else:
            rows += rows_for("TUM", scene, "rgbd", "", inp, "",
                             "no RGB-D config for this sequence in the repository")
    # ---- Replica
    rroot = os.path.join(DS, "replica")
    for scene in sorted(os.listdir(rroot)):
        if not os.path.isdir(os.path.join(rroot, scene, "results")):
            continue
        inp = os.path.join(rroot, scene)
        rows += rows_for("Replica", scene, "mono",
                         "Monocular/Replica/replica_%s.yaml" % scene, inp, "")
        rows += rows_for("Replica", scene, "rgbd",
                         "RGB-D/Replica/replica_%s.yaml" % scene, inp, "")
    # ---- ETH3D
    eroot = os.path.join(DS, "eth3d")
    for scene in sorted(os.listdir(eroot)):
        d = os.path.join(eroot, scene)
        if not os.path.isdir(os.path.join(d, "rgb")):
            continue
        rows += rows_for("ETH3D", scene, "mono",
                         "Monocular/ETH3D/eth3d_%s.yaml" % scene, d, "")
        rows += rows_for("ETH3D", scene, "rgbd",
                         "RGB-D/ETH3D/eth3d_%s.yaml" % scene, d,
                         os.path.join(d, "associated.txt"))
    # ---- upstream Photo-SLAM (point-only, one arm)
    for dataset, scene, modality, inp, assoc in [
            ("TUM", s, "mono", os.path.join(DS, "tum", TUM_DIR.get(s, "rgbd_dataset_freiburg3_" + s)), "")
            for s in ["cabinet", "large_cabinet"]]:
        for rep in REPS:
            rows.append({
                "id": "UPSTREAM|%s|%s|%s|rep%d" % (dataset, scene, modality, rep),
                "dataset": dataset, "scene": scene, "modality": modality,
                "system_arm": "UPSTREAM", "repeat": rep,
                "config": "", "input": inp, "assoc": assoc,
                "out_dir": os.path.join(RUNS, "UPSTREAM", scene, modality,
                                        "UP", "rep%d" % rep),
                "blocked_reason": "upstream Photo-SLAM binary not built in this "
                                  "worktree (needs its Gaussian/torch backend)",
            })

    bh, lh = sha(BIN), sha(LIB)
    for r in rows:
        if r["blocked_reason"]:
            r["status"] = "BLOCKED"
        elif not r["input"] or not os.path.exists(r["input"]):
            r["status"] = "BLOCKED"
            r["blocked_reason"] = "input dataset path missing"
        elif not r["config"] or not os.path.exists(os.path.join(TREE, "cfg/ORB_SLAM3", r["config"])):
            r["status"] = "BLOCKED"
            r["blocked_reason"] = "config file missing: %s" % r["config"]
        elif r["modality"] == "rgbd" and r["dataset"] == "TUM" and not r["assoc"]:
            r["status"] = "BLOCKED"
            r["blocked_reason"] = "association file missing"
        else:
            st = os.path.join(r["out_dir"], "status.txt")
            if os.path.exists(st):
                code = None
                for line in open(st):
                    if line.startswith("exit_code="):
                        code = int(line.split("=")[1])
                r["status"] = "COMPLETED" if code == 0 else "FAILED"
            else:
                r["status"] = "PENDING"
        r["build_entry_sha256"] = bh
        r["build_lib_sha256"] = lh

    keys = ["id", "dataset", "scene", "modality", "system_arm", "repeat",
            "status", "blocked_reason", "config", "input", "assoc", "out_dir",
            "build_entry_sha256", "build_lib_sha256"]
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    from collections import Counter
    print("manifest=%s rows=%d" % (MANIFEST, len(rows)))
    print("build: entry=%s lib=%s" % (bh[:16], lh[:16]))
    for k, v in sorted(Counter((r["dataset"], r["status"]) for r in rows).items()):
        print("  %-8s %-10s %d" % (k[0], k[1], v))
    print("  TOTAL " + str(Counter(r["status"] for r in rows)))
    return 0


def status():
    with open(MANIFEST) as f:
        rows = list(csv.DictReader(f))
    from collections import Counter
    print("rows=%d" % len(rows))
    for k, v in sorted(Counter(r["status"] for r in rows).items()):
        print("  %-10s %d" % (k, v))
    for k, v in sorted(Counter((r["dataset"], r["modality"], r["status"]) for r in rows).items()):
        print("  %-8s %-5s %-10s %d" % (k[0], k[1], k[2], v))
    return 0


def run():
    with open(MANIFEST) as f:
        rows = list(csv.DictReader(f))
    os.makedirs(RUNS, exist_ok=True)
    prog = open(os.path.join("/workspace/eval_runs", "matrix_progress.log"), "a")
    done = 0
    for r in rows:
        if r["status"] != "PENDING":
            continue
        out = r["out_dir"]
        if os.path.isdir(out) and os.path.exists(os.path.join(out, "status.txt")):
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if not os.path.isdir(out):
            os.makedirs(out)
        cfg = os.path.join(TREE, "cfg/ORB_SLAM3", r["config"])
        env = dict(os.environ)
        env.update({"EVAL_ARM": r["system_arm"],
                    "PHOTO_SLAM_LINE_MODE": LINE_MODE[r["system_arm"]],
                    "PHOTO_SLAM_PO_LINE": "1", "PHOTO_SLAM_LBA_LINE": "1",
                    "PHOTO_SLAM_LINE_RESCUE": "0", "PHOTO_SLAM_SLAM_ONLY": "1"})
        if r["dataset"] == "Replica":
            mode = "replica-" + r["modality"]
            args = [BIN, VOCAB, cfg, "/dev/null", r["input"], out, mode]
        elif r["modality"] == "mono":
            mode = "mono"
            args = [BIN, VOCAB, cfg, "/dev/null", r["input"], out, mode]
        else:
            mode = "rgbd"
            args = [BIN, VOCAB, cfg, "/dev/null", r["input"], out, mode, r["assoc"]]
        t0 = time.time()
        with open(os.path.join(out, "meta.txt"), "w") as f:
            f.write("id=%s\ncmd=%s\nenv=%s\nstart=%s\n"
                    % (r["id"], " ".join(args),
                       json.dumps({k: env[k] for k in
                                   ("EVAL_ARM", "PHOTO_SLAM_LINE_MODE",
                                    "PHOTO_SLAM_PO_LINE", "PHOTO_SLAM_LBA_LINE",
                                    "PHOTO_SLAM_LINE_RESCUE",
                                    "PHOTO_SLAM_SLAM_ONLY")}),
                       time.strftime("%Y-%m-%dT%H:%M:%S")))
        with open(os.path.join(out, "stdout.log"), "w") as lf:
            rc = subprocess.call(args, env=env, stdout=lf, stderr=subprocess.STDOUT)
        wall = time.time() - t0
        with open(os.path.join(out, "status.txt"), "w") as f:
            f.write("exit_code=%d\nwall_s=%.1f\nend=%s\n"
                    % (rc, wall, time.strftime("%Y-%m-%dT%H:%M:%S")))
        line = "%s exit=%d wall=%.1fs\n" % (r["id"], rc, wall)
        prog.write(line)
        prog.flush()
        done += 1
        print(line.strip(), flush=True)
    print("run() finished, executed=%d" % done)
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    sys.exit({"build": build, "run": run, "status": status}[cmd]())
