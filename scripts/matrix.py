#!/usr/bin/env python3
"""Master manifest + single-instance resumable serial executor (v3).

v3 changes:
  * configs use the REPOSITORY's official files where they exist. ETH3D ships a
    complete RGB-D set with RGBD.DepthMapFactor = 5000 (verified against
    cfg/ORB_SLAM3/RGB-D/ETH3D/eth3d_desk_3.yaml); Replica ships all 16. An
    earlier generator wrote its own copies with factor 1000 -- those are removed
    and recorded as INVALID_CONFIG in eval/DEVIATIONS.md.
  * every path handed to the binary is ABSOLUTE (a relative association path is
    what made 18 RGB-D runs exit 2)
  * single-instance flock; a second scheduler refuses to start
  * frozen per-run timeout (TIMEOUT_S); a timeout is recorded as TIMEOUT
  * status derived from artefacts on every rebuild; a directory without
    status.txt is PARTIAL and is never reused
  * FAILED rows may be retried ONCE into a sibling `_r1` directory; the failed
    directory is kept untouched

Usage: python3 scripts/matrix.py build|run|status
"""
import csv
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter

TREE = "/workspace/deliv/cabinet-eval"
RUNS = "/workspace/eval_runs/v2"
MANIFEST = "/workspace/eval_runs/MASTER_MANIFEST.csv"
LOCK = "/workspace/eval_runs/.matrix.lock"
PROGRESS = "/workspace/eval_runs/matrix_progress.log"
DS = "/workspace/code/SEGS-SLAM/datasets"
BIN = os.path.join(TREE, "bin/eval_frontend")
LIB = os.path.join(TREE, "ORB-SLAM3/lib/libORB_SLAM3.so")
VOCAB = os.path.join(TREE, "ORB-SLAM3/Vocabulary/ORBvoc.txt")
CFG = os.path.join(TREE, "cfg/ORB_SLAM3")

ARMS = ["T0", "P1", "P2"]
REPS = [1, 2, 3]
LINE_MODE = {"T0": "0", "P1": "1", "P2": "2"}
TIMEOUT_S = 1800

# ETH3D mono configs exist for 6 of 11 sequences; the rest use configs derived
# from each sequence's own calibration.txt (MONO has no depth key, so the ETH3D
# depth factor does not enter).
ETH_MONO_GEN = {"mannequin_1", "mannequin_3", "plant_scene_1", "sfm_bench",
                "sfm_house_loop", "sfm_lab_room_1"}

TUM = [
    ("cabinet", "rgbd_dataset_freiburg3_cabinet",
     "Monocular/TUM/tum_freiburg3_cabinet.yaml",
     "RGB-D/TUM/tum_freiburg3_cabinet.yaml",
     "RGB-D/TUM/associations/fr3_cabinet.txt"),
    ("large_cabinet", "rgbd_dataset_freiburg3_large_cabinet",
     "Monocular/TUM/tum_freiburg3_large_cabinet.yaml",
     "RGB-D/TUM/tum_freiburg3_large_cabinet.yaml",
     "RGB-D/TUM/associations/fr3_large_cabinet.txt"),
    ("long_office_household", "rgbd_dataset_freiburg3_long_office_household",
     "Monocular/TUM/tum_freiburg3_long_office_household.yaml",
     "RGB-D/TUM/tum_freiburg3_long_office_household.yaml",
     "RGB-D/TUM/associations/fr3_office.txt"),
    ("structure_notexture_near", "rgbd_dataset_freiburg3_structure_notexture_near",
     "Monocular/TUM/tum_freiburg3_structure_notexture_near_lowfast.yaml",
     "RGB-D/TUM/tum_freiburg3_structure_notexture_near.yaml",
     "RGB-D/TUM/associations/fr3_structure_notexture_near.txt"),
    ("structure_notexture_far", "rgbd_dataset_freiburg3_structure_notexture_far",
     "Monocular/TUM/tum_freiburg3_structure_notexture_far_lowfast.yaml",
     "RGB-D/TUM/tum_freiburg3_structure_notexture_far.yaml",
     "RGB-D/TUM/associations/fr3_structure_notexture_far.txt"),
    ("freiburg1_desk", "rgbd_dataset_freiburg1_desk",
     "Monocular/TUM/tum_freiburg1_desk.yaml",
     "RGB-D/TUM/tum_freiburg1_desk.yaml", "RGB-D/TUM/associations/fr1_desk.txt"),
    ("freiburg1_room", "rgbd_dataset_freiburg1_room",
     "Monocular/TUM/tum_freiburg1_room.yaml",
     "RGB-D/TUM/tum_freiburg1_room.yaml",
     "RGB-D/TUM/associations/fr1_room.txt"),
    ("freiburg2_xyz", "rgbd_dataset_freiburg2_xyz",
     "Monocular/TUM/tum_freiburg2_xyz.yaml",
     "RGB-D/TUM/tum_freiburg2_xyz.yaml", "RGB-D/TUM/associations/fr2_xyz.txt"),
]


def sha(p):
    if not p or not os.path.exists(p):
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rows_for(dataset, scene, modality, cfg, inp, assoc, blocked=""):
    out = []
    for arm in ARMS:
        for rep in REPS:
            out.append({
                "id": "%s|%s|%s|%s|rep%d" % (dataset, scene, modality, arm, rep),
                "dataset": dataset, "scene": scene, "modality": modality,
                "system_arm": arm, "repeat": rep,
                "config": cfg or "", "input": inp or "", "assoc": assoc or "",
                "out_dir": os.path.join(RUNS, dataset, scene, modality, arm,
                                        "rep%d" % rep),
                "blocked_reason": blocked,
            })
    return out


def build():
    rows = []
    for scene, dname, mcfg, rcfg, assoc in TUM:
        inp = os.path.join(DS, "tum", dname)
        rows += rows_for("TUM", scene, "mono", mcfg, inp, "")
        if rcfg:
            rows += rows_for("TUM", scene, "rgbd", rcfg, inp,
                             os.path.join(CFG, assoc))
        else:
            rows += rows_for("TUM", scene, "rgbd", "", inp, "",
                             "no RGB-D config for this sequence in the repository "
                             "and no same-sensor template available to derive one")
    rroot = os.path.join(DS, "replica")
    for scene in sorted(os.listdir(rroot)):
        if not os.path.isdir(os.path.join(rroot, scene, "results")):
            continue
        inp = os.path.join(rroot, scene)
        rows += rows_for("Replica", scene, "mono",
                         "Monocular/Replica/%s.yaml" % scene, inp, "")
        rows += rows_for("Replica", scene, "rgbd",
                         "RGB-D/Replica/%s.yaml" % scene, inp, "")
    eroot = os.path.join(DS, "eth3d")
    for scene in sorted(os.listdir(eroot)):
        d = os.path.join(eroot, scene)
        if not os.path.isdir(os.path.join(d, "rgb")):
            continue
        mc = ("Monocular/ETH3D_gen/eth3d_%s.yaml" % scene
              if scene in ETH_MONO_GEN else "Monocular/ETH3D/eth3d_%s.yaml" % scene)
        rows += rows_for("ETH3D", scene, "mono", mc, d, "")
        rows += rows_for("ETH3D", scene, "rgbd",
                         "RGB-D/ETH3D/eth3d_%s.yaml" % scene, d,
                         os.path.join(d, "associated.txt"))
    for scene in ("cabinet", "large_cabinet"):
        for rep in REPS:
            rows.append({
                "id": "UPSTREAM|TUM|%s|mono|rep%d" % (scene, rep),
                "dataset": "UPSTREAM", "scene": scene, "modality": "mono",
                "system_arm": "UPSTREAM", "repeat": rep,
                "config": "",
                "input": os.path.join(DS, "tum", "rgbd_dataset_freiburg3_" + scene),
                "assoc": "",
                "out_dir": os.path.join(RUNS, "UPSTREAM", scene, "mono", "UP",
                                        "rep%d" % rep),
                "blocked_reason": "upstream Photo-SLAM binary does not exist in this "
                                  "worktree (its Gaussian/torch backend is not "
                                  "built); T0 of this fork is not a substitute",
            })

    bh, lh = sha(BIN), sha(LIB)
    for r in rows:
        if r["blocked_reason"]:
            r["status"] = "BLOCKED"
        elif not r["input"] or not os.path.exists(r["input"]):
            r["status"], r["blocked_reason"] = "BLOCKED", "input dataset path missing"
        elif not r["config"] or not os.path.exists(os.path.join(CFG, r["config"])):
            r["status"] = "BLOCKED"
            r["blocked_reason"] = "config file missing: %s" % r["config"]
        elif r["modality"] == "rgbd" and r["dataset"] == "TUM" and not os.path.exists(r["assoc"]):
            r["status"] = "BLOCKED"
            r["blocked_reason"] = "association file missing: %s" % r["assoc"]
        else:
            st = os.path.join(r["out_dir"], "status.txt")
            if os.path.exists(st):
                code, note = None, ""
                for line in open(st):
                    if line.startswith("exit_code="):
                        code = int(line.split("=")[1])
                    if line.startswith("note="):
                        note = line.split("=", 1)[1].strip()
                if code == 0:
                    r["status"] = "COMPLETED"
                elif note.startswith("INVALID"):
                    r["status"] = "INVALID"
                elif note.startswith("TIMEOUT"):
                    r["status"] = "TIMEOUT"
                else:
                    r["status"] = "FAILED"
            elif os.path.isdir(r["out_dir"]):
                r["status"] = "PARTIAL"
                r["blocked_reason"] = ("directory exists without status.txt "
                                       "(interrupted); never reused")
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
    print("manifest=%s rows=%d entry=%s lib=%s"
          % (MANIFEST, len(rows), bh[:16], lh[:16]))
    for k, v in sorted(Counter((r["dataset"], r["status"]) for r in rows).items()):
        print("  %-9s %-10s %d" % (k[0], k[1], v))
    print("  TOTAL", dict(Counter(r["status"] for r in rows)))
    return 0


def status():
    with open(MANIFEST) as f:
        rows = list(csv.DictReader(f))
    print("rows=%d" % len(rows))
    for k, v in sorted(Counter(r["status"] for r in rows).items()):
        print("  %-10s %d" % (k, v))
    for k, v in sorted(Counter((r["dataset"], r["modality"], r["status"])
                               for r in rows).items()):
        print("  %-9s %-5s %-10s %d" % (k[0], k[1], k[2], v))
    return 0


def run():
    lk = open(LOCK, "w")
    try:
        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another scheduler holds %s -- refusing to start a second" % LOCK)
        return 2
    with open(MANIFEST) as f:
        rows = list(csv.DictReader(f))
    prog = open(PROGRESS, "a")
    # Policy: a crash (SIGSEGV/SIGABRT) is NOT retried until it is attributed.
    # 19 SIGSEGV have been traced to Optimizer.cc:17472
    # (LocalBundleAdjustmentWithLine: delete vP1; delete vP2;) reached from
    # LoopClosing::MergeLocalWithLine, so those cells are left as FAILED rather
    # than burned on a retry that cannot succeed.
    def crash_only(r):
        st = os.path.join(r["out_dir"], "status.txt")
        if not os.path.exists(st):
            return False
        for line in open(st):
            if line.startswith("exit_code="):
                try:
                    return int(line.split("=")[1]) in (-11, -6)
                except ValueError:
                    return False
        return False

    todo = [r for r in rows
            if r["status"] == "PENDING"
            or (r["status"] in ("FAILED", "PARTIAL") and not crash_only(r))]
    print("to execute: %d" % len(todo), flush=True)
    n = 0
    for r in todo:
        retry = (r["status"] in ("FAILED", "PARTIAL"))
        out = r["out_dir"] + ("_r1" if retry else "")
        if os.path.isdir(out):
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        os.makedirs(out)
        cfg = os.path.join(CFG, r["config"])
        env = dict(os.environ)
        env.update({"EVAL_ARM": r["system_arm"],
                    "PHOTO_SLAM_LINE_MODE": LINE_MODE[r["system_arm"]],
                    "PHOTO_SLAM_PO_LINE": "1", "PHOTO_SLAM_LBA_LINE": "1",
                    "PHOTO_SLAM_LINE_RESCUE": "0",
                    "PHOTO_SLAM_SLAM_ONLY": "1"})
        if r["dataset"] == "Replica":
            args = [BIN, VOCAB, cfg, "/dev/null", r["input"], out,
                    "replica-" + r["modality"]]
        elif r["modality"] == "mono":
            args = [BIN, VOCAB, cfg, "/dev/null", r["input"], out, "mono"]
        else:
            args = [BIN, VOCAB, cfg, "/dev/null", r["input"], out, "rgbd", r["assoc"]]
        with open(os.path.join(out, "meta.txt"), "w") as f:
            f.write("id=%s\nretry=%d\ncmd=%s\nenv=%s\nstart=%s\n"
                    % (r["id"], int(retry), " ".join(args),
                       json.dumps({k: env[k] for k in
                                   ("EVAL_ARM", "PHOTO_SLAM_LINE_MODE",
                                    "PHOTO_SLAM_PO_LINE", "PHOTO_SLAM_LBA_LINE",
                                    "PHOTO_SLAM_LINE_RESCUE",
                                    "PHOTO_SLAM_SLAM_ONLY")}),
                       time.strftime("%Y-%m-%dT%H:%M:%S")))
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
        line = "%s retry=%d exit=%d wall=%.1fs %s\n" % (r["id"], int(retry), rc,
                                                        wall, note)
        prog.write(line)
        prog.flush()
        n += 1
        print(line.strip(), flush=True)
    print("run() finished, executed=%d" % n)
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    sys.exit({"build": build, "run": run, "status": status}[cmd]())
