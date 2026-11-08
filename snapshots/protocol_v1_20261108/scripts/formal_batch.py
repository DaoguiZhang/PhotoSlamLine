#!/usr/bin/env python3
"""Serial, resumable formal batch for the locked full-GPU-mapping version.

Design rules:
  * ONE binary and ONE set of libraries for the whole batch. Their SHA-256 is
    checked before every run; if it changes the batch aborts.
  * Full and OFF share the binary, the ORB config, the sequence and the seed.
    The only difference is the Gaussian YAML: OFF has the three effective
    structural-loss weights set to zero.
  * Repeats are the outer loop; condition order alternates per repeat
    (odd: full,off / even: off,full) so machine drift is not tied to a
    condition.
  * A run directory is never overwritten. Completed runs are skipped on resume.
"""
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time

EXP = "/workspace/code/PSL-fullmap-exp-v1"
SRC = "/workspace/code/PSL-fullmap-lock-v1"
MANIFEST = os.environ.get("MANIFEST", os.path.join(EXP, "manifest.csv"))
OUT = os.environ.get("OUT", os.path.join(EXP, "runs", "formal-v1"))
REPEATS = int(os.environ.get("REPEATS", "5"))
TIMEOUT = int(os.environ.get("RUN_TIMEOUT", "7200"))
VOC = f"{SRC}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
LIBS = [
    f"{SRC}/ORB-SLAM3/lib/libORB_SLAM3.so",
    f"{SRC}/lib/libgaussian_mapper.so",
    f"{SRC}/ORB-SLAM3/Thirdparty/g2o/lib/libg2o.so",
    f"{SRC}/ORB-SLAM3/Thirdparty/DBoW2/lib/libDBoW2.so",
]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract(d):
    log = os.path.join(d, "stdout.log")
    text = open(log, encoding="utf-8", errors="replace").read() if os.path.exists(log) else ""
    iters = re.findall(r"Total Iterations:\s*(\d+)", text)
    psnrs = []
    for root, _, files in os.walk(d):
        if "psnr.txt" in files:
            p = os.path.join(root, "psnr.txt")
            for line in open(p, encoding="utf-8", errors="replace"):
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    psnrs.append(float(parts[1]))
            break
    ply = None
    for root, _, files in os.walk(d):
        if "point_cloud.ply" in files:
            ply = os.path.join(root, "point_cloud.ply")
            break
    n_vert, ply_size = None, None
    if ply:
        ply_size = os.path.getsize(ply)
        with open(ply, "rb") as f:
            head = f.read(1024).decode("ascii", errors="replace")
        m = re.search(r"element vertex\s+(\d+)", head)
        n_vert = int(m.group(1)) if m else None
    return dict(
        total_iterations=int(iters[-1]) if iters else 0,
        psnr_entries=len(psnrs),
        mean_psnr=(sum(psnrs) / len(psnrs)) if psnrs else None,
        ply=ply, ply_size=ply_size, ply_vertices=n_vert,
        line_loss_diag_lines=len(re.findall(r"\[LINE-LOSS\]", text)),
        model_empty=("Gaussian model is empty" in text),
    )


def load_manifest():
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    rows = load_manifest()
    print(f"manifest cells={len(rows)} repeats={REPEATS} timeout={TIMEOUT}s", flush=True)

    if not os.path.isfile(VOC):
        print(f"MISSING vocabulary: {VOC}", file=sys.stderr)
        return 2

    binary_sha = {}
    for r in rows:
        b = os.path.join(SRC, "bin", r["binary"])
        binary_sha[r["binary"]] = sha(b)
    lib_sha = {p: sha(p) for p in LIBS}
    print("binary hashes:", json.dumps(binary_sha, indent=2), flush=True)
    print("library hashes:", json.dumps(lib_sha, indent=2), flush=True)

    summary_path = os.path.join(OUT, "summary.csv")
    fields = ["cell", "family", "dataset", "modality", "rep", "seed", "condition",
              "cond_order", "exit", "wall_s", "total_iterations", "psnr_entries",
              "mean_psnr", "ply", "ply_size", "ply_vertices",
              "line_loss_diag_lines", "model_empty", "accept",
              "binary_sha256", "libORB_SLAM3_sha256", "libgaussian_mapper_sha256",
              "orb_cfg_sha256", "gauss_cfg_sha256"]
    if not os.path.exists(summary_path):
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = (f"{SRC}/ORB-SLAM3/lib:{SRC}/ORB-SLAM3/Thirdparty/g2o/lib:"
                              f"{SRC}/ORB-SLAM3/Thirdparty/DBoW2/lib:{SRC}/lib")
    env.update({
        "PHOTO_SLAM_LINE_MODE": "2",
        "PHOTO_SLAM_LINE_LOOP": "0",
        "PHOTO_SLAM_SHADOW_LINE": "0",
        "PHOTO_SLAM_SLAM_ONLY": "0",
        "PHOTO_SLAM_NO_PACING": "0",
    })
    for k in ("PHOTO_SLAM_DEBUG_LINE_LOSS", "PHOTO_SLAM_DEBUG_LINE_EDGES",
              "PHOTO_SLAM_DEBUG_LINE_GATE", "PHOTO_SLAM_DEBUG_LINE_RESCUE",
              "PHOTO_SLAM_DEBUG_LINE_SCAN", "PHOTO_SLAM_DEBUG_MERGE",
              "PHOTO_SLAM_DEBUG_MONO_INIT", "PHOTO_SLAM_DEBUG_LINE_LOOP",
              "PHOTO_SLAM_DEBUG_POSTFIX"):
        env.pop(k, None)

    for rep in range(1, REPEATS + 1):
        order = ["full", "off"] if rep % 2 == 1 else ["off", "full"]
        for r in rows:
            for cond in order:
                cell, bname = r["cell"], r["binary"]
                d = os.path.join(OUT, cell, f"r{rep}_{cond}")
                status_path = os.path.join(d, "status.json")
                if os.path.exists(status_path):
                    try:
                        st = json.load(open(status_path))
                        if st.get("accept"):
                            print(f"SKIP completed {cell} r{rep} {cond}", flush=True)
                            continue
                    except Exception:
                        pass
                if os.path.isdir(d) and os.listdir(d):
                    print(f"SKIP non-empty incomplete {d}", flush=True)
                    continue

                # Hard guard: never let the batch continue on a changed build.
                if sha(os.path.join(SRC, "bin", bname)) != binary_sha[bname]:
                    print(f"ABORT: binary changed: {bname}", file=sys.stderr)
                    return 3
                for p in LIBS:
                    if sha(p) != lib_sha[p]:
                        print(f"ABORT: library changed: {p}", file=sys.stderr)
                        return 3

                os.makedirs(d, exist_ok=True)
                gauss = r["gauss_cfg"] if cond == "full" else r["off_cfg"]
                argv = [os.path.join(SRC, "bin", bname), VOC, r["orb_cfg"], gauss,
                        r["seq_dir"]]
                if r["assoc"]:
                    argv.append(r["assoc"])
                argv += [d + "/", "no_viewer"]

                run_env = dict(env)
                run_env["PHOTO_SLAM_SEED"] = str(rep - 1)
                meta = {
                    "cell": cell, "family": r["family"], "dataset": r["dataset"],
                    "modality": r["modality"], "condition": cond,
                    "cond_order": ",".join(order), "rep": rep, "seed": rep - 1,
                    "binary": argv[0], "binary_sha256": binary_sha[bname],
                    "libORB_SLAM3": LIBS[0], "libORB_SLAM3_sha256": lib_sha[LIBS[0]],
                    "libgaussian_mapper": LIBS[1],
                    "libgaussian_mapper_sha256": lib_sha[LIBS[1]],
                    "orb_cfg": r["orb_cfg"], "orb_cfg_sha256": sha(r["orb_cfg"]),
                    "gauss_cfg": gauss, "gauss_cfg_sha256": sha(gauss),
                    "seq_dir": r["seq_dir"], "assoc": r["assoc"],
                    "frames": r["frames"], "argv": argv,
                    "ld_library_path": run_env["LD_LIBRARY_PATH"],
                    "line_mode": 2, "line_loop": 0, "shadow_line": 0,
                    "slam_only": 0, "no_pacing": 0,
                }
                json.dump(meta, open(os.path.join(d, "run.meta.json"), "w"), indent=2)
                open(os.path.join(d, "cmd.txt"), "w").write(" ".join(argv) + "\n")
                for src, name in ((r["orb_cfg"], "orb_settings.used.yaml"),
                                  (gauss, "gaussian_settings.used.yaml")):
                    open(os.path.join(d, name), "wb").write(open(src, "rb").read())

                log = open(os.path.join(d, "stdout.log"), "w")
                t0 = time.time()
                try:
                    rc = subprocess.run(argv, env=run_env, stdout=log,
                                        stderr=subprocess.STDOUT,
                                        timeout=TIMEOUT).returncode
                except subprocess.TimeoutExpired:
                    rc = 124
                finally:
                    log.close()
                wall = round(time.time() - t0, 1)
                m = extract(d)
                accept = (rc == 0 and m["total_iterations"] > 0
                          and m["psnr_entries"] > 0 and (m["ply_vertices"] or 0) > 0)
                st = dict(meta, exit=rc, wall_s=wall, accept=accept, **m)
                json.dump(st, open(status_path, "w"), indent=2)

                row = {k: st.get(k) for k in fields}
                row["mean_psnr"] = ("" if st["mean_psnr"] is None
                                    else f"{st['mean_psnr']:.6f}")
                row["orb_cfg_sha256"] = meta["orb_cfg_sha256"]
                row["gauss_cfg_sha256"] = meta["gauss_cfg_sha256"]
                row["libORB_SLAM3_sha256"] = meta["libORB_SLAM3_sha256"]
                row["libgaussian_mapper_sha256"] = meta["libgaussian_mapper_sha256"]
                row["binary_sha256"] = meta["binary_sha256"]
                with open(summary_path, "a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=fields).writerow(row)
                print(f"[r{rep} {cell} {cond}] exit={rc} wall={wall}s "
                      f"iters={m['total_iterations']} psnr={m['mean_psnr']} "
                      f"verts={m['ply_vertices']} ACCEPT={accept}", flush=True)
    print("BATCH DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
