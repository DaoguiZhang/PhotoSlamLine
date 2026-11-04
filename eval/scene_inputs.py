#!/usr/bin/env python3
"""How many input frames a cell is *supposed* to process, from the frozen input.

The number of rows a run managed to write is not the number of frames it should
have processed: a crashed run writes fewer, and using its own row count as the
denominator would silently shrink the run it is judged on. The rules below mirror
exactly what `examples/eval_frontend.cpp` feeds to `Track*`:

  mono (TUM/ETH3D)  <seq>/timestamps.txt if present, else <seq>/rgb.txt
  rgbd              the association file named by the manifest
  replica-*         consecutive results/frame%06d.jpg starting at 0

They are pure functions of the dataset directory, so they can be recomputed at
any time and compared against `final_state.csv` to decide whether the input was
processed completely.
"""
import glob
import os

DS = "/workspace/code/SEGS-SLAM/datasets"
ASSOC_ROOT = "/workspace/deliv/cabinet-eval/cfg/ORB_SLAM3"

TUM_DIRS = {"freiburg1_desk": "rgbd_dataset_freiburg1_desk",
            "freiburg1_room": "rgbd_dataset_freiburg1_room",
            "freiburg2_xyz": "rgbd_dataset_freiburg2_xyz"}

TUM_ASSOC = {
    "cabinet": "RGB-D/TUM/associations/fr3_cabinet.txt",
    "large_cabinet": "RGB-D/TUM/associations/fr3_large_cabinet.txt",
    "long_office_household": "RGB-D/TUM/associations/fr3_office.txt",
    "structure_notexture_near":
        "RGB-D/TUM/associations/fr3_structure_notexture_near.txt",
    "structure_notexture_far":
        "RGB-D/TUM/associations/fr3_structure_notexture_far.txt",
    "freiburg1_desk": "RGB-D/TUM/associations/fr1_desk.txt",
    "freiburg1_room": "RGB-D/TUM/associations/fr1_room.txt",
    "freiburg2_xyz": "RGB-D/TUM/associations/fr2_xyz.txt",
}


def seq_dir(dataset, scene):
    if dataset == "TUM":
        return os.path.join(DS, "tum",
                            TUM_DIRS.get(scene, "rgbd_dataset_freiburg3_" + scene))
    if dataset == "ETH3D":
        return os.path.join(DS, "eth3d", scene)
    if dataset == "Replica":
        return os.path.join(DS, "replica", scene)
    raise SystemExit("unknown dataset " + dataset)


def _lines(path):
    if not os.path.exists(path):
        return None
    n = 0
    with open(path) as f:
        for line in f:
            if line.strip() and not line.lstrip().startswith("#"):
                n += 1
    return n


def input_timestamps(dataset, scene, modality):
    """The exact timestamp list the entry feeds to Track*, from the frozen input.

    This is the denominator source: it does not depend on what a run managed to
    process, so a crashed run cannot shrink the population it is judged on.
    """
    d = seq_dir(dataset, scene)
    if dataset == "Replica":
        n = len(glob.glob(os.path.join(d, "results", "frame*.jpg")))
        return [i / 30.0 for i in range(n)]
    if modality == "rgbd":
        path = (os.path.join(ASSOC_ROOT, TUM_ASSOC[scene])
                if dataset == "TUM" else os.path.join(d, "associated.txt"))
        ts = []
        with open(path) as f:
            for line in f:
                s = line.split()
                if not s or s[0].startswith("#"):
                    continue
                # association rows: "rgb_ts rgb depth_ts depth"; the entry feeds
                # the RGB timestamp.
                ts.append(float(s[0]))
        return sorted(ts)
    path = os.path.join(d, "timestamps.txt")
    if not os.path.exists(path):
        path = os.path.join(d, "rgb.txt")
    ts = []
    with open(path) as f:
        for line in f:
            s = line.split()
            if not s or s[0].startswith("#"):
                continue
            ts.append(float(s[0]))
    return ts


def gt_evaluable_count(input_ts, gt_ts, max_dt):
    """How many INPUT frames have a GT sample within max_dt (binary search)."""
    import bisect
    n = 0
    for t in input_ts:
        i = bisect.bisect_left(gt_ts, t)
        best = None
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(gt_ts):
                d = abs(gt_ts[j] - t)
                if best is None or d < best:
                    best = d
        if best is not None and best <= max_dt:
            n += 1
    return n


def expected_inputs(dataset, scene, modality):
    """Frame count the entry will feed, or None if it cannot be determined."""
    d = seq_dir(dataset, scene)
    if dataset == "Replica":
        return len(glob.glob(os.path.join(d, "results", "frame*.jpg")))
    if modality == "rgbd":
        if dataset == "TUM":
            a = TUM_ASSOC.get(scene)
            if a is None:
                return None
            return _lines(os.path.join(ASSOC_ROOT, a))
        # ETH3D ships its association file next to the frames
        return _lines(os.path.join(d, "associated.txt"))
    # mono
    n = _lines(os.path.join(d, "timestamps.txt"))
    if n is None:
        n = _lines(os.path.join(d, "rgb.txt"))
    return n
