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
