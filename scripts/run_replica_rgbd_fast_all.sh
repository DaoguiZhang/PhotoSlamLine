#!/usr/bin/env bash
set -euo pipefail

cd /workspace/code/PSL-perf

dataset_root=/workspace/code/SEGS-SLAM/datasets/replica
base_cfg=results/quad_office0/config/gaussian_quad.yaml
scenes=(office0 office1 office2 office3 office4 room0 room1 room2)

# 开跑前检查全部场景，避免跑到一半才发现路径缺失。
test -x bin/replica_rgbd
test -f ORB-SLAM3/Vocabulary/ORBvoc.txt
test -f "$base_cfg"

for scene in "${scenes[@]}"; do
    if [[ ! -d "$dataset_root/$scene/results" ]]; then
        printf '缺少数据目录：%s\n' "$dataset_root/$scene/results"
        exit 1
    fi
    camera_cfg="cfg/ORB_SLAM3/RGB-D/Replica/$scene.yaml"
    if [[ ! -f "$camera_cfg" ]]; then
        printf '缺少相机配置：%s\n' "$camera_cfg"
        exit 1
    fi
done

mkdir -p results
batch_root=$(mktemp -d "$PWD/results/replica_fast_XXXXXXXX")
result_root="$batch_root/results/replica_rgbd_0"
mkdir -p "$result_root" "$batch_root/logs"

# 创建独立配置，不修改原文件。
python3 - "$base_cfg" "$batch_root/gaussian.yaml" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text()
changes = {
    "Optimization.refined_gaussian_max_iter_num": "0",
    "Optimization.weight_line_coherence": "0.001",
    "Optimization.weight_line_shape_ecc": "0.002",
    "Optimization.weight_line_shape_ori": "0.005",
    "Record.record_ground_truth_image": "0",
}
for key, value in changes.items():
    pattern = r"(?m)^" + re.escape(key) + r"\s*:[^\n]*$"
    text, count = re.subn(pattern, f"{key}: {value}", text)
    if count != 1:
        raise SystemExit(f"配置项缺失或重复：{key}")
pathlib.Path(sys.argv[2]).write_text(text)
PY

git rev-parse HEAD > "$batch_root/commit.txt"
git status --short > "$batch_root/source_status.txt"
sha256sum bin/replica_rgbd "$batch_root/gaussian.yaml" \
    > "$batch_root/checksums.txt"

# 关闭已知调试/性能分析开关。
unset CUDA_LAUNCH_BLOCKING
unset PHOTO_SLAM_RASTER_AGG_SELFCHECK
unset PHOTO_SLAM_RASTER_PROF
unset PHOTO_SLAM_GPU_PROF

export PHOTO_SLAM_RASTER_BACKEND=C
export PHOTO_SLAM_LINE_MODE=2
export PHOTO_SLAM_GBA_APPLY_CORRECTION=0
export PHOTO_SLAM_APPEND_STEPS=0

printf 'scene,exit_code,result_dir\n' > "$batch_root/status.csv"
printf '本批输出目录：%s\n' "$batch_root"

failures=0

for scene in "${scenes[@]}"; do
    scene_out="$result_root/$scene"
    scene_log="$batch_root/logs/$scene.log"
    mkdir -p "$scene_out"

    printf '\n开始：%s\n日志：%s\n' "$scene" "$scene_log"

    if bin/replica_rgbd \
        ORB-SLAM3/Vocabulary/ORBvoc.txt \
        "cfg/ORB_SLAM3/RGB-D/Replica/$scene.yaml" \
        "$batch_root/gaussian.yaml" \
        "$dataset_root/$scene" \
        "$scene_out" \
        no_viewer > "$scene_log" 2>&1
    then
        run_exit=0
        printf '完成：%s\n' "$scene"
    else
        run_exit=$?
        failures=$((failures + 1))
        printf '失败：%s，退出码=%s；继续下一个场景\n' \
            "$scene" "$run_exit"
    fi

    printf '%s,%s,%s\n' "$scene" "$run_exit" "$scene_out" \
        >> "$batch_root/status.csv"
done

printf '\n全部运行结束，失败场景数：%s\n' "$failures"
printf '结果目录：%s\n' "$result_root"
printf '状态汇总：%s/status.csv\n' "$batch_root"

if (( failures > 0 )); then
    exit 1
fi
