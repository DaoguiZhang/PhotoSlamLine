#!/bin/bash

# ==========================================
# 单例测试脚本 (请在 Photo-SLAM 根目录下运行)
# ==========================================

# 我们拿 campus_small_1, garden_large_1 场景来做测试
SEQ="garden_large_1"
DATASET_ROOT="/workspace/code/Photo-SLAM/datasets/RoverData"

echo "================================================="
echo "🚀 开始单测 rover 场景: ${SEQ}"
echo "================================================="

../bin/rover_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/Rover/rover_${SEQ}.yaml \
    ../cfg/gaussian_mapper/RGB-D/Rover/rover_${SEQ}.yaml \
    ${DATASET_ROOT}/${SEQ} \
    ${DATASET_ROOT}/${SEQ}/associated_2000.txt \
    ../results/rover_single_test/${SEQ} \
    no_viewer 
    
# 💡 提示：如果你想看看实时建图的弹窗画面，可以把上面那行 `no_viewer` 删掉。

echo "================================================="
echo "✅ 单测结束！请检查 ./results/rover_single_test/${SEQ} 文件夹是否有输出。"
echo "================================================="
