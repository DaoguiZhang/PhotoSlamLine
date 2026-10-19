#!/bin/bash

# ==========================================
# 单例测试脚本 (请在 Photo-SLAM 根目录下运行)
# ==========================================

# 我们拿 desk_3 场景来做测试
SEQ="mannequin_3"
DATASET_ROOT="/workspace/code/Photo-SLAM/datasets/eth3d"

echo "================================================="
echo "🚀 开始单测 ETH3D 场景: ${SEQ}"
echo "================================================="

../bin/eth3d_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/ETH3D/eth3d_${SEQ}.yaml \
    ../cfg/gaussian_mapper/RGB-D/ETH3D/eth3d_${SEQ}.yaml \
    ${DATASET_ROOT}/${SEQ} \
    ${DATASET_ROOT}/${SEQ}/associated.txt \
    ../results/eth3d_single_test/${SEQ} \
    no_viewer 
    
# 💡 提示：如果你想看看实时建图的弹窗画面，可以把上面那行 `no_viewer` 删掉。

echo "================================================="
echo "✅ 单测结束！请检查 ./results/eth3d_single_test/${SEQ} 文件夹是否有输出。"
echo "================================================="
