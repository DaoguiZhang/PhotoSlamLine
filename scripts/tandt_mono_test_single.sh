#!/bin/bash

# ==========================================
# 单例测试脚本 (请在 Photo-SLAM 根目录下运行)
# ==========================================

# 我们拿 train, garden_large_1 场景来做测试
SEQ="train"
DATASET_ROOT="/workspace/code/Photo-SLAM/datasets/tandt_db"

echo "================================================="
echo "🚀 开始单测 tandt 场景: ${SEQ}"
echo "================================================="

# 🌟 已经为你移除了不属于单目的 associated_2000.txt 参数
../bin/tandt_mono \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/Monocular/tandt/tandt_${SEQ}.yaml \
    ../cfg/gaussian_mapper/Monocular/tandt/tandt_${SEQ}.yaml \
    ${DATASET_ROOT}/${SEQ} \
    ../results/tandt_mono_single_test/${SEQ} \
    no_viewer
    
# 💡 提示：如果你想看看实时建图的弹窗画面，可以把上面那行 `no_viewer` 删掉。

echo "================================================="
echo "✅ 单测结束！请检查 ./results/tandt_mono_single_test/${SEQ} 文件夹是否有输出。"
echo "================================================="
