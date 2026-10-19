#!/bin/bash

# ==========================================
# 1. 定义 ETH3D 的所有 11 个场景
# ==========================================
SCENES=(
    "desk_3" "mannequin_1" "mannequin_3" "planar_2" "planar_3" 
    "plant_scene_1" "sfm_bench" "sfm_garden" "sfm_house_loop" 
    "sfm_lab_room_1" "table_7"
)

# ==========================================
# 2. 定义数据集在容器中的绝对基础路径
# ==========================================
DATASET_ROOT="/workspace/code/Photo-SLAM/datasets/eth3d"

# ==========================================
# 3. 外层循环：跑 5 次 (0 到 4)
# ==========================================
for i in 0 1 2 3 4
do
    echo "================================================="
    echo "🚀 开始执行第 $i 轮测试..."
    echo "================================================="

    # 4. 内层循环：遍历所有场景
    for SEQ in "${SCENES[@]}"
    do
        echo "🏃 正在处理场景: ${SEQ} (第 $i 轮)"

        ../bin/eth3d_rgbd \
            ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
            ../cfg/ORB_SLAM3/RGB-D/ETH3D/eth3d_${SEQ}.yaml \
            ../cfg/gaussian_mapper/RGB-D/ETH3D/eth3d_${SEQ}.yaml \
            ${DATASET_ROOT}/${SEQ} \
            ${DATASET_ROOT}/${SEQ}/associated.txt \
            ../results/eth3d_rgbd_${i}/${SEQ} \
            no_viewer
            
        echo "✅ 场景 ${SEQ} 执行完毕！"
        echo "-------------------------------------------------"
    done
done

echo "🎉 所有 5 轮 ETH3D 测试已全部运行完毕！"
