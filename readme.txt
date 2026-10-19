

PHOTO_SLAM_LINE_MODE=2    # 0=纯点 1=仅线前端 2=完整点线
PHOTO_SLAM_LINE_LOOP=0    # 0=标准点回环（已验收）
PHOTO_SLAM_SHADOW_LINE=0  # 1=跳过 MapLine 建/删/融合
PHOTO_SLAM_SLAM_ONLY=0    # 1=跳过 Gaussian（仅 tum_rgbd.cpp 支持）



=============================replica d================================================
这是replica d 数据的脚本运行
先运行完整线回环的单序列冒烟：
RUNS=1 \
SEQS="office0" \
RESULT_SET=replica_rgbd_lc_smoke \
PHOTO_SLAM_LINE_LOOP=2 \
bash scripts/replica_rgbd.sh

成功后运行全部数据：
RUNS=5 \
RESULT_SET=replica_rgbd_lc \
PHOTO_SLAM_LINE_LOOP=2 \
bash scripts/replica_rgbd.sh


如果想复现原来的标准点回环结果，则把最后一项改成：
PHOTO_SLAM_LINE_LOOP=0


===========================replica mono =======================================
运行所有的relica mono的数据脚本
RUNS=5 \
SEQS="office0 office1 office2 office3 office4 room0 room1 room2" \
RESULT_SET=replica_mono_lc \
PHOTO_SLAM_LINE_MODE=2 \
PHOTO_SLAM_LINE_LOOP=2 \
PHOTO_SLAM_SHADOW_LINE=0 \
PHOTO_SLAM_SLAM_ONLY=0 \
PHOTO_SLAM_DEBUG_MONO_INIT=0 \
bash scripts/replica_mono.sh

后台运行：
mkdir -p results

nohup env \
RUNS=5 \
SEQS="office0 office1 office2 office3 office4 room0 room1 room2" \
RESULT_SET=replica_mono_lc \
PHOTO_SLAM_LINE_MODE=2 \
PHOTO_SLAM_LINE_LOOP=2 \
PHOTO_SLAM_SHADOW_LINE=0 \
PHOTO_SLAM_SLAM_ONLY=0 \
PHOTO_SLAM_DEBUG_MONO_INIT=0 \
bash scripts/replica_mono.sh \
> results/replica_mono_lc_batch.log 2>&1 &


运行replica mono 数据的单个数据office
RUNS=1 \
SEQS="office0" \
RESULT_SET=replica_mono_lc_smoke \
PHOTO_SLAM_LINE_MODE=2 \
PHOTO_SLAM_LINE_LOOP=2 \
PHOTO_SLAM_SHADOW_LINE=0 \
PHOTO_SLAM_SLAM_ONLY=0 \
PHOTO_SLAM_DEBUG_MONO_INIT=1 \
bash scripts/replica_mono.sh




==========================tum_mono rgbd================================
freiburg3_long_office_household 是当前最重要的真实回环序列：
RUNS=1 \
SEQS="freiburg3_long_office_household" \
RESULT_SET=tum_rgbd_lc_smoke \
PHOTO_SLAM_LINE_MODE=2 \
PHOTO_SLAM_LINE_LOOP=2 \
PHOTO_SLAM_SHADOW_LINE=0 \
PHOTO_SLAM_SLAM_ONLY=0 \
PHOTO_SLAM_NO_PACING=0 \
PHOTO_SLAM_DEBUG_LINE_LOOP=1 \
bash scripts/tum_rgbd.sh

检查结果：
tail -100 \
results/tum_rgbd_lc_smoke/rgbd_dataset_freiburg3_long_office_household/run.log
重点确认：
- EXIT=0；
- Loop detected；
- GBA 完成并写回；
- 线 GBA 顶点和边数大于 0；
- 无 NaN、Segmentation fault、Bus error；
- 如果 linePairs=0，应明确记录自动回退到点 Sim3。


运行全部 TUM RGB-D
RUNS=5 \
SEQS="freiburg1_desk freiburg2_xyz freiburg3_long_office_household" \
RESULT_SET=tum_rgbd_lc \
PHOTO_SLAM_LINE_MODE=2 \
PHOTO_SLAM_LINE_LOOP=2 \
PHOTO_SLAM_SHADOW_LINE=0 \
PHOTO_SLAM_SLAM_ONLY=0 \
PHOTO_SLAM_NO_PACING=0 \
PHOTO_SLAM_DEBUG_LINE_LOOP=0 \
bash scripts/tum_rgbd.sh

后台运行：
mkdir -p results

nohup env \
RUNS=5 \
SEQS="freiburg1_desk freiburg2_xyz freiburg3_long_office_household" \
RESULT_SET=tum_rgbd_lc \
PHOTO_SLAM_LINE_MODE=2 \
PHOTO_SLAM_LINE_LOOP=2 \
PHOTO_SLAM_SHADOW_LINE=0 \
PHOTO_SLAM_SLAM_ONLY=0 \
PHOTO_SLAM_NO_PACING=0 \
PHOTO_SLAM_DEBUG_LINE_LOOP=0 \
bash scripts/tum_rgbd.sh \
> results/tum_rgbd_lc_batch.log 2>&1 &

5. 评价注意事项
TUM RGB-D 是有尺度传感器，主要报告 SE(3) 对齐 ATE：
evo_ape tum groundtruth.txt CameraTrajectory_TUM.txt -a

Sim(3) 可以作为辅助诊断：
evo_ape tum groundtruth.txt CameraTrajectory_TUM.txt -as
不要只报告 Sim(3)，否则可能隐藏尺度错误。


=======================tum mono=====================================
grep refined_gaussian_max_iter_num \
cfg/gaussian_mapper/Monocular/TUM/tum_mono.yaml

先运行 fr3 Mono
RUNS=1 \
SEQS="freiburg3_long_office_household" \
RESULT_SET=tum_mono_lc_smoke \
PHOTO_SLAM_LINE_MODE=2 \
PHOTO_SLAM_LINE_LOOP=2 \
PHOTO_SLAM_SHADOW_LINE=0 \
PHOTO_SLAM_SLAM_ONLY=0 \
PHOTO_SLAM_NO_PACING=0 \
PHOTO_SLAM_DEBUG_MONO_INIT=1 \
PHOTO_SLAM_DEBUG_LINE_LOOP=1 \
bash scripts/tum_mono.sh
查看日志：
tail -100 \
results/tum_mono_lc_smoke/rgbd_dataset_freiburg3_long_office_household/run.log

重点确认：
- EXIT=0；
- Mono 初始化成功；
- 无 NaN、Segmentation fault、Bus error；
- 是否检测到回环；
- 线 GBA 是否加入线顶点和线边；
- 如果 linePairs=0，是否自动回退到点 Sim3；
- 是否出现 ResetActiveMap 或第二地图。

运行全部 TUM Mono
RUNS=5 \
SEQS="freiburg1_desk freiburg2_xyz freiburg3_long_office_household" \
RESULT_SET=tum_mono_lc \
PHOTO_SLAM_LINE_MODE=2 \
PHOTO_SLAM_LINE_LOOP=2 \
PHOTO_SLAM_SHADOW_LINE=0 \
PHOTO_SLAM_SLAM_ONLY=0 \
PHOTO_SLAM_NO_PACING=0 \
PHOTO_SLAM_DEBUG_MONO_INIT=0 \
PHOTO_SLAM_DEBUG_LINE_LOOP=0 \
bash scripts/tum_mono.sh

后台运行：
mkdir -p results

nohup env \
RUNS=5 \
SEQS="freiburg1_desk freiburg2_xyz freiburg3_long_office_household" \
RESULT_SET=tum_mono_lc \
PHOTO_SLAM_LINE_MODE=2 \
PHOTO_SLAM_LINE_LOOP=2 \
PHOTO_SLAM_SHADOW_LINE=0 \
PHOTO_SLAM_SLAM_ONLY=0 \
PHOTO_SLAM_NO_PACING=0 \
PHOTO_SLAM_DEBUG_MONO_INIT=0 \
PHOTO_SLAM_DEBUG_LINE_LOOP=0 \
bash scripts/tum_mono.sh \
> results/tum_mono_lc_batch.log 2>&1 &

监控：
tail -f results/tum_mono_lc_batch.log

5. Mono 评价注意事项
TUM Mono 没有绝对尺度，必须使用 Sim(3) 对齐：
evo_ape tum groundtruth.txt CameraTrajectory_TUM.txt -as
不要把无尺度的 SE(3) ATE 作为 Mono 主指标。
同时建议报告：
- Sim(3) ATE；
- RPE translation/rotation；
- 匹配帧覆盖率；
- 地图数量；
- ResetActiveMap 次数；
- 初始化帧和初始化尺度；
- 最终 KeyFrame、MapPoint、MapLine 数量。
结果目录为：
results/tum_mono_lc/run_1/rgbd_dataset_freiburg1_desk/
results/tum_mono_lc/run_1/rgbd_dataset_freiburg2_xyz/
results/tum_mono_lc/run_1/rgbd_dataset_freiburg3_long_office_household/




室外场景：

# 1) 先编译（新目标）
cmake --build /workspace/code/Photo-SLAM-L/build --target tandt_mono -j

# 2) 直接跑 train（已有 rgb.txt，500 帧）
bash scripts/tandt_mono.sh

# 3) 其他场景（自动生成 rgb.txt 后运行）
RUNS=1 SEQS="train family truck" PHOTO_SLAM_NO_PACING=1 bash scripts/tandt_mono.sh

# 4) SLAM-only / 纯点对照
SEQS="train" PHOTO_SLAM_SLAM_ONLY=1 bash scripts/tandt_mono.sh
SEQS="train" PHOTO_SLAM_LINE_MODE=0 bash scripts/tandt_mono.sh

# 单独的 rgb.txt 生成（默认 FPS=5，仅缺文件时生成；FORCE=1 覆盖）
bash scripts/make_tandt_rgb_txt.sh
SEQS="temple" FORCE=1 FPS=5.0 bash scripts/make_tandt_rgb_txt.sh

# 旧的单序列脚本也可用（在 scripts/ 目录下运行）
cd scripts && bash tandt_mono_test_single.sh