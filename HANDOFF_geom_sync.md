# HANDOFF — geom-sync / gba-recovery-v1

状态：**gba-recovery-v1 受控实验完成；结果为负：PoseAndMap 在该事件上
即时质量略差、相同步数无稳定收益、计入同步耗时后不占优。**
HEAD：本次提交。工作树干净（构建产物一并提交）。

## 本轮新增

- `GaussianMapperLine` 记忆体快照：Gaussian 全参数、6 组 Adam
  `(step, exp_avg, exp_avg_sq)`、`iteration_`、`active_sh_`/`default_sh_`、
  EMA、`kfs_used_times_`、每 KF 位姿/剩余使用次数/金字塔预算/tracking error。
- 事件内两臂实验（env `PHOTO_SLAM_GBA_RECOVERY=<dir>`）：
  PoseOnly 与 PoseAndMap 使用同一快照、同一事件位姿、同一 500 步训练视图序列；
  C 后端、结构损失 0、opacity reset 0、Gaussian 数冻结；先 10 步重复检查，
  再按 1→2 / 2→1 顺序各跑一对。
- 评价：PSNR/SSIM/L1（24 个固定视角，训练不使用）、离线 LPIPS、残差图；
  记录 sync/train/total 时间与逐帧 CSV。
- `applyGbaCorrections` 统计增加 `nonzero_moved_rows`、`translation_mean/max`；
  GBA-Apply 日志增加参考 KF 分类（in_event / scene_not_event / absent_scene /
  unassigned / newer_than_event）。
- 脚本：`scripts/build_two_level.sh`、`scripts/lpips_gba_recovery.py`、
  `scripts/plot_gba_recovery.py`。

## 关键结果（同一事件，evt=1，245 KFs）

- 更正：moved=104159，nonzero=99881，line=238，mean 位移 3.3 mm，
  max 172 mm；缺失关联 169 行全部是“场景内但事件中标记 bad/剔除的参考 KF”。
- 重复检查：step0 参数/Adam/调度/渲染完全一致；训练后张量因 CUDA 原子归约
  顺序存在 ULP 级漂移（已如实记录）。
- 顺序 1：step0 ΔPSNR −0.0231 dB，ΔSSIM −0.00123，ΔLPIPS +0.00111；
  step500 ΔPSNR −0.0485 dB。顺序 2 同向（step500 −0.0023 dB，属波动）。
- 总恢复时间：PoseOnly 3.694 s，PoseAndMap 3.787 s（sync 0.016 vs 0.107 s）。
- 结论：不称改善、不称加速；保留正确性修复，不追加模块追求正结果。

## 产物

- `reports/gba_recovery_v1_report.md`（方法、原始表、决策、限制）
- `reports/gba_recovery_v1/summary.csv`、`lpips.csv`、`repeat_check.csv`、
  `common_state.txt`、`event_evt1.txt`、`<arm>/metrics.csv`、逐帧 CSV
- `reports/gba_recovery_v1/curves/{psnr,ssim,lpips}_vs_step.png`、
  `psnr_vs_total_time.png`

## 限制

- 单事件、单序列、各顺序一对；24 固定评价视角；无 depth 输出，故无深度误差。
- LPIPS 因本机 cuDNN 版本冲突只能用 CPU 计算（在已保存 PNG 上）。
- 单参考 KF 刚性近似，不精确复现 GBA 的逐点调整。

## 下一条具体命令（复现实验）

```bash
cd /workspace/code/PSL-perf
./scripts/build_two_level.sh
PHOTO_SLAM_RASTER_BACKEND=C PHOTO_SLAM_LINE_LOSS_EVERY=0 \
PHOTO_SLAM_GBA_RECOVERY=abl_out/gba_recovery_v1 \
PHOTO_SLAM_GBA_RECOVERY_STEPS=500 \
PHOTO_SLAM_GBA_RECOVERY_EVAL_STEPS=0,10,50,100,250,500 \
PHOTO_SLAM_GBA_RECOVERY_VIEWS=24 \
  bin/tum_rgbd ORB-SLAM3/Vocabulary/ORBvoc.txt \
  cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
  cfg/gaussian_mapper/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
  /workspace/code/SEGS-SLAM/datasets/tum/rgbd_dataset_freiburg3_long_office_household \
  cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg3_long_office_household.txt \
  /tmp/gba_recovery_run no_viewer 2>&1 | tee /tmp/gba_recovery.log
python3 scripts/lpips_gba_recovery.py abl_out/gba_recovery_v1 --device cpu
python3 scripts/plot_gba_recovery.py abl_out/gba_recovery_v1
```

