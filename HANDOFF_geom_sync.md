# HANDOFF — geom-sync / gba-recovery（v1 + 机制诊断）

状态：**v1 受控实验为负；本轮机制诊断未发现新的生产实现错误（LR 恢复
缺口的诊断缺陷已修），G 变形收益不稳定，结论维持负结果。GBA 变形改为
可选（`PHOTO_SLAM_GBA_APPLY_CORRECTION=0` 关闭，默认仍开）。**
HEAD：本次提交。工作树干净（构建产物一并提交）。

## 机制诊断（本轮，v1 结果不变）

- LOOP 事件范围采集原先落在 point-only 处理器与 `#if 0` 副本中，导致
  `available=0`；已移入 live `combineMappingOperations_withLine()` 的
  `LoopClosingBA` 分支，现报告 `available=1, kfs=242`。早期运行的 LOOP
  数据缺失，明确标注，不补造。
- 重复性：step0 参数/Adam/Adam step/LR/调度/视图序列/渲染逐位相同；
  first update rotation 差 6.3e-3，step10 rotation 3.6e-2、scaling 1.8e-2、
  xyz 7.6e-4。`post_step_param_max_abs_diff≈0.039` 是 10 步累积，**不是
  ULP 级**，也不构成恢复失败的证据。first-backward 梯度逐位相同但 first
  update 仍有差异，说明旋转组的更新路径未被指纹完全覆盖（仪器缺口，未
  发现恢复错误）。`quat_angle_deg` 的 180 是哨兵值，忽略。
- step0 分解（24 视角，同冻结态）：MeansOnly +0.030 dB、14/24 变好、
  最差 −1.17 dB；Full +0.030 dB、最差 −1.18 dB；MeansOnly 与 Full
  最大差 0.013 dB → 即时效应由中心平移主导，仅作诊断，不作为默认修复。
- 范围：GBA rot 0.331/0.573/0.933 deg（mean/p95/max），trans
  0.0190/0.0419/0.0431 m，预测像素位移 mean 1.19–3.21 px、p95 7.43、
  max 17.3；LOOP rot 0.485/1.229/1.282 deg，trans 0.0248/0.0469/0.0476 m，
  像素 mean 0–8.04、p95 11.81、max 25.7。LOOP 只大约 1.5–2.5 倍（同量级），
  暂不做 LOOP 控制的 500 步矩阵。
- 未关联行：本轮 0；上一条事件 35 行且 35/35 在全部评价视角可见。v1 日志
  中的 169 行（参考 KF 被事件剔除）本轮未复现，未重锚、未新增 CUDA 统计。
- 逐视角 ΔPSNR 与像素位移无单调关系（view 11 位移最大且 −1.17 dB；
  view 21 位移大却 +0.33 dB），故不能只用均值位移判断影响，也不能只用
  全图均值 PSNR 排除局部收益/损失。
- 回归：默认运行仍变形（moved=82825，计数 82842→82842）；
  `PHOTO_SLAM_GBA_APPLY_CORRECTION=0` 时 moved=0、相机位姿照常更新。

报告与原始数据：`reports/gba_recovery_diag_report.md`、
`reports/gba_recovery_diag/`（CSV/TXT，不含大图）。

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
- 重复检查：step0 参数/Adam/调度/渲染完全一致；训练后张量差异为优化更新
  路径的累积漂移（0.039 量级，非 ULP 级，见机制诊断一节）。
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

