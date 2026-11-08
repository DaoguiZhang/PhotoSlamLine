# room1 pyramid 层数最小对照与三次交替重复

日期：2026-10-23  
范围：仅 room1；仅 `GausPyramid.num_sub_levels: 3 → 2`；冻结 A；不开发加速方法，不改源码，不重编译，不覆盖原配置/结果。

## 0. 保护现场

- 开始前 `pgrep` 无 `replica_rgbd` / `eval_checkpoint_snapshot` 进程；GPU 无本实验进程。
- 冻结 A 全部 `SHA256SUMS` 校验通过：
  - 二进制 `/workspace/code/PSL-perf/evidence/frozen_A_clean_20261023/bin/replica_rgbd`
  - SHA256 `5919857e7a89947b2c5256a8d47ccaf7ca6952f65a9f96218754543d9989d6ae`
- **实际动态库路径已核实**：
  - 不设 `LD_LIBRARY_PATH` 时，二进制 RUNPATH 会解析到 `/tmp/psl-densify-a/...`，这是诊断树，不是本轮允许的库。
  - 本轮每次运行均显式设置 `LD_LIBRARY_PATH=/workspace/code/PSL-perf/evidence/frozen_A_clean_20261023/bin`。
  - 该路径下 `libgaussian_mapper.so`、`libgaussian_viewer.so`、`libORB_SLAM3.so`、`libcuda_rasterizer.so`、`libsimple_knn.so`、`libimgui.so`、`libDBoW2.so`、`libg2o.so` 全部解析到冻结目录；每个 trial 的 `resolved_project_libs.txt` 已保存。
- 未修改源码，未重编译，未覆盖旧配置、日志或结果；所有新结果写入新根目录。

## 1. 配置隔离

从当前 A8 实际使用的配置逐字节复制：

- 基准：`evidence/room1_pyramid_ablation_20261023/config/room1_A8_pyramid3.yaml`
  - SHA256 `62fadbf35156c1bea52b0965b49be236d59d8260882961bc100ac9dc1fc129f7`
  - 与 `cfg/gaussian_mapper/RGB-D/Replica/A_baseline/room1.yaml` 相同
- 对照：`evidence/room1_pyramid_ablation_20261023/config/room1_A8_pyramid2.yaml`
  - SHA256 `8e3292bacecaa924820521232381d6f9f161069f64007219fa2648a655ea6462`

文本 diff 只有一行：

```diff
-GausPyramid.num_sub_levels: 3
+GausPyramid.num_sub_levels: 2
```

按键解析比较，唯一不同键为 `GausPyramid.num_sub_levels`：`3 → 2`。没有用旧配置整体替换，没有混入别名兼容或用户意图参数。

## 2. 六次干净运行

每轮严格串行交替，真实退出码均来自进程本身，未经过 `tee`。

| 轮次 | trial | 退出码 | 配置记录 SHA | 结果路径 | wall s | TotalOperation s |
|---|---|---|---:|---|---:|---:|
| 1 | r1_pyramid3 | 0 | `62fadbf3…` | `results/room1_pyramid_ablation_20261023/r1_pyramid3/room1` | 93 | 89.6996 |
| 1 | r1_pyramid2 | 0 | `8e3292ba…` | `.../r1_pyramid2/room1` | 91 | 88.4509 |
| 2 | r2_pyramid2 | 0 | `8e3292ba…` | `.../r2_pyramid2/room1` | 90 | 87.9682 |
| 2 | r2_pyramid3 | 0 | `62fadbf3…` | `.../r2_pyramid3/room1` | 94 | 90.8117 |
| 3 | r3_pyramid3 | 0 | `62fadbf3…` | `.../r3_pyramid3/room1` | 90 | 87.2290 |
| 3 | r3_pyramid2 | 0 | `8e3292ba…` | `.../r3_pyramid2/room1` | 92 | 88.3027 |

共同运行环境：

- `PHOTO_SLAM_LINE_MODE=2`
- `PHOTO_SLAM_LINE_LOOP=0`
- `PHOTO_SLAM_GBA_APPLY_CORRECTION=0`（Gaussian deformation 关闭）
- `PHOTO_SLAM_APPEND_STEPS=0`
- `PHOTO_SLAM_SEED=0`
- `PHOTO_SLAM_RASTER_BACKEND=A`
- `no_viewer`
- `STAGE_COUNTERS/ADAM_TOPO/ONLINE_CHECKPOINT/SHUTDOWN_DIAG/RASTER_AGG_SELFCHECK/RASTER_PROF/GPU_PROF/GBA_RECOVERY_DIAG` 全部 unset
- 未增加精修，未改变 loss、LR、增密或优化器顺序。

A8 历史运行只作背景观察，未计入本轮三次统计。

## 3. 全量原生训练 GS PSNR

指标只来自项目原有导出的 `psnr_gaussian_splatting.txt`；没有调用 Photo-SLAM-eval，没有新增评价器。

| trial | 记录数 | 首帧 | 全量均值 | median | 帧 std |
|---|---:|---:|---:|---:|---:|
| r1_pyramid3 | 181 | 33.3924636841 | 32.7391586725 | 33.1879501343 | 3.1767503077 |
| r1_pyramid2 | 176 | 29.8809394836 | 35.5814104297 | 36.2774047851 | 2.8689526580 |
| r2_pyramid2 | 169 | 34.2610893250 | 35.5815385220 | 35.8681640625 | 2.0943986974 |
| r2_pyramid3 | 188 | 28.9857463837 | 32.1732054365 | 32.3942661286 | 3.3075611156 |
| r3_pyramid3 | 173 | 31.6408252716 | 32.0827286671 | 31.7040157318 | 3.1888687418 |
| r3_pyramid2 | 174 | 33.5562477112 | 35.2808051164 | 35.4446239471 | 2.2461308889 |

配置聚合（三次运行的均值，`sample std` 为三次运行均值的样本标准差）：

| 配置 | 运行均值 | 三次运行 sample std |
|---|---:|---:|
| pyramid2 | **35.4812513560** | 0.1735915474 |
| pyramid3 | 32.3316975920 | 0.3557596246 |

全量运行均值差 `pyramid2 - pyramid3`：

- r1：`+2.8422517572 dB`
- r2：`+3.4083330855 dB`
- r3：`+3.1980764493 dB`
- 三轮平均：`+3.1495537640 dB`，三轮样本标准差 `0.2861430688 dB`

## 4. 按 `img_name` 配对的共同训练帧

映射来源：每个结果目录已有的 `natural_end/cameras.csv`，用 `fid → img_name` 映射；没有按 KF 序号逐行配对。

| 轮次 | 公共帧 | pyramid3 公共帧均值 | pyramid2 公共帧均值 | Δ(p2-p3) | 每帧差 std | p2 更好/更差 |
|---|---:|---:|---:|---:|---:|---:|
| r1 | 31 | 33.9498299629 | 36.4094483160 | +2.4596183531 | 3.7393948388 | 28 / 3 |
| r2 | 23 | 32.7472294517 | 35.9627470763 | +3.2155176246 | 2.1968865349 | 22 / 1 |
| r3 | 35 | 33.2828075409 | 36.0502713885 | +2.7674638476 | 2.3469495207 | 30 / 5 |

三轮共同帧差值的描述统计：均值 `+2.8141999417 dB`，三轮样本标准差 `0.3801106734 dB`。  
公共帧集合各轮不同，且共同帧不是独立重复；这里的标准差只描述帧间差异，不是三次独立实验的置信区间。

## 5. 时间、步数和模型规模

| trial | optimizer step calls | Adam group counter（不是步数） | KF | Gaussians | line Gaussians |
|---|---:|---:|---:|---:|---:|
| r1_pyramid3 | 10285 | 61134 | 181 | 515835 | 688 |
| r1_pyramid2 | 9226 | 54840 | 176 | 500226 | 972 |
| r2_pyramid2 | 9031 | 53682 | 169 | 474977 | 352 |
| r2_pyramid3 | 10285 | 61134 | 188 | 536884 | 407 |
| r3_pyramid3 | 9953 | 59160 | 173 | 505445 | 1179 |
| r3_pyramid2 | 9240 | 54924 | 174 | 500578 | 525 |

三次均值：

| 配置 | step calls | KF | Gaussians | TotalOperation s | wall s |
|---|---:|---:|---:|---:|---:|
| pyramid2 | 9165.7 | 173.0 | 491927 | 88.2406 | 91.0 |
| pyramid3 | 10174.3 | 180.7 | 519388 | 89.2468 | 92.3 |
| p2 - p3 | -1008.7 | -7.7 | -27461 | -1.006 | -1.33 |

解释边界：

- `optimizer step calls` 是调用计数，**不是** Adam 实际推进次数；Adam group counter 单列，未拿来当步数。
- 本轮没有把 `total_mapping_time.txt` 的整数毫秒累计当作精确完整更新耗时；性能只报告导出的 `TotalOperationTime` 与 wall time。
- KF/Gaussian 数不同导致耗时和步数不可作严格等规模性能比较；这里只报告代价。

## 6. pyramid 机制静态分析

### 已由代码确认

1. **实际分辨率列表**，room1 原图 `1200×680`：
   - `num_sub_levels=2`：`0.25 → 300×170`，`0.5 → 600×340`，然后全分辨率 `1200×680`。
   - `num_sub_levels=3`：`0.125 → 150×85`，`0.25 → 300×170`，`0.5 → 600×340`，然后全分辨率 `1200×680`。
   - 代码：`gaussian_mapper_line.cpp` 中 `kf_gaus_pyramid_factors_[l] = pow(0.5, N-l)`，camera 的宽高按该 factor 缩放；训练图像按该尺寸 `cv::resize`。
2. **每层预算与顺序**：
   - 配置 `GausPyramid.sub_level_times_of_use=15`。
   - 每个新 KF 初始化 `gaus_pyramid_times_of_use_ = [15, …, 15]`，长度等于层数。
   - `GaussianKeyframeLine::getCurrentGausPyramidLevel()` 从 index 0 开始，返回第一个非零层并减 1；全部耗尽后才返回 `num_sub_levels_`，进入全分辨率分支。
   - 因此每个 KF 的固定低分辨率预算是 `15×N`：pyramid2 为 30 次，pyramid3 为 45 次；顺序是从最低分辨率到 0.5，再到全分辨率。
3. **选择不是固定逐 KF 轮转**：
   - 训练视角由 `useOneTrainingErrorSlidingWindowKeyframe()` 在仍有 `remaining_times_of_use_` 的 KF 上按 `tracking_error_` 轮盘赌选择。
   - 选中一次同时消耗普通使用次数和该 KF 的当前 pyramid 层一次。
   - 因此实际某层用了多少步由误差采样决定，不是固定 15 步一块的静态序列。
4. **自然停止时确实可能留有 pyramid 预算**：
   - 本次没有达到 `max_num_iterations`（9–10k 远低于 50.1k）。
   - 使用已有 `used_times/keyframe_used_timesfinal.txt`：
     - pyramid2：169–176 个 KF 中，26–31 个总使用次数 <30；未消耗的低分辨率预算合计 152–186 次。
     - pyramid3：173–188 个 KF 中，83–94 个总使用次数 <45；未消耗的低分辨率预算合计 1107–1314 次。
   - 这说明自然结束时，pyramid3 有更多 KF 尚未走完低分辨率预算，更不可能进入全分辨率阶段。

### 计数推算（下界，不是直接逐层计数）

用最终每个 KF 的总使用次数，减去其固定低分辨率预算，得到最终 KF 的全分辨率选择次数下界：

- pyramid2：4113–4206 次，占全部 step calls 的 `44.6%–45.5%`
- pyramid3：3139–3292 次，占全部 step calls 的 `30.5%–32.9%`

这不是新增的逐步同步诊断，只是对已有 `keyframe_used_timesfinal.txt` 的离线统计；它排除了中途被 cull 的 KF，因此是下界，不是实际全分辨率计数器。

### 机制假设（未证明）

**假设：** pyramid3 多出的 `150×85` 层和每 KF 多出的 15 次低分辨率训练，挤占了全分辨率训练机会，导致细节/清晰度不足和 PSNR 下降。  
支持它的证据是代码路径、六次稳定对照和上述下界；但没有逐层训练计数，也没有把 KF 集合变化与 pyramid 层数变化完全分离。因此不能把“低分辨率训练比例”写成已证明的唯一因果机制。

## 7. 结论

1. **当前配置只改 pyramid 是否恢复质量？**
   - 是。只把 `GausPyramid.num_sub_levels` 从 3 改为 2，三次运行均值为 `35.4813 dB`，当前三次数值均值为 `32.3317 dB`，恢复 `+3.1496 dB`。
   - 这是画质恢复，不是历史 `34.6339 dB` 的精确复现。
2. **三轮方向是否一致？**
   - 是。全量均值差三轮为 `+2.8423 / +3.4083 / +3.1981 dB`；共同帧差值三轮为 `+2.4596 / +3.2155 / +2.7675 dB`，方向全部一致。
3. **时间与模型规模代价？**
   - 本轮 pyramid2 的 Gaussian 均值为 `491,927`，比 pyramid3 的 `519,388` 少 `27,461`（约 `-5.3%`）；KF 少 `7.7` 个。
   - 导出总运行时间均值 `88.24 s` vs `89.25 s`，wall `91.0 s` vs `92.3 s`；本轮没有看到额外时间或模型规模代价。
   - 因为 KF/Gaussian 数不同且运行方差存在，这不是严格性能结论。
4. **哪些机制已确认，哪些仍是假设？**
   - 已确认：分辨率列表、`15×N` 预算、低→高→全分辨率顺序、误差采样选择、自然停止时存在未消耗 pyramid 预算。
   - 仍是假设：额外低分辨率训练通过何种具体优化路径造成 PSNR 下降；缺少逐层训练计数。
5. **完整结果根目录、配置和复现命令：**
   - 结果根目录：`/workspace/code/PSL-perf/results/room1_pyramid_ablation_20261023`
   - 候选配置：`/workspace/code/PSL-perf/cfg/gaussian_mapper/RGB-D/Replica/A_baseline/room1_pyramid2_candidate.yaml`
     - SHA256 `8e3292bacecaa924820521232381d6f9f161069f64007219fa2648a655ea6462`
   - 复现脚本：`/workspace/code/PSL-perf/evidence/room1_pyramid_ablation_20261023/reproduce_room1_pyramid2.sh`
   - 直接命令：

```bash
TAG=my_room1_pyramid2 \
/workspace/code/PSL-perf/evidence/room1_pyramid_ablation_20261023/reproduce_room1_pyramid2.sh
```

- 明细 CSV：
  - `trial_metrics.csv`
  - `round_common_frame_stats.csv`
- 每个 trial 的 `command.txt`、`resolved_project_libs.txt`、`trial_status.txt` 与完整结果目录都已保留。
