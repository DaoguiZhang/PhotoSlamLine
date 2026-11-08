# room1 LOOP ↔ Gaussian mapping 坐标一致性最小修复

## 1. 根因（代码位置）

基线 commit `ec85e80886`，文件 `src/gaussian_mapper_line.cpp`：

- `7073`：`const bool do_correct = (transNorm > 0.10m) || (rotRad > 2deg);`
- `7075-7113`：只有 `do_correct` 才调用
  `gaussians_->scaledTransformVisiblePointsOfKeyframe(...)`。
- `7115`：`pkf->setPose(...)` **在 do_correct 之外**，无条件执行。

因此当 loop 修正量小于阈值时：**相机位姿更新，Gaussian map 不动**。

即使 `do_correct` 为真，旧入口也只处理
`point_not_transformed && point_unstable(exist_since_iter == kf_creation_iter) && visible`
的行（`src/operate_points.cu:95` 的 `scaleAndTransformThenMarkVisiblePoints`）。
不可见行、稳定行、后续新建行会永久停留在旧坐标系。

事件实测（room1，backend C）：

- 117 个 scene KF，113 个位姿变化；
- max translation `0.0361479 m`，max rotation `0.899438 deg`；
- 均低于旧阈值 `0.10 m / 2 deg` → legacy 117/117 跳过 map，仅相机更新。

## 2. 修复

修改文件：

- `src/gaussian_mapper_line.cpp`
- `src/gaussian_model_line.cpp`
- `include/gaussian_mapper_line.h`
- `include/gaussian_model_line.h`
- 新增 `include/loop_event_key.h`（顺序无关 payload fingerprint）
- 新增 `tests/test_loop_coord_sync.cpp`（Gba/LoopKey 契约）
- 新增 `scripts/repro_loop_coord_fix.sh`
- 证据 `evidence/loop_coord_fix_r5/`

核心：

1. `LoopClosingBA` 不再用可见性/创建迭代/阈值决定是否修正。
2. 为每个事件 KF 构造 `G = Tcw_after^-1 * Tcw_before`。
3. 调用 `GaussianModelLine::applyGbaCorrections(...)`，按
   `ref_kf_id_ == kfid && coord_event_id_ < coord_evt` 选行：
   - 每个 Gaussian 只由它的 reference KF 修正一次；
   - 中心/`xyz_init_` anchor：`p' = s * (R p) + t`；
   - 四元数：`q' = qG * q`；
   - line `line_dir_w_`：`d' = R d`；
   - `scaling_`（log 尺度）：`+= log(s)`；
   - xyz Adam 一阶矩同步旋转/缩放；
   - `line cache` 失效重建；
   - 行数不变（`TORCH_CHECK`）。
4. 先改 map，再改相机位姿；`last_applied_gba_event_` 使用模型内单调
   `coord_event_id`（LOOP 的 ORB payload evt 为 0，不能直接当版本号）。
   GBA payload 去重改用 mapper 的 `last_gba_payload_event_`。
5. GBA 默认变形仍为 disabled，未改变用户要求保持的行为。

## 3. 同一事件质量（24 固定视角，零优化步）

事件：ORB payload `evt=0`，mapper coordinate `evt=1`。

`evidence/loop_coord_fix_r5/summary.csv`：

| arm | mean PSNR | Δ vs before |
|---|---:|---:|
| before camera + before model | 30.58399487 | 0 |
| after camera + before model（legacy gate skipped） | 23.47381488 | **−7.110179981** |
| after camera + corrected model（fixed） | 28.98292383 | **−1.60107104** |

恢复 `+5.50910894 dB`（恢复 77.5% 的即时跌幅）；相对修正前仍有
`−1.60107104 dB` 的即时残差。23/24 视角移动。Worst fixed view fid 55：
`−5.16902 dB`。这是 **r5 单事件质量**数据，PSNR 为普通定义。

覆盖日志：

```
[LineLoop-Gaussian] candidates=1262 correctedLineGaussians=1262
 kfsTotal=117 kfsWithPkf=117 kfsCorrected=117 kfsSkippedByGate=0
 corrected=386317 skippedBad=0 skippedNull=0 duplicateSkipped=0
 nonFinite=0 maxTranslation=0.0361479 maxRotationDeg=0.899438
 meanTranslation=0.0172877 meanRotationDeg=0.605443 meanScaleRatio=1
[LoopSync] evt=1 kfs=117 kfsPoseChanged=113 noScene=0
 movedRows=386317 nonzeroMovedRows=351371 lineRowsMoved=1262
 unassignedRows=0 missingEventRows=3801 staleRows=0 scale=1
 count=390118->390118
```

`missingEventRows=3801`：其 `ref_kf_id_` 是不在本 LOOP payload 里的 scene KF，
保持旧坐标并显式报告，不做猜测。

## 4. 真实 LOOP 的 Legacy vs Fixed 100 步恢复

`evidence/loop_coord_fix_r5/recovery/loop_recovery_summary.csv`，同一 pre-LOOP
状态、同一 LOOP 后相机、同一训练视图序列、固定 Gaussian 数量；相机指纹
`340f17cbe078514f` 在四个 arm 完全相同。

| arm | step | ordinary PSNR | **psnr_gaussian_splatting** | SSIM |
|---|---:|---:|---:|---:|
| Legacy（PoseOnly） | 0 | 28.1042 | **28.1116** | 0.8836 |
| Legacy（PoseOnly） | 100 | 31.4389 | **31.4499** | 0.9154 |
| Fixed（PoseAndMap） | 0 | 28.9032 | **28.9127** | 0.8989 |
| Fixed（PoseAndMap） | 100 | 31.5392 | **31.5506** | 0.9164 |

必须区分两个量：

1. **同事件即时修正效果（r5，单事件质量）**：修正前 `28.98292383` 对
   before-camera/before-model `30.58399487`，普通 PSNR 残差
   `−1.60107104 dB`。
2. **Legacy vs Fixed 100 步收益（r2/recovery）**：step0 Fixed `28.9127`，
   Legacy `28.1116`，Fixed−Legacy = `+0.8011 dB`；step100 Fixed `31.5506`，
   Legacy `31.4499`，Fixed−Legacy = `+0.1007 dB`。这两个数是 **Fixed 相对
   Legacy 的收益，不是相对修正前残差**。Legacy 臂的模型未修正，所以它同时
   近似等于本 recovery 的修正前模型；不能把 `+0.1007` 说成“残差缩小到
   0.1007”。

两组都稳定上升，无 NaN/行数变化；相机指纹完全相同
（`340f17cbe078514f`），说明最终相机一致。该 recovery 事件为 114 KF /
111 位姿变化，coverage 行数 385356，`refInEvent=373356`、
`refSceneNotEvent=12000`、`refAbsentScene=0`、`refUnassigned=0`、
`refNewerThanEvent=0`。r5 的单事件质量行数/覆盖是另一组数
（390118 行、`missingEventRows=3801`），两者不可混用。

### 版本/幂等

- `[LoopReplay] evt=1 same_moved=0 same_stale=373356 older_moved=0 older_stale=373356`。
- 同一个 payload/version 第二次消费不再移动 Gaussian；旧版本也不移动。
- **新行 stamp/replay 测试已实际通过**（`src-fix` 的
  `runGbaApplyContractTest`）：在 event 3 后插入一行 `ref_kf_id=10`，该行
  `coord_event_id_` 等于 `lastAppliedGbaEvent()==3`；回放 event 1
  `moved_rows=0` 且新行不动；随后真正的新 event 4 能移动该行。
  实测输出：
  `[GbaContract] ... scale_apply=1 stamp=1 new_replay=1 new_event=1 : PASS`。
- **顺序无关的 LoopKey helper 已实际通过**：
  `[LoopKeyContract] order=1 map=1 match=1 : PASS`。
- 跨 map 目前有 `mapGen` 检查、payload 全 scene 缺失保护和
  `loop_event_key::map_matches` 测试；但**没有构造完整 payload + 跨 map
  的端到端回归**，这一点仍标记为未完成。

## 5. 干净 room1 回归

诊断开关 `PHOTO_SLAM_EVENT_QUALITY_DIR` / `PHOTO_SLAM_COMMON_ABC_DIR` 会运行
反事实 arm，**最终 PSNR 不可用于回归**（这类 run 末端约 18 dB，是诊断本身造成的）。
干净回归必须不设置这两个变量。

干净 fixed run（**真实 LOOP 触发，无诊断开关**，第 4 次干净回归）：
- `optimizer_steps=9146`，`gaussians=491017`，`line_gaussians=1143`，
  `mapping_seconds=71.373`，`kfs=180`
- `psnr_gaussian_splatting`: n=180，mean `29.2972`，last `25.8999`
- 事件：118 个 KF，117 个位姿变化；max `0.0150452 m` / `0.176331 deg`
  （仍低于旧阈值，legacy 会全部跳过 map）；修正 `373284` 行，
  `missingEventRows=4927`，行数 `378211 -> 378211`。
- 旧 current C 对照（不同 run）：mean `26.5209`，last `26.0256`
- old A（不同 run）：mean `31.5032`，last `24.7688`

这是干净 LOOP 回归：fixed 的 mean 比 current C 高 `+2.7763 dB`，last 低
`−0.1257 dB`。它证明修复后真实 LOOP 能触发且不导致终端崩溃/明显退化；
但这是不同 run 的轨迹，不能把均值差全部归因于修复。修复的逐步因果证据仍是
同事件 step0 对照。

## 6. 未解释/剩余差距

- 同事件即时残差：r5 普通 PSNR 为 `−1.6011 dB`（Fixed 28.9829 对 before
  30.5840）。r2 的 `+0.8011 / +0.1007` 是 Fixed 相对 Legacy（未修正模型）的
  **收益**，不是残差；r2 的即时残差应从它自己的 before-model 单独计算，
  本次报告不把这两个数重复解释成“残差恢复”。
- r2/recovery 覆盖分类：`refSceneNotEvent=12000`、`refAbsentScene=0`、
  `refUnassigned=0`；r5 单事件是 `missingEventRows=3801`。两者均表明
  missing 主要是 **LOOP payload 未包含该参考 KF**，不是参考 KF 被剔除，
  也不是未知 anchor；未更正行保持原值。
- 可见性线索（`missing_visibility.csv`）：该事件 24 个评估视图中 fid56 有
  `11667` 个 missing-ref 行可见（fid41/46/51/56/62/67/72/77/82 都在 6k–11.7k）。
  这是**可见性计数，不是遮挡合成后的贡献**；fid55 不在该 24 视图中，因此
  fid55 的缺失行可见性没有直接测量。
- Adam 二阶矩未做各向异性旋转；这是已有文档化近似。

### 6.1 非零 GBA 的 100 步输入盘点

**结论：现有证据不能恢复非零 GBA 的 100 步，缺少事件时刻的完整状态。**
没有用当前模型/新优化器“补跑”，因为那会是新的初始化，不是同事件延续。
完整清单：`evidence/loop_coord_fix_r5/gba_inventory.md`。

| 来源 | 已有 | 缺失/为何不能用 |
|---|---|---|
| r5 `gba_evt1` | 24 视角 step0 的 before/pose/corrected 渲染与 `summary.csv` | 没有 pre-GBA 参数张量、完整 Adam 状态、事件时刻相机、训练序列、RNG/迭代状态；只有 step0 指标 |
| r2/r4 `gba-recovery-*` | `common_state.txt`、`summary.csv`、事件 scope、100 步 LOOP recovery | GBA `world_deltas` 全零（`PoseAndMap moved=0`），不是非零 GBA；模型只在 run 末保存，GBA 发生在 final refinement 之后 |
| r3/r5/r6 | run 日志/最终结果 | 未产生可用非零 GBA 事件 |

要恢复 100 步，至少需要事件瞬间的：6 组 Gaussian 参数（含 line 字段）、
完整 Adam 一阶/二阶矩与 step 计数、所有 KF 的 `Tcw`、`ref_kf_id_`/
`coord_event_id_`/line cache、GBA payload 的 `world_deltas` 与 map generation、
当前迭代/active SH/分辨率/金字塔预算、训练视图序列与 RNG 状态、图像路径和
内参。现有 artifact 均未同时保存这些。r5 的 step0 仍是有效证据：before
`28.2804`，pose-only `24.0143`（`−4.2662`），corrected map `29.2305`
（`+0.9501`，普通 PSNR）。

### 6.2 room1 A/C 诊断对（同一候选 fix，GBA 变形 disabled）

命令使用同一 `src-fix`、同一 ORB/Gaussian 配置、`PHOTO_SLAM_SEED=0`、
`PHOTO_SLAM_GBA_APPLY_CORRECTION=0`、`PHOTO_SLAM_APPEND_STEPS=0`、
`PHOTO_SLAM_LINE_MODE=2`，仅 backend A/C 不同。两端都 `EXIT=0`。

| backend | KF | GS 数 | line GS | optimizer steps | native PSNR mean/last | GS PSNR mean/last | LOOP | GBA |
|---|---:|---:|---:|---:|---:|---:|---|---|
| A | 183 | 546704 | 2725 | 10289 | 30.6536 / 28.3001 | 30.6610 / 28.3001 | 0 | 0 |
| C | 182 | 472461 | 818 | 9189 | 29.9071 / 24.7993 | 29.9150 / 24.8131 | 1 | 1（零位移） |

C 的事件位置：LOOP 日志为 `Loop: mpCurrentKF -> 124`，随后在
`ProcessNewKeyFrame mnId 125` 消费；GBA 在同一 KF 125 的 LBA 结束后、KF 126
之前。日志没有 wall-clock 时间戳，只有 KF/frame 序列位置（事件前最后
`Frame id: 1300`），因此报告 KF 位置而不伪造秒数。A 两组均无 LOOP/GBA。

C 的 LOOP：`evt=1 kfs=120 kfsPoseChanged=116 movedRows=371525
 missingEventRows=4703 count=376228->376228`；相机与地图均实际改变。
C 的 GBA：`evt=1 deltas=121 translation_max=0.00000 moved=0`，即 payload
存在但位移为零；因为 `GBA_APPLY_CORRECTION=0`，地图本来也不会动。日志里的
`APPLIED` 只表示 payload 被接受，不代表非零修正。

A 没有 LOOP/GBA，C 触发了 LOOP，两者 KF 数/轨迹不同。因此 A 全量均值高
`0.7465 dB`、last 高 `3.5008 dB` 只能作为**诊断**，不能把差异归因于
LOOP 修复或 backend。该对没有提供同事件因果比较，也没有验收 GBA。

共同训练帧请按 `cameras.json` 的 `img_name` 配对，而不是按张量行号：
`ac_pair/common_frames.csv` 有 **17** 个共同帧。原生 GS PSNR 的 A−C 为
mean `+1.0327 dB`、median `+0.9844`（A 高 13/17）。按图像时间切成
`frame001300` 前的 15 帧和之后的 2 帧只能作为**图像时间分类**；这些分数都在
shutdown 时计算，不能证明质量在 LOOP 前已经分叉。**最后共同帧**
`frame001625.jpg`（A/C camera id 均为 163）为 A `28.2450`、C `27.9186`、
A−C `+0.3264 dB`。该行是最后一个共同数据帧，不是任一 PSNR 文件最后一行；
全量均值仍只作系统描述。

## 7. 可变拓扑 A0/A1/C 首次分叉（严格 120 步，统一 optimizer 重建）

`topology_divergence_r5` 的 400 步只作探索；`r6/r7` 的严格重放暴露了一个
实验恢复问题：A0 使用 live optimizer，而 A1/C 因行数改变重建 optimizer，
可能混淆对比。`r9` 对 **A0/A1/C 全部**强制走同一条
`restoreRecoverySnapshot(..., force_rebuild=true)` 路径，再重新加载 Adam。
因此本轮采用 `evidence/topology_divergence_r9/`。

一次 room1 C run 在 iteration **6599**（下一次 densify 6600）捕获，162 KF /
370358 GS / 1409 line，`param_fp=b6ee8e82a2773ac6_n370358_nline1409`，
`adam_fp=51d652e14b5cb1cb`。A0/A1/C 各 120 次真实更新，覆盖 6600 和 6700
两次 densify。

RNG 控制：`std::rand()` 只服务 error-guided 视图采样；固定序列正好 120，
没有回落实时采样；clone/split 的 `at::normal` 使用 torch 默认 CPU/CUDA
generator，**捕获并恢复实际 generator 状态**后再开始每臂；每臂还
`std::srand(12345)`。`std::random_device` 只在初始化使用。正常
`trainForOneIterationErrorGuided`、原 densify/clone/split/prune、全部损失和
`opacity_reset_interval=0` 保持不变。

原生 GS PSNR（固定 view，无梯度）：

| arm | final GS | final line | step0 | step1 | step10 | step100 | step120 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 368328 | 1410 | 25.5463 | 25.4741 | 26.65843 | 28.57022 | 28.5705 |
| A1 | 368316 | 1412 | 25.5463 | 25.4741 | 26.65843 | 28.56971 | 28.6052 |
| C | 368324 | 1412 | 25.5463 | 25.4741 | 26.65834 | 28.56924 | 28.6110 |

- step0/step1 三臂 bit-identical；step10 差约 `9e-5 dB`。
- step120：C−A0 `+0.0405 dB`，C−A1 `+0.0058 dB`，A0−A1 `−0.0347 dB`。
  C 与 A1 基本持平，A0 略低。单次严格重放不支持 C 持续退化。
- 第一次完整决策点 6600：完整 prune/clone/split 集合逐行相同
  （prune `7056`、clone `2664`、split `3877`）；增密后、optimizer 更新前
  fingerprint 三臂均为 `89d4f940ea6028d4_n369843_nline1405`；
  `grad_max=0.0127795`、`scale_mean=0.0148415`、`opacity_mean=0.619997`、
  `struct_loss=0.00459062` 也相同。
- 第一个非 RNG、非恢复路径的参数分叉在 **step2**：A0 `a785c1ce...`、
  A1 `c025c732...`、C `c10a1453...`。A0/A1 同 backend、同强制重建路径、
  同 RNG 状态，因此这一步仍有 CUDA/atomic 浮点归约差异，不能单独归因于 C。
- 第一次拓扑分叉在 **6700**：opacity prune `11198/11203/11198`，
  clone `3028/3017/3019`，split `6655/6659/6660`，最终总数
  `368328/368316/368324`；完整集合也均不同（`topo_mask.csv.gz`）。6600 后
  行号不能作为跨臂对应关系。

**结论**：本轮不修改算法、阈值、学习率或精修预算；不做阈值补偿。严格
120 步下未见 C 持续退化；第一次参数分叉在 step2 且 A0/A1 也有；第一次
拓扑分叉在 6700 且 A0/A1 也有。在线 A/C 共同帧 `+1.0327 dB` 的差距仍在，
但不能用 shutdown 分数按图像时间切分来证明 LOOP 前已分叉。

### 7.1 现有在线 Event 时间证据能回答到哪

`evidence/topology_divergence_r9/ONLINE_PHASE.md` 复用已有 event-time
评价（来自另一个 C run，不是本 120 步 capture）：C 的 LOOP 事件 before
`29.4182`，camera-only `27.3609`（`−2.0573`），corrected map `29.0122`
（相对 before `−0.4060`）。即 C 在 LOOP 处存在**真实的瞬时跳变**；固定几何
event-time replay 从 `29.4182` 到 100 步 `31.547`，没有显示事件后持续下降。
GBA event-time 变化很小。

但 A 没有 LOOP 事件，A/C 不是同一事件；现有日志只有 KF/frame 位置和最终
optimizer 步数，没有逐帧 optimizer 步、SH/分辨率次数、队列、停止原因或
事件/帧检查点模型状态。因此目前只能回答“C 的 LOOP 处有跳变”，不能回答
A/C 差距是在 LOOP 前已形成还是事件后扩大。下一步必须用有模型检查点的
在线 A/C 诊断，不能用 shutdown 分数按图像时间切分冒充历史质量。

限制：一次 capture、一次 120 步 replay；A0/A1 不是独立进程重复；RNG 状态
恢复只控制随机流，不保证 CUDA 归约 bitwise 确定。本轮无非零 GBA 事件，GBA
默认策略未改变。证据：`evidence/topology_divergence_r9/REPORT.md`、
`summary.csv`、`pre_update.csv`、三臂 `metrics.csv`/`topology_steps.csv`、
`audit/`。

## 8. 复现

```bash
SRC_FIX=/tmp/psl-room1-control/src-fix \
SRC_AUDIT=/tmp/psl-room1-control/src-fix-audit \
OUT=/tmp/psl-room1-control/repro-loop-fix \
bash /workspace/code/PSL-perf/scripts/repro_loop_coord_fix.sh
```

小版本测试（实际使用，已通过）：

```bash
cd /tmp/psl-room1-control/src-fix
cmake --build build-fix -j8 --target test_loop_coord_sync
./bin/test_loop_coord_sync
# [GbaContract] ... stamp=1 new_replay=1 new_event=1 : PASS
# [LoopKeyContract] order=1 map=1 match=1 : PASS
# [test_loop_coord_sync] PASS
```

A/C 诊断产物与完整命令：`evidence/loop_coord_fix_r5/ac_pair/README.md`。
非零 GBA 输入盘点：`evidence/loop_coord_fix_r5/gba_inventory.md`。

单事件质量命令（实际使用）：

```bash
cd /tmp/psl-room1-control/src-fix-audit
env -u CUDA_LAUNCH_BLOCKING -u PHOTO_SLAM_RASTER_AGG_SELFCHECK \
    -u PHOTO_SLAM_RASTER_PROF -u PHOTO_SLAM_GPU_PROF \
    PHOTO_SLAM_LINE_MODE=2 PHOTO_SLAM_GBA_APPLY_CORRECTION=0 \
    PHOTO_SLAM_APPEND_STEPS=0 PHOTO_SLAM_SEED=0 \
    PHOTO_SLAM_PSNR_DIAG=0 PHOTO_SLAM_RASTER_BACKEND=C \
    PHOTO_SLAM_EVENT_QUALITY_DIR=$PWD/event-quality-fix-r5 \
    PHOTO_SLAM_EVENT_QUALITY_VIEWS=24 \
    timeout --signal=TERM --kill-after=30s 1800 \
    bin/replica_rgbd \
      /workspace/code/PSL-perf/ORB-SLAM3/Vocabulary/ORBvoc.txt \
      /tmp/psl-room1-control/config/room1_orb.yaml \
      /tmp/psl-room1-control/config/room1_gaussian.yaml \
      /workspace/code/SEGS-SLAM/datasets/replica/room1 \
      /tmp/psl-room1-control/src-fix-audit/out-fix-r5 no_viewer
```

## 9. 在线 A/C 模型检查点诊断

`evidence/online_checkpoints_r0/REPORT.md` 记录了本轮唯一一对在线 A/C
（同代码/配置/种子，仅 backend；检查点 input frame `500/1000/1500/end`）。
固定 12 个早期共同帧的 A−C GS PSNR 为：500 `−1.7342`、1000 `−0.8803`、
1500 `+3.2018`、end `+1.7474 dB`；两次运行均无 LOOP/GBA。任何 A/C 分叉在
**frame 500 之前**已经存在；但 C **首次低于** A 的实测点是 1500，且 500/1000
时 C 更高，符号翻转，不能称为单调 C 退化的起点。500 之前没有检查点，不能
进一步定位。详细更新量、KF 数和缺失的队列/分辨率统计见该报告。

来源：生产基线 `ec85e80886e2f0494acb87781e075334bcf2ba58`；诊断补丁
`instrumentation.patch`（SHA-256 `3d7f56e5…`）。报告中的 PSNR 是检查点时刻由
原有 C++ 渲染/GS-PSNR 函数产生并落盘，**没有**再做一次训练结束后的离线重渲染；
快照、相机和 mask 已保留，可用于后续离线复算。

## 10. 检查点离线复算 + 受控续训（r1）

`evidence/checkpoint_followup_r1/REPORT.md` 是后续证据。要点：

- 用新增 `eval_checkpoint_snapshot` 读取已有 snapshot、训练相机和 mask，调用
  原有 `psnr_gaussian_splatting` 复算 8 个检查点；与现场逐帧最大差
  `0.00661 dB`，均值差 `<0.00018 dB`，原表无需修正。
- 固定 12 帧离线 A−C：500 `−1.7342`、1000 `−0.8806`、1500 `+3.2019`、
  end `+1.7476 dB`。
- 1000→1500 的固定帧相对相机变化很小（A 最大 `0.117°/4.84 mm`，C
  `0.094°/3.43 mm`）；`frame000000` 相机完全不变，A/C 仍分别掉
  `3.61/10.50 dB`。因此这一跌落不是相机位姿跳变，不能据此指向局部 BA；
  但具体模型/拓扑原因现有数据不能确定。
- 现有 snapshot 无 Adam/LR/使用状态，不能静默恢复。用一次 C 自然结束的
  活状态做 A/C 固定拓扑续训（0/100/400/800/1230 次更新）：两臂从同一
  `33.4330 dB` 起，1230 次后均约 `35.36 dB`（提升约 `1.93 dB`），A−C
  最大仅 `0.047 dB`，无持续分叉；1230 较 800 略降，非单调。
  这支持“C 终态仍有拟合余量/慢区间损失训练机会”，但 1230 不是等算力，
  不能声称全部差距由步数造成；也未见 C 续训持续变差。
- 新增 `PHOTO_SLAM_STAGE_COUNTERS=1` 的 checkpoint/结束时计数器
  （分辨率/SH 更新、训练像素、KF 使用、队列/进度/结束原因）；LocalBA
  消费步和相机变化仍未捕获，下一次必要在线运行再补，不从旧日志补造。

生产基线相同：`ec85e80886e2f0494acb87781e075334bcf2ba58`；r1 诊断补丁
`104a2d70…`。详细表格、相机/尺度/opacity 统计和限制见该证据目录。


## 11. 极端尺度与冻结渲染反事实（r2）

`evidence/checkpoint_followup_r2/REPORT.md`。只读现有 A/C 1000/1500 快照，
不训练、不调参、不重跑在线。

- `scaling_` 使用 `exp` 后的实际尺度，三轴分开；C 的约 12 最大尺度是
  **普通 Gaussian**，不是线 Gaussian。C1500 最大行 axes
  `[1.119, 0.809, 11.888]`、opacity 1.0、ref KF 0、中心投影在 ref KF 外；
  单轴膨胀，不是线高斯的 longitudinal/transverse 模式。
- 线高斯最大尺度仅约 `1.72`（C1000）/`1.65`（C1500）。线偏心项代码仍是
  `mean(s_perp)/(s_parallel+1e-6)` 的无界形式；**原式仍持续作用**，但当前
  快照与选定异常行的反事实实验不支持它是本次 1500 下降的主要原因
  （`s_parallel<1e-5` 为 0、`s_perp>s_parallel` 为 0，ecc p50 约 `0.00163`、
  max 约 `0.00230`）；没有改损失，也不把该反事实当成生产剪枝建议。
- 预先固定规则（按每类 max(exp(scaling)) 前 20 行，不看 PSNR）做冻结全场景
  渲染 opacity 反事实。线行置零几乎无影响；普通行置零通常**降低**质量
  （C1000 固定 12 帧 `−1.086 dB`，A1000 `−0.302 dB`），仅 C1500 frame0
  有 `+0.160 dB` 的局部改善，但同 checkpoint 固定 12 帧仍 `−0.0086 dB`。
  因此异常尺度行不能解释 1500 的下降，最大尺度不能直接当作有害贡献。
- 结论转向后期训练分配：下一次必要在线运行启用
  `PHOTO_SLAM_STAGE_COUNTERS=1`，记录分辨率/SH 更新、训练像素、每 KF
  使用预算、停止原因；r2 阶段 LocalBA 消费/相机变化仍是未知，本轮已补
  独立 CSV 记录，见下文。
- Adam step 8625 vs 全局 8712 的 87 差已记录为审计项：6 个组 step 全相同，
  代码中的 `trainingSetup`、`replaceTensorToOptimizer`、densify/prune state
  extension 和诊断重建位置已列出；没有逐迭代证据，不能直接叫“漏更新”，
  未改优化器。

基线 `ec85e80886e2f0494acb87781e075334bcf2ba58`；r2 补丁
`ff71241f…`。证据目录同时保存 r2 二进制、库、配置和 `artifact_manifest.sha256`。

## 12. H 阶段保护候选：A/C/H 同批验证

`evidence/checkpoint_followup_r3/REPORT.md`。H 是独立实验选项
`PHOTO_SLAM_RASTER_BACKEND=H`：`active_sh<3` 调用 C，`active_sh>=3` 调用 A；
默认 A 不变，不改 CUDA 公式、损失、精度、预算或固定精修。

- 主机侧分发验收 `test_raster_h_dispatch` PASS；实际 H 运行 4 次切换：
  sh0→C、sh1→C、sh2→C、sh3→A。自然结束实际 backward 计数
  A=7065、C=2999，总数 10064 等于 optimizer updates；与 stage backend
  计数完全一致，且未启用 selfcheck/golden/profiler。
- `PHOTO_SLAM_STAGE_COUNTERS=1` 已真实启用：A/C/H 的每个 checkpoint 和
  natural_end 都有分辨率/SH/backend 更新数、训练像素、KF 使用、队列/
  进度/停止原因、Adam 六组边界。LocalBA 消费已有独立 CSV：短 A 运行
  3 个事件、33 行，记录 optimizer step/iteration、input frame、KF 数/
  范围、old/new 位姿、rot/trans、map/line 点数；事件发生在 step 1/1/124，
  不是在每个 100 边界，不能解释 Adam 差。A 无 LOOP/GBA，C/H 各 1 次
  LOOP、1 次 GBA（121/117 KFs，deformation disabled，moved=0），因此
  H vs A 仍有事件混杂。
- GBA 实际增量：C loop `119 KFs`、pose_changed=0、movedRows=359170、
  nonzeroMoved=0、maxTrans=2.67e-07、maxRot=5.60e-06；H loop `116 KFs`、
  pose_changed=112、movedRows=374605、nonzeroMoved=336452、
  maxTrans=7.10e-03、maxRot=0.2162；两者 Global GBA 均 moved=0。
  C 的 loop 增量在数值上接近零，H 的 loop 不是零；`pose_mismatch=0` 只
  说明检查到的位姿项一致，不等于所有增量都为零。C/H 没有保存完整的
  逐 KF 事件前状态，因此事件混杂仍无法完全消除。
- input_end 与 natural_end 分开：A `9769/178→9770/179`，C
  `8771/180→8772/181`，H `10063/181→10064/182`。质量使用现有离线快照
  C++ GS PSNR；本轮固定早期共同帧只剩 **5** 帧，不能把原 12 帧曲线照搬。
- 固定 5 帧曲线（A/C/H）：500 `28.38/33.63/32.29`，1000
  `35.45/36.75/35.74`，1500 `32.84/27.51/31.62`，input_end
  `34.40/31.16/34.45`，natural_end `34.38/31.18/34.41`。
  natural_end 全部共同 12 帧：A `33.29`、C `30.37`、H `32.46`；
  full-set：A `30.99`、C `28.50`、H `30.07`。H 比 C 好，但比 A
  低 `0.826 dB`（12 共同帧）/`0.921 dB`（full set）；固定 5 帧和 frame0
  的局部改善不足以宣布系统增益。
- H 相比 C 增加 1292 次更新、20.6% 训练像素，但总操作时间
  `89.99 s` vs A `89.02 s`，不是速度收益；且 C/H 事件、KF 集合不同，
  因果受限。H 没有在整体上超过同批 A，故按预先规则保留 A，不做交换顺序
  的第二轮，也不扩阈值/精修。结果记录为 partial/negative candidate。
- Adam 逐 100 诊断已在短 A 运行中定位：iteration 100 的
  `after_backward` 六组 rows=3029、grad=1、Adam step=99、host=99；
  `after_topology` 六组 TensorImpl 全部更换、rows=3419、grad=0、Adam=99；
  `after_step` host=100 但 Adam 仍 99。iteration 200 同样为
  after_backward Adam=198/host=199，after_topology 全换且 grad=0，
  after_step host=200/Adam=198。`prunes.csv` 记录
  `3029→2737`、`4101→3419`、`3989→3430`、`5214→4322` 行变化，
  `replacements.csv` 记录 tensor 5 在 step 50/99/149/198 被替换。
- 运行时 `runtime_config.txt` 显示 `densify_from_iter=0`，而 YAML 写 600：
  代码读取的是 `Optimization.densify_from_iter_`（带尾下划线），YAML 键
  是 `densify_from_iter`，因此拓扑从 iteration 100 就开始。顺序是
  `backward → topology replacement → optimizer.step()`；替换后六个参数
  TensorImpl 全变、梯度为 None，Adam 对六组都不更新，但 host 计数仍加一，
  这正好解释每 100 步丢一次更新（以及 200 步时 198 vs 199）。
- 这是既存顺序，不是 H/C 引入的回归；本轮按规定不移动 optimizer 与
  densify 顺序、不复制父梯度、不改 schedule。`densify_from_iter` 键名
  不匹配作为配置缺陷记录，未修复，因为修复会改变训练行为。
- 运维结论不变：A 继续作为正常运行后端；C/H 仅保留实验入口，不扫描
  切换阈值、不增加精修。H 是 partial/negative candidate，不是生产加速。

上一轮插桩补丁 SHA-256 `3b3f2d4c…`；本轮配置修复补丁见下方
`9b76438c…`。证据目录保留正式二进制、库、配置、快照哈希。

## densify_from_iter 配置修复与 A 基线

- 旧行为：代码读取 `Optimization.densify_from_iter_`，YAML 使用
  `Optimization.densify_from_iter`，缺失节点被 OpenCV 静默读成 0。
- 新行为：规范键优先；仅旧键存在时警告并使用旧键；两键冲突报错；
  两键缺失使用显式默认 500 并打印。运行日志和 `runtime_config.txt`
  同时记录实际 from/until/interval 及来源。
- 解析验证：canonical=600、legacy=600（警告）、冲突退出 1、缺失=500；
  room1 首次普通增密在 iteration 700，不是 600。LocalMapping 补点单独记录。
- 更新计数分开为 `training_backward_rounds`、`optimizer_step_calls` 和
  `adam_advance_total/group`；旧 `optimizer_steps` 不再称为真实参数更新。
- 同构建 A 对照：from0 `9538` calls / `9443` Adam 推进 / 95 次增密；
  from600 `10223` / `10127` / 96 次增密。每个普通增密事件恰好丢一次
  Adam 推进，保持既有 `backward -> topology -> optimizer` 顺序。
- 质量：natural_end 共同图像（n=31）from600 比 from0 平均 `+0.298 dB`，
  中位数 `+0.565 dB`；全帧集 `31.4838` vs `30.6067 dB`，但图像集不同。
  frame_500/1000/1500 共同帧 from0 更高，frame_end 与 natural_end
  from600 更高；单次运行、共同样本小，不作显著性声明。
- 两臂 full run 均在写完 natural_end snapshot 和 `[PhaseTiming]` 后未退出，
  被外部 timeout 终止（exit 124）；该 shutdown 问题另列，未混入本次配置修复。
- 详细报告：`evidence/densify_config_a/REPORT.md`；补丁 SHA-256
  `9b76438c…`。C/H 未运行。

## A 基线收尾：退出 124 与正常退出

- 退出 124 不是 `System::Shutdown()` 死锁：阶段日志显示 mapper/loop closer
  已完成并 join，`main` 卡在 `main_before_viewer_join`。
- 先前命令未传 `no_viewer`；viewer 线程在 `glfwCreateWindow()` 内阻塞，
  `DISPLAY=:28` 存在但无可用的 X server。`ptrace_scope=1` 无法 attach，
  证据为默认关闭的阶段日志与线程状态捕获，不冒称调用栈。
- 修复仅涉及退出生命周期：对已有 viewer 调 `signalStop()` 后再 join，并在
  `no_viewer` 路径下不创建 viewer；未跳过 Shutdown/join/export，未 detach，
  未强制 exit，未吞 124。阶段日志默认关闭，需
  `PHOTO_SLAM_SHUTDOWN_DIAG=1`。
- 400 帧短序列 `no_viewer`：exit 0，总耗时 15.9385 s。
- 诊断开启的 final A/from600：exit 0，from=600，首增密 700；
  `optimizer_step_calls=9655`、Adam/组 `9565`、普通增密 `90`；
  KFs 177、Gaussians 488209；mapping 65.674 s、SLAM_end 86.498 s、
  `TotalOperationTime=89.0689 s`；原生 PSNR 30.8196 dB（177 帧）。
- 该诊断开启运行与 from0 共同 31 帧均值 `31.2581` vs `32.1697`
  （`-0.9116 dB`，13/18）；与历史 A 共同 25 帧 `31.5185` vs `32.8833`
  （`-1.3648 dB`，11/14）。下面干净运行是正常使用基线，二者不可混称。
- 详细报告：`evidence/densify_config_a/REPORT.md`，补丁 SHA-256
  `03f4bcb2…`。C/H 未运行。

## 干净 A 复核与条件核实

- 最终可用运行显式 `no_viewer`、`PHOTO_SLAM_LINE_MODE=2`、
  `PHOTO_SLAM_LINE_LOOP=0`、`GBA_APPLY_CORRECTION=0`、
  `APPEND_STEPS=0`、seed 0、backend A；关闭 stage/Adam/topology、
  online checkpoint、shutdown diag、selfcheck 和 profiler。
- room1 全序列 exit 0：calls `10212`、backward `10212`、Adam/组 `10116`、
  差值 `96`；natural snapshot KFs `185`、Gaussians `520255`、line `390`；
  mapping `70.478 s`、SLAM_end `85.144 s`、总程序 `88.4992 s`；
  原生 PSNR `32.1042 dB`（185 帧）。
- `img_name` 共同帧：干净 A 比 from0 `+1.1027 dB`（n=24，18/6），比历史
  A `+0.7638 dB`（n=31，23/8）；不是代码回退，暂不做源码二分。
- 历史 A 的 `PHOTO_SLAM_LINE_LOOP`、seed、viewer、GBA 开关未完整保留，
  不能称“同条件”；历史高分只作历史观察。当前代码 `LINE_LOOP` 未设置时
  默认 2，因此必须显式写 0。
- 配置别名审计：YAML 的 `Line.SampleStep=0.10`、`max_anisotropy_threshold=100`、
  `absolute_max_ratio=100` 未绑定到代码读取的 `Optimization.*` 键；实际
  生效值分别为 `0.2`、`600`、`1500`。其余 `lambda_anisotropy`、`Line.Sigma`、
  `Line.TopK`、`Line.ViewWeight` 虽未绑定但默认值恰好相同。本轮未改这些值。

## 冻结 A、重复运行、别名修复与批量准备

- 冻结归档：`evidence/frozen_A_clean_20261023/`，含源码快照、补丁、二进制、
  共享库、配置、命令和 `SHA256SUMS`，目录已只读。
- 三次同条件 clean A room1（全部 exit 0，关闭诊断/检查点/selfcheck/profiler）：
  PSNR `32.1042 / 31.3607 / 31.1281 dB`，均值/样本标准差
  `31.5310 ± 0.5099 dB`；KFs `185/183/172`；Gaussians
  `520255/549503/498388`；调用 `10212/10214/10094`；Adam 每组
  `10116/10118/10000`；总耗时 `88.4992/88.5939/85.7869 s`。
  三次共同帧交集只有 16 张，均值/标准差 `33.3668 ± 0.2714 dB`；
  不把帧数当独立重复。
- 历史 A 的 `PHOTO_SLAM_LINE_LOOP`、seed、viewer、GBA 开关仍未知；
  不可当作严格同条件基线。`LINE_LOOP` 未设置时默认 2，命令必须显式写 0。
- 别名兼容补丁 `alias_compat_fix.patch` 在隔离树
  `/tmp/psl-alias-fix` 验证：canonical、alias-only、same、conflict、
  missing 五种规则通过。稳定 canonical 配置与原 clean A 的 66 个实际值
  对比 **0 差异**；用户意图配置只改变 `absolute_max_ratio=100`、
  `max_anisotropy_threshold=100`、`line_sample_step=0.10`。
- 批量脚本 `scripts/run_replica_A_baseline.sh` 已通过 8 场景数据和配置检查，
  room1 完整检查 exit 0；使用冻结二进制、显式稳定配置、`no_viewer`，
  不调用评价项目。

## 8 场景 Replica RGB-D 批量结果

- 命令：`BATCH_TAG=20261023_A8 scripts/run_replica_A_baseline.sh`
- 结果根目录：
  `results/replica_A_baseline_20261023_A8/replica_rgbd_0/`
- office0/1/2/3/4、room0/1/2 全部 **exit 0**；耗时
  `87/86/97/99/94/108/92/108 s`。
- 每场景已检查 `point_cloud.ply`、相机/关键帧 TUM 轨迹、
  `Global_Final_Map_Points.obj`、`TotalOperationTime.txt`、`run.log`、
  `command.txt`、`natural_end/snapshot.pt` 非空。
- 使用冻结 A 二进制和配套库；`no_viewer`；诊断、检查点、selfcheck、
  profiler、追加训练全部关闭。未计算 PSNR，未调用 Photo-SLAM-eval。
- `summary.csv` 仅含 `scene,exit_code,log_path,result_dir`；
  `summary_detailed.csv` 保留配置/二进制哈希。无残留实验进程。
