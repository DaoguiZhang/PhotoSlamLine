# 共同输入 + 完整几何事件流上的 A/V/N 归因对照

日期：2026-11-07
根目录：`/workspace/code/PSL-evtrepl`（新实验独立存放，未覆盖任何旧产物）
前置：`REVIEW_REPORT.md`（独立审查）、`REPORT_AVN_OFFICIAL.md`（官方运行验证 + 上一轮纯 A/V/N 隔离）

本轮**没有**新增算法、损失或参数扫描；没有启动 150/300 项正式批次。
本轮做了三件事：把官方测试账本补清、把两处偶发异常查到根因、把 A/V/N 放到
**同一份共同输入 + 同一条事件流**上重放做可归因对照。

> **后续范围修正（不影响本报告任何数据）**：最终 L 版本随后去掉了 Gaussian mapping 的
> 结构线损失（三个有效权重 0、且不再计算，见 `lossaudit/` 与 `FINAL_VERSION.md` §5）。
> 本报告的 30 次重放**不是**最终配置的结果，而是**归因诊断**（A/V/N 在同一输入、同一事件流上
> 的相对差异）；它的结论（V 未确认画质收益、N 的应用集合不可复现）是随后把 V/N 都设为 0
> 的依据之一。该报告中的 `L_core` 指当时那个 V0N0 **带结构损失**的版本，
> 与现在冻结的 `L_core`（结构损失关闭）**不是同一份配置**，两者不得混引。
> 正式整系统对比为 `O_core` vs 新 `L_core`，两方法 300 项，状态 **NOT_STARTED**。

---

## 一、保护与冻结

详见 `FREEZE.md`。要点：

* 官方副本 `PSL-upstream-official`、历史实验 `PSL-fullmap-lock-v1`（L0）、上一轮
  `PSL-dedupfix`（L1）、`PSL-lavn`、所有 patch 与失败日志**均未改动**。
* 本轮所有产物在 `/workspace/code/PSL-evtrepl/{export,replay,logs,report,diffs}`。
* 六个"版本"中 `O_raw / O_core` 是两个构建，`L_core / L_V / L_N / L_VN`
  是**同一个 L 二进制**（`d82b9ccae0368d72ea0fd52118884f2a`）的两个运行期开关的四种取值。
* 每次运行都回读开关实际取值并断言生效；生效证据是分支计数，不是环境变量打印。
* V/N 关闭不影响 SLAM 自身的回环、GBA、地图合并。
* A2↔A3、A6↔所有替换路径、V↔A7、N↔A5 四处依赖不可安全拆分，未为消融强行制造不一致。

---

## 二、官方测试账本（全部 case / 轮次 / 命令 / 加载库 / 退出码 / 结果）

### 2.1 命令与加载库身份

每 case 一个独立进程，`env -u LD_LIBRARY_PATH` 保证不被当前 shell 的
`LD_LIBRARY_PATH` 覆盖（上一轮踩过：`LD_LIBRARY_PATH` 优先于 RUNPATH，会让
"用别的库跑这个二进制"看起来像结论）。二进制自身从 `/proc/self/maps` 打印真正
映射进来的 `libgaussian_mapper.so` 路径，再由脚本对这个路径取 sha256：

```
[lib] mapped=/workspace/code/PSL-review/O_core/lib/libgaussian_mapper.so size=997216 mtime=...
```

| 版本 | 加载库 | sha256(前 32) |
|---|---|---|
| O_raw | `PSL-review/O_raw/lib/libgaussian_mapper.so` | `0396c6d327dfdbce11fd5ef3754c0487` |
| O_core | `PSL-review/O_core/lib/libgaussian_mapper.so` | `4293b9d7349f8bde2436097e5a278f83` |

账本：`ledger/ledger.csv`；每 (tree, rep, case) 的完整 stdout+stderr 在
`ledger/raw/<tree>_rep<N>_case<M>.log`。命令：`./run_official_ledger.sh 3`。

### 2.2 结果（3 轮 × 8 case × 2 版本）

| case | 内容 | O_raw（3 轮） | O_core（3 轮） |
|---|---|---|---|
| 1 | 恒等变换下四元数写回 | FAILED ×3 | OK ×3 |
| 2 | 恒等变换 no-op | OK ×3 | OK ×3 |
| 3 | 纯平移 | OK ×3 | OK ×3 |
| 4 | 全局 `s=2`：中心/σ/Σ | FAILED ×3 | OK ×3 |
| 5 | 回环 `s=0.5` + 混合选择 | FAILED ×3 | OK ×3 |
| 6 | 变换 + 逆变换 | FAILED ×3 | OK ×3 |
| 7 | 首次 `optimizer.step()` 之前更新 | **SIGSEGV ×3（exit 139）** | OK ×3 |
| 8 | 非法尺度拒绝 + 状态不变 | FAILED ×3 | OK ×3 |

合计 **O_raw 6 OK / 15 FAILED / 3 SIGSEGV**，**O_core 24 OK / 0 失败 / 0 崩溃**。

### 2.3 "4 pass / 9 fail"对"16 项"：差在"把什么算作一项"，而旧文本自己也没对齐

先把事实钉死：`review_official.cpp` 里 `check()` 只有 **8 个**
（`grep -c "    check(" = 8`），**每个 case 恰好 1 个断言**。所以这里从来没有
"13 个断言"或"16 个断言"这种东西——**8 个 case 就是 8 项**。

那么两个旧数字分别是什么：

* **"16 项"** = O_core 上一轮跑的 **2 轮 × 8 case = 16 次运行**，全部通过。
  它数的是**运行次数**。
* **"4 pass / 9 fail"** = O_raw 同一次 2 轮的 16 次运行，其中 4 次通过、
  9 次断言失败、**3 次崩溃**（case 7 两轮都 SIGSEGV，case 1 第一轮 abort）。
  旧文本写的是"4 pass / 9 fail（含 2 次段错误）"——"含"字使它看起来像
  4+9=13 项，而 13 与 16 对不上；并且崩溃实际是 **3 次**（2 次 SIGSEGV + 1 次
  abort/134），不是 2 次。**这是我上一轮的计数错误，本轮更正。**

按上一轮原始日志（`logs/official_O_raw.log`）逐 case 重算：

| 轮次 | OK | FAILED | CRASH |
|---|---|---|---|
| rep1 | 2（case 2, 3） | 4（case 4,5,6,8） | 2（case 1 abort 134 + case 7 SIGSEGV 139） |
| rep2 | 2（case 2, 3） | 5（case 1,4,5,6,8） | 1（case 7 SIGSEGV 139） |
| **合计** | **4** | **9** | **3** |

4 + 9 + 3 = 16 —— 这才与 O_core 的 16 次运行同口径。
所以"数量差异"根本不存在：**两者都是 16 次运行，只是 O_raw 的 16 次里有 3 次是崩溃、
9 次是断言失败、4 次通过；O_core 的 16 次全部通过。**

本轮把账本做成 **3 轮 × 8 case = 24 次运行**（每 case 独立进程、加载库身份已验证、
`env -u LD_LIBRARY_PATH`），结果 **O_raw 6 OK / 15 FAILED / 3 SIGSEGV**、
**O_core 24 OK / 0 失败 / 0 崩溃**，与上表结构完全自洽
（每轮 2 OK / 5 FAILED / 1 CRASH，×3 轮 = 6 / 15 / 3）。

### 2.4 两处 Torch 异常：原文、完整 stderr、根因

**原文（O_core，rep2，case1，完整 stderr，一字未改）：**

```
[lib] mapped=/workspace/code/PSL-review/O_core/lib/libgaussian_mapper.so size=997216 mtime=1794064673
terminate called after throwing an instance of 'std::runtime_error'
  what():  value cannot be converted to type float without overflow
```

**判定过程（不是"GPU 噪声"这种说法）：**

1. **不是加载库问题。** `[lib]` 行证明映射的是预期库，sha256 与上表一致；
   `env -u LD_LIBRARY_PATH` 排除了 shell 覆盖。
2. **不是被测实现问题。** 在 `c10::report_overflow` 上设断点复现，栈顶是：

   ```
   #1 c10::Scalar::toFloat()
   #2 at::native::addcdiv_cuda_kernel(...)
   #5 torch::autograd::VariableType::addcdiv_
   #7 torch::optim::Adam::step(std::function<at::Tensor ()>)
   #8 Fixture::Fixture(bool)
   #9 main
   ```

   也就是说，异常发生在**测试夹具里那次"占位 optimizer.step()"**，在任何一个
   几何断言之前，且栈里没有任何 Photo-SLAM 几何代码。
3. **不是测试输入问题，是未初始化输入问题。** 逐组打印优化器超参，发现：

   ```
   [opt] group 0 lr=0.00015999999595806003 eps=1e-15 beta1=0.9 beta2=0.999
   [opt] group 1 lr=0.0024999999441206455  ...
   [opt] group 2 lr=0.00012499999720603228  ...
   [opt] group 3 lr=-1.0436737635417543e-16    <-- opacity 学习率是垃圾值
   [opt] group 4 lr=0.004999999888241291  ...
   [opt] group 5 lr=0.0010000000474974513 ...
   ```

   根因在 `src/gaussian_parameters.cpp`：

   ```cpp
   opacity_lr_(opacity_lr_),      // 自我初始化：读的是未初始化的同名成员
   // 应为 opacity_lr_(opacity_lr)
   ```

   该成员在"直接构造 `GaussianOptimizationParams`"的路径上是**不确定值**。当这个
   垃圾值超出 float 范围时，torch 的 Adam 第一次 `addcdiv_` 把 `lr / bias_correction1`
   作为 double Scalar 转 float 就会抛出 `value cannot be converted to type float
   without overflow`；垃圾值恰好是极小负数时（如上）则静默通过。
   这解释了"偶发、与断言无关、两个版本都会遇到"的全部特征。
4. **最小复现（不碰任何几何代码）。** `src/repro_overflow.cpp` 只做
   `GaussianModel::createFromPcd → trainingSetup → 零梯度一次 optimizer.step()`：

   | 状态 | 60 次独立进程 | 结果 |
   |---|---|---|
   | 修复前 | 60 | **1 次 abort**（`without overflow`），59 次正常 |
   | 夹具显式 `opt.opacity_lr_ = 0.05f` 后 | 60 | **0 次失败** |

   修复后 group 3 打印 `lr=0.05000000074505806`，异常消失。
5. **本轮处理。** 在 `review_official.cpp` 夹具里显式给 `opacity_lr_` 赋值（注释写明
   原因），随后重跑账本 → 3 轮 × 8 case × 2 版本**全部可重复**，O_core 24/24 OK，
   O_raw 稳定 6 OK / 15 FAILED / 3 SIGSEGV，**没有任何未解释的异常**。

**对冻结的影响（明确结论）：不阻塞。**
异常来自测试夹具的未初始化输入，不属于 A 类修复，也不属于被冻结的库；修复夹具后
同一库同一断言 3 轮全稳定。

**顺带发现的上游缺陷（本轮不改库，只登记）：**
`opacity_lr_(opacity_lr_)` 在生产路径**不受影响**，因为 `readConfigFromFile()` 无条件
用 YAML 覆盖 `opt_params_.opacity_lr_`；61 份 `cfg/gaussian_mapper/**/*.yaml` 中 60 份
显式写了 `Optimization.opacity_lr`，缺这行的一份（`Monocular/tandt/1tandt_train.yaml`）
会被读成 `0.0`。若要修这一行，会把该配置的 opacity 学习率从 0 变成 0.05，
**这是行为变更、必须单独决策**，因此本轮不在冻结时刻顺手改库。

---

## 三、共同输入与完整几何更新事件流

### 3.1 导出什么、写在哪

每个 `(场景, 种子)` 只跑**一次** live 运行，导出到
`export/<scene>_s<seed>/`，随后五种条件都从这些文件重建：

| 文件 | 内容 |
|---|---|
| `snapshot.txt` | 初始、未优化状态：初始关键帧集合（含出生位姿）+ 初始稀疏点云（含线采样点、方向、步长）+ `cameras_extent` + 地图 id + 坐标纪元数；文件末行是内容哈希 |
| `kf_index.txt` | 曾经进入 mapper 的**全部**关键帧：id、相机 id、出生位姿、尺寸、原始图像深度码、`creation_iter`、源文件名 |
| `kf_img/<fid>.png` | mapper 真正训练的**已去畸变**图像（无损 PNG：float 0–1 → 8 bit，深度 m → mm 16 bit；量化对所有条件完全相同） |
| `kf_aux/<fid>.png` | RGB-D 的深度图 |
| `events/eNNNNN_<Type>.txt` | 生产消费端**按顺序**消费的每一条 `MappingOperation`：类型、`mfScale`、`mT`、消费时的 mapper 迭代、逐关键帧新位姿、新地图点、线采样点（位置/颜色/方向）；文件末行是内容哈希 |
| `events/index.csv` | 上面这些事件的紧凑时间线 |
| `run_meta.txt` | live 运行在整个优化预算上的长度（`total_iteration`）、事件数、关键帧数、末态高斯数 |

### 3.2 重放怎么保证"同输入、同事件流"

* 五种条件读**同一批文件**，并在读入时逐个文件校验哈希；任一不匹配直接抛异常退出。
* 每种条件都从 `snapshot.txt` 的 pcd 重新 `createFromPcd`，因此权重、Adam 状态、
  体素网格、关键帧对象、`default_sh_`、`iteration_`、`frozen_opt_steps_`、
  `kf_used_times_`、`libc rand()` 状态全部重建，**不共享任何训练后的模型或可变事件对象**。
* 训练视图序列在运行前一次性生成：**只用训练集关键帧**做固定种子的 shuffle + 循环，
  五种条件共用同一条序列（哈希写进 `summary.txt` 的 `kf_sequence_hash`），
  因此视角顺序不可能成为条件间的差异来源。
* 事件不经过任何"简化"：`injectReplayEvent()` 把导出的事件重建成
  `MappingOperation`（新增 `replayAddKeyFrame/replayAddMapPoint/replayAddLineSample`
  三个只写数据的构造入口），`pushMappingOperation()` 推进**真实的 Atlas 队列**，
  再由 `combineMappingOperations_withLine()` 消费。**移动高斯的是生产消费者本身**，
  不是重放器。
* 事件按 live 迭代号线性映射到 `[1, 5000]` 的重放步轴（`nat` 取自 `run_meta.txt`），
  映射关系在运行前固定，且保持原始顺序；关键帧的 `creation_iter` 用**同一个** `nat`
  映射，保证 `scaledTransformVisiblePointsOfKeyframe()` 里
  `|exist_since_iter - kf_creation_iter| < stable` 这个门限在重放时间轴上
  保持与 live 相同的相对含义，而不是全instant。

### 3.3 已知的不保真之处（如实记录，不掩饰）

1. **重放压缩了 live 的迭代轴。** office0 的 live 长度 6598–6774 次迭代 / 5000 步 ≈ 1.32–1.35×；
   desk3 为 17880 / 19257 / 30100 次迭代 / 5000 步 ≈ **3.6× / 3.9× / 6.0×**。而 `densify_from_iter`、`opacity_reset_interval`、
   SH 升级等**按重放步数**计，所以"事件发生的时刻相对于增密计划"与 live 不完全对齐。
   这对五种条件**完全相同**，因此不损害归因；但它意味着本轮的绝对数值不能与 live
   运行的绝对数值直接比较。
2. **所有关键帧在重放 t=0 就存在**（位姿取它们的出生位姿），而不是像 live 那样逐个入场。
   这样第 0 步的评测视图集合在所有检查点都一致、可比；代价是"关键帧插入时序"未被重放。
   合并事件里 `created=0`（与 live 一致），所以"合并创建新关键帧"这条分支本轮未被覆盖。
3. 图像量化：float→8 bit（RGB）、m→mm（深度）。误差 ≤1/255 与 ≤1 mm，五种条件完全相同。
4. 未重放：ORB-SLAM3 的 IMU 路径、`ScaleRefinement` 的具体数值（本两个场景未触发，
   纪元计数为 0）。

---


## 四、运行矩阵：2 场景 × 3 种子 × 5 条件 = 30 次 Mapping

每个 `(场景, 种子)` 一个进程：先跑一次 live 导出，再在同一二进制里跑五种条件。
逐运行表：`report/input_identity.csv`（30 行）。

| 场景 | 种子 | 输入快照哈希 | 训练序列哈希 | 事件数 | 关键帧 | 初始高斯 |
|---|---|---|---|---|---|---|
| office0_mono | 0 | `18ee7b7f7d298edd` | `5bd2e2a322f64cf4` | 174 | 191 | 3018 |
| office0_mono | 1 | `02b3f7afddfda049` | `0a9831e5300804b0` | 169 | 191 | 3211 |
| office0_mono | 2 | `928f2aa9dbb333d6` | `393c0627a18d50a1` | 168 | 191 | 3080 |
| desk3_rgbd | 0 | `41d527db84982144` | `7ae392dcdcdcbcce` | 497 | 578 | 2534 |
| desk3_rgbd | 1 | `91a7dc5e3517c630` | `e3f0d6b761815176` | 432 | 480 | 2465 |
| desk3_rgbd | 2 | `bd9c98addd5f8389` | `f2bbfb4c6b442ff2` | 494 | 575 | 2552 |

（desk3 的哈希见 CSV；同一 `(场景,种子)` 的 5 行**完全相同**，脚本 `analyze.py`
已做唯一性检查：每个 `(场景,种子)` 只有 1 个 `(input_snapshot_hash,
kf_sequence_hash)` 组合，无分裂。）

每次运行都满足：

* `opt_steps = 5000 = requested_steps`（真实 `optimizer.step()` 次数，不是迭代循环次数）；
* `events_applied = events_in_stream`（事件全投递，无丢弃）；
* `finite = 1`（末态无 NaN/Inf）；
* 30000 次检查点 / 事件的逐视图 PSNR/SSIM 都写进了 `all_steps.csv` 与
  `event_log.csv`（`psnr_eval_before/after`）。

`VN` 就是冻结表里的 **L_VN**，`A` 就是 **L_core**，`V`/`N` 分别是 **L_V**/**L_N**。

---

## 五、公共参考上的评测

* **留出集固定**：按导出索引的 `i % 8 == 7` 取 1/8 关键帧作留出视图，
  其余为训练视图。**留出视图从不进入训练序列**（训练序列只由训练视图构成），
  因此下文的 `psnr_heldout` 是真正的留出视图数字；`psnr_train` 单列，用来区分
  "重建变好"和"泛化变好"。
* **相机与坐标纪元**：评测相机就是场景里的关键帧，事件（回环 / GBA / 合并）
  在移动高斯的同时用 `pkf->setPose()` 更新同一批关键帧位姿，所以**参考相机跟着同一个
  变换走**，不会因为"相机没同步"而错误惩罚一次正确的变换。
* **检查点**：0 / 100 / 500 / 1000 / 3000 / 5000 步，外加每个**几何事件**（LoopClosingBA /
  GlobalBAWriteback / MapMerge / ScaleRefinement）应用前后的 `before/after` 成对评测。
* **耗时**：`cum_mapping_ms` 包含训练 + 事件处理 + 体素维护，**不含评测**；
  每次评测单独记 `eval_ms`。所有条件覆盖同一区间 0–5000 步，等时间比较只在该区间内做，
  不外推。

### 5.1 留出视图 PSNR（3 个种子的均值）

| 场景 | 条件 | 0 | 100 | 500 | 1000 | 3000 | 5000 |
|---|---|---|---|---|---|---|---|
| office0_mono | A | 13.387 | 16.685 | 21.117 | 25.203 | 33.032 | 34.409 |
| office0_mono | A_repeat | 13.387 | 16.586 | 20.945 | 25.199 | 32.985 | 34.469 |
| office0_mono | V | 13.387 | 15.962 | 20.214 | 24.887 | 32.934 | 34.489 |
| office0_mono | N | 13.387 | 16.056 | 20.694 | 24.797 | 32.813 | 34.318 |
| office0_mono | VN | 13.387 | 16.021 | 20.733 | 24.876 | 32.916 | 34.425 |
| desk3_rgbd | A | 8.082 | 9.855 | 11.648 | 13.080 | 16.580 | 18.607 |
| desk3_rgbd | A_repeat | 8.082 | 9.616 | 11.417 | 12.693 | 16.574 | 18.620 |
| desk3_rgbd | V | 8.082 | 9.754 | 11.410 | 13.082 | 16.665 | 18.524 |
| desk3_rgbd | N | 8.082 | 9.828 | 11.757 | 12.942 | 16.563 | 18.588 |
| desk3_rgbd | VN | 8.082 | 9.579 | 11.341 | 12.905 | 16.366 | 18.574 |

第 0 步五行完全相同（`13.387` / `8.082`）是同一份输入的又一处直接证据。

---

## 六、机制与正确性：逐事件记录

逐事件表：`report/all_events.csv`（9670 行，30 次运行的全部事件），字段含
候选数（`ev_points` / `ev_line_samples` / `ev_kfs`）、实际更新数
（`points_transformed` / `line_transformed` / `kfs_corrected`）、跳过原因
（`kfs_skipped`）、体素重建次数与耗时（`voxel_rebuilds` / `voxel_points` /
`voxel_ms`）、事件前后数值检查（`non_finite` / `finite_ok`）、队列完成状态
（`event_ms` 非零即消费完成，退出码 0）。

### 6.1 覆盖情况（这是本轮最重要的边界）

| 场景 | 种子 | 回环 | GBA写回 | 地图合并 | 变换点数 | 变换线高斯 | 体素重建 |
|---|---|---|---|---|---|---|---|
| office0_mono | 0 | 1 | 1 | 0 | 14137 / 14553 / 14345 / 14543 / 14416 | 4 / 4 / 6 / 6 / 3 | **0** |
| office0_mono | 1 | **0** | **0** | 0 | **0（机制未覆盖）** | 0 | **0** |
| office0_mono | 2 | 1 | 1 | 0 | 854–864 | 0 | **0** |
| desk3_rgbd | 0 | 0 | 0 | 1 | 646 / 691 / 779 / 707 / 668 | 3 / 2 / 6 / 3 / 9 | **8**（307k 点，71.4 ms） |
| desk3_rgbd | 1 | 0 | 0 | 1 | 1164 / 1187 / 1211 / 1208 / 1255 | 7 / 12 / 11 / 12 / 8 | **7**（168k 点，39.6 ms） |
| desk3_rgbd | 2 | 0 | 0 | 1 | 631 / 575 / 579 / 562 / 542 | 7 / 10 / 10 / 7 / 9 | **9**（276k 点，65.6 ms） |

（每格按 A / A_repeat / V / N / VN 排列）

* **N 有覆盖**：office0 种子 0、2（GBA 写回 + 回环），desk3 全部种子（地图合并）。
  合计 **9/15 次运行**出现并消费了几何写回。
* **N 未覆盖**：office0 种子 1 的 live 运行**根本没有触发回环或 GBA**（0/0），
  五次运行的 `points_transformed` 全为 0。按规则记为"机制未覆盖"，**不用它证明
  N 有效或无效**——但它的 `N−A` 数值（−0.148 dB）恰好说明噪声有多大。
* **V 有覆盖**：仅 desk3（7–9 次重建 / 168k–311k 点 / 39.6–72.2 ms）。
* **V 未覆盖**：office0 全部种子、以及 live 段。原因不是开关失效，而是
  **office0 的 live 运行从来没有把线采样点送进任何 `MappingOperation`**
  （`ev_line_samples` 全 0，live 日志里 `[VoxelGrid] rebuilt` 出现 0 次）。
  所以 office0 上 `V=1` 与 `V=0` 走的是**同一段代码路径**，其 PSNR 差是纯噪声。

### 6.2 事件流确实带着真实几何更新（不是静态固定几何）

重放中移动高斯的是生产消费者 `combineMappingOperations_withLine()`，日志逐事件给出
`corrected / skipped / pointsTransformed / lineGaussiansTransformed`。
与 live 运行对照（office0 种子 0，同一次事件）：

| | kfsCorrected | kfsSkippedByGate | 变换点数 | lineGaussians |
|---|---|---|---|---|
| live（回环，iter 2348） | 25 | 57 | **9699** | 0 |
| 重放 A（step 1691） | 25 | 57 | **2119** | 0 |
| live（GBA） | 40 | 43 | **20498** | 21 |
| 重放 A（step 1733） | 40 | 43 | **12018** | 4 |

**关键帧级判定完全一致（25/57、40/43）**，说明事件语义被完整重放；
但**被移动的高斯数目少 2.2×（回环）到 1.7×（GBA）**。原因见 §3.3 第 1 条：
`scaledTransformVisiblePointsByKeyframe` 的资格门限是
`|exist_since_iter − kf_creation_iter| < 1`，即只移动"与该关键帧同龄"的高斯；
重放把 live 的 6774 次迭代压到 5000 步，增密历史被压缩，同龄集合因此变小。
desk3 的合并事件则与 live 同量级（重放 542–1255 点 / 3–12 个线高斯，
上一轮 live 观测为 572 点 / 8 个线高斯）。

**因此本轮能证明"机制被触发并作用于真实几何"，但不能用重放的变换点数去代表
live 的变换点数。**

### 6.3 每个几何事件前后的成对评测：一个必须写下来的负面结果

`report/post_event.csv`。同一模型、同一组视图、事件前后的 PSNR 差：

| 场景 | 种子 | A | A_repeat | V | N | VN |
|---|---|---|---|---|---|---|
| office0_mono | 0 | −2.93 | −2.94 | −3.00 | −3.08 | −2.86 |
| office0_mono | 2 | −2.27 | −2.27 | −2.18 | −2.19 | −2.16 |
| desk3_rgbd | 0 | −1.91 | −2.91 | −2.37 | −2.36 | −2.66 |
| desk3_rgbd | 1 | −3.11 | −2.88 | −2.57 | −1.67 | −2.87 |
| desk3_rgbd | 2 | −1.63 | −1.84 | −1.91 | −2.14 | −1.98 |

**每一个几何事件都会让留出视图 PSNR 立刻掉 1.6–3.1 dB**，包括没有 N 的 A 组；
后续训练把它重新拟合回来（office0 从 step 1691 的 25.5 回到 step 3000 的 32.5）。
N 既没有系统性地减轻这个跌落（desk3 三个种子里 N 好 1 个、差 2 个，均值差 +0.16 dB，
跨种子标准差 ~0.9 dB），也没有加重它。

这个跌落的机制是清楚的：写回路径**只移动满足"同龄"门限的那一小部分高斯**，
其余高斯留在旧坐标系，于是事件瞬间地图与相机不同步。这不是重放造成的
（live 的资格集合同样只有总量的 30–50%），而是**生产实现的既有性质**。
它同时解释了为什么 N 难以"修正事件后的地图不一致"：N 用的是同一个逐关键帧、
同一门限的选择性机制，作用范围同样受限。

---

## 七、统计与最终选择

### 7.1 对比（5000 步，留出视图 PSNR，3 种子）

| 场景 | 对比 | 均值 Δ | 标准差 | 各次方向 | 逐种子 |
|---|---|---|---|---|---|
| office0_mono | V−A | **+0.080** | 0.096 | 2正/1负 | +0.181 / −0.009 / +0.068 |
| office0_mono | N−A | **−0.090** | 0.050 | 0正/3负 | −0.059 / −0.148 / −0.064 |
| office0_mono | VN−A | +0.016 | 0.062 | 2/1 | +0.066 / +0.036 / −0.054 |
| office0_mono | VN−V | −0.064 | 0.094 | 1/2 | −0.115 / +0.045 / −0.122 |
| office0_mono | VN−N | +0.106 | 0.088 | 3/0 | +0.126 / +0.184 / +0.010 |
| office0_mono | **A_repeat−A（噪声底）** | **+0.060** | **0.128** | 2/1 | +0.016 / **+0.204** / −0.040 |
| desk3_rgbd | V−A | −0.083 | 0.147 | 1/2 | −0.015 / −0.251 / +0.018 |
| desk3_rgbd | N−A | −0.019 | 0.250 | 2/1 | −0.295 / +0.044 / +0.193 |
| desk3_rgbd | VN−A | −0.033 | 0.272 | 2/1 | −0.336 / +0.051 / +0.188 |
| desk3_rgbd | VN−V | +0.050 | 0.328 | 2/1 | −0.321 / +0.302 / +0.169 |
| desk3_rgbd | VN−N | −0.013 | 0.025 | 1/2 | −0.042 / +0.007 / −0.006 |
| desk3_rgbd | **A_repeat−A（噪声底）** | **+0.013** | **0.159** | 2/1 | −0.159 / +0.041 / +0.156 |

**判读**：所有 |Δ| 都不超过同设置重复波动的量级（噪声底标准差 0.128 / 0.159，
单个种子的重复差最大到 **+0.204 dB**）。因此本轮**不宣布任何显著性**，
不说"V/N 提升了画质"，也不说"V/N 降低了画质"。按事先约定：**没有明确收益就保持
"不确定"**。

### 7.2 耗时与代价（这些是可以直接报的机制级数字）

| 场景 | 全部事件处理（3 种子均值） | 其中体素维护 |
|---|---|---|
| office0_mono | A 773.5 / A_repeat 809.8 / V 873.8 / N 859.6 / VN 813.4 ms | 0（V 未覆盖） |
| desk3_rgbd | A 1580.5 / A_repeat 1579.6 / V 1623.0 / N 1532.5 / VN 1590.6 ms | **V 58.9 ms，VN 60.1 ms（占 3.6–3.8%）** |

V 的代价是**可测的、且不依赖随机性**：desk3 上每次运行 7–9 次网格重建、
168k–311k 个高斯点被复制、39.6–72.2 ms。
端到端 `cum_mapping_ms` 的差（V−A 在 desk3 上 +321 ms）反而被噪声淹没
（A_repeat−A 差 +1544 ms），因为三个条件在同机串行跑、受其它进程影响。

### 7.3 五个判断

1. **A 类修复是否可以冻结：可以。**
   O_core 3 轮 × 8 case = **24/24 通过**，加载库身份经 `/proc/self/maps` + sha256 验证；
   O_raw 稳定 6 OK / 15 FAILED / 3 SIGSEGV（case7 三轮全崩）。
   唯一的异常已定位到测试夹具的未初始化输入（`opacity_lr_(opacity_lr_)`），
   修复后 60/60 次不再出现，**没有未解释异常**。

2. **V 是否减少冗余，代价和画质变化是什么：机制有覆盖的部分（desk3）能测到代价，
   画质方向不确定。**
   * 代价：7–9 次网格重建 / 168k–311k 点 / 39.6–72.2 ms，占事件处理时间的 3.6–3.8%。
   * 冗余：V=1 时同一批线采样点在插入前按"当前存活高斯占用"去重，
     V=0 时按"历史采样键"去重；两者在本轮 desk3 上写入的线高斯数差异没有超出
     重复波动，**没有足够证据说哪个更冗余**（`voxel_committed_keys` /
     `voxel_rejected_duplicates` 已在 summary 里逐运行记录）。
   * 画质：desk3 `V−A = −0.083 ± 0.147`，方向 1 正 2 负，落在噪声内 → **不确定**。
   * office0 上 V 完全未触发，那里的 V−A 只能当噪声读数。

3. **N 是否修正了事件后的地图不一致：没有可测的修正。**
   逐事件成对评测显示所有条件（含无 N 的 A）在事件后都会掉 1.6–3.1 dB，
   N 的均值差 +0.16 dB、跨种子 ±0.9 dB。机理上也不该期望更多：N 复用的是
   同一个"同龄门限"选择性变换，能移动的高斯只占总量的一小部分。

4. **VN 是否存在交互或额外开销：没有可测交互。**
   `VN−V = −0.064 ± 0.094`（office0）、`+0.050 ± 0.328`（desk3）；
   `VN−N = +0.106 ± 0.088`（office0）、`−0.013 ± 0.025`（desk3）。
   若 V、N 独立可加，`VN−A` 应约等于 `(V−A)+(N−A)`；desk3 上实测 −0.033 vs 预测 −0.102，
   office0 上 +0.016 vs −0.010，两者都在噪声内，**既不能证明也不能否定可加性**。
   额外开销：体素维护只出现在 V 开的两组（58.9 / 60.1 ms），N 本身没有可测开销。

5. **哪个版本适合作为正式 Photo-SLAM-L：**
   * **A 类修复（A1–A9）作为默认，必须保留**——这是唯一被官方对照实验证明的正确性差异。
   * **V、N 保留默认开启（= 当前 L1 行为），但论文里必须分别标注为"去重策略变更"和
     "新增一致性机制"，并注明本轮没有测到画质增益。**
   * 若要给"保守基线"，用 **L_core（V0N0）**；要给"候选正式版"，用 **L_VN（V1N1）**。
     两者在本轮 30 次运行中**不可区分**，因此选择应基于机制论证（V 有确定代价、
     N 无确定收益），而不是基于画质数字。
   * 与官方比较用 **O_core**（并声明基线已含公共修复）。

---

## 八、仍未覆盖 / 需要补测

| # | 缺口 | 影响 | 建议补测 |
|---|---|---|---|
| 1 | office0 的 N 在种子 1 完全未触发；office0 全部种子 V 未触发 | 该场景的 V/N 只能当噪声读数 | 换用**必然触发**的场景，或延长 live 运行直到触发；不要用 office0 下 V/N 结论 |
| 2 | 重放压缩了迭代轴（office0 1.32–1.35×，desk3 3.6–6.0×），导致"同龄集合"比 live 小 1.7–2.2× | 重放测到的变换点数不能代表 live | 按 `nat/steps` 同比缩放增密相关阈值（`densify_from_iter` / `opacity_reset_interval` / SH 升级），**作为一次性协议变更**，五条件共用 |
| 3 | 所有关键帧在重放 t=0 就已存在 | "关键帧插入时序"未被重放；合并 `created=0` | 支持"关键帧按事件入场"的重放 |
| 4 | 事件后的 1.6–3.1 dB 跌落只移动了部分高斯 | 这是**生产实现**的性质，不是本轮引入 | 应作为独立问题立项：写回是否应该移动全部可见高斯，而不是"同龄"子集 |
| 5 | 线高斯在 5000 步后几乎被剪光（office0 初始 49 → 末态 0–1） | 线高斯的长期存活能力未解决 | 与去重/写回无关，另立 |
| 6 | desk3 seed 1 进程在**重放完成后**未退出（teardown 卡住 18 分钟，被 kill） | 数据完整（5 个 condition 的 summary/steps/event_log 全部落盘后才卡），不影响本轮任何数字 | 见 §9 |

---

## 九、desk3 seed 1 的 teardown 卡住（如实记录）

现象：replay 全部结束并打印 `[Replay] all conditions finished in 132s`，
5 个条件的 `summary.txt` / `steps.csv` / `event_log.csv` 全部写完，但进程在
`main` 的 `training_thd.join()`（或 `Shutdown()` 的等待）处不再前进，18 分钟后被
`kill -9`。其余 5 个进程正常退出。

定位限制：容器内 `ptrace` 被禁用（`Could not attach to process … ptrace: Inappropriate
ioctl for device`），无法取到线程栈；只知道 54 个线程全部处于 futex 等待，
`ps` 的 40% 是**生命周期平均 CPU**，不是当时的占用。**没有证据表明它发生在
重放计算内部**，也没有证据表明它与 V/N 有关（卡住的是 A 之前的 live 段之后）。

对交付的影响：**不影响数据**（所有产物在卡住前已落盘并 flush；
`du`/`wc -l` 已核对）。

建议的规避（未在本轮生效，避免改动产生 30 次结果的二进制）：
在 `runReplay()` 结束后、所有文件 flush 之后调用 `std::_Exit(0)`，
即"重放跑完就硬退出"，绕开 ORB-SLAM3 的线程收尾竞态。
这是一个**只影响进程退出路径**的补丁，不可能改变任何 mapping 数字。
