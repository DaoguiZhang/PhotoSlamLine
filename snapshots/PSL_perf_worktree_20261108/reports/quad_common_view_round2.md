# 第二轮：共同视角评价与计算预算归因（基于 quad_office0）

本轮**不覆盖** `reports/quad_office0_report.md`；旧表保留，并明确标注为
“各运行自身关键帧训练视图指标”。本轮不修改 CUDA / 结构损失 / 采样 / 增密 / 回环算法，
不重跑四组矩阵。代码保持 `a26e835d03`（quad harness）；二进制与结果标识见
`results/quad_office0/provenance.txt`（`replica_rgbd` `6800cc2d…`、
`replica_rgbd_point` `3b34c1db…`、`libORB_SLAM3.so` `6d9ace12…`、
`libgaussian_mapper.so` `6a87b655…`）。

## 1. 结论修正（保留旧结果，不改数字）

1. L 管线用 C：整体墙钟 **−29.7±4.1 s（−19%）**，mapping 时间 **−32.0±3.4 s（−24.5%）**。
2. L_fast 相对 L_ref 的最终 PSNR **−0.34±0.22 dB**（三轮同向），SSIM −0.0004，
   LPIPS +0.0023（更差）。**不再写“质量保持”**；这只是本次运行、这套评价下的观测。
3. 不能由此推出“C 必然降低质量”：在线地图、位姿、更新次数（L_fast 25.0k vs L_ref 21.4k）
   与参与评价的关键帧集合都不同，属混杂。
4. P 是**同仓库纯点构建**（`USE_LINE_GAUSSIAN=0` + `TrackRGBD` + `LINE_MODE=0`），
   不是实际运行的上游 Photo-SLAM。上游树哈希 `82c3cc06…` 仅作身份核对，未参与比较。
5. L 的画质优势伴随 **2.4× optimizer 更新**；**尚不能归因于线信息本身**（见 §3）。

## 2. 后端 A/C 的实际分发核实

证据取自 `cuda_rasterizer/backward.cu` 中 `BACKWARD::render` **内部的**
`[RasterBackend]` 打印（即真正的 host 端分发处），不是入口参数回显：

| 运行 | 打印 | steps | KFs | mapping s |
|---|---|---|---|---|
| P_ref_r1/r2/r3 | A reference (per-contribution atomicAdd) | 8581 / 8981 / 8781 | 100/102/104 | 42.9 / 44.3 / 44.5 |
| P_fast_r1/r2/r3 | C converged + 5-stage warp tree | 10581 / 11081 / 10581 | 101/106/105 | 41.5 / 44.1 / 42.5 |
| L_ref_r1/r2/r3 | A reference | 21038 / 21497 / 21529 | 112/115/119 | 131.6 / 127.8 / 132.0 |
| L_fast_r1/r2/r3 | C converged + 5-stage warp tree | 25027 / 25058 / 24942 | 119/120/113 | 98.6 / 99.7 / 97.2 |

结论：12/12 次运行确实各自走了 A 或 C；`line_gaussians` 在 P 组恒为 0、L 组非 0
（16–310），与条件定义一致。
**未完成**：单地图、单视图的 A/C 反向内核微基准（无 optimizer.step、预热、清梯度）。
仓库现有微基准入口面向“同一冻结进程”，需要先解决离线加载后可渲染的相机对象
（见 §4 的同一阻塞），本轮未重写性能框架，明确标为未做。

## 3. 为什么 L 有约 2.4 倍更新（代码 + 日志可核）

| cond | KFs | 总 steps | mapping s | 每 KF 平均 steps |
|---|---|---|---|---|
| P_ref | 102 | 8781 | 43.9 | 86 |
| P_fast | 104 | 10750 | 42.7 | 103 |
| L_ref | 115 | 21350 | 130.5 | 186 |
| L_fast | 117 | 25010 | 98.5 | 214 |

两个管线**调度不同**（这是既有实现差异，不是本轮改动）：

- **L（`GaussianMapperLine`）**
  - 在线：每新增关键帧 `Mapper.new_keyframe_times_of_use = 20` 次训练额度，
    `GausPyramid.sub_level_times_of_use = 15`（各金字塔子层），
    `Mapper.local_BA_increased_times_of_use = 0`（本配置不加 BA 额度）。
  - 收尾：`SLAM ended. Resetting remaining_times_of_use for global refinement...` 后，
    每 KF 额度置 1000，并**固定追加** `Optimization.refined_gaussian_max_iter_num = 10000`
    次迭代（`target_stop_iter = iteration + 10000`，`while (iteration < target_stop_iter || isKeepingTraining())`）。
  - 因此 `总 steps ≈ 在线 + 10000`：L_ref 21350 → 在线 ≈ 11350；L_fast 25010 → ≈ 15010。
- **P（`GaussianMapper`）**
  - 每轮循环一次 `trainForOneIteration()`，受 `iterations_ = 30100` 上限与
    SLAM 结束标志约束；收尾循环条件是
    `iter - SLAM_stop_iter <= 0.8*densify_interval || iter % densify_interval <= 0.8*densify_interval
     || isKeepingTraining()`，**没有** `refined_gaussian_max_iter_num` 这样的固定追加预算。
  - 因此 P 的 steps 基本全是在线阶段（8781–10750）。

**关键量化**：扣除 10000 次固定精修后，L_fast 的在线更新 ≈15010，P_fast ≈10750；
L_ref ≈11350 vs P_ref 8781。也就是说 **L 的 2.4× 中约 10000 次（≈40%）来自序列结束后
的固定追加精修**，这部分与线信息无关，任何管线都能获得。剩余差额来自在线每 KF 额度
（20 + 金字塔额度）与更多 KF（+12%）。

已知/未知边界：
- 已知（代码常量 + 本配置 + 日志总数）：上述额度与总步数。
- **未知**（旧日志没有分阶段计数）：在线阶段与精修阶段各自的**实际**步数与耗时精确值；
  上面的在线值由“总数 − 10000”推得，若 `isKeepingTraining()` 延长了精修，则在线步数被高估。
  训练分辨率/金字塔层级的使用分布、SH 阶数分布同样未逐阶段记录。
- 因此本轮**不**声称“线几何导致 2.4×”，只报告“L 的调度含 10000 次固定追加精修”。

## 4. 共同视角评价：未完成，阻塞项与最小补充

**为什么没做完（不是缺少动机，是缺少可运行的离线渲染路径）：**

1. `bin/view_result` 需要 ImGui 交互窗口，无法脚本化；`loadPly(ply, camera_path)` 的
   `camera_path` 是 ORB 风格 `cv::FileStorage` YAML，只创建 **一个 `Camera`**，
   **不创建可渲染的 keyframe/view 对象**。仓库里没有“给定 PLY + 任意位姿直接出图”的入口。
2. 12 次运行保存了 `*_shutdown/ply/point_cloud.ply`、`CameraTrajectory_TUM.txt`、
   `KeyFrameTrajectory_TUM.txt`、`image*/`（训练视图）与 `pose_TUM.txt`；但**没有**保存
   map→GT 的 SE(3) 对齐结果，也没有保存位姿/内参一体的评价相机清单。
   （对齐本身可由 `CameraTrajectory_TUM.txt` + GT 重算，ATEm 的 est-centric SE(3) 流程已在
   `scripts/eval_tum_se3_v2.py` 中实现，可直接复用其对齐。）
3. 因此缺的是**一条离线渲染链路**，不是缺地图。

最小补充（下一步只需做这一件，不改算法）：
- 新增 `examples/eval_common_views.cpp`：`GaussianMapper`/`GaussianMapperLine` + `loadPly()`，
  用 ORB yaml 建 `Camera`，对每个评价位姿构造一个最小 view（位姿、W/H、fx/fy/cx/cy、
  world_view/proj 矩阵）调用现有 `GaussianRenderer(WithLine)::render`，
  只保存 PNG + 每帧 CSV；不训练、不 step、不改任何超参。
- Python 侧：用 `CameraTrajectory_TUM.txt` 与 `pose_TUM.txt` 求 SE(3)（无尺度）对齐，
  把 GT 评价位姿变到该地图坐标系，写出每 run 的 `eval_cameras.txt`；
  24 个固定 frame ID（覆盖全序列、等间隔抽样，固定后写入哈希清单）。
- 指标：**同一批渲染**上算 PSNR/SSIM（复用 `loss_utils` 或 Python 复算并核对）
  与 LPIPS（CPU），旧表另存不混合。
- 同时记录“哪些评价帧曾参与训练/初始化取色”（关键帧 fid 与帧 ID 的交集），
  本轮只称**共同视角评价**，不称严格留出。

在上述链路落地前，任何“共同视角”数字都不应给出；本轮没有用旧表冒充。

## 5. 预算追平实验：未开始，原因与最小方案

**阻塞**：12 次运行从未设置 `PHOTO_SLAM_SNAPSHOT_DIR/AT`，因此没有可恢复的
Gaussian/Adam/调度检查点。若现在“续训”，只能重置优化器与调度状态，
那会是**新的训练**而不是原训练的续跑——按本轮约束，不做这种静默重置。

最小方案（明确命名，作为诊断而非在线调度）：
1. 只跑 P_fast 与 L_fast 各一次，开启现有快照机制（`PHOTO_SLAM_SNAPSHOT_AT` 取序列尾声），
   并在快照中**额外**保存 6 组 Adam `step/exp_avg/exp_avg_sq`、参数组 LR、`iteration_`、
   `kfs_used_times_`（`RecoverySnapshot` 已有这些字段，导出即可）。
2. 用 §4 的共同视角链路评价“自然在线阶段结束点”。
3. 冻结几何，按现有训练规则继续精修，直到达到**预先写入 manifest 的共同累计训练时间预算**；
   预算由本轮 L_fast 的 mapping 时间（98.5 s）与 P_fast（42.7 s）确定同一个值，
   并在看到新质量结果前固定。
4. 分别记录在线/追加两阶段的步数与耗时，评价 24 共同视角。
5. 命名：`budget-catchup-diagnostic`；不得称为严格相同在线调度，也不能隔离线初始化作用。

## 6. 对五个问题的回答（当前证据能到哪一步）

1. **共同视角下 3 dB 还剩多少？** 未知。共同视角渲染链路缺失（§4），旧表是训练视图
   （L 训练更久，天然占优），不能回答。
2. **P 是否真的调用 C？同输入下有内核收益？** 调用已证实（12/12 在分发处打印）。
   内核级收益未测（§2）；系统级 P_fast−P_ref 的 mapping 差为 −1.21±0.88 s，不显著。
3. **L 的额外更新来自？** ≈40% 是固定 10000 次收尾精修；其余来自在线每 KF 20 次额度、
   金字塔子层额度与 +12% 的关键帧。已量化到配置常量与总数，分阶段实测步数未知。
4. **预算追平后差距？** 未测（无可恢复检查点，§5）。按 §3，预期会明显缩小，但未验证前
   不作为结论。
5. **哪些支持整条 L 管线、哪些不能归因于线？** 目前能支持的只是
   “整条 L 管线在本评价协议下 PSNR/SSIM/LPIPS 更好，代价是 2.2× 墙钟、2.4× 更新”；
   “线几何本身贡献多少”与“其中多少来自额外计算”都尚无证据，等 §4/§5。
