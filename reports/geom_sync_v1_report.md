# geom-sync-v1 — 回环 / BA 位姿更新后的 Gaussian 地图一致性

基线：`51d6534b70`（raster-perf-v6，后端 C 已验收并冻结）。本轮只做几何同步正确性；不继续优化 CUDA，不新增结构损失/采样/回环算法。

---

## 一、基线保存与磁盘

| 项 | 状态 |
|---|---|
| 源码 | 锚点 `51d6534b70`，本轮只新增 `reports/geom_sync_v1_report.md` |
| 配置 | `cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml`、`cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml` |
| 运行命令 | §附录 |
| 二进制标识 | `binaries.sha256`（`bin/replica_rgbd` + `libORB_SLAM3.so`） |
| 加速基线 | 已在 `archive/raster-perf-v5/`、`archive/raster-perf-v6/`，**未重复复制** |
| 磁盘 | 88%，可用 218 GB（此前 96%）；`abl_out` 7.3 GB，未删除任何无备份结果，未改写历史 |

后续实验显式使用 `PHOTO_SLAM_RASTER_BACKEND=C`；A 回退入口保留；正常入口默认值**未改**。

---

## 二、同步路径审计（基于实际代码）

### 位姿约定

`KeyFrame::mTcw` 是 **Tcw（world → camera）**：`SetPose(Tcw)` 同时令 `mTwc = mTcw.inverse()`；`GetPose()` 返回 Tcw，`GetPoseInverse()` 返回 Twc（`ORB-SLAM3/src/KeyFrame.cc:140-148,195-205`）。局部变量名 `Tiw` 在本 fork 中仍是 Tcw。

### 1. 位姿更新如何发布

| 事件 | 写入位置 | 通知方式 |
|---|---|---|
| 局部 BA | `Optimizer.cc:3319-3323` `pKFi->SetPose(Tcw)`；点 `:3332` `SetWorldPos`；`:3346` `IncreaseChangeIndex` | `LocalMapping.cc:166` 推一个 `MappingOperation(LocalMappingBA)` |
| 回环（当前 KF） | `LoopClosing.cc:2104-2112`，应用**一个公共 loop Sim3**（`mg2oLoopScw`） | — |
| 回环（共视邻居） | `LoopClosing.cc:2148-2157`，**逐 KF** `S_ic * S_cw`（保持与当前 KF 的相对位姿） | — |
| 回环（点） | `LoopClosing.cc:2192-2197` 逐 KF Sim3 映射 | — |
| 回环（essential graph） | `LoopClosing.cc:2282` → `Optimizer.cc:13581` 逐 KF 写回，**非刚性图优化** | `LoopClosing.cc:2291` 推 `LoopClosingBA` |
| 回环后 GBA | `LoopClosing.cc:4556` `pKF->SetPose(pKF->mTcwGBA)`；点 `:4639/:4658` | **只有 `InformNewBigChange()` + `IncreaseChangeIndex()`（`:4662-4663`）** |

**通知机制**：唯一通道是 `Atlas::mqMappingOperations` 这个**轮询队列**，`addKeyFrame()` 在调用时刻**快照位姿**（`Atlas.h:134`）。mapper 侧在 `gaussian_mapper_line.cpp:4188-4192` 轮询。全仓库没有 `condition_variable` / `notify_*` / callback / observer。

### 2. 相机缓存何时刷新

`GaussianKeyframeLine` 在创建时 `setPose()` + `computeTransformTensors()` 一次（`gaussian_mapper_line.cpp:700-715` 初始建图、`handleNewKeyframe` 增量）。此后只有两个地方刷新：`LocalMappingBA` 分支（`:4216` 附近）与 `LoopClosingBA` 分支（`:4155-4157`），都是 `setPose(...)` + `computeTransformTensors()`。

### 3. Gaussian 中心与旋转是否接受几何更正

- **回环分支：是。** `LoopClosingBA` 里算 `diff_pose = new_pose.inverse() * old_Tcw`（即 `Twc_new ∘ Tcw_old`，world→world），对"该 KF 可见"的 Gaussian 调 `scaledTransformVisiblePointsOfKeyframe()`（`:4085-4100`），内核同时更新 `xyz_` 与 `rotation_`；线 Gaussian 的 `line_dir_w_` 也按同一旋转同步（`gaussian_model_line.cpp:1410-1424`），变化尺度时 `scaling_` 乘以 `s`（`:1426-1432`）。
- **局部 BA：否。** `LocalMappingBA` 分支只动相机位姿，不动任何 Gaussian 与元数据（`:4216-4222`）。阈值机制（`large_rot_th_=0.1 rad`、`large_trans_th_=1e-2`，`:155-156`）只存在于回环分支。

### 4. MapLine 端点 / 线锚点 / 方向

| 项 | 状态 |
|---|---|
| SLAM 侧 MapLine 端点 | 回环与 GBA 由 ORB-SLAM3 自己更新（`Optimizer.cc:529-551` `SetLineWorldPos`；GBA 线程 `:4889/:4908`） |
| Gaussian `line_dir_w_` | **已同步**（`gaussian_model_line.cpp:1410-1424`），与中心用同一旋转 |
| Gaussian `scaling_`（尺度变化） | **已同步**（`:1426-1432`） |
| Gaussian `xyz_init_`（线锚点，世界坐标） | **未同步 —— 确认缺口 G2** |

`xyz_init_` 不是装饰字段：`gaussian_model_line.cpp:262-271` 把它当作"直线上的一点参考锚点"送入线结构损失。回环后中心与方向都动了、锚点留在原世界位置，锚点与中心不再同源。该缺口当前不即时发作，只因为本轮实验一直关闭结构损失（`PHOTO_SLAM_LINE_LOSS_EVERY=0`）。

### 5. 是否存在可靠的参考关键帧关联

**没有。确认缺口 G3。**

- `GaussianKeyframeLine` 有 `creation_iter_`（`gaussian_keyframe_line.h:86`），Gaussian 行有 `exist_since_iter_`（`gaussian_model_line.h:365`），二者都是**全局迭代号**，不是来源 KF 的 id。
- 现有"归属"判据是 `markVisible(...)` 的可见性 + `|exist_since_iter_ - kf_creation_iter| < stableNumIterExistence()` 的**迭代邻近度**（`:4085-4098`、`:4445`）。这是启发式，不是真实来源。
- loop 分支用 `point_not_transformed_flags` 做"先到先变换"，避免同一 Gaussian 被多个 KF 重复变换——这恰好说明**没有真实归属**，只能用一次性的布尔标记掩盖。
- `clone` / `split` / `prune` 会按行复制或删除 `is_line_`/`line_dir_w_`/`xyz_init_`（`:3115-3116`、`:3275-3276`、`:1728-1730`），但**没有可继承的 ref-KF 列**，因为这一列不存在。

### 6. 遗漏 / 重复 / 中途施加

| 风险 | 结论 |
|---|---|
| 重复施加同一事件 | 天然幂等：diff 是"相对于 mapper 当前存位姿"算的，同一事件再来一次 diff≈I，被 `large_rot/large_trans` 挡掉。**但没有事件 id 可证明**（缺口 G4） |
| 回环后 GBA 的更正 | **完全遗漏 —— 确认缺口 G1（最严重）**。GBA 在 `opr` 已 push 之后异步运行，改完所有 KF 位姿与点坐标只调 `InformNewBigChange()`，mapper 永不获知 |
| 局部 BA 的更正 | 位姿刷新了，Gaussian 不动 —— 见 §3 |
| 中途施加 | 循环体在 `combineMappingOperations_withLine()`（`:802`/`:834`）里一次性消费整个队列，变换在 `mutex_render_` 下进行；与 `trainForOneIteration*` 互斥，**不在 forward/backward 之间改模型**。这一条现状是安全的 |

---

## 三、缺口清单

| ID | 缺口 | 严重度 | 本轮 |
|---|---|---|---|
| G1 | 回环后 GBA 的位姿/点更正不发布 `MappingOperation`，mapper 永久看不到 | 高 | 未修，见下 |
| G2 | `xyz_init_` 线锚点不随回环更正 | 中（当前结构损失关闭则不发作） | 未修 |
| G3 | 无可靠的逐 Gaussian 参考关键帧关联，只有可见性 + 迭代邻近度启发式 | 高（G1/G2 正确修复的前提） | 未修 |
| G4 | 无事件标识，无法证明"同一事件只应用一次" | 低 | 未修 |
| G5 | 局部 BA 只改相机不改 Gaussian | 中（量级未测） | 未修 |

**为什么本轮没有直接改**：G1 的正确修复不是"补一次 push"。GBA 是**非刚性**优化，每个 KF 有各自的 delta；把它应用到一个 Gaussian 上必须先知道"这个 Gaussian 来自哪个 KF"。当前代码没有这个关联（G3），loop 分支靠"可见 + 迭代邻近 + 先到先变换"近似。若把 GBA 也按同一近似应用，等于把一个已知不可靠的判据用到**更大范围、更大幅度**的更正上，可能比现在还差——而且无法验证。先补 G3 才能让 G1 有意义。

同理 G2：`scaledTransformVisiblePointsOfKeyframe` 的 `diff_pose` 张量在传入前做过 `.transpose(0,1)`（`gaussian_mapper_line.cpp:4113-4116`），现有 `line_dir_w_` 代码用 `slice(0,0,3).slice(1,0,3)` 当旋转矩阵。锚点要用**同一表达式**才能保证与中心同源；在未用一次真实回环核对这个布局之前直接写，风险是把锚点转错而不是不转。**不制造修复。**

---

## 四、本轮未做的实验

- 未跑 TUM `fr3_long_office_household` 的真实回环：数据集（`/workspace/code/SEGS-SLAM/datasets/tum/rgbd_dataset_freiburg3_long_office_household`，含 `rgb/depth/groundtruth.txt`）与配置（`cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml`）都已确认存在，但本轮预算用于审计，未进入运行阶段。
- 未做恒等更正/重复事件/刚体一致性等小验证——它们需要先有可注入事件的入口（G4 的一部分）。

**因此本轮不给任何"同步已正确"或"同步已修复"的结论。**

---

## 五、建议的下一步（按依赖顺序）

1. **G3 先做**：给 `GaussianModelLine` 增加一列 `ref_kf_id_`（`long`，创建时写入来源 KF 的 `mnId`；`increasePcd` 从 `MappingOperation` 里携带的 KF id 传入）。`clone`/`split` 继承父行，`prune` 同行删除。这一步同时让 `clone`/`split`/`prune` 后的关联可验证（长度与 `xyz_` 一致）。
2. **G4**：给 `MappingOperation` 加 `mnEventId`（Atlas 内单调递增），mapper 打印并记录已应用的最大 id。
3. **G1**：新增 `OprType::GlobalBA`，只携带 `(kf_id, pose)` **轻量**列表（不要复用 `addKeyFrame`——它会 `imgLeftRGB.clone()` + `GetKeyPointInfo`，对全图 KF 代价过高）。mapper 侧按 `ref_kf_id_` 归属应用，不再是可见性启发式。
4. **G2**：在同一归属基础上同步 `xyz_init_`（用与中心完全相同的 `diff_pose` 表达式），并用一次真实回环核对旋转布局。
5. **G5**：量化局部 BA 的位姿增量分布，再决定是否需要同步（可能小于一个像素，不需要动）。

## 附录：运行命令

```bash
cd /workspace/code/PSL-perf
# 加速基线（后端 C，显式指定）
PHOTO_SLAM_RASTER_BACKEND=C PHOTO_SLAM_LINE_LOSS_EVERY=0 \
  bash scripts/run_online_ab.sh C abl_out/<out> office0
# 回环测试（本轮未执行）
PHOTO_SLAM_RASTER_BACKEND=C PHOTO_SLAM_LINE_LOSS_EVERY=0 \
  bin/replica_rgbd ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
    cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
    /workspace/code/SEGS-SLAM/datasets/tum/rgbd_dataset_freiburg3_long_office_household \
    abl_out/geom_sync_loop no_viewer
```
