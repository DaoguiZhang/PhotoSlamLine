# GBA 恢复实验收尾：状态一致性结论（基于 d2c8f94283 + 80c68c5cac）

结论先行：

1. “首次 backward 梯度相同、首次 rotation 更新不同” 的**前提不成立**。之前看到
   相同的 `first_backward,rotation` 行是**参数**，不是梯度；同一次捕获里
   `first_backward,grad_rotation` 已经不同（max_abs 1.42e-08，rel-L2 6.0e-06）。
2. 优化器本身完全确定：同一输入离线重放 5 次逐位相同，且**真实训练的首步结果
   与离线重放逐位相同**（raw diff = 0，协方差 diff = 0）。没有外围写入、没有
   张量重绑定、没有重复注册、没有一步更新多次、快照不与运行张量共享 storage。
3. 差异唯一的来源是 CUDA backward 的 1 ULP 梯度差；Adam 的 `eps=1e-15` 低于
   float32 分辨率，把 1 ULP 放大到首步 6.8e-3（rotation），10 步后 4.65e-2。
   所以 0.039 量级是**累积放大**，既不是参数 ULP，也不是恢复缺陷。
4. 修复项只有两处，且都不改训练算法：LR 恢复（只影响重复性检查本身）与形变默认
   关闭（保留显式开启入口）。
5. 500 步复核仍然没有稳定收益：保留负结果，强制形变实验结束。

## 一、默认行为

- GBA 事件照常被消费：`groups`、`last_applied_gba_event_`、事件版本、相机位姿
  更新全部照旧；只有 Gaussian 坐标不做刚性形变。
- 形变默认**关闭**；显式开启：
  `PHOTO_SLAM_GBA_APPLY_CORRECTION=1`。
- legacy LOOP 更正路径完全不变。
- 日志区分两条路径：
  - 默认：`[GBA-Apply] deformation DISABLED (default; set …=1 to enable). Camera poses are still updated; N deltas not applied`
  - 开启：`[GBA-Apply] deformation ENABLED explicitly (…=1)`，随后
    `moved=77759 nonzero_moved=74065`。
- 生产路径回归：默认运行 exit 0、`moved=0`、计数 79760→79760；开启运行 exit 0、
  `moved=77759`、`free(): invalid pointer` 未复现。

## 二、真正送入 Adam 的状态（`adam_audit.csv`）

在 `optimizer_->step()` 的直接入口按参数组记录，rotation 组（第 5 组）摘要：

| 项 | 值 |
|---|---|
| 模型成员 vs 优化器参数 TensorImpl | 相同（`same_impl=1`，`same_data_ptr=1`） |
| shape / stride / dtype / device | `[79760,4]` / `[4,1]` / float / cuda:0 |
| requires_grad / is_leaf | 1 / 1 |
| `p.grad()` | defined，data_ptr 与 `param.grad()` 一致，L2 ≈ 5.5e-3 |
| Adam state 按**参数身份**查找 | found，`step` 15563→15564→…（每步 +1） |
| exp_avg / exp_avg_sq | defined（L2 见文件） |
| max_exp_avg_sq | 未定义（`amsgrad=0`，与选项一致） |
| 参数组选项 | lr=0.001，betas=(0.9,0.999)，eps=1e-15，weight_decay=0，amsgrad=0 |
| 重复注册 | `dup_param_count=1`，`group_param_count=1` |
| 同一步更新次数 | `step_calls_this_iter=1`（全部捕获步） |
| 快照与运行张量共享 storage | 否（`snapshot_shares_storage=0`） |

模型成员指纹与优化器参数身份一致，因此 `paramFingerprint()` 与
`adamFingerprint()` 的组合是可信的（此前只是不知道二者是否指向同一张量）。

## 三、单步隔离重放（无渲染/损失/相机/事件）

对每个复原臂的**真实** rotation 参数、`p.grad()`、Adam state、选项做深拷贝，用同一
`torch::optim::Adam` 在新参数上重放 5 次，记录三个时刻：step 入口、step 返回并
`cuda synchronize()` 后、按生产的 `q/||q||` 归一化后。

`first_update_replay.csv`（RepeatA，RepeatB 相同）：

| stage | 5 次重放 raw max_abs | 四元数角（q/-q 等价） | 协方差 max_abs |
|---|---|---|---|
| entry | 0 | ≤5e-6 deg | 0 |
| after_step | 0 | ≤5e-6 deg | 0 |
| after_normalize | 0 | ≤5e-6 deg | 0 |

`first_update_actual.csv`（真实训练首步 vs 重放）：

| arm | raw max_abs | quat deg | cov max_abs | same_impl | state_step |
|---|---|---|---|---|---|
| RepeatA | 0 | ≤5e-6 deg | 0 | 1 | 15563 |
| RepeatB | 0 | ≤5e-6 deg | 0 | 1 | 15563 |

说明：四元数角的 ≤5e-6 deg 是 float32 表示极限（raw 差恰为 0 是更强的结论）；
旧的 `acos(|dot|)` 写法在 7.5 万行上有约 0.1 deg 的数值地板，见
`gba_recovery_diag_report.md` 中 0.0969 deg 的读数，那一行本身就是地板值。

## 四、两臂输入的差异定位（`first_update_cross.csv`）

| 输入 | A vs B |
|---|---|
| param | 0 |
| **grad（rotation）** | **1.42318e-08** |
| exp_avg | 0 |
| exp_avg_sq | 0 |
| Adam step | 15563 / 15563 |
| lr / eps / wd / betas | 完全相同 |
| 四元数角 / 协方差 | 6.6e-06 deg / 0 |

也就是说：**唯一不同的输入就是首次 backward 的梯度**（1–2 ULP，来自 CUDA
光栅化 backward 的归约顺序），其余全部逐位相同。更新结果对输入是确定的，所以
首步 6.8e-3 的 rotation 差完全是这个梯度差经过 Adam 放大后的结果。

放大机制（实测，不改生产）：rotation 的 `exp_avg_sq` 长期均值只有约 1.76e-6（L2），
`eps=1e-15` 对 float32 的 `sqrt(v)`（≈1e-7 起）等于没有下限，梯度里 1 ULP 的噪声
可以直接改变某些分量的步长符号/大小。首步 rotation 差 6.8e-3，10 步后 4.65e-2
（`repeat_diag.csv`）。这属于**条件数/数值敏感度**，不是实现错误；本轮不修改 eps，
也不放宽容差。

## 五、旧结果受影响范围（LR 恢复缺陷）

- 缺陷：`restoreRecoverySnapshot` 早期版本不恢复 6 个参数组的 LR，导致
  `lr_equal=0`。
- 实际影响范围：**仅重复性检查本身**。训练体在每次迭代渲染前都会重新设置全部
  6 组 LR（`updateLearningRate` 只改第 0 组，其余 5 组用固定学习率
  `setFeature/Opacity/Scaling/RotationLearningRate`），所以任何一步真正送入 Adam 的
  LR 都是当步重新写入的，与快照里的值无关。
- 因此：**没有任何优化曲线因这个缺陷失效**；受影响的只是 `lr_equal` 一行。
  - 受影响运行：所有带 `PHOTO_SLAM_GBA_RECOVERY*` 的运行（v1 `8fba880349` 及更早的
    诊断 run 1），以及它们的 `repeat_check`/`repeat_consistency`。
  - 不受影响：Step 0 诊断（不执行 `optimizer.step()`）、所有生产运行、所有质量曲线。
  - 原 v1 结果按原样保留，不混入修复后的数字；v1 报告已加状态注记。
- 另一处已被推翻的表述：v1 报告里“训练后张量 ULP 级漂移”不成立。实测首步就是
  6.8e-3（rotation），10 步 4.65e-2，属放大后的结果。v1 报告同步更正，数字未改。

## 六、500 步复核（同一事件、同一共同状态、一对两序）

命令见第七节。两臂路径确认不同：
`arm=PoseOnly deformation=OFF(pose-only)` / `moved=0`；
`arm=PoseAndMap deformation=ON(applyGbaCorrections)` / `moved=74005 nonzero=70839`。

| step | PoseOnly PSNR/SSIM | PoseAndMap PSNR/SSIM | ΔPSNR | ΔSSIM |
|---|---|---|---|---|
| 0 | 21.7206 / 0.752931 | 21.7548 / 0.751060 | +0.0342 | −0.00187 |
| 10 | 21.9206 / 0.755762 | 21.9187 / 0.753855 | −0.0019 | −0.00191 |
| 50 | 22.2721 / 0.764424 | 22.2064 / 0.761899 | −0.0657 | −0.00253 |
| 100 | 22.3676 / 0.766307 | 22.3239 / 0.764901 | −0.0437 | −0.00141 |
| 250 | 22.3684 / 0.767156 | 22.3385 / 0.766547 | −0.0299 | −0.00061 |
| 500 | 22.3637 / 0.771013 | 22.3612 / 0.770809 | −0.0025 | −0.00020 |

反向顺序（`_r2`）同样：step500 ΔPSNR −0.0027、ΔSSIM −0.00016。
时间：PoseOnly 2.447/2.406 s，PoseAndMap 2.549/2.532 s（sync 0.015 vs 0.102 s），
计入同步后 PoseAndMap 更慢。

结论：**仍无稳定质量收益，且计入同步更慢**。结束强制 GBA 形变的收益实验，不再扫
比例/权重，不新增模块。

## 七、精确复跑命令

```bash
cd /workspace/code/PSL-perf
./scripts/build_two_level.sh

# (A) 状态审计 + 单步重放（写 adam_audit.csv / first_update_*.csv）
PHOTO_SLAM_RASTER_BACKEND=C PHOTO_SLAM_LINE_LOSS_EVERY=0 \
PHOTO_SLAM_GBA_RECOVERY_DIAG=abl_out/gba_recovery_diag \
PHOTO_SLAM_GBA_RECOVERY_VIEWS=24 \
PHOTO_SLAM_GBA_UPDATE_AUDIT=abl_out/adam_audit.csv \
  bin/tum_rgbd ORB-SLAM3/Vocabulary/ORBvoc.txt \
  cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
  cfg/gaussian_mapper/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
  /workspace/code/SEGS-SLAM/datasets/tum/rgbd_dataset_freiburg3_long_office_household \
  cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg3_long_office_household.txt \
  /tmp/gba_audit_run no_viewer

# (B) 500 步一对两序
PHOTO_SLAM_RASTER_BACKEND=C PHOTO_SLAM_LINE_LOSS_EVERY=0 \
PHOTO_SLAM_GBA_RECOVERY=abl_out/gba_confirm \
PHOTO_SLAM_GBA_RECOVERY_STEPS=500 \
PHOTO_SLAM_GBA_RECOVERY_EVAL_STEPS=0,10,50,100,250,500 \
PHOTO_SLAM_GBA_RECOVERY_VIEWS=24 \
  bin/tum_rgbd ORB-SLAM3/Vocabulary/ORBvoc.txt \
  cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
  cfg/gaussian_mapper/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
  /workspace/code/SEGS-SLAM/datasets/tum/rgbd_dataset_freiburg3_long_office_household \
  cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg3_long_office_household.txt \
  /tmp/gba_confirm no_viewer

# (C) 生产默认（形变关闭）与显式开启
PHOTO_SLAM_RASTER_BACKEND=C PHOTO_SLAM_LINE_LOSS_EVERY=0 \
  bin/tum_rgbd … /tmp/gba_default no_viewer            # moved=0
PHOTO_SLAM_RASTER_BACKEND=C PHOTO_SLAM_LINE_LOSS_EVERY=0 \
PHOTO_SLAM_GBA_APPLY_CORRECTION=1 \
  bin/tum_rgbd … /tmp/gba_enable  no_viewer            # moved=77759
```

原始数据：`reports/gba_state_audit/`（`adam_audit.csv`、
`first_update_replay.csv`、`first_update_actual.csv`、`first_update_cross.csv`、
`repeat_diag.csv`、`repeat_consistency.csv`、`confirm_summary.csv`、
`confirm_<arm>.csv`）。
