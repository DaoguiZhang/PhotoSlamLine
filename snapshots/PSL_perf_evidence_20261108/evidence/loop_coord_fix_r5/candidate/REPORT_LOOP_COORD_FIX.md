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

恢复 `+5.50910894 dB`（恢复 77.5% 的即时跌幅）；残留 `−1.60107104 dB`。
23/24 视角移动。Worst fixed view fid 55：`−5.16902 dB`。

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

## 4. 100 步稳定性

同一预事件状态的 A0/A1/C 100 步（`common-abc-fix-r5`）：

| step | A0 | A1 | C |
|---:|---:|---:|---:|
| 0 | 30.58399487 | 30.58399487 | 30.58399487 |
| 100 | 32.5411311 | 32.54129354 | 32.54093504 |

C−A0 = `−0.00019606 dB`，在 A 重复噪声内。修复只做刚体/Sim(3) 坐标变换，
未改学习率、损失、增密或优化器策略；该 100 步检查不显示新的不稳定。

注意：事件诊断的 100 步来自 pre-event common-state 状态，不是“修后状态再跑 100 步”；
本轮没有额外增加修后 100 步 arm，以避免扩大范围。

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
- 对照 current C：mean `26.5209`，last `26.0256`
- old A：mean `31.5032`，last `24.7688`

这是干净 LOOP 回归：fixed 的 mean 比 current C 高 `+2.7763 dB`，last 低
`−0.1257 dB`。它证明修复后真实 LOOP 能触发且不导致终端崩溃/明显退化；
但这是不同 run 的轨迹，不能把均值差全部归因于修复。修复的逐步因果证据仍是
同事件 step0 对照。

## 6. 未解释/剩余差距

- 同事件即时残留 `−1.6011 dB`。
- `3801` 个 missing rows（参考 KF 不在事件内）以及非事件 KF 拥有的 Gaussian
  仍可能出现在事件视角中；本轮没有逐视角可见性计数来把残留完全归因。
- **GBA 默认变形仍 disabled**。同一 run 的后续 GBA 事件：
  before `28.2804`，pose-only `24.0143`（`−4.2662`），corrected map `29.2305`
  （`+0.9501`）。这是 LOOP 修复后暴露出的另一个独立缺陷，不属于本轮
  “最小 LOOP/Gaussian 衔接修复”的范围，但会影响终端质量。
- Adam 二阶矩未做各向异性旋转；这是已有文档化近似。

## 7. 复现

```bash
SRC_FIX=/tmp/psl-room1-control/src-fix \
SRC_AUDIT=/tmp/psl-room1-control/src-fix-audit \
OUT=/tmp/psl-room1-control/repro-loop-fix \
bash /workspace/code/PSL-perf/scripts/repro_loop_coord_fix.sh
```

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
