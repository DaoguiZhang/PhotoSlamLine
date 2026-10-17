# HANDOFF — geom-sync / gba-recovery（v1 + 状态一致性收尾）

状态：**v1 与复核均为负；强制 GBA 形变实验结束。GBA 形变默认关闭
（`PHOTO_SLAM_GBA_APPLY_CORRECTION=1` 显式开启），事件消费、相机更新、
事件版本与 legacy LOOP 更正全部保留。**
HEAD：本次提交。工作树干净（构建产物一并提交）。

## 本轮结论（详细见 `reports/gba_state_audit_report.md`）

- “首次 backward 梯度相同、首次 rotation 更新不同” 不成立：相同的是**参数**
  （`first_backward,rotation`），梯度行 `first_backward,grad_rotation` 差
  1.42e-08（rel-L2 6.0e-06）。
- 优化器确定：同输入离线重放 5 次逐位相同；真实训练首步与重放**逐位相同**
  （raw 0 / 协方差 0）。无外围写入、无重绑定、无重复注册
  （`dup_param_count=1`）、无一步多次更新（`step_calls_this_iter=1`）、
  快照不共享 storage。
- 模型成员与优化器参数同一张量（`same_impl=1`、`same_data_ptr=1`），
  Adam state 按参数身份命中，`step` 每步 +1；选项 lr 0.001 / betas
  (0.9,0.999) / eps 1e-15 / wd 0 / amsgrad 0。
- 放大机制：`eps=1e-15` 低于 float32 分辨率，1 ULP 梯度差 → 首步 rotation
  6.8e-3 → 10 步 4.65e-2。0.039 是累积放大，不是参数 ULP，也不是恢复缺陷。
  不修改 eps，不放宽容差。
- LR 恢复缺陷只影响 `lr_equal` 检查本身：每次迭代渲染前 6 组 LR 全部重设，
  仅第 0 组与迭代相关。质量曲线均不受影响；Step 0 无 `optimizer.step()`。
  v1 报告已加状态注记，原数字未改。

## 500 步复核（同一事件、同一共同状态，一对两序）

两臂路径确认不同：`arm=PoseOnly deformation=OFF(pose-only) moved=0`；
`arm=PoseAndMap deformation=ON(applyGbaCorrections) moved=74005 nonzero=70839`。

| step | PoseOnly PSNR | PoseAndMap PSNR | ΔPSNR | ΔSSIM |
|---|---|---|---|---|
| 0 | 21.7206 | 21.7548 | +0.0342 | −0.00187 |
| 10 | 21.9206 | 21.9187 | −0.0019 | −0.00191 |
| 50 | 22.2721 | 22.2064 | −0.0657 | −0.00253 |
| 100 | 22.3676 | 22.3239 | −0.0437 | −0.00141 |
| 250 | 22.3684 | 22.3385 | −0.0299 | −0.00061 |
| 500 | 22.3637 | 22.3612 | −0.0025 | −0.00020 |

反序同向（step500 ΔPSNR −0.0027）。时间 PoseOnly 2.447/2.406 s，
PoseAndMap 2.549/2.532 s（sync 0.015 vs 0.102 s）。无稳定收益、计入同步更慢。

## 生产默认

- 默认：`[GBA-Apply] deformation DISABLED (default; …=1 to enable)`，`moved=0`，
  计数不变（79760→79760），exit 0。
- 显式开启：`moved=77759`，exit 0，`free(): invalid pointer` 未复现。
- 渲染后端 C（`PHOTO_SLAM_RASTER_BACKEND=C`）、`PHOTO_SLAM_LINE_LOSS_EVERY=0`。

## 复跑命令

```bash
cd /workspace/code/PSL-perf && ./scripts/build_two_level.sh
# 审计 + 单步重放
PHOTO_SLAM_RASTER_BACKEND=C PHOTO_SLAM_LINE_LOSS_EVERY=0 \
PHOTO_SLAM_GBA_RECOVERY_DIAG=abl_out/gba_recovery_diag \
PHOTO_SLAM_GBA_UPDATE_AUDIT=abl_out/adam_audit.csv \
PHOTO_SLAM_GBA_RECOVERY_VIEWS=24 \
  bin/tum_rgbd ORB-SLAM3/Vocabulary/ORBvoc.txt \
  cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
  cfg/gaussian_mapper/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
  /workspace/code/SEGS-SLAM/datasets/tum/rgbd_dataset_freiburg3_long_office_household \
  cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg3_long_office_household.txt \
  /tmp/gba_audit_run no_viewer
# 500 步一对两序
PHOTO_SLAM_RASTER_BACKEND=C PHOTO_SLAM_LINE_LOSS_EVERY=0 \
PHOTO_SLAM_GBA_RECOVERY=abl_out/gba_confirm \
PHOTO_SLAM_GBA_RECOVERY_STEPS=500 \
PHOTO_SLAM_GBA_RECOVERY_EVAL_STEPS=0,10,50,100,250,500 \
PHOTO_SLAM_GBA_RECOVERY_VIEWS=24 \
  bin/tum_rgbd … /tmp/gba_confirm no_viewer
```

## 产物与限制

- `reports/gba_state_audit_report.md`、`reports/gba_state_audit/`（adam_audit、
  first_update_replay/actual/cross、repeat_diag、confirm_summary、confirm_<arm>）。
- `reports/gba_recovery_v1_report.md`（原数字 + 状态注记）、
  `reports/gba_recovery_diag_report.md`（含更正说明）。
- 限制：单事件、单序列；24 固定评价视角；无深度输出；LPIPS 需 CPU
  （`scripts/lpips_gba_recovery.py --device cpu`）；单参考 KF 刚性近似。
- 不再扫描比例/权重，不新增模块追求正结果。

## office0 四组在线对照（本轮）

- 四条件：P_ref/P_fast（同仓库 `USE_LINE_GAUSSIAN=0` 纯点构建，`TrackRGBD`+`LINE_MODE=0`）
  × A/C 后端；L_ref/L_fast（线构建，`LINE_MODE=2`，结构损失**按权重**置 0）× A/C。
  四组共用同一 `gaussian_quad.yaml`；GBA 形变保持默认关闭。
- 3 轮交错串行（R1 P_ref→P_fast→L_ref→L_fast；R2 反序；R3 交叉），12/12 exit 0。
- 结果：渲染器在 L 上 mapping −32.0±3.4 s（−24.5%）、墙钟 −29.7±4.1 s（−19%），代价 PSNR −0.34±0.22 dB；
  在 P 上无可测收益。线 vs 点：+3.1~3.2 dB / SSIM +0.014 / LPIPS −0.026，代价 2.2× 墙钟、2.4× 更新数；
  ATE 差异 ≤0.5 mm（噪声级）。
- 纯点基线身份：上游 `/workspace/code/Photo-SLAM`（无线符号，树哈希 `82c3cc06…`）已核实但未参与比较；
  本轮 P 为同仓库纯点构建 `P_ro`，不称原版等价。关键：RGB-D 原先只走 `TrackRGBDWithLine`，
  仅 `LINE_MODE=0` 不是纯点前端。
- 报告 `reports/quad_office0_report.md`；一条命令 `ROUNDS=3 ./scripts/quad_office0.sh`；
  产物在 `results/quad_office0/`（已 gitignore，含 `provenance.txt`/`summary.md`/`per_run.csv`）。
- 交付形态：源码/脚本入 Git；二进制与结果独立归档，不再提交到源码 Git。
