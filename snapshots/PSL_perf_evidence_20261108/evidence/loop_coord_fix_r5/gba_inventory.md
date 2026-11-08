# 非零 GBA 100 步输入盘点

## 已有

- `r5/gba_evt1/summary.csv`：非零 GBA 的 24 视角 step0 指标
  （before 28.2804, pose-only 24.0143, corrected 29.2305, ordinary PSNR）。
- `r5/gba_evt1/step0_per_view.csv`、`event_scope.csv`、三组
  `eval_step0/per_frame.csv`。
- `gba-recovery-r2`、`gba-recovery-r4`：`common_state.txt`、`summary.csv`、
  事件 scope、LOOP 100 步 recovery、最终模型/轨迹输出。

## 为什么不能补 100 步

- r5 只保存了 step0 渲染指标，没有保存 pre-GBA 的 Gaussian 参数、完整 Adam
  状态、事件时刻相机、训练序列/RNG、line cache；不能从这些 artifact 恢复
  同一事件并继续优化。
- r2/r4 的 GBA `world_deltas` 全零（`PoseAndMap moved=0`），不是非零 GBA；
  并且 GBA 发生在 final refinement 之后，输出模型是 run 末状态，不能反推
  pre-GBA 状态。
- r3/r5/r6 未产生可用非零 GBA 事件。

## 恢复 100 步至少需要的输入

1. 6 组参数张量（xyz/xyz_init、features_dc/rest、opacity、scaling、rotation，
   含 line 字段），pre-GBA 与 payload 后各一份。
2. 完整 Adam 状态：每组的 exp_avg/exp_avg_sq/step，及 `coord_event_id_`、
   `ref_kf_id_`、line cache。
3. 所有事件 KF 的 `Tcw_before`、`Tcw_after`、`world_deltas`、map generation。
4. 当前 iteration、active SH、分辨率/金字塔预算、训练视图序列、RNG 状态。
5. 训练图像路径与相机内参，且与当时 run 的 eval guard 配置一致。

缺少上述任一项时，静默新建优化器或从最终模型继续都不是同事件 100 步恢复，
本次不做这种替代实验。
