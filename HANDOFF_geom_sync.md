# HANDOFF — geom-sync（参考关键帧锚定的 LOOP/GBA 地图更正）

状态：**RGB-D 下 GlobalBA 已对已有 Gaussian 实际应用并继续训练。**
HEAD：本次提交。工作树干净（构建产物一并提交）。

## 本轮实现（geom-sync-v3）

- `Atlas.h`：GlobalBA 事件载荷改为 `kf_id + Tcw_before + Tcw_after`；
  新增 MapPoint/线采样点的 `ref_kf_id` 并行载荷（-1 = 未知）。
- `LoopClosing.cc`：两处 GBA 写回点发布 before/after 快照
  （`mTcwBefGBA`，未被更正的关键帧 before = after）。
- `GaussianModelLine`：新增 `ref_kf_id_`、`coord_event_id_`；
  覆盖初始点云、普通点追加、inactive 补点、线采样、clone/split、prune、
  `keepSubset`、PLY/checkpoint/full-snapshot 路径。
- 更正：`applyGbaCorrections(G = Tcw_after^-1 * Tcw_before)` 按 ref kf 分组，
  更新中心、四元数、线锚点 `xyz_init_`、线方向 `line_dir_w_`、行缓存；
  SE(3) 不改 scaling。Adam：xyz 一阶矩随 R 旋转，二阶矩保留（近似），
  其余参数组不变。`coord_event_id_` 阻止旧事件重复作用于已更正坐标的新点。
- 回调不创建点：`TORCH_CHECK(count_after == count_before)`；
  `created_during_event=0` 实测。LocalMapping 新点/训练增密仍走原路径并单独计数。
- `scripts/build_two_level.sh`：固定两级构建并校验两 DSO 的
  `MappingOperation` 析构布局一致。

## 验证

| 场景 | 结果 |
|---|---|
| 契约测试（CPU+GPU） | centre/dir/anchor/residual/scale/covariance/cache/counts/duplicate/order 全 PASS |
| TUM fr3_long_office_household 全序列 | exit 0；evt=1 APPLIED；moved=77170（line 134）；stale=0；unassigned=30；missing_event=116；count 77316→77316；sync 79.9 ms；事件后继续 KF 230+ 并正常导出 |
| office0（Replica RGB-D，GPU） | exit 0；无 RefKf/Torch 错误；正常导出 |

事件日志见 `reports/gba_apply_evidence.md`、`reports/gba_tum_final.log`；
契约见 `reports/gba_contract_final.log`。

## 明确限制

- 仅 RGB-D SE(3)；单目 Sim(3)、局部 BA 新策略、高阶 SH 旋转未扩展。
- 单锚刚性近似：同一 ref kf 的所有 Gaussian 施加同一 delta，不宣称精确复现
  GBA 对每个三维点的独立调整。
- 无锚（-1）或锚 kf 不在事件内的行不移动，按 `unassigned/missing_event` 记录。

## 下一条具体命令

```bash
cd /workspace/code/PSL-perf
./scripts/build_two_level.sh
PHOTO_SLAM_RASTER_BACKEND=C PHOTO_SLAM_LINE_LOSS_EVERY=0 \
  bin/tum_rgbd ORB-SLAM3/Vocabulary/ORBvoc.txt \
  cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
  cfg/gaussian_mapper/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
  /workspace/code/SEGS-SLAM/datasets/tum/rgbd_dataset_freiburg3_long_office_household \
  cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg3_long_office_household.txt \
  /tmp/gba_next no_viewer 2>&1 | tee /tmp/gba_next.log
echo "EXIT=${PIPESTATUS[0]}"
grep -E "GBA: Correct|GBA-Apply|Loop Closure Detected" /tmp/gba_next.log
```

