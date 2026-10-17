# HANDOFF — geom-sync（参考关键帧锚定的 LOOP/GBA 地图更正）

状态：**执行中断（上下文预算用尽）**，不是缺数据或技术依赖阻塞。
HEAD：`a066e5d74d` + 本次提交。工作树干净。

## 已实现且编译通过

1. **变换契约**（`f02c52920d`）
   - 修复 legacy 中 `line_dir_w_` 的逆旋转（数值证据：90° 旋转下误差 2.00）。
   - 新增 `xyz_init_` 线锚点同步（此前完全不移动）。
   - 自验证 `runRigidCorrectionContractTest`，`PHOTO_SLAM_SYNC_CONTRACT_TEST=1` 触发。
   - **GPU 中心变换已实测**：identity / translation / Rz90+trans 与 CPU 参考 `max_abs=0`（容差运行前固定 1e-4）。

2. **轻量 GBA 事件通道**（本次提交）
   - `Atlas.h`：`OprType::GlobalBA = 4`；`mvGlobalBAPoses`（仅 `(kf_id, new_Tcw)`，无图像）、`mnEventId`、`mnMapGen`、`reserveGlobalBAPoses/addGlobalBAPose/globalBAPoses/setEventId/eventId/setMapGen/mapGen`；拷贝构造同步搬运。
   - `LoopClosing.cc`：在 GBA 写回完成、`InformNewBigChange()` 之前发布，**采集的是写回后的最终位姿**（含跨生成树传播得到更正的关键帧），`mapGen = Map::GetId()`，`eventId` 单调递增。两个 GBA 线程（普通版与 WithLine 版）都已插入。
   - `gaussian_mapper_line.cpp`：新增 `case GlobalBA`，**只消费并记录，不施加形变**；打印 `evt / map / kfs / in_scene / prev_evt / ACCEPTED|DUPLICATE_REJECTED / accepted / duplicate`。
   - 设计上把"通知通路"与"形变"解耦：形变需要 `ref_kf_id_`，尚未实现，因此这一层明确不施加。

## 已验证

- 全部编译通过（`cd build && make -j8`，无 error）。
- 契约自验证 PASS（CPU 四项 + GPU 中心三项）。
- 默认行为未改变：新事件只打印；legacy 形变路径仍是已修复版本。

## 未验证 / 未实现

- **真实 GBA 事件从未被观测到**。首次尝试用 `bin/replica_rgbd` + TUM 数据集，因数据集布局不符崩溃：
  `filesystem error: cannot open directory [...rgbd_dataset_freiburg3_long_office_household/results]`，exit 134。
  该入口要求 Replica 风格的 `results/` 与 `depth/`；TUM 是 `rgb/` + `depth/` + 关联文件，应用 **`bin/tum_rgbd`** 与 `cfg/ORB_SLAM3/RGB-D/TUM/associations/` 下的关联文件。
- 因此"发布 → 消费 → 版本检查"**尚未取得运行证据**，不得宣称通过。
- `ref_kf_id_` 未实现（里程碑 2），故按锚分组的形变应用未实现（里程碑 5）。
- 整模型检查（协方差 Σ' = R Σ Rᵀ、尺度不变、未选中行不变、线残差、缓存读新值）未做。
- Adam 策略未改；高阶 SH 旋转未做。

## 下一条具体命令

先确认 `bin/tum_rgbd` 的 CLI（`examples/tum_rgbd.cpp`），然后：

```bash
cd /workspace/code/PSL-perf
PHOTO_SLAM_RASTER_BACKEND=C PHOTO_SLAM_LINE_LOSS_EVERY=0 \
  bin/tum_rgbd ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
    cfg/gaussian_mapper/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
    /workspace/code/SEGS-SLAM/datasets/tum/rgbd_dataset_freiburg3_long_office_household \
    abl_out/geom_gba_tum no_viewer
grep -E "Loop Closure Detected|Starting Global Bundle Adjustment|SyncEvent" \
  abl_out/geom_gba_tum/run.log
```

若 `tum_rgbd` 需要关联文件，按 `examples/tum_rgbd.cpp` 的实际参数顺序补齐
`cfg/ORB_SLAM3/RGB-D/TUM/associations/fr3_long_office_household.txt`（如不存在则生成）。

通过判据：日志中出现 `Starting Global Bundle Adjustment` 且随后出现至少一条
`[SyncEvent] GlobalBA evt=N ... ACCEPTED`，再出现同一 evt 时为
`DUPLICATE_REJECTED`；`in_scene` 应接近 `kfs`。

---

## 真实 TUM 运行结果（正确入口，已执行）

- 工作树：`b2e997a5d9`，dirty=0；二进制 `bin/tum_rgbd` sha256 `d321f0a1a98752e8…`
- CLI 已从 `examples/tum_rgbd.cpp` 核实（argc 7 或 8）：
  `argv[1]=ORBvocab, [2]=ORB cfg, [3]=gaussian cfg, [4]=dataset, [5]=associations, [6]=out, [7]=no_viewer`
- 命令（已执行）：
```bash
PHOTO_SLAM_RASTER_BACKEND=C PHOTO_SLAM_LINE_LOSS_EVERY=0 \
  bin/tum_rgbd ORB-SLAM3/Vocabulary/ORBvoc.txt \
  cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
  cfg/gaussian_mapper/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
  /workspace/code/SEGS-SLAM/datasets/tum/rgbd_dataset_freiburg3_long_office_household \
  cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg3_long_office_household.txt \
  abl_out/geom_gba_tum no_viewer
```
- PID 1630967，**真实退出码 134（SIGABRT）**，不是 tee 的成功。
- 崩溃点：第 97 行 `free(): invalid pointer`，紧跟在 KF 2 的
  `LocalMapping:: RunWithLine: end LocalBundleAdjustmentWithLine_Optimization_Plucker_Reg`
  之后；崩溃前最后一条 loop 相关日志是 `LoopClosing bFindedRegion false`。
- 计数（`abl_out/geom_gba_tum/terminal.log`）：
  `Starting Global Bundle Adjustment` = **0**
  `SyncEvent` = **0**
  `Loop Closure Detected` = **0**

**结论：GBA 从未开始，事件链路未被覆盖。** 本轮的 "GBA写回 → 发布 → 消费 → ACCEPTED"
顺序**没有任何运行证据**，不做通过声明。ACCEPTED 语义本来也只是"通知被接受"，
不代表 Gaussian 已更正。

**崩溃归因未确定**：崩溃发生在第一个 loop/GBA 之前，而本轮改动中唯一处于该热路径的是
`MappingOperation` 拷贝构造新增的成员搬运与 mapper 的 `case GlobalBA`（此时不会执行）。
因此更可能是既有 TUM 线路的堆错误，但**未做二分验证，不能断言**。

## 下一条具体命令（二分归因）

```bash
cd /workspace/code/PSL-perf
git worktree add /workspace/code/PSL-bisect b2e997a5d9~1
cd /workspace/code/PSL-bisect && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8 tum_rgbd
cd /workspace/code/PSL-bisect && PHOTO_SLAM_RASTER_BACKEND=C PHOTO_SLAM_LINE_LOSS_EVERY=0 \
  bin/tum_rgbd ORB-SLAM3/Vocabulary/ORBvoc.txt \
  cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
  cfg/gaussian_mapper/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
  /workspace/code/SEGS-SLAM/datasets/tum/rgbd_dataset_freiburg3_long_office_household \
  cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg3_long_office_household.txt \
  /tmp/bisect_tum no_viewer; echo "BISECT_EXIT=$?"
```
若同样 134 崩溃 → 与本轮改动无关，先修既有崩溃（或改用 office0 等已知可跑序列产生回环），
再回来验证事件链路。若正常 → 本轮改动引入，回退 `Atlas.h` 拷贝构造成员后重试。
