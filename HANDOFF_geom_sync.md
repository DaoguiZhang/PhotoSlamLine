# HANDOFF — geom-sync（参考关键帧锚定的 LOOP/GBA 地图更正）

状态：**TUM 崩溃已定位并修复；GlobalBA 事件链路首次取得运行证据。**
HEAD：本次提交（见 `git log -1`）。工作树干净。

## 本轮结论：TUM fr3_long_office_household KF 2 `free(): invalid pointer`

- 根因：**跨 DSO 的 ODR / 类型布局不一致**，不是 GlobalBA 业务逻辑。
  `b2e997a5d9` 给 `MappingOperation` 增加了 `mvGlobalBAPoses/mnEventId/mnMapGen`；
  `libgaussian_mapper.so` 已按新头文件重建，但 `ORB-SLAM3/lib/libORB_SLAM3.so`
  及其 `.o` 仍是旧头文件产物。`LocalMapping.cc` 按旧布局构造对象，内联析构却
  绑定到先加载的 `libgaussian_mapper.so` 中的新析构，越界把 `mvGlobalBAPoses`
  的垃圾指针当 vector 释放。
- 证据：`reports/tum_crash_rootcause.md`、`reports/gdb_crash.log`
  （`~MappingOperation` ← `LocalMapping::RunWithLine`）；析构符号大小
  新 `0x24d` vs 旧 `0x22d`。
- 修复：按原缓存配置重建 `ORB-SLAM3/build`，再重建顶层 `build`。
  **无源码修改**；重建后两个 DSO 的析构均为 `0x24d`。

## 验证（正常 Release 构建）

| 场景 | 结果 |
|---|---|
| TUM 全序列 ×3 | exit 0；崩溃关键字 0；LBA 次数 254/250/241 |
| office0（Replica RGB-D，GPU 路径） | exit 0；正常导出全局点线地图 |
| TUM LOOP/GBA | 1 次回环；`GBA: Correct MapPoints and MapLines`；`[SyncEvent] GlobalBA evt=1 map=0 kfs=234 in_scene=234 prev_evt=0 ACCEPTED accepted=1 duplicate=0` |

原崩溃窗口在三次全序列运行中均未复现。

## 构建注意（本次事故根因）

`ORB-SLAM3` 是独立 CMake 工程，顶层只链接其 `.so`。修改
`ORB-SLAM3/include/*.h` 后必须先 `cmake --build ORB-SLAM3/build -j8`，
再 `cmake --build build -j8`；否则内联符号会按不同布局进入不同 DSO。

## 未实现（不变）

- `ref_kf_id_` 与形变应用未实现；`ACCEPTED` 仅表示通知被接受，不代表 Gaussian 已更正。
- 整模型检查、Adam 策略、高阶 SH 未做。

## 日志

- `reports/tum_crash_rootcause.md` — 根因与符号/时间戳证据
- `reports/gdb_crash.log` — GDB 全线程栈
- `/tmp/fix_tum_{1,2,3}.log`、`/tmp/fix_office0.log` — 验证日志（临时）

## 下一条具体命令

```bash
cd /workspace/code/PSL-perf
PHOTO_SLAM_RASTER_BACKEND=C PHOTO_SLAM_LINE_LOSS_EVERY=0 \
  bin/tum_rgbd ORB-SLAM3/Vocabulary/ORBvoc.txt \
  cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
  cfg/gaussian_mapper/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
  /workspace/code/SEGS-SLAM/datasets/tum/rgbd_dataset_freiburg3_long_office_household \
  cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg3_long_office_household.txt \
  /tmp/fix_tum_next no_viewer 2>&1 | tee /tmp/fix_tum_next.log
echo "EXIT=${PIPESTATUS[0]}"
grep -E "Loop Closure Detected|GBA:|SyncEvent" /tmp/fix_tum_next.log
```

