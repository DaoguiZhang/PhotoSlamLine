# 恢复手册（RESTORE）

适用范围：`PhotoSlamLine` 仓库的归档分支 `source/space-audit-20261108`（标签 `archive-20261108`），
以及本机保留的冻结二进制与数据集。

---

## 0. 先分清三种东西

| 类别 | 在哪里 | 能否从 Git 恢复 |
|---|---|---|
| 源码 / 配置 / 脚本 / 报告 / 小型 CSV | 归档分支 `snapshots/…` | **能**，逐文件 sha256 可核 |
| 冻结二进制（`libgaussian_mapper.so` 等） | 原机 `PSL-final/{lib,bin}`、`PSL-review/bin_*` | **不能**。Git 里只有哈希。需要原文件或重建 |
| 数据集 / PLY / PNG / checkpoint / 运行产物 | 原机磁盘 | **不能**。Git 中没有 |

> 重新编译**不保证**得到相同的 `.so` 哈希（编译器/驱动/CUDA 版本都会影响）。
> 若某结论依赖"就是这个二进制"，必须保留原文件（见 §3）。

---

## 1. 从零恢复源码

```bash
# 1) 取归档分支（浅取即可，内容是单个提交）
git clone --depth 1 --branch source/space-audit-20261108 \
    git@github.com:DaoguiZhang/PhotoSlamLine.git /tmp/psl-restore
cd /tmp/psl-restore

# 2) 核对文件完整性（每个快照一份清单）
for f in snapshots/*.sha256; do
  d=$(basename "$f" .sha256)
  ( cd "snapshots/$d" && sha256sum -c "../$d.sha256" --quiet ) && echo "OK   $d" || echo "FAIL $d"
done
```

期望：12 个快照目录全部 `OK`。

## 2. 恢复某个具体版本

| 想要 | 用哪个目录 | 额外步骤 |
|---|---|---|
| 冻结的 L_core（结构线损失关闭，`538f4629…`） | `snapshots/L_core_final_20261108/` | 放回 `/workspace/code/PSL-final/`，然后按 `BUILD_NOTES` 重建（见 §4） |
| 产生 30 次重放结果的版本 | `snapshots/L_replay_20261108/` | 同上，路径 `/workspace/code/PSL-replay/` |
| 正式基线 O_core | `snapshots/O_core_20261108/` | 路径 `/workspace/code/PSL-review/O_core/` |
| 官方原样 O_raw | `snapshots/O_raw_20261108/` | 路径 `/workspace/code/PSL-review/O_raw/` |
| 30 单元格协议（含 30 行 cfg 哈希） | `snapshots/protocol_baseline_csv_20261108/` | 直接读 CSV，不需要重建 |
| 封存协议配置与报告 | `snapshots/protocol_v1_20261108/` | `runs/` 未归档（16.7 GB），需要时有原目录 |
| 本轮全部证据（门禁/账本/审计/退出回归） | `snapshots/evidence_evtrepl_20261108/` | 见其中 `README`/`REPORT_FINAL.md` |
| 主工作仓库当时的未提交状态 | `snapshots/Photo_SLAM_L_worktree_20261108*/` | `worktree.diff` 是完整 `git diff`，可 `git apply` |

恢复主工作仓库的未提交修改：

```bash
git clone git@github.com:DaoguiZhang/PhotoSlamLine.git /tmp/psl-main
cd /tmp/psl-main
git checkout 0e2f7fc40010            # 或 archive/line-rescue-tracking-20261108
git apply --3way /tmp/psl-restore/snapshots/Photo_SLAM_L_worktree_20261108_GITSTATE/worktree.diff
# 再按 untracked_list_head40.txt / status_porcelain.txt 补齐未跟踪文件（在快照树里找同名文件）
```

---

## 3. 冻结二进制（不在 Git 中）

| 二进制 | 原路径 | sha256（前 32） | 说明 |
|---|---|---|---|
| `libgaussian_mapper.so` | `/workspace/code/PSL-final/lib/` | `538f46296c8e33721ade70e042ae2d5f` | L_core 正式库 |
| `libORB_SLAM3.so` | `/workspace/code/PSL-final/ORB-SLAM3/lib/` | `5a14981faac97ed9db7cf7d834168db2` | 线前端/后端库 |
| `libgaussian_mapper.so` | `/workspace/code/PSL-replay/lib/` | `d82b9ccae0368d72ea0fd52118884f2a` | 30 次重放所用版本 |
| `libgaussian_mapper.so` | `/workspace/code/PSL-review/O_core/lib/` | `4293b9d7349f8bde2436097e5a278f83` | 正式基线 |
| `libgaussian_mapper.so` | `/workspace/code/PSL-review/O_raw/lib/` | `0396c6d327dfdbce11fd5ef3754c0487` | 官方原样 |

校验方式：`sha256sum <文件>`，前 32 位应如上。
**这些文件本轮没有被删除**；若要迁移，请整目录 `cp -a` 并用上表核对。

## 4. 重建源码树（Build）

依赖（实测环境）：Ubuntu 22.04、CUDA 12.x（`arch=120`，RTX 5090）、CMake ≥ 3.22、C++17、
Eigen3、OpenCV 4.10、Pangolin、PyTorch 2.12（libtorch）。

```bash
# 以 L_core 为例
cp -a /path/to/snapshots/L_core_final_20261108 /workspace/code/PSL-final
cd /workspace/code/PSL-final
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release      # 交付里没有 build/，必须重新配置
cmake --build build -j20
sha256sum lib/libgaussian_mapper.so                 # 记录；不承诺等于 538f4629…
```

**不承诺**重建后的 `.so` 哈希与归档时相同；结论复现应以"配置 + 源码 + 运行命令"为准，
二进制哈希只用于识别"是不是当时那一份"。

## 5. 运行命令（可复制的口径）

```bash
W=/workspace/code/PSL-final
export LD_LIBRARY_PATH=$W/ORB-SLAM3/lib:$W/ORB-SLAM3/Thirdparty/g2o/lib:$W/ORB-SLAM3/Thirdparty/DBoW2/lib:$W/lib
export PHOTO_SLAM_LINE_MODE=2 PHOTO_SLAM_LINE_LOOP=0 PHOTO_SLAM_DEDUP_VOXEL=0 PHOTO_SLAM_DEDUP_NOTIFY=0
$W/bin/replica_mono \
  $W/ORB-SLAM3/Vocabulary/ORBvoc.txt \
  $W/cfg/ORB_SLAM3/Monocular/Replica/office0.yaml \
  $W/cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml \
  /path/to/dataset/office0  <outdir>/ no_viewer
```

- L_core 的 Gaussian 配置 = `snapshots/evidence_evtrepl_20261108/cfg_v2/*.nostruct.yaml`（三个结构权重为 0）。
- 重放诊断：`PHOTO_SLAM_REPLAY_DIR/_STEPS/_CONDS`、`PHOTO_SLAM_EVT_AUDIT=1`。
- 已验证过的退出码约定：`0` 正常、`3` Shutdown 超时、`4/5` 重放失败、`6` 仍有未消费事件。

## 6. 数据集

不在 Git 中。原机路径与本轮实测大小见 `SPACE_AUDIT.md` 的"数据集体积"一节；
`snapshots/protocol_baseline_csv_20261108/cells.csv` 里每个单元格都记录了 `seq_dir` 与 `assoc` 的绝对路径，
可作为重建时的对照。
