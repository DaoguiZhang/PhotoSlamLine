# 官方整系统基线对照协议（`PSL-fullmap-exp-v1` 导出）

**来源**：`/workspace/code/PSL-fullmap-exp-v1`（只读导出，源目录未被修改）。
**导出脚本**：`/workspace/code/PSL-lineband-multi/scripts/export_baseline_protocol.py`（可重跑）。
**机器可读副本**（本目录）：

| 文件 | 内容 |
|---|---|
| `cells.csv` | 30 单元格的完整定义（二进制、ORB 标定、Gaussian 配置、OFF 配置、序列目录、关联文件、帧数、三份配置的 sha256、OFF 差异键） |
| `budget_per_cell.csv` | 逐单元格精修预算与预处理参数（25 个键 × 30 行） |
| `calibration.csv` | 逐单元格标定参数（内参、畸变、图像尺寸、通道序、ORB 特征数） |

**源清单哈希**（用于对齐版本）：

| 文件 | sha256 前 16 位 |
|---|---|
| `PSL-fullmap-exp-v1/manifest.csv` | `abd7dba8e3dc1700` |
| `report/formal_paired_delta.csv` | `bae4da857ca2a4ec` |
| `report/formal_summary_cell.csv` | `4279763312e2b3d0` |
| `report/formal_summary_group.csv` | `0e9311e1a9bf1001` |

---

## 0. 边界声明（必须先读）

本协议是**整系统官方基线对照**（full vs off），与上一级目录的窄带消融
（`../SEAL.md` / `../FALLBACK_CHECK.md` —— 固定几何 A/W0/V1/V2 实验）是**两套不同的表**：

| | 官方基线对照（本文档） | 窄带消融（`SEAL.md`） |
|---|---|---|
| 几何 | 完整 SLAM 前端在线跑（每 repeat 重新跑前端） | 固定几何：一次前端 + 冻结快照重放 |
| 条件 | `full`（三线权重 0.001/0.002/0.005）vs `off`（全 0） | A / W0 / V1 / V2 |
| 单元格 | **30** 个 dataset×modality | 8 个（多场景轮） |
| 重复 | 每单元格 **5** 次 | 每单元格 3 种子 |
| 指标 | 全图 PSNR（`psnr.txt` 均值）+ 轨迹 | 边界带 PSNR / F1 / 线段级 |

**禁止混表**：固定几何消融的数字、以及本轮新引入/复用的场景
（office0、room0、fr1_desk、desk_3 等）都**不得**写入整系统主表。
整系统主表只包含下节 30 个单元格，且只包含 `full` / `off` 两个条件。

---

## 1. 实验矩阵

| 项 | 值 |
|---|---|
| 单元格数 | **30**（Replica 8 mono；TUM 3 mono + 3 rgbd；ETH3D 5 mono + 11 rgbd = 16） |
| 每单元格条件数 | 2（`full`、`off`） |
| 每单元格重复次数 | **5**（`rep = 1..5`） |
| 每单元格单独运行数 | 10 |
| 总运行数 | 30 × 5 × 2 = **300** |
| 随机种子 | `PHOTO_SLAM_SEED = str(rep - 1)` → rep1=0 … rep5=4 |
| 条件顺序 | rep 为奇数：`full, off`；rep 为偶数：`off, full`（交替，抵消顺序效应） |
| 输出目录 | `runs/formal-v1/<cell>/r<rep>_<cond>/`（**从不覆盖**：非空即跳过） |

## 2. 30 个单元格

| cell | family | modality | binary | frames |
|---|---|---|---|---|
| replica_office0_mono | Replica | mono | `replica_mono` | 2000 |
| replica_office1_mono | Replica | mono | `replica_mono` | 2000 |
| replica_office2_mono | Replica | mono | `replica_mono` | 2000 |
| replica_office3_mono | Replica | mono | `replica_mono` | 2000 |
| replica_office4_mono | Replica | mono | `replica_mono` | 2000 |
| replica_room0_mono | Replica | mono | `replica_mono` | 2000 |
| replica_room1_mono | Replica | mono | `replica_mono` | 2000 |
| replica_room2_mono | Replica | mono | `replica_mono` | 2000 |
| tum_freiburg1_desk_rgbd | TUM | rgbd | `tum_rgbd` | 序列文件全量 |
| tum_freiburg2_xyz_rgbd | TUM | rgbd | `tum_rgbd` | 序列文件全量 |
| tum_freiburg3_long_office_household_rgbd | TUM | rgbd | `tum_rgbd` | 序列文件全量 |
| tum_freiburg1_desk_mono | TUM | mono | `tum_mono` | 序列文件全量 |
| tum_freiburg2_xyz_mono | TUM | mono | `tum_mono` | 序列文件全量 |
| tum_freiburg3_long_office_household_mono | TUM | mono | `tum_mono` | 序列文件全量 |
| eth3d_desk3_rgbd | ETH3D | rgbd | `eth3d_rgbd` | 序列文件全量 |
| eth3d_mannequin1_rgbd | ETH3D | rgbd | `eth3d_rgbd` | 序列文件全量 |
| eth3d_mannequin3_rgbd | ETH3D | rgbd | `eth3d_rgbd` | 序列文件全量 |
| eth3d_planar2_rgbd | ETH3D | rgbd | `eth3d_rgbd` | 序列文件全量 |
| eth3d_planar3_rgbd | ETH3D | rgbd | `eth3d_rgbd` | 序列文件全量 |
| eth3d_plantscene1_rgbd | ETH3D | rgbd | `eth3d_rgbd` | 序列文件全量 |
| eth3d_sfmbench_rgbd | ETH3D | rgbd | `eth3d_rgbd` | 序列文件全量 |
| eth3d_sfmgarden_rgbd | ETH3D | rgbd | `eth3d_rgbd` | 序列文件全量 |
| eth3d_sfmhouseloop_rgbd | ETH3D | rgbd | `eth3d_rgbd` | 序列文件全量 |
| eth3d_sfmlabroom1_rgbd | ETH3D | rgbd | `eth3d_rgbd` | 序列文件全量 |
| eth3d_table7_rgbd | ETH3D | rgbd | `eth3d_rgbd` | 序列文件全量 |
| eth3d_desk3_mono | ETH3D | mono | `eth3d_mono` | 序列文件全量 |
| eth3d_planar2_mono | ETH3D | mono | `eth3d_mono` | 序列文件全量 |
| eth3d_planar3_mono | ETH3D | mono | `eth3d_mono` | 序列文件全量 |
| eth3d_sfmgarden_mono | ETH3D | mono | `eth3d_mono` | 序列文件全量 |
| eth3d_table7_mono | ETH3D | mono | `eth3d_mono` | 序列文件全量 |

逐单元格的 `orb_cfg`、`gauss_cfg`、`off_cfg`、`seq_dir`、`assoc` 绝对路径与
sha256 见 `cells.csv`。

### 2.1 输入帧列表 / 序列定义

| family | 帧列表来源 | 关联文件 |
|---|---|---|
| Replica | `<seq_dir>/results/frame%04d.jpg`，manifest 指定 `frames = 2000` | 无（RGB-D 由 `results/depth%04d.png` 直接配对） |
| TUM mono | `<seq_dir>/rgb.txt` | 无 |
| TUM rgbd | `<seq_dir>/rgb.txt` | `PSL-fullmap-lock-v1/cfg/ORB_SLAM3/RGB-D/TUM/associations/<seq>.txt` |
| ETH3D mono | `<seq_dir>/rgb.txt` | 无 |
| ETH3D rgbd | `<seq_dir>/rgb.txt` | `<seq_dir>/associated.txt` |

正式运行**不使用**抽帧子集。`PSL-fullmap-exp-v1/inputs/` 下的 200 帧清单
（`eth3d_desk3_rgbd_assoc_200.txt`、`replica_office0_mono_200/`、`tum_fr3_mono_200/`、
`tum_fr3_rgbd_assoc_200.txt`）只用于 `scripts/small_sample.sh` 的冒烟自检，
**不得**作为正式结果。

---

## 3. 命令行、环境与运行顺序

```bash
# 单次运行的实际命令行（由 scripts/formal_batch.py 生成，也写入 run dir 的 cmd.txt）
bin/<binary> <ORBvoc.txt> <orb_cfg> <gauss_cfg 或 off_cfg> <seq_dir> [<assoc>] <run_dir>/ no_viewer
```

固定环境变量：

| 变量 | 值 |
|---|---|
| `LD_LIBRARY_PATH` | `<SRC>/ORB-SLAM3/lib:<SRC>/ORB-SLAM3/Thirdparty/g2o/lib:<SRC>/ORB-SLAM3/Thirdparty/DBoW2/lib:<SRC>/lib` |
| `PHOTO_SLAM_LINE_MODE` | `2` |
| `PHOTO_SLAM_LINE_LOOP` | `0` |
| `PHOTO_SLAM_SHADOW_LINE` | `0` |
| `PHOTO_SLAM_SLAM_ONLY` | `0` |
| `PHOTO_SLAM_NO_PACING` | `0` |
| `PHOTO_SLAM_SEED` | `str(rep - 1)` |
| 所有 `PHOTO_SLAM_DEBUG_*` | **显式 unset**（`DEBUG_LINE_LOSS`、`DEBUG_LINE_EDGES`、`DEBUG_LINE_GATE`、`DEBUG_LINE_RESCUE`、`DEBUG_LINE_SCAN`、`DEBUG_MERGE`、`DEBUG_MONO_INIT`、`DEBUG_LINE_LOOP`、`DEBUG_POSTFIX`） |

运行顺序（外层 repeat，内层 cell）：

```
for rep in 1..5:
    order = ["full","off"] if rep % 2 == 1 else ["off","full"]
    for cell in manifest:            # 30 个
        for cond in order:
            运行 → 写 status.json → 追加一行 summary.csv
```

### 3.1 完整性与防污染守卫

| 守卫 | 行为 |
|---|---|
| `bin/<binary>` 与两个 `.so` 的 sha256 在**每次运行前**重算比对 | 任何变化 → `ABORT` 并返回码 3，整批停止 |
| 已存在 `status.json` 且 `accept == true` | `SKIP completed` |
| 目标目录存在且非空但未完成 | `SKIP non-empty incomplete`（不覆盖、不续跑） |
| 每次运行把 `orb_settings.used.yaml`、`gaussian_settings.used.yaml` 复制进运行目录 | 事后可核对实际生效配置 |
| 每次运行写 `run.meta.json`（含 argv、全部哈希、环境、条件顺序） | 可独立审计 |

---

## 4. 标定

标定**逐单元格**来自该单元格的 ORB-SLAM3 YAML（顺序 `cells.csv` 的 `orb_cfg`），
运行时会作为 `orb_settings.used.yaml` 复制进运行目录。参数表见 `calibration.csv`，键为：

```
Camera.type, Camera.RGB, Camera.width, Camera.height,
Camera1.fx, Camera1.fy, Camera1.cx, Camera1.cy,
Camera1.k1, Camera1.k2, Camera1.p1, Camera1.p2,
ORBextractor.nFeatures
```

例（`eth3d_planar2_rgbd`）：`Camera.type: PinHole`，`Camera1.fx = fy = 726.28741455078`，
`cx = 354.6496887207`，`cy = 186.46566772461`，`k1 = k2 = p1 = p2 = 0`，
`Camera.RGB: 1`，`Camera.width = 739`，`Camera.height = 458`，`ORBextractor.nFeatures = 2000`。

`Camera.RGB` 决定读图时的通道序处理（逐单元格取值见 `calibration.csv`，不得自行改动）。
畸变由 `initUndistortRectifyMapAndMask` 用上表 `k1/k2/p1/p2` 生成去畸变映射与有效掩码，
在送入 SfM/Gaussian 前统一应用；**不修改**原始图像文件。

---

## 5. 预处理

| 项 | 值（30 个单元格全部一致，除非注明） |
|---|---|
| `Model.resolution` | `-1.0`（原生分辨率，不缩放） |
| `Model.sh_degree` | `3` |
| `Model.white_background` | `0` |
| `Model.eval` | `0` |
| `Camera.z_near` / `Camera.z_far` | `0.01` / `100.0` |
| 去畸变 | 见 §4，逐帧应用映射 + 有效掩码 |
| 高斯金字塔 | `GausPyramid.do: 1`；`num_sub_levels` 与 `sub_level_times_of_use` **逐单元格不同**（见 §6.2） |
| 关键帧策略 | `Mapper.new_keyframe_times_of_use` 逐单元格不同（2 / 8 / 20）；`Mapper.min_num_initial_map_kfs` 逐单元格不同（10 / 20 / 28）；`Mapper.cull_keyframes: 0` |

---

## 6. 精修预算（refinement budget）

### 6.1 30 个单元格全部相同的部分

| 键 | 值 |
|---|---|
| `Optimization.position_lr_init` | `0.00032` |
| `Optimization.position_lr_delay_mult` | `0.01` |
| `Optimization.position_lr_max_steps` | `24` |
| `Optimization.feature_lr` | `0.0025` |
| `Optimization.opacity_lr` | `0.05` |
| `Optimization.scaling_lr` | `0.005` |
| `Optimization.rotation_lr` | `0.001` |
| `Optimization.lambda_dssim` | `0.2` |
| `Optimization.densify_grad_threshold` | `0.001` |
| `Model.resolution` | `-1.0` |
| `Model.sh_degree` | `3` |
| `Model.white_background` | `0` |
| `Camera.z_near` / `Camera.z_far` | `0.01` / `100.0` |
| `Mapper.cull_keyframes` | `0` |

### 6.2 逐单元格不同的键（必须从该单元格的 `gauss_cfg` 读取，不可全局统一）

| 键 | 取值 → 单元格数 | 分布 |
|---|---|---|
| `Optimization.max_num_iterations` | `30100` → 27；`301000` → 3 | 只有 TUM rgbd 三个单元格是 301000 |
| `Optimization.position_lr_final` | `0.00032` → 14；`0.00016` → 16 | ETH3D 15 个为 0.00016，但 `eth3d_sfmgarden_mono` 例外为 0.00032；Replica/TUM 全部 0.00032 |
| `Optimization.densify_from_iter` | `600` → 8；`800` → 19；`1000` → 3 | Replica 600；TUM rgbd 1000；其余 800 |
| `Optimization.densify_until_iter` | `15000` → 10；`30000` → 20 | Replica 8 个 + `eth3d_sfmbench_rgbd`、`eth3d_sfmhouseloop_rgbd` |
| `Optimization.densify_min_opacity` | `0.02` → 8；`0.1` → 22 | 只有 Replica 是 0.02 |
| `Optimization.prune_big_point_after_iter` | `15000` → 2；`30000` → 28 | 只有 `eth3d_sfmbench_rgbd`、`eth3d_sfmhouseloop_rgbd` |
| `Mapper.new_keyframe_times_of_use` | `8` → 19；`20` → 3；`2` → 8 | Replica+ETH3D rgbd 8；TUM rgbd 20；TUM mono 与 ETH3D mono 2 |
| `Mapper.min_num_initial_map_kfs` | `20` → 22；`10` → 3；`28` → 5 | TUM rgbd 10；ETH3D mono 28；其余 20 |
| `GausPyramid.num_sub_levels` | `2` → 15；`3` → 15 | Replica 8 + TUM 6 + `eth3d_sfmgarden_mono` 为 2；其余 15 个 ETH3D 为 3 |
| `GausPyramid.sub_level_times_of_use` | `8` → 27；`15` → 3 | 只有 TUM rgbd 是 15 |

**注意**：`eth3d_sfmgarden_mono` 在 `position_lr_final`（0.00032）与 `num_sub_levels`（2）
上与其余 4 个 ETH3D mono 单元格不同，这是源配置的既有差异，本协议**照原样保留**，
不做统一化——若统一化，整系统主表就与已跑出的 `runs/formal-v1/` 结果不再对应。

精确到单元格的数值见 `budget_per_cell.csv`（25 列 × 30 行）；
每个运行目录里的 `gaussian_settings.used.yaml` 是实际生效值的最终依据。

---

## 7. `full` 与 `off` 的唯一差异

`full` 用 manifest 的 `gauss_cfg`；`off` 用
`PSL-fullmap-exp-v1/configs/off/formal/<cell>.off.yaml`（由 `scripts/make_off_cfg.py`
从对应 `gauss_cfg` 机械生成）。对 30 个单元格逐一做 `diff` 的校验结果：

| 检查项 | 结果 |
|---|---|
| 变化的行数 | 全部 30 个单元格均为 **6** 行（3 删 + 3 增） |
| 变化的键 | 全部 30 个单元格均只有这 3 个键：`Optimization.weight_line_coherence`、`Optimization.weight_line_shape_ecc`、`Optimization.weight_line_shape_ori` |
| 变化内容 | `0.001 / 0.002 / 0.005` → `0.0 / 0.0 / 0.0` |
| 其余所有键 | 逐字节相同 |

即：`off` 条件**只**关掉三项线一致性/形状损失，其余（预算、学习率、稠密化、金字塔、
关键帧策略、预处理、标定）与 `full` 完全一致。校验可用
`python3 scripts/export_baseline_protocol.py` 重跑（输出 `off-config diffs unexpected: none`）。

---

## 8. 必须产出的评测产物

### 8.1 每个运行目录 `runs/formal-v1/<cell>/r<rep>_<cond>/`

| 产物 | 说明 |
|---|---|
| `<iter>_shutdown/psnr.txt` | 逐评估帧 PSNR；`mean_psnr` 由该文件所有行取均值得到（`psnr_entries` = 行数） |
| `<iter>_shutdown/dssim.txt` | 逐帧 DSSIM |
| `<iter>_shutdown/psnr_gaussian_splatting.txt` | 对照用 GS 口径 PSNR |
| `<iter>_shutdown/render_time.txt`、`image/`、`ply/` | 渲染耗时、渲染图、点云 |
| `point_cloud.ply`（在运行目录下递归查找） | 最终点云；校验 `element vertex` 数（`ply_vertices`）与字节数（`ply_size`） |
| `CameraTrajectory_TUM.txt`、`CameraTrajectory_EuRoC.txt`、`CameraTrajectory_KITTI.txt` | 相机轨迹（三种格式） |
| `KeyFrameTrajectory_TUM.txt` | 关键帧轨迹 |
| `TotalOperationTime.txt`、`TrackingTime.txt`、`total_mapping_time.txt`、`GpuPeakUsageMB.txt` | 时间与显存 |
| `cmd.txt` | 实际命令行 |
| `orb_settings.used.yaml`、`gaussian_settings.used.yaml` | 实际生效配置（从源配置按字节复制） |
| `run.meta.json` | argv、各文件 sha256、环境变量、条件顺序、rep/seed |
| `status.json` | 上述 + `exit`、`wall_s`、`total_iterations`、`psnr_entries`、`mean_psnr`、`accept`、`model_empty`、`line_loss_diag_lines` |
| `stdout.log` | 完整日志 |
| `used_times` | 每阶段耗时记录 |

### 8.2 批次级

| 产物 | 字段 |
|---|---|
| `runs/formal-v1/summary.csv` | `cell, family, dataset, modality, rep, seed, condition, cond_order, exit, wall_s, total_iterations, psnr_entries, mean_psnr, ply, ply_size, ply_vertices, line_loss_diag_lines, model_empty, accept, binary_sha256, libORB_SLAM3_sha256, libgaussian_mapper_sha256, orb_cfg_sha256, gauss_cfg_sha256` |

### 8.3 报告级（`report/`）

| 文件 | 内容 |
|---|---|
| `formal_paired_delta.csv` | 逐单元格配对差值：`family, dataset, modality, cell, n_pairs, d_psnr_mean, d_psnr_std`（off − full，按 rep 配对） |
| `formal_summary_cell.csv` | 逐单元格 full/off 均值与标准差、有效重复数 |
| `formal_summary_group.csv` | family × modality 汇总：`family, modality, full_psnr_mean, full_psnr_std, full_repeats_used, full_cells, off_psnr_mean, off_psnr_std, off_repeats_used, off_cells` |

当前已有示例（同 `PSL-fullmap-exp-v1/report/`）：
`formal_summary_group.csv` 中 ETH3D mono 为 full 22.894 ± 0.304（5 次 × 5 单元格）、
off 23.190 ± 0.153（5 × 5）；ETH3D rgbd 为 full 26.685 ± 0.072（5 × 11）、
off 26.749 ± 0.126（5 × 11）。

### 8.4 单次运行的接受判据

```python
accept = (returncode == 0
          and total_iterations > 0
          and psnr_entries > 0
          and (ply_vertices or 0) > 0)
```

`accept = false` 的运行**保留在目录里**（不删除、不覆盖），但在汇总时必须显式标出
`exit`、`psnr_entries`、`ply_vertices`，不得静默丢弃。

---

## 9. 复现整批的命令

```bash
cd /workspace/code/PSL-fullmap-exp-v1
python3 scripts/formal_batch.py          # 30 cell × 5 rep × 2 cond = 300 次运行
python3 scripts/formal_report.py         # 生成 report/formal_*.csv
```

小样本冒烟（不是正式结果）：

```bash
bash scripts/small_sample.sh             # 用 inputs/ 下 200 帧清单跑少量 cell
```

本导出本身的复现：

```bash
cd /workspace/code/PSL-lineband-multi
python3 scripts/export_baseline_protocol.py   # 重写 baseline_protocol/*.csv 并打印不变量检查
```
