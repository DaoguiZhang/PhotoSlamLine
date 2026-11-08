# room1 高画质复现与旧配置单因子恢复

日期：2026-10-23  
范围：room1，冻结 A 二进制，同一数据/seed/LINE_MODE=2/LINE_LOOP=0/GBA off/append off/no_viewer/诊断 off。  
未改代码、未改后端、未重建、未覆盖旧结果。

## 0. 结论先行

1. **旧配置恢复了画质，但没有逐位复现旧文件。**
   - 旧目标：173 条指标记录，首帧 `33.7540245056`，全量均值 `34.6338976932`。
   - 旧配置（声明值 canonical 化）：173 条指标记录，首帧 `32.8744659400`，原生 C++ GS 全量均值 `35.6117353052`。
   - 均值比旧目标高 `+0.9778376 dB`，首帧低 `-0.8795586 dB`。因此是“画质恢复”，不是“同一文件复现”。
2. **当前掉画质的主因已用单因子实验定位到 `GausPyramid.num_sub_levels`。**
   - 旧配置 `num_sub_levels=2` → `35.6117353 dB`。
   - 只把这一项恢复成当前值 `3` → `31.5378915 dB`（`-4.0738438 dB`）。
   - 只恢复 `position_lr_final=0.00016` → `35.6734069 dB`（影响很小）。
   - 只恢复 `voxel_size=0.10` → `36.0820786 dB`（没有下降）。
3. **剩余与旧目标的约 `0.98 dB` 差距不能归因于这些配置键。**
   - 旧目标的运行源码/二进制没有保存；旧目标为 `iteration=7611`、末点 `475430`，旧配置在冻结 A 上为 `9133`、`532901`。
   - 目标目录没有 `command.txt`、快照或相机清单，LINE_MODE/LINE_LOOP/GBA/backend/viewer/诊断状态和旧二进制哈希均**未知**。
   - 旧 YAML 中的 `Line.*` / 顶层 `max_anisotropy_threshold` / `absolute_max_ratio` 别名是否曾被旧运行时读取，**没有可证明的旧源码或运行日志**。

## 1. 精确文件身份

### 1.1 旧高画质目标（按内容绑定）

- 结果目录：
  `/workspace/code/Photo-SLAM-L/Temp/0_iterations/replica_rgbd/replica_rgbd_0/room1`
- 指标文件：
  `/workspace/code/Photo-SLAM-L/Temp/0_iterations/replica_rgbd/replica_rgbd_0/room1/7611_shutdown/psnr_gaussian_splatting.txt`
  - SHA256 `d27bef902f811cbf64476d91d3a0a85092697875b5e153f5d2e14050afcb4f2a`
  - 173 条，首行 `0 33.7540245056`
  - 全量均值 `34.6338976932`，median `34.4938659668`，min `29.6085453033`，max `39.0343704224`，std `2.3065226029`
  - mtime `2026-08-05 14:17:53.702844667 UTC`
- 旧 Gaussian 配置：
  `/workspace/code/Photo-SLAM-L/Temp/0_iterations/replica_rgbd/replica_rgbd.yaml`
  - SHA256 `c7d4037a534e0a9d1ecf9bc87ffcfb0f8b12d1097134b109faff98ae22debe4c`
  - mtime `2026-08-05 14:08:11.459024624 UTC`
- 旧批次日志：
  `/workspace/code/Photo-SLAM-L/Temp/0_iterations/replica_rgbd/log.txt`
  - SHA256 `7e6df9af438c2d43cc9017062ea9bb00937088ef4cd9434796475d020653fc29`
  - 该日志的 seed0 room1 `eval.txt` 值为 `33.09262585926056`，与目标目录 `eval.txt` 完全一致，因此目标目录可绑定为该批次 seed0 的 `replica_rgbd_0`。
- 旧目标其它可读事实：
  - `KeyFrameTrajectory_TUM.txt` 只有 156 行；173 是指标记录数，不是轨迹导出 KF 数。
  - `gaussian_training_log.txt` 末行 `7611 ... 475430`。
  - `Global_Final_Map_Lines.obj` 写明 `Total lines: 31`；`Global_Final_Map_Points.obj` 为 `12859` 行。
  - 旧 `eval.txt` 的 `psnr: 33.09262585926056` 不是本报告使用的 `psnr_gaussian_splatting`，不能与目标首行/均值混用。
- 旧运行的二进制、环境变量、LINE_MODE/LINE_LOOP/GBA、backend、viewer、shutdown 诊断：**未知**。
  - 当前 `/workspace/code/Photo-SLAM-L/bin/replica_rgbd` mtime `2026-10-15 12:30`，SHA256 `abf404a07b9403ed716fee94b752cffa4dec10dcb66b533e9d14808954a7e1f1`，晚于旧目标，不能当作旧二进制。

### 1.2 新低画质目标

- 结果目录：
  `/workspace/code/PSL-perf/results/replica_A_baseline_20261023_A8/replica_rgbd_0/room1`
- 指标文件：
  `.../10480_shutdown/psnr_gaussian_splatting.txt`
  - SHA256 `36000dd2831f710ae37ce274ee9badebd1ff1744db20099a7354d5fa280d4773`
  - 182 条，首行 `0 31.0604000092`
  - 在线均值 `31.9849529266`；快照原生复核 `31.9848955805`
- 快照：
  `.../natural_end/snapshot.pt`
  - SHA256 `3417cd440c806defb2fdde18b95cf6584ceba9d3448289fb5bbb9166ef3dd3b8`
- 配置：`cfg/gaussian_mapper/RGB-D/Replica/A_baseline/room1.yaml`
  - SHA256 `62fadbf35156c1bea52b0965b49be236d59d8260882961bc100ac9dc1fc129f7`
- 运行身份见 `.../command.txt`：冻结 A、seed0、LINE_MODE=2、LINE_LOOP=0、GBA apply=0、append=0、no_viewer、诊断 off。

### 1.3 冻结二进制

- `/workspace/code/PSL-perf/evidence/frozen_A_clean_20261023/bin/replica_rgbd`
- SHA256 `5919857e7a89947b2c5256a8d47ccaf7ca6952f65a9f96218754543d9989d6ae`
- 所有复现运行均使用该二进制；未重建、未替换库。

## 2. 旧 YAML 与当前稳定配置逐键差异

旧文件精确副本：
`evidence/repro_room1_oldconfig_20261023/config/room1_old_yaml_exact.yaml`  
与旧文件逐字节相同（`cmp` 返回 0），SHA256 `c7d4037a...`。

| 键 | 旧 YAML | 当前 A8 | 说明 |
|---|---:|---:|---|
| `GausPyramid.num_sub_levels` | 2 | 3 | **主因**，单因子实验见 §4 |
| `Optimization.position_lr_final` | 0.00032 | 0.00016 | 单因子影响很小 |
| `Optimization.voxel_size` | 0.25 | 0.10 | 单因子没有降低画质 |
| `Optimization.max_num_iterations` | 30100 | 50100 | 实际只到 7611–10480，未触顶 |
| `Record.training_report_interval` | 1 | 0 | 仅训练日志采样，不改优化 |
| 顶层 `absolute_max_ratio` | 100.0（别名） | canonical 1500.0 | 旧运行时是否读取：**未知** |
| 顶层 `max_anisotropy_threshold` | 100.0（别名） | canonical 600.0 | 旧运行时是否读取：**未知** |
| 顶层 `lambda_anisotropy` | 0.01（别名） | canonical 0.01 | 数值相同 |
| `Line.SampleStep` | 0.15（别名） | canonical 0.2 | 旧运行时是否读取：**未知** |
| `Line.ViewWeight` | 2.0（别名） | canonical 2.0 | 数值相同 |
| `Line.Sigma` | 3.0（别名） | canonical 3.0 | 数值相同 |
| `Line.TopK` | 3（别名） | canonical 3 | 数值相同 |

旧 YAML 读取器证据：
- `backup/Photo-SLAM-L/src/gaussian_mapper_line.cpp`（mtime 2026-05-18）只读
  `Optimization.line_sample_step`、`Optimization.max_anisotropy_threshold`、
  `Optimization.absolute_max_ratio`；不读 `Line.*` 或顶层别名。
- 当前 PSL-perf 源码同样只读 canonical 键。
- 未找到 2026-08-05 的旧源码或旧二进制，因此“旧运行时实际值”不能闭合证明；只能给出两个整配置实验：
  - `room1_oldconfig_effective.yaml`：把旧声明别名转成 canonical，argv 使其生效。
  - `room1_oldruntime_canonical.yaml`：旧 canonical 键生效，旧别名按冻结 A 默认值（0.2/600/1500）。

## 3. 整配置复现结果

所有 mean 均为冻结 A 快照的 native C++ `psnr_gaussian_splatting` 重渲染；旧目标无快照，只列其保存的在线文件值。

| 运行 | 指标数 | 首帧 | 全量均值 | median | std | iteration | gaussians | line gauss | mapping s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 旧目标（无快照） | 173 | 33.7540245056 | 34.6338976932 | 34.4938660 | 2.3065226 | 7611 | 475430 | 未知 | 未知 |
| 当前 A8 | 182 | 31.0604000092 | 31.9848955805 | 31.7638130 | 2.8134025 | 10480 | 527959 | 605 | 70.793 |
| 旧配置（别名 canonical 生效） | 173 | 32.8744659400 | **35.6117353052** | 35.9919815 | 2.7017719 | 9133 | 532901 | 0 | 69.805 |
| 旧运行时等效（别名留默认） | 185 | 33.6039352400 | 35.8297602265 | 36.0981712 | 2.5786473 | 9275 | 528864 | 671 | 72.776 |
| 旧配置 + pyramid3 | 166 | 31.1880378700 | 31.5378915141 | 30.6577368 | 2.9157492 | 9537 | 563787 | 0 | 68.875 |
| 旧配置 + poslr016 | 180 | 30.4183464100 | 35.6734069188 | 35.8598938 | 2.2487709 | 9543 | 523625 | 0 | 70.840 |
| 旧配置 + voxel010 | 176 | 32.6468582200 | 36.0820786412 | 36.6881084 | 2.5022068 | 9546 | 515432 | 0 | 70.720 |

运行目录：
- `results/repro_room1_oldconfig_20261023/replica_rgbd_0/room1`
- `results/repro_room1_oldruntime_20261023/replica_rgbd_0/room1`
- `results/repro_room1_oldconfig_pyramid3_20261023/replica_rgbd_0/room1`
- `results/repro_room1_oldconfig_poslr016_20261023/replica_rgbd_0/room1`
- `results/repro_room1_oldconfig_voxel010_20261023/replica_rgbd_0/room1`

在线 `psnr_gaussian_splatting.txt` 与 native 重渲染差异：
- 旧配置在线均值 `35.6117565398`，native `35.6117353052`。
- 当前 A8 在线 `31.9849529266`，native `31.9848955805`。
- 差异只来自浮点/重渲染路径，未改变 PSNR 公式。

已保存配置（不覆盖当前 A8 配置）：
- `cfg/gaussian_mapper/RGB-D/Replica/A_baseline/room1_old_recovered.yaml`
  - SHA256 `bf686855bd859c68540ec6ff19d3abaa48493a0336095f52b127480465af30f0`
- `cfg/gaussian_mapper/RGB-D/Replica/A_baseline/room1_oldruntime_canonical.yaml`
  - SHA256 `105222bffa6afddb1d63e3fb87c9f46d682a9da5e4c6e3ced96dc2f937086643`

## 4. 单因子恢复：只改回一个当前值

基准：`room1_oldconfig_effective.yaml`，native `35.6117353 dB`。

| 只恢复的当前改动 | 指标数 | 首帧 | 均值 | 相对旧配置 | 结论 |
|---|---:|---:|---:|---:|---|
| `GausPyramid.num_sub_levels: 2 → 3` | 166 | 31.1880379 | 31.5378915 | **-4.0738438** | 主因，单独就能把画质打回当前水平 |
| `Optimization.position_lr_final: 0.00032 → 0.00016` | 180 | 30.4183464 | 35.6734069 | +0.0616716 | 不是掉画质原因 |
| `Optimization.voxel_size: 0.25 → 0.10` | 176 | 32.6468582 | 36.0820786 | +0.4703433 | 不是掉画质原因 |

因此当前低画质目标的主因证据是 `GausPyramid.num_sub_levels=3`，不是 position LR、voxel、行采样或各向异性阈值。

## 5. 公共帧统计（按 `img_name`，未做 KF 序号相减）

脚本从各自 `natural_end/cameras.csv` 的 `fid → img_name` 映射合并，按绝对图片路径取交集。

| 对比 | 公共帧 | 旧配置/旧运行时均值 | 当前均值 | Δ | 更好/更差 |
|---|---:|---:|---:|---:|---:|
| 旧配置 vs 当前 | 24 | 37.0337354 | 32.7718372 | **+4.2618990** | 24 / 0 |
| 旧运行时等效 vs 当前 | 35 | 36.5532204 | 33.5493768 | +3.0038436 | 28 / 7 |

公共帧只有 24 / 35，是因为两种配置的 KF 集合明显分叉（173 vs 182、185 vs 182），不是采样挑选。
明细：
- `evidence/repro_room1_oldconfig_20261023/common_frames_by_img_name.csv`（旧配置 vs 当前，24 行）

旧目标没有 `cameras.csv`/快照，`psnr_gaussian_splatting.txt` 只有连续序号 0..172，无法可靠映射到 `img_name`。
因此旧目标与当前**只报告聚合值**，不做任何 KF 序号相减。

## 6. 剩余差距与下一步证据

### 已证明

- 旧配置能恢复画质：native `35.61 dB`，高于旧目标 `34.63 dB`。
- 当前 A8 的低画质由 `GausPyramid.num_sub_levels=3` 主导：单因子 `-4.07 dB`。
- 同一冻结二进制、同一数据/seed/LINE_MODE=2/LINE_LOOP=0/GBA off/no_viewer 下比较，排除代码/后端改动。

### 仍未闭合

- 旧目标与旧配置差 `+0.98 dB`（旧配置更高），首帧差 `-0.88 dB`（旧配置更低），不是同一结果。
- 旧目标没有保存快照，无法 native 重渲染复核，也没有保存相机 `img_name` 映射。
- 旧目标旧源码/旧二进制/环境/命令均未保存；旧目标 `7611` 步、`475430` 点，而旧配置 `9133` 步、`532901` 点，说明旧运行时训练预算/实现与冻结 A 不同。
- 旧别名是否被读取无法由现存证据判定：`Line.*`/顶层别名版得 173 条，旧默认版得 185 条，两者都不是旧目标的完整数值复现。

### 下一步（需用户确认后才做）

1. 若目标是“恢复高画质基线”：保存并使用 `room1_old_recovered.yaml`，它以 `num_sub_levels=2` 为主因恢复画质。
2. 若目标是“逐位复现旧文件”：必须先找到 2026-08-05 前后的旧二进制或源码快照；仅靠当前冻结 A 和旧 YAML 不能证明能精确复现。
3. 不建议继续扫描配置组合；当前证据已把当前回归定位到单一键。
