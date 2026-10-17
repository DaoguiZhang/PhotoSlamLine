# office0 四组在线对照：P_ref / P_fast / L_ref / L_fast

## 0. 结论

| 对照 | 结果 |
|---|---|
| **P_fast − P_ref**（渲染器 C vs A，纯点） | 质量与定位无差别（ΔPSNR −0.13±0.38 dB，ΔATE +0.3 mm 属噪声）；mapping 时间 −1.2±0.9 s，总墙钟 +0.5±0.7 s → **纯点管线无渲染器收益** |
| **L_fast − L_ref**（渲染器 C vs A，L 配置） | **mapping 时间 −32.0±3.4 s（130.5→98.5 s，−24.5%），总墙钟 −29.7±4.1 s（−19%）**；代价是 PSNR −0.34±0.22 dB（三轮同向）、SSIM −0.0004、LPIPS +0.0023；ATE 不变 |
| **L_fast − P_fast**（线 vs 点，同用 C） | PSNR +3.19±0.74 dB、SSIM +0.0143、LPIPS −0.0238（更好）；ATE +0.2 mm（噪声级）；墙钟 +66.5 s、mapping +55.8 s |
| **L_fast − P_ref** | PSNR +3.07±0.36 dB、SSIM +0.0144、LPIPS −0.026；ATE +0.5±0.6 mm；墙钟 +66.9±1.3 s |

归因：**速度收益完全来自渲染器，且只在 L 管线出现**（点管线的渲染占比太小，无法测出）；
线方法以约 2.2× 墙钟（2.4× mapping）换取 +3.1~3.2 dB / +0.014 SSIM / −0.026 LPIPS 的重建质量，
定位精度没有可分辨的改善（L 略差 0.2–0.5 mm，在 3 轮散布内）。渲染器 C 在 L 上让质量略降
0.34 dB 是真实但很小的权衡，本轮不做调参追求正结果。

## 1. 固定版本、配置与标识

- 源码：基线 `5e34226b0c3f7416355973becece41b898e8815b`（GBA 状态一致性收尾）+ 本轮 quad 补丁
  （点专用构建目标、`[QuadMetrics]` 遥测、点管线 `TrackRGBD` 入口、共享配置生成、脚本），
  补丁已提交为 `4c1b86bb99`（含本报告）。发车时工作树为"基线 + 同一补丁"（`dirty_files=47`
  含运行产物），因此**以二进制哈希为准确身份**：
  - `bin/replica_rgbd` = `6800cc2d014b813259e82b7c0de467c42f197955c0c03bc8ed6ba6f6b52bd962`
  - `bin/replica_rgbd_point` = `3b34c1db8478b3e5fa074b0861b0ab0afac1ecd7dbe8cf32d3afe8a3072e112e`
  - `ORB-SLAM3/lib/libORB_SLAM3.so` = `6d9ace129739168019ba3f1e737363bdd4848b84984ec8b998d855c10c2d35ac`
  - `lib/libgaussian_mapper.so` = `6a87b655cd46a4a1ed9a958515e423880beee57904e9ece1b34d0120fa529636`
  - 构建：`scripts/build_two_level.sh`（ORB-SLAM3 先行 + DSO 布局检查 `0x27d`），
    `CMAKE_BUILD_TYPE=Release`，`CMAKE_CUDA_ARCHITECTURES=52`，无额外 CXX/CUDA flags。
- 配置：四组**共用同一份** `gaussian_quad.yaml`（sha256 `30f5deda…`），相对
  `cfg/gaussian_mapper/RGB-D/Replica/office0.yaml` 只有两处差别：
  - `Optimization.weight_line_coherence/shape_ecc/shape_ori = 0.0`（**用实际权重**关闭结构项，
    不动前端/初始化；因此本轮 L 不是论文 Full）；
  - `Record.record_ground_truth_image = 1`（只为离线 LPIPS 存 GT，不参与训练）。
  ORB 配置 `office0.yaml` sha256 `a39fd203…`，四组相同。
- 渲染后端：`PHOTO_SLAM_RASTER_BACKEND=A|C`（A=参考逐贡献 atomicAdd；C=收敛控制流+5 级 warp 树）。
- GBA 形变：默认关闭（本轮从未设 `PHOTO_SLAM_GBA_APPLY_CORRECTION`）；事件消费、相机位姿更新、
  applied-event 版本与既有 LOOP 更正不变。日志确认 `[GBA-Apply] deformation DISABLED (default …)`。
- 数据/播放：同一 `replica/office0`，2000 帧，同一 ORB 配置、同一 `assemble` 方式（dataset 目录原样），
  末端规则由各管线自身 `_shutdown` 收尾逻辑决定（不再额外裁剪）。
- 产物归档：`results/`（已加入 `.gitignore`）与二进制**不进源码 Git**；`provenance.txt` 记录 commit/哈希/配置。
  未重写任何历史。

## 2. 纯点基线身份（核实结果）

- 上游快照：`/workspace/code/Photo-SLAM`（CVPR 2024 上游树）。证据：`include/ src/ cuda_rasterizer/`
  中无 `MapLine`/`KeyLine`/`LSDextractor` 任何符号；入口只调用 `TrackRGBD`；源码树 tar 哈希
  `82c3cc061487edfd0ab4f60f9f24b7bf307591ea2e523939ec944c6a85be4660`；其 `bin/replica_rgbd`
  sha256 `db7b842737fe241559d692a5ab1cc8b8c7d91323d325655ac7b937d4a28b4cb7`。
  该上游二进制**没有参与本轮比较**（缺少同一套评价/遥测实现，混用会破坏评价协议一致性）。
- 本轮的 P 是**同仓库纯点构建** `P_ro`，不是上游二进制，也不冒充：
  - 编译期 `USE_LINE_GAUSSIAN=0` → `GaussianMapper`（无线种子/线增密/线损失）；
  - 入口按该宏改调 `System::TrackRGBD`（**帧内不做 LSD 线提取**）；
  - 运行期 `PHOTO_SLAM_LINE_MODE=0` → `LocalMapping::Run` + `LoopClosing::Run`（BA/LC 无线顶点/边）。
  - 关键发现：RGB-D 原先**总是**调用 `TrackRGBDWithLine`，所以仅 `LINE_MODE=0` 并不是纯点前端；
    逐轮日志确认 P 组 `line_gaussians=0`。
  - 与上游的已知差异（因此不能称"等价于原版"）：perf 分支在共享代码里的渲染器 A/C 调度、mapper
    预算/调度、跟踪与回环修复、Atlas 事件（GBA 位姿）通路。这些差异对四组是共同的。
- 没有把线 mapper 或线算法复制进点基线：点构建只多了一个编译宏、一个入口分支和一个
  `add_slam_executable` 目标。

## 3. 四个条件

| 条件 | 二进制 | LINE_MODE | 后端 | 说明 |
|---|---|---|---|---|
| P_ref | `replica_rgbd_point` | 0 | A | 纯点前端 + 点 Gaussian mapper |
| P_fast | `replica_rgbd_point` | 0 | C | 同上，仅渲染器不同 |
| L_ref | `replica_rgbd` | 2 | A | 点线前端、线地图、线种子/线感知增密全部保留；结构损失权重 0 |
| L_fast | `replica_rgbd` | 2 | C | 同上，仅渲染器不同 |

L 实际保留的行为（日志/产物验证）：`[LineMode] … MODE=2 | PO_LINE=1 LBA_LINE=1`，线端点参与
Pose-Opt 与 Local BA，线几何参与建图与增密，`line_gaussians` > 0（本轮 16–310，随线数量波动）。
结构项通过权重 0 关闭，未用会同时改变前端/初始化的开关。

## 4. 执行方式

- 先冒烟：120/300 帧固定子集，四组各一次，确认后端、`LineMode`、`line_gaussians`、快照与指标可用
  （`results/quad_smoke`）。冒烟同时暴露了 0 关键帧时点管线收尾会挂死的问题（见 §7）。
- 正式：**3 轮 × 4 组，串行、无并发**，顺序预先排定并交错：
  R1 `P_ref → P_fast → L_ref → L_fast`；R2 `L_fast → L_ref → P_fast → P_ref`；
  R3 `P_fast → P_ref → L_fast → L_ref`。
- 一条命令：`ROUNDS=3 scripts/quad_office0.sh`（见 §6）。
- **不是冻结几何的受控配对**：在线前端跨进程，KF/高斯/更新次数必然不同；下表如实报告这些差异。

## 5. 结果（3 轮，均值 ± 标准差）

### 5.1 逐轮

| cond | r | exit | wall s | map s | opt steps | KF | Gauss | line G | PSNR | SSIM | LPIPS | ATE(m) | cov | VRAM MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P_ref | 1 | 0 | 58.3 | 42.9 | 8581 | 100 | 184588 | 0 | 39.80 | 0.9688 | 0.0725 | 0.0047 | 1.000 | 2860 |
| P_ref | 2 | 0 | 60.5 | 44.3 | 8981 | 102 | 181129 | 0 | 39.54 | 0.9679 | 0.0749 | 0.0053 | 1.000 | 3148 |
| P_ref | 3 | 0 | 59.8 | 44.5 | 8781 | 104 | 187240 | 0 | 39.33 | 0.9670 | 0.0788 | 0.0047 | 1.000 | 3082 |
| P_fast | 1 | 0 | 58.9 | 41.5 | 10581 | 101 | 189139 | 0 | 40.02 | 0.9705 | 0.0705 | 0.0051 | 1.000 | 3056 |
| P_fast | 2 | 0 | 61.6 | 44.1 | 11081 | 106 | 188000 | 0 | 39.01 | 0.9663 | 0.0744 | 0.0050 | 1.000 | 3282 |
| P_fast | 3 | 0 | 59.5 | 42.5 | 10581 | 105 | 192756 | 0 | 39.25 | 0.9673 | 0.0745 | 0.0054 | 1.000 | 3008 |
| L_ref | 1 | 0 | 157.5 | 131.6 | 21038 | 112 | 198559 | 22 | 43.06 | 0.9831 | 0.0453 | 0.0053 | 1.000 | 2940 |
| L_ref | 2 | 0 | 153.0 | 127.8 | 21497 | 115 | 193513 | 47 | 43.10 | 0.9830 | 0.0463 | 0.0053 | 1.000 | 2970 |
| L_ref | 3 | 0 | 158.0 | 132.0 | 21529 | 119 | 206696 | 48 | 42.73 | 0.9821 | 0.0497 | 0.0054 | 1.000 | 3116 |
| L_fast | 1 | 0 | 126.2 | 98.6 | 25027 | 119 | 203123 | 17 | 42.54 | 0.9822 | 0.0486 | 0.0053 | 1.000 | 3030 |
| L_fast | 2 | 0 | 128.0 | 99.7 | 25058 | 120 | 217685 | 16 | 43.00 | 0.9832 | 0.0450 | 0.0052 | 1.000 | 3076 |
| L_fast | 3 | 0 | 125.2 | 97.2 | 24942 | 113 | 211695 | 310 | 42.33 | 0.9816 | 0.0545 | 0.0057 | 1.000 | 2932 |

### 5.2 汇总

| cond | PSNR | SSIM | LPIPS | ATE(m) | cov | wall s | map s | opt steps | KF | Gauss | VRAM MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P_ref | 39.56 ± 0.24 | 0.9679 ± 0.0009 | 0.0754 ± 0.0032 | 0.00490 ± 0.00004 | 1.000 | 59.6 ± 1.1 | 43.9 ± 0.9 | 8781 ± 200 | 102 ± 2 | 184k ± 3k | 3030 ± 151 |
| P_fast | 39.43 ± 0.53 | 0.9680 ± 0.0022 | 0.0731 ± 0.0023 | 0.00517 ± 0.00024 | 1.000 | 60.0 ± 1.4 | 42.7 ± 1.3 | 10750 ± 290 | 104 ± 3 | 190k ± 2k | 3115 ± 146 |
| L_ref | 42.96 ± 0.20 | 0.9827 ± 0.0005 | 0.0471 ± 0.0023 | 0.00533 ± 0.00008 | 1.000 | 156.2 ± 2.8 | 130.5 ± 2.3 | 21350 ± 275 | 115 ± 4 | 200k ± 7k | 3009 ± 94 |
| L_fast | 42.62 ± 0.34 | 0.9823 ± 0.0008 | 0.0494 ± 0.0048 | 0.00539 ± 0.00029 | 1.000 | 126.5 ± 1.4 | 98.5 ± 1.2 | 25010 ± 60 | 117 ± 4 | 211k ± 7k | 3013 ± 74 |

### 5.3 四个对照（逐轮差值的均值 ± sd）

| 对照 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔATE(m) | Δwall s | Δmapping s |
|---|---|---|---|---|---|---|
| P_fast − P_ref | −0.129 ± 0.379 | +0.0001 ± 0.0016 | −0.0023 ± 0.0020 | +0.00027 ± 0.00055 | +0.46 ± 0.69 | −1.21 ± 0.88 |
| L_fast − L_ref | **−0.342 ± 0.216** | −0.0004 ± 0.0005 | +0.0023 ± 0.0032 | +0.00006 ± 0.00020 | **−29.72 ± 4.13** | **−31.99 ± 3.41** |
| L_fast − P_fast | +3.194 ± 0.742 | +0.0143 ± 0.0026 | −0.0238 ± 0.0050 | +0.00021 ± 0.00005 | +66.45 ± 0.78 | +55.79 ± 1.14 |
| L_fast − P_ref | +3.065 ± 0.364 | +0.0144 ± 0.0010 | −0.0260 ± 0.0034 | +0.00049 ± 0.00058 | +66.91 ± 1.33 | +54.58 ± 1.60 |

读法：
- 渲染器收益（L）是**时间**上的，−24.5% mapping / −19% 墙钟；代价是 0.34 dB，三轮同向但很小。
- 渲染器在 P 上没有可测收益：mapping −1.2 s（±0.9），墙钟 +0.5 s（±0.7）；点管线里光栅化耗时占比低。
- 线方法的重建收益大且稳定（+3.1~3.2 dB、SSIM +0.014、LPIPS −0.026），代价是 2.2× 墙钟、
  更多 KF（117 vs 103）、更多高斯（211k vs 187k）、2.4× optimizer 更新（25.0k vs 10.8k）。
- 定位：四组 ATE 均在 4.9–5.4 mm，差异 ≤0.5 mm，与轮间散布同量级 → **线方法未改善定位**。

## 6. 一条命令 + 指标定义

```bash
cd /workspace/code/PSL-perf
./scripts/build_two_level.sh            # 两级构建，含 DSO 布局校验
ROUNDS=3 ./scripts/quad_office0.sh      # 12 次串行运行 + 自动汇总
# 产物：results/quad_office0/{provenance.txt, runs/, logs/, per_run.csv, summary.md}
```

指标定义（四组完全一致）：
- **PSNR/SSIM**：每组收尾时 `renderAndRecordAllKeyframes("_shutdown")` 对**每个关键帧**渲染一次，
  用两个构建共用的 `loss_utils::ssim/psnr` 计算（`psnr.txt` / `dssim.txt`，后者存的就是 SSIM），
  取关键帧均值。**这是训练视图（关键帧位姿 + 训练用取色）指标，不是留出视角**。
- **LPIPS**：对同一批 `image/` 与 `image_gt/` 用 `scripts/eval_snapshot_lpips.py --device cpu` 离线计算，
  CPU 时间不计入 mapping。
- **ATE**：`scripts/eval_tum_se3_v2.py`，估计轨迹中心化的 SE(3) RMSE（无 Sim(3)、无对齐缩放），
  GT = `pose_TUM.txt`；**覆盖率** = 写出位姿数 / 2000，四组均为 1.000。
- **mapping 时间**：每次迭代内 render→optimizer step 的累计墙钟（`[QuadMetrics] mapping_seconds`），
  不含前端跟踪、收尾保存和 LPIPS；**总墙钟** = 进程启动到退出（runner 用 `date +%s%N`）。
- **optimizer 更新数**：`optimizer_->step()` 实际调用次数（`[QuadMetrics] optimizer_steps`）。
- **峰值显存**：runner 以 200 ms 采样 `nvidia-smi --query-compute-apps` 的**本进程** used_memory。
- 不使用整数毫秒日志推断微小差异；所有差值以 3 轮散布报告。

## 7. 限制与异常

- 单场景（office0）、单数据集、3 轮；不自动扩展其他序列。
- 在线前端跨进程：关键帧、高斯数、更新次数与训练视图序列都不同，**不能把这些运行当作共享
  冻结几何的受控配对**；这也是 P 与 L 之间只能做"完整系统比较"的原因。
- PSNR/SSIM/LPIPS 是训练视图指标，会偏向训练更充分的管线（L 的更新次数约为 P 的 2.4×），
  不能当作留出视角的重建质量。
- 峰值显存是进程级采样（200 ms），±150 MiB 的差异不应解读为显著。
- 冒烟阶段发现：当序列过短导致 **0 关键帧**时，点管线的收尾训练循环会在空场景上挂住（本次被
  runner 超时/手工终止）。这只影响退化输入，2000 帧正式运行 12/12 均 exit 0；已在报告中标注，
  本轮不修改该行为（不新增算法/调度改动）。
- L_fast 第 3 轮 `line_gaussians=310`（其余 16–48）：线数量随前端匹配显著波动，属在线不确定性，
  已如实记录，不做筛选。
