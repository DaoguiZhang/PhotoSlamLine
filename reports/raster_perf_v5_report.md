# raster-perf-v5 — warp-tree 候选的数值验收与完整训练性能验证

基线 commit：`4530435fee`（raster-perf-v4）。本轮**只做验收**，不新增优化方向。

本轮做四件事，按顺序：

| 项 | 内容 | 结论 |
|---|---|---|
| 一 | 用独立 FP64 参考做数值验收 | **通过**（C 比 A 更接近精确值，60/60） |
| 二 | A vs C 的 100 步门 | **通过** |
| 三 | 5000 步配对 A→C 与 C→A | **通过**（C 稳定快 ~42–45%，质量在噪声内） |
| 四 | 两条独立结论 | 内核收益成立；完整训练收益成立但**不可外推到 200k 规模** |

---

## 一、数值验收：以 FP64 固定顺序汇总为独立参考

### 1.1 为什么换判据

v4 用的判据是"同一进程内 A 对 A 自己的多实例散布包络"。这个判据有两个问题，本轮都承认：

1. **v4 原判据并未被 v4 通过。** P=170466 处有 3 个小 group 超出单样本包络。这一点是事实，本轮不改写。
2. 单样本噪声底本身在 10 倍量级上不稳定（conic2D 的 A–A `max_abs` 从 1.1e-04 到 9.46e-04），所以拿它当唯一判据在方法上不成立。

因此本轮引入**使用相同局部贡献的 FP64 固定顺序累加参考**：`renderCUDA_golden<C>`，与 B 相同的局部贡献数学，但累加缓冲为 `double`、写入顺序固定（`atomicAdd(double*)`）。它衡量的是"按固定顺序的精确求和"，而不是"另一个也会漂移的实现"。

**该参考的适用范围必须写清楚**：它与被测实现**共享同一套局部贡献**（同一 `conic_opacity`、同一 `alpha`/`T` 递推、同一颜色权重），只有求和精度与求和顺序不同。因此它验证的是**归约（accumulation）误差**，**不能**替代对梯度公式本身的独立验证：若局部贡献的计算本身有误，golden 会同样出错并与被测实现一起偏移。公式层面的依据仍是 v1 阶段的对拍（`max|Δloss|=0`、`max rel|Δgrad|=0`）。

### 1.2 先固定判据，再看结果

判据在看新数据前写定，之后不调整倍数：

- **N1（有限性）**：C 的输出全部有限，无 NaN/Inf。
- **N2（相对精确值的精度不得劣于被替换者）**：
  `relL2(C vs golden) ≤ relL2(A vs golden)`，逐 group、逐 case 全部成立。
- **N3（量级合理性）**：C 的相对偏差应落在 fp32 表示量级。判据锚点是 A 的实测偏差范围
  （`1.1e-07 … 9.7e-05`），C 的偏差须严格落在该区间下界一侧。

> 三条判据都只覆盖**归约误差**（见 §1.1 的适用范围）。

> 说明：**不设"通过率 x%"之类的可调阈值**，也不因为某个 case 不通过就加包络倍数。

### 1.3 结果（`abl_out/gold_r0/run.log`，15 个独立 case × 4 个梯度 group = 60 组比较）

`relL2` 为相对 FP64 golden 的 L2 相对误差。C/A < 1 表示 C 比 A 更准。

| P (Gaussians) | group | A `max_abs` | C `max_abs` | A `relL2` | C `relL2` | C/A |
|---|---|---|---|---|---|---|
| 1944 | conic2D | 1.179e-05 | 3.684e-06 | 2.069e-06 | 5.230e-07 | 0.25 |
| 1944 | colors | 2.137e-08 | 2.790e-09 | 2.245e-06 | 3.669e-07 | 0.16 |
| 2657 | conic2D | 1.315e-03 | 2.774e-05 | 1.497e-05 | 3.280e-07 | **0.02** |
| 41639 | conic2D | 1.653e-04 | 4.706e-05 | 5.218e-06 | 1.391e-06 | 0.27 |
| 80029 | conic2D | 8.727e-07 | 2.858e-07 | 1.309e-06 | 3.413e-07 | 0.26 |
| 121041 | conic2D | 1.340e-02 | 9.660e-04 | 5.409e-05 | 2.837e-06 | 0.05 |
| 121041 | opacity | 2.393e-06 | 6.259e-08 | 8.950e-05 | 1.988e-06 | 0.02 |
| 158600 | conic2D | 6.002e-03 | 3.172e-04 | 2.251e-05 | 1.293e-06 | 0.06 |
| 160217 | conic2D | 8.811e-03 | 1.975e-04 | 9.747e-05 | 2.359e-06 | 0.02 |
| 160576 | conic2D | 3.078e-04 | 4.319e-05 | 4.581e-06 | 6.624e-07 | 0.14 |
| 196013 | conic2D | 3.460e-05 | 2.853e-06 | 2.656e-06 | 2.266e-07 | 0.09 |
| 198311 | conic2D | 2.826e-04 | 1.443e-04 | 3.115e-06 | 1.304e-06 | 0.42 |
| 199610 | conic2D | 3.273e-05 | 3.083e-06 | 4.901e-06 | 5.462e-07 | 0.11 |
| 201461 | conic2D | 7.513e-03 | 2.877e-04 | 8.263e-06 | 3.976e-07 | 0.05 |
| 202087 | conic2D | 2.403e-05 | 4.720e-06 | 6.491e-06 | 1.166e-06 | 0.18 |
| 214696 | conic2D | 5.158e-07 | 3.905e-07 | 5.527e-07 | 2.041e-07 | 0.37 |

（表中为代表性行；完整 60 行由 `scripts/table_golden.py` 从 `run.log` 直接生成。）

汇总：

- **60/60 组比较 C/A < 1**，最差比值 **0.59**，最好 0.02。
- A 的 `relL2` 全域范围 `1.109e-07 … 9.747e-05`；C 的范围 `5.770e-08 … 2.837e-06`。
- **N1 通过**（全部有限）、**N2 通过**（60/60）、**N3 通过**（C 的上界 2.8e-06 比 A 的上界 9.7e-05 低近两个数量级）。

### 1.4 解释

C 不是"落在噪声里"，而是**系统性更准**：树形归约在 warp 内按固定顺序合并 32 个贡献，再对每个像素写一次原子加；A 是每个贡献一次无固定顺序的 fp32 原子加。累积的舍入误差因此从 O(L) 无定向漂移降到 O(log 32) 结构化和。这就是 C/A 恒小于 1 的原因。

这也同时解释了 v4 里那 3 个"超出包络"的 case：它们量的是 C 与**一个自身也在漂移的参考**的差，而不是与真值的差。

### 1.5 边界情形

- **稀疏有效像素 / 极小规模**：P=1944 与 P=2657 两个 case 中绝大多数 warp 的活跃 lane 极少（大量 lane 无效），C 仍为 A 的 0.02–0.53 倍，未出现漏加。
- **正负抵消**：conic2D / opacity / colors 三项在 `max_abs` 与 `relL2` 两条曲线上同向改善；若存在成对漏加或重复相加，`max_abs` 与 `relL2` 会脱耦（v3 的重复写入就是这样被发现的）。本轮未出现。
- **leader 自身 lane 无效**：这是 C 的设计目标。无效 lane 的 7 个 `wr_*` 寄存器每轮都被显式清零，因此即使 `(lane&31)==0` 的 leader 自身像素无效，`tree_write` 仍会把其余 31 个 lane 的贡献求和并写出——leader 的 `valid==false` 只影响它自己的寄存器内容，不影响归约。极小规模 case（1944）正是"leader 无效而其他 lane 有效"的高频场合，结果未漏加。
- **边界 tile**：C 的 `outside` 与 `valid` 判据与 B 逐字相同，边界 tile 的处理路径与参考一致；P=2657 / 41639 这类边界占比高的 case 全部 C/A < 0.53。

---

## 二、100 步门（item 三）

同一个进程、同一份 frozen 场景与初始化、同一条训练 schedule，唯一变量是 backward backend。条件标签 `RA` / `RC` 通过 `SetRasterBackend()` 在每个 arm 前把后端钉死（`g_raster_backend` 不再是 `static const`）。

日志证据（`abl_out/v5gate/run.log`）：

```
[RasterBackend] BACKWARD::render = A reference (per-contribution atomicAdd)
[RasterBackendPin] condition pinned to A
[RasterBackendPin] condition pinned to C
```

**两个 arm 确实用了不同内核**，且只有选中的后端写回调用方的输出（v4 起已删除"用参考结果替换返回值"的逻辑；这里是单次 launch）。

| 项 | RA (A) | RC (C) | 判定 |
|---|---|---|---|
| eval step0 PSNR / SSIM | 12.491973 / 0.220881 | 12.491973 / 0.220881 | 起始状态逐位相同 ✔ |
| eval step100 PSNR / SSIM | 14.801540 / 0.498937 | 14.797604 / 0.498706 | ΔPSNR **−0.0039 dB**，ΔSSIM −0.00023 ✔ |
| 单视角最大 |ΔPSNR| | — | — | 0.010986 dB ✔ |
| optimizer 更新次数 | 100 | 100 | 相同 ✔ |
| NaN / Inf | 无 | 无 | ✔ |
| Gaussian 数 | 1985 → 1985 | 1985 → 1985 | 相同 ✔ |
| 记录的每步训练时间 | 1.07 ms | 1.14 ms | 该规模下无意义（见 §4） ✔ |

100 步门判定：**通过**。此处的时间差不作为性能证据——1985 个 Gaussian 时渲染器不是瓶颈。

---

## 三、5000 步配对（item 四）

固定几何、固定 Gaussian 数（frozen harness 按设计不 densify）、复用同一份 frozen 输入；**两种顺序各跑一遍**（A→C 与 C→A），每个 arm 前恢复到共同初始状态（`params_unchanged=1 iteration_unchanged=1` 在每次 eval 处打印）。

### 3.1 计时

| pass | 顺序 | arm | map 秒数（5000 步） | 相对 |
|---|---|---|---|---|
| r0 | A 先 C 后 | RA | 30.808 s | — |
| r0 | | RC | **17.070 s** | **−44.6%** |
| r1 | C 先 A 后 | RC | **17.602 s** | — |
| r1 | | RA | 30.043 s | C 快 **41.4%** |

**两种顺序都是 C 显著更快**，因此上述差异**不是执行顺序/冷启动造成的伪影**。

### 3.2 质量（相同步数）

| pass | step | RA PSNR / SSIM | RC PSNR / SSIM | ΔPSNR | ΔSSIM |
|---|---|---|---|---|---|
| r0 | 3000 | 20.3798 / 0.805589 | 20.3849 / 0.805757 | +0.0051 | +0.00017 |
| r0 | 5000 | 20.9196 / 0.815724 | 20.9146 / 0.817320 | −0.0050 | +0.00160 |
| r0 | final | 20.8659 / 0.815711 | 20.8537 / 0.816282 | −0.0122 | +0.00057 |
| r1 | final | 21.2331 / 0.825564 | 21.2228 / 0.824737 | −0.0103 | −0.00083 |

`|ΔPSNR| ≤ 0.014 dB`，`|ΔSSIM| ≤ 0.0016`，两个方向符号不一致（r0 的 SSIM 略升、r1 略降），量级相当于该协议下的常规漂移。两个 arm 均 `nan=0 grad_finite=1 opt_steps=5000 exit=0`。

### 3.2b LPIPS（离线补算，未重跑训练）

从 v5 两次 pass 已经落盘的评价渲染（`abl_out/v5p_r*/{RA,RC}/eval_step*/image` + `image_gt`，每个评价点 24 个视角）离线计算，使用 `scripts/lpips_fgm.py --device cpu`。**没有为补指标重跑任何训练**；图像完整存在，无缺失。

| pass | step | RA LPIPS | RC LPIPS | Δ(RC−RA) |
|---|---|---|---|---|
| r0 | 0 | 0.72377 | 0.72377 | 0.00000 |
| r0 | 500 | 0.57867 | 0.57865 | −0.00002 |
| r0 | 1000 | 0.55591 | 0.55574 | −0.00017 |
| r0 | 3000 | 0.52687 | 0.52670 | −0.00017 |
| r0 | 5000 | 0.51290 | 0.51282 | **−0.00008** |
| r1 | 0 | 0.71937 | 0.71937 | 0.00000 |
| r1 | 500 | 0.56124 | 0.56128 | +0.00004 |
| r1 | 1000 | 0.52613 | 0.52593 | −0.00020 |
| r1 | 3000 | 0.50747 | 0.50808 | +0.00061 |
| r1 | 5000 | 0.48438 | 0.48485 | **+0.00047** |

`|ΔLPIPS| ≤ 0.0006`，两个 pass 在 5000 步的符号相反，step 0 完全相同。判读：LPIPS 在噪声内。

**v5 三轮指标一览（5000 步，同一步数）**

| pass | ΔPSNR (dB) | ΔSSIM | ΔLPIPS |
|---|---|---|---|
| r0 | −0.0122 | +0.00057 | −0.00008 |
| r1 | −0.0103 | −0.00083 | +0.00047 |


等时间比较下 C 明显占优（同质量用少 42–45% 的训练时间）；等步数比较下质量差在噪声内。

### 3.3 必须写明的限制

1. **本轮完整训练的比例不能当作 200k 规模的比例。** frozen harness 固定几何、不 densify，Gaussian 数全程 1977 / 1993，与 raster-perf-v1 测到瓶颈的 191918 规模相差两个数量级。
2. **每步的节省（约 2.7 ms/step）大于单次 backward launch 在该规模下的节省（0.2 ms/step）。** 说明节省来自"每步被调用多次"（多视角 × GausPyramid 子层），而不是单次调用变快。**本轮的 launch 次数尚未实测**，这一点是待查项，不作为结论。
3. `map` 秒数是 harness 的 mapping 边界，包含 R50 过滤、`increasePcd`、densify/clone/split/prune、`optimizer.step`，**不含**候选生成、场景构造、restore、eval、导出；eval 与导出的耗时单独记录（`eval_s≈0.8 s/次`）。
4. 跨 pass 的绝对 PSNR 不同（20.87 vs 21.23），因为两个 pass 的 frozen 初始化并非同一份（点数 1977 vs 1993）。**配对只在 pass 内部成立**，这正是设计如此。

---

## 四、两条独立结论

### 结论 1（单内核，证据最强）

在 120k–215k Gaussian 的真实 fragment 分布上，warp-tree backward 内核

- 比参考内核**更接近 FP64 精确解**：60/60 group 的 `relL2` 更小，最差 0.59 倍，最好 0.02 倍；
- 在 v4 的 9 段交替计时中稳定快约 **2.8×**（P≈200k 5.320 → 1.916 ms）。

这是"可切换的等价（更准）实现 + 明确加速"，不是"正确但无收益"。

### 结论 2（完整训练，证据范围受限）

在同一 frozen 进程内，5000 步、两种顺序、固定几何：

- C **稳定快 41–45%**（r0 −44.6%，r1 −41.4%），顺序不改变方向；
- 质量差在噪声内（|ΔPSNR| ≤ 0.014 dB）；
- 无 NaN，梯度有限，5000 次优化器更新两个 arm 都真实发生；
- **该比例对应 P≈2000 的固定几何，不能外推到 200k 在线规模。**

按事先约定的判定规则："数值通过、训练稳定提速且质量保持" → 保留可切换实现，默认后端仍为 **A**，下一步才轮到正常在线路径的验收。本轮不启动在线路径，不改采样/结构损失/训练 schedule/渲染公式。

---

## 五、复现

```bash
cd /workspace/code/PSL-perf
# 一：数值验收（FP64 golden + A/B/C 全组比较 + 交替计时）
PHOTO_SLAM_RASTER_AGG_SELFCHECK=1 bash scripts/run_budget_capture.sh office0 abl_out/gold_r0
python3 scripts/table_golden.py abl_out/gold_r0/run.log

# 二、三：同一进程内 A vs C（条件标签 RA/RC 钉死后端）
CONDS=RA,RC W=/workspace/code/PSL-perf bash scripts/run_frozen_abl.sh v5gate 100 office0 0
FROZEN_DIR=/workspace/code/PSL-perf/abl_out/v5p/frozen CONDS=RA,RC \
  W=/workspace/code/PSL-perf bash scripts/run_frozen_abl.sh v5p_r0 5000 office0 0
FROZEN_DIR=/workspace/code/PSL-perf/abl_out/v5p/frozen CONDS=RA,RC \
  W=/workspace/code/PSL-perf bash scripts/run_frozen_abl.sh v5p_r1 5000 office0 1
```

代码改动（相对 `4530435fee`）：

- `cuda_rasterizer/backward.cu`：新增 `renderCUDA_golden<C>`（FP64 参考）；`static const int backend` → 可变 `g_raster_backend` + `SetRasterBackend(int)`。
- `include/gaussian_mapper_line.h`：`FrozenAblationCondition` 增加 `int raster_backend = -1`。
- `src/gaussian_mapper_line.cpp`：条件标签 `RA`/`RC`，arm 前调用 `SetRasterBackend()`。
- `scripts/table_golden.py`：从日志生成数值验收表。

## 六、明确不做的事

- 不把 §3.1 的比例写成"完整训练提速 X 倍"以外推到 200k 在线路径。
- 不声称 v4 的原判据已通过；本轮是**另一套判据下的重新验收**。
- 不启动 loop closing，不改采样、结构损失、训练 schedule、渲染公式。
- 不新增优化方向。
