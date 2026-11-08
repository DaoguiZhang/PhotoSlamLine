# 版本冻结：本阶段正式版本清单

日期：2026-11-07（范围修正版：L_core 去掉 Gaussian mapping 的结构线损失）
**未改动**：`PSL-upstream-official`、`PSL-fullmap-lock-v1`（L0，300 次历史）、
`PSL-fullmap-exp-v1`、`PSL-dedupfix`（L1）、`PSL-lavn`、`PSL-replay`、`PSL-review`、
全部旧 `cfg/**`、全部历史结果。

> **本轮唯一的范围修正**：最终 L 版本（`L_core`）**不再包含** Gaussian mapping 的结构线损失
> （`weight_line_coherence` / `weight_line_shape_ecc` / `weight_line_shape_ori` 三个有效权重全为 0），
> 且这些无效损失项**不再计算**（`[LossAudit] structural_active_steps=0`、`loss_is_photometric_only=1`）。
> 该修正**只作用于新冻结配置**：历史 Full/OFF 实验、结构损失的实现代码、封存版本一律保留、未改。
> 因此 **`L_core-Full` 不再是正式必跑方法**，正式对比为 `O_core` vs `L_core` 两个方法。
> 与此同时**没有**关闭：线检测、线匹配、点线相机优化、线初始化与既定增密机制——
> 线元数据/张量结构全部保留（`is_line_`、`line_dir_w_`、`line_anchor_` 等仍被初始化、克隆、分裂使用）。

## 1. 候选与身份

| 代号 | 目录 | 角色 | 身份（sha256 前 32） |
|---|---|---|---|
| **O_core** | `PSL-review/O_core` | **正式对比基线**（官方 + 公共工程修复） | `libgaussian_mapper.so 4293b9d7349f8bde2436097e5a278f83` |
| O_raw | `PSL-review/O_raw` | 官方原样（仅构建兼容），只作对照 | `0396c6d327dfdbce11fd5ef3754c0487` |
| **L_core** | `PSL-final`，`V=0 N=0` | **正式 L 基线**（A 类修复，无 V/N） | 见下 |
| L_VN（实验） | `PSL-final`，`V=1 N=1` | **实验版本**，非默认 | 同上 |
| L_V / L_N（实验） | `PSL-final`，`V=1 N=0` / `V=0 N=1` | 仅用于消融 | 同上 |
| **PSL-final** | `/workspace/code/PSL-final` | 本阶段冻结版本（源码 + 构建） | `libgaussian_mapper.so 538f46296c8e33721ade70e042ae2d5f…`<br>`libORB_SLAM3.so 5a14981faac97ed9db7cf7d834168db2` |
| 上一轮重放版 | `PSL-replay` | 产生 30 次重放结果的版本（保留） | `d82b9ccae0368d72ea0fd52118884f2a` |
| L0（历史） | `PSL-fullmap-lock-v1` | 300 次历史实验，未动 | `98cc87928caa592823650048aa78d194` |

`PSL-final` = `PSL-replay` + `diffs_final/exit_fix_and_audit.patch`：
退出修复、事件审计、有效参数打印、例程退出码、结构损失参与度审计与跳过守卫。
**未改**任何跟踪/建图/优化器数学（结构损失表达式一字未动，只在权重全 0 时整块跳过）。
与 PSL-replay 的差异仅 6 个文件（`diffs_final/exit_fix_and_audit.patch`，810 行 / 22 hunk），
全部可用 patch 复核。

**L_core 的完整定义（新冻结配置）**：

| 组成 | 取值 |
|---|---|
| A 类工程修复（去重/写回/变换契约等） | 保留 |
| 线前端（检测/匹配/点线相机优化/线初始化/线高斯） | 保留 |
| 结构线损失（coherence / shape_ecc / shape_ori） | **全部 0，且不计算** |
| 窄带光度加权 | 本树**不存在**该损失项（`grep band/narrow` 无命中），等价于关闭 |
| V（`PHOTO_SLAM_DEDUP_VOXEL`） | **0** |
| N（`PHOTO_SLAM_DEDUP_NOTIFY`） | **0** |
| `lambda_anisotropy`（通用各向异性惩罚） | 0.01（**不是**线结构项，保留） |

对应配置文件：`PSL-evtrepl/cfg_v2/{office0_mono,desk3_rgbd}.nostruct.yaml`（= 原 cfg 只改这 3 行）。
**重要复用事实**：封存协议 `PSL-fullmap-exp-v1/configs/off/formal/*.off.yaml` 与这里的 L_core
在这三个键上**完全相同**（都是 0.0），因此 L_core 的 Gaussian 配置不需要新发明一套。

## 2. 正式版本的有效开关（全部显式列出）

| 开关 | 默认 | 含义 | 正式版取值 |
|---|---|---|---|
| `PHOTO_SLAM_LINE_MODE` | 2 | 点线前端 | 2 |
| `PHOTO_SLAM_LINE_LOOP` | 2（本实验 0） | 回环中的线约束 | **0**（与历史 300 次协议一致） |
| `PHOTO_SLAM_LINE_RESCUE` | 0 | 线救援 | 0 |
| `PHOTO_SLAM_SHADOW_LINE` | 0 | 影子线 | 0 |
| `PHOTO_SLAM_DEDUP_VOXEL` (V) | 1 | 体素"当前存活"去重 | **正式版 0**；实验版 1 |
| `PHOTO_SLAM_DEDUP_NOTIFY` (N) | 1 | GBA/合并写回通知 | **正式版 0**；实验版 1 |
| `PHOTO_SLAM_GAUSS_LOOP_TRANS_M` / `_ROT_DEG` | 0.10 / 2 | 写回门限 | 未改动 |
| `PHOTO_SLAM_SEED` | 未设 | 随机种子 | 由调度器显式给定 |
| `PHOTO_SLAM_EVT_AUDIT(_KFS/_SAMPLES)` | 关 | 生产链事件审计 | 仅诊断，正式运行关闭 |
| `PHOTO_SLAM_PRINT_EFF_PARAMS` | 关 | 打印有效优化器超参 | 正式运行开启（只多一行日志） |
| `Optimization.weight_line_coherence` | 0.001（cfg） | 线方向一致性结构正则 | **L_core = 0.0** |
| `Optimization.weight_line_shape_ecc` | 0.002（cfg） | 线形状偏心约束 | **L_core = 0.0** |
| `Optimization.weight_line_shape_ori` | 0.005（cfg） | 线形状朝向约束 | **L_core = 0.0** |
| `Optimization.lambda_anisotropy` | 0.01（cfg） | 通用各向异性惩罚（**非线结构项**） | 保留 0.01 |

> **正式版默认 = V0N0（L_core）。** 理由见 `REPORT_FINAL.md` §5：
> V 有确定维护成本、画质作用未确认且 office0 上完全未触发；N 的"应用集合"不可复现，
> 其一致性契约**未被验证**。旧封存版本的默认值一个都没有改。

## 3. 有效优化器超参（实测，非推断）

`PHOTO_SLAM_PRINT_EFF_PARAMS=1`，office0_mono，PSL-final：

```
[EffParams] group=0 nparams=1 lr=3.73615e-05 eps=1e-15 beta1=0.9 beta2=0.999 weight_decay=0   (xyz, lr_init*extent)
[EffParams] group=1 nparams=1 lr=0.0025      ...   (feature_dc = Optimization.feature_lr)
[EffParams] group=2 nparams=1 lr=0.000125    ...   (feature_rest = feature_lr/20)
[EffParams] group=3 nparams=1 lr=0.05        ...   (opacity = Optimization.opacity_lr)  <-- 有限、等于配置
[EffParams] group=4 nparams=1 lr=0.005       ...   (scaling)
[EffParams] group=5 nparams=1 lr=0.001       ...   (rotation)
```

全部有限；与 `cfg/gaussian_mapper/**` 一致。group 0 因乘 `spatial_lr_scale` 而小于
`position_lr_init`（0.00032×0.1168），这是既有行为，不是配置不符。
**上游 `opacity_lr_(opacity_lr_)` 自初始化缺陷在生产路径不可达**（parser 无条件覆盖），
本轮未改库、未改 cfg。

## 4. 交付物

| 文件 | 内容 |
|---|---|
| `GATES.md` | 7 项门禁 PASS/FAIL/NOT_COVERED |
| `REPORT_FINAL.md` | 本阶段报告（5 条结论） |
| `RUN_LEDGER.csv` | 逐宿主进程：映射完成 / 产物完整 / 退出状态 / 墙钟 |
| `report/*.csv` | 30 次运行的逐检查点、逐事件、机制、对比、事后评测 |
| `exitreg/summary.csv` + `exitreg/*.log` | 退出与审计回归（4 个用例） |
| `lossaudit/summary.csv` + `lossaudit/*.log` | 结构损失**不参与**优化的读回与审计（OFF/ON 对照，2 个场景 × 4 次） |
| `cfg_v2/*.nostruct.yaml` | L_core 的 Gaussian 配置（三个结构权重 = 0.0，其余逐字未改） |
| `loss_audit_check.sh` | 结构损失短跑审计脚本（200 步，replay-only） |
| `diffs_final/exit_fix_and_audit.patch` | 退出修复 + 事件审计 + 结构损失审计 |
| `diffs/` | 重放harness、O_core 数学修复、O_raw 构建兼容、测试源 |
| `FORMAL_PLAN/` | 正式对比 manifest + 调度器（两方法 300 项，本轮**未执行**，理由见其中 README） |

## 5. 结构损失「确实不参与优化」的证据（不只是改了一个没人读的 YAML 键）

做法：把**解析后**的三个权重在 `run()` 开头打印为 `[LossCfg]`，并在两条训练路径的损失组装点
计数 `[LossAudit]`。若三个权重为 0，整块结构损失代码**根本不执行**（不分配张量、不启动线损失核、
不往 `loss` 上加任何东西），计数为 0；若权重非 0，计数随迭代增长。因此"配置回读"和"实际参与"
是同一个可观测事实。

`lossaudit/summary.csv`（2 场景 × OFF/ON，各 200 步，replay-only，V0N0）：

| scene | cfg | steps | exit | structural_active | active_steps | coherence_terms | shape_terms | loss_is_photometric_only |
|---|---|---|---|---|---|---|---|---|
| office0_mono | **OFF（= L_core）** | 200 | 0 | **0** | **0** | 0 | 0 | **1** |
| office0_mono | ON（对照） | 200 | 0 | 1 | 200 | 200 | 200 | 0 |
| desk3_rgbd | **OFF（= L_core）** | 200 | 0 | **0** | **0** | 0 | 0 | **1** |
| desk3_rgbd | ON（对照） | 200 | 0 | 1 | 200 | 200 | 200 | 0 |

原始读回（`lossaudit/*.log`）：

```
[LossCfg] w_line_coherence=0 w_line_shape_ecc=0 w_line_shape_ori=0 structural_active=0 anisotropy=0.01 line_band_terms=0 (no narrow-band photometric weighting in this tree)
[LossAudit] steps=200 structural_active_steps=0 coherence_terms=0 shape_terms=0 narrow_band_terms=0 w=0/0/0 structural_enabled=0 loss_is_photometric_only=1
[LossCfg] w_line_coherence=0.001 w_line_shape_ecc=0.002 w_line_shape_ori=0.005 structural_active=1 anisotropy=0.01 line_band_terms=0 ...
[LossAudit] steps=200 structural_active_steps=200 coherence_terms=200 shape_terms=200 narrow_band_terms=0 w=0.001/0.002/0.005 structural_enabled=1 loss_is_photometric_only=0
```

ON 组是**正对照**：它证明该审计能检测到"结构损失参与了优化"，因此 OFF 组的 0 不是审计失灵。
`exit_code=0` 全部正常，无孤儿进程、无 `terminate`。

**旁证（未连带关闭的东西仍在工作）**：同一次 OFF 运行中 `[EvtAudit]`/线高斯变换计数正常
（此前回环事件审计 3795 个对象、其中包括线高斯 `line_dir'=Rd` 校验通过），说明线检测、匹配、
点线相机优化、线初始化与增密都还在。

## 6. 论文措辞规则（本轮范围修正要求）

| 允许 | 禁止 |
|---|---|
| "结构正则在已测设置下未获得稳定渲染收益，且带来额外计算成本，因此最终配置不采用"（保留消融表） | "结构正则是本方法的正式组件" |
| "本方法在两个方法的正式比较中（见 §正式对比）……" | "关闭结构损失 ⇒ 整系统优于 Photo-SLAM"（两者不等价） |
| "V/N 为实验性开关，未进入正式版" | 把 V/N 写成正式贡献 |
| 消融结果按"无稳定收益 + 额外成本"描述 | 用 OFF 配置的单个数字替代整系统比较 |
