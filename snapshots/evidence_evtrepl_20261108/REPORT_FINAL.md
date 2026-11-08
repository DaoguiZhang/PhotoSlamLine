# Photo-SLAM-L 本阶段收尾报告

日期：2026-11-07（范围修正版）
版本：**PSL-final** `libgaussian_mapper.so 538f46296c8e33721ade70e042ae2d5f…`、
`libORB_SLAM3.so 5a14981faac97ed9db7cf7d834168db2`
前置证据：`REPORT_REPLAY_AVN.md`（官方验证 / 共同输入重放）、`GATES.md`（8 项门禁）、
`RUN_LEDGER.csv`（逐宿主进程）、`exitreg/`（退出与审计回归）、`lossaudit/`（结构损失不参与优化）、
`FORMAL_PLAN/`（两方法 300 项正式对比状态）

本轮**没有**新增算法、损失或权重扫描；没有重跑 30 条件批次；没有改动任何旧目录、旧 cfg、
旧默认值；**没有**删除结构线损失的实现代码，只是在新冻结配置里把三个有效权重设为 0 并跳过计算。

> **范围修正（本轮唯一的方法定义变更）**：最终 L 版本改为 **L_core = A 类修复 + 线前端 + 点线相机优化
> + 线初始化与增密，但 Gaussian mapping 的结构线损失关闭（三权重 0、不计算），V=0 N=0**。
> 因此 **`L_core-Full` 不再是正式必跑方法**，正式对比为 `O_core` vs `L_core` 两个方法。
> 证据见 §3.4；措辞约束见 §5.3。

---

## 1. 30 次条件运行的账本（三件事分开记）

`RUN_LEDGER.csv`。三列互不替代：`conditions_complete`（映射是否跑完 5 个条件）、
`products_complete`（每个条件的 summary/steps/event_log 是否非空）、`host_exit`（承载进程是否正常退出）。

| 场景 | 种子 | 条件 | 产物 | 宿主退出 | 墙钟 |
|---|---|---|---|---|---|
| office0_mono | 0/1/2 | 5/5 | 完整 | `EXIT_0` | 320 / 316 / 320 s |
| desk3_rgbd | 0 | 5/5 | 完整 | `EXIT_0` | 239 s |
| desk3_rgbd | **1** | **5/5** | **完整** | **`KILLED_SIGKILL`** | **1412 s**（含 18 min 挂起） |
| desk3_rgbd | 2 | 5/5 | 完整 | `EXIT_0` | 233 s |

desk3 seed1 **不写成 exit=0**。它的 30 个检查点行、5 份 `summary.txt`、9670 行事件记录都完整，
所以映射数据可用；但退出状态、墙钟与异常单独标注。
`host_pid` 对 5 个进程是 `not-captured`（当轮没采集 `$!`）——正式调度器已改为必记 PID。
**没有为改退出码重跑这 30 次。**

## 2. 宿主挂起：实测栈、真实等待关系、修复

### 2.1 拿栈的方式
容器 `kernel.yama.ptrace_scope=1` 且 `/proc/sys` 只读 → **无法 attach**。
改为**在 gdb 下启动**（父进程可跟踪子进程），用 watchdog 发 `SIGALRM` 让 gdb 停下来再
`thread apply all bt`。脚本 `probe_exit.sh`。

### 2.2 实测到的东西（不是"54 个线程都在 futex"这种猜测）

第一次探测就抓到一个**可复现的缺陷形态**（58 个栈）：

```
Thread 54 (mapper)  __pthread_kill ← __GI_raise ← abort ← std::terminate
                    ← __cxa_throw ← GaussianMapperLine::runReplay() [clone .cold]
                    ← GaussianMapperLine::run()
Thread 3 (LocalMapping) g2o::HyperGraph::clear ← OptimizableGraph::~OptimizableGraph
                    ← Optimizer::LocalBundleAdjustmentWithLine_Optimization_Plucker_Reg
                    ← LocalMapping::RunWithLine()
main                System::Shutdown() → usleep
```

结论（与"猜"的区别在于有栈）：
1. `runReplay()` 的任何失败路径（坏快照哈希、缺图、开关未生效）**逃出线程函数** →
   `std::terminate` → `abort`。abort 在多线程进程里要停所有线程，而此刻 LocalMapping
   正在 g2o 图析构里；这条路径**既可能直接杀死进程，也可能卡在 terminate/unwind 上**
   —— 这正是"产物都写完了但进程不退出、退出码永远不写"的来源。
2. `Shutdown()` 的等待 `while(!local->isFinished() || !loop->isFinished() || loop->isRunningGBA()) usleep(5000)`
   **无界且无诊断**：一旦某个谓词永久为真，进程就静默停在那里，没有任何信息留下。

### 2.3 修复（`diffs_final/exit_fix_and_audit.patch`，19 hunk）

| 位置 | 修复 |
|---|---|
| `runReplay()` 两个调用点 | `try/catch` → `mapper_exit_code_=4/5` + `[MapperExit] FAILED reason=…`，不再 terminate |
| `System::Shutdown()` | 等待**有界 180 s**；每 10 s 打印 `local_mapping / loop_closing / gba` 哪个未满足；超时置 `mbShutdownTimedOut`、**不 join**、明确报失败 |
| 退出前 | 检查 `hasMappingOperation()`；仍有未消费事件 → `exit_code=6` |
| 例程 | `return mapper->exitCode()`；`Shutdown` 超时 → 打印后 `_Exit(3)` |

**没有**用 `_Exit(0)` 假装成功：成功路径永远是正常 `return 0`；`_Exit(3)` 只在
"已判定超时、join 会 UB" 的失败路径上。

### 2.4 退出回归（`exitreg/summary.csv`）

| 用例 | 期望 | 实测 |
|---|---|---|
| 正常完成（4 条件） | 0、队列空、产物齐、无孤儿 | **rc=0 / orphan=no / terminate=no / products=yes / pending_operations=0** |
| 提前失败（坏重放目录） | 非 0、不 abort、无孤儿 | **rc=4 / orphan=no / terminate=no**（修前是 terminate→abort） |

修复后**在最终二进制（`538f4629…`）上重跑了同一套回归（4 个用例，含回环与合并事件审计）**，
`exitreg/summary.csv` 每行都与当前冻结二进制对应，不与旧哈希混用：

| 用例 | 场景 / 步数 | exit | orphan | terminate | 产物 | 备注 |
|---|---|---|---|---|---|---|
| `normal_exit` | office0_mono / 150，4 条件 | **0** | no | no | yes | `pending_operations=0` |
| `early_fail` | office0_mono / 150，坏快照 | **4** | no | no | no | `FAILED reason=replay: bad snapshot` |
| `audit_loop` | office0_mono / 2000，`CONDS=A` | **0** | no | no | yes | 66 行 `[EvtAudit]` |
| `audit_merge` | desk3_rgbd / 2900，`CONDS=A` | **0** | no | no | yes | 10 行 `[EvtAudit]` |

---

## 3. 事件一致性：逐对象逐字段的数值审计

`PHOTO_SLAM_EVT_AUDIT=1` 在**生产消费点**（`scaledTransformVisiblePointsOfKeyframe` 调用前后）
对每个被移动的对象按闭式解逐字段核对，并检查未被移动的对象是否一位都没变。
判据用 kernel 真正收到的 `(R, t, s)`，不是重算的近似。

### 3.1 回环（office0 seed0，step 1691）

```
[EvtAudit][TOTAL] kfsAudited=4 sampledObjects=80 newlyTransformed=3799
  changedNotTransformed=0 maxDeltaUntransformed=0
  maxErrXyz=1.19209e-07 maxErrRot=1.78814e-07 maxErrLogScale=0
  maxErrAnchor=1.19209e-07 maxErrDir=0 maxErrSigma=4.76837e-07
```
（最终二进制 `538f4629…`，2000 步，`PHOTO_SLAM_REPLAY_CONDS=A`；`newlyTransformed` 在
3142–3799 之间浮动，是 GPU 增密顺序造成的，属已知不确定度，不是审计不稳定。）

* `xyz' = sRp+t`、`q' = q_R⊗q`、`log σ' = log σ + log s`、`anchor' = sRa+t`、
  `line_dir' = Rd`（**仅线高斯**）、`Σ' = s²RΣRᵀ` —— 全部在 float 舍入量级。
* 未选中对象 `maxDeltaUntransformed = 0`：掩码之外没有副作用。
* 这次审计还纠正了我自己第一版判据的错误（原先要求所有对象的 `line_dir_w_` 都旋转）；
  生产代码本来就是"只旋转线高斯"，按类分开判据后误差归零。**没有为了好看改动生产语义。**

### 3.2 合并写回（N）：**应用集合不可复现，契约未验证**

同一导出（desk3 seed1）、同样 5000 步、同一版本族：

| 运行 | 合并事件变换点数 | 变换线高斯 | 被审计 KF 的 `newlyTransformed` |
|---|---|---|---|
| 上一轮 A/V/N 的 A 条件（5 条件同进程，A 排第 5） | **1164** | 7 | — |
| PSL-final 审计 run1（`CONDS=A`） | **0** | 0 | 0（10 个 KF 全为 0） |
| PSL-final 审计 run2（`CONDS=A`） | **0** | 0 | 0 |

原因：资格判据是 `|exist_since_iter − kf_creation_iter| < 1`，要求"高斯出生迭代与该关键帧
出生迭代恰好相差 <1 次"。这是两个**独立演进的计数器**之间的整数巧合，会被
GPU 增密（原子加/规约顺序）带来的微小差异整体翻转：一次选中 1164 个，另一次 0 个。

**因此：N 的通知被生产、被入队、被消费（分支计数可查），但"它修正了受影响区域"这一契约
没有得到验证。** 按规则 N **保留为实验功能**，论文不得声称已实现完整 GBA/合并地图一致性。

### 3.3 事件后 PSNR 下降：只作诊断，且已排除三种解释

* 不是评价参考不同步：评测相机就是被 `setPose()` 更新的同一批关键帧。
* 不是掩码/状态更新错误：§3.1 的逐字段审计误差 ≤2.4e-7、未选中对象零变化。
* 不是"几何规则错"：写回只移动"同龄"子集是**生产实现既有语义**，我没有为了让 PSNR 好看去改它。
* 结论：事件瞬间 PSNR 掉 1.6–3.1 dB 是"只移动子集"的直接后果，属**已知一致性缺口**，
  本轮**只登记、不修**（改它会改变几何契约，需要单独立项与官方路径对照）。

### 3.4 结构线损失：确实不参与最终优化（本轮范围修正的核心证据）

不是为了"改一个 YAML 键"交差：把**解析后**的三个权重在 `run()` 开头打印为 `[LossCfg]`，
并在两条训练路径的损失组装处对“结构项是否真的进了 `loss`”计数为 `[LossAudit]`
（权重全 0 时整块跳过：不分配张量、不启动线损失核、不向 `loss` 加任何东西）。

| scene | cfg | steps | exit | structural_active | active_steps | coherence_terms | shape_terms | photometric_only |
|---|---|---|---|---|---|---|---|---|
| office0_mono | **OFF = L_core** | 200 | 0 | **0** | **0** | 0 | 0 | **1** |
| office0_mono | ON（正对照） | 200 | 0 | 1 | 200 | 200 | 200 | 0 |
| desk3_rgbd | **OFF = L_core** | 200 | 0 | **0** | **0** | 0 | 0 | **1** |
| desk3_rgbd | ON（正对照） | 200 | 0 | 1 | 200 | 200 | 200 | 0 |

* 正对照（ON）说明审计**能**检到参与，所以 OFF 的 0 不是“审计看不出来”。
* `line_band_terms=0`：本树**不存在**窄带光度加权项（`grep band/narrow` 无命中），等价于已关闭。
* **未连带关闭**：线检测、线匹配、点线相机优化、线初始化与既定增密；线元数据/张量结构
  （`is_line_`、`line_dir_w_`、`line_anchor_`）全部保留，`clone/split` 与初始化仍依赖它们。
  `lambda_anisotropy=0.01` 是通用各向异性惩罚，**不属于**线结构正则，保留。
* 历史保留：旧 Full/OFF 实验、结构损失实现、`PSL-fullmap-exp-v1` 封存协议一律未改；
  本轮**没有**重新证明结构损失有没有用，旧结论原样有效。

### 3.5 自查发现并修掉一个**审计工具缺陷**（不是生产缺陷）

在最终二进制上重跑退出回归时，`[EvtAudit][TOTAL]` 突然报出 `maxErrSigma=8.04e+19`；
加上"最差对象"归因后重跑变成 `inf`，而同一批对象的 log-scale 完全正常
（`−3.7/−3.3/−4.2` 等）、`maxErrLogScale=0`、`maxErrXyz=1.2e-7`。数值量级与正常尺度**矛盾**，
说明问题在审计代码本身：

```cpp
// 错（旧审计）—— asDiagonal() 包装了一个惰性表达式，包装对象比它引用的临时量活得久
return Rq * d.cwiseAbs2().asDiagonal() * Rq.transpose();
// 对（现在）—— 先把 d² 和 D 落地成有名对象
Eigen::Vector3f d2 = d.cwiseAbs2();
Eigen::Matrix3f D  = d2.asDiagonal();
return Eigen::Matrix3f(Rq * D * Rq.transpose());
```

修前/修后（同场景、同 2000 步、同审计参数）：

| | maxErrSigma |
|---|---|
| 修前（观察到的两个值） | `8.0412e+19`、`inf` |
| 修后 | **`1.49012e-08`**（float 舍入量级） |

**因此必须更正一句话**：上一轮报告的 `maxErrSigma=0` **不能**算作协方差一致性的证据——
那次只是那个缺陷恰好没读出垃圾值。现在协方差字段的证据是修后实测的 `1.49e-08`，
并且审计会同时打印 `sigmaWorstIdx / sigmaWorstLogScale0 / sigmaWorstLogScale1`，
任何异常都能落到具体对象上，而不是只给一个汇总数。

生产代码**没有**因此改动：只改了审计里的一个 lambda，以及增加了一行诊断输出。
本条如实保留，因为它属于"不能把可信度建立在可能失灵的仪器上"。

---

## 4. V/N 研究结论（封存）

| 项 | V（体素"当前存活"去重） | N（GBA/合并写回） |
|---|---|---|
| 是否被触发 | desk3 全部种子：7–9 次重建 / 168k–311k 点 / 39.6–72.2 ms（占事件处理 3.6–3.8%）；**office0 全部未触发**（live 从未把线采样送进任何 op） | office0 种子 0、2（GBA+回环），desk3 全部（合并）；office0 种子 1 未触发 |
| 代价 | 有确定维护成本 | 无可测开销 |
| 画质 | 3 种子 5000 步：desk3 `V−A = −0.083 ± 0.147`，office0（未触发）`+0.080 ± 0.096`；**都落在同设置重复波动内**（`A_repeat−A` 标准差 0.128/0.159，单次最大 +0.204） | `N−A = −0.019 ± 0.250`（desk3）/ `−0.090 ± 0.050`（office0）；同样在噪声内 |
| 契约 | — | **未验证**（§3.2 应用集合不可复现） |

**V 本轮结论**：有实际维护成本，画质作用未确认，office0 机制未覆盖。
**N 本轮结论**：通知被消费并实际更新 Gaussian（有分支计数与逐字段审计），
但完整一致性与画质收益不能仅凭更新计数证明，且应用集合不稳定。

---

## 5. 最终方法采用哪些模块 / 哪些能力未覆盖

### 5.1 正式版 = **L_core（V0N0，结构线损失关闭）**：A 类修复 + 原点线管线

| 模块 | 是否进入正式版 | 依据 |
|---|---|---|
| A1 `insert_rot_to_rots` | **是** | O_raw 旋转矩阵误差 0.796；O_core 24/24 |
| A2 全局 `scaling_ += log s` | **是** | O_raw σ 误差 12.4 / Σ 387 |
| A3 回环路径 σ 同步 | **是** | O_raw 完全不更新 σ（σ 误差 2.33） |
| A4 `line_dir_w_` 用 R | **是** | L0 方向误差 2.00；本轮审计 maxErrDir=0（线高斯） |
| A5 `xyz_init_` 随变换（掩码） | **是** | 审计 maxErrAnchor=1.2e-7 |
| A6 Adam 空状态 | **是** | O_raw case7 3/3 SIGSEGV |
| A7 体素提交时序 + 同批去重 | **是** | 与 V 的策略部分解耦；时序部分是正确性 |
| A8 Voxel Report 标签/闭合 | **是** | 日志缺陷 |
| A9 非法尺度拒绍 | **是** | O_raw 写出 NaN |
| 线前端/点线相机优化/线初始化/增密 | **是** | 范围修正明确要求**不得**关闭 |
| **结构线损失（三权重）** | **否（本轮范围修正：全 0 且不计算）** | §3.4：`active_steps=0`、`photometric_only=1`；在已测设置下未获得稳定渲染收益且增加计算成本 |
| 窄带光度加权 | 否（本树无此项） | `line_band_terms=0` |
| V（当前存活去重） | **否（实验版）** | §4 |
| N（写回通知） | **否（实验版）** | §4 契约未验证 |
| 退出修复 / 诊断 / 损失审计 | **是（工程）** | §2、§3.4；不计入方法创新 |

### 5.2 尚未覆盖

1. event 后 1.6–3.1 dB 跌落（生产实现一致性缺口）——只登记，未修。
2. N 的应用集合稳定性（`|Δiter| < 1` 整数巧合）——未找到既能保语义又能稳定的判据。
3. office0 上的 V/N 机制覆盖（该数据集不触发）。
4. 合并事件"创建新关键帧"分支（重放里 `created=0`）。
5. 重放压缩迭代轴（office0 1.32–1.35×，desk3 3.6–6.0×）导致"同龄集合"比 live 小 1.7–2.2×。
6. 正式对比（`O_core` vs `L_core` 两方法 × 30 格 × 5 次 = **300 项**）**NOT_STARTED**
   （`FORMAL_PLAN/STATE.md`；两个前置：`O_core`/`L_core` 共同条件待拍板、≈25 h 墙钟与产物存放位置）。
7. 结构线损失的“报除”结论只在**已测设置**（当前预算/步调/三数据集）下成立，
   未在其他设置下重新扫描（本阶段明确不做新损失/权重扫描）。

### 5.3 论文措辞约束（本轮范围修正要求）

| 允许写 | 不得写 |
|---|---|
| “结构正则在已测设置下未获得稳定渲染收益，且额外增加计算成本，因此最终配置不采用”（保留消融表） | “结构正则是本方法的正式组件” |
| “本方法相对基线的整系统结论以两方法正式比较为准（当前 NOT_STARTED）” | “关闭结构损失 ⇒ 整系统优于 Photo-SLAM”（两者不等价） |
| “V/N 为实验性开关，未进入正式版” | 把 V/N 写成已验收的贡献 |
| 消融按“无稳定收益 + 额外成本”描述 | 用 OFF 配置的单个数字替代整系统对比 |

---

## 6. 主任务状态

| 问题 | 回答 |
|---|---|
| 工程修复是否验收 | **是**：门禁 1–4、6、7、8 PASS；门禁 5 部分通过（对象级数值 PASS，集合级 NOT_VERIFIED，但因 N=0 不再阻塞） |
| 正式实验是否完成 | **否**：两方法 300 项 **NOT_STARTED**，两个前置条件未满足（`FORMAL_PLAN/README.md` §3/§5） |
| V/N 研究结论 | §4，两者均**未确认画质收益**，N 契约未验证；正式版 V=N=0 |
| 最终方法采用哪些模块 | §5.1：A1–A9 + 线前端/点线相机优化/线初始化/增密 + 退出修复；**结构线损失关闭**（三权重 0、不计算）；V/N 关闭 |
| 尚未覆盖的能力 | §5.2（含正式对比未启动） |

**因此本阶段的主任务（含正式对比）尚不能宣布全部完成。**
已完成并可验收的是：A 类修复、官方 24 次验证、30 次重放与账本、退出修复与回归、
事件链的逐字段审计、结构线损失“确实不参与”的读回与审计、以及正式对比的全部前置准备
（两方法 300 项计划、调度器、L_core 配置）。
