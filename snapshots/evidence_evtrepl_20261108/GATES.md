# 本阶段验收门禁（一次性，逐项 PASS / FAIL / NOT_COVERED）

版本：**PSL-final**（= PSL-replay + 退出修复 + 事件审计 + 有效参数打印）
证据根目录：`/workspace/code/PSL-evtrepl`

| # | 门禁 | 结论 | 证据 |
|---|---|---|---|
| 1 | 核心数学变换与优化器契约 | **PASS** | 46/0（`test_dedup_transform`，A 类修复后）；官方 8 case → O_core 24/24；最终二进制上的事件审计：xyz ≤1.2e-7、rotation ≤1.8e-7、log-scale 0、anchor ≤1.2e-7、**direction 0**、covariance ≤4.8e-7（修掉审计自身的 Eigen 临时量缺陷后实测，见 §自查），未更新对象 `changedNotTransformed=0 / maxDeltaUntransformed=0` |
| 2 | 官方公共修复验证 | **PASS** | `ledger/ledger.csv`：O_raw 6 OK / 15 FAILED / 3 SIGSEGV，O_core 24 OK；每 case 独立进程 + `[lib] mapped=` + sha256 |
| 3 | 正常退出、提前失败及资源清理 | **PASS** | `exitreg/summary.csv`（最终二进制 `538f4629…`，4 个用例）：正常 `rc=0 / orphan=no / terminate=no / products=yes / pending_operations=0`；坏输入 `rc=4 / orphan=no / terminate=no`（修前是 `std::terminate`→abort） |
| 4 | 事件生产—队列—消费链 | **PASS** | 30 次重放 `events_applied = events_in_stream`；消费端分支计数逐事件落盘；退出时 `pending_operations=0` |
| 5 | 事件前后对象集合及数值一致性 | **PARTIAL → 该能力标 NOT_VERIFIED** | 回环：审计 4 个 KF / 3795 个对象，字段全部吻合闭式解。**合并写回：同一输入、同一 5000 步，一次移动 1164 个对象/7 个线高斯，另两次（同一二进制、`CONDS=A`）移动 0 个** → 应用集合不可复现，见 §4 |
| 6 | 输入、产物、版本与实际加载身份 | **PASS（30 次运行部分标注）** | `report/input_identity.csv`（同种子 5 条件哈希一致）、`RUN_LEDGER.csv`（区分"映射完成/产物完整/宿主进程退出"，desk3 seed1 = `KILLED_SIGKILL`）、`ARTEFACT_HASHES.txt`、`[lib] mapped=` |
| 7 | 剩余能力边界与论文声明一致 | **PASS（须按 §4 措辞）** | 见 `REPORT_FINAL.md` §5 声明清单；**新增**：结构正则不得再写为正式组件 |
| 8 | 结构线损失确实不参与最终优化（范围修正项） | **PASS** | `lossaudit/summary.csv`：OFF（= L_core）`structural_active=0 / active_steps=0 / coherence_terms=0 / shape_terms=0 / loss_is_photometric_only=1`；ON 正对照 `1 / 200 / 200 / 200 / 0`；2 个场景一致，`exit_code=0` |

## 逐项说明

### 门禁 1（PASS）
- 生产链路审计不是"更新计数"，而是**逐对象逐字段**与闭式相似变换比对：
  `xyz'=sRp+t`、`q'=q_R⊗q`、`log σ'=log σ+log s`、`anchor'=sRa+t`、`line_dir'=Rd`（**仅线高斯**）、
  `Σ'=s²RΣRᵀ`。误差见上表，全部在 float 舍入量级。
- 未选中的对象**一位都没变**（`maxDeltaUntransformed=0`），说明掩码以外没有副作用。
- 该审计发现并修正了我自己第一版的判据错误（把 `line_dir_w_` 当作对所有对象都旋转）；
  生产代码本来就是"只旋转线高斯"，审计按类别分开判据后误差归零。
- **自查修正（重要）**：重跑时审计报出 `maxErrSigma=8.04e+19` 与 `inf`，而同批对象 log-scale 完全正常；
  定位为审计里 `d.cwiseAbs2().asDiagonal()` 让包装对象比它引用的惰性临时量活得久，读到垃圾。
  改为先落地 `d2`、`D` 后实测 `maxErrSigma=4.8e-7`。**因此上一轮的 `maxErrSigma=0` 不得引用为证据**，
  现以修后值为准（生产代码未改，只改审计 lambda + 增加最差对象诊断）。

### 门禁 2（PASS）
- 见 `ledger/`。另：上游 `GaussianOptimizationParams` 的 `opacity_lr_(opacity_lr_)` 自初始化
  缺陷已定性（`repro_overflow`：修前 60 次 1 次 abort，修后 60/60 通过），
  **本轮没有把它算作算法收益，也没有改动任何旧 cfg**。

### 门禁 3（PASS）
修复内容（`diffs_final/exit_fix_and_audit.patch`，19 个 hunk）：
1. `runReplay()` 的两处调用点加 `try/catch`，失败→`mapper_exit_code_=4/5`，
   不再逃逸成 `std::terminate`/`abort`（这是 desk3 seed1 挂起的最可能形态：
   线程停在 `__cxa_throw → std::terminate` 上，54 个线程全部 park）。
2. `System::Shutdown()` 的等待从"无界 + 无诊断"改为
   **有界（180 s）+ 每 10 s 报告哪个谓词未满足**（`local_mapping / loop_closing / gba`）；
   超时置 `mbShutdownTimedOut`，**不 join、不假成功**，主程序打印后以 **3** 退出。
3. 退出前检查 `hasMappingOperation()`；若还有未消费事件→ `exit_code=6`。
4. 例程把 `mapper->exitCode()` 作为进程退出码返回。
**没有**使用 `_Exit(0)` 绕过；`_Exit(3)` 只出现在**已判定失败**的路径上（join 未完成时不能正常返回）。

### 门禁 4（PASS）
- 30 次运行：`events_applied == events_in_stream`（见 `report/input_identity.csv`）。
- 退出时 `pending_operations=0`（`exitreg/*.log` 的 `[MapperExit]` 行）。
- 消费端分支计数（corrected / skipped / pointsTransformed / lineGaussiansTransformed /
  voxel rebuilds）逐事件写入 `report/all_events.csv`（9670 行）。

### 门禁 5（PARTIAL，相关能力 NOT_VERIFIED）
- **已证**：事件被消费；被选中的对象其 6 个字段全部符合闭式解；未选中对象不变。
- **未证**：被选中的**集合**本身稳定。同一导出、同一 5000 步、同一二进制：
  | 运行 | 合并事件变换点数 | 变换线高斯 |
  |---|---|---|
  | PSL-replay 5 条件 A（A 为第 5 个条件） | **1164** | 7 |
  | PSL-final 审计 run1（`CONDS=A`） | **0** | 0 |
  | PSL-final 审计 run2（`CONDS=A`） | **0** | 0 |
  原因是资格判据 `|exist_since_iter − kf_creation_iter| < 1`：要求"高斯出生迭代与该关键帧
  出生迭代恰好相差不到 1 次"，这是两个独立演进的计数器之间的**整数巧合**，
  会被 GPU 增密（原子加/规约顺序）的微小差异整体翻转。
  **因此 N 的"修正受影响区域"契约未被验证**，N 只能作为实验功能。

### 门禁 6（PASS，含标注）
`RUN_LEDGER.csv` 把三件事分开：`conditions_complete / products_complete / host_exit`。
desk3 seed1：5/5 条件完成、产物完整、`host_exit=KILLED_SIGKILL`（操作员 18 分钟后 kill，
**不是 exit 0**），墙钟 1412 s（正常 desk3 约 233 s）。
`host_pid` 列对 5 个进程标 `not-captured`（当轮未采集 `$!`）；正式批次调度器已改为记录 PID。

### 门禁 7（PASS）
见 `REPORT_FINAL.md` §5：哪些模块进了正式版、哪些结论只能说"未确认"。
**措辞约束（范围修正）**：最终配置不含结构线损失，因此论文**不得**再把结构正则写成正式组件；
保留消融而非删表，描述为"在已测设置下未获得稳定渲染收益，且额外增加计算成本"；
也**不得**把"关闭结构损失"等同于"整系统优于 Photo-SLAM"——整系统结论以两方法正式比较为准。

### 门禁 8（PASS）——结构线损失不参与优化
- 判据不是"YAML 写了 0"，而是**解析后权重 + 损失组装点计数**：`[LossCfg]` 打印 `opt_params_`
  里真正被使用的三个权重；`[LossAudit]` 在每个训练迭代计数，只有权重为正时结构项才进入 `loss`。
- OFF（`cfg_v2/*.nostruct.yaml`，即 L_core）：`w=0/0/0`、`structural_active=0`、
  `structural_active_steps=0`、`coherence_terms=0`、`shape_terms=0`、`loss_is_photometric_only=1`。
- ON 正对照（原 cfg，0.001/0.002/0.005）：`1 / 200 / 200 / 200 / 0` → 审计能检到参与，
  故 OFF 的 0 不是"审计看不出来"。
- 范围边界：**未**连带关闭线检测、线匹配、点线相机优化、线初始化与增密；线元数据/张量结构保留。
  `lambda_anisotropy=0.01` 是通用各向异性惩罚，**不属于**线结构正则，保留。
