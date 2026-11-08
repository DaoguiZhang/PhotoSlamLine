# 版本冻结记录（本轮）

日期：2026-11-07
根目录：`/workspace/code/PSL-evtrepl`（新实验全部写在这里，旧目录一个字节都没动）

## 1. 冻结了什么

| 代号 | 目录 / 产物 | 含义 | 关键身份 |
|---|---|---|---|
| **O_raw** | `/workspace/code/PSL-review/O_raw` | 官方 `f8bfb2f` + **仅构建兼容**补丁 | `libgaussian_mapper.so` `0396c6d327dfdbce11fd5ef3754c0487` |
| **O_core** | `/workspace/code/PSL-review/O_core` | O_raw + 公共工程修复（A 类） | `libgaussian_mapper.so` `4293b9d7349f8bde2436097e5a278f83` |
| **L_core** | `PSL-replay` 的 `V0N0` 条件 | Photo-SLAM-L + A 类修复，V/N 关 | 同一二进制，见下 |
| **L_V** | `PSL-replay` 的 `V1N0` 条件 | L_core + V | 同上 |
| **L_N** | `PSL-replay` 的 `V0N1` 条件 | L_core + N | 同上 |
| **L_VN** | `PSL-replay` 的 `V1N1` 条件 | L_core + V + N（= 当前 L1 行为） | 同上 |
| **PSL-replay** | `/workspace/code/PSL-replay` | 本轮唯一使用的 L 二进制 | `libgaussian_mapper.so` `d82b9ccae0368d72ea0fd52118884f2a`（2026-11-07 16:40 构建；批次 16:43 启动，构建时间早于全部运行） |
| **L0（历史）** | `/workspace/code/PSL-fullmap-lock-v1` | 300 次历史实验，**未改动** | `98cc87928caa592823650048aa78d194` |
| **L1（上一轮）** | `/workspace/code/PSL-dedupfix` | 上一轮工程修复树，**未改动** | `240789f5cd96cfdf542b96b251236bc4` |
| **PSL-lavn（上一轮）** | `/workspace/code/PSL-lavn` | 上一轮 A/V/N 用的树，**未改动** | `609fcb3181124cd3f2e1abf60ceb9fae` |

**L_core / L_V / L_N / L_VN 不是四个构建**，而是同一个 `PSL-replay/lib/libgaussian_mapper.so`
在运行期由两个开关取不同值得到的四种行为，避免"不同构建"造成的混淆：

```
PHOTO_SLAM_DEDUP_VOXEL=0|1   (V)   1 = 当前存活去重 + O(1) 失效 + 惰性重建
PHOTO_SLAM_DEDUP_NOTIFY=0|1  (N)   1 = 发出并消费 GlobalBAWriteback / MapMerge
```

生产路径仍按环境变量读取；本轮离线重放为了在**一个进程里**跑五种条件，增加了
`DedupVoxelOverrideSlot()/DedupNotifyOverrideSlot()`（`-1` = 回到环境变量）。
`runReplay()` 在每种条件开始时把值写进 override，**并立即回读 `DedupVoxelLive()` /
`DedupNotifyEnabled()` 断言取值生效**，不生效就抛异常终止：

```cpp
ORB_SLAM3::DedupVoxelOverrideSlot() = cond.v;
ORB_SLAM3::DedupNotifyOverrideSlot() = cond.n;
const bool v_on = ORB_SLAM3::DedupVoxelLive();
const bool n_on = ORB_SLAM3::DedupNotifyEnabled();
if (v_on != (cond.v != 0) || n_on != (cond.n != 0)) throw ...;
```

生效证据不是打印环境变量，而是**分支计数**（见 `report/mechanism.csv`）：

* V：`scene_->voxel_stats()` 的 `invalidations / rebuilds / rebuild_points / rebuild_ms`
  在每个条件下从 0 重新统计（`resetVoxelStats()`），逐事件写进 `event_log.csv`。
* N：消费端 `[GeometryWriteback] … corrected= created= skipped= pointsTransformed=
  lineGaussiansTransformed= nonFinite=` 的真实分支计数被回填到 `event_log.csv`。
* 若某次运行 N 的目标事件根本没发生，CSV 里对应列全为 0，报告里记为"机制未覆盖"，
  **不用它证明 N 有效或无效**。

## 2. V/N 关闭时 SLAM 自身行为不变

N=0 只跳过 `LoopClosing` 里的两次 `pushMappingOperation`（GBA 写回、地图合并写回）。
`RunGlobalBundleAdjustmentWithLine`、`MergeLocal`、`MergeLocalWithLine` 本身照常执行，
日志里仍能看到 `Full GBA`、`Num exec`、`merging maps finished`。

V=0 只把体素去重退回旧语义（历史采样键、接受即提交、不失效、不重建），不影响
ORB-SLAM3 任何部分。

## 3. 不能安全拆开的地方（故未强行拆）

| 依赖 | 说明 |
|---|---|
| A2 ↔ A3 | 全局尺度与回环尺度必须使用同一 `+= log(s)` 语义；拆开会让两条路径对"σ 如何随相似变换变化"给出不同答案。 |
| A6（Adam 空状态）↔ 所有替换路径 | 任何 `replaceTensorToOptimizer` 调用都依赖它；即使 V=N=0 也保留，否则仍可能段错误（O_raw case7 已实测 3/3 段错误）。 |
| V 与 A7 | V 的"提交时序"部分本身就是 A7。V=0 只回退"历史→当前 + 惰性重建"这一策略部分，不会制造不一致状态（只是去重语义更弱更旧）。 |
| N 与 A5 | N 的消费路径用 `xyz_init_` 同步锚点；N=0 时该路径不执行，A5 仍服务回环路径。 |

## 4. 源补丁

| 文件 | 内容 |
|---|---|
| `diffs/O_raw_compat.patch` | 官方源码在 torch 2.12 下的构建兼容（6 处 `c10::guts::to_string`） |
| `diffs/O_core_math.patch` | 公共工程修复（A 类） |
| `diffs/replay_harness.patch` | `PSL-lavn → PSL-replay` 的全部改动（导出器 + 重放器 + 开关 override + 计数回填），1296 行 |
| `diffs/review_official.cpp` | 官方测试源（含 `[lib]` 加载身份打印） |
| `diffs/repro_overflow.cpp` | 异常最小复现源 |
| `diffs/run_official_ledger.sh` | 官方测试账本生成脚本 |
