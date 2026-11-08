# 本轮候选修复结论（阶段性）

LOOP漏更正路径已定位并做最小修复；同一 LOOP 事件的相机/地图一致性和可重放性有强证据，但后续非零 GBA 的 100 步验收仍缺输入。

- 候选补丁：`evidence/loop_coord_fix_r5/candidate/loop_coord_fix.patch`
- 基线：`ec85e80886`（`feat/mapper-budget-v1`，生产树 dirty 1715）
- 配置哈希、二进制哈希：见 `candidate/manifest.sha256`
- 复现脚本：`scripts/repro_loop_coord_fix.sh`

已完成的强证据（真实 LOOP）：

- 同事件零步（r5，普通 PSNR）：before `30.58399487`，legacy camera-only
  `23.47381488`（`−7.110179981`），fixed map+camera `28.98292383`
  （相对修正前 `−1.60107104`）。
- 干净 LOOP 回归：118 KF，117 个位姿变化，修正 `373284` 行，
  `missingEventRows=4927`，行数不变；GS PSNR n=180 mean `29.2972`，last `25.8999`。
- Legacy vs Fixed 100 步（r2/recovery，GS PSNR）：step0 `28.1116` vs
  `28.9127`（Fixed−Legacy `+0.8011`），step100 `31.4499` vs `31.5506`
  （`+0.1007`）。这两个是 **Fixed 相对 Legacy 的收益，不是相对修正前残差**；
  四 arm 相机指纹同为 `340f17cbe078514f`。
- 幂等：`[LoopReplay] evt=1 same_moved=0 same_stale=373356
  older_moved=0 older_stale=373356`。
- 覆盖（r2/recovery）：`refSceneNotEvent=12000`、`refAbsentScene=0`、
  `refUnassigned=0`；missing 主要是 payload 未包含参考 KF。r5 单事件
  `missingEventRows=3801`，两者是不同事件，不能混用。fid56 有 11667 个
  missing-ref 行可见；fid55 不在 24 视图网格内。
- **新行 stamp/replay 行级测试已通过**：event 3 后插入新行，stamp=`3`；
  回放 event 1 不移动；event 4 移动。输出
  `[GbaContract] ... stamp=1 new_replay=1 new_event=1 : PASS`。
- **LoopKey 顺序无关测试已通过**：
  `[LoopKeyContract] order=1 map=1 match=1 : PASS`。

尚未验收：

- 跨 map 的完整 payload 端到端回归（当前有 `mapGen` 检查、全 scene 缺失保护
  和 `map_matches` helper 测试）。
- 非零后续 GBA 的 PoseOnly vs PoseAndMap 100 步：输入不齐，未做。r5 只有
  step0 渲染指标；r2/r4 的 GBA `world_deltas` 为零；r3/r5/r6 无可用非零
  事件。缺失输入清单见 `../gba_inventory.md`。没有静默重置优化器补跑。

仅诊断、不作验收：

- room1 A/C 已跑一对（同 fix/config/GBA-disabled，仅 backend 不同）：A 无
  LOOP/GBA，C 有 1 次 LOOP（120 KF，116 位姿变化，371525 行移动）和 1 次
  零位移 GBA。A 183 KF / native mean 30.6536 / last 28.3001；C 182 KF /
  29.9071 / 24.7993。按 common `img_name` 的 17 帧原生 GS A−C 为
  `+1.0327 dB`（median `+0.9844`，A 高 13/17）；最后共同帧
  `frame001625` 为 A `28.2450`、C `27.9186`。因轨迹不同，不能把全量差异
  归因于修复或 backend。证据见 `../ac_pair/`。
- 可变拓扑严格 120 步（6599 状态、完整 Adam/相机/120 帧序列、每臂恢复实际
  CPU/CUDA RNG 状态、A0/A1/C 统一强制重建 optimizer）：6600 的完整
  prune/clone/split 集合和 pre-update fingerprint 均相同；step2 才出现
  非 RNG 参数分叉，A0/A1 自身也有；6700 出现首次拓扑分叉。step120 GS PSNR 为
  `28.5705/28.6052/28.6110`，C−A0 `+0.0405`、C−A1 `+0.0058`、A0−A1
  `−0.0347 dB`。该单次严格重放仍只支持“未观察到 C 持续降低质量”，
  不改阈值。证据见 `../topology_divergence_r9/REPORT.md`。

本结论不使用 pre-event A0/A1/C 100 步作为“修复后稳定”的证明。

## 一句话状态

- 已修并验证：LOOP 的相机/地图一致性、事件版本幂等、新行 stamp、顺序无关 key。
- 可变拓扑严格 120 步：6600 完整决策相同，step2 非 RNG 参数分叉（A0/A1 也有），
  6700 首次拓扑分叉；C 未出现持续退化。
- 未完成：非零 GBA 100 步（缺事件时刻状态）；跨 map 端到端回归。
- A/C 在线全量差是诊断，不是修复效果的因果验收。
