# 磁盘与仓库整理审计（SPACE_AUDIT）

日期：2026-11-08　执行者：容器内 root（`zdg-MS-7E20`）
原则：**先盘点 → 再归档 → 远端核验 → 恢复测试 → 最后才白名单清理**。
本轮**没有**使用 `git clean -fdx`、`rm -rf PSL-*`、`docker system prune`，也没有移动/改名任何活跃工程目录。

---

## 0. 结论摘要

| 问题 | 结论 |
|---|---|
| 空间大户 | `/workspace/code`（**638 GB**，独立 ext4 挂载）与 `/tmp`（**66 GB**，overlay 可写层） |
| 容器内能不能解释 `df` | **不能**。容器可见约 **733 GB**，而设备已用 **1.7 TB**；差额在宿主侧，容器内没有 docker CLI / socket，无法只读查看 |
| 已释放 | **20.87 GB**（可用空间 24.7 GB → **44.4 GB**） |
| 本轮最安全的可释放项 | garden 数据集 zip（18.52 GB，已做逐文件核对）、两个旧 `build_*` 对象目录（2.16 GB）、`Train.zip`（0.19 GB） |
| 明确**没有**释放的大块 | 数据集 285.8 GB、`Temp/` 80.9 GB、`results/` 52.3 GB、`.git` 19.7 GB、旧备份 tar+bundle 35.9 GB |
| 另发现的可释放机会（未执行，需你拍板） | SEGS-SLAM 与 Photo-SLAM 下 **Replica 11.69 GB + RoverData 14.91 GB 完全同名同尺寸重复**（合计 26.6 GB）；旧 worktree 约 1.8 GB |
| 正式 300 项实验预算 | 按封存协议自己 300 次运行的实测：**56.9 MB/次**，300 次 ≈ **16.7 GB** → 当前 44 GB **够用**，但只剩 ~27 GB 余量 |
| 是否已上传 Git | 源码/配置/脚本/报告/小 CSV：**是**（分支 `source/space-audit-20261108`，标签 `archive-20261108-r2`）；数据集/PLY/PNG/checkpoint/二进制：**否** |

---

## 1. 度量口径（避免误读）

* `df -hT /` 报的是 **overlay 底层设备** `nvme0n1p2`（1.8 T），不是本容器独占配额。
* `/workspace/code` 是**另一个 ext4 挂载**，同一块设备：
  `/dev/nvme0n1p2 /workspace/code ext4 rw,relatime 0 0`（见 `/proc/mounts`）。
  因此两处 `df` 数字相同，**不能相加**。
* 文件级统计用 `find /workspace/code -xdev -type f -printf '%s\t%p\n'`
  （1 685 427 个文件），再按目录/扩展名聚合；**不重复计数，也不跨挂载点**。
* 没有历史容量快照，本轮**不声称**"某天突然增长了 X"。

### 1.1 df / inode

```
Filesystem     Type     Size  Used Avail Use% Mounted on
overlay        overlay  1.8T  1.7T   44G  98% /
/dev/nvme0n1p2 ext4     1.8T  1.7T   44G  98% /workspace/code

Filesystem     Inodes   IUsed   IFree IUse% 
overlay        122028032 5056054 116971978 5%    ← inode 不紧张
```

### 1.2 为什么容器内统计解释不了 df

| 范围 | 实测 |
|---|---|
| `/workspace/code` | 638 GB |
| `/tmp` | 66 GB（清理后 ~63 GB） |
| `/usr` | 19 GB |
| `/root` | 7 GB（其中 `.vscode-server` 5.4 GB） |
| `/opt` ` /var` `/srv` `/home` … | ≈ 4 GB |
| **容器可见合计** | **≈ 733 GB** |
| 设备已用 | **1 700 GB** |
| **差额** | **≈ 970 GB，不在本容器可见挂载内** |

容器内 `command -v docker` → 无；`/var/run/docker.sock` → 不存在。
**因此本轮不提供 `docker system df -v` 的输出**（拿不到就不写）。
需要你在宿主机执行（只读）：

```bash
# 宿主机（不要在容器内操作 /var/lib/docker）
docker system df -v
docker ps -a --size --format 'table {{.ID}}\t{{.Names}}\t{{.Size}}'
sudo du -xhd1 /var/lib/docker /var/lib/containerd 2>/dev/null | sort -h | tail -20
sudo journalctl --disk-usage
sudo find / -xdev -type f -size +20G -printf '%s %p\n' 2>/dev/null | sort -nr | head -20
```

---

## 2. 一级容量统计

### 2.1 按项目（`/workspace/code`，GB）

| 路径 | 占用 | 说明 |
|---|---|---|
| `/workspace/code/Photo-SLAM` | **276.31** | 数据集 + 官方基线的 `Temp/` 渲染输出 |
| `/workspace/code/SEGS-SLAM` | **75.99** | 数据集（与 Photo-SLAM 重复 26.6 GB）+ `Temp/` |
| `/workspace/code/Photo-SLAM-L` | **60.29** | 主工作仓库（build 6.1 G、Temp 21 G、.git 19.2 G、datasets） |
| `/workspace/code/Photo-slam-l-readme` | **37.14** | 早期工作副本 + `loop_results` |
| `/workspace/code/PSL-pml-ablation` | **27.58** | 消融实验（`camdiag_20261028` 17.9 GB） |
| `/workspace/code/PSL-perf` | **22.85** | worktree（`experiments/` 17.9 GB、`abl_out/` 150 MB） |
| `/workspace/code/PSL-lineband-multi-results` | **19.96** | 线带多场景结果 |
| `/workspace/code/PSL-verify` | **18.81** | 验证树（build/lib/结果） |
| `/workspace/code/Photo-SLAM-L_backups` | **17.91** | 单个 `history.bundle`（17.9 GB） |
| `/workspace/code/PSL-fullmap-exp-v1` | **16.74** | 封存协议 + 300 次历史运行（16.67 GB） |
| `PSL-lineband-scene-results` / `-densify-formal-results` / `-lineband-formal-results` / `-fgm-ablation` | 5.24 / 4.50 / 4.42 / 3.85 | 结果目录 |
| `psl-initdiag` / `PSL-finalstate` / `PSL-diag2` / `PSL-initb` / `PSL-statfix` / `PSL-fgm-formal-results` / `PSL-evtrepl` / `PSL-review` | 2.56 / 1.56 / 1.56 / 1.54 / 1.53 / 1.41 / 1.29 / 1.27 | 诊断与证据 |
| `PSL-final` / `PSL-replay` / `PSL-fullmap-lock-v1` | 0.39 / 0.38 / 0.38 | **冻结版本（体积小，必须保留）** |

### 2.2 按类别

| 类别 | 占用 | 备注 |
|---|---|---|
| `*/datasets/` | **285.8 GB** | 其中 83.2 GB 为 `.zip`（本轮删掉 18.7 GB 后余 64.5 GB） |
| `*/Temp/` | **80.9 GB** | 运行渲染输出（PNG/PLY） |
| `*/results/` | **52.3 GB** | 实验产物 |
| `*/.git/` | **19.7 GB** | 主要是 Photo-SLAM-L（19.2 GB / 14 个 pack） |
| `*/build*/` | **13.5 GB** | 可重建 |

### 2.3 按扩展名（全 `/workspace/code`）

```
227.76G  452515 .png      98.19G 1053147 .jpg      86.91G    3957 .ply
 83.24G      11 .zip      22.55G   47553 .txt      19.57G      14 .pack
 18.22G       7 .mp4      17.91G       1 .bundle   12.44G      18 .tgz
 11.48G    7254 .o         7.02G     511 .pt         6.95G     522 .so
```

`.txt` 22.55 GB 主要来自 `used_times/keyframe_used_times.txt` 与运行日志，**不是源码**。

---

## 3. 空间大户表（逐项）

| # | 绝对路径 | 占用 | 类型 | 活跃 | 唯一 | 已有备份 | 建议 | 预计释放 |
|---|---|---|---|---|---|---|---|---|
| 1 | `/workspace/code/Photo-SLAM/datasets/campus_large_2023-07-20.zip` | 44.80 GB | 数据集压缩包 | 否 | **是（未解压）** | 否 | **保留**：解压才算冗余 | 0 |
| 2 | `/workspace/code/Photo-SLAM/datasets/campus_small_2024-02-19.zip` | 18.33 GB | 同上 | 否 | **是（未解压）** | 否 | **保留** | 0 |
| 3 | `/workspace/code/Photo-SLAM/datasets/garden_large_2024-05-29_1.zip` | 18.52 GB | 数据集包 | 否 | 否（已解压） | 否 | **已删**（逐文件核对通过） | **18.52** |
| 4 | `/workspace/code/Photo-SLAM-L_backups/20261013_125650/history.bundle` | 17.91 GB | git bundle | 否 | **待确认** | 自身即备份 | 先用 `git bundle verify` + 与 origin refs 比对再决定 | 0（待验证） |
| 5 | `/workspace/backups/Photo-SLAM-L-formal-v1-20261106-072419.tar` | 18.45 GB | 时间点整包 | 否 | 与盘上目录重复（未逐字节验证） | 有 `.sha256` | 保留：是封存协议的唯一时间点副本 | 0 |
| 6 | `/workspace/code/Photo-SLAM-L/.git/objects` | 19.21 GB | git 历史 | **是** | 是 | **GitHub 有远程** | 保留；如需瘦身只能重写历史（本轮禁止） | 0 |
| 7 | `*/datasets/`（Replica/TUM/ETH3D/Rover/TandT） | ~200 GB | 数据集 | 是 | 部分重复 | 否 | SEGS-SLAM 侧 26.6 GB 重复→**待你决定** | 26.6（待批） |
| 8 | `Photo-SLAM/Temp/` + `Photo-SLAM-L/Temp/` + `SEGS-SLAM/Temp/` | 80.9 GB | 渲染输出 | 否 | 部分是唯一 | 否 | 逐目录判定（见 §3.1），本轮**未删** | 0 |
| 9 | `PSL-fullmap-exp-v1/runs/formal-v1` | 16.67 GB | 300 次运行产物 | 是（协议基准） | 是 | 否 | 保留（也是 300 项预算的实测依据） | 0 |
| 10 | `PSL-pml-ablation/experiments/camdiag_20261028` | 17.94 GB | 消融产物 | 否 | 是 | 否 | 需外部归档目标后再删 | 0 |
| 11 | `/tmp/psl-room1-control` | 32.41 GB | 运行产物 | 否 | **是（被 evidence 引用）** | 否 | **保留**：`PSL-perf/evidence/checkpoint_followup_r1` 直接引用其路径 | 0 |
| 12 | `/tmp/psl-densify-a` | 3.93 GB | 旧构建 + 输出 | 否 | 是 | 否 | **保留**：冻结二进制的 `RUNPATH`/`LD_LIBRARY_PATH` 指向它 | 0 |
| 13 | `/tmp/build_asan`, `/tmp/build_eval_new` | 1.17 + 1.05 GB | 纯 `.o` 对象 | 否 | 否（可重建） | 否 | **已删** | **2.16** |
| 14 | `/tmp/plucker_e2e` | 2.34 GB | 运行产物 | 否 | 是 | 否 | 保留（未归档） | 0 |
| 15 | `*/build*/`（13.5 GB） | 13.5 GB | 构建中间件 | 部分 | 否 | 源码在 Git | 保留（活跃工程要重新编译代价高） | 0 |
| 16 | 旧 worktree（`PSL-fgm-ablation` 等 7 个 + `cabinet-eval`） | ~6.4 GB | git worktree | 否 | **有未提交改动** | diff 已归档 | `git worktree remove` 需 `--force`，**本轮不做** | 0 |
| 17 | `Photo-SLAM/datasets/tandt_db.zip` | 0.64 GB | 数据集包 | 否 | **内容与解压目录不同名** | 否 | 保留（未证明冗余） | 0 |
| 18 | `/workspace/code/Photo-SLAM/datasets/Train.zip` | 0.19 GB | 数据集包 | 否 | 否（301/301 一致） | 否 | **已删** | **0.19** |
| 19 | `/root/.vscode-server` | 5.35 GB | 编辑器/扩展 | 是 | 否 | 否 | 保留（删了会断连） | 0 |

### 3.1 本轮对 `Temp/` 的处理与原因

`Temp/` 共 80.9 GB，但**逐目录语义不同**：`Photo-SLAM-L/Temp/0_iterations/.../room1` 是历史
room1 结果的**唯一副本**（见 `photo-slam-l-room1-old-recovered` 记录），`Photo-SLAM/Temp/10000_iterations`
是官方基线渲染。按前一轮"必须先有独立归档目标"的规则，**本轮一律未删**，只登记。

---

## 4. `/workspace/backups` 里的旧 18 GB 备份

`/workspace/backups/Photo-SLAM-L-formal-v1-20261106-072419.tar` = **18 451 957 760 B**（≈18.45 GB），
旁边有 `.sha256`。按文件名与创建命令（`tar -cf ... -C /workspace/code PSL-fullmap-exp-v1 PSL-fullmap-lock-v1`）
它只覆盖**这两个目录**。

**它没有覆盖**后来新增的内容：
`PSL-final`（新冻结 L_core）、`PSL-evtrepl`（本轮全部证据）、`cfg_v2/`、`lossaudit/`、
以及 `PSL-fullmap-exp-v1/runs/` 之后新增的任何运行。
所以它不是"当前状态的备份"，只能算"11-06 时间点的协议快照"。本轮**未删除**。

另有 `/workspace/backups/psl-20261013-131651/` 22 MB（10-13 的旧小备份）。

---

## 5. 本轮实际执行与释放

| 动作 | 对象 | 验证方式 | 释放 |
|---|---|---|---|
| 删除 | `Photo-SLAM/datasets/garden_large_2024-05-29_1.zip` | `zipfile` 逐条比对：**58 925/58 925** 条目名称+大小与解压目录完全一致，0 处大小不符；另取 5 个文件做 CRC 抽样（含 `groundtruth.txt`、3 张 png），全部一致；解压目录**多出** 1 个 `realsense_D435i/associated.txt`（1.08 MB，本地生成） | **18.52 GB** |
| 删除 | `/tmp/build_asan`、`/tmp/build_eval_new` | 内容仅 37 / 36 个 `.o`；无脚本引用（grep 全仓库 0 命中）；无 ELF `RUNPATH` 指向 `/tmp` | **2.16 GB** |
| 删除 | `Photo-SLAM/datasets/Train.zip` | 301/301 条目名称+大小与 `Train/` 一致，0 差异 | **0.19 GB** |
| 合计 | | | **20.87 GB** |

`df` 变化（`/workspace/code`）：`Avail 24 747 290 624 B` → `44 633 939 968 B`
（清理后为 `44G`，98%）。

---

## 6. 仍未备份、因而不能删的

1. **数据集**（285.8 GB），其中 `campus_large` / `campus_small` 两个 zip 是唯一形态。
2. **运行产物**（`Temp/` 80.9 GB、`results/` 52.3 GB、`runs/formal-v1` 16.67 GB）：
   PLY/PNG 无法从 Git 恢复，容器内也**没有**第二个持久挂载可作目标。
3. **冻结二进制**（`PSL-final/lib`、`PSL-review/bin_O_core` 等）：Git 只存哈希。
4. **旧 git 历史备份**：`history.bundle`（17.91 GB）、`/workspace/backups/*.tar`（18.45 GB）。
5. **`/tmp/psl-room1-control`**（32.41 GB）、**`/tmp/psl-densify-a`**（3.93 GB）：被证据/二进制引用。

> 结论：**容器内没有可用的"大文件归档目标"**（`/workspace/backups` 与源在同一块设备）。
> 大文件归档必须先由你在宿主机挂载外部存储，再按 `CLEANUP_LEDGER.csv` 的清单搬运。
> 这一条没有绕过，也没有用"再打一个 tar"假装完成。

---

## 7. 正式 300 项实验的空间预算（实测）

封存协议自己的 300 次运行就在 `/workspace/code/PSL-fullmap-exp-v1/runs/formal-v1`：

```
run dirs       : 300
total bytes    : 17 912 594 619  (16.68 GB)
per run        : 56.9 MB
单次内容       : NNNNN_shutdown/ (105 MB 以内的 ply+局部产物) + 轨迹 4 个 txt + 计时/显存 txt
```

* 新计划 300 项（O_core 150 + L_core 150，30 单元格 × 5 次）按同一口径 ≈ **16.7 GB**。
* 当前可用 **44 GB** → **满足**，但余量只有 ~27 GB；若同时保留新旧两批产物、或打开逐迭代存图，
  余量会被迅速吃掉。
* 建议：正式批次输出写到**独立 run 根目录**（`/workspace/code/runs/formal-2026xx/`），
  产物保留 `NNNNN_shutdown` + 轨迹 + 计时，**不要**每轮保留全量渲染 PNG。

---

## 8. 以后的保存规则（已写入归档 README）

1. **少量活跃代码目录**（`PSL-final`、`PSL-review/O_core`、`Photo-SLAM-L`）；
   历史版本用 **Git 分支/标签**（`source/*`、`archive-*`），不再靠复制整棵树。
2. **数据集只读共享**：一份权威副本 + 只读挂载或软链接，禁止每个项目各拷一份。
3. **结果写入持久存储的独立 run 目录**，命名含版本哈希；不要写进源码树。
4. **不要每轮复制整套 `build/`、`lib/` 和依赖**：worktree 共享 `.git`，构建产物用独立 build dir。
5. 任何"删除"必须先满足：已在 Git 或独立备份可恢复 + 恢复验证通过 + 有清单记录。
