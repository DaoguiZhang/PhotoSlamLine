# Photo-SLAM-L 归档说明（2026-11-08）

本目录是 **源码 + 小型证据** 的归档副本，存放在 GitHub 仓库
`git@github.com:DaoguiZhang/PhotoSlamLine.git` 的**独立归档分支**上，
**不合并、不覆盖任何默认分支或历史分支**。

- 归档分支：`source/space-audit-20261108`
- 不可变标签：`archive-20261108`
- 归档根目录：`snapshots/`

## 1. 这个归档**包含**什么

| 目录 | 内容 | 文件数 | 体积 |
|---|---|---|---|
| `snapshots/L_core_final_20261108/` | 冻结的 L_core 源码树（`PSL-final`）：`include/ src/ examples/ cuda_rasterizer/ ORB-SLAM3/ (不含 Thirdparty/Vocabulary/evaluation) cfg/ scripts/ tests/ CMakeLists.txt` + 该轮证据副本 | 482 | 15 MB |
| `snapshots/L_replay_20261108/` | 产生 30 次重放结果的版本（`PSL-replay`）同口径源码 | 482 | 15 MB |
| `snapshots/O_core_20261108/` | 正式对比基线 `PSL-review/O_core`（官方 + 公共工程修复）源码 | 234 | 5.2 MB |
| `snapshots/O_raw_20261108/` | 官方原样构建 `PSL-review/O_raw` 源码（仅作对照） | 234 | 5.2 MB |
| `snapshots/protocol_v1_20261108/` | 封存协议 `PSL-fullmap-exp-v1` 的 `configs/ manifest.csv report/ scripts/`（**不含** `runs/` 16.7 GB） | 209 | 2.2 MB |
| `snapshots/protocol_baseline_csv_20261108/` | 30 单元格协议机器可读件（`cells.csv` / `budget_per_cell.csv` / `calibration.csv` / `BASELINE_PROTOCOL.md`） | 4 | 68 KB |
| `snapshots/evidence_evtrepl_20261108/` | `PSL-evtrepl` 报告、`report/*.csv`、`exitreg/`、`lossaudit/`、`diffs/`、`cfg_v2/`、`FORMAL_PLAN/` | 74 | 5.3 MB |
| `snapshots/Photo_SLAM_L_worktree_20261108/` | 主工作仓库 `Photo-SLAM-L` 的**工作区**源码/配置/脚本（非 HEAD） | 505 | 35 MB |
| `snapshots/Photo_SLAM_L_worktree_20261108_GITSTATE/` | 上者的 `HEAD`、`git status`、`git diff` 全文 | 5 | 112 KB |
| `snapshots/PSL_perf_worktree_20261108/` | `PSL-perf`（worktree，`feat/mapper-budget-v1`）源码 + `archive/` | 818 | 26 MB |
| `snapshots/PSL_perf_evidence_20261108/` | `PSL-perf/evidence/` 的报告与 CSV（**不含** vendored `eval_site/`） | 599 | 14 MB |
| `snapshots/WORKTREE_GITSTATE/` | 其余 9 个 git worktree 的 `HEAD / status / diff` 快照 | 27 | 1.6 MB |
| `snapshots/*.sha256` | 每个快照目录的逐文件 sha256 清单 | — | — |

合计：**约 123 MB / 3 966 个文件**。

## 2. 这个归档**不包含**什么（必须另行处理）

| 类别 | 体积 | 位置 | 说明 |
|---|---|---|---|
| 原始数据集 | ~286 GB | `/workspace/code/*/datasets/` | Replica / TUM / ETH3D / Rover / TandT / campus / garden；Git 与归档均不放 |
| 渲染与训练产物 | ~81 GB | 各项目 `Temp/`（80.9 GB） | PNG/PLY 序列，可由运行重建，但部分是唯一结果 |
| 实验结果目录 | ~52 GB | 各项目 `results/`、`PSL-fullmap-exp-v1/runs/`（16.7 GB） | 只归档其中的 CSV/manifest，图像与 PLY 未归档 |
| 冻结二进制 | 每份 6–40 MB | `PSL-final/{lib,bin}`、`PSL-review/bin_*` | **保留原文件，不在 Git 中**；哈希见 `EXPERIMENT_INDEX.csv` |
| 旧 git 历史备份 | 35.9 GB | `/workspace/code/Photo-SLAM-L_backups/`、`/workspace/backups/` | 见 `SPACE_AUDIT.md`；本轮未删 |
| 第三方源码 | — | `*/ORB-SLAM3/Thirdparty/`、`opencv-*`、`pcl-*` | 上游可重取，不归档 |

> **不要把"已上传 Git"理解成"模型和原始结果也已保存"**：Git 里只有源码、配置、脚本、
> 小型 CSV/报告/哈希。PLY、PNG、checkpoint、数据集、二进制都不在 Git 里。

## 3. 白名单规则（本轮实际使用的规则）

只接受扩展名白名单：`.h .hpp .hh .cpp .cc .cu .cuh .c .py .sh .yaml .yml .json .csv .md .txt .log .cmake CMakeLists.txt Makefile .patch .diff .proto .gitignore`，
且单文件 < 20 MB。
剪除目录：`build/ build_*/ lib/ bin/ bin_*/ libs/ Temp/ results/ datasets/ .git/ viewer/
Thirdparty/ Vocabulary/ evaluation/ runs/ export/ replay/ logs/ out_*/ experiments/ abl_out/
checkpoints/ ply/ image/ node_modules/ __pycache__/ eval_site/`。

**从未使用** `git add .` 或 `git add -A`；每次提交的文件集由上述规则生成并用
`snapshots/*.sha256` 固定。

## 4. 凭据扫描

在提交前对全部 3 966 个文件做了模式扫描（私钥块、`ghp_`/`github_pat_`/`AKIA`/`sk-`/`xox?-` 令牌、
`*.pem *.key id_rsa* .env *secret* *credential*`）：**未发现**任何凭据或私钥文件。
`/root/.ssh/` 不在归档范围内。
命中 "password/token" 字样的文件仅为第三方库（`tinyply.h`、`fontTools`、`argcomplete`），
无实际值。
