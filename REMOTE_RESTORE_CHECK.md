# 远端恢复核验（REMOTE_RESTORE_CHECK）

日期：2026-11-08
远端：`git@github.com:DaoguiZhang/PhotoSlamLine.git`（SSH key `/root/.ssh/id_ed25519`，推送权限已验证：
`ssh -T git@github.com` → `Hi DaoguiZhang! You've successfully authenticated`）

本轮**没有**动 `main`、没有任何已有分支被强推或改写；只新建了 refs。

---

## 1. 本轮新建的远端 refs

| ref | 指向 commit | 内容 | 用途 |
|---|---|---|---|
| `refs/heads/source/space-audit-20261108` | `ee86ac9ae67fe67b36f2dc45a7214addf1a86860` | 12 个快照目录 + `README_ARCHIVE.md` / `RESTORE.md` / `EXPERIMENT_INDEX.csv` | **主归档分支** |
| `refs/tags/archive-20261108` | `444a3f35c11614fd2de7c14da95475ab414595f2`（annotated → tag obj `678b5f02…`） | 首个归档提交 | 不可变标签（rev1） |
| `refs/tags/archive-20261108-r2` | `ee86ac9ae67fe67b36f2dc45a7214addf1a86860`（annotated → tag obj `2ce122a4…`） | rev1 + 3 个被 .gitignore 隐藏的文件 | **推荐恢复点** |
| `refs/heads/archive/line-rescue-tracking-20261108` | `0e2f7fc4001019c61984fc7e266b80f188c8a12e` | Photo-SLAM-L 工作仓库 HEAD（此前**只存在于本地**） | 真实运行过的源码 |
| `refs/heads/archive/mapper-budget-v1-20261108` | `ec85e80886e2f0494acb87781e075334bcf2ba58` | PSL-perf HEAD（此前**只存在于本地**） | 真实运行过的源码 |

核验命令与结果（`git ls-remote`）：

```
ee86ac9ae67fe67b36f2dc45a7214addf1a86860  refs/heads/source/space-audit-20261108
444a3f35c11614fd2de7c14da95475ab414595f2  refs/tags/archive-20261108^{}
ee86ac9ae67fe67b36f2dc45a7214addf1a86860  refs/tags/archive-20261108-r2^{}
0e2f7fc4001019c61984fc7e266b80f188c8a12e  refs/heads/archive/line-rescue-tracking-20261108
ec85e80886e2f0494acb87781e075334bcf2ba58  refs/heads/archive/mapper-budget-v1-20261108
```

与本地 `git rev-parse` 逐一相等。

---

## 2. 独立恢复测试（不使用本地对象）

```bash
rm -rf /tmp/restore_check2
git clone --depth 1 --branch source/space-audit-20261108 \
    git@github.com:DaoguiZhang/PhotoSlamLine.git /tmp/restore_check2/psl
# → HEAD ee86ac9ae67fe67b36f2dc45a7214addf1a86860
# → tracked files 3688 ; snapshots/ 120 258 082 B
```

逐文件校验（每个快照目录一份清单）：

```
manifest_failures=0
```

12 份清单全部通过：`L_core_final` 482、`L_replay` 482、`O_core` 234、`O_raw` 234、
`protocol_v1` 209、`protocol_baseline_csv` 4、`evidence_evtrepl` 74、
`Photo_SLAM_L_worktree` 505、`Photo_SLAM_L_worktree_GITSTATE` 5、
`PSL_perf_worktree` 818、`PSL_perf_evidence` 599、`WORKTREE_GITSTATE` 27。

### 2.1 第一次校验的失败与修正（如实保留）

第一次 clone 后有 **3 个文件缺失**：

```
.vscode/settings.json                                   FAILED open or read
Test/test_initial_cam_pose/before/2_Keyframe_Camera_after.txt   FAILED open or read
Test/test_initial_cam_pose/before/2_Keyframe_Camera_before.txt  FAILED open or read
```

原因**不是**传输损坏，而是快照树里带着 `Photo-SLAM-L/.gitignore`
（第 59 行 `Test/`、第 75 行 `.vscode/`），`git add` 静默跳过了它们——清单里有、提交里没有。
处理方式：`git add -f` 这 3 个文件（commit `ee86ac9`），**没有**删减清单来"让它通过"。
重新 clone 校验后 `manifest_failures=0`。

---

## 3. 从远端能恢复什么 / 不能恢复什么

| 能否从这批 refs 恢复 | 内容 |
|---|---|
| **能** | `PSL-final`、`PSL-replay`、`O_core`、`O_raw` 的源码/配置/脚本/测试；封存协议的 `configs/ manifest report`；30 单元格协议 CSV；本轮全部证据（报告、CSV、退出回归日志、损失审计日志、patch）；`Photo-SLAM-L`/`PSL-perf` 工作区源码与完整 `git diff`；9 个 worktree 的 `HEAD/status/diff` |
| **不能** | 数据集、PLY、PNG、checkpoint、运行产物（`runs/`、`Temp/`、`replay/`、`export/`）、冻结二进制本体 |

冻结二进制的身份仍以哈希固定（`RESTORE.md` §3）：

```
libgaussian_mapper.so (L_core) 538f46296c8e33721ade70e042ae2d5f…  (原文件保留在 PSL-final/lib)
libORB_SLAM3.so                5a14981faac97ed9db7cf7d834168db2…
libgaussian_mapper.so (O_core) 4293b9d7349f8bde2436097e5a278f83…
```

**不承诺**重新编译得到相同 `.so` 哈希；需要"就是这个二进制"的结论，必须依赖原文件（本轮未删）。

---

## 4. 依赖与构建（保存但不重编）

* 依赖版本、`cmake` 流程、运行命令、退出码约定、环境变量取值：`RESTORE.md` §4/§5。
* 关键实测：`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j20`；
  CUDA arch 120 / RTX 5090 / torch 2.12 / OpenCV 4.10 / Eigen3 / Pangolin。
* 本轮**没有**为每个历史版本重新编译。

---

## 5. 本地验收（清理后）

```bash
$ ldd $PSL-final/bin/replica_mono | grep -E 'gaussian_mapper|ORB_SLAM3'
 libgaussian_mapper.so => /workspace/code/PSL-final/lib/libgaussian_mapper.so
 libORB_SLAM3.so       => /workspace/code/PSL-final/ORB-SLAM3/lib/libORB_SLAM3.so

$ LD_LIBRARY_PATH=PSL-review/O_core/lib:... ldd PSL-review/bin_O_core
 libgaussian_mapper.so => /workspace/code/PSL-review/O_core/lib/libgaussian_mapper.so
 libORB_SLAM3.so       => /workspace/code/PSL-review/O_core/ORB-SLAM3/lib/libORB_SLAM3.so

$ git -C Photo-SLAM-L worktree list | wc -l   → 10   （worktree 全部有效）
$ ls PSL-final/ORB-SLAM3/Vocabulary/ORBvoc.txt → 存在（145 MB）
$ df -hT /workspace/code → 1.8T  1.7T  44G  98%
```

L_core 与 O_core 的库加载路径、词袋、配置目录均未受影响。
