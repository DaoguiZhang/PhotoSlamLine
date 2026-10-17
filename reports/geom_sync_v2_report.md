# geom-sync-v2 — 变换契约、锚点同步与自验证

基线：`7a2671bf35`（geom-sync-v1 审计）。本轮**不重复审计**；后端 C 不变，CUDA 加速实现未改，无新增结构损失/采样/训练调度变更。

## 状态总览

| 项 | 状态 |
|---|---|
| 一、矩阵契约 | **完成并数值验证**（CPU 参考，四组用例，误差 0） |
| 一、GPU 内核中心路径 | **仅由布局推导确认，未实测**（见 §1.3） |
| 二、`ref_kf_id_` 锚定关系 | **未做** |
| 三、轻量更正事件 | **未做** |
| 四、统一应用回环与 GBA | **未做** |
| 五、七项小验证 | **仅第 5 项的前置（契约）完成** |
| 六、真实回环运行 | **未做** |

## 一、变换契约

### 1.1 实际布局链（读代码得出）

1. `EigenMatrix2TorchTensor(M)`（`include/tensor_utils.h:180-194`）先 `transposeInPlace()`，再用 `from_blob` 按行主序解释 → 返回 `T[i][j] = M(i,j)`。
2. mapper 传给模型的 `diff_pose` 是 `T.transpose(0,1)`（`src/gaussian_mapper_line.cpp:4113-4116`），所以**到手的是逻辑上的 `M^T`**。
3. CUDA 内核拿到 `diff_pose.contiguous()`：该视图连续化后**正是 M 的列主序**，与 `transformPoint4x3` 的索引方式（`matrix[0],[4],[8],[12]`）吻合 → **中心路径本来是对的**。
4. 但**按索引切片**看到的是 `M^T`，因此旧代码 `diff_pose.slice(0,0,3).slice(1,0,3)` 得到的是 **R^T，不是 R**。

### 1.2 发现的既有缺陷（已修）

旧代码把这个 `R^T` 当作旋转矩阵去转 `line_dir_w_`，等于给线方向施加了**逆旋转**。数值化：方向 `(1,0,0)`、绕 Z 轴 90°，正确结果 `(0,1,0)`，旧表达式给出 `(0,-1,0)`，误差 **2.00**（单位向量直径）。

新入口（`src/gaussian_model_line.cpp`，`scaledTransformVisiblePointsOfKeyframe` 内）：

```cpp
torch::Tensor M = diff_pose.transpose(0, 1).contiguous();          // 真 M
torch::Tensor R = M.slice(0, 0, 3).slice(1, 0, 3);                 // 真 R
torch::Tensor t = M.slice(0, 0, 3).slice(1, 3, 4).transpose(0,1);  // 真 t（第 3 列）
// 行向量约定： (R d)^T = d^T R^T
line_dir_w_ : d' = d @ R^T          （只转不平移、不缩放）
xyz_init_   : a' = s * (a @ R^T) + t（与中心同一刚体映射）
```
并在两处赋值后调用既有 `invalidateLineCache()`，避免原张量变了而缓存切片仍旧。

**自查**：第一版把平移读成"矩阵第 3 行"（那是 `(0,0,0,1)`），被同一个自验证当场判 FAIL（recover/anchor 误差 1.00），修正为"第 3 列"后才归零。这是本轮唯一一次自我纠错，记录在此而不是埋在提交里。

### 1.3 验证结果（CPU 参考，`PHOTO_SLAM_SYNC_CONTRACT_TEST=1`）

```
[SyncContract] identity       recover=0.00e+00 direction=0.00e+00 anchor=0.00e+00 old_expression=0.00e+00
[SyncContract] translation    recover=0.00e+00 direction=0.00e+00 anchor=0.00e+00 old_expression=0.00e+00
[SyncContract] Rz90+trans     recover=0.00e+00 direction=0.00e+00 anchor=0.00e+00 old_expression=2.00e+00
[SyncContract] Rz90+scale2    recover=0.00e+00 direction=0.00e+00 anchor=0.00e+00 old_expression=2.00e+00
[SyncContract] worst recover=0.00e+00 direction=0.00e+00 anchor=0.00e+00 | old expression error=2.00e+00 | PASS (new entry)
```

四组用例覆盖：恒等、纯平移、90° 旋转+平移、90° 旋转+平移+尺度 2。检查项为中心之外的三项：旋转/平移恢复、方向、锚点。

**边界**：该测试是纯 CPU，验证的是"从实际传入的张量布局恢复真 R/t 并施加到方向与锚点"这一层。**GPU 内核作用于中心的路径本轮没有实测**，只由 §1.1 的列主序推导确认；GPU 侧实测需要单独的张量对拍（下一步）。

## 二～六、未完成部分与阻塞点

未做，且不是"缺输入"而是**本轮预算用尽**。按依赖顺序，阻塞链明确：

1. `ref_kf_id_`（§二）是后续一切的前提：没有逐 Gaussian 的真实来源关键帧，GBA 的非刚性更正无法按组应用。
2. 事件（§三）需要先定义 payload 与 `ref_kf_id_` 的消费语义。
3. 统一应用（§四）依赖 1+2。
4. 七项小验证中，第 1/2/3/4/7 项都需要事件入口才能构造。
5. 真实回环（§六）依赖 1-4。

数据集与配置已确认存在（`/workspace/code/SEGS-SLAM/datasets/tum/rgbd_dataset_freiburg3_long_office_household`、`cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml`），但**没有运行**。

## 七、本轮的实际收益与限制

**收益**：线方向的回环更正此前是反向的（误差 2.00）；现在与中心同源。线锚点 `xyz_init_` 此前完全不随更正移动，现在使用与中心完全相同的刚体映射。这两项在结构损失开启时才会影响损失值，本轮结构损失仍关闭，因此**不影响已验收的加速数据**。

**限制**：
- 只覆盖 `line_dir_w_` / `xyz_init_` / 中心；MapLine 端点的同步仍由 ORB-SLAM3 自身负责，本轮的 Gaussian 元数据更新**不替代** MapLine 几何更新。
- 归属仍是既有的"可见性 + 迭代邻近"启发式，**不是**真实来源；本轮没有改变这一点。
- Adam 一阶/二阶矩在几何更正后仍保留旧值（只有 `scaling_` 走 `replaceTensorToOptimizer`）。策略本轮**未改**，也未重置任何参数组——这是明确的遗留项，不是已完成项。
- SH 方向基底：SH 系数是视角方向上的颜色基，与高斯朝向/旋转无关；回环只改世界位置与旋转，不改 SH 系数。此结论未经实验验证，列为待核对项。

## 八、复现命令

```bash
cd /workspace/code/PSL-perf
cd build && make -j8          # 仅 src/gaussian_model_line.cpp + src/gaussian_mapper_line.cpp
cd ..
PHOTO_SLAM_SYNC_CONTRACT_TEST=1 CONDS=RA W=/workspace/code/PSL-perf \
  bash scripts/run_frozen_abl.sh synctest 20 office0 0
grep SyncContract abl_out/synctest/run.log
```
