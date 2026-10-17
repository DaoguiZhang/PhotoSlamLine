# 第三轮：共同视角评价工具（实现结果与硬阻塞）

## 已完成
- 新增 headless 入口 `examples/eval_common_views.cpp`（编译通过，CMake 目标
  `eval_common_views`）：加载 PLY + ORB 内参，对给定 Tcw 列表逐视角渲染并写 PNG/CSV，
  不依赖 ImGui，不训练、不 step、不改算法。字段转换由 `GaussianModelLine::loadPly`
  与 `renderWithLine` 完成（EXPLICIT 不做二次 exp/sigmoid/SH 变换）。
- 新增 `scripts/eval_common_views.py`：24 个确定性帧 ID（0,83,…,1909）、SE(3) Umeyama
  （无尺度）把 GT `T_wc` 变到各 run 地图坐标系后取逆写 `eval_cameras.txt`，
  调用上面的入口，并对同一批 PNG 用**同一实现**算 PSNR/SSIM（numpy）与 LPIPS（CPU）。

## 硬阻塞（实测，不是推断）
12 次运行保存的 `*_shutdown/ply/input.ply` 是**可视化点云**，不是 Gaussian 参数：

```
element vertex N
property float x/y/z  property float nx/ny/nz  property uchar red/green/blue
end_header
```

P 与 L 的 PLY 头完全相同，均**缺少** `f_rest_*`、`opacity`、`scale_*`、`rot_*`。
`tinyply` 明确报错 `property keys were not found: f_rest_0.. , opacity, scale_0.., rot_0..`。
因此这些地图**无法离线渲染**：没有不透明度、尺度、旋转与高阶 SH，任何渲染结果都不是
当时的 Gaussian 场。改用 `GaussianModel` 的 `loadPly` 也一样（它只认 xyz/法向/颜色，
其余字段缺失即失败或默认为零）。

结论：**共同视角指标无法由这 12 份归档地图得到**，与渲染入口无关；缺的是地图本身。

## 最小补充（明确、可执行，未擅自执行）
1. 在 `savePly`（或 `GaussianModelLine` 导出路径）增加 Gaussian 参数导出：
   xyz、f_dc/f_rest、opacity、scale、rot、is_line 等（与 `loadPly` 字段严格对称）。
2. 只跑 **4 次**（P_ref/P_fast/L_ref/L_fast 各一次，`ROUNDS=1`），保留同一配置与调度，
   文件落到 `*_shutdown/gaussian.ply`；不需要恢复 Adam，不影响现有 12 次结果。
3. 用本轮已实现的 `scripts/eval_common_views.py` 对 4 份可用地图出共同视角表
   （旧 12 次表继续标为训练视图，不混合）。
4. 验收：用某个 run 的关键帧估计位姿重渲染，与原 `_shutdown/image/*.jpg` 比较；
   另检查同图 LPIPS≈0、RGB/BGR、PSNR/SSIM 单实现、mask 不因空洞缩小、无梯度不改参数。

## 未做
- 共同视角四组指标（被上述阻塞阻断）。
- A/C 内核微基准（同一 P 地图不可渲染，同样依赖第 1 步导出）。
