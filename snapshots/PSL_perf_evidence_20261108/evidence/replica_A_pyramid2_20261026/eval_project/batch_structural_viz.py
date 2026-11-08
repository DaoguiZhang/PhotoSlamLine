import os
import glob
import json
import numpy as np
import pyvista as pv
from plyfile import PlyData
import argparse
import math

def focal2fov(focal, dist):
    """将焦距转换为 FoV (弧度)"""
    return 2 * math.atan(dist / (2 * focal))

def run_batch_viz_from_folder(base_dir, threshold=1000.0, step=10):
    # 1. 自动定位 PLY 文件
    # 匹配模式: base_dir/point_cloud/iteration_*/point_cloud.ply
    ply_pattern = os.path.join(base_dir, "point_cloud", "iteration_*", "*.ply")
    ply_files = glob.glob(ply_pattern)
    if not ply_files:
        print(f"错误: 在 {base_dir} 下找不到任何 .ply 文件")
        return
    ply_path = ply_files[0]
    print(f"已定位 PLY: {ply_path}")

    # 2. 读取 cameras.json
    cam_json_path = os.path.join(base_dir, "cameras.json")
    if not os.path.exists(cam_json_path):
        print(f"错误: 找不到 {cam_json_path}")
        return
    with open(cam_json_path, 'r') as f:
        camera_data = json.load(f)

    # 3. 加载并处理点云数据 (各向异性分析)
    print("正在处理高斯模型结构...")
    plydata = PlyData.read(ply_path)
    v = plydata['vertex']
    positions = np.stack([v['x'], v['y'], v['z']], axis=1)
    
    fields = v.data.dtype.names
    s_keys = ['scale_0', 'scale_1', 'scale_2'] if 'scale_0' in fields else ['scaling_0', 'scaling_1', 'scaling_2']
    # 3DGS 通常存储的是 log scaling
    scales = np.exp(np.stack([v[s_keys[0]], v[s_keys[1]], v[s_keys[2]]], axis=1))
    anisotropy = np.max(scales, axis=1) / (np.min(scales, axis=1) + 1e-6)

    # 结构化点筛选 (亮红色骨架)
    mask = anisotropy > threshold
    struct_pos = positions[mask]
    struct_colors = np.log10(anisotropy[mask]) # 使用 Log 增强颜色对比

    # 4. 设置 PyVista 绘图器
    output_dir = os.path.join(base_dir, "structural_paper_renders")
    os.makedirs(output_dir, exist_ok=True)
    
    plotter = pv.Plotter(off_screen=True, window_size=[1600, 1200])
    plotter.background_color = "white"

    # 绘制淡灰色背景 (Context)
    plotter.add_mesh(pv.PolyData(positions), color="#E0E0E0", opacity=0.08, point_size=2, name="bg")

    # 绘制亮红色结构线 (Ours: Structural Scaffolding)
    if len(struct_pos) > 0:
        pcd_struct = pv.PolyData(struct_pos)
        pcd_struct['Anisotropy_Log'] = struct_colors
        plotter.add_mesh(pcd_struct, scalars='Anisotropy_Log', cmap="Reds", 
                         point_size=10.0, render_points_as_spheres=True, 
                         name="struct", show_scalar_bar=False)

    # 5. 按照 JSON 里的位姿进行批量渲染
    print(f"开始渲染... 每隔 {step} 帧保存一张图")
    for i in range(0, len(camera_data), step):
        cam = camera_data[i]
        
        # 提取位姿
        pos = np.array(cam["position"])
        # rotation 是 3x3 矩阵
        R = np.array(cam["rotation"])
        
        # 提取内参并计算 FoV (PyVista 需要角度)
        fovx = focal2fov(cam["fx"], cam["width"])
        fovy = focal2fov(cam["fy"], cam["height"])
        view_angle = math.degrees(fovy)

        # 计算相机关键向量 (3DGS -> PyVista 转换)
        # Z轴方向为前进方向
        direction = R @ np.array([0, 0, 1])
        focal_point = pos + direction
        # Y轴取反 (3DGS/OpenCV Y向下 -> PyVista Y向上)
        view_up = R @ np.array([0, -1, 0])

        # 更新相机状态
        plotter.camera.position = pos
        plotter.camera.focal_point = focal_point
        plotter.camera.up = view_up
        plotter.camera.view_angle = view_angle
        
        # 渲染并保存
        plotter.render()
        out_name = os.path.join(output_dir, f"frame_{cam['id']:06d}_struct.png")
        plotter.screenshot(out_name)
        
        if i % 100 == 0:
            print(f"已保存: {out_name}")

    plotter.close()
    print(f"\n渲染完成! 结果保存在: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 只需要输入包含 cameras.json 的文件夹路径
    parser.add_argument("-d", "--dir", type=str, required=True, 
                        help="包含 cameras.json 和 point_cloud/ 的主文件夹路径")
    parser.add_argument("-t", "--threshold", type=float, default=200.0, 
                        help="结构化各向异性阈值 (建议 800-1500)")
    parser.add_argument("-s", "--step", type=int, default=1, 
                        help="采样步长")
    
    args = parser.parse_args()
    run_batch_viz_from_folder(args.dir, args.threshold, args.step)
