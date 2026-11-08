import os
import glob
import json
import numpy as np
import pyvista as pv
from plyfile import PlyData
import argparse
import math

def focal2fov(focal, dist):
    """将焦距转换为弧度制 FoV"""
    return 2 * math.atan(dist / (2 * focal))

def run_fusion_viz(base_dir, threshold=1000.0, step=10):
    # 1. 自动定位 PLY 文件 ( iteration_*/point_cloud.ply )
    ply_pattern = os.path.join(base_dir, "point_cloud", "iteration_*", "*.ply")
    ply_files = glob.glob(ply_pattern)
    if not ply_files:
        print(f"Error: No .ply found in {base_dir}")
        return
    ply_path = ply_files[0]

    # 2. 读取相机位姿和参数
    cam_json_path = os.path.join(base_dir, "cameras.json")
    if not os.path.exists(cam_json_path):
        print(f"Error: Missing {cam_json_path}")
        return
    with open(cam_json_path, 'r') as f:
        camera_data = json.load(f)

    # 获取默认分辨率（如果第一帧有的话）
    default_w = camera_data[0].get("width", 1200)
    default_h = camera_data[0].get("height", 680)

    # 3. 处理 Gaussian 点云
    print(f"Processing PLY for structural features: {os.path.basename(ply_path)}")
    plydata = PlyData.read(ply_path)
    v = plydata['vertex']
    positions = np.stack([v['x'], v['y'], v['z']], axis=1)
    
    fields = v.data.dtype.names
    s_keys = ['scale_0', 'scale_1', 'scale_2'] if 'scale_0' in fields else ['scaling_0', 'scaling_1', 'scaling_2']
    scales = np.exp(np.stack([v[s_keys[0]], v[s_keys[1]], v[s_keys[2]]], axis=1))
    anisotropy = np.max(scales, axis=1) / (np.min(scales, axis=1) + 1e-6)

    # 提取结构化点 (红色骨架)
    mask = anisotropy > threshold
    struct_pos = positions[mask]
    struct_colors = np.log10(anisotropy[mask]) # 使用 Log 映射提升动态范围

    # 4. 初始化绘图器 (先用默认尺寸，后续循环中可动态调整)
    plotter = pv.Plotter(off_screen=True, window_size=[default_w, default_h])
    plotter.background_color = "black" # 设为黑色，方便后期融合

    # 绘制淡灰色背景点云 (只提供微弱的轮廓参考)
    plotter.add_mesh(pv.PolyData(positions), color="#333333", opacity=0.05, point_size=1, name="bg_context")

    # 绘制亮红色结构骨架
    if len(struct_pos) > 0:
        pcd_struct = pv.PolyData(struct_pos)
        pcd_struct['Anisotropy_Log'] = struct_colors
        plotter.add_mesh(pcd_struct, scalars='Anisotropy_Log', cmap="Reds", 
                         point_size=8.0, render_points_as_spheres=True, 
                         name="structural_scaffolding", show_scalar_bar=False)

    # 5. 循环渲染
    output_dir = os.path.join(base_dir, "structural_fusion_layers")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Rendering {len(camera_data)//step} views...")
    for i in range(0, len(camera_data), step):
        cam = camera_data[i]
        
        # 动态更新窗口尺寸 (如果相机参数中有定义)
        curr_w = cam.get("width", default_w)
        curr_h = cam.get("height", default_h)
        plotter.window_size = [curr_w, curr_h]

        # 提取位姿和 FoV
        pos = np.array(cam["position"])
        R = np.array(cam["rotation"])
        fovy = focal2fov(cam["fy"], curr_h)
        
        # 坐标系转换 (Z前, Y下 -> PyVista)
        direction = R @ np.array([0, 0, 1])
        focal_point = pos + direction
        view_up = R @ np.array([0, -1, 0])

        # 设置相机
        plotter.camera.position = pos
        plotter.camera.focal_point = focal_point
        plotter.camera.up = view_up
        plotter.camera.view_angle = math.degrees(fovy)

        plotter.render()
        out_name = os.path.join(output_dir, f"frame_{cam['id']:06d}_layer.png")
        plotter.screenshot(out_name)
        
        if i % 100 == 0:
            print(f"Saved layer: {out_name} ({curr_w}x{curr_h})")

    plotter.close()
    print(f"Finished! Layers saved in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, required=True, help="Path containing cameras.json")
    parser.add_argument("-t", "--threshold", type=float, default=1000.0)
    parser.add_argument("-s", "--step", type=int, default=10)
    args = parser.parse_args()
    
    run_fusion_viz(args.dir, args.threshold, args.step)
