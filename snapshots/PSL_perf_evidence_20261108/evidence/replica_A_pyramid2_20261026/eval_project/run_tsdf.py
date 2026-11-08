import os
import numpy as np
import open3d as o3d
import json
from tqdm import tqdm

def run_tsdf_from_json(result_path):
    # 加载元数据
    json_path = os.path.join(result_path, "tsdf_meta.json")
    with open(json_path, "r") as f:
        meta_data = json.load(f)

    # 1. 使用第一帧的内参初始化
    first = meta_data[0]["intrinsics"]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        first["width"], first["height"], first["fx"], first["fy"], first["cx"], first["cy"]
    )

    # 2. 初始化 TSDF (0.01m = 1cm 精度)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.03,
        sdf_trunc=0.06,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    print(f"正在融合 {len(meta_data)} 帧...")
    for frame in tqdm(meta_data):
        color_path = os.path.join(result_path, "image", frame["img_name"])
        depth_path = os.path.join(result_path, "depth_data", frame["depth_path"])
        
        if not os.path.exists(depth_path) or not os.path.exists(color_path):
            continue

        color_img = o3d.io.read_image(color_path)
        depth_np = np.load(depth_path)
        
        if depth_np.ndim == 3: depth_np = depth_np[0]
        
        # 转换深度图 (假设渲染深度单位是米)
        depth_o3d = o3d.geometry.Image(depth_np.astype(np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img, depth_o3d, depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False
        )

        # 处理位姿 (从 JSON 取出 12列转 4x4)
        c2w = np.eye(4)
        c2w[:3, :4] = np.array(frame["pose_12"]).reshape(3, 4)
        w2c = np.linalg.inv(c2w)

        volume.integrate(rgbd, intrinsic, w2c)

    # 3. 提取 Mesh
    print("提取 Mesh 中...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    output_path = os.path.join(result_path, "Photoslam_L_Mesh.obj")
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"成功保存 Mesh 至: {output_path}")

if __name__ == "__main__":
    # 修改为你的结果路径
    RESULT_DIR = "/home/zdg/zdg/CPlusPlusProjects/Photo-SLAM/results/replica_rgbd_0/room1" 
    run_tsdf_from_json(RESULT_DIR)
