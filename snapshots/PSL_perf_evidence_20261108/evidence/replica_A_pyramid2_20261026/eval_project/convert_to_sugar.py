import json
import os
import numpy as np
import struct
import argparse
import shutil
from scipy.spatial.transform import Rotation as R

def save_cameras_bin(cameras_data, output_path):
    """保存为 COLMAP cameras.bin"""
    with open(output_path, "wb") as f:
        f.write(struct.pack("<Q", 1))
        cam = cameras_data[0]
        cam_id, model_id = 1, 1
        width, height = int(cam['width']), int(cam['height'])
        fx, fy = float(cam['fx']), float(cam['fy'])
        cx, cy = width / 2.0, height / 2.0
        f.write(struct.pack("<iiQQ", cam_id, model_id, width, height))
        f.write(struct.pack("<dddd", fx, fy, cx, cy))
        f.flush()
        os.fsync(f.fileno())
    return os.path.exists(output_path)

def save_images_bin(cameras_data, output_path):
    """保存为 COLMAP images.bin"""
    with open(output_path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras_data)))
        for i, cam in enumerate(cameras_data):
            image_id = i + 1
            pos_wc = np.array(cam['position'])
            rot_wc = np.array(cam['rotation'])
            T_wc = np.eye(4)
            T_wc[:3, :3] = rot_wc
            T_wc[:3, 3] = pos_wc
            T_cw = np.linalg.inv(T_wc)
            q = R.from_matrix(T_cw[:3, :3]).as_quat()
            q_colmap = [q[3], q[0], q[1], q[2]] 
            t_colmap = T_cw[:3, 3]
            camera_id = 1
            pure_name = os.path.basename(cam['img_name']) + "\0" 
            f.write(struct.pack("<idddddddi", image_id, *q_colmap, *t_colmap, camera_id))
            f.write(pure_name.encode('utf-8'))
            f.write(struct.pack("<Q", 0))
        f.flush()
        os.fsync(f.fileno())
    return os.path.exists(output_path)

def convert_ply_to_points3D_bin(ply_path, output_bin_path):
    """读取 input.ply 并转换为 COLMAP points3D.bin"""
    print(f"正在转换点云: {ply_path} -> points3D.bin")
    
    try:
        with open(ply_path, 'rb') as f:
            # 1. 跳过文件头
            header_size = 0
            while True:
                line = f.readline()
                header_size += len(line)
                if b"end_header" in line:
                    break
            
            # 2. 读取二进制顶点数据
            # Photo-SLAM-L 的 input.ply 通常包含: x, y, z (f4), nx, ny, nz (f4), r, g, b (u1)
            # 或者是简单的 x, y, z (f4), r, g, b (u1)
            # 这里我们根据常见格式尝试解析
            f.seek(header_size)
            
            # 尝试解析格式: 3个float(xyz) + 3个float(法向) + 3个uint8(rgb)
            # 这是 Photo-SLAM 源码中常见的存储方式
            dt = np.dtype([
                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                ('r', 'u1'), ('g', 'u1'), ('b', 'u1')
            ])
            
            # 如果你的 PLY 没有法向，请改用这个 dtype:
            # dt = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
            
            data = np.fromfile(f, dtype=dt)
            
            if len(data) == 0:
                # 如果按上述格式没读到数据，尝试不带法向的格式
                f.seek(header_size)
                dt_no_normal = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
                data = np.fromfile(f, dtype=dt_no_normal)

            points = np.stack([data['x'], data['y'], data['z']], axis=1)
            colors = np.stack([data['r'], data['g'], data['b']], axis=1)

        # 3. 写入 COLMAP points3D.bin
        with open(output_bin_path, "wb") as f_bin:
            num_points = len(points)
            f_bin.write(struct.pack("<Q", num_points))
            for i in range(num_points):
                point_id = i + 1
                xyz = points[i].astype(np.float64)
                rgb = colors[i].astype(np.uint8)
                error = 0.0
                # COLMAP points3D 格式: id(Q), x,y,z(ddd), r,g,b(BBB), error(d), track_len(Q)
                f_bin.write(struct.pack("<QdddBBB", point_id, *xyz, *rgb))
                # 补全最后两个字段: error 和 track_length
                f_bin.write(struct.pack("<dQ", error, 0))
        
        print(f"✅ 点云转换完成: 共 {num_points} 个点")
        return True

    except Exception as e:
        print(f"⚠️ 点云解析失败: {e}")
        print("生成空 points3D.bin 以保证目录结构完整")
        with open(output_bin_path, "wb") as f_bin:
            f_bin.write(struct.pack("<Q", 0))
        return False

def main():
    parser = argparse.ArgumentParser(description="全自动 Photo-SLAM-L 到 SuGaR 格式转换器")
    parser.add_argument("--input_dir", "-i", required=True, help="包含 cameras.json 和 input.ply 的文件夹路径")
    parser.add_argument("--output_dir", "-o", required=True, help="SuGaR 输出根目录")
    
    args = parser.parse_args()

    # 1. 路径定义
    input_path = os.path.abspath(args.input_dir)
    json_file = os.path.join(input_path, "cameras.json")
    input_ply = os.path.join(input_path, "input.ply")
    # 假设 point_cloud.ply 在子文件夹下，根据 Photo-SLAM 习惯查找
    point_cloud_src = os.path.join(input_path, "point_cloud/iteration_9251/point_cloud.ply") # 需根据实际迭代数确认
    
    output_root = os.path.abspath(args.output_dir)
    sparse_path = os.path.join(output_root, "sparse/0")
    images_dst = os.path.join(output_root, "images")
    
    os.makedirs(sparse_path, exist_ok=True)
    os.makedirs(images_dst, exist_ok=True)

    # 2. 读取 JSON
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 3. 转换位姿
    save_cameras_bin(data, os.path.join(sparse_path, "cameras.bin"))
    save_images_bin(data, os.path.join(sparse_path, "images.bin"))
    
    # 4. 转换点云 (input.ply -> points3D.bin)
    if os.path.exists(input_ply):
        convert_ply_to_points3D_bin(input_ply, os.path.join(sparse_path, "points3D.bin"))
    else:
        print("⚠️ 找不到 input.ply，生成空 points3D.bin")
        with open(os.path.join(sparse_path, "points3D.bin"), "wb") as f: f.write(struct.pack("<Q", 0))

    # 5. 复制图片
    print("📸 正在复制原图...")
    for cam in data:
        src = cam['img_name']
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(images_dst, os.path.basename(src)))

    # 6. 处理最终高斯模型 (自动寻找最深层级的 point_cloud.ply)
    # 在输入目录下递归查找第一个 point_cloud.ply
    found_pc = False
    for root, dirs, files in os.walk(input_path):
        if "point_cloud.ply" in files:
            pc_path = os.path.join(root, "point_cloud.ply")
            shutil.copy2(pc_path, os.path.join(output_root, "output_model.ply"))
            print(f"✅ 已找到并复制模型: {pc_path} -> output_model.ply")
            found_pc = True
            break
    
    if not found_pc:
        print("❌ 未能在目录下找到 point_cloud.ply，请手动复制并重命名。")

    print(f"\n🎉 转换大功告成！SuGaR 输入目录已准备好: {output_root}")

if __name__ == "__main__":
    main()
