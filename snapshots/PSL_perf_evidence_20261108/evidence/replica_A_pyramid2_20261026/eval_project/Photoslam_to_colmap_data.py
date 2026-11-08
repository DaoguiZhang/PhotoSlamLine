import json
import os
import numpy as np
import struct
from scipy.spatial.transform import Rotation as R

def save_cameras_bin(cameras_data, output_path):
    # COLMAP PINHOLE model: [f, f, cx, cy] -> [fx, fy, cx, cy]
    # 假设主点在中心
    with open(output_path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras_data)))
        # 我们假设所有图像使用同一个相机参数(Replica常用配置)
        cam = cameras_data[0]
        cam_id = 1
        model_id = 1 # PINHOLE
        width = cam['width']
        height = cam['height']
        fx = cam['fx']
        fy = cam['fy']
        cx = width / 2.0
        cy = height / 2.0
        f.write(struct.pack("<iiQQ", cam_id, model_id, width, height))
        f.write(struct.pack("<dddd", fx, fy, cx, cy))

def save_images_bin(cameras_data, output_path):
    with open(output_path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras_data)))
        for i, cam in enumerate(cameras_data):
            image_id = i + 1
            # 注意：Photo-SLAM-L json 存的是 T_wc (Camera to World)
            # COLMAP 需要的是 T_cw (World to Camera)
            pos_wc = np.array(cam['position'])
            rot_wc = np.array(cam['rotation'])
            
            # 构建 4x4 T_wc
            T_wc = np.eye(4)
            T_wc[:3, :3] = rot_wc
            T_wc[:3, 3] = pos_wc
            
            # 求逆得到 T_cw
            T_cw = np.linalg.inv(T_wc)
            
            # 提取四元数 (qw, qx, qy, qz)
            q = R.from_matrix(T_cw[:3, :3]).as_quat() # 返回 [x,y,z,w]
            q_colmap = [q[3], q[0], q[1], q[2]] # 变为 [w,x,y,z]
            t_colmap = T_cw[:3, 3]
            
            camera_id = 1
            name = cam['img_name'] + "\0" # 必须以空字符结尾
            
            f.write(struct.pack("<idddddddi", image_id, *q_colmap, *t_colmap, camera_id))
            f.write(name.encode('utf-8'))
            
            # COLMAP 每张图后面还要跟 2D 点信息，SuGaR 优化时通常不需要这些，可以填 0
            f.write(struct.pack("<Q", 0))

def create_empty_points3D_bin(output_path):
    # SuGaR 需要这个文件存在，即使没有点
    with open(output_path, "wb") as f:
        f.write(struct.pack("<Q", 0))

def convert_photoslam_to_sugar(json_file, output_dir):
    os.makedirs(os.path.join(output_dir, "sparse/0"), exist_ok=True)
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    save_cameras_bin(data, os.path.join(output_dir, "sparse/0/cameras.bin"))
    save_images_bin(data, os.path.join(output_dir, "sparse/0/images.bin"))
    create_empty_points3D_bin(os.path.join(output_dir, "sparse/0/points3D.bin"))
    
    print(f"转换完成！COLMAP 文件已保存至 {output_dir}/sparse/0/")

if __name__ == "__main__":
    # 使用方法：
    # 1. 找到 Photo-SLAM-L 保存的 cameras.json
    # 2. 指定你想创建的 SuGaR 输入根目录
    JSON_PATH = "./results/replica_rgbd_0/office0/10000_shutdown/cameras.json"
    SUGAR_ROOT = "./sugar_input_office0"
    
    convert_photoslam_to_sugar(JSON_PATH, SUGAR_ROOT)
