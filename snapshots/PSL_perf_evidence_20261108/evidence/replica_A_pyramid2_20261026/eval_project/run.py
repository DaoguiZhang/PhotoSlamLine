import os
import sys
import time
import torch
from tqdm import tqdm, trange
import numpy as np
from PIL import Image
import json
import glob
from renderer import render
from argparse import ArgumentParser, Namespace
from gaussian_model import GaussianModel
from utils import *
from pytorch_msssim import ssim
import yaml
import cv2
import matplotlib.pyplot as plt
import json

# This container's cuDNN sublibrary set is mismatched for the installed torch.
# SSIM uses ordinary 2D convolution; disable cuDNN for the offline evaluator
# only so it falls back to torch's native CUDA kernel.
torch.backends.cudnn.enabled = False


def save_depth_image(depth_tensor, path):
    """ 将深度 Tensor 转换为伪彩色图片并保存 """
    depth_np = depth_tensor.detach().cpu().numpy()
    
    # 自动降维：从 [1, H, W] 或 [3, H, W] 变为 [H, W]
    if depth_np.ndim == 3:
        depth_np = depth_np[0]
    elif depth_np.ndim == 4:
        depth_np = depth_np.squeeze()
        if depth_np.ndim == 3: depth_np = depth_np[0]

    mask = depth_np > 0
    if not mask.any(): return
    
    # 归一化 (1%-99% 分位数，防止极端值干扰)
    vmin = np.percentile(depth_np[mask], 1)
    vmax = np.percentile(depth_np[mask], 99)
    depth_norm = (depth_np - vmin) / (vmax - vmin + 1e-5)
    depth_norm = np.clip(depth_norm, 0, 1)
    
    # 使用 Turbo 色图（更清晰地观察百叶窗细节）
    color_map = plt.get_cmap('turbo')
    depth_color = color_map(depth_norm)[:, :, :3]
    
    depth_img = Image.fromarray(np.uint8(depth_color * 255))
    depth_img.save(path)

def loadReplica(path):
    color_paths = sorted(glob.glob(os.path.join(path, "results/frame*.jpg")))
    tstamp = [float(color_path.split("/")[-1].replace("frame", "").replace(".jpg", "").replace(".png", "")) for color_path in color_paths]
    return color_paths, tstamp

def loadTUM(path):
    if os.path.exists(os.path.join(path, "rgb3")):
        color_paths = sorted(glob.glob(os.path.join(path, "rgb3/*.png")))
    else:
        color_paths = sorted(glob.glob(os.path.join(path, "rgb/*.png")))
    tstamp = [float(color_path.split("/")[-1].replace("frame", "").replace(".jpg", "").replace(".png", "")) for color_path in color_paths]
    return color_paths, tstamp

def loadKITTI(path):
    color_paths = sorted(glob.glob(os.path.join(path, "image_2/*.png")))
    tstamp = np.loadtxt(os.path.join(path, "times.txt"), delimiter=' ', dtype=np.unicode_).astype(np.float32)
    return color_paths, tstamp

def loadEuRoC(path):
    color_paths = sorted(glob.glob(os.path.join(path, 'mav0/cam0/data/*.png')))    
    tstamp = [np.float64(x.split('/')[-1][:-4])/1e9 for x in color_paths]
    return color_paths, tstamp

def associate_frames(tstamp_image, tstamp_pose, max_dt=0.08):
    associations = []
    for i, t in enumerate(tstamp_image):
        j = np.argmin(np.abs(tstamp_pose - t))
        if (np.abs(tstamp_pose[j] - t) < max_dt):
            associations.append((i, j))
    return associations

if __name__ == "__main__":
    parser = ArgumentParser(description="evaluation script parameters")
    parser.add_argument("result_path", type=str, default = None)
    parser.add_argument("gt_path", type=str, default = None)
    parser.add_argument("--correct_scale", action="store_true")
    parser.add_argument("--show_plot", action="store_true")
    args = parser.parse_args()
    
    sh_degree = 3
    gaussians = GaussianModel(sh_degree)
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    dirs = os.listdir(args.result_path)
    
    width, height, fovx, fovy = 0,0,0,0
    render_time = 0
    for file_name in dirs:
        # Result roots also contain files such as eval_guard_shutdown.txt.
        # Only directories are shutdown checkpoint directories.
        if not os.path.isdir(os.path.join(args.result_path, file_name)):
            continue
        if "shutdown" in file_name:
            iter = file_name.split("_")[0]
            ply_path = os.path.join(args.result_path, file_name, "ply/point_cloud/iteration_{}".format(iter), "point_cloud.ply")
            gaussians.load_ply(ply_path)
            with open (os.path.join(args.result_path, file_name, "ply", "cameras.json"), "r") as fin:
                camera_paras = json.load(fin)
            width, height, fx, fy = camera_paras[0]["width"], camera_paras[0]["height"], camera_paras[0]["fx"], camera_paras[0]["fy"]
            fovx = focal2fov(fx, width)
            fovy = focal2fov(fy, height)
            render_time_path = os.path.join(args.result_path, file_name, "render_time.txt")
            if os.path.exists(render_time_path):
                render_time_data = np.loadtxt(render_time_path, delimiter=' ', dtype=np.str_)
                render_time = render_time_data[:, 1].astype(np.float32)

    if "replica" in args.gt_path.lower():
        gt_color_paths, gt_tstamp = loadReplica(args.gt_path)
    elif "kitti" in args.gt_path.lower():
        gt_color_paths, gt_tstamp = loadKITTI(args.gt_path)   
    elif "euroc" in args.gt_path.lower():
        gt_color_paths, gt_tstamp = loadEuRoC(args.gt_path)  
    else:
        gt_color_paths, gt_tstamp = loadTUM(args.gt_path)
    
    pose_path = os.path.join(args.result_path, "CameraTrajectory_TUM.txt")
    # --- 核心修复：检查轨迹文件是否存在 ---
    #if not os.path.exists(pose_path):
    #    print(f"⚠️ 跳过场景 {file_name}: 找不到轨迹文件 {pose_path}")
    #    continue
    # ------------------------------------
    poses, tstamp = loadPose(pose_path)
    associations = associate_frames(tstamp, gt_tstamp)
    
    # 预设畸变参数
    distortion, K = None, None
    if os.path.isfile(os.path.join(args.gt_path, "camera.yaml")):
        with open(os.path.join(args.gt_path, "camera.yaml"), 'r') as fin:
            camera_model = yaml.safe_load(fin)
        K = np.eye(3)
        K[0,0], K[1,1], K[0,2], K[1,2] = camera_model["fx"], camera_model["fy"], camera_model["cx"], camera_model["cy"]
        distortion = np.array(camera_model['distortion']) if 'distortion' in camera_model else None
        re_u, re_v = cv2.initUndistortRectifyMap(K, distortion, None, K, (camera_model["W"], camera_model["H"]), m1type=cv2.CV_32FC1)

    os.makedirs(os.path.join(args.result_path, "image"), exist_ok=True)
    os.makedirs(os.path.join(args.result_path, "depth"), exist_ok=True) # 创建深度文件夹

    psnr_list, ssim_list, lpips_list, time_list = [], [], [], []
    # 在循环开始前初始化一个列表
    tsdf_meta_data = []
    
    for index in trange(len(associations), desc="eval"):
        (result_indx, gt_indx) = associations[index]
        w2c = np.linalg.inv(poses[result_indx])
        cam = MiniCam(width, height, fovx, fovy, w2c)
        
        t0 = time.time()
        out = render(cam, gaussians, background)
        render_image = out["render"] # [3, H, W]
        render_depth = out["depth"]
        t1 = time.time() - t0

        # 处理 GT
        gt_img = Image.open(gt_color_paths[gt_indx])
        gt_np = np.array(gt_img)
        if gt_np.ndim < 3: gt_np = np.stack([gt_np]*3, axis=-1)
        
        if distortion is not None:
            gt_np = cv2.remap(gt_np, re_u, re_v, interpolation=cv2.INTER_LINEAR, borderValue=[0,0,0])

        gt_torch = torch.from_numpy(gt_np).to("cuda").float() / 255.0 # [H, W, 3]

        # 核心：对齐维度为 [1, 3, H, W] 计算指标
        img_a = render_image.unsqueeze(0) # [1, 3, H, W]
        img_b = gt_torch.permute(2, 0, 1).unsqueeze(0) # [1, 3, H, W]

        psnr_val = mse2psnr(img2mse(img_a, img_b), "cuda").item()
        ssim_val = ssim(img_a, img_b, data_range=1.0).item()
        lpips_val = LPIPS.calculate(img_a, img_b, "cuda").item()

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        lpips_list.append(lpips_val)
        time_list.append(t1)

        # 保存图片
        img_name = gt_color_paths[gt_indx].split("/")[-1]
        #rgb_save = (img_a.squeeze().permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        # 增加 .detach() 来切断梯度追踪
        rgb_save = (img_a.detach().squeeze().permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(rgb_save).save(os.path.join(args.result_path, "image", img_name))
        #save_depth_image(render_depth, os.path.join(args.result_path, "depth", "depth_" + img_name))
        depth_base_name = img_name.replace(".jpg", ".npy").replace(".png", ".npy")
        #depth_save_path = os.path.join(args.result_path, "depth_data", depth_base_name)
        #os.makedirs(os.path.dirname(depth_save_path), exist_ok=True)
        #np.save(depth_save_path, render_depth.detach().cpu().numpy())
        
        
        # --- 保存位姿与内参信息 (核心修改) ---
        c2w = np.linalg.inv(w2c)
        # 将 c2w 矩阵转为平铺的 12 个数，并计算 TUM 格式的四元数
        R_mat = c2w[:3, :3]
        t_vec = c2w[:3, 3]
        quat = Rotation.from_matrix(R_mat).as_quat() # x, y, z, w
    
        frame_info = {
            "timestamp": tstamp[result_indx],
            "img_name": img_name,
            "depth_path": depth_base_name,
            "pose_12": c2w[:3, :4].reshape(-1).tolist(), # 12列格式
            "tum_pose": [tstamp[result_indx], t_vec[0], t_vec[1], t_vec[2], quat[0], quat[1], quat[2], quat[3]],
            "intrinsics": {
                "width": int(width), "height": int(height),
                "fx": float(fx), "fy": float(fy),
                "cx": float(width / 2.0), "cy": float(height / 2.0) # 假设主点在中心
            }
        }
        tsdf_meta_data.append(frame_info)
    
    # 循环结束后，保存这个 JSON 文件
    with open(os.path.join(args.result_path, "tsdf_meta.json"), "w") as f:
        json.dump(tsdf_meta_data, f, indent=4)

    with open(os.path.join(args.result_path, "TrackingTime.txt"), "r") as fin:
        tracking_time = fin.readlines()
    #print(tracking_time)
    tracking_time = np.array(tracking_time[:-3]).astype(np.float32)
    #tracking_time = np.loadtxt(os.path.join(args.result_path, "TrackingTime.txt"), delimiter=' ', dtype=np.unicode_)
    #tracking_time = tracking_time[:-3].astype(np.float32) # supposed last three lines is comment
    #np.savetxt(os.path.join(args.result_path, "renderng_time.txt"), time_list)
    #np.savetxt(os.path.join(args.result_path, "eval.txt"), [np.mean(psnr_list), np.mean(ssim_list), np.mean(lpips_list)])
    with open(os.path.join(args.result_path, "eval.txt"), "w") as fout:
        fout.write("psnr: {}\n".format(np.mean(psnr_list)))
        fout.write("ssim: {}\n".format(np.mean(ssim_list)))
        fout.write("lpips: {}\n".format(np.mean(lpips_list)))
        fout.write("tracking s: {}\n".format(np.mean(tracking_time)))
        fout.write("tracking FPS: {}\n".format(1/np.mean(tracking_time)))

        fout.write("rendering ms: {}\n".format(np.mean(render_time)))
        fout.write("rendering FPS: {}\n".format(1000/np.mean(render_time)))

    #gt = view.original_image[0:3, :, :]

    #### pose evaluation
    import evo
    from evo.core.trajectory import PoseTrajectory3D
    from evo.tools import file_interface
    from evo.core import sync
    import evo.main_ape as main_ape
    from evo.core.metrics import PoseRelation
    # load estimated poses
    traj_est = file_interface.read_tum_trajectory_file(pose_path)
    # load gt pose
    if "kitti" in args.gt_path.lower():
        def loadKITTIPose(gt_path):
            scene = gt_path.split("/")[-1]
            gt_file = gt_path.replace(scene, 'poses/{}.txt'.format(scene))
            pose_quat = []
            with open(gt_file, "r") as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    line = lines[i].split()
                    #print(line)
                    c2w = np.array(list(map(float, line))).reshape(3, 4)
                    #print(c2w)
                    quat = np.zeros(7)
                    quat[:3] = c2w[:3, 3]
                    quat[3:] = Rotation.from_matrix(c2w[:3, :3]).as_quat()
                    pose_quat.append(quat)
            pose_quat = np.array(pose_quat)

            return pose_quat
        
        pose_quat = loadKITTIPose(args.gt_path)
        traj_ref = PoseTrajectory3D(positions_xyz=pose_quat[:,:3],         
                                        orientations_quat_wxyz=pose_quat[:,3:],         
                                        timestamps=np.array(gt_tstamp))

        #scene = args.gt_path.split("/")[-1]
        #gt_file = args.gt_path.replace(scene, 'poses/{}.txt'.format(scene))
        #print(gt_file)
        #traj_ref = file_interface.read_kitti_poses_file(gt_file)
    elif "replica" in args.gt_path.lower():
        gt_file = os.path.join(args.gt_path, 'pose_TUM.txt')
        traj_ref = file_interface.read_tum_trajectory_file(gt_file)
        #if not os.path.isfile(gt_file):
    elif "euroc" in args.gt_path.lower():
        gt_file = os.path.join(args.gt_path, 'mav0/state_groundtruth_estimate0/data.csv')
        traj_ref = file_interface.read_euroc_csv_trajectory(gt_file)
    else:
        gt_file = os.path.join(args.gt_path, 'groundtruth.txt')   
        traj_ref = file_interface.read_tum_trajectory_file(gt_file)

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff=0.1)
    result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=args.correct_scale)
    result_rotation_part = main_ape.ape(traj_ref, traj_est, est_name='rot', pose_relation=PoseRelation.rotation_part, 
                                        align=True, correct_scale=args.correct_scale)
    #使用角度的误差
    #result_rotation_part = main_ape.ape(
    #traj_ref, traj_est, 
    #est_name='rot', 
    #pose_relation=PoseRelation.rotation_angle_deg, # <--- 关键修改：计算角度误差(度)
    #align=True, 
    #correct_scale=False # <--- 关键修改：旋转不需要缩放
#)

    out_path=os.path.join(args.result_path, "metrics_traj.txt")
    with open(out_path, 'w') as fp:
        fp.write(result.pretty_str())
        fp.write(result_rotation_part.pretty_str())
    print(result)

    if args.show_plot:
        from evo.tools import plot
        from evo.tools.plot import PlotMode
        import matplotlib.pyplot as plt
        import copy
        traj_est_aligned = copy.deepcopy(traj_est)
        traj_est_aligned.align(traj_ref, correct_scale=True)
        fig = plt.figure()
        traj_by_label = {
            "estimate (not aligned)": traj_est,
            "estimate (aligned)": traj_est_aligned,
            "reference": traj_ref
        }
        plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
        plt.show()
