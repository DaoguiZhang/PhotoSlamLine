import numpy as np
import pyvista as pv
import torch
import cv2
import os
import math
import argparse
from plyfile import PlyData
from renderer import render
from gaussian_model import GaussianModel

# 1. 完善 MiniCam 类，确保包含 sh_degree 属性
class MiniCam:
    def __init__(self, width, height, fovx, fovy, world_view_transform, full_proj_transform, camera_center, sh_degree=3):
        self.image_width = width
        self.image_height = height
        self.FoVx = fovx
        self.FoVy = fovy
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.camera_center = camera_center
        # 核心修复：必须有这个属性，render 才能渲染出彩色
        self.active_sh_degree = sh_degree 

class PhotoSlamProViewer:
    def __init__(self, ply_path, threshold=1000.0):
        self.device = torch.device("cuda")
        
        print(f"Loading Gaussian Model: {ply_path}")
        self.gaussians = GaussianModel(sh_degree=3)
        self.gaussians.load_ply(ply_path)
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        
        self.plotter = pv.Plotter(title="Photo-SLAM-L: Advanced Structural & Color Overlay")
        self.plotter.background_color = [0.05, 0.05, 0.05] 
        
        self._load_geometry_for_pv(ply_path, threshold)
        
        self.plotter.add_key_event('r', self.render_and_blend_color)
        print("\n" + "="*60)
        print("操作指南:")
        print("  - 鼠标左键旋转视图，滚轮缩放")
        print("  - 按下 'R' 键：执行高质量渲染并保存叠加图")
        print("="*60)

    def _load_geometry_for_pv(self, path, threshold):
        print(f"Extracting structural features (Threshold: {threshold})...")
        ply = PlyData.read(path)
        v = ply['vertex']
        fields = v.data.dtype.names
        
        pos = np.stack([v['x'], v['y'], v['z']], axis=1)
        s_keys = ['scale_0', 'scale_1', 'scale_2'] if 'scale_0' in fields else ['scaling_0', 'scaling_1', 'scaling_2']
        scales = np.exp(np.stack([v[s_keys[0]], v[s_keys[1]], v[s_keys[2]]], axis=1))
        
        aniso = np.max(scales, axis=1) / (np.min(scales, axis=1) + 1e-6)
        mask = aniso > threshold
        
        struct_cloud = pv.PolyData(pos[mask])
        self.plotter.add_mesh(struct_cloud, color=[1.0, 0.1, 0.1], 
                             point_size=5, name="struct_mesh", 
                             render_points_as_spheres=True,
                             label=f"Anisotropy > {threshold}")
        self.plotter.add_legend()

    def render_and_blend_color(self):
        print("\n[Action] 正在同步相机并启动混合渲染...")
        
        # 1. 获取 PyVista 当前窗口的相机参数
        cam = self.plotter.camera
        w, h = self.plotter.window_size
        
        pos = np.array(cam.position)
        focal = np.array(cam.focal_point)
        up = np.array(cam.up)

        # 2. 构造 World-to-Camera (W2C) 矩阵
        # 在 3DGS 中，相机坐标系是：X右, Y下, Z前
        z = (focal - pos)
        z /= np.linalg.norm(z)
        x = np.cross(z, up)
        x /= np.linalg.norm(x)
        y = np.cross(x, z) # 此时 y 是向上
        
        # 核心：为了对齐 3DGS 渲染器，Y轴取负（变为向下）
        R = np.stack([x, -y, z]) 
        T = -R @ pos
        
        # 构造 4x4 矩阵并转置（因为 Rasterizer 接收的是 column-major 风格的 tensor）
        world_view = np.eye(4)
        world_view[:3, :3] = R
        world_view[:3, 3] = T
        w2v_tensor = torch.tensor(world_view, device="cuda").float().cuda().transpose(0, 1)

        # 3. 构造投影矩阵 (Projection Matrix)
        fovy = math.radians(cam.view_angle)
        fovx = 2 * math.atan(math.tan(fovy / 2) * (w / h))
        
        z_near, z_far = 0.01, 100.0
        tan_x, tan_y = math.tan(fovx / 2), math.tan(fovy / 2)
        
        # 使用 standard OpenGL 风格投影矩阵
        proj = np.zeros((4, 4))
        proj[0, 0] = 1 / tan_x
        proj[1, 1] = 1 / tan_y
        proj[2, 2] = z_far / (z_far - z_near)
        proj[2, 3] = -(z_far * z_near) / (z_far - z_near)
        proj[3, 2] = 1.0 
        proj_tensor = torch.tensor(proj, device="cuda").float().cuda().transpose(0, 1)

        # 关键点：full_proj 必须由 w2v 和 proj 相乘得到
        full_proj_tensor = (w2v_tensor.transpose(0, 1) @ proj_tensor.transpose(0, 1)).transpose(0, 1)

        # 4. 调用渲染
        view_cam = MiniCam(w, h, fovx, fovy, w2v_tensor, full_proj_tensor, 
                           torch.tensor(pos, device="cuda").float(),
                           sh_degree=self.gaussians.active_sh_degree)

        with torch.no_grad():
            out = render(view_cam, self.gaussians, self.background)
        
        render_img = out["render"].detach().cpu().numpy().transpose(1, 2, 0)
        print(f"Rendering Stats -> Max: {render_img.max():.4f}, Mean: {render_img.mean():.4f}")

        # 5. 后处理与混合 (逻辑同上)
        if render_img.max() > 0:
            render_bgr = cv2.cvtColor(np.clip(render_img * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite("render_debug_color.png", render_bgr)
            
            # 截取 PyVista 红色骨架
            self.plotter.background_color = "black"
            geo_img = self.plotter.screenshot(None, return_img=True)
            geo_bgr = cv2.cvtColor(cv2.resize(geo_img, (w, h)), cv2.COLOR_RGB2BGR)
            self.plotter.background_color = [0.05, 0.05, 0.05]
            
            mask = (geo_bgr[:, :, 2] > 50) 
            hybrid_img = render_bgr.copy()
            hybrid_img[mask] = cv2.addWeighted(render_bgr[mask], 0.3, geo_bgr[mask], 0.7, 0).reshape(-1, 3)
            cv2.imwrite("eval_hybrid_overlay.png", hybrid_img)
            print(">>> 混合成功，已保存 eval_hybrid_overlay.png")
        else:
            print(">>> 警告：当前视角渲染结果为空，请尝试移动相机视角或缩放。")

    def show(self):
        self.plotter.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True)
    parser.add_argument('--threshold', '-t', type=float, default=1000.0)
    args = parser.parse_args()
    
    viewer = PhotoSlamProViewer(args.input, threshold=args.threshold)
    viewer.show()
