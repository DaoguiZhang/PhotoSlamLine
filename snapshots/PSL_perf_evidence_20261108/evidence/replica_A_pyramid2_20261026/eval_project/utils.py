import os
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation
from skimage.metrics import structural_similarity as calculate_ssim
import lpips
import torch

# 更加稳健的 Lambda 函数
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x, rank: -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(rank))
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

class LPIPS:
    loss_fn_alex = None

    @staticmethod
    def calculate(img_a, img_b, rank):
        """
        自适应维度处理：无论输入是 [3, H, W] 还是 [1, 3, H, W] 都能正确处理
        """
        def prepare(img):
            # 1. 确保是 FloatTensor
            img = img.to(torch.float32)
            
            # 2. 如果是 4 维 [B, C, H, W]
            if img.dim() == 4:
                # 如果通道在最后 [1, H, W, 3]，转为 [1, 3, H, W]
                if img.shape[-1] == 3:
                    return img.permute(0, 3, 1, 2)
                return img
            
            # 3. 如果是 3 维 [C, H, W] 或 [H, W, 3]
            if img.dim() == 3:
                # [H, W, 3] -> [3, H, W]
                if img.shape[-1] == 3:
                    img = img.permute(2, 0, 1)
                # 增加 Batch 维度 -> [1, 3, H, W]
                return img.unsqueeze(0)
            
            return img

        # 预处理图片，确保 LPIPS 收到的是 [1, 3, H, W]
        t_a = prepare(img_a).to(rank)
        t_b = prepare(img_b).to(rank)
        
        # LPIPS 期望数值范围在 [-1, 1] 之间，如果输入是 [0, 1]，进行映射
        if t_a.max() <= 1.0:
            t_a = t_a * 2.0 - 1.0
            t_b = t_b * 2.0 - 1.0

        if LPIPS.loss_fn_alex == None:
            # 这里的 version='0.1' 对应较新的 lpips 库
            LPIPS.loss_fn_alex = lpips.LPIPS(net='alex', version='0.1').to(rank)
        
        with torch.no_grad():
            return LPIPS.loss_fn_alex(t_a, t_b)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = np.tan((fovY / 2))
    tanHalfFovX = np.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

class MiniCam:
    def __init__(self, width, height, fovx, fovy, world_view_transform):
        self.image_width = width
        self.image_height = height  
        self.FoVx = fovx
        self.FoVy = fovy
        self.znear = 0.01
        self.zfar = 100.0
        
        # 确保输入是 torch tensor
        if not isinstance(world_view_transform, torch.Tensor):
            world_view_transform = torch.tensor(world_view_transform, dtype=torch.float32)
            
        self.world_view_transform = world_view_transform.transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

def loadPose(path):
    """
    带文件检查的轨迹加载
    """
    if not os.path.exists(path):
        print(f"Warning: Pose file not found at {path}")
        return None, None
        
    try:
        # 适配不同版本的 numpy
        pose_data = np.loadtxt(path, delimiter=' ', dtype=str)
        if pose_data.size == 0:
            return None, None
            
        pose_vecs = pose_data[:, 1:].astype(np.float32)
        tstamp = pose_data[:, 0].astype(np.float64)
        
        poses = []
        for pose_vec in pose_vecs:
            pose = np.eye(4)
            # pose_vec 顺序通常为 [tx, ty, tz, qx, qy, qz, qw]
            pose[:3, :3] = Rotation.from_quat(pose_vec[3:]).as_matrix()
            pose[:3, 3] = pose_vec[:3]
            poses.append(pose)
        return poses, tstamp
    except Exception as e:
        print(f"Error loading pose: {e}")
        return None, None

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def focal2fov(focal, pixels):
    return 2*np.arctan(pixels/(2*focal))
