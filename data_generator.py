import torch
import numpy as np
from scipy.spatial.transform import Rotation as R


def generate_synthetic_data(batch_size=8, num_rays=2048, depth_range=[3, 30], num_rot_views=2):
    # 直接指定设备为GPU（避免后续to(DEVICE)的额外开销）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 显式指定GPU
    u = torch.randint(0, 640, (num_rays,), device=device)
    v = torch.randint(0, 480, (num_rays,), device=device)

    # 1. 相机内参直接在GPU生成
    K = torch.tensor([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=torch.float32, device=device)

    # 2. 原始3D场景在GPU生成
    scene_3d_points = []
    gt_depth_list = []
    for _ in range(batch_size):
        # 像素坐标直接在GPU生成
        u = torch.randint(0, 640, (num_rays,), device=device)
        v = torch.randint(0, 480, (num_rays,), device=device)
        pixel_hom = torch.stack([u, v, torch.ones(num_rays, device=device)], dim=-1).float()
        # 射线方向计算（全程GPU）
        ray_dir = torch.matmul(torch.inverse(K), pixel_hom.T).T  # 矩阵运算在GPU
        ray_dir = torch.nn.functional.normalize(ray_dir, dim=-1)
        # 真值深度在GPU生成
        gt_depth = torch.rand(num_rays, device=device) * (depth_range[1] - depth_range[0]) + depth_range[0]
        gt_depth_list.append(gt_depth)
        # 3D点坐标（GPU计算）
        scene_3d = ray_dir * gt_depth.unsqueeze(-1)
        scene_3d_points.append(scene_3d)

    # 3. 旋转视角数据也直接在GPU生成（省略部分重复代码，核心是所有张量都用device=device）
    all_rays = []
    all_ray_3d = []
    all_cam_extrinsic = []
    all_gt_depth = []
    all_gt_anchor_idx = []
    for b in range(batch_size):
        scene_3d = scene_3d_points[b]
        original_gt_depth = gt_depth_list[b]
        for v in range(num_rot_views):
            # SO(3)旋转矩阵在GPU生成
            rot_euler = np.random.uniform(-30, 30, 3) * np.pi / 180
            R_so3 = torch.tensor(
                R.from_euler('xyz', rot_euler).as_matrix(),
                dtype=torch.float32,
                device=device  # 直接放GPU
            )
            # 齐次外参矩阵在GPU
            extrinsic = torch.eye(4, dtype=torch.float32, device=device)
            extrinsic[:3, :3] = R_so3
            all_cam_extrinsic.append(extrinsic)
            # 旋转后3D点和射线（全程GPU计算）
            rotated_3d = torch.matmul(R_so3, scene_3d.T).T  # GPU矩阵乘法
            rotated_ray_dir = torch.nn.functional.normalize(rotated_3d, dim=-1)
            all_rays.append(rotated_ray_dir)
            all_ray_3d.append(rotated_3d)
            all_gt_depth.append(original_gt_depth)
            # 锚框索引在GPU生成
            gt_anchor_idx = torch.zeros(num_rays, dtype=torch.long, device=device)
            # ... 锚框索引计算不变 ...
            all_gt_anchor_idx.append(gt_anchor_idx)

    # 所有返回张量已在GPU，无需再to(DEVICE)
    return {
        "rays": torch.stack(all_rays),
        "ray_3d_coords": torch.stack(all_ray_3d),
        "gt_depth": torch.stack(all_gt_depth),
        "gt_anchor_idx": torch.stack(all_gt_anchor_idx),
        "anchor_centers": torch.tensor([4.5, 9, 18, 27], dtype=torch.float32, device=device),
        "anchor_radius": torch.tensor([0.9, 3.0, 3.0, 3.0], dtype=torch.float32, device=device),
        "K": K,
        "cam_extrinsic": torch.stack(all_cam_extrinsic),
        "batch_size": batch_size,
        "num_rot_views": num_rot_views
    }