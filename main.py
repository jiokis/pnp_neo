import argparse
import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train.stage2_train import main as train_main
from utils.visual_utils import VisualUtils
from utils.eval_utils import EvaluationMetrics
from model.full_model import DroneCrossViewModel
from config.base_config import BaseConfig

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="无人机跨视角匹配 - 阶段二（尺度预测优化）")
    parser.add_argument("--mode", type=str, default="train",
                       choices=["train", "eval", "visualize"],
                       help="运行模式：train（训练）、eval（评估）、visualize（可视化）")
    parser.add_argument("--weight_path", type=str, default="./weights/stage2_checkpoint/best_val_checkpoint.pth",
                       help="评估/可视化使用的权重路径")
    parser.add_argument("--data_path", type=str, default="./data/real/EuRoC/",
                       help="评估/可视化使用的数据路径")
    return parser.parse_args()

def eval_mode(args):
    """评估模式"""
    print("="*50)
    print("进入评估模式")
    print("="*50)

    # 加载模型
    model = DroneCrossViewModel().to(BaseConfig.DEVICE)
    checkpoint = torch.load(args.weight_path, map_location=BaseConfig.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"已加载权重: {args.weight_path}")

    # 加载数据（简化版，实际需调用data模块）
    from data.real.data_parser import RealDataParserFactory
    from data.real.preprocess import RealDataPreprocessor
    from config.data_config import DataConfig

    parser = RealDataParserFactory.get_parser("euroc", args.data_path)
    preprocessor = RealDataPreprocessor()
    eval_metrics = EvaluationMetrics()

    # 评估单序列
    img_names = sorted(os.listdir(parser.cam0_path))[:10]
    depth_names = sorted(os.listdir(parser.depth_path))[:10]
    total_metrics = {}

    with torch.no_grad():
        for img_name, depth_name in zip(img_names, depth_names):
            # 解析数据
            data = parser.parse_data(img_name, depth_name)
            data = preprocessor(data)
            # 模型预测
            pred_dict = model({
                "ray_dir": data["ray_dir"].unsqueeze(0),
                "ray_3d": data["ray_3d"].unsqueeze(0),
                "image": data["image"].unsqueeze(0)
            })
            # 计算指标
            metrics = eval_metrics.compute_scale_metrics(
                pred_dict["pred_depth"].squeeze(0),
                data["depth"]
            )
            # 累积指标
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v

    # 计算平均值
    avg_metrics = {k: v / len(img_names) for k, v in total_metrics.items()}
    print("评估结果:")
    for k, v in avg_metrics.items():
        print(f"  {k}: {v:.4f}")

def visualize_mode(args):
    """可视化模式"""
    print("="*50)
    print("进入可视化模式")
    print("="*50)

    # 加载模型和数据
    model = DroneCrossViewModel().to(BaseConfig.DEVICE)
    checkpoint = torch.load(args.weight_path, map_location=BaseConfig.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    from data.real.data_parser import RealDataParserFactory
    from data.real.preprocess import RealDataPreprocessor
    from config.data_config import DataConfig

    parser = RealDataParserFactory.get_parser("euroc", args.data_path)
    preprocessor = RealDataPreprocessor()
    visualizer = VisualUtils()

    # 可视化单帧数据
    img_names = sorted(os.listdir(parser.cam0_path))[:2]
    depth_names = sorted(os.listdir(parser.depth_path))[:2]

    with torch.no_grad():
        for i, (img_name, depth_name) in enumerate(zip(img_names, depth_names)):
            data = parser.parse_data(img_name, depth_name)
            data = preprocessor(data)
            pred_dict = model({
                "ray_dir": data["ray_dir"].unsqueeze(0),
                "ray_3d": data["ray_3d"].unsqueeze(0),
                "image": data["image"].unsqueeze(0)
            })

            # 深度图可视化
            visualizer.visualize_depth_map(
                pred_dict["pred_depth"].squeeze(0),
                data["depth"],
                f"depth_visualize_{i}.png"
            )

            # 隐空间可视化（仅第一帧）
            if i == 0:
                visualizer.visualize_latent_space(
                    pred_dict["latent_vecs"],
                    data["ray_3d"].unsqueeze(0),
                    "latent_visualize.png"
                )

        # 跨视角匹配可视化
        data0 = parser.parse_data(img_names[0], depth_names[0])
        data0 = preprocessor(data0)
        data1 = parser.parse_data(img_names[1], depth_names[1])
        data1 = preprocessor(data1)

        pred0 = model({
            "ray_dir": data0["ray_dir"].unsqueeze(0),
            "ray_3d": data0["ray_3d"].unsqueeze(0),
            "image": data0["image"].unsqueeze(0)
        })
        pred1 = model({
            "ray_dir": data1["ray_dir"].unsqueeze(0),
            "ray_3d": data1["ray_3d"].unsqueeze(0),
            "image": data1["image"].unsqueeze(0)
        })

        # 计算匹配索引
        match_idx = torch.argmin(
            torch.cdist(pred1["latent_vecs"].squeeze(0), pred0["latent_vecs"].squeeze(0)),
            dim=-1
        )

        visualizer.visualize_cross_view_matching(
            data0["image"],
            data1["image"],
            match_idx,
            data0["uv"],
            data1["uv"],
            "cross_view_match.png"
        )

    print("所有可视化结果已保存！")

def main():
    args = parse_args()
    if args.mode == "train":
        train_main()
    elif args.mode == "eval":
        eval_mode(args)
    elif args.mode == "visualize":
        visualize_mode(args)
    else:
        print("无效模式！请选择 train/eval/visualize")

if __name__ == "__main__":
    main()