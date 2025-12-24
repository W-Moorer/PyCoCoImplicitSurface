#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用生成的CFPU输入文件进行重建尝试（做法A：曲线约束耦合进 exactinterp residual）
"""

import sys
import os
import json
import numpy as np
import time
import multiprocessing
import argparse

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath("."))

# 全局配置：是否默认使用GPU加速（True=默认使用GPU，False=默认使用CPU）
DEFAULT_USE_GPU = True

# 尝试导入CuPy以支持GPU加速
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("CuPy导入成功，GPU加速可用")
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    print("CuPy导入失败，将仅使用CPU计算")

from src.cfpurecon import cfpurecon
from visualize.visualize_cfpu_input import load_cfpu

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CFPU重建脚本')
    parser.add_argument('--model', type=str, default='LinkedGear',
                        help='模型名称（默认：PressureLubricatedCam）')
    parser.add_argument('--gridsize', type=int, default=256,
                        help='网格大小（默认：256）')
    parser.add_argument('--use_gpu', action='store_true',
                        help='使用GPU加速（默认：使用全局配置DEFAULT_USE_GPU的值）')
    parser.add_argument('--threads', type=int, default=-1,
                        help='线程数量（默认：-1，使用全部线程）')
    return parser.parse_args()


def main():
    """使用生成的CFPU输入文件进行重建尝试"""
    args = parse_arguments()

    use_gpu = (args.use_gpu or DEFAULT_USE_GPU) and GPU_AVAILABLE
    if (args.use_gpu or DEFAULT_USE_GPU) and not GPU_AVAILABLE:
        print("警告：请求使用GPU，但CuPy不可用，将使用CPU")

    if use_gpu:
        if args.threads == -1 or args.threads > 1:
            args.threads = 1

    input_dir = rf"output\cfpu_input\{args.model}_surface_cellnormals_cfpu_input"

    mode_str = "GPU" if use_gpu else "CPU"
    output_dir = rf"output\{args.model}_recon_{mode_str}"
    os.makedirs(output_dir, exist_ok=True)

    print("读取CFPU输入文件...")
    nodes, normals, patches, radii, feature_count = load_cfpu(input_dir)

    print(f"读取完成！")
    print(f"节点数量: {nodes.shape[0]}")
    print(f"法向量数量: {normals.shape[0]}")
    print(f"补丁数量: {patches.shape[0]}")
    print(f"半径数量: {radii.shape[0] if radii is not None else 0}")
    print(f"特征点数量: {feature_count}")

    if nodes.shape[0] != normals.shape[0]:
        print("警告：节点数量与法向量数量不匹配！")
        return 1
    if patches.shape[0] != radii.shape[0]:
        print("警告：补丁数量与半径数量不匹配！")
        return 1

    gridsize = args.gridsize

    # ===== 新增：读取尖锐边曲线约束（如果存在）=====
    curve_points = None
    curve_patch_map = None

    curve_raw_path = os.path.join(input_dir, "sharp_curve_points_raw.npy")
    if os.path.exists(curve_raw_path):
        curve_points = np.load(curve_raw_path)
        print(f"加载曲线点: {curve_raw_path} | shape={curve_points.shape}")
    else:
        print("未找到 sharp_curve_points_raw.npy：本次不使用曲线约束")

    curve_map_path = os.path.join(input_dir, "sharp_curve_feature_patch_map.json")
    if os.path.exists(curve_map_path):
        with open(curve_map_path, "r", encoding="utf-8") as f:
            curve_patch_map = json.load(f)
        print(f"加载曲线映射: {curve_map_path} | patches={len(curve_patch_map)}")
    else:
        print("未找到 sharp_curve_feature_patch_map.json：将由 cfpurecon 在运行时按球域自动分配（默认仅feature patch）")

    print(f"\n开始CFPU重建，网格大小: {gridsize}...")
    print(f"使用GPU加速: {use_gpu}")
    print(f"线程数量: {args.threads if args.threads != -1 else '全部线程'}")
    start_time = time.time()

    try:
        potential, X, Y, Z = cfpurecon(
            x=nodes,
            nrml=normals,
            y=patches,
            gridsize=gridsize,
            reginfo={
                'exactinterp': 1,
                'nrmlreg': 1,
                'nrmllambda': 1e-6,
                'nrmlschur': 1,
                'potreg': 1,
                'potlambda': 1e-4
            },
            n_jobs=args.threads,
            progress=lambda current, total: print(f"进度: {current}/{total}", end="\r"),
            progress_stage=lambda stage, info: print(f"\n阶段: {stage}"),
            patch_radii_file=os.path.join(input_dir, "radii.txt"),
            patch_radii_in_world_units=True,
            patch_radii_enforce_coverage=False,
            use_gpu=use_gpu,

            # ===== 新增参数：做法A曲线约束 =====
            curve_points=curve_points,
            curve_points_in_unit=False,          # 我们传的是 raw(world) 坐标
            curve_patch_map=curve_patch_map,
            curve_max_points_per_patch=200,
            curve_only_feature_patches=True
        )

        end_time = time.time()
        print(f"\nCFPU重建完成，耗时: {end_time - start_time:.2f}秒")

        print("保存重建结果...")

        potential_path = os.path.join(output_dir, "potential.npy")
        np.save(potential_path, potential)
        print(f"潜在场保存到: {potential_path}")

        grid_path = os.path.join(output_dir, "grid.npz")
        np.savez(grid_path, X=X, Y=Y, Z=Z)
        print(f"网格坐标保存到: {grid_path}")

        config_path = os.path.join(output_dir, "recon_config.txt")
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(f"模型名称: {args.model}\n")
            f.write(f"节点数量: {nodes.shape[0]}\n")
            f.write(f"补丁数量: {patches.shape[0]}\n")
            f.write(f"网格大小: {gridsize}\n")
            f.write(f"重建耗时: {end_time - start_time:.2f}秒\n")
            f.write(f"潜在场形状: {potential.shape}\n")
            f.write(f"使用GPU加速: {use_gpu}\n")
            f.write(f"线程数量: {args.threads if args.threads != -1 else '全部线程'}\n")
            f.write(f"使用曲线约束: {curve_points is not None}\n")
            if curve_points is not None:
                f.write(f"曲线点数量: {curve_points.shape[0]}\n")
                f.write(f"每patch最大曲线点: 200\n")
        print(f"重建配置保存到: {config_path}")

        print("\n重建完成！")
        return 0

    except Exception as e:
        print(f"\nCFPU重建失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
