#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main_3_cfpu_recon_specified.py

目的：
- 标准 CFPU 重建脚本。
- 移除 curve_points 相关逻辑。
- 移除 also_save_base 对比逻辑。

行为：
- 读取 CFPU 输入（nodes/normals/patches/radii/feature_count）
- 调用 src.cfpurecon.cfpurecon(...) 进行重建
- 输出：
  - potential.npy
  - grid.npz (X/Y/Z)
  - recon_config.txt
"""

import sys
import os
import time
import argparse
import numpy as np

# 添加项目根目录到 Python 路径
sys.path.append(os.path.abspath("."))

DEFAULT_USE_GPU = True

# 尝试导入 CuPy 以支持 GPU 加速（cfpurecon 内部也会二次判断）
try:
    import cupy as cp  # noqa: F401
    GPU_AVAILABLE = True
    try:
        cp.ones(10)  # smoke test
        print("CuPy GPU加速可用")
    except Exception as e:
        print(f"CuPy导入成功，但GPU不可用: {e}")
        GPU_AVAILABLE = False
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy导入失败，将仅使用CPU计算")

from src.cfpurecon import cfpurecon
from visualize.visualize_cfpu_input import load_cfpu


def parse_arguments():
    p = argparse.ArgumentParser(
        description="CFPU重建脚本"
    )
    p.add_argument("--model", type=str, default="LinkedGear", help="模型名称")
    p.add_argument("--gridsize", type=int, default=256, help="网格大小")
    p.add_argument("--use_gpu", action="store_true", help="使用GPU加速（默认跟随 DEFAULT_USE_GPU）")
    p.add_argument("--threads", type=int, default=-1, help="线程数量（-1=全部）")

    # 输入/输出目录（默认保持与原脚本一致）
    p.add_argument(
        "--input_dir", type=str, default=None,
        help="CFPU输入目录（默认: output\\cfpu_input\\{model}_surface_cellnormals_cfpu_input）"
    )
    p.add_argument(
        "--output_dir", type=str, default=None,
        help="输出目录（默认: output\\{model}_recon_{CPU/GPU}）"
    )

    # 控制 curve_points 影响范围（缩放 feature patch 半径）
    p.add_argument(
        "--feature_scale", type=float, default=1.0,
        help="对 feature patches 半径整体缩放（<1 更局部；默认 1.0）"
    )

    return p.parse_args()


def _default_input_dir(model: str) -> str:
    return rf"output\cfpu_input\{model}_surface_cellnormals_cfpu_input"


def _default_output_dir(model: str, use_gpu: bool) -> str:
    mode_str = "GPU" if use_gpu else "CPU"
    return rf"output\{model}_recon_{mode_str}"


def main():
    args = parse_arguments()

    use_gpu = (args.use_gpu or DEFAULT_USE_GPU) and GPU_AVAILABLE
    if (args.use_gpu or DEFAULT_USE_GPU) and not GPU_AVAILABLE:
        print("警告：请求使用GPU，但CuPy不可用，将使用CPU")
    if use_gpu:
        # 避免 CPU 多线程 + GPU 混用
        if args.threads == -1 or args.threads > 1:
            args.threads = 1

    input_dir = args.input_dir or _default_input_dir(args.model)
    output_dir = args.output_dir or _default_output_dir(args.model, use_gpu)
    os.makedirs(output_dir, exist_ok=True)

    # 读取 CFPU 输入
    print(f"读取CFPU输入: {input_dir}")
    nodes, normals, patches, radii, feature_count = load_cfpu(input_dir)

    print("读取完成！")
    print(f"节点数量: {nodes.shape[0]}")
    print(f"法向量数量: {normals.shape[0]}")
    print(f"补丁数量: {patches.shape[0]}")
    print(f"半径数量: {radii.shape[0] if radii is not None else 0}")
    print(f"特征点数量: {feature_count}")

    if nodes.shape[0] != normals.shape[0]:
        raise ValueError("节点数量与法向量数量不匹配！")
    if patches.ndim != 2 or patches.shape[1] != 3:
        raise ValueError(f"补丁坐标维度错误！ patches.shape={patches.shape}")

    # feature_mask：前 feature_count 个 patch 被认为是 feature patches
    M = patches.shape[0]
    feature_mask = None
    try:
        fc = int(feature_count)
        if fc > 0:
            feature_mask = (np.arange(M) < fc)
    except Exception:
        feature_mask = None

    # 执行重建
    print("开始CFPU重建...")
    start_time = time.time()
    
    potential, X, Y, Z = cfpurecon(
        x=nodes,
        nrml=normals,
        y=patches,
        gridsize=args.gridsize,
        reginfo={
            "exactinterp": 1,
            "nrmlreg": 1,
            "nrmllambda": 1e-6,
            "nrmlschur": 1,
            "potreg": 1,
            "potlambda": 1e-4
        },
        n_jobs=args.threads,
        progress=lambda cur, tot: print(f"进度: {cur}/{tot}", end="\r"),
        progress_stage=lambda stage, info: print(f"\n阶段: {stage}"),
        patch_radii_file=os.path.join(input_dir, "radii.txt"),
        patch_radii_in_world_units=True,
        patch_radii_enforce_coverage=False,
        feature_mask=feature_mask,
        feature_scale=float(args.feature_scale),
        use_gpu=use_gpu,
    )
    
    print(f"\nCFPU重建完成，耗时: {time.time() - start_time:.2f}秒")

    # 保存结果
    np.save(os.path.join(output_dir, "potential.npy"), potential)
    np.savez(os.path.join(output_dir, "grid.npz"), X=X, Y=Y, Z=Z)
    print("已保存 potential.npy / grid.npz")

    # 保存配置
    config_path = os.path.join(output_dir, "recon_config.txt")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("Standard CFPU recon (Clean)\n")
        f.write(f"model: {args.model}\n")
        f.write(f"input_dir: {input_dir}\n")
        f.write(f"output_dir: {output_dir}\n")
        f.write(f"gridsize: {args.gridsize}\n")
        f.write(f"use_gpu: {use_gpu}\n")
        f.write(f"threads: {args.threads}\n")
        f.write(f"feature_scale: {args.feature_scale}\n")
    print(f"已保存配置: {config_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())