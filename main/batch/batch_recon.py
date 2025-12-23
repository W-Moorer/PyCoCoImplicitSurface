#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量CFPU重建脚本

该脚本会批量处理output/cfpu_input文件夹中的所有CFPU输入数据，
为每个模型执行CFPU重建，并将结果保存到对应的输出目录中。
"""

import sys
import os
import numpy as np
import time
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath("."))

# 全局配置：是否默认使用GPU加速（True=默认使用GPU，False=默认使用CPU）
DEFAULT_USE_GPU = False

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
    parser = argparse.ArgumentParser(description='批量CFPU重建脚本')
    parser.add_argument('--gridsize', type=int, default=256,
                        help='网格大小（默认：256）')
    parser.add_argument('--use_gpu', action='store_true',
                        help='使用GPU加速（默认：使用全局配置DEFAULT_USE_GPU的值）')
    parser.add_argument('--threads', type=int, default=-1,
                        help='线程数量（默认：-1，使用全部线程）')
    return parser.parse_args()


def process_single_model(cfpu_dir, output_base_dir, gridsize=256, use_gpu=False, threads=-1):
    """
    处理单个CFPU模型的函数
    
    Args:
        cfpu_dir: CFPU输入数据目录
        output_base_dir: 输出基础目录
        gridsize: 网格大小
        
    Returns:
        bool: 处理是否成功
    """
    model_name = cfpu_dir.name
    print(f"\n开始处理模型: {model_name}")
    
    # 创建输出目录
    mode_str = "GPU" if use_gpu else "CPU"
    output_dir = output_base_dir / model_name.replace('_surface_cellnormals_cfpu_input', f'_recon_{mode_str}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 检查是否已经处理过
    config_file = output_dir / "recon_config.txt"
    if config_file.exists():
        print(f"模型 {model_name} 已经处理过，跳过")
        return True
    
    try:
        # 读取CFPU输入文件
        print("读取CFPU输入文件...")
        nodes, normals, patches, radii, feature_count = load_cfpu(cfpu_dir)
        
        print(f"读取完成！")
        print(f"节点数量: {nodes.shape[0]}")
        print(f"法向量数量: {normals.shape[0]}")
        print(f"补丁数量: {patches.shape[0]}")
        print(f"半径数量: {radii.shape[0] if radii is not None else 0}")
        print(f"特征点数量: {feature_count}")
        
        # 检查数据一致性
        if nodes.shape[0] != normals.shape[0]:
            print(f"警告：节点数量与法向量数量不匹配！")
            return False
        
        if patches.shape[0] != radii.shape[0]:
            print(f"警告：补丁数量与半径数量不匹配！")
            return False
        
        # 开始CFPU重建
        print(f"开始CFPU重建，网格大小: {gridsize}...")
        print(f"使用GPU加速: {use_gpu}")
        print(f"线程数量: {threads if threads != -1 else '全部线程'}")
        start_time = time.time()
        
        # 执行CFPU重建
        potential, X, Y, Z = cfpurecon(
            x=nodes,
            nrml=normals,
            y=patches,
            gridsize=gridsize,
            reginfo={
                'exactinterp': 1,
                'nrmlreg': 1,          # 添加正则化
                'nrmllambda': 1e-6,     # 正则化系数
                'nrmlschur': 1,         # 使用Schur补加速计算
                'potreg': 1,            # 势能正则化
                'potlambda': 1e-6        # 势能正则化系数
            },
            n_jobs=threads,  # 使用指定的线程数
            progress=lambda current, total: print(f"进度: {current}/{total}", end="\r"),
            progress_stage=lambda stage, info: print(f"\n阶段: {stage}"),
            patch_radii_file=cfpu_dir / "radii.txt",
            patch_radii_in_world_units=True,
            patch_radii_enforce_coverage=False,
            use_gpu=use_gpu
        )
        
        end_time = time.time()
        print(f"\nCFPU重建完成，耗时: {end_time - start_time:.2f}秒")
        
        # 保存重建结果
        print("保存重建结果...")
        
        # 保存潜在场
        potential_path = output_dir / "potential.npy"
        np.save(potential_path, potential)
        print(f"潜在场保存到: {potential_path}")
        
        # 保存网格坐标
        grid_path = output_dir / "grid.npz"
        np.savez(grid_path, X=X, Y=Y, Z=Z)
        print(f"网格坐标保存到: {grid_path}")
        
        # 保存重建配置（使用UTF-8编码避免乱码）
        config_path = output_dir / "recon_config.txt"
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(f"模型名称: {model_name}\n")
            f.write(f"节点数量: {nodes.shape[0]}\n")
            f.write(f"补丁数量: {patches.shape[0]}\n")
            f.write(f"网格大小: {gridsize}\n")
            f.write(f"重建耗时: {end_time - start_time:.2f}秒\n")
            f.write(f"潜在场形状: {potential.shape}\n")
            f.write(f"使用GPU加速: {use_gpu}\n")
            f.write(f"线程数量: {threads if threads != -1 else '全部线程'}\n")
        print(f"重建配置保存到: {config_path}")
        
        print(f"模型 {model_name} 重建完成！")
        return True
        
    except Exception as e:
        print(f"\n模型 {model_name} 重建失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 保存错误信息
        error_file = output_dir / "error_log.txt"
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write(f"模型: {model_name}\n")
            f.write(f"错误信息: {str(e)}\n")
            f.write(f"错误详情:\n")
            traceback.print_exc(file=f)
        
        return False


def main():
    """批量CFPU重建主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 检查GPU可用性和用户选择
    use_gpu = (args.use_gpu or DEFAULT_USE_GPU) and GPU_AVAILABLE
    if (args.use_gpu or DEFAULT_USE_GPU) and not GPU_AVAILABLE:
        print("警告：请求使用GPU，但CuPy不可用，将使用CPU")

    if use_gpu:
        # GPU 模式下不建议再开大量 CPU 线程（会造成 GPU kernel 过碎 + 争用），这里强制单线程驱动 GPU
        if args.threads == -1 or args.threads > 1:
            args.threads = 1
    
    # 输入目录（CFPU输入数据目录）
    cfpu_input_dir = Path(r"output\cfpu_input")
    
    # 输出基础目录
    output_base_dir = Path(r"output")
    
    # 网格大小
    gridsize = args.gridsize
    
    print("开始批量CFPU重建...")
    print(f"CFPU输入目录: {cfpu_input_dir}")
    print(f"输出基础目录: {output_base_dir}")
    print(f"网格大小: {gridsize}")
    print(f"使用GPU加速: {use_gpu}")
    print(f"线程数量: {args.threads if args.threads != -1 else '全部线程'}")
    
    # 获取所有CFPU输入目录
    cfpu_dirs = []
    if cfpu_input_dir.exists():
        for item in cfpu_input_dir.iterdir():
            if item.is_dir() and '_surface_cellnormals_cfpu_input' in item.name:
                cfpu_dirs.append(item)
    
    if not cfpu_dirs:
        print(f"在 {cfpu_input_dir} 中没有找到CFPU输入目录")
        return 1
    
    print(f"找到 {len(cfpu_dirs)} 个CFPU模型目录")
    
    # 统计处理结果
    success_count = 0
    failed_count = 0
    
    # 批量处理所有模型
    for i, cfpu_dir in enumerate(cfpu_dirs, 1):
        print(f"\n{'='*60}")
        print(f"处理进度: {i}/{len(cfpu_dirs)}")
        
        if process_single_model(cfpu_dir, output_base_dir, gridsize, use_gpu, args.threads):
            success_count += 1
        else:
            failed_count += 1
    
    # 输出处理结果
    print(f"\n{'='*60}")
    print("批量重建完成！")
    print(f"成功处理: {success_count} 个模型")
    print(f"处理失败: {failed_count} 个模型")
    print(f"输出目录: {output_base_dir}")
    
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())