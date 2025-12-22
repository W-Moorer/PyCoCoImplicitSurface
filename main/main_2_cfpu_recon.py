#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用生成的CFPU输入文件进行重建尝试
"""

import sys
import os
import numpy as np
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath("."))

# 尝试导入CuPy以支持GPU加速
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("CuPy导入成功，将使用GPU加速")
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    print("CuPy导入失败，将使用CPU计算")

from src.cfpurecon import cfpurecon
from visualize.visualize_cfpu_input import load_cfpu

def main():
    """使用生成的CFPU输入文件进行重建尝试"""
    # 输入目录（使用我们生成的CFPU输入数据）
    input_dir = r"output\cfpu_input\zuheti_surface_cellnormals_cfpu_input"
    
    # 输出目录
    output_dir = r"output\zuheti_surface_cellnormals_cfpu_recon"
    os.makedirs(output_dir, exist_ok=True)
    
    print("读取CFPU输入文件...")
    
    # 读取CFPU输入文件
    nodes, normals, patches, radii, feature_count = load_cfpu(input_dir)
    
    print(f"读取完成！")
    print(f"节点数量: {nodes.shape[0]}")
    print(f"法向量数量: {normals.shape[0]}")
    print(f"补丁数量: {patches.shape[0]}")
    print(f"半径数量: {radii.shape[0] if radii is not None else 0}")
    print(f"特征点数量: {feature_count}")
    
    # 检查数据一致性
    if nodes.shape[0] != normals.shape[0]:
        print("警告：节点数量与法向量数量不匹配！")
        return 1
    
    if patches.shape[0] != radii.shape[0]:
        print("警告：补丁数量与半径数量不匹配！")
        return 1
    
    # 根据用户要求，设置网格大小为128
    gridsize = 128  # 用户要求的网格大小
    
    print(f"\n开始CFPU重建，网格大小: {gridsize}...")
    print(f"使用GPU加速: {GPU_AVAILABLE}")
    start_time = time.time()
    
    try:
        # 添加正则化参数来解决矩阵奇异问题
        potential, X, Y, Z = cfpurecon(
            x=nodes,
            nrml=normals,
            y=patches,
            gridsize=gridsize,
            reginfo={
                'exactinterp': 1,
                'nrmlreg': 1,          # 添加正则化
                'nrmllambda': 1e-6,     # 正则化系数，调整这个值可以解决矩阵奇异问题
                'nrmlschur': 1,         # 使用Schur补加速计算
                'potreg': 1,            # 势能正则化
                'potlambda': 1e-6        # 势能正则化系数
            },
            n_jobs=4,  # 使用4个线程加速
            progress=lambda current, total: print(f"进度: {current}/{total}", end="\r"),
            progress_stage=lambda stage, info: print(f"\n阶段: {stage}")
        )
        
        end_time = time.time()
        print(f"\nCFPU重建完成，耗时: {end_time - start_time:.2f}秒")
        
        # 保存重建结果
        print("保存重建结果...")
        
        # 保存潜在场
        potential_path = os.path.join(output_dir, "potential.npy")
        np.save(potential_path, potential)
        print(f"潜在场保存到: {potential_path}")
        
        # 保存网格坐标
        grid_path = os.path.join(output_dir, "grid.npz")
        np.savez(grid_path, X=X, Y=Y, Z=Z)
        print(f"网格坐标保存到: {grid_path}")
        
        # 保存重建配置
        config_path = os.path.join(output_dir, "recon_config.txt")
        with open(config_path, 'w') as f:
            f.write(f"节点数量: {nodes.shape[0]}\n")
            f.write(f"补丁数量: {patches.shape[0]}\n")
            f.write(f"网格大小: {gridsize}\n")
            f.write(f"重建耗时: {end_time - start_time:.2f}秒\n")
            f.write(f"潜在场形状: {potential.shape}\n")
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
