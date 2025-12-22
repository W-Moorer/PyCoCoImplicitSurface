#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算模型可视化 - 四视图布局

该脚本用于可视化CFPU重建结果，提供四个不同角度的视图，不需要原始网格模型。
"""

import numpy as np
import pyvista as pv
from pathlib import Path
import argparse


def main():
    """
    可视化CFPU重建结果，提供四个不同角度的视图
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--recon_dir', type=str, 
                   default='output/leftGear_surface_cellnormals_cfpu_recon_new',
                   help='CFPU重建结果输出目录')
    ap.add_argument('--window_size', type=int, nargs=2, default=[1600, 1200],
                   help='可视化窗口大小 (width height)')
    ap.add_argument('--isosurface', type=float, default=0.0,
                   help='等值面值')
    ap.add_argument('--opacity', type=float, default=1,
                   help='重建曲面不透明度')
    args = ap.parse_args()
    
    # 设置工作目录 - 脚本在main文件夹，需要上一级目录作为根目录
    root = Path(__file__).resolve().parents[1]
    
    # 读取重建结果
    print(f"读取重建结果文件...")
    recon_dir = root / args.recon_dir
    
    # 检查重建结果文件是否存在
    potential_path = recon_dir / 'potential.npy'
    grid_path = recon_dir / 'grid.npz'
    
    if not potential_path.exists() or not grid_path.exists():
        print(f"错误：重建结果文件不存在于 {recon_dir}")
        print("请先运行demo_cfpu_recon.py或main_build_cfpu_input.py生成重建结果")
        return 1
    
    # 加载重建数据
    potential = np.load(potential_path)
    grid_data = np.load(grid_path)
    X = grid_data['X']
    Y = grid_data['Y']
    Z = grid_data['Z']
    
    print(f"读取完成！")
    print(f"重建结果信息：")
    print(f"  潜在场形状: {potential.shape}")
    print(f"  网格形状: {X.shape}")
    
    # 准备重建后的等值面
    print(f"创建可视化...")
    
    # 创建结构化网格并提取等值面
    sg = pv.StructuredGrid(X, Y, Z)
    sg['potential'] = potential.ravel(order='F')
    recon_surface = sg.contour(isosurfaces=[args.isosurface])
    
    # 创建单视图可视化窗口
    plotter = pv.Plotter(window_size=args.window_size)
    
    # 添加重建后的等值面
    plotter.add_mesh(
        recon_surface, 
        color='lightblue', 
        specular=0.3, 
        smooth_shading=True, 
        opacity=args.opacity
    )
    
    # 设置初始视角为等轴测视图
    plotter.view_isometric()
    
    # 添加坐标轴
    plotter.add_axes()
    
    # 显示标题
    plotter.add_title("CFPU重建结果可视化")
    
    # 显示底部提示
    plotter.add_text("左键拖动旋转，滚轮缩放，右键平移", position='lower_left', font_size=10, color='black')
    
    # 显示可视化窗口
    plotter.show()
    
    return 0


if __name__ == '__main__':
    exit(main())