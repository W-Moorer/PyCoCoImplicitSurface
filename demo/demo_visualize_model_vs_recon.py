#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化对比原始网格模型与重构后的曲面
"""

import numpy as np
import pyvista as pv
from pathlib import Path
import argparse


def main():
    """
    可视化对比原始网格模型与重构后的曲面
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_mesh', type=str, 
                   default='input/leftGear_surface_cellnormals.vtp',
                   help='原始三角网格模型文件')
    ap.add_argument('--recon_dir', type=str, 
                   default='output/leftGear_surface_cellnormals_cfpu_recon_new',
                   help='CFPU重建结果输出目录')
    args = ap.parse_args()
    
    # 设置工作目录 - 脚本在demo文件夹，需要上一级目录作为根目录
    root = Path(__file__).resolve().parents[1]
    
    # 读取原始三角网格模型
    print(f"读取原始三角网格模型...")
    input_mesh = root / args.input_mesh
    original_mesh = pv.read(input_mesh)
    
    # 读取重建结果
    print(f"读取重建结果文件...")
    recon_dir = root / args.recon_dir
    
    # 检查重建结果文件是否存在
    potential_path = recon_dir / 'potential.npy'
    grid_path = recon_dir / 'grid.npz'
    
    if not potential_path.exists() or not grid_path.exists():
        print(f"错误：重建结果文件不存在于 {recon_dir}")
        print("请先运行demo_cfpu_recon.py生成重建结果")
        return 1
    
    # 加载重建数据
    potential = np.load(potential_path)
    grid_data = np.load(grid_path)
    X = grid_data['X']
    Y = grid_data['Y']
    Z = grid_data['Z']
    
    print(f"读取完成！")
    print(f"原始模型信息：")
    print(f"  点数量: {original_mesh.n_points}")
    print(f"  面数量: {original_mesh.n_cells}")
    print(f"重建结果信息：")
    print(f"  潜在场形状: {potential.shape}")
    print(f"  网格形状: {X.shape}")
    
    # 准备重建后的等值面
    print(f"创建可视化...")
    
    # 创建结构化网格并提取等值面
    sg = pv.StructuredGrid(X, Y, Z)
    sg['potential'] = potential.ravel(order='F')
    recon_surface = sg.contour(isosurfaces=[0.0])
    
    # 创建单视图可视化窗口，同一个视图中显示两个模型
    plotter = pv.Plotter(window_size=[1200, 1000])
    
    plotter.add_title("原始网格模型与CFPU重建曲面对比")
    
    # 添加原始网格模型（灰色，半透明，带边缘）
    plotter.add_mesh(
        original_mesh, 
        color='lightgray', 
        specular=0.1, 
        smooth_shading=True, 
        opacity=0.4,
        show_edges=True,
        edge_color='gray',
        name='original_mesh',
        label='原始模型'
    )
    
    # 添加重建后的等值面（蓝色，半透明，更突出）
    plotter.add_mesh(
        recon_surface, 
        color='lightblue', 
        specular=0.3, 
        smooth_shading=True, 
        opacity=0.7,
        name='recon_surface',
        label='重建曲面'
    )
    
    # 添加坐标轴
    plotter.add_axes()
    
    # 添加图例
    plotter.add_legend(bcolor='w')
    
    # 添加底部提示
    plotter.add_text("灰色：原始模型 | 蓝色：重建曲面", position='upper_right', font_size=10)
    plotter.add_text("左键拖动旋转，滚轮缩放，右键平移", position='lower_left', font_size=10)
    
    # 显示可视化窗口
    plotter.show()
    
    return 0


if __name__ == '__main__':
    exit(main())
