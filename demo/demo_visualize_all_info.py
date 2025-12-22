#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化修改后的segment_mesh函数生成的所有信息，包括区域分割、尖锐边缘和间断点
"""

import sys
import os
import numpy as np
import pyvista as pv
import pickle

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath("."))

from src.precompute import read_mesh

def main():
    # 输入和输出文件路径
    input_path = r"output\leftGear_surface_cellnormals\leftGear_segmented_test.vtp"
    base_name = input_path.rsplit('.', 1)[0]
    
    # 读取分割后的网格
    print("读取分割后的网格...")
    mesh = read_mesh(input_path, compute_split_normals=False)
    
    # 读取尖锐边缘信息
    sharp_edges_path = base_name + '_sharp_edges.pkl'
    with open(sharp_edges_path, 'rb') as f:
        sharp_data = pickle.load(f)
    sharp_edges = sharp_data['sharp_edges']
    sharp_edge_lines = sharp_data['sharp_edge_lines']
    
    # 读取尖锐边缘片段和间断点信息
    segments_path = base_name + '_sharp_segments.pkl'
    with open(segments_path, 'rb') as f:
        segments_data = pickle.load(f)
    segments = segments_data['segments']
    junctions = segments_data['junctions']
    turn_points = segments_data['turn_points']
    
    print(f"网格信息：{mesh.n_points} 个点，{mesh.n_cells} 个面")
    print(f"区域数量：{len(np.unique(mesh.cell_data['RegionId']))}")
    print(f"尖锐边缘数量：{len(sharp_edges)}")
    print(f"尖锐边缘片段数量：{len(segments)}")
    print(f"尖锐连接点数量：{len(junctions)}")
    print(f"间断点数量：{len(turn_points)}")
    
    # 构建尖锐边缘的PolyData
    def build_lines_poly(lines, colors):
        pts = []
        segs = []
        for i, ln in enumerate(lines):
            a = ln[0]
            b = ln[1]
            pts.append(a)
            pts.append(b)
            segs.append([2, 2*i, 2*i+1])
        poly = pv.PolyData()
        if pts:
            poly.points = np.array(pts)
            poly.lines = np.array(segs)
            poly["color_idx"] = np.array(colors, dtype=np.uint8)
        return poly
    
    # 为尖锐边缘分配颜色（凸边为红色，凹边为蓝色）
    colors = []
    for e in sharp_edges:
        colors.append(1 if e.get('is_convex', True) else 0)
    
    lines_poly = build_lines_poly(sharp_edge_lines, colors)
    
    # 创建可视化
    p = pv.Plotter(off_screen=True)
    
    # 显示分割后的网格，按区域着色
    p.add_mesh(mesh, scalars='RegionId', cmap='jet', opacity=0.5, show_edges=False, 
              scalar_bar_args={'title': 'Region ID'})
    
    # 显示尖锐边缘
    if lines_poly.n_points > 0:
        p.add_mesh(lines_poly, scalars='color_idx', cmap=['blue', 'red'], 
                  render_lines_as_tubes=True, line_width=3,
                  scalar_bar_args={'title': 'Edge Type'})
    
    # 显示尖锐连接点（绿色）
    if junctions:
        junction_pts = np.array([mesh.points[int(j)] for j in junctions])
        p.add_points(junction_pts, color='green', point_size=12, render_points_as_spheres=True, 
                    label='Junctions')
    
    # 显示间断点（紫色）
    if turn_points:
        turn_pts = np.array([mesh.points[int(tp)] for tp in turn_points])
        p.add_points(turn_pts, color='purple', point_size=15, render_points_as_spheres=True, 
                    label='Turn Points')
    
    # 设置可视化选项
    p.add_axes()
    p.enable_eye_dome_lighting()
    p.background_color = 'white'
    
    # 保存可视化结果
    output_image = base_name + '_all_info_visualization.png'
    print(f"保存可视化结果到 {output_image}...")
    p.show(screenshot=output_image, auto_close=False)
    p.close()
    
    print("可视化完成！")

if __name__ == "__main__":
    main()
