#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化生成尖锐边缘片段和间断点可视化
不需要单独输出基础尖锐边缘信息，直接生成包含间断点的可视化结果
"""

import sys
import os
import json

# 添加项目根目录到Python路径，确保能导入precompute模块
sys.path.append(os.path.abspath("."))

from src.precompute import read_mesh, detect_sharp_edges, detect_sharp_junctions_degree, build_sharp_segments

def main():
    # 输入文件路径
    input_path = r"input\zuheti_surface_cellnormals.vtp"
    
    # 输出目录
    output_dir = r"output\zuheti_surface_cellnormals"
    
    # 参数设置
    angle_threshold = 30.0
    edge_split_threshold = None
    require_step_face_id_diff = False
    angle_turn_threshold = 90.0  # 间断点角度阈值
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 读取网格文件
    print("读取网格文件...")
    mesh = read_mesh(input_path, compute_split_normals=False)
    
    # 2. 检测尖锐边缘
    print("检测尖锐边缘...")
    sharp_edges, sharp_edge_lines = detect_sharp_edges(mesh, angle_threshold, edge_split_threshold, require_step_face_id_diff)
    print(f"检测到 {len(sharp_edges)} 条尖锐边缘")
    
    # 3. 检测尖锐连接点
    print("检测尖锐连接点...")
    junctions = detect_sharp_junctions_degree(mesh, sharp_edges)
    print(f"检测到 {len(junctions)} 个尖锐连接点")
    
    # 4. 确保有单元格法线
    print("计算单元格法线...")
    if 'Normals' not in mesh.cell_data:
        mesh.compute_normals(inplace=True, cell_normals=True, point_normals=False, split_vertices=False)
    
    # 5. 构建尖锐边缘片段，包含间断点信息
    print("构建尖锐边缘片段...")
    segments = build_sharp_segments(sharp_edges, junctions, mesh.points, mesh.cell_data['Normals'], angle_turn_threshold)
    print(f"构建了 {len(segments)} 个尖锐边缘片段")
    
    # 6. 计算并打印间断点数量
    turn_points = set()
    for seg in segments:
        turn_points.update(seg['turn_splits'])
    print(f"检测到 {len(turn_points)} 个间断点")
    
    # 7. 保存片段信息为JSON文件，用于可视化
    segments_debug_path = os.path.join(output_dir, "sharp_segments_debug.json")
    print(f"保存片段信息到 {segments_debug_path}...")
    
    # 构建调试信息结构
    debug_segments = []
    for gid, seg in enumerate(segments):
        verts = [int(v) for v in seg['vertices']]
        vpts = [mesh.points[v].tolist() for v in verts]
        edges_list = [(int(a), int(b)) for (a, b) in seg['edges']]
        
        debug_segments.append({
            'id': int(gid),
            'closed': bool(seg['closed']),
            'vertices': verts,
            'vertex_points': vpts,
            'edges': edges_list,
            'turn_splits': [int(x) for x in seg.get('turn_splits', [])]
        })
    
    with open(segments_debug_path, 'w') as f:
        json.dump({'segments': debug_segments}, f)
    
    # 8. 运行可视化脚本，生成包含间断点的可视化结果
    print("生成包含间断点的可视化结果...")
    screenshot_path = os.path.join(output_dir, "sharp_edges_with_turn_points.png")
    sharp_edges_pkl = os.path.join(output_dir, "zuheti_surface_cellnormals_segmented_sharp_edges.pkl")
    
    # 使用系统命令运行可视化脚本
    cmd = f"python visualize/visualize_sharp_edges_junctions.py --input {input_path} --sharp_edges_pkl {sharp_edges_pkl} --segments_debug {segments_debug_path} --off_screen --screenshot {screenshot_path}"
    os.system(cmd)
    
    print("\n自动化处理完成！")
    print(f"生成的结果文件：")
    print(f"  - 尖锐边缘片段信息：{segments_debug_path}")
    print(f"  - 包含间断点的可视化：{screenshot_path}")

if __name__ == "__main__":
    main()
