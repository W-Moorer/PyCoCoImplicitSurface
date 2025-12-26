import argparse
import os
import sys
import numpy as np
import pyvista as pv

# 添加项目根目录到路径，确保能导入 src.precompute
sys.path.append(os.path.abspath("."))

try:
    from src.precompute import (
        read_mesh, 
        detect_sharp_edges, 
        detect_sharp_junctions_degree, 
        build_sharp_segments
    )
except ImportError:
    print("错误: 无法导入 src.precompute。请确保在项目根目录下运行此脚本。")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Visualize raw segment topology: Green=Closed, Red=Open")
    parser.add_argument('--input', type=str, required=True, help="Input mesh path (vtp/stl/ply)")
    parser.add_argument('--angle', type=float, default=40.0, help="Sharp edge angle threshold")
    parser.add_argument('--line_width', type=int, default=6, help="Line width for visualization")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: File not found {args.input}")
        return

    print(f"Reading mesh: {args.input}")
    mesh = read_mesh(args.input)

    # 1. 提取尖锐边
    print(f"Detecting sharp edges (Angle={args.angle})...")
    sharp_edges, _ = detect_sharp_edges(mesh, angle_threshold=args.angle)
    print(f"  -> Found {len(sharp_edges)} sharp edges")

    if not sharp_edges:
        print("No sharp edges found.")
        return

    # 2. 构建拓扑段 (Segments)
    junctions = detect_sharp_junctions_degree(mesh, sharp_edges)
    
    # 获取 cell normals (build_sharp_segments 需要)
    if 'cell_normals' not in mesh.cell_data:
        mesh.compute_normals(inplace=True, cell_normals=True, point_normals=False)
    cell_normals = mesh.cell_data.get('Normals', mesh.cell_data.get('cell_normals'))

    print("Building segments...")
    segments = build_sharp_segments(sharp_edges, junctions, mesh.points, cell_normals)
    print(f"  -> Built {len(segments)} segments")

    # 3. 可视化
    p = pv.Plotter()
    p.add_mesh(mesh, color='white', opacity=0.1, style='surface', show_edges=False)

    closed_count = 0
    open_count = 0

    for i, seg in enumerate(segments):
        # 获取该段的所有顶点坐标
        # seg['vertices'] 是点索引列表
        pts = mesh.points[seg['vertices']]
        
        if len(pts) < 2:
            continue

        is_closed = seg['closed']
        
        # 颜色逻辑：闭环=绿，开环=红
        color = 'green' if is_closed else 'red'
        label = "Closed Segments" if is_closed and closed_count == 0 else \
                ("Open Segments" if not is_closed and open_count == 0 else None)
        
        if is_closed:
            closed_count += 1
        else:
            open_count += 1

        # 绘制线段
        line = pv.lines_from_points(pts)
        p.add_mesh(line, color=color, line_width=args.line_width, render_lines_as_tubes=True, label=label)

        # 如果是开环，把起点和终点特意标出来，方便看缺口
        if not is_closed:
            ends = np.vstack([pts[0], pts[-1]])
            p.add_points(ends, color='yellow', point_size=10, render_points_as_spheres=True)

    print(f"Visualization ready:")
    print(f"  - Closed (Green): {closed_count}")
    print(f"  - Open (Red)    : {open_count}")
    
    p.add_legend()
    p.add_axes()
    p.show()

if __name__ == "__main__":
    main()