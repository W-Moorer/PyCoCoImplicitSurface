import argparse
import os
import sys
import numpy as np
import pyvista as pv

# 添加项目根目录到路径
sys.path.append(os.path.abspath("."))

try:
    from src.precompute import (
        read_mesh, 
        detect_sharp_edges, 
        detect_sharp_junctions_degree
    )
except ImportError:
    print("错误: 无法导入 src.precompute。")
    sys.exit(1)

# ==============================================================================
# 修复后的 Segments 构建逻辑 (包含 Turn Angle Check)
# ==============================================================================
def build_sharp_segments_fixed(edges, junctions, points, cell_normals=None, angle_turn_threshold=90.0):
    def calculate_turn_angle(prev_idx, cur_idx, next_idx):
        v1 = points[cur_idx] - points[prev_idx]
        v2 = points[next_idx] - points[cur_idx]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-12 or n2 < 1e-12: return 0.0
        v1 /= n1; v2 /= n2
        dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
        return np.degrees(np.arccos(dot))

    adj = {}
    for i, e in enumerate(edges):
        u, v = int(e['point1_idx']), int(e['point2_idx'])
        adj.setdefault(u, []).append(i)
        adj.setdefault(v, []).append(i)

    visited = set()
    segments = []

    # Stage 1: Starts
    starts = [p for p in adj.keys() if (p in junctions) or (len(adj[p]) != 2)]
    
    for s in starts:
        for start_edge_idx in list(adj[s]):
            if start_edge_idx in visited: continue
            
            path = [s]
            curr_edge_idx = start_edge_idx
            visited.add(curr_edge_idx)
            
            e_obj = edges[curr_edge_idx]
            u, v = int(e_obj['point1_idx']), int(e_obj['point2_idx'])
            curr = v if u == s else u
            prev = s
            path.append(curr)
            
            while True:
                if (curr in junctions) or (len(adj[curr]) != 2) or (curr == s): break
                candidates = [ne for ne in adj[curr] if ne != curr_edge_idx]
                if not candidates: break

                best_next = None
                best_angle = 1e9
                for ne_idx in candidates:
                    e_next = edges[ne_idx]
                    nu, nv = int(e_next['point1_idx']), int(e_next['point2_idx'])
                    nxt_node = nv if nu == curr else nu
                    ang = calculate_turn_angle(prev, curr, nxt_node)
                    if ang < best_angle:
                        best_angle = ang
                        best_next = ne_idx
                
                if best_next is None: break
                
                # Turn Angle Check
                if best_angle > angle_turn_threshold:
                    break 
                
                if best_next in visited: break
                visited.add(best_next)
                curr_edge_idx = best_next
                e_obj = edges[curr_edge_idx]
                u, v = int(e_obj['point1_idx']), int(e_obj['point2_idx'])
                prev = curr
                curr = v if u == curr else u
                path.append(curr)

            is_closed = (len(path) > 2 and path[0] == path[-1])
            segments.append({'vertices': path, 'closed': is_closed})

    # Stage 2: Loops
    all_indices = set(range(len(edges)))
    remaining = all_indices - visited
    while remaining:
        start_idx = remaining.pop()
        visited.add(start_idx)
        e_obj = edges[start_idx]
        u, v = int(e_obj['point1_idx']), int(e_obj['point2_idx'])
        path = [u, v]
        prev = u
        curr = v
        curr_idx = start_idx
        
        while True:
            candidates = [ne for ne in adj[curr] if ne != curr_idx]
            if not candidates: break

            best_next = None
            best_angle = 1e9
            for ne_idx in candidates:
                e_next = edges[ne_idx]
                nu, nv = int(e_next['point1_idx']), int(e_next['point2_idx'])
                nxt_node = nv if nu == curr else nu
                ang = calculate_turn_angle(prev, curr, nxt_node)
                if ang < best_angle:
                    best_angle = ang
                    best_next = ne_idx
            
            if best_next is None: break

            # Turn Angle Check
            if best_angle > angle_turn_threshold:
                 break 

            if best_next in visited: break
            visited.add(best_next)
            remaining.discard(best_next)
            curr_idx = best_next
            e_obj = edges[curr_idx]
            u, v = int(e_obj['point1_idx']), int(e_obj['point2_idx'])
            prev = curr
            curr = v if u == curr else u
            path.append(curr)
            
        is_closed = (len(path) > 2 and path[0] == path[-1])
        segments.append({'vertices': path, 'closed': is_closed})

    return segments

def main():
    parser = argparse.ArgumentParser(description="Visualize raw segment topology: Green=Closed, Red=Open")
    parser.add_argument('--input', type=str, required=True, help="Input mesh path (vtp/stl/ply)")
    parser.add_argument('--angle', type=float, default=40.0, help="Sharp edge angle threshold")
    parser.add_argument('--turn_angle', type=float, default=90.0, help="Turn angle threshold")
    parser.add_argument('--line_width', type=int, default=6, help="Line width for visualization")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: File not found {args.input}")
        return

    print(f"Reading mesh: {args.input}")
    mesh = read_mesh(args.input)

    print(f"Detecting sharp edges (Angle={args.angle})...")
    sharp_edges, _ = detect_sharp_edges(mesh, angle_threshold=args.angle)
    
    if not sharp_edges:
        print("No sharp edges found.")
        return

    junctions = detect_sharp_junctions_degree(mesh, sharp_edges)
    
    if 'cell_normals' not in mesh.cell_data:
        mesh.compute_normals(inplace=True, cell_normals=True, point_normals=False)
    cell_normals = mesh.cell_data.get('Normals', mesh.cell_data.get('cell_normals'))

    print(f"Building segments (Turn Angle={args.turn_angle})...")
    # 使用本地修复函数
    segments = build_sharp_segments_fixed(sharp_edges, junctions, mesh.points, cell_normals, angle_turn_threshold=args.turn_angle)
    print(f"  -> Built {len(segments)} segments")

    p = pv.Plotter()
    p.add_mesh(mesh, color='white', opacity=0.1, style='surface', show_edges=False)

    closed_count = 0
    open_count = 0

    for i, seg in enumerate(segments):
        pts = mesh.points[seg['vertices']]
        if len(pts) < 2: continue

        is_closed = seg['closed']
        color = 'green' if is_closed else 'red'
        label = "Closed Segments" if is_closed and closed_count == 0 else \
                ("Open Segments" if not is_closed and open_count == 0 else None)
        
        if is_closed: closed_count += 1
        else: open_count += 1

        line = pv.lines_from_points(pts)
        p.add_mesh(line, color=color, line_width=args.line_width, render_lines_as_tubes=True, label=label)

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