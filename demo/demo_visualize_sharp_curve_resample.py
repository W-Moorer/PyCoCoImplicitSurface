import argparse
import os
import sys
import numpy as np
import pyvista as pv
from scipy.interpolate import splprep, splev

# 添加路径以导入 precompute
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

def _safe_normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n

# ==============================================================================
# 1. 修复后的 Segments 构建逻辑 (Turn Angle Check)
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
                if best_angle > angle_turn_threshold: break # Sharp Turn Break
                
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
            if best_angle > angle_turn_threshold: break # Sharp Turn Break

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

# ==============================================================================
# 2. 改进的法向计算逻辑 (Alignment + Vertex Averaging)
# ==============================================================================
def compute_aligned_vertex_normals(vert_ids, edge_map, cell_normals, is_closed):
    """
    计算每个顶点的平滑 Side1/Side2 法向。
    1. 获取边法向
    2. 解决边与边之间的法向翻转 (Consistency Alignment)
    3. 将边法向平均到顶点 (Vertex Averaging)
    """
    # --- A. 获取原始边法向序列 ---
    raw_edge_normals = [] # List of (n1, n2)
    for i in range(len(vert_ids) - 1):
        u, v = vert_ids[i], vert_ids[i+1]
        key = tuple(sorted((u, v)))
        edge = edge_map.get(key)
        if edge:
            f1, f2 = int(edge['face1']), int(edge['face2'])
            n1 = cell_normals[f1]
            n2 = cell_normals[f2]
            raw_edge_normals.append((n1, n2))
        else:
            # 异常兜底
            raw_edge_normals.append((np.array([0,0,1.0]), np.array([0,0,1.0])))

    if not raw_edge_normals:
        return None, None

    # --- B. 法向一致性对齐 (Alignment) ---
    aligned_edge_normals = []
    # 初始化
    prev_n1, prev_n2 = raw_edge_normals[0]
    aligned_edge_normals.append((prev_n1, prev_n2))
    
    for i in range(1, len(raw_edge_normals)):
        curr_n1, curr_n2 = raw_edge_normals[i]
        
        # 计算距离代价
        dist_direct = np.linalg.norm(curr_n1 - prev_n1) + np.linalg.norm(curr_n2 - prev_n2)
        dist_swap   = np.linalg.norm(curr_n1 - prev_n2) + np.linalg.norm(curr_n2 - prev_n1)
        
        # 如果交换后更接近，则交换
        if dist_swap < dist_direct:
            curr_n1, curr_n2 = curr_n2, curr_n1
            
        aligned_edge_normals.append((curr_n1, curr_n2))
        prev_n1, prev_n2 = curr_n1, curr_n2

    # 如果是闭环，还需要检查最后一条边和第一条边是否需要翻转（处理莫比乌斯环情况）
    # 这里简单处理：假设非莫比乌斯环，不做额外全局优化

    # --- C. 计算顶点法向 (Vertex Averaging) ---
    vn1_list = []
    vn2_list = []
    
    num_verts = len(vert_ids)
    num_edges = len(aligned_edge_normals)
    
    for i in range(num_verts):
        inc_n1 = []
        inc_n2 = []
        
        # 入边 (Edge i-1)
        if i > 0:
            en1, en2 = aligned_edge_normals[i-1]
            inc_n1.append(en1)
            inc_n2.append(en2)
        elif is_closed: 
            # 闭环起点的入边是最后一条边
            en1, en2 = aligned_edge_normals[-1]
            inc_n1.append(en1)
            inc_n2.append(en2)
            
        # 出边 (Edge i)
        if i < num_edges:
            en1, en2 = aligned_edge_normals[i]
            inc_n1.append(en1)
            inc_n2.append(en2)
        elif is_closed:
            # 闭环终点的出边是第一条边
            en1, en2 = aligned_edge_normals[0]
            inc_n1.append(en1)
            inc_n2.append(en2)
            
        # 平均
        if inc_n1:
            avg_n1 = np.mean(inc_n1, axis=0)
            avg_n2 = np.mean(inc_n2, axis=0)
            vn1_list.append(_safe_normalize(avg_n1))
            vn2_list.append(_safe_normalize(avg_n2))
        else:
            vn1_list.append(np.array([0,0,1.0]))
            vn2_list.append(np.array([0,0,1.0]))
            
    return np.array(vn1_list), np.array(vn2_list)

# ==============================================================================
# 3. 拟合与重采样 (包含法向 B 样条拟合)
# ==============================================================================
def fit_and_resample_segment_advanced(segment_points_ids, mesh_points, step_size, is_closed=False, edge_data_map=None, cell_normals=None):
    raw_points = mesh_points[segment_points_ids]
    n_raw = raw_points.shape[0]
    if n_raw < 2: return None, None, None, None

    # 1. 弦长参数化
    dists = np.linalg.norm(raw_points[1:] - raw_points[:-1], axis=1)
    u_cum = np.concatenate(([0], np.cumsum(dists)))
    total_length = u_cum[-1]
    if total_length < 1e-12: return None, None, None, None
    t_params = u_cum / total_length

    # 2. 计算平滑顶点法向
    vn1_arr, vn2_arr = compute_aligned_vertex_normals(segment_points_ids, edge_data_map, cell_normals, is_closed)
    if vn1_arr is None: return None, None, None, None

    # 3. B-Spline 拟合 (位置 + 法向)
    k = min(3, n_raw - 1)
    
    # 准备拟合
    # 如果 closed=True，raw_points首尾重合，splprep 需要 per=1
    # 注意：Normals 也是首尾重合的（因为我们是按顶点算的）
    try:
        # 位置拟合
        tck_pos, _ = splprep(raw_points.T, u=t_params, s=0.0, k=k, per=1 if is_closed else 0)
        # 法向拟合 (分别对 Side1 和 Side2)
        tck_n1, _  = splprep(vn1_arr.T, u=t_params, s=0.0, k=k, per=1 if is_closed else 0)
        tck_n2, _  = splprep(vn2_arr.T, u=t_params, s=0.0, k=k, per=1 if is_closed else 0)
    except Exception as e:
        # 降级线性
        print(f"拟合降级 (k=1): {e}")
        tck_pos, _ = splprep(raw_points.T, u=t_params, s=0.0, k=1, per=0)
        tck_n1, _  = splprep(vn1_arr.T, u=t_params, s=0.0, k=1, per=0)
        tck_n2, _  = splprep(vn2_arr.T, u=t_params, s=0.0, k=1, per=0)

    # 4. 重采样
    num_samples = max(2, int(np.ceil(total_length / step_size)))
    u_new = np.linspace(0, 1, num_samples, endpoint=True)

    # 评估位置
    new_points = np.array(splev(u_new, tck_pos)).T
    derivatives = np.array(splev(u_new, tck_pos, der=1)).T
    tangents = np.array([_safe_normalize(d) for d in derivatives])

    # 评估法向 (插值得到的法向需要归一化)
    raw_res_n1 = np.array(splev(u_new, tck_n1)).T
    raw_res_n2 = np.array(splev(u_new, tck_n2)).T
    
    res_n1 = np.array([_safe_normalize(n) for n in raw_res_n1])
    res_n2 = np.array([_safe_normalize(n) for n in raw_res_n2])

    return new_points, tangents, res_n1, res_n2

def process_mesh_curves(mesh, angle_threshold=40.0, step_size=0.1, turn_angle=90.0):
    print(f"正在提取尖锐边 (Angle={angle_threshold})...")
    sharp_edges, _ = detect_sharp_edges(mesh, angle_threshold=angle_threshold)
    print(f"检测到 {len(sharp_edges)} 条尖锐边")
    if not sharp_edges: return [], [], [], []

    junctions = detect_sharp_junctions_degree(mesh, sharp_edges)
    
    if 'cell_normals' not in mesh.cell_data:
        mesh.compute_normals(inplace=True, cell_normals=True, point_normals=False)
    cell_normals = mesh.cell_data['Normals'] if 'Normals' in mesh.cell_data else mesh.cell_data['cell_normals']

    print(f"构建 Segment (Turn Threshold={turn_angle})...")
    segments = build_sharp_segments_fixed(sharp_edges, junctions, mesh.points, cell_normals, angle_turn_threshold=turn_angle)
    print(f"构建了 {len(segments)} 条有序 Segment")

    edge_map = {}
    for e in sharp_edges:
        u, v = int(e['point1_idx']), int(e['point2_idx'])
        edge_map[tuple(sorted((u, v)))] = e

    all_pts, all_tan, all_n1, all_n2 = [], [], [], []

    for seg in segments:
        pts, tan, n1, n2 = fit_and_resample_segment_advanced(
            segment_points_ids=seg['vertices'],
            mesh_points=mesh.points,
            step_size=step_size,
            is_closed=seg['closed'],
            edge_data_map=edge_map,
            cell_normals=cell_normals
        )
        
        if pts is not None:
            all_pts.append(pts)
            all_tan.append(tan)
            all_n1.append(n1)
            all_n2.append(n2)

    return all_pts, all_tan, all_n1, all_n2

def visualize_results(mesh, resampled_data):
    pts_list, tan_list, n1_list, n2_list = resampled_data
    if not pts_list:
        print("未生成任何曲线数据。")
        return

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='white', opacity=0.3, style='surface', show_edges=False)

    for i, pts in enumerate(pts_list):
        if len(pts) > 1:
            line = pv.lines_from_points(pts)
            plotter.add_mesh(line, color='blue', line_width=4, render_lines_as_tubes=True)

    all_pts = np.vstack(pts_list)
    all_tan = np.vstack(tan_list)
    all_n1 = np.vstack(n1_list)
    all_n2 = np.vstack(n2_list)
    
    diag_len = mesh.length
    arrow_scale = diag_len / 150.0 

    poly_tan = pv.PolyData(all_pts)
    poly_tan['vectors'] = all_tan
    plotter.add_mesh(poly_tan.glyph(orient='vectors', scale=False, factor=arrow_scale * 1.2), color='green', label="Tangent")

    poly_n1 = pv.PolyData(all_pts)
    poly_n1['vectors'] = all_n1
    plotter.add_mesh(poly_n1.glyph(orient='vectors', scale=False, factor=arrow_scale), color='gold', label="Normal Side 1 (Smooth)")

    poly_n2 = pv.PolyData(all_pts)
    poly_n2['vectors'] = all_n2
    plotter.add_mesh(poly_n2.glyph(orient='vectors', scale=False, factor=arrow_scale), color='cyan', label="Normal Side 2 (Smooth)")

    plotter.add_legend()
    plotter.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--angle', type=float, default=40.0)
    parser.add_argument('--step', type=float, default=0.05)
    parser.add_argument('--turn_angle', type=float, default=90.0)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print("File not found")
        return

    mesh = read_mesh(args.input)
    actual_step = args.step if args.step > 0 else mesh.length * 0.01
    
    results = process_mesh_curves(mesh, angle_threshold=args.angle, step_size=actual_step, turn_angle=args.turn_angle)
    visualize_results(mesh, results)

if __name__ == "__main__":
    main()