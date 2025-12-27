import argparse
import os
import sys
import numpy as np
import pyvista as pv
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree

# 导入 src
sys.path.append(os.path.abspath("."))
try:
    from src.precompute import (
        read_mesh, 
        detect_sharp_edges, 
        detect_sharp_junctions_degree, 
        build_sharp_segments
    )
    import src.precompute as precompute_module
except ImportError:
    print("错误: 无法导入 src.precompute。")
    sys.exit(1)

def _safe_normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n

# ==============================================================================
# 1. 辅助工具
# ==============================================================================
def extract_faces_and_centroids(mesh):
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    points = mesh.points
    face_pts = points[faces]
    centroids = face_pts.mean(axis=1)
    return faces, centroids

def check_convexity_robust(p1, p2, n1, n2, c1, c2):
    """
    稳健的凹凸性判定:
    - 凸边(Convex): Face 2 向下折弯，C2 在 Face 1 平面下方 (dot < 0)
    - 凹边(Concave): Face 2 向上折弯，C2 在 Face 1 平面上方 (dot > 0)
    """
    mid = (p1 + p2) * 0.5
    dist = np.dot(c2 - mid, n1)
    return "convex" if dist < 0 else "concave"

# ==============================================================================
# 2. Monkey Patch: Segments 构建
# ==============================================================================
def build_sharp_segments_fixed_turn(edges, junctions, points, cell_normals=None, angle_turn_threshold=90.0):
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
                best_next = None; best_angle = 1e9
                for ne_idx in candidates:
                    e_next = edges[ne_idx]
                    nu, nv = int(e_next['point1_idx']), int(e_next['point2_idx'])
                    nxt_node = nv if nu == curr else nu
                    ang = calculate_turn_angle(prev, curr, nxt_node)
                    if ang < best_angle: best_angle = ang; best_next = ne_idx
                
                if best_next is None: break
                if best_angle > angle_turn_threshold: break 
                
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
            best_next = None; best_angle = 1e9
            for ne_idx in candidates:
                e_next = edges[ne_idx]
                nu, nv = int(e_next['point1_idx']), int(e_next['point2_idx'])
                nxt_node = nv if nu == curr else nu
                ang = calculate_turn_angle(prev, curr, nxt_node)
                if ang < best_angle: best_angle = ang; best_next = ne_idx
            
            if best_next is None: break
            if best_angle > angle_turn_threshold: break 

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

precompute_module.build_sharp_segments = build_sharp_segments_fixed_turn

# ==============================================================================
# 3. 法向对齐与顶点平均
# ==============================================================================
def compute_aligned_vertex_normals(vert_ids, edge_map, cell_normals, is_closed):
    raw_edge_normals = [] 
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
            raw_edge_normals.append((np.array([0,0,1.0]), np.array([0,0,1.0])))

    if not raw_edge_normals: return None, None

    aligned_edge_normals = []
    prev_n1, prev_n2 = raw_edge_normals[0]
    aligned_edge_normals.append((prev_n1, prev_n2))
    
    for i in range(1, len(raw_edge_normals)):
        curr_n1, curr_n2 = raw_edge_normals[i]
        dist_direct = np.linalg.norm(curr_n1 - prev_n1) + np.linalg.norm(curr_n2 - prev_n2)
        dist_swap   = np.linalg.norm(curr_n1 - prev_n2) + np.linalg.norm(curr_n2 - prev_n1)
        if dist_swap < dist_direct:
            curr_n1, curr_n2 = curr_n2, curr_n1
        aligned_edge_normals.append((curr_n1, curr_n2))
        prev_n1, prev_n2 = curr_n1, curr_n2

    vn1_list = []
    vn2_list = []
    num_verts = len(vert_ids)
    num_edges = len(aligned_edge_normals)
    
    for i in range(num_verts):
        inc_n1 = []
        inc_n2 = []
        
        # 入边
        if i > 0:
            en1, en2 = aligned_edge_normals[i-1]
            inc_n1.append(en1); inc_n2.append(en2)
        elif is_closed: 
            last_n1, last_n2 = aligned_edge_normals[-1]
            first_n1, first_n2 = aligned_edge_normals[0]
            dist_direct = np.linalg.norm(last_n1 - first_n1) + np.linalg.norm(last_n2 - first_n2)
            dist_swap   = np.linalg.norm(last_n1 - first_n2) + np.linalg.norm(last_n2 - first_n1)
            if dist_swap < dist_direct:
                inc_n1.append(last_n2); inc_n2.append(last_n1)
            else:
                inc_n1.append(last_n1); inc_n2.append(last_n2)
        
        # 出边
        if i < num_edges:
            en1, en2 = aligned_edge_normals[i]
            inc_n1.append(en1); inc_n2.append(en2)
        elif is_closed:
            en1, en2 = aligned_edge_normals[0]
            inc_n1.append(en1); inc_n2.append(en2)
            
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
# 4. 拟合与重采样
# ==============================================================================
def fit_and_resample_segment_advanced(segment_points_ids, mesh_points, step_size, is_closed=False, edge_data_map=None, cell_normals=None):
    raw_points = mesh_points[segment_points_ids]
    n_raw = raw_points.shape[0]
    if n_raw < 2: return None, None, None, None, None, None

    dists = np.linalg.norm(raw_points[1:] - raw_points[:-1], axis=1)
    u_cum = np.concatenate(([0], np.cumsum(dists)))
    total_length = u_cum[-1]
    if total_length < 1e-12: return None, None, None, None, None, None
    t_params = u_cum / total_length

    vn1_arr, vn2_arr = compute_aligned_vertex_normals(segment_points_ids, edge_data_map, cell_normals, is_closed)
    if vn1_arr is None: return None, None, None, None, None, None

    k = min(3, n_raw - 1)
    try:
        tck_pos, _ = splprep(raw_points.T, u=t_params, s=0.0, k=k, per=1 if is_closed else 0)
        tck_n1, _  = splprep(vn1_arr.T, u=t_params, s=0.0, k=k, per=1 if is_closed else 0)
        tck_n2, _  = splprep(vn2_arr.T, u=t_params, s=0.0, k=k, per=1 if is_closed else 0)
    except Exception:
        tck_pos, _ = splprep(raw_points.T, u=t_params, s=0.0, k=1, per=0)
        tck_n1, _  = splprep(vn1_arr.T, u=t_params, s=0.0, k=1, per=0)
        tck_n2, _  = splprep(vn2_arr.T, u=t_params, s=0.0, k=1, per=0)

    num_samples = max(2, int(np.ceil(total_length / step_size)))
    u_new = np.linspace(0, 1, num_samples, endpoint=True)

    new_points = np.array(splev(u_new, tck_pos)).T
    derivatives = np.array(splev(u_new, tck_pos, der=1)).T
    tangents = np.array([_safe_normalize(d) for d in derivatives])

    raw_res_n1 = np.array(splev(u_new, tck_n1)).T
    raw_res_n2 = np.array(splev(u_new, tck_n2)).T
    
    res_n1 = np.array([_safe_normalize(n) for n in raw_res_n1])
    res_n2 = np.array([_safe_normalize(n) for n in raw_res_n2])

    return new_points, tangents, res_n1, res_n2, t_params, u_new

# ==============================================================================
# 5. 生成 Band Points (基于凹凸性的几何定向)
# ==============================================================================
def compute_band_points_geometric(new_pts, new_tans, n1s, n2s, 
                                  t_raw, vert_ids_raw, u_new, 
                                  edge_map, cell_normals, face_centroids, 
                                  mesh_points, offset, mesh_obj=None):
    """
    【核心改进】
    废弃全局投票，改用“凹凸性+另一侧法向”的局部几何约束。
    - 对于凸边(Convex): Side 1 偏移应大致反向于 Side 2 法向量的投影。
    - 对于凹边(Concave): Side 1 偏移应大致同向于 Side 2 法向量的投影。
    """
    idx_intervals = np.searchsorted(t_raw, u_new, side='right') - 1
    idx_intervals = np.clip(idx_intervals, 0, len(vert_ids_raw) - 2)

    final_p1s = []
    final_p2s = []
    convexity_flags = []
    
    # 临时缓存上一帧的凸凹状态，防止在个别点计算失败时抖动
    last_convexity = "convex" 

    for i in range(len(new_pts)):
        curr_p = new_pts[i]
        t = new_tans[i]
        n1 = n1s[i]
        n2 = n2s[i]
        
        # 1. 计算基准偏移向量 (在切平面内，垂直于t)
        e1 = np.cross(t, n1)
        if np.linalg.norm(e1) < 1e-6: e1 = np.cross(t, n2)
        e1 = _safe_normalize(e1)
        
        e2 = np.cross(n2, t)
        if np.linalg.norm(e2) < 1e-6: e2 = np.cross(n1, t)
        e2 = _safe_normalize(e2)

        # 2. 获取原始几何信息以判断凹凸
        idx = idx_intervals[i]
        v_a = vert_ids_raw[idx]
        v_b = vert_ids_raw[idx+1]
        key = tuple(sorted((v_a, v_b)))
        edge = edge_map.get(key)
        
        is_convex = last_convexity
        if edge:
            f1, f2 = int(edge['face1']), int(edge['face2'])
            raw_n1 = cell_normals[f1]
            raw_n2 = cell_normals[f2]
            c1, c2 = face_centroids[f1], face_centroids[f2]
            # 判定凹凸
            is_convex = check_convexity_robust(mesh_points[v_a], mesh_points[v_b], raw_n1, raw_n2, c1, c2)
            last_convexity = is_convex
        
        convexity_flags.append(is_convex)

        # 3. 几何定向修正
        # Side 1 修正:
        dot_1 = np.dot(e1, n2)
        
        flip_1 = 1.0
        if is_convex == "convex":
            if dot_1 > 0: flip_1 = -1.0
        else:
            if dot_1 < 0: flip_1 = -1.0
            
        # Side 2 修正:
        dot_2 = np.dot(e2, n1)
        
        flip_2 = 1.0
        if is_convex == "convex":
            if dot_2 > 0: flip_2 = -1.0
        else:
            if dot_2 < 0: flip_2 = -1.0

        final_p1s.append(curr_p + offset * e1 * flip_1)
        final_p2s.append(curr_p + offset * e2 * flip_2)

    final_p1s = np.array(final_p1s)
    final_p2s = np.array(final_p2s)
    
    # 【新增】: 执行越界检测与修复
    if mesh_obj is not None:
        final_p1s, bad1 = validate_and_fix_points_batch(mesh_obj, cell_normals, final_p1s, n1s, new_pts)
        final_p2s, bad2 = validate_and_fix_points_batch(mesh_obj, cell_normals, final_p2s, n2s, new_pts)
        
        n_bad = np.sum(bad1) + np.sum(bad2)
        if n_bad > 0:
            pass # 可以打印日志: print(f"Fixed {n_bad} out-of-bounds points.")

    return final_p1s, n1s, final_p2s, n2s, convexity_flags

# ==============================================================================
# 5.5. 越界检测与修复 (Validate and Fix)
# ==============================================================================
def validate_and_fix_points_batch(mesh, cell_normals, candidates, expected_normals, fallback_points, threshold=0.5):
    '''
    批量验证点是否越界 (Out of Bounds Detection)
    基于法向一致性校验: dot(expected, found) < threshold => Out of Bounds
    
    原理:
    如果偏移点跑到了完全不同的面上(例如从顶面跑到了侧面), 
    其最近邻面的法向量(Found)会与期望法向量(Expected/Start)发生剧烈偏转(接近垂直).
    '''
    if len(candidates) == 0:
        return candidates, np.zeros(0, dtype=bool)

    # 1. 寻找最近面索引
    # find_closest_cell 返回的是 cell index
    try:
        cell_ids = mesh.find_closest_cell(candidates)
    except AttributeError:
        # 兼容旧版 PyVista 或某些特殊情况
        print("Warning: find_closest_cell not available or failed. Skipping validation.")
        return candidates, np.zeros(len(candidates), dtype=bool)
    
    # 2. 获取最近面的法向量
    valid_mask = (cell_ids != -1)
    
    found_normals = np.zeros_like(candidates)
    # 使用传入的 cell_normals 数组 (N_cells, 3)
    if np.any(valid_mask):
        found_normals[valid_mask] = cell_normals[cell_ids[valid_mask]]
    
    # 3. 计算点积 (Cosine similarity)
    # expected_normals 是该点原本所属流形的法向 (Spline N1/N2)
    dots = np.sum(expected_normals * found_normals, axis=1)
    
    # 4. 判定与修复
    # Dot < Threshold (e.g. 0.5 for 60 degrees) => Severe mismatch (Cliff detected)
    bad_mask = (dots < threshold) | (~valid_mask)
    
    fixed_points = candidates.copy()
    # 策略: 回退到原始中心线 (fallback_points)
    # 也可以尝试减小 Offset，但回退是最安全的做法，避免产生悬空几何
    fixed_points[bad_mask] = fallback_points[bad_mask]
    
    return fixed_points, bad_mask

# ==============================================================================
# 6. 主程序
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--step', type=float, default=0.05)
    parser.add_argument('--offset', type=float, default=0.02)
    parser.add_argument('--angle', type=float, default=40.0)
    parser.add_argument('--turn_angle', type=float, default=90.0)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print("File not found")
        return

    print(f"Reading: {args.input}")
    mesh = read_mesh(args.input)
    if args.step <= 0: args.step = mesh.length * 0.01
    if args.offset <= 0: args.offset = args.step * 0.5

    print("Precomputing...")
    edges, _ = detect_sharp_edges(mesh, args.angle)
    junctions = detect_sharp_junctions_degree(mesh, edges)
    
    if 'cell_normals' not in mesh.cell_data:
        mesh.compute_normals(inplace=True, cell_normals=True)
    cell_normals = mesh.cell_data.get('Normals', mesh.cell_data.get('cell_normals'))
    faces_arr, face_centroids = extract_faces_and_centroids(mesh)

    print(f"Building Segments (Turn Threshold = {args.turn_angle} deg)...")
    segments = precompute_module.build_sharp_segments(
        edges, junctions, mesh.points, cell_normals, 
        angle_turn_threshold=args.turn_angle
    )
    
    edge_map = {}
    for e in edges:
        k = tuple(sorted((int(e['point1_idx']), int(e['point2_idx']))))
        edge_map[k] = e

    p = pv.Plotter()
    p.add_mesh(mesh, color='white', opacity=0.2, style='surface', show_edges=False)

    print("Generating Smooth Bands (Geometric Orientation)...")
    
    has_convex = False
    has_concave = False

    for seg in segments:
        res = fit_and_resample_segment_advanced(
            seg['vertices'], mesh.points, args.step, 
            is_closed=seg['closed'], 
            edge_data_map=edge_map, 
            cell_normals=cell_normals
        )
        
        if res is None: continue
        pts, tans, smooth_n1, smooth_n2, t_raw, u_new = res

        # 【调用新的几何定向函数】
        p1, n1, p2, n2, flags = compute_band_points_geometric(
            pts, tans, smooth_n1, smooth_n2,
            t_raw, seg['vertices'], u_new,
            edge_map, cell_normals, face_centroids, 
            mesh.points, args.offset,
            mesh_obj=mesh  # 传入 mesh 对象用于 KDTree 查询
        )

        n_cvx = flags.count("convex")
        n_ccv = flags.count("concave")
        color = 'orange' if n_cvx > n_ccv else 'purple'
        if color == 'orange': has_convex = True
        else: has_concave = True
        
        line = pv.lines_from_points(pts)
        p.add_mesh(line, color=color, line_width=4, render_lines_as_tubes=True)
        
        scale = mesh.length / 150.0
        
        p.add_mesh(pv.PolyData(p1), color='gold', point_size=5, render_points_as_spheres=True)
        tmp1 = pv.PolyData(p1); tmp1['n'] = n1
        p.add_mesh(tmp1.glyph(orient='n', scale=False, factor=scale), color='gold')

        p.add_mesh(pv.PolyData(p2), color='cyan', point_size=5, render_points_as_spheres=True)
        tmp2 = pv.PolyData(p2); tmp2['n'] = n2
        p.add_mesh(tmp2.glyph(orient='n', scale=False, factor=scale), color='cyan')

    p.add_legend([['Convex', 'orange'], ['Concave', 'purple'], ['Side 1 (Smooth)', 'gold'], ['Side 2 (Smooth)', 'cyan']])
    print(f"Total Segments: {len(segments)}")
    p.show()

if __name__ == "__main__":
    main()