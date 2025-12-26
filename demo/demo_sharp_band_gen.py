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

# ==============================================================================
# 1. 辅助工具
# ==============================================================================
def extract_faces_and_centroids(mesh):
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    points = mesh.points
    face_pts = points[faces]
    centroids = face_pts.mean(axis=1)
    return faces, centroids

def check_convexity_signed(p1, p2, n1, n2):
    edge = p2 - p1
    edge /= (np.linalg.norm(edge) + 1e-12)
    s = np.dot(edge, np.cross(n1, n2))
    return "convex" if s > 0 else "concave"

# ==============================================================================
# 2. 【核心修复】Monkey Patch: 恢复 Turn Angle 切断逻辑
# ==============================================================================
def build_sharp_segments_fixed_turn(edges, junctions, points, cell_normals=None, angle_turn_threshold=90.0):
    """
    修复版组装逻辑：
    1. 包含 Turn Angle 检测：遇到尖角(>threshold)必须断开。
    2. 包含闭环检测：如果首尾重合，标记为 Closed。
    """
    # 辅助：计算转折角 (0~180度)
    def calculate_turn_angle(prev_idx, cur_idx, next_idx):
        v1 = points[cur_idx] - points[prev_idx]
        v2 = points[next_idx] - points[cur_idx]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-12 or n2 < 1e-12:
            return 0.0
        v1 /= n1
        v2 /= n2
        dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
        # dot=1 -> 0度(直线), dot=0 -> 90度, dot=-1 -> 180度(回头)
        # 我们需要的是偏转角。acos(dot) 是两向量夹角。
        # 如果是直线，v1和v2同向，dot=1，acos=0。
        return np.degrees(np.arccos(dot))

    adj = {}
    for i, e in enumerate(edges):
        u, v = int(e['point1_idx']), int(e['point2_idx'])
        adj.setdefault(u, []).append(i)
        adj.setdefault(v, []).append(i)

    visited = set()
    segments = []

    # --- Stage 1: 从端点/Junction出发 ---
    starts = [p for p in adj.keys() if (p in junctions) or (len(adj[p]) != 2)]
    
    for s in starts:
        # 可能有多个方向从 s 出发
        for start_edge_idx in list(adj[s]):
            if start_edge_idx in visited:
                continue
            
            # 初始化路径
            path = [s]
            curr_edge_idx = start_edge_idx
            visited.add(curr_edge_idx)
            
            e_obj = edges[curr_edge_idx]
            u, v = int(e_obj['point1_idx']), int(e_obj['point2_idx'])
            curr = v if u == s else u
            prev = s
            path.append(curr)
            seg_edges = [e_obj]
            turn_splits = []

            # 路径追踪
            while True:
                # 停止条件1：遇到 Junction 或 端点
                if (curr in junctions) or (len(adj[curr]) != 2):
                    break
                # 停止条件2：回到起点
                if curr == s:
                    break
                
                # 找下一条边 (度数为2，必有一条未访问或是回头路)
                # 注意：这里需要预判一下转折角，如果有多个选择（极少见），选最直的
                candidates = []
                for ne_idx in adj[curr]:
                    if ne_idx != curr_edge_idx: # 不走回头路
                        candidates.append(ne_idx)
                
                if not candidates:
                    break

                # 选择最佳下一条边（通常只有1条，但在拓扑异常处可能有重叠）
                best_next = None
                best_angle = 1e9
                
                # 确定 next node 坐标来算角度
                for ne_idx in candidates:
                    e_next = edges[ne_idx]
                    nu, nv = int(e_next['point1_idx']), int(e_next['point2_idx'])
                    nxt_node = nv if nu == curr else nu
                    
                    ang = calculate_turn_angle(prev, curr, nxt_node)
                    if ang < best_angle:
                        best_angle = ang
                        best_next = ne_idx

                # 【修复逻辑】：检查转折角
                if best_next is not None:
                    # 如果转折角太大，强制断开！
                    if best_angle > angle_turn_threshold:
                        # print(f"  [Break] Sharp Turn detected: {best_angle:.1f}° at node {curr}")
                        turn_splits.append(curr)
                        break # <--- 必须 break，形成开环线段
                    
                    # 否则继续走
                    next_edge_idx = best_next
                    if next_edge_idx in visited:
                        break # 闭环接上了已访问的部分

                    visited.add(next_edge_idx)
                    curr_edge_idx = next_edge_idx
                    e_obj = edges[curr_edge_idx]
                    u, v = int(e_obj['point1_idx']), int(e_obj['point2_idx'])
                    
                    prev = curr # 更新 prev
                    curr = v if u == curr else u
                    path.append(curr)
                    seg_edges.append(e_obj)
                else:
                    break

            is_closed = (len(path) > 2 and path[0] == path[-1])
            segments.append({'vertices': path, 'edges': seg_edges, 'closed': is_closed, 'turn_splits': turn_splits})

    # --- Stage 2: 剩余的完美闭环 ---
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
        seg_edges = [e_obj]
        turn_splits = []
        
        while True:
            # 找下一条
            candidates = [ne for ne in adj[curr] if ne != curr_idx]
            if not candidates: break

            # 计算角度选路
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

            # 【修复逻辑】：Stage 2 也要检查转折
            if best_angle > angle_turn_threshold:
                 turn_splits.append(curr)
                 break # 闭环因为尖角被切断成开环

            if best_next in visited:
                break
            
            visited.add(best_next)
            remaining.discard(best_next)
            curr_idx = best_next
            e_obj = edges[curr_idx]
            u, v = int(e_obj['point1_idx']), int(e_obj['point2_idx'])
            prev = curr
            curr = v if u == curr else u
            path.append(curr)
            seg_edges.append(e_obj)
            
        is_closed = (len(path) > 2 and path[0] == path[-1])
        segments.append({'vertices': path, 'edges': seg_edges, 'closed': is_closed, 'turn_splits': turn_splits})

    return segments

# 覆盖原模块函数
precompute_module.build_sharp_segments = build_sharp_segments_fixed_turn

# ==============================================================================
# 3. 拟合与重采样 (保持不变)
# ==============================================================================
def fit_and_resample(seg, points, step):
    verts = seg['vertices']
    raw = points[verts]
    if len(raw) < 2: return None, None, None, None, None
    
    dists = np.linalg.norm(raw[1:] - raw[:-1], axis=1)
    u_cum = np.concatenate(([0], np.cumsum(dists)))
    total = u_cum[-1]
    if total < 1e-9: return None, None, None, None, None
    t = u_cum / total
    
    k = min(3, len(raw)-1)
    try:
        tck, _ = splprep(raw.T, u=t, s=0.0, k=k, per=1 if seg['closed'] else 0)
    except:
        tck, _ = splprep(raw.T, u=t, s=0.0, k=1, per=0)
        
    n_samp = max(2, int(np.ceil(total/step)))
    u_new = np.linspace(0, 1, n_samp, endpoint=True)
    pts = np.array(splev(u_new, tck)).T
    ders = np.array(splev(u_new, tck, der=1)).T
    norms = np.linalg.norm(ders, axis=1)[:,None]
    norms[norms<1e-12] = 1.0
    tans = ders / norms
    
    return pts, tans, t, verts, u_new

# ==============================================================================
# 4. 计算 Band Points (包含法向一致性修复)
# ==============================================================================
def compute_band_points_consistent(new_pts, new_tans, t_raw, v_raw, u_new, 
                                   edge_map, cell_normals, face_centroids, 
                                   mesh_points, offset):
    idx_intervals = np.searchsorted(t_raw, u_new, side='right') - 1
    idx_intervals = np.clip(idx_intervals, 0, len(v_raw) - 2)

    p1_list, n1_list = [], []
    p2_list, n2_list = [], []
    convexity_flags = []

    prev_n1 = None
    prev_n2 = None

    for i, idx in enumerate(idx_intervals):
        curr_p = new_pts[i]
        curr_t = new_tans[i]
        
        v_a = v_raw[idx]
        v_b = v_raw[idx+1]
        
        key = tuple(sorted((v_a, v_b)))
        edge = edge_map.get(key)
        
        if not edge:
            n1 = prev_n1 if prev_n1 is not None else np.array([0,0,1.0])
            n2 = prev_n2 if prev_n2 is not None else np.array([0,0,1.0])
            c1, c2 = curr_p, curr_p
            is_convex = "unknown"
        else:
            f1, f2 = int(edge['face1']), int(edge['face2'])
            n_candidate_A = cell_normals[f1]
            n_candidate_B = cell_normals[f2]
            c_candidate_A = face_centroids[f1]
            c_candidate_B = face_centroids[f2]
            
            # --- 法向一致性对齐 ---
            if prev_n1 is None:
                n1, n2 = n_candidate_A, n_candidate_B
                c1, c2 = c_candidate_A, c_candidate_B
            else:
                dist_direct = np.linalg.norm(n_candidate_A - prev_n1) + np.linalg.norm(n_candidate_B - prev_n2)
                dist_swap   = np.linalg.norm(n_candidate_A - prev_n2) + np.linalg.norm(n_candidate_B - prev_n1)
                
                if dist_swap < dist_direct:
                    n1, n2 = n_candidate_B, n_candidate_A
                    c1, c2 = c_candidate_B, c_candidate_A
                else:
                    n1, n2 = n_candidate_A, n_candidate_B
                    c1, c2 = c_candidate_A, c_candidate_B

            is_convex = check_convexity_signed(mesh_points[v_a], mesh_points[v_b], n1, n2)

        prev_n1 = n1
        prev_n2 = n2
        convexity_flags.append(is_convex)

        # 偏移计算
        e1 = np.cross(curr_t, n1) 
        e2 = np.cross(n2, curr_t)
        
        vn1 = np.linalg.norm(e1)
        if vn1 < 1e-6: e1 = np.cross(curr_t, n2)
        else: e1 /= vn1
        
        vn2 = np.linalg.norm(e2)
        if vn2 < 1e-6: e2 = np.cross(n1, curr_t)
        else: e2 /= vn2

        if np.dot(c1 - curr_p, e1) < 0: e1 = -e1
        if np.dot(c2 - curr_p, e2) < 0: e2 = -e2

        p1_list.append(curr_p + offset * e1)
        n1_list.append(n1)
        p2_list.append(curr_p + offset * e2)
        n2_list.append(n2)

    return (np.array(p1_list), np.array(n1_list), 
            np.array(p2_list), np.array(n2_list), 
            convexity_flags)

# ==============================================================================
# 5. 主程序
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--step', type=float, default=0.05)
    parser.add_argument('--offset', type=float, default=0.02)
    parser.add_argument('--angle', type=float, default=40.0)
    # 默认阈值设为 60 或 90 度，确保直角处断开
    parser.add_argument('--turn_angle', type=float, default=60.0, help="Turn split threshold")
    args = parser.parse_args()

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

    # 构建 Segments (包含修复后的 Turn Check)
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

    print("Generating Bands...")
    
    has_convex = False
    has_concave = False

    for seg in segments:
        pts, tans, t, v, u_new = fit_and_resample(seg, mesh.points, args.step)
        if pts is None: continue

        p1, n1, p2, n2, flags = compute_band_points_consistent(
            pts, tans, t, v, u_new,
            edge_map, cell_normals, face_centroids, 
            mesh.points, args.offset
        )

        n_cvx = flags.count("convex")
        n_ccv = flags.count("concave")
        color = 'orange' if n_cvx > n_ccv else 'purple'
        if color == 'orange': has_convex = True
        else: has_concave = True
        
        # 绘制
        line = pv.lines_from_points(pts)
        p.add_mesh(line, color=color, line_width=4, render_lines_as_tubes=True)
        
        scale = mesh.length / 150.0
        
        p.add_mesh(pv.PolyData(p1), color='gold', point_size=5, render_points_as_spheres=True)
        tmp1 = pv.PolyData(p1); tmp1['n'] = n1
        p.add_mesh(tmp1.glyph(orient='n', scale=False, factor=scale), color='gold')

        p.add_mesh(pv.PolyData(p2), color='cyan', point_size=5, render_points_as_spheres=True)
        tmp2 = pv.PolyData(p2); tmp2['n'] = n2
        p.add_mesh(tmp2.glyph(orient='n', scale=False, factor=scale), color='cyan')

    p.add_legend([['Convex', 'orange'], ['Concave', 'purple'], ['Side 1', 'gold'], ['Side 2', 'cyan']])
    print(f"Total Segments: {len(segments)}")
    p.show()

if __name__ == "__main__":
    main()