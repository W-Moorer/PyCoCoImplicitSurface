import argparse
import os
import sys
import numpy as np
import pyvista as pv
from scipy.interpolate import splprep, splev

# 添加路径以导入 precompute
sys.path.append(os.path.abspath("."))
from src.precompute import (
    read_mesh, 
    detect_sharp_edges, 
    detect_sharp_junctions_degree, 
    build_sharp_segments
)

def _safe_normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n

def fit_and_resample_segment(segment_points_ids, mesh_points, step_size, is_closed=False, edge_data_map=None, cell_normals=None):
    """
    对单个 Segment 进行 B 样条拟合与重采样。
    包含：自动闭合检测、断点桥接、法向插值。
    """
    # 获取原始点坐标
    raw_points = mesh_points[segment_points_ids]
    n_raw = raw_points.shape[0]

    if n_raw < 2:
        return None, None, None, None

    # --- 1. 自动闭合检测 (Auto-Bridge Gap) ---
    # 如果标记为未闭合，但首尾距离非常近，说明是算法漏检测了中间的边。
    # 我们需要手动“桥接”这个缺口。
    was_forced_closed = False
    if not is_closed:
        start_pt = raw_points[0]
        end_pt = raw_points[-1]
        gap_dist = np.linalg.norm(start_pt - end_pt)
        
        # 阈值：步长的 3 倍，或者一个固定的小距离（如模型包围盒的 1%）
        # 这里用 step_size * 3.0 比较鲁棒
        gap_threshold = max(step_size * 3.0, 1e-3)
        
        if gap_dist < gap_threshold:
            # print(f"  [Info] 发现微小缺口 ({gap_dist:.4f})，强制闭合曲线。")
            # 物理上追加起点到末尾，形成闭环
            raw_points = np.vstack([raw_points, raw_points[0:1]])
            # 更新 IDs 列表，方便后续查法向（把起点 ID 加到末尾）
            segment_points_ids = segment_points_ids + [segment_points_ids[0]]
            
            is_closed = True
            was_forced_closed = True
            n_raw = raw_points.shape[0]

    # --- 2. 弦长参数化 ---
    dists = np.linalg.norm(raw_points[1:] - raw_points[:-1], axis=1)
    u_cum = np.concatenate(([0], np.cumsum(dists)))
    total_length = u_cum[-1]
    
    if total_length < 1e-12:
        return raw_points[:1], np.array([[0,0,1]]), np.array([[0,0,1]]), np.array([[0,0,1]])
    
    t_params = u_cum / total_length

    # --- 3. B-Spline 拟合 ---
    k = min(3, n_raw - 1)
    
    try:
        # per=1 要求首尾点一致，我们已经通过 raw_points 保证了这一点
        tck, _ = splprep(raw_points.T, u=t_params, s=0.0, k=k, per=1 if is_closed else 0)
    except Exception as e:
        # 降级处理
        tck, _ = splprep(raw_points.T, u=t_params, s=0.0, k=1, per=0)

    # --- 4. 重采样 ---
    num_samples = max(2, int(np.ceil(total_length / step_size)))
    
    # 闭合曲线：endpoint=True 确保首尾相接
    u_new = np.linspace(0, 1, num_samples, endpoint=True)

    new_points = np.array(splev(u_new, tck)).T
    derivatives = np.array(splev(u_new, tck, der=1)).T
    tangents = np.array([_safe_normalize(d) for d in derivatives])

    # --- 5. 属性回溯 (两侧法向) ---
    idx_intervals = np.searchsorted(t_params, u_new, side='right') - 1
    idx_intervals = np.clip(idx_intervals, 0, n_raw - 2)

    res_n1 = []
    res_n2 = []

    for i, idx in enumerate(idx_intervals):
        p_a = segment_points_ids[idx]
        p_b = segment_points_ids[idx + 1]
        
        # 查询 Edge Map
        key = tuple(sorted((p_a, p_b)))
        edge_info = edge_data_map.get(key)
        
        if edge_info:
            f1 = int(edge_info['face1'])
            f2 = int(edge_info['face2'])
            n1 = cell_normals[f1]
            n2 = cell_normals[f2]
        else:
            # --- 关键修复：处理桥接段的法向 ---
            # 如果这是我们强制连上的段（edge_map 里没有），
            # 我们就沿用上一个点的法向，或者进行插值。
            if was_forced_closed and idx == n_raw - 2:
                # 这是最后一段桥接段，取列表里的第一个有效法向（闭环）
                # 为了简单，直接取上一段的法向，或者取整个 list 的第一个（因为是闭环）
                if len(res_n1) > 0:
                    n1 = res_n1[0] # 让它和起点平滑接上
                    n2 = res_n2[0]
                else:
                    n1 = np.array([0, 0, 1.0])
                    n2 = np.array([0, 0, 1.0])
            else:
                # 其他情况的缺数据
                n1 = np.array([0, 0, 1.0])
                n2 = np.array([0, 0, 1.0])
        
        res_n1.append(n1)
        res_n2.append(n2)

    return new_points, tangents, np.array(res_n1), np.array(res_n2)


def process_mesh_curves(mesh, angle_threshold=40.0, step_size=0.1):
    print(f"正在提取尖锐边 (Angle={angle_threshold})...")
    sharp_edges, _ = detect_sharp_edges(mesh, angle_threshold=angle_threshold)
    print(f"检测到 {len(sharp_edges)} 条尖锐边")

    if not sharp_edges:
        return [], [], [], []

    junctions = detect_sharp_junctions_degree(mesh, sharp_edges)
    
    if 'cell_normals' not in mesh.cell_data:
        mesh.compute_normals(inplace=True, cell_normals=True, point_normals=False)
    
    cell_normals = mesh.cell_data['Normals'] if 'Normals' in mesh.cell_data else mesh.cell_data['cell_normals']

    segments = build_sharp_segments(sharp_edges, junctions, mesh.points, cell_normals)
    print(f"构建了 {len(segments)} 条有序 Segment")

    edge_map = {}
    for e in sharp_edges:
        u, v = int(e['point1_idx']), int(e['point2_idx'])
        edge_map[tuple(sorted((u, v)))] = e

    all_pts, all_tan, all_n1, all_n2 = [], [], [], []

    closed_count = 0
    forced_closed_count = 0

    for seg in segments:
        pts, tan, n1, n2 = fit_and_resample_segment(
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

    for pts in pts_list:
        if len(pts) > 1:
            line = pv.lines_from_points(pts)
            plotter.add_mesh(line, color='blue', line_width=4, render_lines_as_tubes=True)

    all_pts = np.vstack(pts_list)
    all_tan = np.vstack(tan_list)
    all_n1 = np.vstack(n1_list)
    all_n2 = np.vstack(n2_list)
    
    # 箭头大小调整
    diag_len = mesh.length
    arrow_scale = diag_len / 200.0 

    # 1. 切向 (绿色)
    poly_tan = pv.PolyData(all_pts)
    poly_tan['vectors'] = all_tan
    arrows_tan = poly_tan.glyph(orient='vectors', scale=False, factor=arrow_scale * 1.2)
    plotter.add_mesh(arrows_tan, color='green', label="Tangent")

    # 2. 侧向法线1 (黄色)
    poly_n1 = pv.PolyData(all_pts)
    poly_n1['vectors'] = all_n1
    arrows_n1 = poly_n1.glyph(orient='vectors', scale=False, factor=arrow_scale)
    plotter.add_mesh(arrows_n1, color='yellow', label="Normal Side 1")

    # 3. 侧向法线2 (青色)
    poly_n2 = pv.PolyData(all_pts)
    poly_n2['vectors'] = all_n2
    arrows_n2 = poly_n2.glyph(orient='vectors', scale=False, factor=arrow_scale)
    plotter.add_mesh(arrows_n2, color='cyan', label="Normal Side 2")

    plotter.add_legend()
    plotter.add_axes()
    plotter.show()

def main():
    parser = argparse.ArgumentParser(description="Demo: Sharp Edge B-Spline Resampling")
    parser.add_argument('--input', type=str, required=True, help="Input mesh path (vtp/stl/ply)")
    parser.add_argument('--angle', type=float, default=40.0, help="Sharp edge angle threshold")
    parser.add_argument('--step', type=float, default=0.05, help="Resampling step size (absolute units)")
    
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: File not found {args.input}")
        return

    print(f"Reading mesh: {args.input}")
    mesh = read_mesh(args.input)

    actual_step = args.step
    if actual_step <= 0:
        actual_step = mesh.length * 0.01
        
    print(f"Using step size: {actual_step:.4f}")

    results = process_mesh_curves(mesh, angle_threshold=args.angle, step_size=actual_step)
    
    print("Visualizing...")
    visualize_results(mesh, results)

if __name__ == "__main__":
    main()