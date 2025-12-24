import argparse
import os
import numpy as np
import pyvista as pv
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.precompute import read_mesh, detect_sharp_edges, detect_sharp_junctions_degree

def load_sharp_edges(mesh, angle_threshold, edge_split_threshold, require_step_face_id_diff):
    # 直接计算尖锐边缘，不再加载pkl文件
    edges, lines = detect_sharp_edges(mesh, angle_threshold=angle_threshold, edge_split_threshold=edge_split_threshold, require_step_face_id_diff=require_step_face_id_diff)
    return edges, lines

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    # 移除sharp_edges_pkl参数
    ap.add_argument('--angle_threshold', type=float, default=30.0)
    ap.add_argument('--edge_split_threshold', type=float, default=None)
    ap.add_argument('--require_step_face_id_diff', action='store_true')
    ap.add_argument('--point_size', type=int, default=14)
    ap.add_argument('--line_width', type=int, default=4)
    ap.add_argument('--mesh_opacity', type=float, default=0.25)
    ap.add_argument('--segments_debug', default=None)
    ap.add_argument('--turn_point_size', type=int, default=18)
    ap.add_argument('--off_screen', action='store_true')
    ap.add_argument('--screenshot', default=None)
    args = ap.parse_args()
    mesh = read_mesh(args.input, compute_split_normals=False)
    # 调用修改后的load_sharp_edges函数，不再传入pkl路径
    edges, edge_lines = load_sharp_edges(mesh, args.angle_threshold, args.edge_split_threshold, args.require_step_face_id_diff)
    junctions = detect_sharp_junctions_degree(mesh, edges)
    colors = []
    for e in edges:
        colors.append(1 if e.get('is_convex', True) else 0)
    lines_poly = build_lines_poly(edge_lines, colors)
    p = pv.Plotter(off_screen=args.off_screen)
    p.add_mesh(mesh, color='lightgray', opacity=max(min(args.mesh_opacity, 1.0), 0.0))
    if lines_poly.n_points > 0:
        p.add_mesh(lines_poly, scalars='color_idx', cmap=['blue','red'], render_lines_as_tubes=True, line_width=max(1, int(args.line_width)), show_scalar_bar=False)
    if junctions:
        pts = np.array([mesh.points[int(j)] for j in junctions])
        p.add_points(pts, color='green', point_size=max(1, int(args.point_size)), render_points_as_spheres=True)
    if args.segments_debug and os.path.exists(args.segments_debug):
        try:
            with open(args.segments_debug, 'r') as f:
                d = json.load(f)
            segs = d.get('segments', [])
            turn_ids = set()
            for s in segs:
                for v in s.get('turn_splits', []) or []:
                    turn_ids.add(int(v))
            if len(turn_ids) > 0:
                tp = np.array([mesh.points[i] for i in sorted(list(turn_ids))])
                p.add_points(tp, color='purple', point_size=max(1, int(args.turn_point_size)), render_points_as_spheres=True)
        except Exception:
            pass
    p.add_axes()
    p.enable_eye_dome_lighting()
    p.background_color = 'white'
    if args.screenshot:
        p.show(auto_close=False)
        p.screenshot(args.screenshot)
        p.close()
    else:
        p.show()

if __name__ == '__main__':
    main()