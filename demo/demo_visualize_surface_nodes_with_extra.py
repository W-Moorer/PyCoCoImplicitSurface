#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化 CFPU input 中的：
- nodes.txt（蓝）
- patches.txt（红）
- sharp_segments_debug.json（海军蓝线：sharp edges 分段）
- sharp_curve_points_raw.npy + sharp_curve_group_id.npy（曲线采样点/连线：闭环=橙色，开环=蓝色）
- sharp_band_points_raw.npy（绿点）

v5 改动点（修复闭环判定 + 强制首尾连接 + 闭环/开环分色）：
1) 不再只依赖 segments_debug 里的 "closed" 字段。若满足：
   - turn_splits 为空 且 (edges 构成 cycle)
   则认为是闭环（即使 "closed" 字段缺失或为 False）。
2) 曲线连线时：若该 gid 判定为闭环，则强制连 last->first，并用橙色绘制；否则蓝色。
3) 曲线点顺序：默认按该 segment 的 vertex_points 弧长排序（近邻投影到顶点链），避免顺序被打散后出现跨越连线。

用法示例：
python demo_visualize_sharpband_segments_and_bspline_v5.py --cfpu_dir output\\cfpu_input\\XXX_cfpu_input --mesh input\\...\\XXX.vtp --show_segments --show_curve --show_band
"""
import argparse
import json
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pyvista as pv

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


def _load_txt3(path: str) -> np.ndarray:
    a = np.loadtxt(path)
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.shape[1] > 3:
        a = a[:, :3]
    return a


def _add_polyline(plotter: pv.Plotter, pts: np.ndarray, color: str, width: float = 3.0, closed: bool = False):
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return
    if closed:
        pts2 = np.vstack([pts, pts[:1]])
    else:
        pts2 = pts
    n = pts2.shape[0]
    lines = np.empty((n - 1, 3), dtype=np.int64)
    for i in range(n - 1):
        lines[i] = [2, i, i + 1]
    poly = pv.PolyData(pts2)
    poly.lines = lines.ravel()
    plotter.add_mesh(poly, color=color, line_width=max(1.0, float(width)), render_lines_as_tubes=True)


def _group_indices(group_id: np.ndarray) -> Dict[int, np.ndarray]:
    gid = np.asarray(group_id).reshape(-1)
    out: Dict[int, List[int]] = {}
    for i, g in enumerate(gid):
        gg = int(g)
        out.setdefault(gg, []).append(i)
    return {k: np.asarray(v, dtype=int) for k, v in out.items()}


def _cumlen(poly: np.ndarray) -> np.ndarray:
    """累积弧长（按给定顺序）"""
    P = np.asarray(poly, dtype=float)
    if P.ndim != 2 or P.shape[0] == 0:
        return np.zeros((0,), dtype=float)
    d = np.linalg.norm(P[1:] - P[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    return s


def _order_points_by_segment_arclen(curve_pts: np.ndarray, seg_vpts: np.ndarray) -> np.ndarray:
    """
    依据 curve 点在 seg_vpts 顶点链上的“最近顶点弧长”来排序。
    目标：稳定显示连通性/闭合性，避免出现跨越连线。
    """
    pts = np.asarray(curve_pts, dtype=float)
    ref = np.asarray(seg_vpts, dtype=float)
    if pts.shape[0] < 2 or ref.shape[0] < 2 or (cKDTree is None):
        return np.arange(pts.shape[0], dtype=int)

    tree = cKDTree(ref)
    _, vidx = tree.query(pts, k=1)
    vidx = np.asarray(vidx, dtype=int).reshape(-1)

    s = _cumlen(ref)
    key = s[np.clip(vidx, 0, s.shape[0] - 1)]
    order = np.lexsort((np.arange(key.shape[0]), key))
    return order.astype(int)


def _is_cycle_from_edges(edges: List[Tuple[int, int]]) -> bool:
    """判断无向边集合是否构成单一 cycle：所有顶点度=2 且连通 且 |E|=|V|>=3"""
    if not edges:
        return False
    adj: Dict[int, List[int]] = {}
    deg: Dict[int, int] = {}
    for a, b in edges:
        a = int(a); b = int(b)
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    verts = list(adj.keys())
    if len(verts) < 3:
        return False
    # degree==2
    if any(deg[v] != 2 for v in verts):
        return False
    # |E| == |V|
    if len(edges) != len(verts):
        return False
    # connected
    stack = [verts[0]]
    vis = set([verts[0]])
    while stack:
        v = stack.pop()
        for nb in adj.get(v, []):
            if nb not in vis:
                vis.add(nb)
                stack.append(nb)
    return len(vis) == len(verts)

DIR = "output\cfpu_input\LinkedGear_surface_cellnormals_cfpu_input"
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfpu_dir', type=str, default=DIR)
    ap.add_argument('--mesh', default=None)
    ap.add_argument('--show_nodes', action='store_true', default=True)
    ap.add_argument('--show_patches', action='store_true', default=True)
    ap.add_argument('--show_segments', action='store_true', default=False)
    ap.add_argument('--show_curve', action='store_true', default=True)
    ap.add_argument('--show_band', action='store_true', default=False)

    ap.add_argument('--node_size', type=float, default=3.5)
    ap.add_argument('--patch_size', type=float, default=8.0)
    ap.add_argument('--seg_line_width', type=float, default=4.0)
    ap.add_argument('--curve_point_size', type=float, default=5.0)
    ap.add_argument('--curve_line_width', type=float, default=8.0)
    ap.add_argument('--band_size', type=float, default=5.0)

    ap.add_argument('--order_curve_by_segment', action='store_true', default=True)
    ap.add_argument('--verbose', action='store_true', default=False)

    args = ap.parse_args()
    cfpu_dir = args.cfpu_dir

    nodes_path = os.path.join(cfpu_dir, 'nodes.txt')
    patches_path = os.path.join(cfpu_dir, 'patches.txt')
    seg_debug_path = os.path.join(cfpu_dir, 'sharp_segments_debug.json')
    curve_pts_path = os.path.join(cfpu_dir, 'sharp_curve_points_raw.npy')
    curve_gid_path = os.path.join(cfpu_dir, 'sharp_curve_group_id.npy')
    band_path = os.path.join(cfpu_dir, 'sharp_band_points_raw.npy')

    p = pv.Plotter(window_size=(1400, 1000))
    p.background_color = 'white'

    if args.mesh and os.path.exists(args.mesh):
        try:
            m = pv.read(args.mesh)
            p.add_mesh(m, color='lightgray', opacity=0.25)
        except Exception as e:
            print(f'[demo] mesh 读取失败：{e}')

    bbox_pts = []

    # nodes
    if args.show_nodes and os.path.exists(nodes_path):
        nodes = _load_txt3(nodes_path)
        bbox_pts.append(nodes)
        p.add_points(nodes, color='blue', point_size=max(1.0, float(args.node_size)), render_points_as_spheres=True)

    # patches
    if args.show_patches and os.path.exists(patches_path):
        patches = _load_txt3(patches_path)
        bbox_pts.append(patches)
        p.add_points(patches, color='red', point_size=max(1.0, float(args.patch_size)), render_points_as_spheres=True)

    # segments_debug
    seg_closed: Dict[int, bool] = {}
    seg_vpts: Dict[int, np.ndarray] = {}
    seg_edges: Dict[int, List[Tuple[int, int]]] = {}
    seg_turns: Dict[int, List[int]] = {}

    if os.path.exists(seg_debug_path):
        try:
            with open(seg_debug_path, 'r', encoding='utf-8') as f:
                d = json.load(f) or {}
            segs = d.get('segments', []) or []
            segs = sorted(segs, key=lambda x: int(x.get('id', 0)))

            for s in segs:
                gid = int(s.get('id', 0))
                turns = [int(x) for x in (s.get('turn_splits', []) or [])]
                edges = [(int(a), int(b)) for (a, b) in (s.get('edges', []) or [])]
                vpts = np.asarray(s.get('vertex_points', []) or [], dtype=float)

                seg_turns[gid] = turns
                seg_edges[gid] = edges
                if vpts.ndim == 2 and vpts.shape[0] >= 2:
                    seg_vpts[gid] = vpts

                closed_raw = s.get('closed', None)  # 可能缺失
                closed_infer = (len(turns) == 0) and _is_cycle_from_edges(edges)
                seg_closed[gid] = bool(closed_raw) or bool(closed_infer)

                if args.verbose and closed_raw is not None and (bool(closed_raw) != bool(seg_closed[gid])):
                    print(f'[demo][info] gid={gid} closed_raw={closed_raw} -> closed_infer={closed_infer} turn_splits={len(turns)} edges={len(edges)}')

            if args.show_segments:
                for gid, vpts in seg_vpts.items():
                    _add_polyline(p, vpts, color='navy', width=args.seg_line_width, closed=bool(seg_closed.get(gid, False)))
        except Exception as e:
            print(f'[demo] segments_debug 读取失败：{e}')

    # curve
    if args.show_curve and os.path.exists(curve_pts_path) and os.path.exists(curve_gid_path):
        curve_pts = np.load(curve_pts_path)
        curve_gid = np.load(curve_gid_path)

        if curve_pts.ndim == 2 and curve_pts.shape[0] > 0:
            bbox_pts.append(curve_pts)

        gid_to_idx = _group_indices(curve_gid)

        n_closed, n_open = 0, 0

        for gid, idxs in gid_to_idx.items():
            pts = np.asarray(curve_pts[idxs], dtype=float)
            if pts.shape[0] < 2:
                continue

            closed = bool(seg_closed.get(int(gid), False))

            # 排序（如果没有 seg_vpts 或无 scipy 就回退原序）
            if args.order_curve_by_segment and (int(gid) in seg_vpts):
                order = _order_points_by_segment_arclen(pts, seg_vpts[int(gid)])
                pts = pts[order]

            color = 'orange' if closed else 'blue'
            if closed:
                n_closed += 1
            else:
                n_open += 1

            p.add_points(pts, color=color, point_size=max(1.0, float(args.curve_point_size)), render_points_as_spheres=True)
            _add_polyline(p, pts, color=color, width=args.curve_line_width, closed=closed)

        if args.verbose:
            print(f'[demo] curve groups: closed={n_closed}, open={n_open}')

    # band
    if args.show_band and os.path.exists(band_path):
        band = np.load(band_path)
        if band.ndim == 2 and band.shape[0] > 0:
            p.add_points(band, color='green', point_size=max(1.0, float(args.band_size)), render_points_as_spheres=True)

    p.add_axes()
    p.enable_eye_dome_lighting()
    p.show()


if __name__ == '__main__':
    main()
