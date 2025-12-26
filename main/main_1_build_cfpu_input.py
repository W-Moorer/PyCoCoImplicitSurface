import argparse
import os
import sys
import json
import numpy as np

# 添加项目根目录到Python路径
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)

from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree


def _get_builder():
    from src.precompute import build_cfpu_input
    return build_cfpu_input


def _import_sharp_tools():
    # 都来自你提供的 precompute.py（单文件整合版 / src.precompute）
    from src.precompute import (
        read_mesh,
        detect_sharp_edges,
        detect_sharp_junctions_degree,
        build_sharp_segments,
    )
    return read_mesh, detect_sharp_edges, detect_sharp_junctions_degree, build_sharp_segments


def _safe_normalize(v):
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return v / n


def _extract_faces_triangles(mesh):
    """
    pyvista PolyData faces: [3, a,b,c, 3, a,b,c, ...]
    """
    faces = mesh.faces
    faces = np.asarray(faces)
    if faces.ndim != 1 or faces.size % 4 != 0:
        # 尝试 reshape
        faces = faces.reshape(-1, 4)
    else:
        faces = faces.reshape(-1, 4)
    return faces[:, 1:].astype(np.int64)


def _compute_sharp_Lmin(points, faces, sharp_edges):
    """
    从 sharp_edges 的 face1/face2 收集到的三角形集合中，取最短边。
    若收集失败则退化为全局最短边。
    """
    face_ids = set()
    for e in sharp_edges:
        if not isinstance(e, dict):
            continue
        if 'face1' in e:
            face_ids.add(int(e['face1']))
        if 'face2' in e:
            face_ids.add(int(e['face2']))

    def tri_min_edge(fid_list):
        tri = faces[np.asarray(fid_list, dtype=np.int64)]
        pa = points[tri[:, 0]]
        pb = points[tri[:, 1]]
        pc = points[tri[:, 2]]
        lab = np.linalg.norm(pa - pb, axis=1)
        lbc = np.linalg.norm(pb - pc, axis=1)
        lca = np.linalg.norm(pc - pa, axis=1)
        return float(np.min(np.concatenate([lab, lbc, lca], axis=0)))

    if len(face_ids) > 0:
        Lmin = tri_min_edge(sorted(list(face_ids)))
    else:
        # fallback: 全局
        tri = faces
        pa = points[tri[:, 0]]
        pb = points[tri[:, 1]]
        pc = points[tri[:, 2]]
        lab = np.linalg.norm(pa - pb, axis=1)
        lbc = np.linalg.norm(pb - pc, axis=1)
        lca = np.linalg.norm(pc - pa, axis=1)
        Lmin = float(np.min(np.concatenate([lab, lbc, lca], axis=0)))

    return max(Lmin, 1e-12)


def _fit_bspline_and_sample_equal_arclen_with_tangent(
    P,
    step,
    oversample=2000,
    closed: bool = False,
    min_samples: int = 2,
):
    """
    对 polyline 点 P (N,3) 做 B样条（N不足自动降阶），按近似等弧长步长 step 采样。
    - closed=True：按闭合曲线处理（periodic spline + 弧长含首尾闭合段），采样不包含重复端点。
    - min_samples：每条 segment 至少采样点数下限（避免过稀）。

    返回：
      C (Q,3) 采样点
      T (Q,3) 单位切向
    """
    P = np.asarray(P, dtype=float)
    if P.shape[0] == 0:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=float)
    if P.shape[0] == 1:
        return P.copy(), np.tile(np.array([[0.0, 0.0, 1.0]], dtype=float), (1, 1))

    # 若闭合且首尾重复，去掉末尾重复点（避免 per=1 时退化）
    if closed and P.shape[0] >= 3:
        if np.linalg.norm(P[0] - P[-1]) < 1e-12:
            P = P[:-1]

    # 去掉连续重复点，避免 splprep 报错
    d = np.linalg.norm(P[1:] - P[:-1], axis=1)
    keep = np.ones(P.shape[0], dtype=bool)
    keep[1:] = d > 1e-12
    P = P[keep]
    if P.shape[0] == 1:
        return P.copy(), np.tile(np.array([[0.0, 0.0, 1.0]], dtype=float), (1, 1))

    # 2 点：线性（闭合在 2 点上没有意义，按 open 处理）
    if P.shape[0] == 2:
        L = float(np.linalg.norm(P[1] - P[0]))
        n = max(int(min_samples), int(np.ceil(L / max(step, 1e-12))) + 1)
        t = np.linspace(0.0, 1.0, n)
        C = (1 - t)[:, None] * P[0] + t[:, None] * P[1]
        T = np.tile(_safe_normalize(P[1] - P[0])[None, :], (C.shape[0], 1))
        return C, T

    k = min(3, P.shape[0] - 1)
    try:
        tck, _u = splprep([P[:, 0], P[:, 1], P[:, 2]], s=0.0, k=k, per=1 if closed else 0)
    except TypeError:
        # 兼容老版本 scipy：可能没有 per 参数
        tck, _u = splprep([P[:, 0], P[:, 1], P[:, 2]], s=0.0, k=k)

    # 高密度采样近似弧长
    M = max(int(oversample), 200)
    uu = np.linspace(0.0, 1.0, M, endpoint=not closed)
    Q = np.vstack(splev(uu, tck)).T

    if Q.shape[0] < 2:
        C = Q[:1].copy()
        T = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=float), (1, 1))
        return C, T

    if not closed:
        seg = np.linalg.norm(Q[1:] - Q[:-1], axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(cum[-1])
        uu_ext = uu
        cum_ext = cum
    else:
        seg = np.linalg.norm(Q[1:] - Q[:-1], axis=1)
        closure = float(np.linalg.norm(Q[0] - Q[-1]))
        seg2 = np.concatenate([seg, [closure]], axis=0)
        cum_ext = np.concatenate([[0.0], np.cumsum(seg2)], axis=0)  # (M+1,)
        uu_ext = np.concatenate([uu, [1.0]], axis=0)               # (M+1,)
        total = float(cum_ext[-1])

    if total <= 1e-12:
        C = Q[:1].copy()
        T = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=float), (1, 1))
        return C, T

    step = max(float(step), total / 10000.0)

    if not closed:
        n = max(int(min_samples), int(np.floor(total / step)) + 1)
        svals = np.linspace(0.0, total, n, endpoint=True)
    else:
        n = max(int(min_samples), int(np.ceil(total / step)))
        svals = np.linspace(0.0, total, n, endpoint=False)

    u_new = np.interp(svals, cum_ext, uu_ext)
    C = np.vstack(splev(u_new, tck)).T

    # 切向来自 spline 一阶导
    dC = np.vstack(splev(u_new, tck, der=1)).T
    T = np.zeros_like(dC)
    for i in range(dC.shape[0]):
        T[i] = _safe_normalize(dC[i])

    # 极端情况下导数可能退化，再用差分兜底
    bad = np.linalg.norm(T, axis=1) < 1e-8
    if np.any(bad):
        Cd = np.gradient(C, axis=0)
        for i in np.where(bad)[0]:
            T[i] = _safe_normalize(Cd[i])

    return C, T


def _point_segment_distance(p, a, b):
    """点到线段距离（返回 dist, t(0..1)）"""
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-18:
        return float(np.linalg.norm(p - a)), 0.0
    t = float(np.dot(p - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    q = a + t * ab
    return float(np.linalg.norm(p - q)), t


def export_sharp_curve_constraints(
    input_mesh_path: str,
    out_dir: str,
    angle_threshold: float,
    edge_split_threshold: float,
    require_step_face_id_diff: bool,
    curve_step: float,
    curve_step_factor: float,
    curve_oversample: int,
    max_curve_points_per_feature_patch: int,
    curve_min_samples: int = 2,
    tol_ratio_for_b1: float = 0.01,   # 用于生成 tol_geom_default = tol_ratio*sharp_Lmin
):
    """
    导出尖锐边曲线约束点（做法A），并额外导出B1需要的信息（tangent、两侧法向、sharp_Lmin）。

    输出到 out_dir：
      - sharp_curve_points_raw.npy           (Q,3) 原坐标
      - sharp_curve_points_unit.npy          (Q,3) unit box 坐标
      - sharp_curve_group_id.npy             (Q,)  segment id
      - sharp_curve_tangents.npy             (Q,3) 单位切向（B1用）
      - sharp_curve_n1.npy                   (Q,3) 一侧面法向（B1用）
      - sharp_curve_n2.npy                   (Q,3) 另一侧面法向（B1用）
      - sharp_curve_meta.json                元信息（合并写，不覆盖丢字段）
      - sharp_curve_feature_patch_map.json   (可选) feature patch -> curve indices
    """
    nodes_path = os.path.join(out_dir, "nodes.txt")
    patches_path = os.path.join(out_dir, "patches.txt")
    radii_path = os.path.join(out_dir, "radii.txt")
    featcnt_path = os.path.join(out_dir, "feature_count.txt")

    if not os.path.exists(nodes_path):
        print(f"[sharp-curve] 跳过：未找到 {nodes_path}")
        return

    nodes = np.loadtxt(nodes_path)
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        print(f"[sharp-curve] 跳过：nodes.txt 形状异常 {nodes.shape}")
        return

    # unit box 变换（和 nodes 一致）
    minxx = nodes.min(axis=0)
    maxxx = nodes.max(axis=0)
    scale = float(np.max(maxxx - minxx))
    if scale <= 0:
        scale = 1.0

    read_mesh, detect_sharp_edges, detect_sharp_junctions_degree, build_sharp_segments = _import_sharp_tools()
    mesh = read_mesh(input_mesh_path, compute_split_normals=False)

    # faces / points
    points = np.asarray(mesh.points, dtype=float)
    faces = _extract_faces_triangles(mesh)

    # cell normals（用于 n1/n2）
    if 'cell_normals' in mesh.cell_data:
        cell_normals = np.asarray(mesh.cell_data['cell_normals'], dtype=float)
    else:
        try:
            mesh.compute_normals(inplace=True, cell_normals=True, point_normals=False, split_vertices=False)
        except Exception:
            mesh.compute_normals(inplace=True)
        # pyvista常用字段名为 'Normals'
        cell_normals = np.asarray(mesh.cell_data.get('Normals', np.zeros((mesh.n_cells, 3))), dtype=float)

    # 1) 检测尖锐边
    sharp_edges, _lines = detect_sharp_edges(
        mesh,
        angle_threshold=angle_threshold,
        edge_split_threshold=edge_split_threshold,
        require_step_face_id_diff=require_step_face_id_diff
    )

    # 没有尖锐边：输出空文件，保持流水线稳定
    if (sharp_edges is None) or (len(sharp_edges) == 0):
        print(f"[sharp-curve] 未检测到尖锐边：{os.path.basename(input_mesh_path)}")
        np.save(os.path.join(out_dir, "sharp_curve_points_raw.npy"), np.empty((0, 3)))
        np.save(os.path.join(out_dir, "sharp_curve_points_unit.npy"), np.empty((0, 3)))
        np.save(os.path.join(out_dir, "sharp_curve_group_id.npy"), np.empty((0,), dtype=np.int32))
        np.save(os.path.join(out_dir, "sharp_curve_tangents.npy"), np.empty((0, 3)))
        np.save(os.path.join(out_dir, "sharp_curve_n1.npy"), np.empty((0, 3)))
        np.save(os.path.join(out_dir, "sharp_curve_n2.npy"), np.empty((0, 3)))

        meta_path = os.path.join(out_dir, "sharp_curve_meta.json")
        old = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    old = json.load(f) or {}
            except Exception:
                old = {}
        old.update({
            "n_points": 0,
            "minxx": minxx.tolist(),
            "scale": scale,
            "source": "none",
            "sharp_Lmin": 0.0,
            "tol_ratio_default": float(tol_ratio_for_b1),
            "tol_geom_default": 0.0,
        })
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(old, f, ensure_ascii=False, indent=2)
        return

    # 2) 计算 sharp_Lmin（B1推 tol_geom 的依据）
    sharp_Lmin = _compute_sharp_Lmin(points, faces, sharp_edges)
    tol_geom_default = float(tol_ratio_for_b1) * float(sharp_Lmin)

    # 3) 分段成 polyline/loop
    junctions = detect_sharp_junctions_degree(mesh, sharp_edges)
    # build_sharp_segments 的 cell_normals 参数在你版本里并不参与计算，给个占位即可
    segments = build_sharp_segments(
        sharp_edges, junctions, points, cell_normals, angle_turn_threshold=90.0
    )

    # 3.1) 输出 segments_debug（用于后续 sharp-band 与可视化对齐，避免二次分段导致 gid 不一致）
    # 格式对齐 demo_auto_turn_points.py：{'segments':[{'id','closed','vertices','vertex_points','edges','turn_splits'}]}
    try:
        debug_segments = []
        for gid, seg in enumerate(segments):
            verts = [int(v) for v in (seg.get('vertices', []) or [])]
            edges_list = [(int(a), int(b)) for (a, b) in (seg.get('edges', []) or [])]
            vpts = [points[v].tolist() for v in verts] if len(verts) > 0 else []
            debug_segments.append({
                'id': int(gid),
                'closed': bool(seg.get('closed', False)),
                'vertices': verts,
                'vertex_points': vpts,
                'edges': edges_list,
                'turn_splits': [int(x) for x in (seg.get('turn_splits', []) or [])],
            })
        with open(os.path.join(out_dir, "sharp_segments_debug.json"), "w", encoding="utf-8") as f:
            json.dump({'segments': debug_segments}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[sharp-curve] segments_debug 输出失败（不影响主流程）：{e}")


    # 用于 edge -> record 映射（拿 face1/face2）
    edge_info = {}
    for e in sharp_edges:
        a = int(e['point1_idx'])
        b = int(e['point2_idx'])
        k = (a, b) if a < b else (b, a)
        edge_info[k] = e

    # 4) 每条 segment：B 样条 + 等弧长采样，并为每个曲线点赋 tangent / n1 / n2
    curve_pts_all = []
    curve_tan_all = []
    curve_n1_all = []
    curve_n2_all = []
    gid_all = []

    for gid, seg in enumerate(segments):
        verts = seg.get('vertices', None)
        if verts is None:
            continue
        verts = [int(v) for v in verts]
        if len(verts) < 2:
            continue

        P = points[np.asarray(verts, dtype=int)]

        # 步长：优先 curve_step；否则 curve_step_factor * (相邻点平均距离)
        if curve_step is not None and curve_step > 0:
            step = float(curve_step)
        else:
            d = np.linalg.norm(P[1:] - P[:-1], axis=1)
            mean_d = float(np.mean(d)) if d.size > 0 else 0.0
            if mean_d <= 0:
                mean_d = scale * 1e-3
            step = max(curve_step_factor * mean_d, 1e-12)

        C, T = _fit_bspline_and_sample_equal_arclen_with_tangent(P, step=step, oversample=curve_oversample, closed=bool(seg.get('closed', False)), min_samples=int(curve_min_samples))
        if C.shape[0] == 0:
            continue

        # 构造本 segment 的 edge 列表，用于找每个曲线点最近的 sharp edge，从而拿 face1/face2 normals
        seg_edges = seg.get('edges', [])
        seg_edge_a = []
        seg_edge_b = []
        seg_fn1 = []
        seg_fn2 = []
        seg_mid = []

        for (u, v) in seg_edges:
            u = int(u); v = int(v)
            k = (u, v) if u < v else (v, u)
            rec = edge_info.get(k, None)
            if rec is None:
                continue
            f1 = int(rec.get('face1', -1))
            f2 = int(rec.get('face2', -1))
            if not (0 <= f1 < cell_normals.shape[0]) or not (0 <= f2 < cell_normals.shape[0]):
                continue
            a = points[u]
            b = points[v]
            seg_edge_a.append(a)
            seg_edge_b.append(b)
            seg_fn1.append(_safe_normalize(cell_normals[f1]))
            seg_fn2.append(_safe_normalize(cell_normals[f2]))
            seg_mid.append(0.5 * (a + b))

        seg_edge_a = np.asarray(seg_edge_a, dtype=float)
        seg_edge_b = np.asarray(seg_edge_b, dtype=float)
        seg_fn1 = np.asarray(seg_fn1, dtype=float)
        seg_fn2 = np.asarray(seg_fn2, dtype=float)
        seg_mid = np.asarray(seg_mid, dtype=float)

        if seg_mid.shape[0] > 0:
            tree_mid = cKDTree(seg_mid)

        n1_list = []
        n2_list = []
        for p in C:
            if seg_mid.shape[0] == 0:
                n1_list.append(np.array([0.0, 0.0, 1.0], dtype=float))
                n2_list.append(np.array([0.0, 0.0, 1.0], dtype=float))
                continue

            # 先用 midpoint KDTree 粗筛，再精算点到线段距离
            kq = min(8, seg_mid.shape[0])
            _, idxs = tree_mid.query(p, k=kq)
            idxs = np.atleast_1d(idxs)

            best = None
            best_i = None
            for ii in idxs:
                a = seg_edge_a[int(ii)]
                b = seg_edge_b[int(ii)]
                dist, _t = _point_segment_distance(p, a, b)
                if best is None or dist < best:
                    best = dist
                    best_i = int(ii)

            n1_list.append(seg_fn1[best_i])
            n2_list.append(seg_fn2[best_i])

        N1 = np.asarray(n1_list, dtype=float)
        N2 = np.asarray(n2_list, dtype=float)

        curve_pts_all.append(C)
        curve_tan_all.append(T)
        curve_n1_all.append(N1)
        curve_n2_all.append(N2)
        gid_all.append(np.full((C.shape[0],), gid, dtype=np.int32))

    if not curve_pts_all:
        print(f"[sharp-curve] 分段后没有可用曲线：{os.path.basename(input_mesh_path)}")
        np.save(os.path.join(out_dir, "sharp_curve_points_raw.npy"), np.empty((0, 3)))
        np.save(os.path.join(out_dir, "sharp_curve_points_unit.npy"), np.empty((0, 3)))
        np.save(os.path.join(out_dir, "sharp_curve_group_id.npy"), np.empty((0,), dtype=np.int32))
        np.save(os.path.join(out_dir, "sharp_curve_tangents.npy"), np.empty((0, 3)))
        np.save(os.path.join(out_dir, "sharp_curve_n1.npy"), np.empty((0, 3)))
        np.save(os.path.join(out_dir, "sharp_curve_n2.npy"), np.empty((0, 3)))

        meta_path = os.path.join(out_dir, "sharp_curve_meta.json")
        old = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    old = json.load(f) or {}
            except Exception:
                old = {}
        old.update({
            "n_points": 0,
            "minxx": minxx.tolist(),
            "scale": scale,
            "source": "segments(empty)",
            "sharp_Lmin": float(sharp_Lmin),
            "tol_ratio_default": float(tol_ratio_for_b1),
            "tol_geom_default": float(tol_geom_default),
        })
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(old, f, ensure_ascii=False, indent=2)
        return

    curve_pts = np.vstack(curve_pts_all)
    curve_tan = np.vstack(curve_tan_all)
    curve_n1 = np.vstack(curve_n1_all)
    curve_n2 = np.vstack(curve_n2_all)
    group_id = np.concatenate(gid_all)

    # 去重：只在“每个 group 内”去重。
    # 重要：不同 group 之间允许坐标重复（junction/共享端点很常见），否则会导致：
    # - 闭合环被“拆掉一个点”而看似不闭合
    # - 段与段在共享端点处出现缺口/跳连（可视化与 sharp-band 都会受影响）
    keep = np.zeros((curve_pts.shape[0],), dtype=bool)
    for gid in np.unique(group_id):
        idx = np.where(group_id == gid)[0]
        if idx.size == 0:
            continue
        pts_g = curve_pts[idx]
        key_g = np.round(pts_g, decimals=10)
        _, uniq_local = np.unique(key_g, axis=0, return_index=True)
        uniq_local = np.sort(uniq_local)
        keep[idx[uniq_local]] = True

    curve_pts = curve_pts[keep]
    curve_tan = curve_tan[keep]
    curve_n1 = curve_n1[keep]
    curve_n2 = curve_n2[keep]
    group_id = group_id[keep]

    curve_unit = (curve_pts - minxx) / scale

    # --- 保存（做法A + B1需要） ---
    np.save(os.path.join(out_dir, "sharp_curve_points_raw.npy"), curve_pts)
    np.save(os.path.join(out_dir, "sharp_curve_points_unit.npy"), curve_unit)
    np.save(os.path.join(out_dir, "sharp_curve_group_id.npy"), group_id)
    np.save(os.path.join(out_dir, "sharp_curve_tangents.npy"), curve_tan)
    np.save(os.path.join(out_dir, "sharp_curve_n1.npy"), curve_n1)
    np.save(os.path.join(out_dir, "sharp_curve_n2.npy"), curve_n2)

    # --- meta：合并写（不覆盖丢字段） ---
    meta_path = os.path.join(out_dir, "sharp_curve_meta.json")
    old = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                old = json.load(f) or {}
        except Exception:
            old = {}

    new_meta = {
        "source": "detect_sharp_edges + build_sharp_segments + cubic_bspline",
        "n_points": int(curve_pts.shape[0]),
        "n_segments": int(len(segments)),
        "minxx": minxx.tolist(),
        "scale": scale,
        "curve_step": None if curve_step is None else float(curve_step),
        "curve_step_factor": float(curve_step_factor),
        "curve_oversample": int(curve_oversample),
        "curve_min_samples": int(curve_min_samples),
        "angle_threshold": float(angle_threshold),
        "edge_split_threshold": None if edge_split_threshold is None else float(edge_split_threshold),
        "require_step_face_id_diff": bool(require_step_face_id_diff),

        # B1关键字段
        "sharp_Lmin": float(sharp_Lmin),
        "tol_ratio_default": float(tol_ratio_for_b1),
        "tol_geom_default": float(tol_geom_default),

        "note": "meta 为合并写；B1 需要 sharp_Lmin 推导 tol_geom；epsilon 在 main_3 内由 tol_geom+θ+h 推导"
    }
    old.update(new_meta)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(old, f, ensure_ascii=False, indent=2)

    # 5) （可选）输出 feature patch -> curve indices 映射（保留你原逻辑）
    try:
        if os.path.exists(patches_path) and os.path.exists(radii_path) and os.path.exists(featcnt_path):
            patches = np.loadtxt(patches_path)
            radii = np.loadtxt(radii_path)
            feature_count = int(open(featcnt_path, "r").read().strip())

            feature_count = max(0, min(int(feature_count), patches.shape[0]))
            if feature_count > 0 and curve_unit.shape[0] > 0:
                patches_unit = (patches[:feature_count] - minxx) / scale
                radii_unit = radii[:feature_count] / scale

                tree = cKDTree(curve_unit)
                mapping = {}
                for i in range(feature_count):
                    idx = tree.query_ball_point(patches_unit[i], float(radii_unit[i]))
                    if not idx:
                        continue
                    if (max_curve_points_per_feature_patch is not None) and (len(idx) > int(max_curve_points_per_feature_patch)):
                        idx = np.asarray(idx, dtype=int)
                        d = np.linalg.norm(curve_unit[idx] - patches_unit[i], axis=1)
                        order = np.argsort(d)
                        idx = idx[order[:int(max_curve_points_per_feature_patch)]].tolist()
                    mapping[str(i)] = list(map(int, idx))

                with open(os.path.join(out_dir, "sharp_curve_feature_patch_map.json"), "w", encoding="utf-8") as f:
                    json.dump(mapping, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[sharp-curve] patch 映射输出失败（不影响主流程）：{e}")

    print(f"[sharp-curve] 导出完成：{os.path.basename(input_mesh_path)} | points={curve_pts.shape[0]} | sharp_Lmin={sharp_Lmin:.3e} -> {out_dir}")




def append_sharp_band_oriented_points(
    input_mesh_path: str,
    out_dir: str,
    angle_threshold: float,
    edge_split_threshold: float,
    require_step_face_id_diff: bool,
    band_tol_geom: float | None = None,
    band_offset_scale: float = 1.0,
    band_offset_cap_ratio: float = 0.25,
    band_max_points: int = 0,
):
    """
    在尖锐边曲线附近生成“两侧偏移的 oriented points”（点+法向），并追加到 nodes.txt / normals.txt。

    目的：
      - 把“尖锐顶点的偏移法向约束”推广到整条尖锐边中段，减少沿边波浪起伏。
      - 偏移距离 δ 受控：δ = min(band_offset_scale*tol_geom, band_offset_cap_ratio*local_step)

    偏移方向：
      给定切向 t、侧法向 n1/n2
        e1 = normalize(t × n1)
        e2 = normalize(n2 × t)
      再用 face1/face2 三角形质心 c1/c2 选符号，保证 e1 指向 face1 内部、e2 指向 face2 内部：
        if dot(c1 - p, e1) < 0: e1 = -e1
        if dot(c2 - p, e2) < 0: e2 = -e2

    输出：
      - nodes.txt / normals.txt 追加 2*K 个点（每个曲线点两侧各一个）
      - 额外保存（调试用）：
          sharp_band_points_raw.npy, sharp_band_normals.npy, sharp_band_meta.json
    """
    nodes_path = os.path.join(out_dir, "nodes.txt")
    normals_path = os.path.join(out_dir, "normals.txt")
    if (not os.path.exists(nodes_path)) or (not os.path.exists(normals_path)):
        print(f"[sharp-band] 跳过：缺少 nodes/normals：{out_dir}")
        return

    # 读取尖锐曲线数据（来自 export_sharp_curve_constraints）
    p_raw_path = os.path.join(out_dir, "sharp_curve_points_raw.npy")
    t_path = os.path.join(out_dir, "sharp_curve_tangents.npy")
    n1_path = os.path.join(out_dir, "sharp_curve_n1.npy")
    n2_path = os.path.join(out_dir, "sharp_curve_n2.npy")
    gid_path = os.path.join(out_dir, "sharp_curve_group_id.npy")
    meta_path = os.path.join(out_dir, "sharp_curve_meta.json")

    if not (os.path.exists(p_raw_path) and os.path.exists(t_path) and os.path.exists(n1_path) and os.path.exists(n2_path) and os.path.exists(gid_path)):
        print(f"[sharp-band] 跳过：缺少 sharp_curve_*.npy：{out_dir}")
        return

    curve_pts = np.load(p_raw_path)
    curve_tan = np.load(t_path)
    curve_n1 = np.load(n1_path)
    curve_n2 = np.load(n2_path)
    group_id = np.load(gid_path)

    if curve_pts.size == 0:
        print(f"[sharp-band] 跳过：无尖锐曲线点：{os.path.basename(input_mesh_path)}")
        return

    # tol_geom：优先参数，否则读 meta.tol_geom_default，否则用 1e-4 * scale 兜底
    tol_geom = None
    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f) or {}
        except Exception:
            meta = {}
    if band_tol_geom is not None and band_tol_geom > 0:
        tol_geom = float(band_tol_geom)
    elif "tol_geom_default" in meta and float(meta.get("tol_geom_default", 0.0)) > 0:
        tol_geom = float(meta["tol_geom_default"])
    else:
        # fallback：用 bbox 尺度估一个极小值
        nodes0 = np.loadtxt(nodes_path)
        minxx = nodes0.min(axis=0)
        maxxx = nodes0.max(axis=0)
        scale = float(np.max(maxxx - minxx)) if float(np.max(maxxx - minxx)) > 0 else 1.0
        tol_geom = 1e-4 * scale

    tol_geom = float(max(tol_geom, 1e-15))
    delta0 = float(max(band_offset_scale, 0.0)) * tol_geom

    # 为确定偏移符号：重新读 mesh 并复现“曲线点 -> 最近 sharp edge -> face1/face2”
    read_mesh, detect_sharp_edges, detect_sharp_junctions_degree, build_sharp_segments = _import_sharp_tools()
    mesh = read_mesh(input_mesh_path, compute_split_normals=False)
    points = np.asarray(mesh.points, dtype=float)
    faces = _extract_faces_triangles(mesh)

    # cell normals（用于 face 法向），以及 face centroids
    if 'cell_normals' in mesh.cell_data:
        cell_normals = np.asarray(mesh.cell_data['cell_normals'], dtype=float)
    else:
        # 兜底：用三角形几何计算
        tri = points[faces]
        nn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        cell_normals = np.asarray([_safe_normalize(v) for v in nn], dtype=float)
    face_centroids = points[faces].mean(axis=1)

    ret = detect_sharp_edges(
        mesh,
        angle_threshold=angle_threshold,
        edge_split_threshold=edge_split_threshold,
        require_step_face_id_diff=require_step_face_id_diff
    )
    # detect_sharp_edges 可能返回 (sharp_edges, sharp_edge_lines) 或仅返回 sharp_edges
    if isinstance(ret, tuple):
        sharp_edges = ret[0]
    else:
        sharp_edges = ret
    if (sharp_edges is None) or (len(sharp_edges) == 0):
        print(f"[sharp-band] 跳过：未检测到尖锐边（无法确定偏移符号）：{os.path.basename(input_mesh_path)}")
        return

    # segments：优先加载 export 阶段写出的 sharp_segments_debug.json，
    # 避免在 sharp-band 阶段二次 build_sharp_segments 导致 gid 顺序不一致（会造成“断点/错段/偏置到错误边”）
    segments = None
    seg_debug_path = os.path.join(out_dir, "sharp_segments_debug.json")
    if os.path.exists(seg_debug_path):
        try:
            with open(seg_debug_path, "r", encoding="utf-8") as f:
                d = json.load(f) or {}
            segs = d.get("segments", []) or []
            # 按 id 排序，确保 gid 与导出阶段一致
            segs = sorted(segs, key=lambda x: int(x.get("id", 0)))
            segments = []
            for s in segs:
                segments.append({
                    "vertices": [int(v) for v in (s.get("vertices", []) or [])],
                    "edges": [(int(a), int(b)) for (a, b) in (s.get("edges", []) or [])],
                    "closed": bool(s.get("closed", False)),
                    "turn_splits": [int(v) for v in (s.get("turn_splits", []) or [])],
                })
        except Exception as e:
            print(f"[sharp-band] 读取 sharp_segments_debug.json 失败，将回退到 build_sharp_segments：{e}")
            segments = None

    if segments is None:
        # junctions: 兼容不同签名 detect_sharp_junctions_degree(mesh, sharp_edges) / (sharp_edges)
        try:
            junctions = detect_sharp_junctions_degree(mesh, sharp_edges)
        except TypeError:
            junctions = detect_sharp_junctions_degree(sharp_edges)

        segments = build_sharp_segments(
            sharp_edges, junctions, points, cell_normals, angle_turn_threshold=90.0
        )

    # edge -> record（拿 face1/face2）
    edge_info = {}
    for e in sharp_edges:
        a = int(e['point1_idx'])
        b = int(e['point2_idx'])
        k = (a, b) if a < b else (b, a)
        edge_info[k] = e

    # 按 segment(gid) 处理：为该 gid 的 curve 点找到对应 edge 的 face1/face2 centroid
    # 先建立 gid->indices（保持原数组顺序，便于 local_step 估计）
    gid_to_idx = {}
    for i, g in enumerate(group_id.tolist()):
        gid_to_idx.setdefault(int(g), []).append(i)

    extra_pts = []
    extra_nrm = []
    used_delta = []

    # helper：估计每个点的 local_step（按 gid 内顺序）
    def _estimate_local_step(idx_list, closed: bool = False):
        n = len(idx_list)
        if n <= 1:
            return np.full((n,), np.nan, dtype=float)
        P = curve_pts[np.asarray(idx_list, dtype=int)]
        ds = np.full((n,), np.nan, dtype=float)

        if closed and n >= 3:
            for j in range(n):
                jp = (j - 1) % n
                jn = (j + 1) % n
                d_prev = np.linalg.norm(P[j] - P[jp])
                d_next = np.linalg.norm(P[jn] - P[j])
                ds[j] = min(d_prev, d_next)
        else:
            for j in range(n):
                d_prev = np.linalg.norm(P[j] - P[j - 1]) if j - 1 >= 0 else np.nan
                d_next = np.linalg.norm(P[j + 1] - P[j]) if j + 1 < n else np.nan
                ds[j] = np.nanmin([d_prev, d_next])

        # 兜底：用中位数填 NaN/0
        med = float(np.nanmedian(ds)) if np.isfinite(np.nanmedian(ds)) else float(np.mean(np.linalg.norm(np.diff(P, axis=0), axis=1)))
        if not np.isfinite(med) or med <= 0:
            med = 1.0
        ds = np.where((~np.isfinite(ds)) | (ds <= 0), med, ds)
        return ds

    for gid, seg in enumerate(segments):
        if gid not in gid_to_idx:
            continue
        idx_list = gid_to_idx[gid]
        if len(idx_list) == 0:
            continue

        # segment 的 edges -> midpoints KDTree（和 export_sharp_curve_constraints 一致）
        seg_edges = seg.get('edges', [])
        seg_edge_a = []
        seg_edge_b = []
        seg_face1 = []
        seg_face2 = []
        seg_mid = []

        for (u, v) in seg_edges:
            u = int(u); v = int(v)
            k = (u, v) if u < v else (v, u)
            rec = edge_info.get(k, None)
            if rec is None:
                continue
            f1 = int(rec.get('face1', -1))
            f2 = int(rec.get('face2', -1))
            if not (0 <= f1 < faces.shape[0]) or not (0 <= f2 < faces.shape[0]):
                continue
            a = points[u]
            b = points[v]
            seg_edge_a.append(a)
            seg_edge_b.append(b)
            seg_face1.append(f1)
            seg_face2.append(f2)
            seg_mid.append(0.5 * (a + b))

        if len(seg_mid) == 0:
            continue

        seg_edge_a = np.asarray(seg_edge_a, dtype=float)
        seg_edge_b = np.asarray(seg_edge_b, dtype=float)
        seg_face1 = np.asarray(seg_face1, dtype=int)
        seg_face2 = np.asarray(seg_face2, dtype=int)
        seg_mid = np.asarray(seg_mid, dtype=float)
        tree_mid = cKDTree(seg_mid)

        # gid 内的 local_step
        ds = _estimate_local_step(idx_list, closed=bool(seg.get('closed', False)))

        # 可选：限制总点数（每 gid）
        if band_max_points and band_max_points > 0 and len(idx_list) > int(band_max_points):
            # 均匀抽样
            keep = np.linspace(0, len(idx_list) - 1, int(band_max_points)).round().astype(int)
            idx_list = [idx_list[i] for i in keep.tolist()]
            ds = ds[keep]

        # 逐点生成两侧偏移点
        for jj, i in enumerate(idx_list):
            p = curve_pts[i]
            t = _safe_normalize(curve_tan[i])
            n1 = _safe_normalize(curve_n1[i])
            n2 = _safe_normalize(curve_n2[i])

            # 候选偏移方向（在各自切平面内、垂直于边）
            e1 = np.cross(t, n1)
            e2 = np.cross(n2, t)
            if np.linalg.norm(e1) < 1e-12:
                # 兜底：尝试用 n2
                e1 = np.cross(t, n2)
            if np.linalg.norm(e2) < 1e-12:
                e2 = np.cross(n1, t)
            e1 = _safe_normalize(e1)
            e2 = _safe_normalize(e2)

            # 找最近 edge（用 midpoints 粗筛 + 点线段距离精筛）
            _, idxs = tree_mid.query(p, k=min(8, seg_mid.shape[0]))
            if np.isscalar(idxs):
                idxs = [int(idxs)]
            best = None
            best_i = None
            for ii in idxs:
                a = seg_edge_a[int(ii)]
                b = seg_edge_b[int(ii)]
                dist, _ = _point_segment_distance(p, a, b)
                if best is None or dist < best:
                    best = dist
                    best_i = int(ii)
            if best_i is None:
                continue

            f1 = int(seg_face1[best_i])
            f2 = int(seg_face2[best_i])
            c1 = face_centroids[f1]
            c2 = face_centroids[f2]

            # 用面片质心确定 e1/e2 符号，保证朝向对应面片内部
            if float(np.dot(c1 - p, e1)) < 0.0:
                e1 = -e1
            if float(np.dot(c2 - p, e2)) < 0.0:
                e2 = -e2

            # delta：受 local_step 限制，避免过密点导致局部过拟合
            ds_i = float(ds[jj]) if jj < len(ds) else float(np.nanmedian(ds))
            if not np.isfinite(ds_i) or ds_i <= 0:
                ds_i = 1.0
            delta = min(delta0, float(band_offset_cap_ratio) * ds_i)
            delta = float(max(delta, 1e-15))

            q1 = p + delta * e1
            q2 = p + delta * e2

            extra_pts.append(q1); extra_nrm.append(n1)
            extra_pts.append(q2); extra_nrm.append(n2)
            used_delta.append(delta); used_delta.append(delta)

    if len(extra_pts) == 0:
        print(f"[sharp-band] 未生成任何边带点：{os.path.basename(input_mesh_path)}（可能 segments 匹配失败）")
        return

    extra_pts = np.asarray(extra_pts, dtype=float)
    extra_nrm = np.asarray(extra_nrm, dtype=float)
    used_delta = np.asarray(used_delta, dtype=float)

    # 去重（避免重复点导致局部病态）
    key = np.round(extra_pts, decimals=10)
    _, uniq_idx = np.unique(key, axis=0, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    extra_pts = extra_pts[uniq_idx]
    extra_nrm = extra_nrm[uniq_idx]
    used_delta = used_delta[uniq_idx]

    # 追加到 nodes/normals
    nodes = np.loadtxt(nodes_path)
    normals = np.loadtxt(normals_path)
    if normals.ndim == 1:
        normals = normals.reshape(-1, 3)
    if nodes.ndim == 1:
        nodes = nodes.reshape(-1, 3)

    nodes2 = np.vstack([nodes, extra_pts])
    normals2 = np.vstack([normals, extra_nrm])

    np.savetxt(nodes_path, nodes2)
    np.savetxt(normals_path, normals2)

    # debug 输出
    np.save(os.path.join(out_dir, "sharp_band_points_raw.npy"), extra_pts)
    np.save(os.path.join(out_dir, "sharp_band_normals.npy"), extra_nrm)
    with open(os.path.join(out_dir, "sharp_band_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "n_added": int(extra_pts.shape[0]),
            "tol_geom": float(tol_geom),
            "band_offset_scale": float(band_offset_scale),
            "band_offset_cap_ratio": float(band_offset_cap_ratio),
            "delta_min": float(np.min(used_delta)),
            "delta_max": float(np.max(used_delta)),
            "delta_mean": float(np.mean(used_delta)),
            "note": "两侧偏移 oriented points，用 face centroid 选偏移符号"
        }, f, ensure_ascii=False, indent=2)

    print(f"[sharp-band] 已追加 {extra_pts.shape[0]} 个边带 oriented points 到 nodes/normals | "
          f"delta≈[{np.min(used_delta):.3e},{np.max(used_delta):.3e}] mean={np.mean(used_delta):.3e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+')
    ap.add_argument('--out_root', default=os.path.join('output', 'cfpu_input'))

    ap.add_argument('--angle_threshold', type=float, default=50.0)
    ap.add_argument('--r_small_factor', type=float, default=0.5)
    ap.add_argument('--r_large_factor', type=float, default=3.0)
    ap.add_argument('--edge_split_threshold', type=float, default=None)
    ap.add_argument('--require_step_face_id_diff', action='store_true')

    # --- 导出曲线约束点（做法A） ---
    ap.add_argument('--no_export_sharp_curve', action='store_true',
                    help='不导出尖锐边曲线约束点（默认导出）')
    ap.add_argument('--curve_step', type=float, default=None,
                    help='曲线采样的绝对步长（模型单位）。给了它就忽略 curve_step_factor。')
    ap.add_argument('--curve_step_factor', type=float, default=0.5,
                    help='曲线采样步长系数：step = factor * (segment 相邻顶点平均距离)（默认 0.5）')
    ap.add_argument('--curve_oversample', type=int, default=2000,
                    help='B样条用于近似弧长的过采样点数（默认 2000）')
    ap.add_argument('--curve_min_samples', type=int, default=12,
                    help='每条尖锐边片段最少采样点数下限（默认 12，避免过稀导致断点/不稳定）')
    ap.add_argument('--max_curve_points_per_feature_patch', type=int, default=200,
                    help='输出 feature patch 映射时，每个 feature patch 最多保留多少个曲线点（默认 200）')

    # --- 新增：B1 默认误差比例（tol_geom_default=tol_ratio*sharp_Lmin）---
    ap.add_argument('--b1_tol_ratio', type=float, default=0.01,
                    help='写入 sharp_curve_meta.json：tol_geom_default = b1_tol_ratio * sharp_Lmin（默认 0.01）')

    
    # --- 新增：在尖锐边附近追加“边带 oriented points”（两侧偏移点+法向） ---
    ap.add_argument('--no_append_sharp_band', action='store_true',
                    help='不追加尖锐边边带 oriented points（默认追加，推荐开启以减少沿边波浪）')
    ap.add_argument('--band_tol_geom', type=float, default=None,
                    help='边带偏移的几何误差尺度（模型单位）。默认读取 sharp_curve_meta.json 的 tol_geom_default。')
    ap.add_argument('--band_offset_scale', type=float, default=1.0,
                    help='偏移距离系数：delta0 = band_offset_scale * tol_geom（默认 1.0）')
    ap.add_argument('--band_offset_cap_ratio', type=float, default=0.25,
                    help='偏移距离上限：delta <= band_offset_cap_ratio * local_step（默认 0.25）')
    ap.add_argument('--band_max_points', type=int, default=0,
                    help='每条 sharp segment 最多使用多少曲线点生成边带（0表示不限制，默认 0）')
    args = ap.parse_args()
    inputs = args.inputs
    if not inputs:
        inputs = [
            os.path.join('input', 'smooth_geometry', 'Ellipsoid_surface_cellnormals.vtp'),
            os.path.join('input', 'smooth_geometry', 'Ring_surface_cellnormals.vtp'),
            os.path.join('input', 'smooth_geometry', 'Sphere_surface_cellnormals.vtp'),
            os.path.join('input', 'nonsmooth_geometry', 'Cone_surface_cellnormals.vtp'),
            os.path.join('input', 'nonsmooth_geometry', 'Cylinder_surface_cellnormals.vtp'),
            os.path.join('input', 'nonsmooth_geometry', 'Cube_surface_cellnormals.vtp'),
            os.path.join('input', 'nonsmooth_geometry', 'Prism_surface_cellnormals.vtp'),
            os.path.join('input', 'nonsmooth_geometry', 'TruncatedRing_surface_cellnormals.vtp'),
            os.path.join('input', 'combinatorial_geometry', 'CompositeBody1_surface_cellnormals.vtp'),
            os.path.join('input', 'combinatorial_geometry', 'CompositeBody2_surface_cellnormals.vtp'),
            os.path.join('input', 'complex_geometry', 'Gear_surface_cellnormals.vtp'),
            os.path.join('input', 'complex_geometry', 'LinkedGear_surface_cellnormals.vtp'),
            os.path.join('input', 'complex_geometry', 'Nail_surface_cellnormals.vtp'),
            os.path.join('input', 'complex_geometry', 'PressureLubricatedCam_surface_cellnormals.vtp'),
            os.path.join('input', 'complex_geometry', 'SlidewayRotatingModel_surface_cellnormals.vtp'),
        ]

    build_cfpu_input = _get_builder()

    for inp in inputs:
        base = os.path.splitext(os.path.basename(inp))[0]
        out_dir = os.path.join(args.out_root, base + '_cfpu_input')

        # 1) 生成 CFPU 输入
        build_cfpu_input(
            inp,
            out_dir,
            args.angle_threshold,
            args.r_small_factor,
            args.r_large_factor,
            args.edge_split_threshold,
            args.require_step_face_id_diff
        )

        # 2) 导出尖锐边曲线约束点（做法A）+ B1需要字段
        if not args.no_export_sharp_curve:
            export_sharp_curve_constraints(
                input_mesh_path=inp,
                out_dir=out_dir,
                angle_threshold=args.angle_threshold,
                edge_split_threshold=args.edge_split_threshold,
                require_step_face_id_diff=args.require_step_face_id_diff,
                curve_step=args.curve_step,
                curve_step_factor=args.curve_step_factor,
                curve_oversample=args.curve_oversample,
                max_curve_points_per_feature_patch=args.max_curve_points_per_feature_patch,
                curve_min_samples=args.curve_min_samples,
                tol_ratio_for_b1=args.b1_tol_ratio
            )

            # 3) 新增：追加尖锐边“边带 oriented points”（两侧偏移点+法向）
            if not args.no_append_sharp_band:
                append_sharp_band_oriented_points(
                    input_mesh_path=inp,
                    out_dir=out_dir,
                    angle_threshold=args.angle_threshold,
                    edge_split_threshold=args.edge_split_threshold,
                    require_step_face_id_diff=args.require_step_face_id_diff,
                    band_tol_geom=args.band_tol_geom,
                    band_offset_scale=args.band_offset_scale,
                    band_offset_cap_ratio=args.band_offset_cap_ratio,
                    band_max_points=args.band_max_points,
                )



if __name__ == '__main__':
    main()