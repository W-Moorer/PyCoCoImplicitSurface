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


def _fit_bspline_and_sample_equal_arclen_with_tangent(P, step, oversample=2000):
    """
    对 polyline 点 P (N,3) 做 B样条（N不足自动降阶），按近似等弧长步长 step 采样。
    返回：
      C (Q,3) 采样点
      T (Q,3) 单位切向
    """
    P = np.asarray(P, dtype=float)
    if P.shape[0] == 0:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=float)
    if P.shape[0] == 1:
        return P.copy(), np.tile(np.array([[0.0, 0.0, 1.0]], dtype=float), (1, 1))

    # 去掉连续重复点，避免 splprep 报错
    d = np.linalg.norm(P[1:] - P[:-1], axis=1)
    keep = np.ones(P.shape[0], dtype=bool)
    keep[1:] = d > 1e-12
    P = P[keep]
    if P.shape[0] == 1:
        return P.copy(), np.tile(np.array([[0.0, 0.0, 1.0]], dtype=float), (1, 1))

    # 2 点：线性
    if P.shape[0] == 2:
        L = float(np.linalg.norm(P[1] - P[0]))
        n = max(2, int(np.ceil(L / max(step, 1e-12))) + 1)
        t = np.linspace(0.0, 1.0, n)
        C = (1 - t)[:, None] * P[0] + t[:, None] * P[1]
        T = np.tile(_safe_normalize(P[1] - P[0])[None, :], (C.shape[0], 1))
        return C, T

    k = min(3, P.shape[0] - 1)
    tck, _u = splprep([P[:, 0], P[:, 1], P[:, 2]], s=0.0, k=k)

    # 高密度采样近似弧长
    M = max(int(oversample), 200)
    uu = np.linspace(0.0, 1.0, M)
    Q = np.vstack(splev(uu, tck)).T
    seg = np.linalg.norm(Q[1:] - Q[:-1], axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])

    if total <= 1e-12:
        C = Q[:1].copy()
        T = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=float), (1, 1))
        return C, T

    step = max(float(step), total / 10000.0)
    n = max(2, int(np.floor(total / step)) + 1)
    svals = np.linspace(0.0, total, n)

    u_new = np.interp(svals, cum, uu)
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
    cell_normals_dummy = np.zeros((mesh.n_cells, 3), dtype=float)
    segments = build_sharp_segments(
        sharp_edges, junctions, points, cell_normals_dummy, angle_turn_threshold=90.0
    )

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

        C, T = _fit_bspline_and_sample_equal_arclen_with_tangent(P, step=step, oversample=curve_oversample)
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

    # 去重（同时同步过滤 tangent/n1/n2/group_id）
    key = np.round(curve_pts, decimals=10)
    _, uniq_idx = np.unique(key, axis=0, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    curve_pts = curve_pts[uniq_idx]
    curve_tan = curve_tan[uniq_idx]
    curve_n1 = curve_n1[uniq_idx]
    curve_n2 = curve_n2[uniq_idx]
    group_id = group_id[uniq_idx]

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
    ap.add_argument('--max_curve_points_per_feature_patch', type=int, default=200,
                    help='输出 feature patch 映射时，每个 feature patch 最多保留多少个曲线点（默认 200）')

    # --- 新增：B1 默认误差比例（tol_geom_default=tol_ratio*sharp_Lmin）---
    ap.add_argument('--b1_tol_ratio', type=float, default=0.01,
                    help='写入 sharp_curve_meta.json：tol_geom_default = b1_tol_ratio * sharp_Lmin（默认 0.01）')

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
                tol_ratio_for_b1=args.b1_tol_ratio
            )


if __name__ == '__main__':
    main()
