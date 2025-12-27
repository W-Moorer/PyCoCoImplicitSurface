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


def _fit_bspline_and_sample_equal_arclen(P, step, oversample=2000):
    """
    对 polyline 点 P (N,3) 做三次B样条（N不足自动降阶），然后按近似等弧长步长 step 采样。
    返回 C (Q,3) 采样点。
    """
    P = np.asarray(P, dtype=float)
    if P.shape[0] == 0:
        return np.empty((0, 3), dtype=float)
    if P.shape[0] == 1:
        return P.copy()

    # 去掉连续重复点，避免 splprep 报错
    d = np.linalg.norm(P[1:] - P[:-1], axis=1)
    keep = np.ones(P.shape[0], dtype=bool)
    keep[1:] = d > 1e-12
    P = P[keep]

    if P.shape[0] == 1:
        return P.copy()

    # 2 点：直接线性插值
    if P.shape[0] == 2:
        L = float(np.linalg.norm(P[1] - P[0]))
        n = max(2, int(np.ceil(L / max(step, 1e-12))) + 1)
        t = np.linspace(0.0, 1.0, n)
        return (1 - t)[:, None] * P[0] + t[:, None] * P[1]

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
        return Q[:1].copy()

    step = max(float(step), total / 10000.0)  # 兜底避免极端密度
    n = max(2, int(np.floor(total / step)) + 1)
    svals = np.linspace(0.0, total, n)

    u_new = np.interp(svals, cum, uu)
    C = np.vstack(splev(u_new, tck)).T
    return C


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
):
    """
    导出尖锐边“曲线零值约束点”（做法A要用的 C 点集），并导出 unit 坐标，保证与 cfpurecon 的缩放一致。

    【方案2（避免覆盖 + 补齐 region pair）】
    - 所有本函数输出文件名均带 _spline 后缀，避免覆盖 build_cfpu_input 内部 B1 导出的 sharp_curve_*.npy。
    - 同时额外输出每个曲线点的 region pair（来自 face_region_id.npy + sharp edge 上相邻面片的 region）。

    输出到 out_dir：
      - sharp_curve_points_raw_spline.npy        (Q,3) 原坐标系
      - sharp_curve_points_unit_spline.npy       (Q,3) unit box（按 nodes.txt 的 minxx/scale）
      - sharp_curve_group_id_spline.npy          (Q,)  segment id（与 build_sharp_segments 顺序一致）
      - sharp_curve_tangents_spline.npy          (Q,3) 近似切向（单位化）
      - sharp_curve_region_pair_spline.npy       (Q,2) 每点所属 region pair (ri,rj)，未知为 (-1,-1)
      - sharp_curve_meta_spline.json             元信息
      - sharp_curve_feature_patch_map_spline.json (可选) feature patch -> curve indices（便于 exactinterp 直接用）
    """
    nodes_path = os.path.join(out_dir, "nodes.txt")
    patches_path = os.path.join(out_dir, "patches.txt")
    radii_path = os.path.join(out_dir, "radii.txt")
    featcnt_path = os.path.join(out_dir, "feature_count.txt")
    face_region_path = os.path.join(out_dir, "face_region_id.npy")

    if not os.path.exists(nodes_path):
        print(f"[sharp-curve] 跳过：未找到 {nodes_path}")
        return

    nodes = np.loadtxt(nodes_path)
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        print(f"[sharp-curve] 跳过：nodes.txt 形状异常：{nodes.shape}")
        return

    # unit box 变换（和 cfpurecon 内部一致：minxx + scale）
    minxx = nodes.min(axis=0)
    maxxx = nodes.max(axis=0)
    scale = float(np.max(maxxx - minxx))
    if scale <= 0:
        scale = 1.0

    # （可选）加载 face -> region id（用于输出 region pair）
    face_region_id = None
    if os.path.exists(face_region_path):
        try:
            face_region_id = np.load(face_region_path)
        except Exception as e:
            print(f"[sharp-curve] 警告：读取 face_region_id.npy 失败，将不输出 region pair：{e}")
            face_region_id = None

    read_mesh, detect_sharp_edges, detect_sharp_junctions_degree, build_sharp_segments = _import_sharp_tools()
    mesh = read_mesh(input_mesh_path, compute_split_normals=False)

    # 1) 检测尖锐边（不使用任何 pkl）
    sharp_edges, _lines = detect_sharp_edges(
        mesh,
        angle_threshold=angle_threshold,
        edge_split_threshold=edge_split_threshold,
        require_step_face_id_diff=require_step_face_id_diff
    )

    if (sharp_edges is None) or (len(sharp_edges) == 0):
        print(f"[sharp-curve] 未检测到尖锐边：{os.path.basename(input_mesh_path)}")
        # 仍写一个 meta，便于流水线判断
        with open(os.path.join(out_dir, "sharp_curve_meta_spline.json"), "w", encoding="utf-8") as f:
            json.dump({
                "source": "none",
                "n_points": 0,
                "n_segments": 0,
                "minxx": minxx.tolist(),
                "scale": scale,
                "note": "no sharp edges detected"
            }, f, ensure_ascii=False, indent=2)
        return

    # 2) 分段成多条 polyline/loop
    junctions = detect_sharp_junctions_degree(mesh, sharp_edges)
    # build_sharp_segments 的 cell_normals 参数在你版本里并不参与计算，给个占位即可
    cell_normals_dummy = np.zeros((mesh.n_cells, 3), dtype=float)
    segments = build_sharp_segments(
        sharp_edges, junctions, np.asarray(mesh.points), cell_normals_dummy, angle_turn_threshold=90.0
    )

    # 3) 每条 segment：三次 B 样条 + 等弧长采样
    curve_pts_all = []
    gid_all = []
    tan_all = []
    rp_all = []

    pts_np = np.asarray(mesh.points)

    def _compute_tangents(C: np.ndarray) -> np.ndarray:
        """对采样点序列 C 计算单位切向（简单有限差分）"""
        if C.shape[0] == 0:
            return C.copy()
        T = np.zeros_like(C)
        if C.shape[0] == 1:
            T[0] = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            d = np.diff(C, axis=0)
            T[0] = d[0]
            T[-1] = d[-1]
            if C.shape[0] > 2:
                T[1:-1] = C[2:] - C[:-2]
        nrm = np.linalg.norm(T, axis=1, keepdims=True)
        nrm[nrm < 1e-12] = 1.0
        return T / nrm

    for gid, seg in enumerate(segments):
        verts = seg.get('vertices', None)
        if verts is None:
            continue
        verts = [int(v) for v in verts]
        if len(verts) < 2:
            continue

        P = pts_np[np.asarray(verts, dtype=int)]

        # 步长：优先 curve_step；否则 curve_step_factor * (相邻点平均距离)
        if curve_step is not None and curve_step > 0:
            step = float(curve_step)
        else:
            d = np.linalg.norm(P[1:] - P[:-1], axis=1)
            mean_d = float(np.mean(d)) if d.size > 0 else 0.0
            if mean_d <= 0:
                mean_d = scale * 1e-3
            step = max(curve_step_factor * mean_d, 1e-12)

        C = _fit_bspline_and_sample_equal_arclen(P, step=step, oversample=curve_oversample)
        if C.shape[0] == 0:
            continue

        # --- 计算该 segment 的 region pair（取其 sharp edges 上 face1/face2 对应 region 的众数）---
        pair = (-1, -1)
        if face_region_id is not None:
            pairs = []
            for e in seg.get('edges', []) or []:
                f1 = e.get('face1', None)
                f2 = e.get('face2', None)
                if f1 is None or f2 is None:
                    continue
                try:
                    r1 = int(face_region_id[int(f1)])
                    r2 = int(face_region_id[int(f2)])
                except Exception:
                    continue
                if (r1 < 0) or (r2 < 0) or (r1 == r2):
                    continue
                if r1 < r2:
                    pairs.append((r1, r2))
                else:
                    pairs.append((r2, r1))
            if pairs:
                from collections import Counter
                pair = Counter(pairs).most_common(1)[0][0]

        # --- 收集 ---
        curve_pts_all.append(C)
        gid_all.append(np.full((C.shape[0],), gid, dtype=np.int32))
        tan_all.append(_compute_tangents(C))
        rp_all.append(np.tile(np.asarray(pair, dtype=np.int32)[None, :], (C.shape[0], 1)))

    if not curve_pts_all:
        print(f"[sharp-curve] 分段后没有可用曲线：{os.path.basename(input_mesh_path)}")
        with open(os.path.join(out_dir, "sharp_curve_meta_spline.json"), "w", encoding="utf-8") as f:
            json.dump({
                "source": "detect_sharp_edges + build_sharp_segments + cubic_bspline",
                "n_points": 0,
                "n_segments": int(len(segments)),
                "minxx": minxx.tolist(),
                "scale": scale,
                "note": "no valid segments after processing"
            }, f, ensure_ascii=False, indent=2)
        return

    curve_pts = np.vstack(curve_pts_all)
    group_id = np.concatenate(gid_all, axis=0)
    tangents = np.vstack(tan_all)
    region_pair = np.vstack(rp_all)

    # 去重（避免 exactinterp 线性系统更病态）；对附带数组同步筛选
    key = np.round(curve_pts, decimals=10)
    _, uniq_idx = np.unique(key, axis=0, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    curve_pts = curve_pts[uniq_idx]
    group_id = group_id[uniq_idx]
    tangents = tangents[uniq_idx]
    region_pair = region_pair[uniq_idx]

    curve_unit = (curve_pts - minxx) / scale

    np.save(os.path.join(out_dir, "sharp_curve_points_raw_spline.npy"), curve_pts)
    np.save(os.path.join(out_dir, "sharp_curve_points_unit_spline.npy"), curve_unit)
    np.save(os.path.join(out_dir, "sharp_curve_group_id_spline.npy"), group_id)
    np.save(os.path.join(out_dir, "sharp_curve_tangents_spline.npy"), tangents)
    np.save(os.path.join(out_dir, "sharp_curve_region_pair_spline.npy"), region_pair)

    meta = {
        "source": "detect_sharp_edges + build_sharp_segments + cubic_bspline",
        "n_points": int(curve_pts.shape[0]),
        "n_segments": int(len(segments)),
        "minxx": minxx.tolist(),
        "scale": scale,
        "curve_step": None if curve_step is None else float(curve_step),
        "curve_step_factor": float(curve_step_factor),
        "curve_oversample": int(curve_oversample),
        "face_region_id_available": bool(face_region_id is not None),
        "outputs": {
            "points_raw": "sharp_curve_points_raw_spline.npy",
            "points_unit": "sharp_curve_points_unit_spline.npy",
            "group_id": "sharp_curve_group_id_spline.npy",
            "tangents": "sharp_curve_tangents_spline.npy",
            "region_pair": "sharp_curve_region_pair_spline.npy",
        },
        "note": "本函数输出均带 _spline 后缀，避免覆盖 build_cfpu_input(B1) 的 sharp_curve_*.npy；region_pair 为每点所属 (ri,rj)，未知为 (-1,-1)。",
    }
    with open(os.path.join(out_dir, "sharp_curve_meta_spline.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 4) （可选）输出 feature patch -> curve indices 映射（只对尖锐 patch 做）
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
                    r = max(float(radii_unit[i]), 1e-12)
                    idx = tree.query_ball_point(patches_unit[i], r)
                    if not idx:
                        mapping[str(i)] = []
                        continue
                    if len(idx) > int(max_curve_points_per_feature_patch):
                        idx = np.asarray(idx, dtype=int)
                        d = np.linalg.norm(curve_unit[idx] - patches_unit[i], axis=1)
                        order = np.argsort(d)
                        idx = idx[order[:int(max_curve_points_per_feature_patch)]].tolist()
                    mapping[str(i)] = list(map(int, idx))

                with open(os.path.join(out_dir, "sharp_curve_feature_patch_map_spline.json"), "w", encoding="utf-8") as f:
                    json.dump(mapping, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[sharp-curve] patch 映射输出失败（不影响主流程）：{e}")

    print(f"[sharp-curve] 导出完成（spline）：{os.path.basename(input_mesh_path)} | points={curve_pts.shape[0]} -> {out_dir}")





def export_junctions_kway(
    input_mesh_path: str,
    out_dir: str,
    angle_threshold: float,
    edge_split_threshold: float,
    require_step_face_id_diff: bool,
    degree_th: int = 3,
):
    """
    导出 k-way junction（sharp-edge 图中度数 >= degree_th 的顶点）及其 incident regions。

    输出文件（写入 out_dir）：
      - junction_vertex_id.npy        (J,)  原网格顶点 id
      - junction_points_raw.npy       (J,3) 原坐标系
      - junction_points_unit.npy      (J,3) unit box（按 nodes.txt 的 minxx/scale）
      - junction_regions.json         {"<vid>":[r1,r2,...], ...} incident regions（升序）
      - junction_meta.json            元信息（阈值、缩放、数量等）

    incident regions 的计算：
      - 优先使用 sharp edge dict 中的 face1/face2 -> face_region_id 映射（只统计参与 sharp 的面）
      - 若某 junction 统计为空且 face_region_id 可用，则回退到“该顶点所有相邻三角形”的 region 并集。
    """
    import numpy as _np
    from collections import defaultdict

    nodes_path = os.path.join(out_dir, "nodes.txt")
    face_region_path = os.path.join(out_dir, "face_region_id.npy")

    if not os.path.exists(nodes_path):
        print(f"[junction] 跳过：未找到 {nodes_path}")
        return

    nodes = _np.loadtxt(nodes_path)
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        print(f"[junction] 跳过：nodes.txt 形状异常：{nodes.shape}")
        return

    minxx = nodes.min(axis=0)
    maxxx = nodes.max(axis=0)
    scale = float(_np.max(maxxx - minxx))
    if scale <= 0:
        scale = 1.0

    face_region_id = None
    if os.path.exists(face_region_path):
        try:
            face_region_id = _np.load(face_region_path)
        except Exception as e:
            print(f"[junction] 警告：读取 face_region_id.npy 失败，将只输出 junction 点不输出 regions：{e}")
            face_region_id = None

    read_mesh, detect_sharp_edges, _unused_detect_junc, _unused_build_seg = _import_sharp_tools()
    mesh = read_mesh(input_mesh_path, compute_split_normals=False)

    sharp_edges, _lines = detect_sharp_edges(
        mesh,
        angle_threshold=angle_threshold,
        edge_split_threshold=edge_split_threshold,
        require_step_face_id_diff=require_step_face_id_diff
    )
    if (sharp_edges is None) or (len(sharp_edges) == 0):
        # 写空输出，便于下游流水线一致
        _np.save(os.path.join(out_dir, "junction_vertex_id.npy"), _np.zeros((0,), dtype=_np.int32))
        _np.save(os.path.join(out_dir, "junction_points_raw.npy"), _np.zeros((0, 3), dtype=float))
        _np.save(os.path.join(out_dir, "junction_points_unit.npy"), _np.zeros((0, 3), dtype=float))
        with open(os.path.join(out_dir, "junction_regions.json"), "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "junction_meta.json"), "w", encoding="utf-8") as f:
            json.dump({
                "source": "detect_sharp_edges + degree_on_sharp_graph",
                "n_junctions": 0,
                "degree_th": int(degree_th),
                "minxx": minxx.tolist(),
                "scale": scale,
                "note": "no sharp edges detected"
            }, f, ensure_ascii=False, indent=2)
        print(f"[junction] 无尖锐边：{os.path.basename(input_mesh_path)} -> {out_dir}")
        return

    # 1) 构建 sharp-edge 邻接图 +（可选）incident region 统计
    adj = defaultdict(set)            # vid -> set(vid)
    v2regs = defaultdict(set)         # vid -> set(region_id)

    # sharp_edges 可能是 dict 列表（src.precompute / precompute.py），也可能是 ndarray 边列表；
    # 这里统一按 dict 优先处理。
    if isinstance(sharp_edges, (list, tuple)) and sharp_edges and isinstance(sharp_edges[0], dict):
        for e in sharp_edges:
            try:
                a = int(e.get('point1_idx'))
                b = int(e.get('point2_idx'))
            except Exception:
                continue
            adj[a].add(b); adj[b].add(a)

            if face_region_id is not None:
                f1 = e.get('face1', None)
                f2 = e.get('face2', None)
                if f1 is None or f2 is None:
                    continue
                try:
                    r1 = int(face_region_id[int(f1)])
                    r2 = int(face_region_id[int(f2)])
                except Exception:
                    continue
                if (r1 >= 0): v2regs[a].add(r1); v2regs[b].add(r1)
                if (r2 >= 0): v2regs[a].add(r2); v2regs[b].add(r2)
    else:
        # ndarray 边列表（E,2）
        se = _np.asarray(sharp_edges, dtype=_np.int64)
        if se.ndim == 2 and se.shape[1] >= 2:
            for a, b in se[:, :2]:
                a = int(a); b = int(b)
                adj[a].add(b); adj[b].add(a)

    # 2) k-way junction：degree >= degree_th
    junction_vid = [int(v) for v, nbrs in adj.items() if len(nbrs) >= int(degree_th)]
    junction_vid = sorted(set(junction_vid))
    junction_vid_np = _np.asarray(junction_vid, dtype=_np.int32)

    pts_np = _np.asarray(mesh.points)
    jpts = pts_np[junction_vid_np] if junction_vid_np.size else _np.zeros((0, 3), dtype=float)
    junit = (jpts - minxx) / scale if junction_vid_np.size else _np.zeros((0, 3), dtype=float)

    # 3) incident regions（回退：vertex 相邻 faces）
    regions_map = {}
    if face_region_id is not None and junction_vid_np.size:
        # 准备 vertex->faces 邻接（回退用）
        try:
            faces = _np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
            v2faces = defaultdict(list)
            for fi, tri in enumerate(faces):
                for v in tri:
                    v2faces[int(v)].append(fi)
        except Exception:
            faces = None
            v2faces = None

        for vid in junction_vid:
            regs = set(v2regs.get(int(vid), set()))
            if (not regs) and (v2faces is not None):
                # 回退：该 vertex 的所有相邻三角面 region
                for fi in v2faces.get(int(vid), []):
                    try:
                        rr = int(face_region_id[int(fi)])
                    except Exception:
                        continue
                    if rr >= 0:
                        regs.add(rr)
            regions_map[str(int(vid))] = sorted(int(r) for r in regs)

    # 4) 落盘
    _np.save(os.path.join(out_dir, "junction_vertex_id.npy"), junction_vid_np)
    _np.save(os.path.join(out_dir, "junction_points_raw.npy"), jpts)
    _np.save(os.path.join(out_dir, "junction_points_unit.npy"), junit)

    with open(os.path.join(out_dir, "junction_regions.json"), "w", encoding="utf-8") as f:
        json.dump(regions_map, f, ensure_ascii=False, indent=2)

    meta = {
        "source": "detect_sharp_edges + degree_on_sharp_graph",
        "n_junctions": int(junction_vid_np.size),
        "degree_th": int(degree_th),
        "minxx": minxx.tolist(),
        "scale": scale,
        "face_region_id_available": bool(face_region_id is not None),
        "outputs": {
            "vertex_id": "junction_vertex_id.npy",
            "points_raw": "junction_points_raw.npy",
            "points_unit": "junction_points_unit.npy",
            "regions": "junction_regions.json",
        }
    }
    with open(os.path.join(out_dir, "junction_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[junction] 导出完成：{os.path.basename(input_mesh_path)} | junctions={int(junction_vid_np.size)} -> {out_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+')
    ap.add_argument('--out_root', default=os.path.join('output', 'cfpu_input'))

    ap.add_argument('--angle_threshold', type=float, default=40.0)
    ap.add_argument('--r_small_factor', type=float, default=0.5)
    ap.add_argument('--r_large_factor', type=float, default=3.0)
    ap.add_argument('--edge_split_threshold', type=float, default=None)
    ap.add_argument('--require_step_face_id_diff', action='store_true')

    # --- 新增：导出曲线约束点（做法A需要） ---
    ap.add_argument('--no_export_sharp_curve', action='store_true',
                    help='不导出尖锐边曲线约束点（默认导出）')
    # --- 新增：导出 k-way junction（默认导出） ---
    ap.add_argument('--no_export_junctions', action='store_true',
                    help='不导出 k-way junction 信息（默认导出）')
    ap.add_argument('--junction_degree_th', type=int, default=3,
                    help='判定 junction 的 sharp-edge 图度阈值（k-way，默认 3）')
    ap.add_argument('--curve_step', type=float, default=None,
                    help='曲线采样的绝对步长（模型单位）。给了它就忽略 curve_step_factor。')
    ap.add_argument('--curve_step_factor', type=float, default=0.5,
                    help='曲线采样步长系数：step = factor * (segment 相邻顶点平均距离)（默认 0.5）')
    ap.add_argument('--curve_oversample', type=int, default=2000,
                    help='B样条用于近似弧长的过采样点数（默认 2000）')
    ap.add_argument('--max_curve_points_per_feature_patch', type=int, default=200,
                    help='输出 feature patch 映射时，每个 feature patch 最多保留多少个曲线点（默认 200，防止后续 exactinterp 系统过大）')

    # --- 新增：导出椭球(各向异性)patch信息（patch_frames/patch_axes） ---
    ap.add_argument('--export_patch_aniso', action='store_true',
                    help='导出 patch_frames.npy / patch_axes.npy（用于各向异性椭球patch）；默认不导出')
    ap.add_argument('--aniso_ratio_feature', type=float, default=0.25,
                    help='feature patch 的法向半轴比例 c/a（默认 0.25）')
    ap.add_argument('--aniso_ratio_smooth', type=float, default=0.75,
                    help='smooth patch 的法向半轴比例 c/a（默认 0.75）')
    ap.add_argument('--aniso_k_nn', type=int, default=60,
                    help='估计patch局部PCA时kNN兜底点数（默认 60）')
    ap.add_argument('--aniso_min_points', type=int, default=20,
                    help='估计patch局部PCA所需最少点数（默认 20）')
    ap.add_argument('--aniso_ball_factor', type=float, default=1.0,
                    help='估计patch帧时球邻域半径因子（默认 1.0，使用 r*factor）')



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

    # 可选：导出椭球patch帧/半轴（用于各向异性patch）
    export_patch_aniso_info = None
    if getattr(args, "export_patch_aniso", False):
        try:
            from src.aniso_patch_export import export_patch_aniso_info  # type: ignore
        except Exception:
            from aniso_patch_export import export_patch_aniso_info  # type: ignore

    for inp in inputs:
        base = os.path.splitext(os.path.basename(inp))[0]
        out_dir = os.path.join(args.out_root, base + '_cfpu_input')

        # 1) 生成 CFPU 输入（不传任何 pkl）
        build_cfpu_input(
            inp,
            out_dir,
            args.angle_threshold,
            args.r_small_factor,
            args.r_large_factor,
            args.edge_split_threshold,
            args.require_step_face_id_diff
        )


        # 1.5) （可选）导出各向异性椭球 patch 信息：patch_frames.npy / patch_axes.npy
        if args.export_patch_aniso:
            try:
                from src.aniso_patch_export import export_patch_aniso_info
            except Exception:
                from aniso_patch_export import export_patch_aniso_info
            export_patch_aniso_info(
                output_dir=out_dir,
                ratio_feature=args.aniso_ratio_feature,
                ratio_smooth=args.aniso_ratio_smooth,
                k_nn=args.aniso_k_nn,
                min_points=args.aniso_min_points,
                ball_factor=args.aniso_ball_factor,
            )
        # 1.8) 导出 k-way junction（用于 v3 文档的 k-way 锚定）
        if not args.no_export_junctions:
            export_junctions_kway(
                input_mesh_path=inp,
                out_dir=out_dir,
                angle_threshold=args.angle_threshold,
                edge_split_threshold=args.edge_split_threshold,
                require_step_face_id_diff=args.require_step_face_id_diff,
                degree_th=args.junction_degree_th,
            )

        # 2) 导出尖锐边曲线约束点（做法A要用）
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
                max_curve_points_per_feature_patch=args.max_curve_points_per_feature_patch
            )


if __name__ == '__main__':
    main()
