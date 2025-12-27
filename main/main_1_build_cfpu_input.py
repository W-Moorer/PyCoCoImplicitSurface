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

    输出到 out_dir：
      - sharp_curve_points_raw.npy           (Q,3) 原坐标系
      - sharp_curve_points_unit.npy          (Q,3) unit box 坐标（按 nodes.txt 的 minxx/scale）
      - sharp_curve_group_id.npy             (Q,)  segment id
      - sharp_curve_meta.json                元信息
      - sharp_curve_feature_patch_map.json   (可选) feature patch -> curve indices（便于 exactinterp 直接用）
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

    # unit box 变换（和 cfpurecon 内部一致：minxx + scale）
    minxx = nodes.min(axis=0)
    maxxx = nodes.max(axis=0)
    scale = float(np.max(maxxx - minxx))
    if scale <= 0:
        scale = 1.0

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
        # 输出空文件方便流水线
        np.save(os.path.join(out_dir, "sharp_curve_points_raw.npy"), np.empty((0, 3)))
        np.save(os.path.join(out_dir, "sharp_curve_points_unit.npy"), np.empty((0, 3)))
        np.save(os.path.join(out_dir, "sharp_curve_group_id.npy"), np.empty((0,), dtype=np.int32))
        with open(os.path.join(out_dir, "sharp_curve_meta.json"), "w", encoding="utf-8") as f:
            json.dump({
                "n_points": 0,
                "minxx": minxx.tolist(),
                "scale": scale,
                "source": "none",
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

    pts_np = np.asarray(mesh.points)

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

        curve_pts_all.append(C)
        gid_all.append(np.full((C.shape[0],), gid, dtype=np.int32))

    if not curve_pts_all:
        print(f"[sharp-curve] 分段后没有可用曲线：{os.path.basename(input_mesh_path)}")
        np.save(os.path.join(out_dir, "sharp_curve_points_raw.npy"), np.empty((0, 3)))
        np.save(os.path.join(out_dir, "sharp_curve_points_unit.npy"), np.empty((0, 3)))
        np.save(os.path.join(out_dir, "sharp_curve_group_id.npy"), np.empty((0,), dtype=np.int32))
        with open(os.path.join(out_dir, "sharp_curve_meta.json"), "w", encoding="utf-8") as f:
            json.dump({
                "n_points": 0,
                "minxx": minxx.tolist(),
                "scale": scale,
                "source": "segments(empty)",
            }, f, ensure_ascii=False, indent=2)
        return

    curve_pts = np.vstack(curve_pts_all)
    group_id = np.concatenate(gid_all)

    # 去重（避免 exactinterp 线性系统更病态）
    key = np.round(curve_pts, decimals=10)
    _, uniq_idx = np.unique(key, axis=0, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    curve_pts = curve_pts[uniq_idx]
    group_id = group_id[uniq_idx]

    curve_unit = (curve_pts - minxx) / scale

    np.save(os.path.join(out_dir, "sharp_curve_points_raw.npy"), curve_pts)
    np.save(os.path.join(out_dir, "sharp_curve_points_unit.npy"), curve_unit)
    np.save(os.path.join(out_dir, "sharp_curve_group_id.npy"), group_id)

    meta = {
        "source": "detect_sharp_edges + build_sharp_segments + cubic_bspline",
        "n_points": int(curve_pts.shape[0]),
        "n_segments": int(len(segments)),
        "minxx": minxx.tolist(),
        "scale": scale,
        "curve_step": None if curve_step is None else float(curve_step),
        "curve_step_factor": float(curve_step_factor),
        "curve_oversample": int(curve_oversample),
        "note": "unit 坐标按 nodes.txt 的 minxx/scale 归一化，可直接用于后续 exact interpolation 里把曲线点并入 residual 插值集合"
    }
    with open(os.path.join(out_dir, "sharp_curve_meta.json"), "w", encoding="utf-8") as f:
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
                    idx = tree.query_ball_point(patches_unit[i], float(radii_unit[i]))
                    if not idx:
                        continue
                    # 如果太多，截断为最近的 max_curve_points_per_feature_patch 个
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

    print(f"[sharp-curve] 导出完成：{os.path.basename(input_mesh_path)} | points={curve_pts.shape[0]} -> {out_dir}")


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