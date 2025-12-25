#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CFPU 重建 + B1(bump 修正场) 第一阶段落地版（路线B：分辨率一致化抑制鼓泡/串珠）

路线B核心：
- 你想要的几何尺度 r_raw (由 tol_geom+θ 推导) 可能远小于网格步长 h
- 若 r_raw << h，则“exact interp + 全局RBF + 大epsilon(由3h主导)”会产生串珠/鼓泡
- 解决：强制 r_used >= r_floor_factor * h（让约束尺度与网格解析一致）
  代价：几何误差下界会受 h 限制（暂时为“看起来合理”，后续你要走路线A局部细化才能达标）

输出：
- potential.npy（原 CFPU）
- potential_b1.npy（B1 修正后）
- grid.npz（X,Y,Z）
- recon_config.txt
- b1_meta.json / b1_arc_points.npy
"""

import sys
import os
import json
import numpy as np
import time
import argparse
from scipy.spatial import cKDTree

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath("."))

DEFAULT_USE_GPU = True

# 尝试导入CuPy以支持GPU加速
try:
    import cupy as cp  # noqa: F401
    GPU_AVAILABLE = True
    print("CuPy导入成功，GPU加速可用")
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy导入失败，将仅使用CPU计算")

from src.cfpurecon import cfpurecon
from visualize.visualize_cfpu_input import load_cfpu


# -----------------------------
# 基础工具
# -----------------------------
def _safe_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return v / n


def bump_weight_c1(d: np.ndarray, eps: float) -> np.ndarray:
    """
    C1 bump：B(0)=1, B(eps)=0, 且在 0/eps 处导数为0
    使用 smoothstep：1 - 3t^2 + 2t^3
    """
    eps = max(float(eps), 1e-12)
    t = np.clip(d / eps, 0.0, 1.0)
    return 1.0 - 3.0 * t * t + 2.0 * t * t * t


def trilinear_sample(potential: np.ndarray,
                     X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                     pts: np.ndarray) -> np.ndarray:
    """
    在规则网格 (X,Y,Z + potential[ny,nx,nz]) 上对任意点做三线性插值。
    要求 X,Y,Z 来自 cfpurecon 的 meshgrid(indexing='xy') 输出。
    """
    xs = X[0, :, 0]
    ys = Y[:, 0, 0]
    zs = Z[0, 0, :]

    px = np.clip(pts[:, 0], xs[0], xs[-1])
    py = np.clip(pts[:, 1], ys[0], ys[-1])
    pz = np.clip(pts[:, 2], zs[0], zs[-1])

    ix = np.searchsorted(xs, px) - 1
    iy = np.searchsorted(ys, py) - 1
    iz = np.searchsorted(zs, pz) - 1

    ix = np.clip(ix, 0, len(xs) - 2)
    iy = np.clip(iy, 0, len(ys) - 2)
    iz = np.clip(iz, 0, len(zs) - 2)

    x0 = xs[ix]
    x1 = xs[ix + 1]
    y0 = ys[iy]
    y1 = ys[iy + 1]
    z0 = zs[iz]
    z1 = zs[iz + 1]

    tx = (px - x0) / np.maximum((x1 - x0), 1e-12)
    ty = (py - y0) / np.maximum((y1 - y0), 1e-12)
    tz = (pz - z0) / np.maximum((z1 - z0), 1e-12)

    # potential shape: (ny, nx, nz) with indexing (y,x,z)
    c000 = potential[iy, ix, iz]
    c100 = potential[iy, ix + 1, iz]
    c010 = potential[iy + 1, ix, iz]
    c110 = potential[iy + 1, ix + 1, iz]

    c001 = potential[iy, ix, iz + 1]
    c101 = potential[iy, ix + 1, iz + 1]
    c011 = potential[iy + 1, ix, iz + 1]
    c111 = potential[iy + 1, ix + 1, iz + 1]

    c00 = c000 * (1 - tx) + c100 * tx
    c10 = c010 * (1 - tx) + c110 * tx
    c01 = c001 * (1 - tx) + c101 * tx
    c11 = c011 * (1 - tx) + c111 * tx

    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty

    return c0 * (1 - tz) + c1 * tz


# -----------------------------
# RBF (PHS r^3 + 常数项)
# -----------------------------
def rbf_fit_phs3_centers(centers: np.ndarray, values: np.ndarray, ridge: float = 1e-12):
    """
    PHS r^3 + 常数项（poly=1） 的插值：
        f(x) = sum_j a_j * ||x-c_j||^3 + b
    解:
        [Phi  1][a] = [values]
        [1^T  0][b]   [  0   ]
    """
    centers = np.asarray(centers, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)

    n = centers.shape[0]
    if n == 0:
        raise ValueError("RBF centers is empty")

    diff = centers[:, None, :] - centers[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    Phi = r ** 3
    Phi.flat[:: n + 1] += float(ridge)  # tiny ridge for stability

    P = np.ones((n, 1), dtype=float)
    K = np.block([
        [Phi, P],
        [P.T, np.zeros((1, 1), dtype=float)]
    ])
    rhs = np.concatenate([values, np.zeros(1, dtype=float)], axis=0)

    try:
        sol = np.linalg.solve(K, rhs)
    except np.linalg.LinAlgError:
        sol, *_ = np.linalg.lstsq(K, rhs, rcond=None)

    a = sol[:n]
    b = float(sol[n])
    return a, b


def rbf_eval_phs3(points: np.ndarray, centers: np.ndarray, a: np.ndarray, b: float, chunk: int = 20000):
    """分块评估 f(x)=sum a_j||x-c_j||^3 + b"""
    points = np.asarray(points, dtype=float)
    centers = np.asarray(centers, dtype=float)
    a = np.asarray(a, dtype=float).reshape(-1)

    out = np.empty((points.shape[0],), dtype=float)
    for i in range(0, points.shape[0], chunk):
        q = points[i:i + chunk]
        diff = q[:, None, :] - centers[None, :, :]
        r = np.linalg.norm(diff, axis=2)
        out[i:i + chunk] = (r ** 3) @ a + b
    return out


# -----------------------------
# B1：按 tol_geom 推导 r 与 epsilon，并生成短弧点（路线B：r下限）
# -----------------------------
def derive_r_and_build_arc_points(curve_pts, curve_tan, n1, n2,
                                  tol_geom: float, grid_dx: float,
                                  k_eps: float = 3.0,
                                  arc_points_per_center: int = 3,
                                  max_centers: int = 800,
                                  r_floor_factor: float = 0.5):
    """
    正确逻辑 + 路线B修正：
    - tol_geom: 目标几何误差上界（例如 0.01*Lmin）
    - 每个采样点用 θ 计算允许半径 r_i <= tol_geom / (csc(θ/2)-1)
    - 第一阶段保守取 r_raw = min(r_i)
    - 路线B：r_used = max(r_raw, r_floor_factor * grid_dx)  (抑制亚体素导致的串珠)
    - epsilon = max(k_eps*r_used, 3*grid_dx)
    - 生成弧点：p = c + r_used * dir，dir 在 u1->u2 的球面插值（在法截面内）
    """
    curve_pts = np.asarray(curve_pts, dtype=float)
    curve_tan = np.asarray(curve_tan, dtype=float)
    n1 = np.asarray(n1, dtype=float)
    n2 = np.asarray(n2, dtype=float)

    tol_geom = max(float(tol_geom), 1e-12)
    grid_dx = max(float(grid_dx), 1e-12)
    k_eps = float(k_eps)
    r_floor_factor = float(r_floor_factor)

    u1_valid = []
    u2_valid = []
    idx_valid = []
    r_i_list = []
    theta_list = []

    for i in range(curve_pts.shape[0]):
        t = _safe_normalize(curve_tan[i])
        nn1 = _safe_normalize(n1[i])
        nn2 = _safe_normalize(n2[i])

        u1 = nn1 - float(np.dot(nn1, t)) * t
        u2 = nn2 - float(np.dot(nn2, t)) * t
        nu1 = float(np.linalg.norm(u1))
        nu2 = float(np.linalg.norm(u2))
        if nu1 < 1e-10 or nu2 < 1e-10:
            continue
        u1 /= nu1
        u2 /= nu2

        dot12 = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
        theta = float(np.arccos(dot12))
        if theta < 1e-4:
            continue

        s = float(np.sin(theta / 2.0))
        if s < 1e-8:
            continue

        denom = (1.0 / s) - 1.0  # csc(theta/2)-1
        if denom < 1e-8:
            continue

        r_i = tol_geom / denom
        if not np.isfinite(r_i) or r_i <= 0:
            continue

        u1_valid.append(u1)
        u2_valid.append(u2)
        idx_valid.append(i)
        r_i_list.append(r_i)
        theta_list.append(theta)

    if len(r_i_list) == 0:
        return np.empty((0, 3), dtype=float), None, None, None, None

    r_i_arr = np.asarray(r_i_list, dtype=float)
    theta_arr = np.asarray(theta_list, dtype=float)

    r_raw = float(np.min(r_i_arr))
    r_raw = max(r_raw, 1e-12)

    # -------- 路线B关键：抬升到可解析尺度 --------
    r_floor = max(1e-12, r_floor_factor * grid_dx)
    r_used = max(r_raw, r_floor)

    epsilon = max(k_eps * r_used, 3.0 * grid_dx)

    # 生成弧点（控制数量）
    per_center = max(1, int(arc_points_per_center))
    total = len(idx_valid) * per_center
    stride = int(np.ceil(total / max_centers)) if total > max_centers else 1

    if per_center == 1:
        alphas = [0.5]
    else:
        alphas = [(k + 1) / (per_center + 1) for k in range(per_center)]

    arc_pts = []
    used = 0
    for j in range(0, len(idx_valid), stride):
        i = idx_valid[j]
        u1 = u1_valid[j]
        u2 = u2_valid[j]
        c = curve_pts[i]

        dot12 = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
        theta = float(np.arccos(dot12))
        sin_theta = float(np.sin(theta))
        if abs(sin_theta) < 1e-8:
            continue

        for a in alphas:
            w1 = np.sin((1.0 - a) * theta) / sin_theta
            w2 = np.sin(a * theta) / sin_theta
            d = _safe_normalize(w1 * u1 + w2 * u2)
            arc_pts.append(c + r_used * d)

        used += 1
        if used * per_center >= max_centers:
            break

    arc_pts = np.asarray(arc_pts, dtype=float)
    if arc_pts.shape[0] == 0:
        return arc_pts, r_raw, r_used, epsilon, {"r_stats": None, "theta_stats": None}

    # 去重 + 限量
    q = max(r_used * 0.25, 1e-12)
    key = np.round(arc_pts / q).astype(np.int64)
    _, uniq = np.unique(key, axis=0, return_index=True)
    arc_pts = arc_pts[np.sort(uniq)]

    if arc_pts.shape[0] > max_centers:
        arc_pts = arc_pts[::int(np.ceil(arc_pts.shape[0] / max_centers))]

    r_stats = {
        "min": float(np.min(r_i_arr)),
        "p10": float(np.percentile(r_i_arr, 10)),
        "median": float(np.median(r_i_arr)),
        "p90": float(np.percentile(r_i_arr, 90)),
        "max": float(np.max(r_i_arr)),
        "n": int(r_i_arr.size),
    }
    th_stats = {
        "min": float(np.min(theta_arr)),
        "median": float(np.median(theta_arr)),
        "max": float(np.max(theta_arr)),
    }

    extra = {
        "r_stats": r_stats,
        "theta_stats": th_stats,
        "r_raw": float(r_raw),
        "r_floor": float(r_floor),
        "r_used": float(r_used),
        "r_floor_factor": float(r_floor_factor),
    }
    return arc_pts, r_raw, r_used, epsilon, extra


# -----------------------------
# B1：应用到网格 potential 上
# -----------------------------
def apply_b1_bump(potential, X, Y, Z, input_dir, out_dir,
                  tol_geom_override=None, tol_ratio=0.01,
                  max_centers=800, arc_points_per_center=3,
                  k_eps=3.0, band_chunk=20000,
                  r_floor_factor=0.5):
    """
    返回 corrected_potential, b1_meta
    """
    meta_path = os.path.join(input_dir, "sharp_curve_meta.json")
    curve_pts_path = os.path.join(input_dir, "sharp_curve_points_raw.npy")
    tan_path = os.path.join(input_dir, "sharp_curve_tangents.npy")
    n1_path = os.path.join(input_dir, "sharp_curve_n1.npy")
    n2_path = os.path.join(input_dir, "sharp_curve_n2.npy")

    if not os.path.exists(meta_path):
        print("[B1] 未找到 sharp_curve_meta.json，跳过。")
        return None, None
    if not (os.path.exists(curve_pts_path) and os.path.exists(tan_path) and os.path.exists(n1_path) and os.path.exists(n2_path)):
        print("[B1] 缺少 sharp_curve_* .npy 文件，跳过。")
        return None, None

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    curve_pts = np.load(curve_pts_path)
    curve_tan = np.load(tan_path)
    curve_n1 = np.load(n1_path)
    curve_n2 = np.load(n2_path)

    # --- 读取/推导 tol_geom（几何误差容限） ---
    if tol_geom_override is not None:
        tol_geom = float(tol_geom_override)
    else:
        tol_geom = float(meta.get("tol_geom", meta.get("tol_geom_default", meta.get("tol", 0.0))))
        if tol_geom <= 0.0:
            Lmin = float(meta.get("sharp_Lmin", 0.0))
            if Lmin > 0:
                tol_geom = float(tol_ratio) * Lmin

    if tol_geom <= 0.0:
        print("[B1] tol_geom 无法确定（meta缺失或无 Lmin），跳过。")
        return None, None

    # --- 网格步长 h ---
    xs = X[0, :, 0]
    ys = Y[:, 0, 0]
    zs = Z[0, 0, :]
    if len(xs) < 2 or len(ys) < 2 or len(zs) < 2:
        print("[B1] 网格尺寸异常，跳过。")
        return None, None

    hx = float(abs(xs[1] - xs[0]))
    hy = float(abs(ys[1] - ys[0]))
    hz = float(abs(zs[1] - zs[0]))
    h = max(hx, hy, hz)

    # --- 由 tol_geom + θ 推导 r / epsilon，并生成弧点（路线B：r下限） ---
    arc_pts, r_raw, r_used, epsilon, extra = derive_r_and_build_arc_points(
        curve_pts, curve_tan, curve_n1, curve_n2,
        tol_geom=tol_geom,
        grid_dx=h,
        k_eps=k_eps,
        arc_points_per_center=arc_points_per_center,
        max_centers=max_centers,
        r_floor_factor=r_floor_factor
    )

    if arc_pts.shape[0] == 0 or r_used is None or epsilon is None:
        print("[B1] 弧点生成失败，跳过。")
        return None, None

    print(f"[B1-B] tol_geom={tol_geom:.3e}, r_raw={r_raw:.3e}, r_used={r_used:.3e}, epsilon={epsilon:.3e}, grid_dx={h:.3e}, centers={arc_pts.shape[0]}")
    if extra and extra.get("r_stats") is not None:
        rs = extra["r_stats"]
        ts = extra["theta_stats"]
        print(f"[B1] r_i stats: min={rs['min']:.3e}, p10={rs['p10']:.3e}, med={rs['median']:.3e}, p90={rs['p90']:.3e}, max={rs['max']:.3e}, n={rs['n']}")
        print(f"[B1] theta stats(rad): min={ts['min']:.3f}, med={ts['median']:.3f}, max={ts['max']:.3f}")
        print(f"[B1-B] r_floor = {extra['r_floor']:.3e} (= {extra['r_floor_factor']:.2f} * grid_dx)")

    # --- 拟合 RBF：让弧点处 s0 + bump(d/ε)*Δs = 0 ---
    s0_at_arc = trilinear_sample(potential, X, Y, Z, arc_pts)

    # 弧点到中心线距离≈r_used
    bump_r = float(bump_weight_c1(np.array([r_used], dtype=float), epsilon)[0])
    bump_r = max(bump_r, 1e-6)
    target = -s0_at_arc / bump_r

    print(f"[B1] 拟合 RBF (PHS r^3 + const)：bump(r_used/ε)={bump_r:.6f}")
    a, b = rbf_fit_phs3_centers(arc_pts, target, ridge=1e-12)

    # --- 带内修正 ---
    tree = cKDTree(curve_pts)
    corrected = potential.copy()
    ny, nx, nz = corrected.shape

    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    xy_flat = np.stack([XX.reshape(-1), YY.reshape(-1)], axis=1)

    t0 = time.time()
    updated = 0

    for kz in range(nz):
        z = float(zs[kz])
        pts = np.concatenate([xy_flat, np.full((xy_flat.shape[0], 1), z, dtype=float)], axis=1)

        d, _ = tree.query(pts, k=1)
        msk = d <= epsilon
        if not np.any(msk):
            continue

        idx = np.where(msk)[0]
        qpts = pts[idx]

        w = bump_weight_c1(d[idx], epsilon)
        ds = rbf_eval_phs3(qpts, arc_pts, a, b, chunk=band_chunk)

        delta = w * ds
        sl = corrected[:, :, kz].reshape(-1)
        sl[idx] += delta
        corrected[:, :, kz] = sl.reshape((ny, nx))

        updated += idx.size

    t1 = time.time()
    print(f"[B1] 带内修正完成：更新体素数={updated}, 耗时={t1 - t0:.2f}s")

    b1_meta = {
        "tol_geom": float(tol_geom),
        "tol_ratio_used": float(tol_ratio),
        "r_raw": float(r_raw),
        "r_used": float(r_used),
        "r_floor_factor": float(r_floor_factor),
        "epsilon": float(epsilon),
        "k_eps": float(k_eps),
        "grid_dx": float(h),
        "bump_r": float(bump_r),
        "arc_centers": int(arc_pts.shape[0]),
        "curve_points": int(curve_pts.shape[0]),
        "updated_voxels": int(updated),
        "rbf_kernel": "PHS r^3 + const",
        "arc_points_per_center": int(arc_points_per_center),
        "max_centers": int(max_centers),
        "band_chunk": int(band_chunk),
        "extra": extra
    }

    try:
        np.save(os.path.join(out_dir, "b1_arc_points.npy"), arc_pts)
        with open(os.path.join(out_dir, "b1_meta.json"), "w", encoding="utf-8") as f:
            json.dump(b1_meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return corrected, b1_meta


# -----------------------------
# CLI
# -----------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="CFPU重建脚本（含 B1 bump 修正后处理：路线B 版）")
    parser.add_argument("--model", type=str, default="LinkedGear", help="模型名称")
    parser.add_argument("--gridsize", type=int, default=256, help="网格大小")
    parser.add_argument("--use_gpu", action="store_true", help="使用GPU加速（默认跟随 DEFAULT_USE_GPU）")
    parser.add_argument("--threads", type=int, default=-1, help="线程数量（-1=全部）")

    # B1
    parser.add_argument("--b1", action="store_true", help="显式启用 B1（若检测到 sharp_curve_meta.json 也会自动启用）")
    parser.add_argument("--no_b1", action="store_true", help="强制禁用 B1")
    parser.add_argument("--b1_tol_geom", type=float, default=None, help="几何误差容限 tol_geom（世界单位），覆盖 meta/默认 tol_ratio*Lmin")
    parser.add_argument("--b1_tol_ratio", type=float, default=0.01, help="默认 tol_geom = tol_ratio * sharp_Lmin（默认 0.01）")
    parser.add_argument("--b1_max_centers", type=int, default=800, help="RBF 弧点中心最大数量（默认 800）")
    parser.add_argument("--b1_arc_points_per_center", type=int, default=3, help="每个采样点生成弧点数量（默认 3）")
    parser.add_argument("--b1_k_eps", type=float, default=3.0, help="epsilon = max(k_eps*r, 3*h) 中的 k_eps（默认 3）")
    parser.add_argument("--b1_band_chunk", type=int, default=20000, help="带内评估 Δs 的 chunk 大小（默认 20000）")

    # 路线B关键参数
    parser.add_argument("--b1_r_floor_factor", type=float, default=0.5,
                        help="路线B：r_used = max(r_raw, r_floor_factor*grid_dx)。默认0.5；可试1.0抑制鼓泡更强。")
    return parser.parse_args()


def main():
    args = parse_arguments()

    use_gpu = (args.use_gpu or DEFAULT_USE_GPU) and GPU_AVAILABLE
    if (args.use_gpu or DEFAULT_USE_GPU) and not GPU_AVAILABLE:
        print("警告：请求使用GPU，但CuPy不可用，将使用CPU")
    if use_gpu:
        if args.threads == -1 or args.threads > 1:
            args.threads = 1

    input_dir = rf"output\cfpu_input\{args.model}_surface_cellnormals_cfpu_input"
    mode_str = "GPU" if use_gpu else "CPU"
    output_dir = rf"output\{args.model}_recon_{mode_str}"
    os.makedirs(output_dir, exist_ok=True)

    print("读取CFPU输入文件...")
    nodes, normals, patches, radii, feature_count = load_cfpu(input_dir)

    print("读取完成！")
    print(f"节点数量: {nodes.shape[0]}")
    print(f"法向量数量: {normals.shape[0]}")
    print(f"补丁数量: {patches.shape[0]}")
    print(f"半径数量: {radii.shape[0] if radii is not None else 0}")
    print(f"特征点数量: {feature_count}")

    if nodes.shape[0] != normals.shape[0]:
        print("错误：节点数量与法向量数量不匹配！")
        return 1
    if radii is not None and patches.shape[0] != radii.shape[0]:
        print("错误：补丁数量与半径数量不匹配！")
        return 1

    gridsize = args.gridsize

    print(f"\n开始CFPU重建，网格大小: {gridsize}...")
    print(f"使用GPU加速: {use_gpu}")
    print(f"线程数量: {args.threads if args.threads != -1 else '全部线程'}")
    start_time = time.time()

    try:
        potential, X, Y, Z = cfpurecon(
            x=nodes,
            nrml=normals,
            y=patches,
            gridsize=gridsize,
            reginfo={
                "exactinterp": 1,
                "nrmlreg": 1,
                "nrmllambda": 1e-6,
                "nrmlschur": 1,
                "potreg": 1,
                "potlambda": 1e-4
            },
            n_jobs=args.threads,
            progress=lambda cur, tot: print(f"进度: {cur}/{tot}", end="\r"),
            progress_stage=lambda stage, info: print(f"\n阶段: {stage}"),
            patch_radii_file=os.path.join(input_dir, "radii.txt"),
            patch_radii_in_world_units=True,
            patch_radii_enforce_coverage=False,
            use_gpu=use_gpu
        )

        end_time = time.time()
        print(f"\nCFPU重建完成，耗时: {end_time - start_time:.2f}秒")

        # 保存原始结果
        print("保存重建结果...")
        potential_path = os.path.join(output_dir, "potential.npy")
        np.save(potential_path, potential)
        print(f"潜在场保存到: {potential_path}")

        grid_path = os.path.join(output_dir, "grid.npz")
        np.savez(grid_path, X=X, Y=Y, Z=Z)
        print(f"网格坐标保存到: {grid_path}")

        # B1 后处理（自动启用：存在 sharp_curve_meta.json）
        meta_path = os.path.join(input_dir, "sharp_curve_meta.json")
        enable_b1 = (not args.no_b1) and (args.b1 or os.path.exists(meta_path))
        b1_done = False

        if enable_b1:
            print("\n[B1-B] 开始 bump 修正后处理（路线B：r下限）...")
            corrected, _b1_meta = apply_b1_bump(
                potential=potential, X=X, Y=Y, Z=Z,
                input_dir=input_dir, out_dir=output_dir,
                tol_geom_override=args.b1_tol_geom,
                tol_ratio=args.b1_tol_ratio,
                max_centers=args.b1_max_centers,
                arc_points_per_center=args.b1_arc_points_per_center,
                k_eps=args.b1_k_eps,
                band_chunk=args.b1_band_chunk,
                r_floor_factor=args.b1_r_floor_factor
            )
            if corrected is not None:
                b1_path = os.path.join(output_dir, "potential_b1.npy")
                np.save(b1_path, corrected)
                print(f"[B1-B] 修正后的潜在场保存到: {b1_path}")
                b1_done = True
            else:
                print("[B1-B] bump 修正未执行（缺少数据或失败）。")
        else:
            print("\n[B1] 未启用 bump 修正。")

        # 保存配置
        config_path = os.path.join(output_dir, "recon_config.txt")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(f"模型名称: {args.model}\n")
            f.write(f"节点数量: {nodes.shape[0]}\n")
            f.write(f"补丁数量: {patches.shape[0]}\n")
            f.write(f"网格大小: {gridsize}\n")
            f.write(f"重建耗时: {end_time - start_time:.2f}秒\n")
            f.write(f"潜在场形状: {potential.shape}\n")
            f.write(f"使用GPU加速: {use_gpu}\n")
            f.write(f"线程数量: {args.threads if args.threads != -1 else '全部线程'}\n")
            f.write(f"B1 bump: {'ON' if enable_b1 else 'OFF'}\n")
            f.write(f"B1 done: {b1_done}\n")
            if args.b1_tol_geom is not None:
                f.write(f"B1 tol_geom(override): {args.b1_tol_geom}\n")
            f.write(f"B1 tol_ratio(default): {args.b1_tol_ratio}\n")
            f.write(f"B1 max_centers: {args.b1_max_centers}\n")
            f.write(f"B1 arc_points_per_center: {args.b1_arc_points_per_center}\n")
            f.write(f"B1 k_eps: {args.b1_k_eps}\n")
            f.write(f"B1 r_floor_factor(routeB): {args.b1_r_floor_factor}\n")
        print(f"重建配置保存到: {config_path}")

        print("\n重建完成！")
        return 0

    except Exception as e:
        print(f"\nCFPU重建失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
