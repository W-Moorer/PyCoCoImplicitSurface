#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接用“隐式曲面 phi(x)=0”做误差评估（不把等值面三角化、不用 vtkImplicitPolyDataDistance）

- 输入：CFPU recon 的 potential.npy + grid.npz（X,Y,Z）
- 评估点：默认用 ORIGINAL_MESH 的点；也可改成你自己的采样点 NPY/NPZ
- 输出：
  * 距离/残差统计
  * 可视化：在原始网格上渲染误差热力图
"""

from __future__ import annotations
import os
from typing import Tuple, Optional, Dict

import numpy as np

try:
    import pyvista as pv
    from pyvista import _vtk as vtk
except Exception as e:
    raise RuntimeError("需要安装 pyvista：pip install pyvista") from e


# ==================== 参数（按需改） ====================
RECON_DIR = "./output/Gear_recon_GPU"     # 含 potential.npy, grid.npz
ORIGINAL_MESH_PATH = "./input/complex_geometry/Gear_surface_cellnormals.vtp"        # 原始网格（用其点/法向）
ISO_VALUE = 0.0                                                       # 隐式面：phi=ISO_VALUE

# 若你要用“采样点云”评估，把它设为路径；否则设为 None
# 支持：
#   - .npz: 需要 points(N,3) 和 normals(N,3)(可选)
#   - .npy: 支持 (N,6)->xyz+nxnynz 或 (N,3)->xyz
POINTCLOUD_PATH: Optional[str] = None

# 误差模式：
#   "residual"   -> 只算 |phi|
#   "first_order"-> |phi|/||grad||
#   "project"    -> Newton 投影到 phi=ISO_VALUE，再算 ||p-p_proj||
ERROR_MODE = "residual"          # residual/first_order/project
PROJECT_ITERS = 5                       # Newton 迭代次数（3~8 通常够）
MAX_STEP_FACTOR = 0.5                   # 每次投影最大步长=MAX_STEP_FACTOR*grid_spacing

PCTL = 95.0                             # robust Hausdorff: HDp
CLIM = None                             # 可视化色标范围，例如 (0, 0.1) 或 None
SCREENSHOT_PATH = None                  # e.g. "implicit_error.png" or None
SHOW_ISO_WIREFRAME = False               # 是否显示零等值面线框网格
# =======================================================


def load_cfpu_recon(recon_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    potential_path = os.path.join(recon_dir, "potential.npy")
    grid_path = os.path.join(recon_dir, "grid.npz")
    if not os.path.exists(potential_path) or not os.path.exists(grid_path):
        raise FileNotFoundError(f"重建结果文件不存在于 {recon_dir}")
    potential = np.load(potential_path)
    grid_data = np.load(grid_path)
    X = grid_data["X"]
    Y = grid_data["Y"]
    Z = grid_data["Z"]
    return potential, X, Y, Z


def build_structured_grid(potential: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> pv.StructuredGrid:
    # 与你现有代码一致：用 Fortran 顺序放 scalars（确保和 VTK 点顺序对齐）
    sg = pv.StructuredGrid(X, Y, Z)
    sg["potential"] = potential.ravel(order="F")
    return sg


def estimate_grid_spacing(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> float:
    # cfpurecon 里 meshgrid(xx,yy,zz,indexing='xy')，shape = (len(yy), len(xx), len(zz)) :contentReference[oaicite:2]{index=2}
    # 因此：
    #   x 主要沿 axis=1 变化
    #   y 主要沿 axis=0 变化
    #   z 主要沿 axis=2 变化
    eps = 1e-12
    dx = abs(float(X[0, 1, 0] - X[0, 0, 0])) if X.shape[1] > 1 else np.inf
    dy = abs(float(Y[1, 0, 0] - Y[0, 0, 0])) if Y.shape[0] > 1 else np.inf
    dz = abs(float(Z[0, 0, 1] - Z[0, 0, 0])) if Z.shape[2] > 1 else np.inf
    h = min(v for v in (dx, dy, dz) if np.isfinite(v) and v > eps)
    return float(h)


def add_gradient_field(sg: pv.StructuredGrid, scalar_name="potential", grad_name="grad") -> pv.DataSet:
    """
    用 VTK 的 vtkGradientFilter 计算 ∇phi（点数据），比猜 pyvista API 版本更稳。
    """
    gf = vtk.vtkGradientFilter()
    gf.SetInputData(sg)
    gf.SetInputScalars(vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, scalar_name)
    gf.SetResultArrayName(grad_name)
    gf.Update()
    return pv.wrap(gf.GetOutput())


def probe_phi_grad(sg_with_grad: pv.DataSet, points: np.ndarray,
                   scalar_name="potential", grad_name="grad") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    在 points 上采样 phi 和 grad。VTK 内部是 C++，速度远高于 python for-loop。
    """
    pd = pv.PolyData(points)
    sampled = pd.sample(sg_with_grad)  # VTK ProbeFilter

    # 有些点可能在网格外，VTK 会给 mask
    if "vtkValidPointMask" in sampled.array_names:
        mask = sampled["vtkValidPointMask"].astype(bool)
    else:
        mask = np.ones(points.shape[0], dtype=bool)

    phi = np.asarray(sampled[scalar_name]).reshape(-1)
    grad = np.asarray(sampled[grad_name]).reshape(-1, 3)
    return phi, grad, mask


def summarize(d: np.ndarray, pctl: float = 95.0) -> Dict[str, float]:
    if d.size == 0:
        return {"HD": 0.0, "HDp": 0.0, "ASD": 0.0, "STD": 0.0}
    return {
        "HD": float(np.max(d)),
        "HDp": float(np.percentile(d, pctl)),
        "ASD": float(np.mean(d)),
        "STD": float(np.std(d)),
    }


def compute_error(points: np.ndarray,
                  sg_with_grad: pv.DataSet,
                  iso_value: float,
                  mode: str,
                  grid_h: float,
                  project_iters: int,
                  max_step_factor: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回：
      err: (N,) 误差（residual/first_order/project）
      phi: (N,) 采样到的 phi
      n_imp: (N,3) 隐式法向（由 grad 得到）
    """
    eps = 1e-12
    max_step = max_step_factor * grid_h

    if mode == "residual":
        phi, grad, mask = probe_phi_grad(sg_with_grad, points)
        phi0 = phi - iso_value
        gn = np.linalg.norm(grad, axis=1)
        n_imp = grad / np.maximum(gn[:, None], eps)
        err = np.abs(phi0)
        err[~mask] = np.nan
        return err, phi0, n_imp

    if mode == "first_order":
        phi, grad, mask = probe_phi_grad(sg_with_grad, points)
        phi0 = phi - iso_value
        gn = np.linalg.norm(grad, axis=1)
        n_imp = grad / np.maximum(gn[:, None], eps)
        err = np.abs(phi0) / np.maximum(gn, eps)
        err[~mask] = np.nan
        return err, phi0, n_imp

    if mode == "project":
        p = points.astype(float).copy()
        last_phi0 = None
        last_nimp = None
        for _ in range(int(project_iters)):
            phi, grad, mask = probe_phi_grad(sg_with_grad, p)
            phi0 = phi - iso_value
            g2 = np.sum(grad * grad, axis=1)

            # Newton: p <- p - phi/||g||^2 * g
            step = np.zeros_like(phi0)
            good = (mask) & (g2 > 1e-24)
            step[good] = (phi0[good] / g2[good])

            # 限制单步步长，避免跳飞
            dp = (-step[:, None]) * grad
            dp_norm = np.linalg.norm(dp, axis=1)
            scale = np.ones_like(dp_norm)
            too_big = dp_norm > max_step
            scale[too_big] = max_step / np.maximum(dp_norm[too_big], eps)
            p = p + dp * scale[:, None]

            gn = np.linalg.norm(grad, axis=1)
            last_nimp = grad / np.maximum(gn[:, None], eps)
            last_phi0 = phi0

        dist = np.linalg.norm(p - points, axis=1)
        dist[~mask] = np.nan
        return dist, last_phi0, last_nimp

    raise ValueError(f"Unknown mode: {mode}")


def load_pointcloud(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        data = np.load(path)
        pts = np.asarray(data["points"], dtype=float)
        nrm = np.asarray(data["normals"], dtype=float) if "normals" in data.files else None
        return pts, nrm
    if ext == ".npy":
        arr = np.load(path)
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2 or arr.shape[1] not in (3, 6):
            raise ValueError("npy 点云支持 (N,3) 或 (N,6)[xyz+nxnynz]")
        pts = arr[:, 0:3]
        nrm = arr[:, 3:6] if arr.shape[1] == 6 else None
        return pts, nrm
    raise ValueError("仅支持 .npz 或 .npy 点云")


def plot_interactive(original: pv.PolyData,
                     err: np.ndarray,
                     title: str,
                     metrics: Dict[str, float],
                     clim: Optional[Tuple[float, float]] = None,
                     screenshot: Optional[str] = None,
                     implicit_grid: Optional[pv.StructuredGrid] = None,
                     show_iso_wireframe: bool = True,
                     save_paraview: bool = False):
    p = pv.Plotter(window_size=[1200, 900])
    p.set_background("white")

    def _auto_clim(arr: np.ndarray):
        if clim is not None:
            return tuple(clim)
        a = arr[np.isfinite(arr)]
        if a.size == 0:
            return (0.0, 1.0)
        mn = float(np.min(a))
        mx = float(np.max(a))
        if np.isclose(mn, mx):
            mx = mn + 1e-6
        return (mn, mx)

    original = original.copy()
    original["err"] = err
    c = _auto_clim(err)

    # 绘制原始网格上的误差热力图
    actor_original = p.add_mesh(original, scalars="err", cmap="viridis", clim=c,
               smooth_shading=True, show_edges=False, show_scalar_bar=False)
    
    # 绘制隐式场网格
    iso_surface = None
    if implicit_grid is not None and show_iso_wireframe:
        # 提取等值面并绘制为灰色线框
        iso_surface = implicit_grid.contour(isosurfaces=[ISO_VALUE])
        if iso_surface.n_points > 0:
            p.add_mesh(iso_surface, color='gray', style='wireframe', opacity=0.5, line_width=1, 
                      show_scalar_bar=False, name='iso_surface')

    # 手动创建 scalar bar，并设置 Times 字体
    scalar_bar = p.add_scalar_bar(title='Hausdorff Error',
                                  position_x=0.15, position_y=0.02,  # 位置到底部
                                  width=0.7, height=0.05,  # 水平扁条形
                                  n_labels=5, label_font_size=10,
                                  title_font_size=12)
    
    # 从网格的 mapper 获取 lookup table 并设置给 scalar bar，确保颜色同步
    lookup_table = actor_original.mapper.lookup_table
    scalar_bar.SetLookupTable(lookup_table)
    
    scalar_bar.SetOrientation(0)  # 0 for horizontal, 1 for vertical
    
    # 设置 Times 字体
    def set_times_font(text_prop):
        text_prop.SetFontFamilyToTimes()
    
    set_times_font(scalar_bar.GetTitleTextProperty())
    set_times_font(scalar_bar.GetLabelTextProperty())

    txt = (f"HD = {metrics['HD']:.6g}\n"
           f"HD{PCTL:.0f} = {metrics['HDp']:.6g}\n"
           f"ASD = {metrics['ASD']:.6g}\n"
           f"STD = {metrics['STD']:.6g}")
    p.add_text(txt, position="upper_right", font_size=12, color="black")
    p.add_axes()
    p.view_isometric()
    if screenshot:
        p.screenshot(screenshot)
    
    # 保存为 paraview 可打开的格式
    if save_paraview:
        # 创建输出目录
        output_dir = os.path.join("output", "haussdorff_error")
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存带有误差数据的原始网格
        original.save(os.path.join(output_dir, "gear_enhance_cfpu.vtp"))
        
        # 如果有等值面，也保存下来
        if iso_surface is not None and iso_surface.n_points > 0:
            iso_surface.save(os.path.join(output_dir, "gear_enhance_cfpu_iso.vtp"))
        
        print(f"已保存结果到 {output_dir}/gear_enhance_cfpu.vtp")
    
    p.show()


def main():
    print("=" * 80)
    print("隐式曲面误差评估（直接用 phi(x)=0，而非等值面三角网格距离）")
    print("=" * 80)

    print("\n【加载 CFPU 重建结果】")
    potential, X, Y, Z = load_cfpu_recon(RECON_DIR)
    sg = build_structured_grid(potential, X, Y, Z)  # 与你原脚本一致 :contentReference[oaicite:3]{index=3}
    grid_h = estimate_grid_spacing(X, Y, Z)
    print(f"  potential shape: {potential.shape}")
    print(f"  grid spacing ~ {grid_h:g}")

    print("\n【计算梯度场 ∇phi】")
    sg_g = add_gradient_field(sg, "potential", "grad")

    # 评估点来源
    if POINTCLOUD_PATH is not None:
        print("\n【加载点云评估点】")
        pts, nrm = load_pointcloud(POINTCLOUD_PATH)
        original = pv.PolyData(pts)
        if nrm is not None:
            original["Normals"] = nrm
        print(f"  points: {pts.shape[0]}")
    else:
        print("\n【加载原始网格评估点】")
        original = pv.read(ORIGINAL_MESH_PATH)
        if not isinstance(original, pv.PolyData):
            original = original.extract_surface()
        original = original.triangulate()
        if "Normals" not in original.point_data:
            original = original.compute_normals(point_normals=True, cell_normals=False,
                                                auto_orient_normals=True, consistent_normals=True)
        pts = np.asarray(original.points)
        print(f"  mesh points: {original.n_points}")

    print("\n【误差计算】")
    err, phi0, n_imp = compute_error(
        points=pts,
        sg_with_grad=sg_g,
        iso_value=ISO_VALUE,
        mode=ERROR_MODE,
        grid_h=grid_h,
        project_iters=PROJECT_ITERS,
        max_step_factor=MAX_STEP_FACTOR,
    )

    # 统计（忽略 NaN）
    valid = np.isfinite(err)
    d = err[valid]
    metrics = summarize(d, pctl=PCTL)

    print("\n【评估结果】")
    print(f"  ERROR_MODE: {ERROR_MODE}")
    print(f"  HD  : {metrics['HD']:.6g}")
    print(f"  HD{PCTL:.0f}: {metrics['HDp']:.6g}")
    print(f"  ASD : {metrics['ASD']:.6g}")
    print(f"  STD : {metrics['STD']:.6g}")

    # 额外：法向误差（如果有原始法向）
    if "Normals" in original.point_data:
        n_gt = np.asarray(original.point_data["Normals"], dtype=float)
        n_gt_norm = np.linalg.norm(n_gt, axis=1)
        n_gt = n_gt / np.maximum(n_gt_norm[:, None], 1e-12)

        cosv = np.sum(n_gt * n_imp, axis=1)
        cosv = np.clip(np.abs(cosv), -1.0, 1.0)  # 取 abs：不敏感于翻向
        ang = np.degrees(np.arccos(cosv))
        ang = ang[np.isfinite(ang)]
        if ang.size:
            print("\n【法向一致性】(度)")
            print(f"  mean: {float(np.mean(ang)):.4g}")
            print(f"  p95 : {float(np.percentile(ang, 95)):.4g}")
            print(f"  max : {float(np.max(ang)):.4g}")

    print("\n【可视化】")
    title = f"Implicit Surface Error (mode={ERROR_MODE})"
    plot_interactive(original, err, title=title, metrics=metrics, clim=CLIM, 
                     screenshot=SCREENSHOT_PATH, implicit_grid=sg, 
                     show_iso_wireframe=SHOW_ISO_WIREFRAME, save_paraview=True)


if __name__ == "__main__":
    main()
