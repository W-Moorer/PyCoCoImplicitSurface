#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算模型可视化 - 单视图（可选对比/显示B1弧点）

适配输出：
- potential.npy（原CFPU）
- potential_b1.npy（B1 bump修正后）
- grid.npz（X/Y/Z）
- 可选：b1_arc_points.npy / b1_meta.json

默认行为：
- 若存在 potential_b1.npy，优先可视化它；否则可视化 potential.npy
"""

import numpy as np
import pyvista as pv
from pathlib import Path
import argparse
import json


def load_grid(recon_dir: Path):
    grid_path = recon_dir / "grid.npz"
    if not grid_path.exists():
        raise FileNotFoundError(f"grid.npz 不存在：{grid_path}")
    grid_data = np.load(grid_path)
    X = grid_data["X"]
    Y = grid_data["Y"]
    Z = grid_data["Z"]
    return X, Y, Z


def load_potential(recon_dir: Path, which: str):
    """
    which: 'potential' | 'potential_b1'
    """
    fname = "potential.npy" if which == "potential" else "potential_b1.npy"
    fpath = recon_dir / fname
    if not fpath.exists():
        return None, None
    return np.load(fpath), fpath


def make_surface(X, Y, Z, potential, isovalue: float):
    """
    将 (ny,nx,nz) 的 potential 挂到 StructuredGrid 上，并做等值面提取
    注意 ravel(order='F') 以匹配 StructuredGrid 的点序。
    """
    sg = pv.StructuredGrid(X, Y, Z)
    sg["potential"] = potential.ravel(order="F")
    surf = sg.contour(isosurfaces=[isovalue])
    return surf


def try_load_b1_meta(recon_dir: Path):
    meta_path = recon_dir / "b1_meta.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def try_load_b1_arc_points(recon_dir: Path):
    arc_path = recon_dir / "b1_arc_points.npy"
    if not arc_path.exists():
        return None
    try:
        pts = np.load(arc_path)
        if pts.ndim == 2 and pts.shape[1] == 3 and pts.shape[0] > 0:
            return pts
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--recon_dir", type=str,
        default="output/LinkedGear_recon_GPU",
        help="CFPU重建结果输出目录"
    )
    ap.add_argument(
        "--window_size", type=int, nargs=2, default=[1600, 1200],
        help="可视化窗口大小 (width height)"
    )
    ap.add_argument(
        "--isosurface", type=float, default=0.0,
        help="等值面值"
    )
    ap.add_argument(
        "--opacity", type=float, default=1.0,
        help="重建曲面不透明度"
    )

    # --- 新增：选择显示哪个场 ---
    ap.add_argument(
        "--field", type=str, default="auto",
        choices=["auto", "potential", "potential_b1"],
        help="选择显示的潜在场：auto优先 potential_b1；否则选定文件"
    )
    ap.add_argument(
        "--compare", action="store_true",
        help="叠加显示 potential 与 potential_b1（若两者都存在）用于对比"
    )

    # --- 新增：显示B1弧点 ---
    ap.add_argument(
        "--show_b1_arc", action="store_true",
        help="显示 b1_arc_points.npy（若存在）"
    )
    ap.add_argument(
        "--arc_point_size", type=float, default=6.0,
        help="B1弧点显示大小（Point size）"
    )
    args = ap.parse_args()

    # 设置工作目录 - 脚本在main文件夹，需要上一级目录作为根目录
    root = Path(__file__).resolve().parents[1]
    recon_dir = root / args.recon_dir

    if not recon_dir.exists():
        print(f"错误：目录不存在 {recon_dir}")
        return 1

    # grid
    try:
        X, Y, Z = load_grid(recon_dir)
    except Exception as e:
        print(f"错误：读取网格失败：{e}")
        return 1

    # potentials
    pot0, pot0_path = load_potential(recon_dir, "potential")
    pot1, pot1_path = load_potential(recon_dir, "potential_b1")

    if pot0 is None and pot1 is None:
        print(f"错误：未找到 potential.npy 或 potential_b1.npy 于 {recon_dir}")
        return 1

    # 打印基本信息
    print("读取完成！")
    print(f"  recon_dir: {recon_dir}")
    print(f"  grid shape: {X.shape}")
    if pot0 is not None:
        print(f"  potential: {pot0.shape}  ({pot0_path})")
    if pot1 is not None:
        print(f"  potential_b1: {pot1.shape}  ({pot1_path})")

    # 选择显示策略
    show_compare = args.compare and (pot0 is not None) and (pot1 is not None)

    # auto选择逻辑
    chosen = args.field
    if chosen == "auto":
        chosen = "potential_b1" if pot1 is not None else "potential"

    # 生成等值面
    print("创建可视化...")
    surfaces = []

    if show_compare:
        # 对比：两者都显示
        surf0 = make_surface(X, Y, Z, pot0, args.isosurface)
        surf1 = make_surface(X, Y, Z, pot1, args.isosurface)
        surfaces.append(("potential", surf0))
        surfaces.append(("potential_b1", surf1))
    else:
        # 单一显示：按 chosen
        if chosen == "potential_b1" and pot1 is not None:
            surf = make_surface(X, Y, Z, pot1, args.isosurface)
            surfaces.append(("potential_b1", surf))
        else:
            surf = make_surface(X, Y, Z, pot0, args.isosurface)
            surfaces.append(("potential", surf))

    # Plotter
    plotter = pv.Plotter(window_size=args.window_size)

    # 添加曲面（颜色区分对比）
    for name, surf in surfaces:
        if name == "potential_b1":
            color = "orange"
            label = "B1 (potential_b1)"
        else:
            color = "lightblue"
            label = "Base (potential)"

        plotter.add_mesh(
            surf,
            color=color,
            specular=0.3,
            smooth_shading=True,
            opacity=args.opacity,
            label=label
        )

    # 可选：显示B1弧点
    if args.show_b1_arc:
        arc_pts = try_load_b1_arc_points(recon_dir)
        if arc_pts is not None:
            arc_cloud = pv.PolyData(arc_pts)
            plotter.add_mesh(
                arc_cloud,
                render_points_as_spheres=True,
                point_size=float(args.arc_point_size),
                color="red"
            )
            print(f"  已显示 b1_arc_points.npy 点数：{arc_pts.shape[0]}")
        else:
            print("  未找到或无法读取 b1_arc_points.npy，跳过显示弧点。")

    # 标题/说明
    title = "CFPU重建结果"
    if show_compare:
        title += "（对比：potential vs potential_b1）"
    else:
        title += f"（显示：{chosen}）"

    plotter.add_title(title)
    plotter.add_axes()
    plotter.add_text("左键拖动旋转，滚轮缩放，右键平移", position="lower_left",
                     font_size=10, color="black")

    # 显示 B1 meta（如果有）
    meta = try_load_b1_meta(recon_dir)
    if meta is not None:
        # 简短显示关键参数
        lines = []
        for k in ["tol_geom", "r", "epsilon", "grid_dx", "bump_r", "arc_centers", "updated_voxels"]:
            if k in meta:
                lines.append(f"{k}: {meta[k]}")
        if lines:
            plotter.add_text("\n".join(lines), position="upper_left", font_size=10, color="black")

    # 初始视角
    plotter.view_isometric()

    # 图例（对比时更有用）
    if show_compare:
        plotter.add_legend(bcolor="white")

    plotter.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
