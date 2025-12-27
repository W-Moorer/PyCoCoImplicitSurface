#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main_4_cfpu_recon_enhance.py  (curve-only)

目的：
- 将原脚本里“边界增强 #2：Sharp-curve 势场侧增强”的 B1 / bump / RBF / 弧点推导等逻辑全部剔除，
  仅保留【方案A：curve_points 注入 CFPU 的 exact-interp residual correction】。

行为：
- 读取 CFPU 输入（nodes/normals/patches/radii/feature_count）
- 读取 sharp_curve_points_raw.npy 作为 curve_points（世界坐标）
- 可选读取 sharp_curve_feature_patch_map.json 作为 curve_patch_map（加速/稳定地把 curve_points 分配到 patch）
- 调用 src.cfpurecon.cfpurecon(..., curve_points=..., exactinterp=1, ...) 进行重建
- 输出：
  - potential.npy
  - grid.npz (X/Y/Z)
  - curve_points_used.npy（实际喂给 cfpurecon 的曲线点）
  - recon_config.txt

可选：
- --also_save_base：再跑一遍不带 curve_points 的 baseline，保存为 potential_base.npy（用于对比）
"""

import sys
import os
import json
import time
import argparse
import numpy as np

# 添加项目根目录到 Python 路径
sys.path.append(os.path.abspath("."))

DEFAULT_USE_GPU = True

# 尝试导入 CuPy 以支持 GPU 加速（cfpurecon 内部也会二次判断）
try:
    import cupy as cp  # noqa: F401
    GPU_AVAILABLE = True
    try:
        cp.ones(10)  # smoke test
        print("CuPy GPU加速可用")
    except Exception as e:
        print(f"CuPy导入成功，但GPU不可用: {e}")
        GPU_AVAILABLE = False
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy导入失败，将仅使用CPU计算")

from src.cfpurecon import cfpurecon
from visualize.visualize_cfpu_input import load_cfpu


def parse_arguments():
    p = argparse.ArgumentParser(
        description="CFPU重建脚本（curve-only：仅 curve_points 注入 exact-interp residual correction）"
    )
    p.add_argument("--model", type=str, default="LinkedGear", help="模型名称")
    p.add_argument("--gridsize", type=int, default=256, help="网格大小")
    p.add_argument("--use_gpu", action="store_true", help="使用GPU加速（默认跟随 DEFAULT_USE_GPU）")
    p.add_argument("--threads", type=int, default=-1, help="线程数量（-1=全部）")
    # 仅保留 curve（为兼容旧命令行，可传 --enhancer curve）
    p.add_argument("--enhancer", type=str, default="curve", choices=["curve"],
                   help="仅支持 curve：curve_points 注入 exact-interp residual correction")


    # 输入/输出目录（默认保持与原脚本一致）
    p.add_argument(
        "--input_dir", type=str, default=None,
        help="CFPU输入目录（默认: output\\cfpu_input\\{model}_surface_cellnormals_cfpu_input）"
    )
    p.add_argument(
        "--output_dir", type=str, default=None,
        help="输出目录（默认: output\\{model}_recon_{CPU/GPU}）"
    )

    # curve_points
    p.add_argument(
        "--curve_points_file", type=str, default=None,
        help="curve_points 文件路径（默认: <input_dir>/sharp_curve_points_raw.npy）"
    )
    p.add_argument(
        "--curve_patch_map_file", type=str, default=None,
        help="curve_patch_map 映射 json（默认: <input_dir>/sharp_curve_feature_patch_map.json，若存在则使用）"
    )

    # 控制 curve_points 规模（总量/每patch）
    p.add_argument(
        "--curve_max_points", type=int, default=0,
        help="curve_points 总数上限（0=不限制；>0 时做均匀下采样）"
    )
    p.add_argument(
        "--curve_max_points_per_patch", type=int, default=200,
        help="每个patch最多分配的 curve_points 数量（默认 200）"
    )

    # 只让 feature patch 吃 curve_points（与 cfpurecon 的实现保持一致）
    p.add_argument(
        "--curve_only_feature_patches", action="store_true",
        help="只对 feature patches 生效（默认 True）"
    )
    p.set_defaults(curve_only_feature_patches=True)

    # 控制 curve_points 影响范围（缩放 feature patch 半径）
    # 各向异性椭球 patch（需要 input_dir 中存在 patch_frames.npy / patch_axes.npy）
    p.add_argument(
        "--anisotropic_patches", action="store_true",
        help="启用各向异性椭球patch（减少跨面污染；需 patch_frames.npy/patch_axes.npy）"
    )
    p.add_argument(
        "--aniso_dir", type=str, default=None,
        help="各向异性patch信息目录（默认等于 input_dir）"
    )
    p.add_argument(
        "--aniso_query_factor", type=float, default=1.2,
        help="KDTree候选球半径倍数（先球后椭球过滤，>1更稳）"
    )
    p.add_argument(
        "--aniso_min_points", type=int, default=20,
        help="每个patch最少节点数；不足会放宽法向轴或回退到球"
    )
    p.add_argument(
        "--aniso_max_iters", type=int, default=6,
        help="放宽迭代次数（每次放宽法向轴）"
    )
    p.add_argument(
        "--aniso_expand", type=float, default=1.35,
        help="每次放宽法向轴的倍率"
    )

    p.add_argument(
        "--feature_scale", type=float, default=1.0,
        help="对 feature patches 半径整体缩放（<1 更局部；默认 1.0）"
    )

    # 对比用：额外跑 baseline
    p.add_argument(
        "--also_save_base", action="store_true",
        help="额外再跑一遍不带curve_points的CFPU，保存为 potential_base.npy 以便对比（耗时增加）"
    )

    return p.parse_args()


def _default_input_dir(model: str) -> str:
    return rf"output\cfpu_input\{model}_surface_cellnormals_cfpu_input"


def _default_output_dir(model: str, use_gpu: bool) -> str:
    mode_str = "GPU" if use_gpu else "CPU"
    return rf"output\{model}_recon_{mode_str}"


def _maybe_load_curve_patch_map(path: str):
    if path is None or (not os.path.exists(path)):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 期望：dict[str(int)] -> list[int]
        if isinstance(data, dict):
            return data
    except Exception as e:
        print(f"[curve] 警告：读取 curve_patch_map 失败，将退回自动分配。原因：{e}")
    return None


def _downsample_curve_points(curve_pts: np.ndarray, max_points: int) -> np.ndarray:
    if max_points is None or int(max_points) <= 0:
        return curve_pts
    max_points = int(max_points)
    if curve_pts.shape[0] <= max_points:
        return curve_pts
    # 均匀下采样（确定性）
    idx = np.linspace(0, curve_pts.shape[0] - 1, max_points).round().astype(int)
    idx = np.unique(idx)
    return curve_pts[idx]


def main():
    args = parse_arguments()

    use_gpu = (args.use_gpu or DEFAULT_USE_GPU) and GPU_AVAILABLE
    if (args.use_gpu or DEFAULT_USE_GPU) and not GPU_AVAILABLE:
        print("警告：请求使用GPU，但CuPy不可用，将使用CPU")
    if use_gpu:
        # 避免 CPU 多线程 + GPU 混用
        if args.threads == -1 or args.threads > 1:
            args.threads = 1

    input_dir = args.input_dir or _default_input_dir(args.model)
    output_dir = args.output_dir or _default_output_dir(args.model, use_gpu)
    os.makedirs(output_dir, exist_ok=True)

    # 读取 CFPU 输入
    print(f"读取CFPU输入: {input_dir}")
    nodes, normals, patches, radii, feature_count = load_cfpu(input_dir)

    print("读取完成！")
    print(f"节点数量: {nodes.shape[0]}")
    print(f"法向量数量: {normals.shape[0]}")
    print(f"补丁数量: {patches.shape[0]}")
    print(f"半径数量: {radii.shape[0] if radii is not None else 0}")
    print(f"特征点数量: {feature_count}")

    if nodes.shape[0] != normals.shape[0]:
        raise ValueError("节点数量与法向量数量不匹配！")
    if patches.ndim != 2 or patches.shape[1] != 3:
        raise ValueError(f"补丁坐标维度错误！ patches.shape={patches.shape}")

    # feature_mask：前 feature_count 个 patch 被认为是 feature patches
    M = patches.shape[0]
    feature_mask = None
    try:
        fc = int(feature_count)
        if fc > 0:
            feature_mask = (np.arange(M) < fc)
    except Exception:
        feature_mask = None

    # curve_points
    curve_points_file = args.curve_points_file or os.path.join(input_dir, "sharp_curve_points_raw.npy")
    if not os.path.exists(curve_points_file):
        raise FileNotFoundError(
            f"未找到 curve_points 文件：{curve_points_file}。\n"
            f"请先运行 main_1 导出 sharp_curve_points_raw.npy，或用 --curve_points_file 指定。"
        )

    curve_pts = np.load(curve_points_file)
    if curve_pts.ndim != 2 or curve_pts.shape[1] != 3:
        raise ValueError(f"curve_points 形状异常：{curve_pts.shape} ({curve_points_file})")

    curve_pts = _downsample_curve_points(curve_pts, args.curve_max_points)
    np.save(os.path.join(output_dir, "curve_points_used.npy"), curve_pts)
    print(f"curve_points: {curve_pts.shape[0]} points (saved: curve_points_used.npy)")

    # curve_patch_map（可选）
    curve_patch_map_file = args.curve_patch_map_file
    if curve_patch_map_file is None:
        default_map = os.path.join(input_dir, "sharp_curve_feature_patch_map.json")
        curve_patch_map_file = default_map if os.path.exists(default_map) else None
    curve_patch_map = _maybe_load_curve_patch_map(curve_patch_map_file)
    if curve_patch_map is not None:
        print(f"curve_patch_map: enabled ({curve_patch_map_file})")
    else:
        print("curve_patch_map: disabled (auto assign by patch spheres)")

    def _run_cfpu(curve_points=None):
        start_time = time.time()
        pot, X, Y, Z = cfpurecon(
            x=nodes,
            nrml=normals,
            y=patches,
            gridsize=args.gridsize,
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
            feature_mask=feature_mask,
            feature_scale=float(args.feature_scale),
            use_gpu=use_gpu,
            anisotropic_patches=bool(args.anisotropic_patches),
            aniso_dir=(args.aniso_dir or input_dir),
            aniso_query_factor=float(args.aniso_query_factor),
            aniso_min_points=int(args.aniso_min_points),
            aniso_max_iters=int(args.aniso_max_iters),
            aniso_expand=float(args.aniso_expand),

            curve_points=curve_points,
            curve_points_in_unit=False,
            curve_patch_map=curve_patch_map,
            curve_max_points_per_patch=int(args.curve_max_points_per_patch),
            curve_only_feature_patches=bool(args.curve_only_feature_patches),
        )
        print(f"\nCFPU重建完成，耗时: {time.time() - start_time:.2f}秒")
        return pot, X, Y, Z

    # 可选 baseline
    if args.also_save_base:
        print("开始CFPU重建（baseline：不带 curve_points）...")
        pot_base, Xb, Yb, Zb = _run_cfpu(curve_points=None)
        np.save(os.path.join(output_dir, "potential_base.npy"), pot_base)
        np.savez(os.path.join(output_dir, "grid_base.npz"), X=Xb, Y=Yb, Z=Zb)
        print("已保存 potential_base.npy / grid_base.npz")

    # curve-only 重建
    print("开始CFPU重建（curve-only：注入 curve_points）...")
    potential, X, Y, Z = _run_cfpu(curve_points=curve_pts)
    np.save(os.path.join(output_dir, "potential.npy"), potential)
    np.savez(os.path.join(output_dir, "grid.npz"), X=X, Y=Y, Z=Z)
    print("已保存 potential.npy / grid.npz")

    # 保存配置
    config_path = os.path.join(output_dir, "recon_config.txt")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("curve-only CFPU recon\n")
        f.write(f"model: {args.model}\n")
        f.write(f"input_dir: {input_dir}\n")
        f.write(f"output_dir: {output_dir}\n")
        f.write(f"gridsize: {args.gridsize}\n")
        f.write(f"use_gpu: {use_gpu}\n")
        f.write(f"threads: {args.threads}\n")
        f.write(f"curve_points_file: {curve_points_file}\n")
        f.write(f"curve_points_used: {curve_pts.shape[0]}\n")
        f.write(f"curve_patch_map_file: {curve_patch_map_file}\n")
        f.write(f"curve_max_points_per_patch: {args.curve_max_points_per_patch}\n")
        f.write(f"curve_only_feature_patches: {args.curve_only_feature_patches}\n")
        f.write(f"feature_scale: {args.feature_scale}\n")

        f.write(f"anisotropic_patches: {args.anisotropic_patches}\n")

        f.write(f"aniso_dir: {args.aniso_dir or input_dir}\n")

        f.write(f"aniso_query_factor: {args.aniso_query_factor}\n")

        f.write(f"aniso_min_points: {args.aniso_min_points}\n")

        f.write(f"aniso_max_iters: {args.aniso_max_iters}\n")

        f.write(f"aniso_expand: {args.aniso_expand}\n")
        f.write(f"also_save_base: {args.also_save_base}\n")
    print(f"已保存配置: {config_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
