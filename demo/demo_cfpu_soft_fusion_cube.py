#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CFPU + soft fusion demo:
- Build two perpendicular cube faces (x=0, y=0) with synthetic nodes+normals.
- Reconstruct each face with curl-free RBF (CFPU-style) on a single patch (weight=1).
- Fuse with softmax to produce a globally smooth (C∞) field with a rounded "edge".
- Validate edge rounding theory: on x=y line, root t ≈ tau*ln(2).
- Save:
  - slice_fused_z0p5.png
  - fused_isosurface.ply
  - demo_summary.json
"""

import os
import sys
import time
import json
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from skimage.measure import marching_cubes

# 添加项目根目录到sys.path，确保可以导入src模块
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ---------- helpers: data generation ----------
def sample_face_x0(res=13):
    """Face x=0, y,z in [0,1], outward normal (-1,0,0)."""
    y = np.linspace(0.0, 1.0, res)
    z = np.linspace(0.0, 1.0, res)
    yy, zz = np.meshgrid(y, z, indexing="xy")
    pts = np.stack([np.zeros_like(yy), yy, zz], axis=-1).reshape(-1, 3)
    nrml = np.tile(np.array([-1.0, 0.0, 0.0]), (pts.shape[0], 1))
    return pts, nrml


def sample_face_y0(res=13):
    """Face y=0, x,z in [0,1], outward normal (0,-1,0)."""
    x = np.linspace(0.0, 1.0, res)
    z = np.linspace(0.0, 1.0, res)
    xx, zz = np.meshgrid(x, z, indexing="xy")
    pts = np.stack([xx, np.zeros_like(xx), zz], axis=-1).reshape(-1, 3)
    nrml = np.tile(np.array([0.0, -1.0, 0.0]), (pts.shape[0], 1))
    return pts, nrml


# ---------- CFPU single-patch solver (same math as your cfpurecon.py) ----------
def solve_cfpu_single_patch(nodes, normals, order=1, nrmllambda=1e-6, cfpurecon_module=None):
    """
    Solve curl-free RBF system for the normal field on ONE patch,
    then compute potential and apply residual correction to enforce s(nodes)=0.

    This matches the structure used in your cfpurecon.py:
      - curlfree_poly(nodes, order) gives CFP and P
      - build matrix A with dphi blocks, add normal regularization, add polynomial constraints
      - solve for coeffs (vector field) + coeffsp (poly)
      - compute potential on nodes
      - exact interpolation residual correction (nodes-only): solve A1 * corr = [pot_nodes; 0]
    """
    if cfpurecon_module is None:
        raise RuntimeError("Need cfpurecon module. Make sure it is importable and pass it in.")

    nodes = np.asarray(nodes, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    n = nodes.shape[0]
    if n == 0:
        raise ValueError("No nodes provided.")

    CFP, P = cfpurecon_module.curlfree_poly(nodes, order)
    CFPt = CFP.T
    L = CFP.shape[1]

    xx, yy, zz = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    dx = xx.reshape(-1, 1) - xx.reshape(1, -1)
    dy = yy.reshape(-1, 1) - yy.reshape(1, -1)
    dz = zz.reshape(-1, 1) - zz.reshape(1, -1)
    r = np.sqrt(dx * dx + dy * dy + dz * dz)

    # PHS kernels as in your code
    if order == 1:
        eta = -r
        zeta = -np.divide(1.0, r, where=(r != 0))
    elif order == 2:
        eta = r ** 3
        zeta = 3.0 * r
    else:
        raise ValueError("Only order=1 or order=2 supported in this demo.")

    np.fill_diagonal(zeta, 0.0)

    dphi_xx = zeta * dx * dx + eta
    dphi_yy = zeta * dy * dy + eta
    dphi_zz = zeta * dz * dz + eta
    dphi_xy = zeta * dx * dy
    dphi_xz = zeta * dx * dz
    dphi_yz = zeta * dy * dz

    A = np.zeros((3 * n + L, 3 * n + L), dtype=np.float64)
    b = np.zeros((3 * n + L,), dtype=np.float64)

    b[0 : 3 * n : 3] = normals[:, 0]
    b[1 : 3 * n : 3] = normals[:, 1]
    b[2 : 3 * n : 3] = normals[:, 2]

    A[0 : 3 * n : 3, 0 : 3 * n : 3] = dphi_xx
    A[0 : 3 * n : 3, 1 : 3 * n : 3] = dphi_xy
    A[0 : 3 * n : 3, 2 : 3 * n : 3] = dphi_xz
    A[1 : 3 * n : 3, 0 : 3 * n : 3] = dphi_xy
    A[1 : 3 * n : 3, 1 : 3 * n : 3] = dphi_yy
    A[1 : 3 * n : 3, 2 : 3 * n : 3] = dphi_yz
    A[2 : 3 * n : 3, 0 : 3 * n : 3] = dphi_xz
    A[2 : 3 * n : 3, 1 : 3 * n : 3] = dphi_yz
    A[2 : 3 * n : 3, 2 : 3 * n : 3] = dphi_zz

    # normal regularization (same structure)
    A[0 : 3 * n, 0 : 3 * n] += (3.0 * n * float(nrmllambda)) * np.eye(3 * n)

    # polynomial constraints
    A[0 : 3 * n, 3 * n :] = CFP
    A[3 * n :, 0 : 3 * n] = CFPt

    coeffs_all = solve(A, b, assume_a="sym", check_finite=False)
    coeffs = coeffs_all[: 3 * n]
    coeffsp = coeffs_all[3 * n :]

    coeffsx = coeffs[0 : 3 * n : 3]
    coeffsy = coeffs[1 : 3 * n : 3]
    coeffsz = coeffs[2 : 3 * n : 3]

    # potential on nodes
    temp_pot_nodes = (
        np.sum(
            eta
            * (
                dx * coeffsx.reshape(1, -1)
                + dy * coeffsy.reshape(1, -1)
                + dz * coeffsz.reshape(1, -1)
            ),
            axis=1,
        )
        + P @ coeffsp
    )

    # exact interpolation residual correction: enforce s(nodes)=0
    phi2 = (-r if order == 1 else r ** 3)
    A1 = np.ones((n + 1, n + 1), dtype=np.float64)
    A1[:n, :n] = phi2
    A1[-1, -1] = 0.0
    b1 = np.concatenate([temp_pot_nodes, np.array([0.0])], axis=0)

    corr = solve(A1, b1, assume_a="sym", check_finite=False)
    cc = corr[:n]
    c0 = corr[-1]

    return {
        "nodes": nodes,
        "order": order,
        "coeffsx": coeffsx,
        "coeffsy": coeffsy,
        "coeffsz": coeffsz,
        "coeffsp": coeffsp,
        "cc": cc,
        "c0": c0,
    }


def eval_cfpu(model, q_pts, chunk=40000):
    """Evaluate the corrected potential s(q) = temp_pot(q) - residual(q) on query points."""
    nodes = model["nodes"].astype(np.float32)
    coeffsx = model["coeffsx"].astype(np.float32)
    coeffsy = model["coeffsy"].astype(np.float32)
    coeffsz = model["coeffsz"].astype(np.float32)
    coeffsp = model["coeffsp"].astype(np.float32)
    cc = model["cc"].astype(np.float32)
    c0 = np.float32(model["c0"])
    order = int(model["order"])

    out = np.empty((q_pts.shape[0],), dtype=np.float32)

    for s in range(0, q_pts.shape[0], chunk):
        e = min(q_pts.shape[0], s + chunk)
        q = q_pts[s:e].astype(np.float32)

        d = q[:, None, :] - nodes[None, :, :]  # (m,n,3)
        r = np.sqrt(np.sum(d * d, axis=2))      # (m,n)

        if order == 1:
            eta = -r
            phi = -r
        elif order == 2:
            eta = r ** 3
            phi = r ** 3
        else:
            raise ValueError("Only order=1 or 2 supported.")

        dot = d[:, :, 0] * coeffsx[None, :] + d[:, :, 1] * coeffsy[None, :] + d[:, :, 2] * coeffsz[None, :]
        temp = np.sum(eta * dot, axis=1) + (q @ coeffsp.reshape(-1, 1)).reshape(-1)
        corr = phi @ cc + c0
        out[s:e] = temp - corr

    return out


# ---------- soft fusion ----------
def softmax2(a, b, tau):
    """Smooth max: max(a,b) approximated; C∞ for tau>0."""
    m = np.maximum(a, b)
    return m + tau * np.log(np.exp((a - m) / tau) + np.exp((b - m) / tau))


# ---------- export ----------
def save_ply(path, verts_xyz, faces_tri):
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {verts_xyz.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {faces_tri.shape[0]}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for v in verts_xyz:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for tri in faces_tri:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")


def main():
    ap = argparse.ArgumentParser("CFPU two-plane demo + soft fusion at a cube edge")
    ap.add_argument("--face_res", type=int, default=13, help="grid resolution on each face (res x res)")
    ap.add_argument("--order", type=int, default=1, choices=[1, 2], help="PHS order: 1->phi=r, 2->phi=r^3")
    ap.add_argument("--nrmllambda", type=float, default=1e-6, help="normal regularization lambda")
    ap.add_argument("--gridN", type=int, default=256, help="evaluation grid resolution in each dimension")
    ap.add_argument("--domain_min", type=float, default=-0.2)
    ap.add_argument("--domain_max", type=float, default=1.2)
    ap.add_argument("--tau_factor", type=float, default=0.75, help="tau = tau_factor * voxel_h")
    ap.add_argument("--out_dir", type=str, default=r"output\cfpu_soft_fusion_demo_out")
    ap.add_argument("--cfpurecon_path", type=str, default=None,
                    help="Path to cfpurecon.py (optional). If not set, tries importing 'cfpurecon' from sys.path.")
    args = ap.parse_args()

    # 导入cfpurecon模块
    # 优先使用用户指定的路径，否则从src目录导入
    cfpurecon_module = None
    if args.cfpurecon_path is not None:
        # 用户指定了路径
        p = os.path.abspath(args.cfpurecon_path)
        sys.path.insert(0, os.path.dirname(p))
        try:
            import cfpurecon as cfpurecon_module
        except Exception as e:
            raise RuntimeError(f"Cannot import cfpurecon from {args.cfpurecon_path}: {e}") from e
    else:
        # 默认从src目录导入
        try:
            from src import cfpurecon as cfpurecon_module
        except Exception as e:
            raise RuntimeError(
                "Cannot import cfpurecon from src directory. "
                "Make sure the demo is run from the project root, "
                "or use --cfpurecon_path to specify the path to cfpurecon.py"
            ) from e

    out_dir = os.path.abspath(args.out_dir)
    # 如果out_dir是相对路径，将其转换为基于项目根目录的绝对路径
    if not os.path.isabs(args.out_dir):
        out_dir = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    t_start = time.time()

    # 1) synthetic data: two faces
    ptsA, nA = sample_face_x0(res=args.face_res)
    ptsB, nB = sample_face_y0(res=args.face_res)

    # 2) CFPU per face
    t0 = time.time()
    modelA = solve_cfpu_single_patch(ptsA, nA, order=args.order, nrmllambda=args.nrmllambda, cfpurecon_module=cfpurecon_module)
    modelB = solve_cfpu_single_patch(ptsB, nB, order=args.order, nrmllambda=args.nrmllambda, cfpurecon_module=cfpurecon_module)
    solve_time = time.time() - t0

    # 3) high-res grid eval
    N = int(args.gridN)
    xmin, xmax = float(args.domain_min), float(args.domain_max)
    xs = np.linspace(xmin, xmax, N, dtype=np.float32)
    ys = np.linspace(xmin, xmax, N, dtype=np.float32)
    zs = np.linspace(xmin, xmax, N, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="xy")
    q = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    voxel_h = float(xs[1] - xs[0])
    tau = float(args.tau_factor) * voxel_h

    t0 = time.time()
    potA = eval_cfpu(modelA, q, chunk=40000)
    potB = eval_cfpu(modelB, q, chunk=40000)
    eval_time = time.time() - t0

    fused = softmax2(potA, potB, tau)
    fused3 = fused.reshape((N, N, N))  # (y,x,z) logically, but for plotting we use slices

    # 4) numeric checks (proof)
    rng = np.random.default_rng(0)
    M = 8000
    qx = rng.uniform(xmin, xmax, M).astype(np.float32)
    qy = rng.uniform(0.0, 1.0, M).astype(np.float32)
    qz = rng.uniform(0.0, 1.0, M).astype(np.float32)
    qtest = np.stack([qx, qy, qz], axis=1)

    vA = eval_cfpu(modelA, qtest, chunk=8000)
    vB = eval_cfpu(modelB, qtest, chunk=8000)
    idealA = -qtest[:, 0]  # plane x=0 with outward normal (-1,0,0)
    idealB = -qtest[:, 1]  # plane y=0 with outward normal (0,-1,0)
    rmsA = float(np.sqrt(np.mean((vA - idealA) ** 2)))
    rmsB = float(np.sqrt(np.mean((vB - idealB) ** 2)))

    # edge rounding check on diagonal x=y=t at z=mid slice
    mid = N // 2
    sl = fused3[:, :, mid]  # y,x slice at z=mid
    diag = sl[np.arange(N), np.arange(N)]
    sg = np.sign(diag)
    idx = np.where(sg[:-1] * sg[1:] <= 0)[0]
    if idx.size > 0:
        i = int(idx[0])
        x0, x1 = float(xs[i]), float(xs[i + 1])
        y0, y1 = float(diag[i]), float(diag[i + 1])
        t_root = x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0)
    else:
        t_root = float("nan")
    t_pred = float(tau * math.log(2.0))

    # 5) save slice plot z≈0.5
    z_target = 0.5
    iz = int(np.argmin(np.abs(zs - z_target)))
    slice_f = fused3[:, :, iz]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(
        slice_f,
        origin="lower",
        extent=[xmin, xmax, xmin, xmax],
        aspect="equal",
    )
    ax.set_title(f"CFPU two-face demo + soft fusion (z≈{zs[iz]:.3f}, grid {N}^3, tau={tau:.6f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.contour(xs, ys, slice_f, levels=[0.0])
    ax.axvline(0.0)  # ideal plane x=0
    ax.axhline(0.0)  # ideal plane y=0
    plt.colorbar(im, ax=ax)

    png_path = os.path.join(out_dir, "slice_fused_z0p5.png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 6) extract 0-isosurface mesh from fused field
    # marching_cubes expects volume indexing (z,y,x) for spacing; our fused3 is (y,x,z)
    vol = np.transpose(fused3, (2, 0, 1))  # (z,y,x)
    verts, faces, norms, vals = marching_cubes(vol, level=0.0, spacing=(voxel_h, voxel_h, voxel_h))

    # verts are in (z,y,x); convert to world (x,y,z)
    vz, vy, vx = verts[:, 0], verts[:, 1], verts[:, 2]
    vxw = xmin + vx
    vyw = xmin + vy
    vzw = xmin + vz
    verts_world = np.stack([vxw, vyw, vzw], axis=1)

    ply_path = os.path.join(out_dir, "fused_isosurface.ply")
    save_ply(ply_path, verts_world, faces)

    total_time = time.time() - t_start

    summary = {
        "nodes_per_face": int(ptsA.shape[0]),
        "face_res": int(args.face_res),
        "order": int(args.order),
        "nrmllambda": float(args.nrmllambda),
        "grid_N": int(N),
        "domain": [xmin, xmax],
        "voxel_h": float(voxel_h),
        "tau": float(tau),
        "tau_factor": float(args.tau_factor),
        "solve_time_s": float(solve_time),
        "eval_time_s": float(eval_time),
        "total_time_s": float(total_time),
        "rms_faceA_vs_-x": float(rmsA),
        "rms_faceB_vs_-y": float(rmsB),
        "diag_root_t_measured": float(t_root),
        "diag_root_t_theory_tau_ln2": float(t_pred),
        "slice_png": png_path,
        "mesh_ply": ply_path,
    }
    summary_path = os.path.join(out_dir, "demo_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nSaved:")
    print(" -", png_path)
    print(" -", ply_path)
    print(" -", summary_path)


if __name__ == "__main__":
    main()
