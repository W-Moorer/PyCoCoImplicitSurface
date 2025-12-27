
import os
import json
import numpy as np
from scipy.spatial import cKDTree

def _normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n

def _safe_orthogonal(n):
    # pick an axis not parallel to n
    n = _normalize(n)
    if abs(n[0]) < 0.9:
        a = np.array([1.0, 0.0, 0.0])
    else:
        a = np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, a)
    if np.linalg.norm(t1) < 1e-12:
        a = np.array([0.0, 0.0, 1.0])
        t1 = np.cross(n, a)
    t1 = _normalize(t1)
    t2 = _normalize(np.cross(n, t1))
    t1 = _normalize(np.cross(t2, n))
    return t1, t2

def export_patch_aniso_info(
    output_dir: str,
    ratio_feature: float = 0.25,
    ratio_smooth: float = 0.75,
    k_nn: int = 60,
    min_points: int = 20,
    ball_factor: float = 1.0,
    max_iters: int = 6
):
    """
    Read CFPU input (nodes/normals/patches/radii/feature_count) from output_dir
    and export anisotropic ellipsoid patch info:

      patch_frames.npy: (M,3,3) columns = [t1,t2,n]
      patch_axes.npy:   (M,3)   = (a,b,c) in world units, where a=b=r_k, c=r_k*ratio

    Notes:
    - We first use a ball query (r_k*ball_factor) to get candidates, then PCA on the tangent plane.
    - If too few points, fallback to kNN.
    - If PCA is degenerate, fallback to an arbitrary orthogonal basis.
    - ratio_feature applies to first feature_count patches, ratio_smooth to the rest.
    """
    output_dir = os.path.abspath(output_dir)
    nodes_path   = os.path.join(output_dir, "nodes.txt")
    normals_path = os.path.join(output_dir, "normals.txt")
    patches_path = os.path.join(output_dir, "patches.txt")
    radii_path   = os.path.join(output_dir, "radii.txt")

    nodes   = np.loadtxt(nodes_path).astype(float)
    normals = np.loadtxt(normals_path).astype(float)
    patches = np.loadtxt(patches_path).astype(float)
    radii   = np.loadtxt(radii_path).astype(float)

    # ensure shapes
    if nodes.ndim == 1: nodes = nodes.reshape(1, -1)
    if normals.ndim == 1: normals = normals.reshape(1, -1)
    if patches.ndim == 1: patches = patches.reshape(1, -1)
    if radii.ndim == 0: radii = radii.reshape(1)

    feature_count = 0
    fc_path = os.path.join(output_dir, "feature_count.txt")
    if os.path.exists(fc_path):
        try:
            feature_count = int(open(fc_path, "r", encoding="utf-8").read().strip())
        except Exception:
            feature_count = 0

    tree = cKDTree(nodes)

    M = patches.shape[0]
    frames = np.zeros((M, 3, 3), dtype=np.float64)
    axes   = np.zeros((M, 3), dtype=np.float64)

    n_nodes = nodes.shape[0]
    k_nn = max(1, int(k_nn))
    min_points = max(4, int(min_points))  # PCA needs at least a few points

    for i in range(M):
        y = patches[i]
        r = float(radii[i]) if i < len(radii) else float(radii[-1])
        ratio = float(ratio_feature) if i < feature_count else float(ratio_smooth)

        # candidate selection: ball then fallback to kNN
        ball = r * float(ball_factor)
        idx = tree.query_ball_point(y, ball)

        if len(idx) < min_points:
            k = min(k_nn, n_nodes)
            d, idx_knn = tree.query(y, k=k)
            idx = np.atleast_1d(idx_knn).tolist()

        # average normal -> n
        ns = normals[idx]
        n = _normalize(np.sum(ns, axis=0))
        if np.linalg.norm(n) < 1e-12:
            # fallback normal if all-zero
            n = np.array([1.0, 0.0, 0.0], dtype=float)

        # tangent-plane PCA
        P = nodes[idx] - y[None, :]
        P = P - (P @ n)[:, None] * n[None, :]  # project to tangent plane

        C = P.T @ P
        # eigen decomposition for principal tangent directions
        try:
            w, V = np.linalg.eigh(C)
            t1 = V[:, int(np.argmax(w))]
            t1 = t1 - float(np.dot(t1, n)) * n
            if float(np.linalg.norm(t1)) < 1e-12:
                t1, t2 = _safe_orthogonal(n)
            else:
                t1 = _normalize(t1)
                t2 = _normalize(np.cross(n, t1))
                if float(np.linalg.norm(t2)) < 1e-12:
                    t1, t2 = _safe_orthogonal(n)
                else:
                    t1 = _normalize(np.cross(t2, n))
        except Exception:
            t1, t2 = _safe_orthogonal(n)

        # columns = [t1,t2,n]
        frames[i, :, :] = np.stack([t1, t2, n], axis=1)

        a = r
        b = r
        c = max(1e-12, r * ratio)
        axes[i, :] = np.array([a, b, c], dtype=np.float64)

    np.save(os.path.join(output_dir, "patch_frames.npy"), frames)
    np.save(os.path.join(output_dir, "patch_axes.npy"), axes)

    meta = dict(
        ratio_feature=float(ratio_feature),
        ratio_smooth=float(ratio_smooth),
        k_nn=int(k_nn),
        min_points=int(min_points),
        ball_factor=float(ball_factor),
        max_iters=int(max_iters),
        feature_count=int(feature_count),
        note="patch_frames columns=[t1,t2,n]; patch_axes=(a,b,c) in WORLD units"
    )
    with open(os.path.join(output_dir, "patch_aniso_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return frames, axes, meta
