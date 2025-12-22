#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Compare default patch radii (computed like cfpurecon) vs radii.txt.
#
# Usage:
#   python compare_patch_radii.py --input_dir /path/to/cfpu_input --out_dir /path/to/out
#
# Expected files in input_dir:
#   nodes.txt   (N,3)
#   patches.txt (M,3)
#   radii.txt   (M,)

import argparse
import os
import csv
import numpy as np
from scipy.spatial import cKDTree


def _load_txt(path, ndmin=1):
    arr = np.loadtxt(path, dtype=float)
    arr = np.asarray(arr, dtype=float)
    if ndmin == 1:
        return arr.reshape(-1)
    return arr


def _summ(x: np.ndarray):
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0:
        return {"n": 0}
    p = np.percentile(x, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    return {
        "n": int(x.size),
        "min": float(p[0]),
        "p01": float(p[1]),
        "p05": float(p[2]),
        "p25": float(p[3]),
        "median": float(p[4]),
        "p75": float(p[5]),
        "p95": float(p[6]),
        "p99": float(p[7]),
        "max": float(p[8]),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
    }


def _print_summ(name, s):
    keys = ["n", "min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max", "mean", "std"]
    print(f"\n[{name}]")
    for k in keys:
        if k in s:
            print(f"  {k:>7s}: {s[k]}")


def compute_default_radii_world(nodes: np.ndarray, patches: np.ndarray, delta: float = 1.0):
    """
    Default radii used in cfpurecon (world units):

      1) H = max nearest-neighbor distance among patch centers
      2) r0 = (1+delta)*H/2   (delta=1 -> r0=H)
      3) radii[:] = r0
      4) coverage fix: ensure every node is covered by at least one patch
         (grow the nearest patch radius to include uncovered nodes)
    """
    M = patches.shape[0]
    if M == 0:
        return np.zeros((0,), dtype=float)

    tree_y = cKDTree(patches)
    nn = tree_y.query(patches, k=2)[0][:, 1]  # nearest neighbor distance (excluding itself)
    H = float(np.max(nn)) if nn.size else 0.0
    r0 = (1.0 + float(delta)) * H / 2.0
    radii = np.full(M, r0, dtype=float)

    tree_x = cKDTree(nodes)

    covered = np.zeros(nodes.shape[0], dtype=bool)
    idx_cache = [None] * M
    for k in range(M):
        ids = tree_x.query_ball_point(patches[k], float(radii[k]))
        idx_cache[k] = np.asarray(ids, dtype=int)
        covered[idx_cache[k]] = True

    missing = np.where(~covered)[0]
    while missing.size > 0:
        i = int(missing[0])
        cp_id = int(tree_y.query(nodes[i], k=1)[1])
        p_dist = float(tree_y.query(nodes[i], k=1)[0])
        new_r = max(float(radii[cp_id]), 1.01 * p_dist)
        if new_r > radii[cp_id]:
            radii[cp_id] = new_r
            ids = tree_x.query_ball_point(patches[cp_id], float(radii[cp_id]))
            idx_cache[cp_id] = np.asarray(ids, dtype=int)
            covered[idx_cache[cp_id]] = True
        else:
            # safety: avoid rare infinite-loop corner
            covered[i] = True
        missing = np.where(~covered)[0]

    return radii


def patch_counts(nodes: np.ndarray, patches: np.ndarray, radii: np.ndarray):
    tree_x = cKDTree(nodes)
    M = patches.shape[0]
    counts = np.zeros(M, dtype=int)
    covered = np.zeros(nodes.shape[0], dtype=bool)
    for k in range(M):
        ids = tree_x.query_ball_point(patches[k], float(radii[k]))
        counts[k] = len(ids)
        if len(ids) > 0:
            covered[np.asarray(ids, dtype=int)] = True
    missing = int(np.sum(~covered))
    return counts, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="directory containing nodes.txt/patches.txt/radii.txt")
    ap.add_argument("--out_dir", default=None, help="output directory (default: input_dir)")
    ap.add_argument("--delta", type=float, default=1.0, help="delta in default radii formula (cfpurecon uses 1.0)")
    ap.add_argument("--no_plots", action="store_true", help="skip plots")
    args = ap.parse_args()

    in_dir = args.input_dir
    out_dir = args.out_dir or in_dir
    os.makedirs(out_dir, exist_ok=True)

    nodes = _load_txt(os.path.join(in_dir, "nodes.txt"), ndmin=2)
    patches = _load_txt(os.path.join(in_dir, "patches.txt"), ndmin=2)
    radii_new = _load_txt(os.path.join(in_dir, "radii.txt"), ndmin=1)

    if radii_new.shape[0] != patches.shape[0]:
        raise ValueError(f"radii.txt length ({radii_new.shape[0]}) != patches ({patches.shape[0]})")

    print(f"N nodes: {nodes.shape[0]} | M patches: {patches.shape[0]}")

    radii_def = compute_default_radii_world(nodes, patches, delta=args.delta)

    cnt_def, miss_def = patch_counts(nodes, patches, radii_def)
    cnt_new, miss_new = patch_counts(nodes, patches, radii_new)

    eps = 1e-12
    ratio = radii_new / np.maximum(radii_def, eps)
    diff = radii_new - radii_def
    adiff = np.abs(diff)

    _print_summ("default radii (world units)", _summ(radii_def))
    _print_summ("new radii (world units; radii.txt)", _summ(radii_new))
    _print_summ("abs diff |new-default|", _summ(adiff))
    _print_summ("signed diff new-default", _summ(diff))
    _print_summ("ratio new/default", _summ(ratio))

    print("\n[coverage / density]")
    print(f"  missing nodes (default): {miss_def} / {nodes.shape[0]}")
    print(f"  missing nodes (new)    : {miss_new} / {nodes.shape[0]}")
    _print_summ("nodes per patch (default)", _summ(cnt_def.astype(float)))
    _print_summ("nodes per patch (new)", _summ(cnt_new.astype(float)))

    K = min(30, patches.shape[0])
    top = np.argsort(-adiff)[:K]
    csv_path = os.path.join(out_dir, "radii_compare_top.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patch_id", "default_r", "new_r", "abs_diff", "signed_diff", "ratio", "cnt_default", "cnt_new"])
        for k in top:
            w.writerow([int(k), float(radii_def[k]), float(radii_new[k]), float(adiff[k]), float(diff[k]), float(ratio[k]), int(cnt_def[k]), int(cnt_new[k])])
    print(f"\nWrote: {csv_path}")

    if not args.no_plots:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.hist(radii_def, bins=60, alpha=0.7, label="default")
        plt.hist(radii_new, bins=60, alpha=0.7, label="new")
        plt.legend()
        plt.title("Patch radii histogram (world units)")
        plt.xlabel("radius")
        plt.ylabel("count")
        p1 = os.path.join(out_dir, "radii_hist.png")
        plt.savefig(p1, dpi=200, bbox_inches="tight")

        plt.figure()
        plt.hist(ratio, bins=60)
        plt.title("Radii ratio (new/default)")
        plt.xlabel("ratio")
        plt.ylabel("count")
        p2 = os.path.join(out_dir, "radii_ratio_hist.png")
        plt.savefig(p2, dpi=200, bbox_inches="tight")

        plt.figure()
        plt.scatter(radii_def, radii_new, s=6)
        lo = min(float(np.min(radii_def)), float(np.min(radii_new)))
        hi = max(float(np.max(radii_def)), float(np.max(radii_new)))
        plt.plot([lo, hi], [lo, hi])
        plt.title("Default vs new radii (world units)")
        plt.xlabel("default radius")
        plt.ylabel("new radius")
        p3 = os.path.join(out_dir, "radii_scatter.png")
        plt.savefig(p3, dpi=200, bbox_inches="tight")

        print(f"Wrote: {p1}\nWrote: {p2}\nWrote: {p3}")


if __name__ == "__main__":
    main()
