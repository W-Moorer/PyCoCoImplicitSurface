#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step_sample_points_gmsh_occ.py

参考你的脚本流程（Gmsh 网格 + pythonOCC Face 绑定 + Face 上法向），做点云采样并输出 PLY/NPY。

依赖：
    pip install gmsh numpy pythonocc-core
可选：
    pip install vtk   # 若你想另外导出 VTK 自己加

用法示例：
    python step_sample_points_gmsh_occ.py --step steps/zuheti.step --out_prefix zuheti_pc
    python step_sample_points_gmsh_occ.py --step steps/zuheti.step --unit mm --export_unit m --out_prefix zuheti_pc_m
    python step_sample_points_gmsh_occ.py --step steps/zuheti.step --r 0.002 --lam 3.0 --out_prefix dense
"""

import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import gmsh

# -------------------- pythonOCC imports --------------------
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED


# ========================= utils =========================
def print_flush(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def read_step_shape(step_file: str):
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_file)
    if status != 1:
        raise RuntimeError(f"读取 STEP 文件失败: {step_file}")
    reader.TransferRoots()
    return reader.OneShape()


def read_step_faces(step_file: str):
    shape = read_step_shape(step_file)
    faces = []
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        faces.append(ex.Current())
        ex.Next()
    return shape, faces


def build_face_aabb(faces):
    boxes = []
    for face in faces:
        bbox = Bnd_Box()
        brepbndlib.Add(face, bbox)  # 新版推荐
        boxes.append(bbox)
    return boxes


def bbox_diag_of_shape(shape) -> float:
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    dx, dy, dz = (xmax - xmin), (ymax - ymin), (zmax - zmin)
    return float(math.sqrt(dx * dx + dy * dy + dz * dz))


def auto_guess_scale_to_m(diag_raw: float) -> float:
    # 经验启发：对角线 > 50 通常是 mm 量级
    return 1e-3 if diag_raw > 50.0 else 1.0


# ========================= Gmsh mesh =========================
def gmsh_init():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)


def gmsh_finalize():
    gmsh.finalize()


def gmsh_import_step(step_file: str):
    gmsh.model.add("model_from_step")
    gmsh.model.occ.importShapes(step_file)
    gmsh.model.occ.synchronize()


def gmsh_set_options(lc_min: float, lc_max: float, use_curvature: bool, optimize: bool, smoothing_steps: int):
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_min)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_max)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1 if use_curvature else 0)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal (2D)

    if optimize:
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    else:
        gmsh.option.setNumber("Mesh.Optimize", 0)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)

    gmsh.option.setNumber("Mesh.Smoothing", smoothing_steps)


def gmsh_generate_surface_mesh():
    gmsh.model.mesh.generate(2)


def gmsh_extract_surface_triangles():
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    node_tags = np.asarray(node_tags, dtype=np.int64)
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    tag_to_index = {tag: i for i, tag in enumerate(node_tags)}

    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim=2)

    tri_list = []
    for etype, eNodeTags in zip(elemTypes, elemNodeTags):
        if etype == 2:  # 3-node triangle
            eNodeTags = np.asarray(eNodeTags, dtype=np.int64).reshape(-1, 3)
            tri_idx = np.vectorize(tag_to_index.get)(eNodeTags)
            tri_list.append(tri_idx)

    if not tri_list:
        raise RuntimeError("没有找到 3-node 三角单元（etype=2）")

    triangles = np.vstack(tri_list).astype(np.int32)
    vertices = coords
    return vertices, triangles


# ========================= Face binding =========================
def bbox_contains_point(bbox: Bnd_Box, p: np.ndarray, tol: float) -> bool:
    is_void = False
    if hasattr(bbox, "IsVoid"):
        is_void = bbox.IsVoid()
    elif hasattr(bbox, "isVoid"):
        is_void = bbox.isVoid()
    if is_void:
        return True
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    return (xmin - tol <= p[0] <= xmax + tol and
            ymin - tol <= p[1] <= ymax + tol and
            zmin - tol <= p[2] <= zmax + tol)


def distance_point_to_face(p_xyz: np.ndarray, face) -> float:
    """投影距离（越小越近）；失败返回 inf"""
    try:
        surf = BRep_Tool.Surface(face)
        pnt = gp_Pnt(float(p_xyz[0]), float(p_xyz[1]), float(p_xyz[2]))
        proj = GeomAPI_ProjectPointOnSurf(pnt, surf)
        if proj.NbPoints() > 0:
            return float(proj.LowerDistance())
    except Exception:
        pass
    return float("inf")


def bind_triangles_to_faces(vertices, triangles, faces, face_boxes, aabb_tol: float):
    tri_face_ids = np.zeros((triangles.shape[0],), dtype=np.int32)
    nf = len(faces)

    print_flush(f"[Bind] triangles={len(triangles)}, faces={nf}")
    for i, tri in enumerate(triangles):
        centroid = vertices[tri].mean(axis=0)

        best_fid = 0
        best_dist = float("inf")

        for fid in range(nf):
            if not bbox_contains_point(face_boxes[fid], centroid, aabb_tol):
                continue
            dist = distance_point_to_face(centroid, faces[fid])
            if dist < best_dist:
                best_dist = dist
                best_fid = fid

        tri_face_ids[i] = best_fid

        if (i + 1) % 2000 == 0:
            print_flush(f"[Bind] {i+1}/{len(triangles)} triangles...")

    print_flush("[Bind] done.")
    return tri_face_ids


# ========================= geometry on face =========================
def project_point_to_face(pt: np.ndarray, face) -> Optional[Tuple[np.ndarray, float, float]]:
    """返回 (proj_xyz, u, v)；失败返回 None"""
    try:
        pnt = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
        surf = BRep_Tool.Surface(face)
        proj = GeomAPI_ProjectPointOnSurf(pnt, surf)
        if proj.NbPoints() == 0:
            return None
        u, v = proj.LowerDistanceParameters()
        q = proj.NearestPoint()
        return np.array([q.X(), q.Y(), q.Z()], np.float64), float(u), float(v)
    except Exception:
        return None


def normal_and_curvature_on_face(proj_xyz: np.ndarray, face, u: float, v: float, tol: float):
    """
    法向：用 surf.D1 求 du,dv，然后 n=du×dv；按 face orientation 修正
    曲率：尽量用 GeomLProp_SLProps（取不到就返回 0）
    """
    # default
    n_out = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    kmax = 0.0
    kmin = 0.0

    try:
        surf = BRep_Tool.Surface(face)

        # normal via D1
        P = gp_Pnt()
        D1u = gp_Vec()
        D1v = gp_Vec()
        surf.D1(u, v, P, D1u, D1v)
        n = D1u.Crossed(D1v)
        if n.Magnitude() > 0:
            n.Normalize()
        else:
            return n_out, kmax, kmin

        if face.Orientation() == TopAbs_REVERSED:
            n.Reverse()

        n_out = np.array([float(n.X()), float(n.Y()), float(n.Z())], dtype=np.float64)
        nn = np.linalg.norm(n_out)
        if nn > 0:
            n_out /= nn

        # curvature (best-effort)
        try:
            props = GeomLProp_SLProps(surf, float(u), float(v), 2, float(tol))
            # IsCurvatureDefined 并非所有 wrapper 都有，try/except 兜底
            kmax = float(props.MaxCurvature())
            kmin = float(props.MinCurvature())
        except Exception:
            kmax, kmin = 0.0, 0.0

        return n_out, kmax, kmin
    except Exception:
        return n_out, kmax, kmin


# ========================= sampling =========================
def tri_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))


@dataclass
class TriInfo:
    v0: int
    v1: int
    v2: int
    area: float
    face_id: int


def build_triangle_infos(vertices, triangles, tri_face_ids) -> Tuple[List[TriInfo], np.ndarray]:
    infos: List[TriInfo] = []
    areas = np.zeros((triangles.shape[0],), dtype=np.float64)
    for i, (t, fid) in enumerate(zip(triangles, tri_face_ids)):
        a, b, c = vertices[t[0]], vertices[t[1]], vertices[t[2]]
        A = tri_area(a, b, c)
        areas[i] = A
        if A > 1e-18:
            infos.append(TriInfo(int(t[0]), int(t[1]), int(t[2]), float(A), int(fid)))
    cdf = np.cumsum(areas / max(areas.sum(), 1e-30))
    return infos, cdf


def sample_point_in_triangle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    u = random.random()
    v = random.random()
    if u + v > 1.0:
        u = 1.0 - u
        v = 1.0 - v
    return a + u * (b - a) + v * (c - a)


def curvature_multiplier(kmax: float, kmin: float, D_m: float, lam: float, kD_max: float) -> float:
    k = max(abs(kmax), abs(kmin))
    kD = k * D_m
    kD = max(0.0, min(kD, kD_max))
    return 1.0 + lam * kD


class VariableRadiusPoisson:
    def __init__(self, cell: float):
        self.cell = float(cell)
        self.grid: Dict[Tuple[int, int, int], List[int]] = {}
        self.points: List[np.ndarray] = []
        self.normals: List[np.ndarray] = []
        self.curv: List[float] = []
        self.radius: List[float] = []

    def _key(self, p: np.ndarray) -> Tuple[int, int, int]:
        return (int(math.floor(p[0] / self.cell)),
                int(math.floor(p[1] / self.cell)),
                int(math.floor(p[2] / self.cell)))

    def try_add(self, p: np.ndarray, n: np.ndarray, c: float, r: float) -> bool:
        k0 = self._key(p)
        rng = int(math.ceil(r / self.cell))
        for dx in range(-rng, rng + 1):
            for dy in range(-rng, rng + 1):
                for dz in range(-rng, rng + 1):
                    kk = (k0[0] + dx, k0[1] + dy, k0[2] + dz)
                    if kk not in self.grid:
                        continue
                    for idx in self.grid[kk]:
                        q = self.points[idx]
                        rq = self.radius[idx]
                        if np.linalg.norm(p - q) < min(r, rq):
                            return False

        idx = len(self.points)
        self.points.append(p)
        self.normals.append(n)
        self.curv.append(float(c))
        self.radius.append(float(r))
        self.grid.setdefault(k0, []).append(idx)
        return True

    def to_arrays(self):
        return (np.asarray(self.points, np.float64),
                np.asarray(self.normals, np.float64),
                np.asarray(self.curv, np.float64))


# ========================= output =========================
def write_ply_ascii(path: str, points: np.ndarray, normals: np.ndarray, curvature: np.ndarray):
    n = points.shape[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property float nx\nproperty float ny\nproperty float nz\n")
        f.write("property float curvature\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = points[i]
            nx, ny, nz = normals[i]
            c = float(curvature[i])
            f.write(f"{x} {y} {z} {nx} {ny} {nz} {c}\n")


# ========================= main =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, help="input STEP file")
    ap.add_argument("--out_prefix", default="cloud", help="output prefix")

    ap.add_argument("--seed", type=int, default=0, help="random seed (0 disables)")

    ap.add_argument("--unit", choices=["auto", "m", "mm"], default="auto",
                    help="how to interpret STEP units (only affects export_unit=m scaling)")
    ap.add_argument("--export_unit", choices=["model", "m"], default="model",
                    help="export points in model units or meters")
    ap.add_argument("--scale_override", type=float, default=0.0,
                    help="force scale_to_m (model * scale_to_m = m). If >0 overrides unit/auto.")

    # density self-adaptive
    ap.add_argument("--r", type=float, default=0.003, help="base spacing ratio: s0 = r * bbox_diag")
    ap.add_argument("--lam", type=float, default=2.0, help="curvature strength (bigger => denser at high curvature)")
    ap.add_argument("--kD_max", type=float, default=50.0, help="cap on k*D")
    ap.add_argument("--tol", type=float, default=1e-7, help="geom tolerance")

    # gmsh mesh options
    ap.add_argument("--use_curvature_mesh", action="store_true", help="enable gmsh curvature-based refinement")
    ap.add_argument("--optimize_mesh", action="store_true", help="enable gmsh optimize")
    ap.add_argument("--smoothing", type=int, default=10, help="gmsh smoothing steps")

    # binding
    ap.add_argument("--aabb_tol", type=float, default=1e-4, help="AABB tol for face candidate filter")

    # sampling
    ap.add_argument("--cand_mult", type=float, default=8.0, help="candidates multiplier")
    ap.add_argument("--max_cand", type=int, default=800000, help="max candidates per batch")
    ap.add_argument("--max_batches", type=int, default=6, help="max batches to reach target")
    ap.add_argument("--max_points", type=int, default=300000, help="hard cap on output points")

    args = ap.parse_args()

    if not os.path.exists(args.step):
        raise FileNotFoundError(args.step)

    if args.seed != 0:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # ---- OCC read faces + bbox ----
    print_flush("=== [OCC] read STEP faces + bbox ===")
    shape, faces = read_step_faces(args.step)
    if len(faces) == 0:
        raise RuntimeError("No faces found in STEP.")
    face_boxes = build_face_aabb(faces)

    diag_raw = bbox_diag_of_shape(shape)
    if diag_raw <= 0:
        raise RuntimeError("Invalid bbox diagonal.")
    print_flush(f"[OCC] faces={len(faces)} bbox_diag_raw={diag_raw}")

    # scale_to_m (only for export_unit=m and curvature normalization)
    if args.scale_override > 0:
        scale_to_m = float(args.scale_override)
    else:
        if args.unit == "m":
            scale_to_m = 1.0
        elif args.unit == "mm":
            scale_to_m = 1e-3
        else:
            scale_to_m = auto_guess_scale_to_m(diag_raw)

    D_m = diag_raw * scale_to_m  # for k*D normalization
    s0 = args.r * diag_raw       # base spacing in MODEL UNITS (unitless adaptive)
    s0_m = s0 * scale_to_m

    print_flush(f"[Scale] scale_to_m={scale_to_m} D_m={D_m}")
    print_flush(f"[Density] s0(model)={s0}  s0(m)={s0_m}")

    # ---- Gmsh mesh ----
    # 用 s0 来设置网格尺寸范围，让网格“跟密度对齐”
    lc_min = max(s0 * 0.5, 1e-12)
    lc_max = max(s0 * 2.0, lc_min * 1.1)

    print_flush("=== [Gmsh] import STEP + mesh ===")
    gmsh_init()
    try:
        gmsh_import_step(args.step)
        gmsh_set_options(
            lc_min=lc_min,
            lc_max=lc_max,
            use_curvature=args.use_curvature_mesh,
            optimize=args.optimize_mesh,
            smoothing_steps=args.smoothing,
        )
        gmsh_generate_surface_mesh()
        vertices, triangles = gmsh_extract_surface_triangles()
    finally:
        gmsh_finalize()

    print_flush(f"[Gmsh] |V|={len(vertices)} |T|={len(triangles)}")

    # ---- Bind triangles -> faces (与你参考代码一致的思路) ----
    print_flush("=== [Bind] triangles -> STEP faces ===")
    tri_face_ids = bind_triangles_to_faces(vertices, triangles, faces, face_boxes, aabb_tol=args.aabb_tol)

    # ---- Build triangle CDF for area-weighted sampling ----
    print_flush("=== [Sample] build triangle areas ===")
    areas = np.zeros((triangles.shape[0],), dtype=np.float64)
    for i, tri in enumerate(triangles):
        a, b, c = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        areas[i] = tri_area(a, b, c)
    area_sum = float(areas.sum())
    if area_sum <= 0:
        raise RuntimeError("Mesh area is zero.")
    cdf = np.cumsum(areas / area_sum)

    # 目标点数：N ≈ A / s0^2
    N_target = int(area_sum / max(s0 * s0, 1e-30))
    N_target = max(1000, min(N_target, args.max_points))
    print_flush(f"[Sample] mesh_area={area_sum}  N_target≈{N_target}")

    # 曲率最密时的最小间距，用于 Poisson 网格 cell
    m_max = 1.0 + args.lam * args.kD_max
    s_min = s0 / math.sqrt(max(m_max, 1e-12))
    sampler = VariableRadiusPoisson(cell=s_min)

    # ---- Sampling loop ----
    print_flush("=== [Sample] start sampling ===")
    for batch in range(args.max_batches):
        if len(sampler.points) >= N_target:
            break

        need = N_target - len(sampler.points)
        # 候选点数：按 need 放大
        Ncand = int(min(args.max_cand, max(5000, need * args.cand_mult)))
        accepted_before = len(sampler.points)

        for j in range(Ncand):
            # pick triangle by CDF
            r = random.random()
            ti = int(np.searchsorted(cdf, r, side="right"))
            ti = min(max(ti, 0), triangles.shape[0] - 1)

            tri = triangles[ti]
            fid = int(tri_face_ids[ti])
            face = faces[fid]

            a, b, c = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            p = sample_point_in_triangle(a, b, c)

            proj = project_point_to_face(p, face)
            if proj is None:
                continue
            p_proj, u, v = proj

            n, kmax, kmin = normal_and_curvature_on_face(p_proj, face, u, v, tol=args.tol)
            curv = max(abs(kmax), abs(kmin))

            m = curvature_multiplier(kmax, kmin, D_m, args.lam, args.kD_max)
            s_local = s0 / math.sqrt(max(m, 1e-12))

            sampler.try_add(p_proj, n, curv, s_local)

            # 进度打印（避免“看起来没输出”）
            if (j + 1) % 50000 == 0:
                print_flush(f"[Sample][batch {batch}] cand {j+1}/{Ncand} accepted={len(sampler.points)}/{N_target}")

        accepted_after = len(sampler.points)
        print_flush(f"[Sample] batch {batch}: +{accepted_after-accepted_before} accepted, total={accepted_after}/{N_target}")

    points, normals, curvature = sampler.to_arrays()
    print_flush(f"=== [Done] final points={points.shape[0]} ===")

    # ---- export scaling ----
    if args.export_unit == "m":
        points_out = points * scale_to_m
    else:
        points_out = points

    # ---- outputs ----
    out_ply = args.out_prefix + ".ply"
    out_pts = args.out_prefix + "_points.npy"
    out_nrm = args.out_prefix + "_normals.npy"
    out_curv = args.out_prefix + "_curvature.npy"
    out_npz = args.out_prefix + ".npz"

    write_ply_ascii(out_ply, points_out, normals, curvature)
    np.save(out_pts, points_out)
    np.save(out_nrm, normals)
    np.save(out_curv, curvature)
    np.savez_compressed(
        out_npz,
        points=points_out,
        normals=normals,
        curvature=curvature,
        bbox_diag_raw=diag_raw,
        scale_to_m=scale_to_m,
        s0_model=s0,
        s0_m=s0_m,
        N_target=N_target,
        export_unit=args.export_unit,
    )

    print_flush(f"PLY: {out_ply}")
    print_flush(f"NPY: {out_pts}, {out_nrm}, {out_curv}")
    print_flush(f"NPZ: {out_npz}")


if __name__ == "__main__":
    main()
