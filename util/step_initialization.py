#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step_to_mesh_normals_vtk.py

一步到位工具：
1. 使用 Gmsh 从 STEP 几何划分表面三角网格；
2. 使用 pythonocc 重新读取 STEP，提取所有 Face；
3. 为每个三角形绑定一个 STEP Face（通过质心 + 投影）；
4. 对每个三角形的每个顶点，在对应 STEP Face 上计算法向量；
   - 注意：同一几何位置在不同三角中是“不同顶点”，法向量可以不同（比如立方体棱边）；
5. 输出：
   - NPZ：vertices, triangles, tri_face_ids, tri_vertex_normals
   - VTK1：原始网格 + cell_normals + step_face_id（看面法向）
   - VTK2：打散顶点网格 + vertex_normal + step_face_id（在 ParaView 里用 Glyph 看“节点法向箭头”）

依赖：
    pip install gmsh numpy pythonocc-core vtk
"""
import struct
import os
import sys
import numpy as np

# ===================== 全局配置（只改这里就行） =====================

# 输入 STEP 文件
STEP_FILE = r".\model\step_model\complex_geometry\PressureLubricatedCam.step"
# 输出缩放系数：1e-3 表示将 mm 转为 m
# OUTPUT_SCALE_FACTOR = 1e-3
OUTPUT_SCALE_FACTOR = 1
# ---------- Gmsh 网格参数 ----------
# 网格维度：
MESH_DIM = 2    # 2 = 表面三角网格（一般用这个就行） 3 = 生成体网格（四面体等），但我们还是只提取表面三角来算法向
# 尺寸控制（单位与 STEP 模型一致）：
LC_MIN = 0.01   # 最小特征长度（越小网格越细）
LC_MAX = 0.25   # 最大特征长度（越大网格越粗）
# 是否启用基于曲率的自动加密（曲率大/转角大的地方更密）
USE_CURVATURE = True
# 是否做网格质量优化 / 平滑（间接改善单元角度）
MESH_OPTIMIZE          = True
MESH_SMOOTHING_STEPS   = 10
# 是否导出 Gmsh 的原始网格文件（可选）
EXPORT_GMSH_MSH = False
GMSH_MSH_FILE   = None
# 是否导出 Gmsh 的 vtk 网格（可选）
EXPORT_GMSH_VTK = False
GMSH_VTK_FILE   = None
# ---------- STEP Face 绑定 & 法向参数 ----------
# 三角形质心落在 STEP 面 AABB 中的容差（米/毫米同 STEP 单位）
AABB_TOL = 1e-4
# (u,v) 是否落在 Face trim 内的判定容差（用于 FaceClassifier，越大越“宽松”）
FACE_TRIM_TOL = 1e-7
# ---------- 输出：网格 + 法向 ----------
EXPORT_NPZ        = False
OUT_NPZ_FILE      = None

EXPORT_VTK        = True
OUT_VTK_SURFACE   = r".\model\vtp_model\complex_geometry\PressureLubricatedCam_surface_cellnormals.vtp"   # 原始网格 + cell_normals
OUT_VTK_VERT      = r".\model\vtp_model\complex_geometry_check\PressureLubricatedCam_vertex_normals_check.vtp"        # 打散顶点 + vertex_normal（看节点箭头）
OBJ_PATH          = r".\model\obj_model\complex_geometry\PressureLubricatedCam.obj"

# ===================== Gmsh 部分 =====================

import gmsh


def gmsh_init():
    gmsh.initialize(sys.argv)
    gmsh.option.setNumber("General.Terminal", 1)  # 在终端打印 Gmsh 信息


def gmsh_finalize():
    gmsh.finalize()


def gmsh_import_step_geometry(step_file: str):
    """
    使用 Gmsh OpenCASCADE 内核导入 STEP 几何
    """
    gmsh.model.add("model_from_step")
    if not os.path.exists(step_file):
        raise FileNotFoundError(f"STEP 文件不存在: {step_file}")
    gmsh.model.occ.importShapes(step_file)
    gmsh.model.occ.synchronize()


def gmsh_set_mesh_options():
    """
    设置 Gmsh 网格选项：尺寸、曲率加密、质量优化等。
    """
    # 全局尺寸控制：类似“最大/最小边长”的近似控制
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", LC_MIN)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", LC_MAX)

    # 基于曲率的加密：曲率大的地方更小单元
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature",
                          1 if USE_CURVATURE else 0)

    # 元素阶数：1 = 线性单元（我们要三节点三角）
    gmsh.option.setNumber("Mesh.ElementOrder", 1)

    # 网格算法（2D，选择 Frontal 作为例子）
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # 6 = Frontal

    # 质量优化 / 平滑
    if MESH_OPTIMIZE:
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    else:
        gmsh.option.setNumber("Mesh.Optimize", 0)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)

    gmsh.option.setNumber("Mesh.Smoothing", MESH_SMOOTHING_STEPS)


def gmsh_generate_mesh(dim: int = 2):
    """
    生成网格：dim=2 表面网格，dim=3 体网格。
    """
    gmsh.model.mesh.generate(dim)


def gmsh_extract_surface_triangles():
    """
    从 Gmsh 当前模型中提取 2D 三角单元的节点坐标和拓扑。

    返回：
        vertices: (N, 3) float64  —— 所有节点坐标
        triangles: (M, 3) int32   —— 0-based 节点索引
    """
    # 所有节点：nodeTags 是节点编号，coords 是 [x1,y1,z1,x2,y2,z2,...]
    nodeTags, coords, _ = gmsh.model.mesh.getNodes()
    nodeTags = np.asarray(nodeTags, dtype=np.int64)
    coords   = np.asarray(coords, dtype=np.float64).reshape(-1, 3)

    # tag -> index 的映射（因为 tag 不一定从 1 开始连续）
    tag_to_index = {tag: i for i, tag in enumerate(nodeTags)}

    # 获取 2D 单元（维度=2）
    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim=2)

    tri_conn_idx = []
    for etype, eTags, eNodeTags in zip(elemTypes, elemTags, elemNodeTags):
        # Gmsh 中 3-node triangle 的类型号是 2
        if etype == 2:
            eNodeTags = np.asarray(eNodeTags, dtype=np.int64).reshape(-1, 3)
            tri_idx = np.vectorize(tag_to_index.get)(eNodeTags)
            tri_conn_idx.append(tri_idx)

    if not tri_conn_idx:
        raise RuntimeError("没有找到 3-node 三角单元（element type 2）。")

    triangles = np.vstack(tri_conn_idx).astype(np.int32)
    vertices  = coords

    print(f"Gmsh 网格：|V| = {len(vertices)}, |T| = {len(triangles)}")
    return vertices, triangles


# ===================== pythonocc：STEP Face + 法向 =====================

try:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.gp import gp_Pnt, gp_Vec
    from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED

    # --- 可选：用于判断 (u,v) 是否落在 Face 的 trim 内，以及更稳健的法向 ---
    try:
        from OCC.Core.BRepClass import BRepClass_FaceClassifier
        from OCC.Core.gp import gp_Pnt2d
        from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON
    except Exception:
        BRepClass_FaceClassifier = None
        gp_Pnt2d = None
        TopAbs_IN = None
        TopAbs_ON = None

    try:
        from OCC.Core.GeomLProp import GeomLProp_SLProps
    except Exception:
        GeomLProp_SLProps = None
except Exception as e:
    raise RuntimeError("需要 pythonocc-core，请先安装：pip install pythonocc-core") from e


def read_step_faces(step_file: str):
    """
    用 pythonocc 读取 STEP，并提取所有 TopoDS_Face。
    """
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_file)
    if status != 1:
        raise RuntimeError(f"读取 STEP 文件失败: {step_file}")
    reader.TransferRoots()
    shape = reader.OneShape()

    faces = []
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        faces.append(ex.Current())
        ex.Next()
    print(f"STEP faces: {len(faces)}")
    return faces


def build_face_aabb(faces):
    """
    为每个 STEP Face 构造 Bnd_Box AABB。
    """
    boxes = []
    for face in faces:
        bbox = Bnd_Box()
        brepbndlib.Add(face, bbox)
        boxes.append(bbox)
    return boxes



def _normalize_vec3(v):
    n = float(np.linalg.norm(v))
    if n > 0.0:
        return (v / n).astype(np.float64), n
    return v.astype(np.float64), n


def _triangle_geom_normal(vertices, tri):
    p0, p1, p2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
    ng = np.cross(p1 - p0, p2 - p0)
    ng, nrm = _normalize_vec3(ng)
    if nrm <= 0.0:
        return None
    return ng


def _uv_is_inside_face_trim(face, u, v, tol2d=FACE_TRIM_TOL):
    """判断参数点 (u,v) 是否落在 face 的 trim 域内(IN/ON)。
    如果 BRepClass_FaceClassifier 不可用，则返回 True（跳过该检查）。
    """
    if BRepClass_FaceClassifier is None or gp_Pnt2d is None or TopAbs_IN is None or TopAbs_ON is None:
        return True

    try:
        puv = gp_Pnt2d(float(u), float(v))
        # 不同 pythonocc 版本 API 略有差异：优先用 Perform；失败再尝试构造函数方式
        clf = BRepClass_FaceClassifier()
        try:
            clf.Perform(face, puv, float(tol2d))
        except Exception:
            clf = BRepClass_FaceClassifier(face, puv, float(tol2d))
        st = clf.State()
        return (st == TopAbs_IN) or (st == TopAbs_ON)
    except Exception:
        return False


def _face_normal_at_uv(face, surf, u, v):
    """在给定 face 的参数 (u,v) 处计算单位法向。
    优先使用 GeomLProp_SLProps（更稳）；否则退回到 D1(u,v) 的叉乘。
    返回 np.ndarray(3,) 或 None。
    """
    try:
        if GeomLProp_SLProps is not None:
            props = GeomLProp_SLProps(surf, float(u), float(v), 1, 1e-12)
            if hasattr(props, "IsNormalDefined") and (not props.IsNormalDefined()):
                return None
            ndir = props.Normal()  # gp_Dir
            n = np.array([ndir.X(), ndir.Y(), ndir.Z()], dtype=np.float64)
        else:
            # fallback: D1 cross
            P = gp_Pnt()
            D1u = gp_Vec()
            D1v = gp_Vec()
            surf.D1(float(u), float(v), P, D1u, D1v)
            nv = D1u.Crossed(D1v)
            if nv.Magnitude() <= 0.0:
                return None
            nv.Normalize()
            n = np.array([nv.X(), nv.Y(), nv.Z()], dtype=np.float64)

        if face.Orientation() == TopAbs_REVERSED:
            n = -n

        n, nrm = _normalize_vec3(n)
        if nrm <= 0.0:
            return None
        return n
    except Exception:
        return None


def bind_triangles_to_faces(vertices, triangles, faces, face_boxes, tol=AABB_TOL):
    """
    对每个三角形绑定一个 STEP Face（更稳健版本）：
      1) 质心 AABB 粗筛候选 Face；
      2) 将质心投影到候选面的底层曲面，得到 (u,v,dist)；
      3) **检查 (u,v) 是否落在 Face 的 trim 域内 (IN/ON)**，仅这些候选参与竞争；
      4) 若 trim 后没有候选，则退回到“不过滤 trim”的候选；
      5) 选择规则：先最小化投影距离 dist；若 dist 在很小阈值内相等，则用
         三角形几何法向 ng 与 face 法向 nf 的点积 dot(ng,nf) 做兜底，取更一致者。
         这样能解决“重合面/反向面”导致的随机选面问题。

    返回：
        tri_face_ids: (M,) int32
            -1 表示未能可靠绑定（后续可退回用三角形几何法向兜底）。
    """
    num_faces = len(faces)
    num_tris = len(triangles)

    # 距离并列判定阈值：与模型尺度相关
    bbox_diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    dist_eps = max(1e-9, 1e-6 * max(1.0, bbox_diag))

    tri_face_ids = np.full((num_tris,), -1, dtype=np.int32)

    for i, tri in enumerate(triangles):
        pts = vertices[tri]
        centroid = pts.mean(axis=0)
        pnt = gp_Pnt(float(centroid[0]), float(centroid[1]), float(centroid[2]))

        ng = _triangle_geom_normal(vertices, tri)

        # --- 1) AABB 粗筛 ---
        cand_fids = []
        for fid in range(num_faces):
            bbox = face_boxes[fid]

            # Bnd_Box 是否为空
            is_void = False
            if hasattr(bbox, "isVoid"):
                is_void = bbox.isVoid()
            elif hasattr(bbox, "IsVoid"):
                is_void = bbox.IsVoid()

            if not is_void:
                xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
                if not (xmin - tol <= centroid[0] <= xmax + tol and
                        ymin - tol <= centroid[1] <= ymax + tol and
                        zmin - tol <= centroid[2] <= zmax + tol):
                    continue
            cand_fids.append(fid)

        # AABB 一个都没命中时，兜底：全量 face（避免 best_fid 默认为 0 的灾难）
        if not cand_fids:
            cand_fids = list(range(num_faces))

        best_fid = -1
        best_dist = float("inf")
        best_dot = -1e100

        def _scan_candidates(check_trim: bool):
            nonlocal best_fid, best_dist, best_dot
            found = False
            for fid in cand_fids:
                face = faces[fid]
                try:
                    surf = BRep_Tool.Surface(face)
                    proj = GeomAPI_ProjectPointOnSurf(pnt, surf)
                    if proj.NbPoints() <= 0:
                        continue

                    dist = float(proj.LowerDistance())
                    u, v = proj.LowerDistanceParameters()

                    if check_trim and (not _uv_is_inside_face_trim(face, u, v, FACE_TRIM_TOL)):
                        continue

                    # 兜底评分：与三角几何法向的一致性
                    dot = -1e100
                    if ng is not None:
                        nf = _face_normal_at_uv(face, surf, u, v)
                        if nf is not None:
                            dot = float(nf[0]*ng[0] + nf[1]*ng[1] + nf[2]*ng[2])

                    # 选择：dist 优先；dist 并列时 dot 更大者胜
                    if dist < best_dist - dist_eps:
                        best_dist = dist
                        best_fid = fid
                        best_dot = dot
                        found = True
                    elif abs(dist - best_dist) <= dist_eps and dot > best_dot:
                        best_dist = dist
                        best_fid = fid
                        best_dot = dot
                        found = True
                except Exception:
                    continue
            return found

        ok = _scan_candidates(check_trim=True)
        if not ok:
            _scan_candidates(check_trim=False)

        tri_face_ids[i] = int(best_fid)

    print("三角形 → STEP Face 绑定完成（trim 检查 + 法向一致性兜底）。")
    return tri_face_ids

def project_point_to_face(pt, face):
    """
    将一个 3D 点投影到给定 STEP 面上，失败则返回原点坐标。
    """
    try:
        pnt = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
        surf = BRep_Tool.Surface(face)
        proj = GeomAPI_ProjectPointOnSurf(pnt, surf)
        if proj.NbPoints() > 0:
            q = proj.NearestPoint()
            return np.array([q.X(), q.Y(), q.Z()], np.float64)
    except Exception:
        pass
    return np.array(pt, dtype=np.float64)


def compute_normal_on_face(pt, face):
    """
    计算给定 STEP Face 上某点处的法向量：
    - 使用 surface.D1(u,v,...) 得到一阶偏导；
    - n = du × dv，并归一化；
    - 若 face.Orientation() == TopAbs_REVERSED，则翻转法向方向。
    """
    try:
        pnt = gp_Pnt(*[float(x) for x in pt])
        surf = BRep_Tool.Surface(face)
        proj = GeomAPI_ProjectPointOnSurf(pnt, surf)
        if proj.NbPoints() == 0:
            return np.array([0.0, 0.0, 1.0], np.float64)

        u, v = proj.LowerDistanceParameters()
        P = gp_Pnt()
        D1u = gp_Vec()
        D1v = gp_Vec()
        surf.D1(u, v, P, D1u, D1v)

        n = D1u.Crossed(D1v)
        if n.Magnitude() > 0.0:
            n.Normalize()
        else:
            return np.array([0.0, 0.0, 1.0], np.float64)

        # 面方向
        if face.Orientation() == TopAbs_REVERSED:
            n.Reverse()

        return np.array([n.X(), n.Y(), n.Z()], np.float64)
    except Exception:
        return np.array([0.0, 0.0, 1.0], np.float64)



def compute_triangle_vertex_normals(vertices, triangles, faces, tri_face_ids):
    """
    对每个三角形的三个顶点：
    - 若该三角形未能可靠绑定 face（fid==-1），则直接使用三角形几何法向做兜底；
    - 否则：
        * 将顶点投影到绑定的 Face；
        * (u,v) 若不在 trim 域内(IN/ON)，则回退到三角形几何法向；
        * 否则在 (u,v) 处计算 Face 法向。
    返回：
        tri_vertex_normals: (M, 3, 3) float64
    """
    num_tris = len(triangles)
    tri_vertex_normals = np.zeros((num_tris, 3, 3), dtype=np.float64)

    for i, tri in enumerate(triangles):
        fid = int(tri_face_ids[i])
        ng = _triangle_geom_normal(vertices, tri)
        if ng is None:
            ng = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        # 未绑定或越界：整片用几何法向
        if fid < 0 or fid >= len(faces):
            tri_vertex_normals[i, :, :] = ng[None, :]
            continue

        face = faces[fid]
        try:
            surf = BRep_Tool.Surface(face)
        except Exception:
            tri_vertex_normals[i, :, :] = ng[None, :]
            continue

        for local_vid, v_idx in enumerate(tri):
            pt = vertices[int(v_idx)]
            try:
                pnt = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
                proj = GeomAPI_ProjectPointOnSurf(pnt, surf)
                if proj.NbPoints() <= 0:
                    tri_vertex_normals[i, local_vid, :] = ng
                    continue

                u, v = proj.LowerDistanceParameters()
                if not _uv_is_inside_face_trim(face, u, v, FACE_TRIM_TOL):
                    tri_vertex_normals[i, local_vid, :] = ng
                    continue

                n = _face_normal_at_uv(face, surf, u, v)
                tri_vertex_normals[i, local_vid, :] = n if n is not None else ng
            except Exception:
                tri_vertex_normals[i, local_vid, :] = ng

    print("三角形顶点法向量计算完成。")
    return tri_vertex_normals

def save_npz_result(path, vertices, triangles, tri_face_ids, tri_vertex_normals):
    np.savez(
        path,
        vertices=vertices,
        triangles=triangles,
        tri_face_ids=tri_face_ids,
        tri_vertex_normals=tri_vertex_normals,
    )
    print(f"✅ NPZ 已保存: {path}")
    print("  - vertices           :", vertices.shape)
    print("  - triangles          :", triangles.shape)
    print("  - tri_face_ids       :", tri_face_ids.shape)
    print("  - tri_vertex_normals :", tri_vertex_normals.shape)


def export_vtk_surface_with_cell_normals(
    path, vertices, triangles, tri_face_ids, tri_vertex_normals
):
    """
    使用原始顶点 & 拓扑构建一个 vtkPolyData：
    - Points = Gmsh 节点
    - Polys  = 三角单元
    - CellData:
        * cell_normals: 每个三角形一个法向（来自 tri_vertex_normals 平均）
        * step_face_id: 每个三角形对应的 STEP 面 ID
    """
    try:
        import vtk
    except Exception as e:
        print("⚠️ 未安装 vtk，无法导出 VTK 文件。请先安装：pip install vtk")
        print("  错误信息：", e)
        return

    num_points = vertices.shape[0]
    num_cells = triangles.shape[0]

    points = vtk.vtkPoints()
    points.SetNumberOfPoints(num_points)
    for i, v in enumerate(vertices):
        points.SetPoint(i, float(v[0]), float(v[1]), float(v[2]))

    polys = vtk.vtkCellArray()
    for tri in triangles:
        cell = vtk.vtkTriangle()
        cell.GetPointIds().SetId(0, int(tri[0]))
        cell.GetPointIds().SetId(1, int(tri[1]))
        cell.GetPointIds().SetId(2, int(tri[2]))
        polys.InsertNextCell(cell)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(polys)

    # cell 法向量 = 顶点法向平均
    normals = vtk.vtkDoubleArray()
    normals.SetName("cell_normals")
    normals.SetNumberOfComponents(3)
    normals.SetNumberOfTuples(num_cells)
    for i in range(num_cells):
        n = tri_vertex_normals[i].mean(axis=0)
        normals.SetTuple(i, (float(n[0]), float(n[1]), float(n[2])))
    polydata.GetCellData().SetNormals(normals)

    # STEP Face ID
    face_id_array = vtk.vtkIntArray()
    face_id_array.SetName("step_face_id")
    face_id_array.SetNumberOfComponents(1)
    face_id_array.SetNumberOfTuples(num_cells)
    for i, fid in enumerate(tri_face_ids):
        face_id_array.SetTuple1(i, int(fid))
    polydata.GetCellData().AddArray(face_id_array)

    # 创建目录
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(polydata)
    if writer.Write() == 1:
        print(f"✅ VTK（原始网格+cell_normals）已导出: {path}")
    else:
        print("⚠️ 写入 VTK 文件失败：", path)


def export_vtk_per_tri_vertex_normals(
    path, vertices, triangles, tri_face_ids, tri_vertex_normals
):
    """
    为了在 ParaView 里直接查看“节点法向箭头（共用几何点可以有不同法向）”：
    - 把网格完全“打散”为 per-triangle 顶点：
        * 对每个三角形的 3 个顶点，各复制一个点（拓扑上不共享）；
        * 对应的 point data 里写 tri_vertex_normals[i, j, :]；
    - cell_data 里写 step_face_id。
    在 ParaView 中：
        - 打开本文件
        - Filters > Glyph
        - Vectors 选择 vertex_normal
        - Glyph Type = Arrow
        - 调整 Scale Factor，即可看到节点法向箭头
    """
    try:
        import vtk
    except Exception as e:
        print("⚠️ 未安装 vtk，无法导出 VTK 文件。请先安装：pip install vtk")
        print("  错误信息：", e)
        return

    num_tris = triangles.shape[0]

    points = vtk.vtkPoints()
    vertex_normals = vtk.vtkDoubleArray()
    vertex_normals.SetName("vertex_normal")
    vertex_normals.SetNumberOfComponents(3)

    polys = vtk.vtkCellArray()
    face_id_array = vtk.vtkIntArray()
    face_id_array.SetName("step_face_id")
    face_id_array.SetNumberOfComponents(1)

    for i, tri in enumerate(triangles):
        pid0 = pid1 = pid2 = None

        for local_vid in range(3):
            v_idx = int(tri[local_vid])
            v = vertices[v_idx]
            n = tri_vertex_normals[i, local_vid]

            pid = points.InsertNextPoint(float(v[0]), float(v[1]), float(v[2]))
            vertex_normals.InsertNextTuple((float(n[0]), float(n[1]), float(n[2])))

            if local_vid == 0:
                pid0 = pid
            elif local_vid == 1:
                pid1 = pid
            else:
                pid2 = pid

        cell = vtk.vtkTriangle()
        cell.GetPointIds().SetId(0, pid0)
        cell.GetPointIds().SetId(1, pid1)
        cell.GetPointIds().SetId(2, pid2)
        polys.InsertNextCell(cell)

        face_id_array.InsertNextTuple1(int(tri_face_ids[i]))

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(polys)
    polydata.GetPointData().SetNormals(vertex_normals)
    polydata.GetCellData().AddArray(face_id_array)

    # 创建目录
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(polydata)
    if writer.Write() == 1:
        print(f"✅ VTK（打散顶点+vertex_normal）已导出: {path}")
    else:
        print("⚠️ 写入 VTK 文件失败：", path)


def export_obj(path, vertices, triangles, tri_vertex_normals):
    """
    导出为 OBJ 文件格式
    """
    # 创建目录
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'w') as f:
        # 写入顶点和法向量
        for i in range(vertices.shape[0]):
            v = vertices[i]
            f.write(f'v {v[0]} {v[1]} {v[2]}\n')
        
        for i in range(tri_vertex_normals.shape[0]):
            for j in range(3):
                n = tri_vertex_normals[i, j]
                f.write(f'vn {n[0]} {n[1]} {n[2]}\n')
        
        # 写入面，OBJ 文件索引从 1 开始
        for i in range(triangles.shape[0]):
            tri = triangles[i]
            # 每个三角形的三个顶点对应三个法向量，法向量索引为 i*3+1, i*3+2, i*3+3
            f.write(f'f {tri[0]+1}//{i*3+1} {tri[1]+1}//{i*3+2} {tri[2]+1}//{i*3+3}\n')
    
    print(f"✅ OBJ 文件已导出: {path}")



# ===================== 主流程 =====================

def main():
    # ---------- 1) Gmsh：STEP → 网格 ----------
    gmsh_init()
    try:
        print("=== [Gmsh] 导入 STEP 几何 ===")
        gmsh_import_step_geometry(STEP_FILE)

        print("=== [Gmsh] 设置网格参数 ===")
        gmsh_set_mesh_options()

        print("=== [Gmsh] 生成网格 ===")
        gmsh_generate_mesh(MESH_DIM)

        if EXPORT_GMSH_MSH:
            gmsh.write(GMSH_MSH_FILE)
            print("Gmsh .msh 已写出:", GMSH_MSH_FILE)

        if EXPORT_GMSH_VTK:
            gmsh.write(GMSH_VTK_FILE)
            print("Gmsh .vtk 已写出:", GMSH_VTK_FILE)

        print("=== [Gmsh] 提取表面三角网格 ===")
        vertices, triangles = gmsh_extract_surface_triangles()

    finally:
        gmsh_finalize()

    # ---------- 2) pythonocc：STEP Face + 法向 ----------
    print("=== [OCC] 读取 STEP Faces ===")
    faces = read_step_faces(STEP_FILE)
    face_boxes = build_face_aabb(faces)

    print("=== [OCC] 绑定三角形 → STEP Face ===")
    tri_face_ids = bind_triangles_to_faces(vertices, triangles, faces, face_boxes, tol=AABB_TOL)

    print("=== [OCC] 计算每个三角顶点法向量 ===")
    tri_vertex_normals = compute_triangle_vertex_normals(vertices, triangles, faces, tri_face_ids)

    if OUTPUT_SCALE_FACTOR != 1.0:
        print(f"=== [Post-Process] 对坐标应用缩放系数: {OUTPUT_SCALE_FACTOR} ===")
        vertices *= OUTPUT_SCALE_FACTOR
    # ----------------------------------------------------

    # ---------- 3) 输出 ----------
    if EXPORT_NPZ:
        print("=== 保存 NPZ 结果 ===")
        save_npz_result(OUT_NPZ_FILE, vertices, triangles, tri_face_ids, tri_vertex_normals)

    

    if EXPORT_VTK:
        print("=== 导出 VTK（原始网格 + cell_normals） ===")
        export_vtk_surface_with_cell_normals(
            OUT_VTK_SURFACE, vertices, triangles, tri_face_ids, tri_vertex_normals
        )

        print("=== 导出 VTK（打散顶点 + vertex_normal） ===")
        export_vtk_per_tri_vertex_normals(
            OUT_VTK_VERT, vertices, triangles, tri_face_ids, tri_vertex_normals
        )
    
    # 导出 OBJ 文件
    print("=== 导出 OBJ 文件 ===")
    export_obj(OBJ_PATH, vertices, triangles, tri_vertex_normals)

    print("✅ 全流程完成。")


if __name__ == "__main__":
    main()
