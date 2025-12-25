"""
precompute_single.py

这是一个“单文件版本”的 precompute 库：将原始的以下模块完整合并到一个 Python 文件中：
- kdtree.py
- mesh_io.py
- topology.py
- cfpu_input.py
- segment.py

目标：在一个文件内保留所有原始函数接口与具体实现，确保调用时可覆盖原有功能。

注意：
- 该文件仅做“代码合并/去相对导入”处理；算法实现与原模块保持一致。
- 需要依赖：numpy、pyvista、scipy，以及 util.curlfree_poly（若不存在则尝试 cfpurecon.curlfree_poly）。
"""
import os
import json
import collections
import itertools
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from scipy.linalg import solve
from scipy.interpolate import splprep, splev

# curlfree_poly 依赖：从cfpurecon模块导入
try:
    from src.cfpurecon import curlfree_poly
except Exception:
    try:
        from .cfpurecon import curlfree_poly
    except Exception:
        raise ImportError("无法导入curlfree_poly函数，请确保cfpurecon.py文件存在于src目录中")

# =============================================================================
# kdtree.py
# =============================================================================
def build_kdtree(points):
    return cKDTree(points)

def query_radius(tree, points, r):
    return tree.query_ball_point(points, r)

# =============================================================================
# mesh_io.py
# =============================================================================
def read_mesh(path, compute_split_normals=False):
    mesh = pv.read(path)
    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()
    if compute_split_normals and 'Normals' not in mesh.point_data:
        mesh.compute_normals(inplace=True, split_vertices=True)
    return mesh

# =============================================================================
# topology.py
# =============================================================================
def geometry_hash(points, decimals=6):
    _, inv = np.unique(np.round(points, decimals=decimals), axis=0, return_inverse=True)
    num_geo = int(inv.max()) + 1
    geo_repr = np.full(num_geo, -1, dtype=int)
    for pid in range(points.shape[0]):
        g = int(inv[pid])
        if geo_repr[g] == -1:
            geo_repr[g] = pid
    return inv, num_geo, geo_repr

def geo_point_to_face_corners(faces, inv):
    d = collections.defaultdict(list)
    n_cells = faces.shape[0]
    for face_id in range(n_cells):
        for i in range(3):
            p_id = faces[face_id, i]
            geo_id = inv[p_id]
            d[geo_id].append((face_id, i))
    return d

def build_edge_to_faces(faces, inv):
    m = collections.defaultdict(list)
    n_cells = faces.shape[0]
    for face_id in range(n_cells):
        p_ids = faces[face_id]
        g_ids = [inv[p] for p in p_ids]
        e0 = tuple(sorted((g_ids[0], g_ids[1])))
        e1 = tuple(sorted((g_ids[1], g_ids[2])))
        e2 = tuple(sorted((g_ids[2], g_ids[0])))
        m[e0].append(face_id)
        m[e1].append(face_id)
        m[e2].append(face_id)
    return m

def k_ring_faces(faces, inv, start_face_id, k):
    edge_to_faces = build_edge_to_faces(faces, inv)
    adj = collections.defaultdict(set)
    for fl in edge_to_faces.values():
        if len(fl) < 2:
            continue
        for i in range(len(fl)):
            for j in range(i + 1, len(fl)):
                a = fl[i]
                b = fl[j]
                adj[a].add(b)
                adj[b].add(a)
    visited = set([start_face_id])
    frontier = set([start_face_id])
    for _ in range(k):
        new_frontier = set()
        for f in frontier:
            new_frontier.update(adj.get(f, set()))
        new_frontier -= visited
        visited |= new_frontier
        frontier = new_frontier
    return visited

def face_segmentation(mesh, angle_threshold=30.0, edge_split_threshold=None, require_step_face_id_diff=False):
    n_cells = mesh.n_cells
    points = mesh.points
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    normals = None
    for key in ('Normals', 'normals', 'vertex_normal'):
        if key in mesh.point_data:
            normals = mesh.point_data[key]
            break
    use_cell_normals = False
    cell_normals = None
    if normals is None and 'cell_normals' in mesh.cell_data:
        cell_normals = mesh.cell_data['cell_normals']
        if not isinstance(cell_normals, np.ndarray):
            cell_normals = np.asarray(cell_normals)
        nrm = np.linalg.norm(cell_normals, axis=1)
        nrm[nrm == 0] = 1.0
        cell_normals = (cell_normals.T / nrm).T
        use_cell_normals = True
    elif normals is None:
        mesh.compute_normals(inplace=True, split_vertices=True)
        normals = mesh.point_data['Normals']
    inv, num_geo, geo_repr = geometry_hash(points)
    geo_corners = geo_point_to_face_corners(faces, inv)
    is_boundary_geo = None
    if use_cell_normals:
        cos_node = np.cos(np.radians(angle_threshold))
        is_boundary_geo = np.zeros(num_geo, dtype=bool)
        for geo_id, corners in geo_corners.items():
            if len(corners) <= 1:
                continue
            fids = [fid for (fid, _) in corners]
            N = cell_normals[fids]
            if N.shape[0] <= 1:
                continue
            dots = np.clip(N @ N.T, -1.0, 1.0)
            min_dot = float(np.min(dots + np.eye(dots.shape[0]) * 2.0))
            if min_dot < cos_node:
                is_boundary_geo[geo_id] = True
    parent = list(range(n_cells))
    def find(i):
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]
    def union(i, j):
        ri = find(i)
        rj = find(j)
        if ri != rj:
            parent[ri] = rj
    if edge_split_threshold is None:
        edge_split_threshold = angle_threshold
    cos_threshold = np.cos(np.radians(edge_split_threshold))
    edge_to_faces = build_edge_to_faces(faces, inv)
    for edge, face_list in edge_to_faces.items():
        if len(face_list) < 2:
            continue
        if use_cell_normals:
            gA, gB = edge
            base_face = face_list[0]
            for other_face in face_list[1:]:
                n1 = cell_normals[base_face]
                n2 = cell_normals[other_face]
                dot = float(np.dot(n1, n2))
                dot = np.clip(dot, -1.0, 1.0)
                should_block = (is_boundary_geo[gA] and is_boundary_geo[gB]) and (dot < cos_threshold)
                if require_step_face_id_diff and ('step_face_id' in mesh.cell_data):
                    fid_arr = mesh.cell_data['step_face_id']
                    if not isinstance(fid_arr, np.ndarray):
                        fid_arr = np.asarray(fid_arr)
                    should_block = should_block and (int(fid_arr[base_face]) != int(fid_arr[other_face]))
                if should_block:
                    continue
                union(base_face, other_face)
        else:
            base_face = face_list[0]
            for other_face in face_list[1:]:
                base_p_ids = faces[base_face]
                other_p_ids = faces[other_face]
                base_g_ids = [inv[p] for p in base_p_ids]
                other_g_ids = [inv[p] for p in other_p_ids]
                shared_g_ids = set(base_g_ids) & set(other_g_ids)
                if len(shared_g_ids) < 2:
                    continue
                is_smooth = True
                cos2 = np.cos(np.radians(angle_threshold))
                for g_id in shared_g_ids:
                    idx_base = base_g_ids.index(g_id)
                    pid_base = base_p_ids[idx_base]
                    idx_other = other_g_ids.index(g_id)
                    pid_other = other_p_ids[idx_other]
                    n1 = normals[pid_base]
                    n2 = normals[pid_other]
                    dot = float(np.dot(n1, n2))
                    dot = np.clip(dot, -1.0, 1.0)
                    if dot < cos2:
                        is_smooth = False
                        break
                if is_smooth:
                    union(base_face, other_face)
    final_labels = np.array([find(i) for i in range(n_cells)])
    _, mapped_labels = np.unique(final_labels, return_inverse=True)
    if cell_normals is None:
        try:
            mesh.compute_normals(inplace=True, cell_normals=True, point_normals=False, split_vertices=False)
        except Exception:
            mesh.compute_normals(inplace=True)
        cell_normals = mesh.cell_data['Normals']
    return mapped_labels, cell_normals, edge_to_faces, inv

def compute_boundary_geo(mesh, inv, angle_threshold):
    points = mesh.points
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    num_geo = int(inv.max()) + 1
    geo_corners = geo_point_to_face_corners(faces, inv)
    cell_normals = None
    if 'cell_normals' in mesh.cell_data:
        cell_normals = mesh.cell_data['cell_normals']
        if not isinstance(cell_normals, np.ndarray):
            cell_normals = np.asarray(cell_normals)
        nrm = np.linalg.norm(cell_normals, axis=1)
        nrm[nrm == 0] = 1.0
        cell_normals = (cell_normals.T / nrm).T
    else:
        try:
            mesh.compute_normals(inplace=True, cell_normals=True, point_normals=False, split_vertices=False)
        except Exception:
            mesh.compute_normals(inplace=True)
        cell_normals = mesh.cell_data['Normals']
    cos_node = np.cos(np.radians(angle_threshold))
    is_boundary_geo = np.zeros(num_geo, dtype=bool)
    for geo_id, corners in geo_corners.items():
        if len(corners) <= 1:
            continue
        fids = [fid for (fid, _) in corners]
        N = cell_normals[fids]
        if N.shape[0] <= 1:
            continue
        dots = np.clip(N @ N.T, -1.0, 1.0)
        min_dot = float(np.min(dots + np.eye(dots.shape[0]) * 2.0))
        if min_dot < cos_node:
            is_boundary_geo[geo_id] = True
    return is_boundary_geo

def classify_adjacency(mapped_labels, cell_normals, edge_to_faces):
    adjacency_info = collections.defaultdict(dict)
    for edge, face_list in edge_to_faces.items():
        if len(face_list) < 2:
            continue
        rids = {mapped_labels[fid] for fid in face_list}
        if len(rids) == 2:
            ra, rb = sorted(list(rids))
            if rb in adjacency_info[ra]:
                continue
            faces_a = [fid for fid in face_list if mapped_labels[fid] == ra]
            faces_b = [fid for fid in face_list if mapped_labels[fid] == rb]
            n_a_avg = np.mean(cell_normals[faces_a], axis=0)
            n_b_avg = np.mean(cell_normals[faces_b], axis=0)
            n_a_avg /= np.linalg.norm(n_a_avg)
            n_b_avg /= np.linalg.norm(n_b_avg)
            dot = float(np.dot(n_a_avg, n_b_avg))
            dot = np.clip(dot, -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(dot))
            edge_type = 'convex' if angle_deg < 90.0 else 'concave'
            adjacency_info[ra][rb] = edge_type
            adjacency_info[rb][ra] = edge_type
    return adjacency_info

def detect_sharp_edges(mesh, angle_threshold=30.0, edge_split_threshold=None, require_step_face_id_diff=False):
    points = mesh.points
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    normals = None
    for key in ('Normals', 'normals', 'vertex_normal'):
        if key in mesh.point_data:
            normals = mesh.point_data[key]
            break
    use_cell_normals = False
    cell_normals = None
    if normals is None and 'cell_normals' in mesh.cell_data:
        cell_normals = mesh.cell_data['cell_normals']
        if not isinstance(cell_normals, np.ndarray):
            cell_normals = np.asarray(cell_normals)
        nrm = np.linalg.norm(cell_normals, axis=1)
        nrm[nrm == 0] = 1.0
        cell_normals = (cell_normals.T / nrm).T
        use_cell_normals = True
    elif normals is None:
        mesh.compute_normals(inplace=True, split_vertices=True)
        normals = mesh.point_data['Normals']
    inv, num_geo, geo_repr = geometry_hash(points)
    edge_to_faces = build_edge_to_faces(faces, inv)
    sharp_edges = []
    sharp_edge_lines = []
    if edge_split_threshold is None:
        edge_split_threshold = angle_threshold
    cos_threshold = np.cos(np.radians(edge_split_threshold))
    if use_cell_normals:
        is_boundary_geo = compute_boundary_geo(mesh, inv, angle_threshold)
        for edge, face_list in edge_to_faces.items():
            if len(face_list) < 2:
                continue
            gA, gB = edge
            base_face = face_list[0]
            for other_face in face_list[1:]:
                n1 = cell_normals[base_face]
                n2 = cell_normals[other_face]
                dot = float(np.dot(n1, n2))
                dot = np.clip(dot, -1.0, 1.0)
                should_block = (is_boundary_geo[gA] and is_boundary_geo[gB]) and (dot < cos_threshold)
                if require_step_face_id_diff and ('step_face_id' in mesh.cell_data):
                    fid_arr = mesh.cell_data['step_face_id']
                    if not isinstance(fid_arr, np.ndarray):
                        fid_arr = np.asarray(fid_arr)
                    should_block = should_block and (int(fid_arr[base_face]) != int(fid_arr[other_face]))
                if should_block:
                    angle_deg = np.degrees(np.arccos(dot))
                    is_convex = angle_deg < 90.0
                    pt1_id = geo_repr[gA]
                    pt2_id = geo_repr[gB]
                    sharp_edges.append({'point1_idx': pt1_id, 'point2_idx': pt2_id, 'geo_point1_idx': gA, 'geo_point2_idx': gB, 'is_convex': is_convex, 'angle': angle_deg, 'face1': base_face, 'face2': other_face})
                    sharp_edge_lines.append([points[pt1_id], points[pt2_id]])
    else:
        for edge, face_list in edge_to_faces.items():
            if len(face_list) < 2:
                continue
            base_face = face_list[0]
            for other_face in face_list[1:]:
                base_p_ids = faces[base_face]
                other_p_ids = faces[other_face]
                base_g_ids = [inv[p] for p in base_p_ids]
                other_g_ids = [inv[p] for p in other_p_ids]
                shared_g_ids = set(base_g_ids) & set(other_g_ids)
                if len(shared_g_ids) < 2:
                    continue
                for g_id in list(shared_g_ids)[:2]:
                    idx_base = base_g_ids.index(g_id)
                    pid_base = base_p_ids[idx_base]
                    idx_other = other_g_ids.index(g_id)
                    pid_other = other_p_ids[idx_other]
                    n1 = normals[pid_base]
                    n2 = normals[pid_other]
                    dot = float(np.dot(n1, n2))
                    dot = np.clip(dot, -1.0, 1.0)
                    if dot < cos_threshold:
                        angle_deg = np.degrees(np.arccos(dot))
                        is_convex = angle_deg < 90.0
                        sg = list(shared_g_ids)
                        gA, gB = sg[0], sg[1] if len(sg) > 1 else sg[0]
                        pt1_id = geo_repr[gA]
                        pt2_id = geo_repr[gB]
                        sharp_edges.append({'point1_idx': pt1_id, 'point2_idx': pt2_id, 'geo_point1_idx': gA, 'geo_point2_idx': gB, 'is_convex': is_convex, 'angle': angle_deg, 'face1': base_face, 'face2': other_face})
                        sharp_edge_lines.append([points[pt1_id], points[pt2_id]])
                        break
    return sharp_edges, sharp_edge_lines

def detect_sharp_junctions_strict(mesh, sharp_edges, angle_threshold=30.0):
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    if 'cell_normals' in mesh.cell_data:
        cell_normals = mesh.cell_data['cell_normals']
        if not isinstance(cell_normals, np.ndarray):
            cell_normals = np.asarray(cell_normals)
    else:
        try:
            mesh.compute_normals(inplace=True, cell_normals=True, point_normals=False, split_vertices=False)
        except Exception:
            mesh.compute_normals(inplace=True)
        cell_normals = mesh.cell_data['Normals']
    p2f = collections.defaultdict(list)
    for fid in range(faces.shape[0]):
        a = int(faces[fid, 0]); b = int(faces[fid, 1]); c = int(faces[fid, 2])
        p2f[a].append(fid); p2f[b].append(fid); p2f[c].append(fid)
    pts = set()
    for e in sharp_edges:
        pts.add(int(e['point1_idx']))
        pts.add(int(e['point2_idx']))
    cos_thr = np.cos(np.radians(angle_threshold))
    junctions = set()
    for p in pts:
        fl = p2f.get(p, [])
        if len(fl) < 3:
            continue
        N = cell_normals[np.array(fl, dtype=int)]
        nrm = np.linalg.norm(N, axis=1)
        nrm[nrm == 0] = 1.0
        N = (N.T / nrm).T
        idxs = list(range(N.shape[0]))
        found = False
        for i, j, k in itertools.combinations(idxs, 3):
            v1 = N[i]; v2 = N[j]; v3 = N[k]
            d12 = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
            d23 = float(np.clip(np.dot(v2, v3), -1.0, 1.0))
            d31 = float(np.clip(np.dot(v3, v1), -1.0, 1.0))
            if (d12 < cos_thr) and (d23 < cos_thr) and (d31 < cos_thr):
                found = True
                break
        if found:
            junctions.add(p)
    return junctions

def detect_sharp_junctions_degree(mesh, sharp_edges):
    E = set()
    for e in sharp_edges:
        a = int(e['point1_idx'])
        b = int(e['point2_idx'])
        if a == b:
            continue
        k = (a, b) if a < b else (b, a)
        E.add(k)
    deg = {}
    for a, b in E:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    junctions = {p for p, d in deg.items() if d >= 3}
    return junctions

def _point_to_faces(faces):
    d = collections.defaultdict(list)
    for fid in range(faces.shape[0]):
        a, b, c = int(faces[fid, 0]), int(faces[fid, 1]), int(faces[fid, 2])
        d[a].append(fid)
        d[b].append(fid)
        d[c].append(fid)
    return d

def detect_sharp_junctions(mesh, sharp_edges, angle_threshold=30.0):
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    # use cell normals
    if 'cell_normals' in mesh.cell_data:
        cell_normals = mesh.cell_data['cell_normals']
        if not isinstance(cell_normals, np.ndarray):
            cell_normals = np.asarray(cell_normals)
    else:
        try:
            mesh.compute_normals(inplace=True, cell_normals=True, point_normals=False, split_vertices=False)
        except Exception:
            mesh.compute_normals(inplace=True)
            
        cell_normals = mesh.cell_data['Normals']
    p2f = _point_to_faces(faces)
    cos_thr = np.cos(np.radians(angle_threshold))
    pts = set()
    for e in sharp_edges:
        pts.add(int(e['point1_idx']))
        pts.add(int(e['point2_idx']))
    junctions = set()
    for p in pts:
        fl = p2f.get(p, [])
        if not fl:
            continue
        clusters = []
        for fid in fl:
            n = cell_normals[int(fid)]
            add_new = True
            for i, c in enumerate(clusters):
                dot = float(np.clip(np.dot(n, c), -1.0, 1.0))
                if dot >= cos_thr:
                    # merge: simple average
                    clusters[i] = (clusters[i] + n) / 2.0
                    add_new = False
                    break
            if add_new:
                clusters.append(n)
        if len(clusters) >= 3:
            junctions.add(p)
    return junctions

def build_sharp_segments(sharp_edges, junctions, points, cell_normals, angle_turn_threshold=90.0):
    adj = collections.defaultdict(set)
    E = set()
    for e in sharp_edges:
        a = int(e['point1_idx'])
        b = int(e['point2_idx'])
        if a == b:
            continue
        adj[a].add(b)
        adj[b].add(a)
        key = (a, b) if a < b else (b, a)
        E.add(key)
    # utility: angle between consecutive geometric segments (prev->cur) and (cur->next)
    def turn_angle(prev, cur, nxt):
        v1 = points[int(cur)] - points[int(prev)]
        v2 = points[int(nxt)] - points[int(cur)]
        n1 = float(np.linalg.norm(v1)); n2 = float(np.linalg.norm(v2))
        if n1 <= 1e-12 or n2 <= 1e-12:
            return 0.0
        v1 /= n1; v2 /= n2
        dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
        return float(np.degrees(np.arccos(dot)))
    visited = set()
    segments = []
    def mark_edge(u, v):
        k = (u, v) if u < v else (v, u)
        visited.add(k)
    # paths starting from junctions or degree!=2
    starts = [p for p in adj.keys() if (p in junctions) or (len(adj[p]) != 2)]
    for s in starts:
        for nb in list(adj[s]):
            k = (s, nb) if s < nb else (nb, s)
            if k in visited:
                continue
            path = [s]
            cur = nb
            prev = s
            turn_splits = []
            mark_edge(prev, cur)
            while True:
                path.append(cur)
                deg = len(adj[cur])
                if (cur in junctions) or (deg != 2):
                    break
                # pick next neighbor that is most continuous w.r.t previous edge tangent
                nxts = [x for x in adj[cur] if x != prev]
                if not nxts:
                    break
                best_nxt = None
                best_ang = None
                for cand in nxts:
                    ang = turn_angle(prev, cur, cand)
                    if (best_ang is None) or (ang < best_ang):
                        best_ang = ang
                        best_nxt = cand
                if (best_ang is not None) and (best_ang > float(angle_turn_threshold)):
                    turn_splits.append(int(cur))
                    break
                nxt = best_nxt if best_nxt is not None else nxts[0]
                k2 = (cur, nxt) if cur < nxt else (nxt, cur)
                if k2 in visited:
                    break
                mark_edge(cur, nxt)
                prev, cur = cur, nxt
            edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            segments.append({'vertices': path, 'edges': edges, 'closed': False, 'turn_splits': turn_splits})
    # remaining edges potentially form closed loops
    remaining = [e for e in E if e not in visited]
    used_nodes = set()
    for e in remaining:
        if e in visited:
            continue
        u, v = e
        # start at u and follow until loop closes
        path = [u]
        cur = v
        prev = u
        mark_edge(prev, cur)
        while True:
            path.append(cur)
            nxts = [x for x in adj[cur] if x != prev]
            if not nxts:
                break
            best_nxt = None
            best_ang = None
            for cand in nxts:
                ang = turn_angle(prev, cur, cand)
                if (best_ang is None) or (ang < best_ang):
                    best_ang = ang
                    best_nxt = cand
            if (best_ang is not None) and (best_ang > float(angle_turn_threshold)):
                break
            nxt = best_nxt if best_nxt is not None else nxts[0]
            k2 = (cur, nxt) if cur < nxt else (nxt, cur)
            if k2 in visited:
                break
            mark_edge(cur, nxt)
            prev, cur = cur, nxt
            if cur == path[0]:
                break
        edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        segments.append({'vertices': path, 'edges': edges, 'closed': (len(path) > 1 and path[-1] == path[0]), 'turn_splits': []})
    return segments

# =============================================================================
# cfpu_input.py
# =============================================================================
def extract_surface_regions(input_path, angle_threshold=30.0, prefer_cell_region=True):
    mesh = read_mesh(input_path, compute_split_normals=False)
    if 'Normals' not in mesh.point_data:
        mesh.compute_normals(inplace=True, split_vertices=True)
    points = mesh.points
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    normals = None
    for key in ('Normals', 'normals', 'vertex_normal'):
        if key in mesh.point_data:
            normals = mesh.point_data[key]
            break
    if prefer_cell_region and ('RegionId' in mesh.cell_data):
        labels = mesh.cell_data['RegionId']
        _ml, cell_normals, _e2f, _inv = face_segmentation(mesh, angle_threshold)
        mapped_labels = labels
    else:
        mapped_labels, cell_normals, _e2f, _inv = face_segmentation(mesh, angle_threshold)
    uniq = sorted(list(set(map(int, mapped_labels.tolist()))))
    regions = []
    for rid in uniq:
        fids = np.where(mapped_labels == rid)[0]
        vset = set()
        for fid in fids:
            tri = faces[int(fid)]
            vset.add(int(tri[0])); vset.add(int(tri[1])); vset.add(int(tri[2]))
        gpids = sorted(list(vset))
        gp2l = {pid: i for i, pid in enumerate(gpids)}
        pts_local = np.array([points[pid] for pid in gpids], dtype=float)
        fcs_local = np.array([[gp2l[int(faces[fid, 0])], gp2l[int(faces[fid, 1])], gp2l[int(faces[fid, 2])]] for fid in fids], dtype=int)
        fn_local = np.array([cell_normals[int(fid)] for fid in fids], dtype=float)
        cn_local = []
        for fid in fids:
            tri = faces[int(fid)]
            if normals is not None:
                cn_local.append([normals[int(tri[0])].tolist(), normals[int(tri[1])].tolist(), normals[int(tri[2])].tolist()])
            else:
                n = cell_normals[int(fid)].tolist()
                cn_local.append([n, n, n])
        cn_local = np.array(cn_local, dtype=float)
        vfaces = {pid: [] for pid in gpids}
        for li, fid in enumerate(fids):
            tri = faces[int(fid)]
            vfaces[int(tri[0])].append(li)
            vfaces[int(tri[1])].append(li)
            vfaces[int(tri[2])].append(li)
        vn_local = np.zeros((len(gpids), 3), dtype=float)
        for pid, li_list in vfaces.items():
            if li_list:
                vnorm = np.mean(fn_local[li_list], axis=0)
                nv = float(np.linalg.norm(vnorm))
                vn_local[gp2l[pid]] = vnorm / max(nv, 1e-12)
        regions.append({'region_id': int(rid), 'points': pts_local, 'faces': fcs_local, 'face_normals': fn_local, 'corner_normals': cn_local, 'vertex_normals': vn_local, 'global_point_ids': gpids, 'global_face_ids': fids.tolist()})
    return regions

def save_regions(output_dir, regions, fmt='json'):
    os.makedirs(output_dir, exist_ok=True)
    for r in regions:
        rid = int(r['region_id'])
        if fmt == 'npz':
            np.savez(os.path.join(output_dir, f'region_{rid}.npz'), points=r['points'], faces=r['faces'], corner_normals=r['corner_normals'], face_normals=r['face_normals'])
        else:
            data = {
                'region_id': rid,
                'points': r['points'].tolist(),
                'faces': r['faces'].tolist(),
                'corner_normals': r['corner_normals'].tolist(),
                'face_normals': r['face_normals'].tolist(),
            }
            with open(os.path.join(output_dir, f'region_{rid}.json'), 'w') as f:
                json.dump(data, f)

def _unique_edges_point_ids(faces):
    e = set()
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        e.add(tuple(sorted((a, b))))
        e.add(tuple(sorted((b, c))))
        e.add(tuple(sorted((c, a))))
    return list(e)

def _avg_edge_length(points, faces):
    edges = _unique_edges_point_ids(faces)
    if not edges:
        return 0.0
    l = []
    for a, b in edges:
        l.append(np.linalg.norm(points[a] - points[b]))
    return float(np.mean(l)) if l else 0.0

def _poisson_disk(points, radius):
    if points.shape[0] == 0 or radius <= 0:
        return np.empty((0, 3))
    sel = []
    tree = None
    for p in points:
        if not sel:
            sel.append(p)
            tree = cKDTree(np.array(sel))
            continue
        if tree.query(p, k=1)[0] > radius:
            sel.append(p)
            tree = cKDTree(np.array(sel))
    return np.array(sel)

def _hermite_sample(p0, p1, t0, t1, step_len):
    d = p1 - p0
    L = float(np.linalg.norm(d))
    if L == 0:
        return np.array([p0]), np.array([t0])
    n = int(np.ceil(L / max(step_len, 1e-12)))
    n = max(n, 1)
    u = np.linspace(0.0, 1.0, n + 1)
    u2 = u*u
    u3 = u2*u
    h00 = 2*u3 - 3*u2 + 1
    h10 = u3 - 2*u2 + u
    h01 = -2*u3 + 3*u2
    h11 = u3 - u2
    pts = (h00.reshape(-1, 1) * p0.reshape(1, -1)) + (h10.reshape(-1, 1) * t0.reshape(1, -1)) + (h01.reshape(-1, 1) * p1.reshape(1, -1)) + (h11.reshape(-1, 1) * t1.reshape(1, -1))
    dh00 = 6*u2 - 6*u
    dh10 = 3*u2 - 4*u + 1
    dh01 = -6*u2 + 6*u
    dh11 = 3*u2 - 2*u
    tg = (dh00.reshape(-1, 1) * p0.reshape(1, -1)) + (dh10.reshape(-1, 1) * t0.reshape(1, -1)) + (dh01.reshape(-1, 1) * p1.reshape(1, -1)) + (dh11.reshape(-1, 1) * t1.reshape(1, -1))
    return pts, tg

def _compute_segment_tangents(verts, edge_info, cell_normals, points):
    ts = []
    es = []
    for i in range(len(verts)-1):
        a = int(verts[i]); b = int(verts[i+1])
        k = (a, b) if a < b else (b, a)
        e = edge_info.get(k, None)
        if e is None:
            t = points[b] - points[a]
        else:
            nA = cell_normals[int(e['face1'])]
            nB = cell_normals[int(e['face2'])]
            t = np.cross(nA, nB)
        n = float(np.linalg.norm(t))
        if n == 0:
            t = points[b] - points[a]
            n = float(np.linalg.norm(t))
        t = t / max(n, 1e-12)
        ts.append(t)
        es.append((a, b, e))
    T = []
    for i in range(len(verts)):
        if i == 0:
            T.append(ts[0])
        elif i == len(verts)-1:
            T.append(ts[-1])
        else:
            v = ts[i-1] + ts[i]
            n = float(np.linalg.norm(v))
            if n == 0:
                v = ts[i]
                n = float(np.linalg.norm(v))
            T.append(v / max(n, 1e-12))
    return np.array(T), es

def _fit_global_bspline(P):
    d = np.linalg.norm(P[1:] - P[:-1], axis=1) if P.shape[0] > 1 else np.array([1.0])
    u = np.zeros(P.shape[0], dtype=float)
    if P.shape[0] > 1:
        u[1:] = np.cumsum(d)
    L = float(u[-1]) if u[-1] > 0 else 1.0
    u = u / L
    x = P[:, 0]
    y = P[:, 1]
    z = P[:, 2]
    k_use = min(3, max(1, P.shape[0]-1))
    tck, _u = splprep([x, y, z], u=u, s=0.0, k=k_use)
    return tck, u, 1

def _curvature_radius(tck, u):
    k = tck[2]
    Dx, Dy, Dz = splev(u, tck, der=1)
    D = np.vstack([Dx, Dy, Dz]).T
    V = np.linalg.norm(D, axis=1)
    if k >= 2:
        D2x, D2y, D2z = splev(u, tck, der=2)
        D2 = np.vstack([D2x, D2y, D2z]).T
        C = np.linalg.norm(np.cross(D, D2), axis=1)
        kappa = C / np.maximum(V**3, 1e-12)
        R = np.divide(1.0, np.maximum(kappa, 1e-12))
    else:
        R = np.full(V.shape[0], 1e9, dtype=float)
    return R, V

def _point_to_faces_map(faces):
    d = {}
    for fid in range(faces.shape[0]):
        a = int(faces[fid, 0]); b = int(faces[fid, 1]); c = int(faces[fid, 2])
        la = d.get(a, None)
        if la is None:
            d[a] = [fid]
        else:
            la.append(fid)
        lb = d.get(b, None)
        if lb is None:
            d[b] = [fid]
        else:
            lb.append(fid)
        lc = d.get(c, None)
        if lc is None:
            d[c] = [fid]
        else:
            lc.append(fid)
    return d

def _adaptive_params(tck, u0=0.0, u1=1.0, min_step=1e-5, max_step=1e-2, curvature_factor=0.1):
    """
    基于曲率的自适应采样
    
    参数：
        tck: B样条曲线的参数（t, c, k）
        u0: 起始参数值
        u1: 结束参数值
        min_step: 最小步长（避免过密采样）
        max_step: 最大步长（避免过稀采样）
        curvature_factor: 曲率影响因子（值越大，曲率对步长影响越大）
    
    返回：
        us: 采样参数值数组
    """
    us = [u0]
    u = u0
    while u < u1 - 1e-9:
        R, V = _curvature_radius(tck, np.array([u]))
        rad = float(R[0])
        vel = float(V[0])
        
        # 基于曲率半径动态调整步长：曲率越大（rad越小），步长越小
        step_mm = curvature_factor * rad
        step_mm = max(min_step, min(step_mm, max_step))
        
        # 计算参数步长：速度是曲线的切向量长度
        du = step_mm / max(vel, 1e-9)
        
        # 确保不超过u1，并且步长不太小
        un = min(u + du, u1)
        if un - u < 1e-6:
            un = u + 1e-6
        
        us.append(un)
        u = un
    return np.array(us, dtype=float)

def interpolate_normals(tck, u_samples, end_normals):
    """
    沿B样条曲线插值法向量
    
    参数：
        tck: B样条曲线的参数（t, c, k）
        u_samples: 采样参数值数组
        end_normals: 端点法向量，格式为(nA_start, nA_end, nB_start, nB_end)
            nA_start: 起点处面片A的法向量
            nA_end: 终点处面片A的法向量
            nB_start: 起点处面片B的法向量
            nB_end: 终点处面片B的法向量
    
    返回：
        nA_samples: 采样点处面片A的法向量数组
        nB_samples: 采样点处面片B的法向量数组
    """
    nA_start, nA_end, nB_start, nB_end = end_normals
    
    # 归一化输入法向量
    nA_start = nA_start / np.linalg.norm(nA_start)
    nA_end = nA_end / np.linalg.norm(nA_end)
    nB_start = nB_start / np.linalg.norm(nB_start)
    nB_end = nB_end / np.linalg.norm(nB_end)
    
    # 计算采样点相对于曲线长度的位置
    total_length = 0.0
    lengths = [0.0]
    
    # 计算各采样点处的曲线长度
    for i in range(1, len(u_samples)):
        u0 = u_samples[i-1]
        u1 = u_samples[i]
        
        # 在u0到u1之间取10个点计算弧长
        us_segment = np.linspace(u0, u1, 10)
        points_segment = np.array(splev(us_segment, tck)).T
        
        # 计算弧长
        segment_length = 0.0
        for j in range(1, len(points_segment)):
            segment_length += np.linalg.norm(points_segment[j] - points_segment[j-1])
        
        total_length += segment_length
        lengths.append(total_length)
    
    # 归一化长度到[0, 1]
    if total_length > 0:
        lengths = np.array(lengths) / total_length
    else:
        lengths = np.linspace(0, 1, len(u_samples))
    
    # 沿曲线长度进行线性插值
    nA_samples = []
    nB_samples = []
    
    for l in lengths:
        # 线性插值法向量
        nA = (1 - l) * nA_start + l * nA_end
        nB = (1 - l) * nB_start + l * nB_end
        
        # 归一化插值后的法向量
        nA = nA / np.linalg.norm(nA)
        nB = nB / np.linalg.norm(nB)
        
        nA_samples.append(nA)
        nB_samples.append(nB)
    
    return np.array(nA_samples), np.array(nB_samples)

def optimized_dual_offset(Cj, tj, nA, nB, curvature, edge_length, base_offset=1e-5):
    """
    优化的双侧偏移采样
    
    参数：
        Cj: 曲线上的采样点
        tj: 采样点处的切线向量
        nA: 面片A的法向量
        nB: 面片B的法向量
        curvature: 采样点处的曲率
        edge_length: 原始边缘的长度
        base_offset: 基础偏移距离
    
    返回：
        pL: 左岸偏移点
        pR: 右岸偏移点
        dL: 左岸偏移方向
        dR: 右岸偏移方向
    """
    # 归一化输入法向量和切线向量
    nA = nA / np.linalg.norm(nA)
    nB = nB / np.linalg.norm(nB)
    tj = tj / np.linalg.norm(tj)
    
    # 计算垂直于切线的方向
    dL = np.cross(nA, tj)
    dR = np.cross(tj, nB)
    
    # 归一化偏移方向
    dLn = np.linalg.norm(dL)
    dRn = np.linalg.norm(dR)
    
    if dLn == 0 or dRn == 0:
        return None, None, None, None
    
    dL = dL / dLn
    dR = dR / dRn
    
    # 基于曲率和边缘长度动态调整偏移距离
    # 曲率大的区域使用较小的偏移距离，避免过度偏移
    # 曲率小的区域使用较大的偏移距离，提高采样精度
    offset_distance = base_offset * edge_length
    offset_distance = offset_distance / max(curvature, 1e-6)
    offset_distance = max(1e-7, min(offset_distance, 1e-3))
    
    # 生成偏移点
    pL = Cj + offset_distance * dL
    pR = Cj + offset_distance * dR
    
    return pL, pR, dL, dR

def _local_cfpu_solver(x_local, ui):
    n = x_local.shape[0]
    if n == 0:
        return None
    CFP, P = curlfree_poly(x_local, 1)
    CFPt = CFP.T
    xx = x_local[:, 0]
    yy = x_local[:, 1]
    zz = x_local[:, 2]
    dx = xx.reshape(-1, 1) - xx.reshape(1, -1)
    dy = yy.reshape(-1, 1) - yy.reshape(1, -1)
    dz = zz.reshape(-1, 1) - zz.reshape(1, -1)
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    b = np.zeros(3*n + 3)
    b[0:3*n:3] = ui[:, 0]
    b[1:3*n:3] = ui[:, 1]
    b[2:3*n:3] = ui[:, 2]
    A = np.zeros((3*n + 3, 3*n + 3))
    eta_temp = -r
    zeta_temp = -np.divide(1.0, r, where=(r!=0))
    dphi_xx = zeta_temp * dx**2 + eta_temp
    dphi_yy = zeta_temp * dy**2 + eta_temp
    dphi_zz = zeta_temp * dz**2 + eta_temp
    dphi_xy = zeta_temp * dx * dy
    dphi_xz = zeta_temp * dx * dz
    dphi_yz = zeta_temp * dy * dz
    A[0:3*n:3, 0:3*n:3] = dphi_xx
    A[0:3*n:3, 1:3*n:3] = dphi_xy
    A[0:3*n:3, 2:3*n:3] = dphi_xz
    A[1:3*n:3, 0:3*n:3] = dphi_xy
    A[1:3*n:3, 1:3*n:3] = dphi_yy
    A[1:3*n:3, 2:3*n:3] = dphi_yz
    A[2:3*n:3, 0:3*n:3] = dphi_xz
    A[2:3*n:3, 1:3*n:3] = dphi_yz
    A[2:3*n:3, 2:3*n:3] = dphi_zz
    A[0:3*n, 3*n:] = CFP
    A[3*n:, 0:3*n] = CFPt
    try:
        coeffs = solve(A + 3*n*1e-4*np.eye(3*n + 3), b, assume_a='sym', check_finite=False)
    except Exception:
        coeffs = np.linalg.lstsq(A + 3*n*1e-4*np.eye(3*n + 3), b, rcond=None)[0]
    coeffsp = coeffs[3*n:]
    coeffs = coeffs[:3*n]
    coeffsx = coeffs[0:3*n:3]
    coeffsy = coeffs[1:3*n:3]
    coeffsz = coeffs[2:3*n:3]
    temp_potential_nodes = np.sum(eta_temp * (dx * coeffsx.reshape(1, -1) + dy * coeffsy.reshape(1, -1) + dz * coeffsz.reshape(1, -1)), axis=1) + P @ coeffsp
    A1 = np.ones((n+1, n+1))
    A1[0:n, 0:n] = -r
    A1[-1, -1] = 0.0
    b1 = np.concatenate([temp_potential_nodes, np.array([0.0])])
    try:
        coeffs_correction = solve(A1, b1, assume_a='sym', check_finite=False)
    except Exception:
        coeffs_correction = np.linalg.lstsq(A1, b1, rcond=None)[0]
    coeffs_correction_const = coeffs_correction[-1]
    coeffs_correction_vec = coeffs_correction[:-1]
    return (x_local, coeffsx, coeffsy, coeffsz, coeffsp, coeffs_correction_vec, coeffs_correction_const)

def _eval_potential(xp, solver):
    x_local, coeffsx, coeffsy, coeffsz, coeffsp, cvec, cconst = solver
    dx = xp[0].reshape(-1, 1) - x_local[:, 0].reshape(1, -1)
    dy = xp[1].reshape(-1, 1) - x_local[:, 1].reshape(1, -1)
    dz = xp[2].reshape(-1, 1) - x_local[:, 2].reshape(1, -1)
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    etab = -r
    phib = -r
    Pb = curlfree_poly(xp.reshape(1, 3), 1)[1]
    temp_potential = np.sum(etab * (dx * coeffsx.reshape(1, -1) + dy * coeffsy.reshape(1, -1) + dz * coeffsz.reshape(1, -1)), axis=1) + Pb @ coeffsp
    potential_correction = phib @ cvec + cconst
    return float(temp_potential - potential_correction)

def _project_smooth(points_ref, normals_ref, seeds, k=64, iters=8):
    if seeds.shape[0] == 0:
        return seeds, np.empty((0, 3))
    tree = cKDTree(points_ref)
    out_pts = []
    out_nrm = []
    for c in seeds:
        d, idx = tree.query(c, k=min(k, points_ref.shape[0]))
        x_local = points_ref[idx]
        ui = normals_ref[idx]
        if x_local.shape[0] < 8:
            out_pts.append(c)
            out_nrm.append(normals_ref[idx[0]])
            continue
        solver = _local_cfpu_solver(x_local, ui)
        if solver is None:
            out_pts.append(c)
            out_nrm.append(normals_ref[idx[0]])
            continue
        xp = c.copy()
        for _ in range(iters):
            s0 = _eval_potential(xp, solver)
            h = max(np.mean(d), 1e-6) * 0.05
            gx = (_eval_potential(np.array([xp[0]+h, xp[1], xp[2]]), solver) - s0) / h
            gy = (_eval_potential(np.array([xp[0], xp[1]+h, xp[2]]), solver) - s0) / h
            gz = (_eval_potential(np.array([xp[0], xp[1], xp[2]+h]), solver) - s0) / h
            g = np.array([gx, gy, gz])
            gnorm2 = float(np.dot(g, g))
            if gnorm2 <= 1e-16:
                break
            step = s0 / gnorm2
            xp = xp - step * g
            if abs(s0) < h * 1e-3:
                break
        gn = np.linalg.norm(g)
        if gn == 0:
            out_pts.append(xp)
            out_nrm.append(normals_ref[idx[0]])
        else:
            out_pts.append(xp)
            out_nrm.append(g / gn)
    return np.array(out_pts), np.array(out_nrm)

def _compute_radii(nodes, patches, feature_count, r_small, r_large, n_min=8):
    if patches.shape[0] == 0:
        return np.empty((0,), dtype=float)
    tree = cKDTree(nodes)
    radii = np.zeros(patches.shape[0], dtype=float)
    for i in range(patches.shape[0]):
        base = r_small if i < feature_count else r_large
        k = min(n_min, nodes.shape[0])
        d = tree.query(patches[i], k=k)[0]
        dn = float(d[-1]) if np.ndim(d) > 0 else float(d)
        radii[i] = max(base, dn)
    return radii

# =============================================================================
# B1: 生成 sharp_Lmin + 尖锐边中心线采样（点/切向/两侧法向）并落盘
# =============================================================================
def _compute_sharp_Lmin(points: np.ndarray, faces: np.ndarray, sharp_edges: list) -> float:
    """从 sharp_edges 关联到的 face1/face2 三角形集合中，取最短边长。若失败则退化为全局最短边。"""
    sharp_face_ids = set()
    for e in sharp_edges:
        if isinstance(e, dict):
            if 'face1' in e: sharp_face_ids.add(int(e['face1']))
            if 'face2' in e: sharp_face_ids.add(int(e['face2']))

    def tri_min_edge(tri_ids):
        tri = faces[np.asarray(tri_ids, dtype=int)]
        pa = points[tri[:, 0]]
        pb = points[tri[:, 1]]
        pc = points[tri[:, 2]]
        lab = np.linalg.norm(pa - pb, axis=1)
        lbc = np.linalg.norm(pb - pc, axis=1)
        lca = np.linalg.norm(pc - pa, axis=1)
        return float(np.min(np.concatenate([lab, lbc, lca], axis=0)))

    if len(sharp_face_ids) > 0:
        Lmin = tri_min_edge(sorted(list(sharp_face_ids)))
    else:
        # fallback: 全局最短边
        tri = faces
        pa = points[tri[:, 0]]
        pb = points[tri[:, 1]]
        pc = points[tri[:, 2]]
        lab = np.linalg.norm(pa - pb, axis=1)
        lbc = np.linalg.norm(pb - pc, axis=1)
        lca = np.linalg.norm(pc - pa, axis=1)
        Lmin = float(np.min(np.concatenate([lab, lbc, lca], axis=0)))

    return max(Lmin, 1e-12)


def export_sharp_curve_for_b1(output_dir: str,
                             input_path: str,
                             points: np.ndarray,
                             faces: np.ndarray,
                             cell_normals: np.ndarray,
                             sharp_edges: list,
                             sample_step: float = None,
                             tol_ratio: float = 0.01,
                             max_points: int = 50000):
    """
    落盘：
      - sharp_curve_points_raw.npy  (Q,3)
      - sharp_curve_tangents.npy    (Q,3)
      - sharp_curve_n1.npy          (Q,3)
      - sharp_curve_n2.npy          (Q,3)
      - sharp_curve_meta.json       (含 sharp_Lmin / tol_geom 等)
    """
    os.makedirs(output_dir, exist_ok=True)
    if sharp_edges is None or len(sharp_edges) == 0:
        return

    sharp_Lmin = _compute_sharp_Lmin(points, faces, sharp_edges)
    tol_geom = float(tol_ratio) * sharp_Lmin  # 你的误差目标（默认 1% Lmin）

    # 采样步长：不给就取 0.25*sharp_Lmin（足够密以做 KDTree 距离近似）
    if sample_step is None:
        sample_step = 0.25 * sharp_Lmin
    sample_step = max(float(sample_step), 1e-12)

    def _safe_normalize(v):
        n = float(np.linalg.norm(v))
        if n < 1e-12:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        return v / n

    curve_pts = []
    curve_tan = []
    curve_n1 = []
    curve_n2 = []

    for e in sharp_edges:
        if not isinstance(e, dict):
            continue
        a = int(e['point1_idx'])
        b = int(e['point2_idx'])
        p1 = points[a]
        p2 = points[b]
        seg = p2 - p1
        L = float(np.linalg.norm(seg))
        if L < 1e-12:
            continue
        t = seg / L

        f1 = int(e.get('face1', -1))
        f2 = int(e.get('face2', -1))
        n1 = _safe_normalize(cell_normals[f1]) if (0 <= f1 < cell_normals.shape[0]) else np.array([0.0, 0.0, 1.0], dtype=float)
        n2 = _safe_normalize(cell_normals[f2]) if (0 <= f2 < cell_normals.shape[0]) else n1.copy()

        m = max(2, int(np.ceil(L / sample_step)) + 1)
        ts = np.linspace(0.0, 1.0, m, dtype=float)
        for s in ts:
            p = (1.0 - s) * p1 + s * p2
            curve_pts.append(p)
            curve_tan.append(t)
            curve_n1.append(n1)
            curve_n2.append(n2)

    if len(curve_pts) == 0:
        return

    curve_pts = np.asarray(curve_pts, dtype=float)
    curve_tan = np.asarray(curve_tan, dtype=float)
    curve_n1 = np.asarray(curve_n1, dtype=float)
    curve_n2 = np.asarray(curve_n2, dtype=float)

    # 去重（避免重复点导致后续奇异）
    q = max(sample_step * 0.25, 1e-12)
    key = np.round(curve_pts / q).astype(np.int64)
    _, uniq_idx = np.unique(key, axis=0, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    curve_pts = curve_pts[uniq_idx]
    curve_tan = curve_tan[uniq_idx]
    curve_n1 = curve_n1[uniq_idx]
    curve_n2 = curve_n2[uniq_idx]

    # 限制最大点数
    if curve_pts.shape[0] > max_points:
        stride = int(np.ceil(curve_pts.shape[0] / max_points))
        curve_pts = curve_pts[::stride]
        curve_tan = curve_tan[::stride]
        curve_n1 = curve_n1[::stride]
        curve_n2 = curve_n2[::stride]

    np.save(os.path.join(output_dir, "sharp_curve_points_raw.npy"), curve_pts)
    np.save(os.path.join(output_dir, "sharp_curve_tangents.npy"), curve_tan)
    np.save(os.path.join(output_dir, "sharp_curve_n1.npy"), curve_n1)
    np.save(os.path.join(output_dir, "sharp_curve_n2.npy"), curve_n2)

    meta_path = os.path.join(output_dir, "sharp_curve_meta.json")
    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f) or {}
        except Exception:
            meta = {}

    meta.update({
        "input_path": input_path,
        "sharp_Lmin": float(sharp_Lmin),
        "tol_ratio_default": float(tol_ratio),
        "tol_geom_default": float(tol_geom),
        "curve_sample_step": float(sample_step),
        "num_curve_points": int(curve_pts.shape[0]),
        "sharp_edges_count": int(len(sharp_edges)),
        "note": "B1 requires sharp_Lmin to derive tol_geom; epsilon is derived later in main_3."
    })
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def build_cfpu_input(input_path, output_dir, angle_threshold=30.0, r_small_factor=0.5, r_large_factor=3.0, edge_split_threshold=None, require_step_face_id_diff=False):
    """
    构建CFPU输入数据，根据用户需求：
    1. 不补充额外节点，只使用原始三角网格顶点
    2. 平滑区域使用泊松盘采样
    3. 尖锐边的每个点都是patch
    4. 尖锐边法向量分离，使用1e-5微小距离偏移
    5. 半径重新划分：拓扑自适应初值 + 兜底
    """
    mesh = read_mesh(input_path, compute_split_normals=False)
    points = mesh.points
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    mapped_labels, cell_normals, edge_to_faces, inv = face_segmentation(mesh, angle_threshold)
    avg_len = _avg_edge_length(points, faces)
    
    # 1. 检测尖锐边和尖锐顶点
    sharp_edges, _lines = detect_sharp_edges(mesh, angle_threshold=angle_threshold, edge_split_threshold=edge_split_threshold, require_step_face_id_diff=require_step_face_id_diff)
    
    # 收集尖锐顶点ID
    sharp_vertex_ids = set()
    for e in sharp_edges:
        sharp_vertex_ids.add(int(e['point1_idx']))
        sharp_vertex_ids.add(int(e['point2_idx']))
    sharp_vertex_ids = sorted(list(sharp_vertex_ids))
    
    # 顶点 -> incident sharp edge records（用于凸/凹判断投票）
    pid_to_incident_edges = collections.defaultdict(list)
    for e in sharp_edges:
        a = int(e['point1_idx'])
        b = int(e['point2_idx'])
        pid_to_incident_edges[a].append(e)
        pid_to_incident_edges[b].append(e)

    # 2. 构建尖锐边的法向量分离（使用1e-5微小距离偏移）
    edge_info = {}
    for e in sharp_edges:
        a = int(e['point1_idx'])
        b = int(e['point2_idx'])
        k = (a, b) if a < b else (b, a)
        edge_info[k] = e
    
    # 3. 准备节点数据 - 不补充额外节点，只使用原始点
    nodes = points
    
    # 4. 计算每个节点的法向量
    # 对于尖锐顶点，使用相邻面法向量的平均值
    # 对于平滑顶点，使用相邻面法向量的平均值
    p2f = _point_to_faces_map(faces)
    normals = np.zeros_like(points)
    for pid in range(points.shape[0]):
        fl = p2f.get(pid, [])
        if fl:
            # 计算相邻面法向量的平均值
            avg_normal = np.mean(cell_normals[fl], axis=0)
            normals[pid] = avg_normal / np.linalg.norm(avg_normal) if np.linalg.norm(avg_normal) > 1e-9 else np.array([0, 0, 1])
        else:
            normals[pid] = np.array([0, 0, 1])
    
    # # 5. 尖锐边法向量分离：处理具有多个法向量的节点
    # # 5.1 识别具有多个法向量的尖锐边节点
    # # 只有尖锐边上的节点才需要拆分（根据用户要求）
    # nodes_with_multiple_normals = set()
    
    # # 5.2 收集每个尖锐边节点的所有法向量
    # node_to_normals = {}
    # for pid in sharp_vertex_ids:  # 只处理尖锐边上的节点
    #     fl = p2f.get(pid, [])
    #     if len(fl) > 1:
    #         # 获取相邻面的法向量
    #         face_normals = cell_normals[fl]
            
    #         # 去重法向量（使用小容差）
    #         unique_normals = []
    #         for fn in face_normals:
    #             is_unique = True
    #             for un in unique_normals:
    #                 if np.linalg.norm(fn - un) < 1e-5:
    #                     is_unique = False
    #                     break
    #             if is_unique:
    #                 unique_normals.append(fn)
            
    #         if len(unique_normals) >= 2:
    #             # 这个尖锐边节点具有多个法向量，需要处理
    #             nodes_with_multiple_normals.add(pid)
    #             node_to_normals[pid] = unique_normals
    
    # print(f"检测到具有多个法向量的尖锐边节点数量: {len(nodes_with_multiple_normals)}")
    
    # # 5.3 面内偏移拆分：对具有多个法向量的节点，按每个所属面片生成一个偏移节点
    # new_nodes = []
    # new_normals = []
    
    # # 计算面片质心
    # face_centroids = []
    # for fid in range(faces.shape[0]):
    #     tri = faces[fid]
    #     centroid = np.mean(points[tri], axis=0)
    #     face_centroids.append(centroid)
    # face_centroids = np.array(face_centroids)
    
    # # 计算偏移比例（根据模型尺寸自适应）
    # bbox_diag = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    # abs_inset = 1e-5      # 你期望的固定偏移（单位=模型单位，比如 m）
    # rel_inset = 0.002     # 你说的小模型用的比例（0.2% 对角线）
    # inset_dist = min(abs_inset, rel_inset * bbox_diag)
    
    # # 5.4 生成新的偏移节点
    # for pid in range(points.shape[0]):
    #     if pid in nodes_with_multiple_normals:
    #         # 这个节点具有多个法向量，需要按每个面片生成偏移节点
    #         v = points[pid]
    #         fl = p2f.get(pid, [])
            
    #         for fid in fl:
    #             # 获取该节点在当前面片中的位置
    #             tri = faces[fid]
    #             li = int(np.where(tri == pid)[0][0])  # 局部顶点索引
                
    #             # 计算从顶点向质心的偏移
    #             c = face_centroids[fid]
    #             d = c - v
    #             L = np.linalg.norm(d)
    #             if L < 1e-12:
    #                 continue
                
    #             # 计算偏移系数，确保偏移距离合理且新节点在三角形内部
    #             alpha = min(0.33, inset_dist / L)
    #             p_in = v + alpha * d  # 凸组合，确保在三角形内部
                
    #             # 获取当前面片的法向量作为新节点的法向量
    #             n = cell_normals[fid]
                
    #             # 添加到新节点和法向量列表
    #             new_nodes.append(p_in)
    #             new_normals.append(n)
    #     else:
    #         # 这个节点只有一个法向量，直接使用原始节点
    #         new_nodes.append(points[pid])
    #         new_normals.append(normals[pid])
    
    # # 转换为numpy数组
    # new_nodes = np.array(new_nodes)
    # new_normals = np.array(new_normals)

    
#     # 5. 尖锐边法向量分离（方案1）：只在尖锐边点上拆分，且按“尖锐边切断的一环面连通分量(面簇)”拆分
#     #    —— 每个面簇生成 1 个节点 + 1 个法向（不再按每个面生成节点）

#     # 5.0 准备：尖锐边集合（用几何点id构成key，避免 split 后点id不一致）
#     sharp_geo_edges = set()
#     for e in sharp_edges:
#         gA = int(e.get('geo_point1_idx', inv[int(e['point1_idx'])]))
#         gB = int(e.get('geo_point2_idx', inv[int(e['point2_idx'])]))
#         if gA == gB:
#             continue
#         sharp_geo_edges.add((gA, gB) if gA < gB else (gB, gA))

#     # 5.0.1 工具函数
#     def _normalize(v):
#         n = float(np.linalg.norm(v))
#         return v / max(n, 1e-12)

#     def _tri_area(fid):
#         a, b, c = faces[int(fid)]
#         v0, v1, v2 = points[int(a)], points[int(b)], points[int(c)]
#         return 0.5 * float(np.linalg.norm(np.cross(v1 - v0, v2 - v0)))

#     # 参考你给的 test_separate.py：将“其他法向合力”投影到当前法向切平面，取反方向作为分离方向
#     # （只用于决定偏移方向，不用于分组）
#     def _separate_dir_from_normals(n_target, n_others):
#         n1 = _normalize(n_target)
#         if not n_others:
#             return np.zeros(3, dtype=float)
#         n_sum = np.sum(np.asarray(n_others, dtype=float), axis=0)
#         v_proj = n_sum - float(np.dot(n_sum, n1)) * n1
#         if float(np.linalg.norm(v_proj)) < 1e-12:
#             return np.zeros(3, dtype=float)
#         return _normalize(-v_proj)

# # 如果你发现整体凸凹反了，把这个改成 -1 即可（无需改其它逻辑）
#     CONVEX_SIGN = -1

#     def _edge_is_convex_signed(e):
#         """
#         用有符号二面角的符号判断凸/凹（需要 mesh 的法向大体一致朝外）
#         返回 True=凸, False=凹
#         """
#         # 注意：detect_sharp_edges 里 edge key 来自 sorted(geo ids)，所以 point1_idx/point2_idx 的方向是稳定的
#         p1 = int(e['point1_idx'])
#         p2 = int(e['point2_idx'])

#         n1 = _normalize(cell_normals[int(e['face1'])])
#         n2 = _normalize(cell_normals[int(e['face2'])])

#         edge_dir = _normalize(points[p2] - points[p1])  # 稳定方向
#         s = float(np.dot(edge_dir, np.cross(n1, n2))) * CONVEX_SIGN

#         # s 的符号区分凸/凹（若反了就把 CONVEX_SIGN 取 -1）
#         return s > 0.0

#     # 5.1 计算偏移距离（你现在的“固定优先，小模型比例”的规则保留）
#     bbox_diag = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
#     abs_inset = 1e-4
#     rel_inset = 1e-4
#     inset_dist = min(abs_inset, rel_inset * bbox_diag)
#     print(f"偏移距离: {inset_dist}")

#     # 5.2 预计算面片质心（用于兜底方向/轻微向内移动）
#     face_centroids = np.mean(points[faces], axis=1)  # (n_faces, 3)

#     # 5.3 对每个“尖锐边上的点 pid”，用尖锐边切断一环面，得到面簇（连通分量）
#     #     两个面在 pid 处可连通 <=> 它们共享的边 (pid, q) 不是尖锐边
#     from collections import defaultdict

#     pid_to_face_groups = {}     # pid -> [ [fid...], [fid...], ... ]
#     pid_to_group_normals = {}   # pid -> [n0, n1, ...]
#     split_pids = set()

#     for pid in sharp_vertex_ids:
#         fl = p2f.get(pid, [])
#         if len(fl) <= 1:
#             continue

#         gpid = int(inv[int(pid)])

#         # 建 edge -> faces 映射（只看经过 pid 的两条边）
#         edge2faces = defaultdict(list)
#         for fid in fl:
#             tri = faces[int(fid)]
#             # tri 中除 pid 外的两个点
#             for q in tri:
#                 q = int(q)
#                 if q == int(pid):
#                     continue
#                 gq = int(inv[q])
#                 key = (gpid, gq) if gpid < gq else (gq, gpid)
#                 edge2faces[key].append(int(fid))

#         # union-find 在 fl 上做连通分量
#         parent = {int(fid): int(fid) for fid in fl}

#         def find(x):
#             while parent[x] != x:
#                 parent[x] = parent[parent[x]]
#                 x = parent[x]
#             return x

#         def union(a, b):
#             ra, rb = find(a), find(b)
#             if ra != rb:
#                 parent[ra] = rb

#         # 共享同一条“非尖锐边”的 face union 在一起
#         for edge_key, fids in edge2faces.items():
#             if edge_key in sharp_geo_edges:
#                 continue  # 尖锐边：切断
#             if len(fids) >= 2:
#                 base = fids[0]
#                 for other in fids[1:]:
#                     union(base, other)

#         groups = defaultdict(list)
#         for fid in fl:
#             groups[find(int(fid))].append(int(fid))

#         # 只有真的被“尖锐边切开”成 >=2 个面簇才拆分
#         if len(groups) >= 2:
#             split_pids.add(int(pid))
#             face_groups = list(groups.values())
#             pid_to_face_groups[int(pid)] = face_groups

#             # 每个面簇算一个“簇法向”（建议面积加权）
#             g_normals = []
#             for g in face_groups:
#                 nsum = np.zeros(3, dtype=float)
#                 wsum = 0.0
#                 for fid in g:
#                     w = _tri_area(fid)
#                     nsum += w * cell_normals[int(fid)]
#                     wsum += w
#                 if wsum < 1e-12:
#                     n = _normalize(np.mean(cell_normals[np.array(g, dtype=int)], axis=0))
#                 else:
#                     n = _normalize(nsum)
#                 g_normals.append(n)
#             pid_to_group_normals[int(pid)] = g_normals

#     print(f"检测到需要拆分的尖锐边节点数量(按面簇): {len(split_pids)}")

#     # 5.4 生成新的偏移节点：每个面簇 1 个节点（不再按每个面生成）
#     new_nodes = []
#     new_normals = []

#     for pid in range(points.shape[0]):
#         if pid not in split_pids:
#             new_nodes.append(points[pid])
#             new_normals.append(normals[pid])
#             continue

#         v = points[pid]
#         face_groups = pid_to_face_groups[pid]
#         group_normals = pid_to_group_normals[pid]
#         K = len(face_groups)
#         # face -> group index
#         face_to_group = {}
#         for gi, g in enumerate(face_groups):
#             for fid in g:
#                 face_to_group[int(fid)] = gi

#         # 每个 group 的凸/凹投票（只统计跨 group 的 incident sharp edges）
#         incident = pid_to_incident_edges.get(pid, [])
#         group_is_convex = [None] * K
#         for gi in range(K):
#             votes = []
#             for e in incident:
#                 f1 = int(e['face1'])
#                 f2 = int(e['face2'])
#                 if (f1 in face_to_group) and (f2 in face_to_group) and (face_to_group[f1] != face_to_group[f2]):
#                     if (face_to_group[f1] == gi) or (face_to_group[f2] == gi):
#                         votes.append(_edge_is_convex_signed(e))
#             if votes:
#                 group_is_convex[gi] = (sum(votes) >= (len(votes) / 2.0))  # 多数投票

#         for i in range(K):
#             n_i = group_normals[i]
#             others = [group_normals[j] for j in range(K) if j != i]

#             # 优先用“切平面投影分离方向”（来自你的参考逻辑）
#             d = _separate_dir_from_normals(n_i, others)

#             # 兜底：如果分离方向退化，用“指向本组面簇质心方向”并投影到切平面
#             if float(np.linalg.norm(d)) < 1e-12:
#                 cg = np.mean(face_centroids[np.array(face_groups[i], dtype=int)], axis=0)
#                 t = cg - v
#                 t = t - float(np.dot(t, n_i)) * n_i  # 投影到切平面
#                 if float(np.linalg.norm(t)) < 1e-12:
#                     d = np.array([1.0, 0.0, 0.0], dtype=float)  # 最后兜底
#                 else:
#                     d = _normalize(t)
#                 # ---------- 关键：凸边取负方向，凹边取正方向 ----------
#                 if group_is_convex[i] is True:
#                     d = -d  # 凸 => 负方向
#                 # group_is_convex[i] is False => 凹 => 正方向（不变）
#                 # None => 未判定，不处理

#                 # ---------- 关键守护：确保 d 指向该面簇“内部” ----------
#                 # 否则即便凸凹翻了，也可能跑出扇区导致“离面”
#                 cg = np.mean(face_centroids[np.array(face_groups[i], dtype=int)], axis=0)
#                 t_in = cg - v
#                 t_in = t_in - float(np.dot(t_in, n_i)) * n_i  # 投影到切平面
#                 if float(np.linalg.norm(t_in)) > 1e-12:
#                     t_in = _normalize(t_in)
#                     if float(np.dot(d, t_in)) < 0.0:
#                         d = -d

#             p_new = v + inset_dist * d
#             new_nodes.append(p_new)
#             new_normals.append(n_i)

#     new_nodes = np.asarray(new_nodes, dtype=float)
#     new_normals = np.asarray(new_normals, dtype=float)

    # 5. 尖锐边法向量分离（方案1）：只在尖锐边点上拆分，且按“尖锐边切断的一环面连通分量(面簇)”拆分
    #    —— 每个面簇生成 1 个节点 + 1 个法向（不再按每个面生成节点）
    #
    #    【本版本包含你要求的全部改动】
    #    1) 偏移距离：每个点 = 0.02 * (该点一环最短边)，并加上限 cap = 0.05 * avg_len
    #    2) 偏移方向：先用“切平面投影分离方向”，再用 凸/凹 (signed dihedral) 决定正负（仅 K==2 时）
    #    3) 最后再做一次“朝本面簇内部”的方向校正，避免偏移跑出扇区
    # 5.0 准备：尖锐边集合（用几何点id构成key，避免 split 后点id不一致）
    sharp_geo_edges = set()
    for e in sharp_edges:
        gA = int(e.get('geo_point1_idx', inv[int(e['point1_idx'])]))
        gB = int(e.get('geo_point2_idx', inv[int(e['point2_idx'])]))
        if gA == gB:
            continue
        sharp_geo_edges.add((gA, gB) if gA < gB else (gB, gA))

    # 5.0.0 顶点 -> incident sharp edge records（用于凸/凹投票）
    pid_to_incident_edges = collections.defaultdict(list)
    for e in sharp_edges:
        a = int(e['point1_idx'])
        b = int(e['point2_idx'])
        pid_to_incident_edges[a].append(e)
        pid_to_incident_edges[b].append(e)

    # 5.0.1 工具函数
    def _normalize(v):
        n = float(np.linalg.norm(v))
        return v / max(n, 1e-12)

    def _tri_area(fid):
        a, b, c = faces[int(fid)]
        v0, v1, v2 = points[int(a)], points[int(b)], points[int(c)]
        return 0.5 * float(np.linalg.norm(np.cross(v1 - v0, v2 - v0)))

    # 参考 test_separate.py：将“其他法向合力”投影到当前法向切平面，取反方向作为分离方向
    def _separate_dir_from_normals(n_target, n_others):
        n1 = _normalize(n_target)
        if not n_others:
            return np.zeros(3, dtype=float)
        n_sum = np.sum(np.asarray(n_others, dtype=float), axis=0)
        v_proj = n_sum - float(np.dot(n_sum, n1)) * n1
        if float(np.linalg.norm(v_proj)) < 1e-12:
            return np.zeros(3, dtype=float)
        return _normalize(-v_proj)

    # --- 凸/凹：用 signed dihedral 的符号判定（注意：若整体反了，把 CONVEX_SIGN 改成 -1 即可） ---
    CONVEX_SIGN = +1  # 若你发现“凸/凹翻了”，只改这里为 -1

    def _edge_is_convex_signed(e):
        """
        返回 True=凸, False=凹
        使用：s = dot(edge_dir, cross(n1, n2))
        """
        p1 = int(e['point1_idx'])
        p2 = int(e['point2_idx'])
        f1 = int(e['face1'])
        f2 = int(e['face2'])

        n1 = _normalize(cell_normals[f1])
        n2 = _normalize(cell_normals[f2])
        edge_dir = _normalize(points[p2] - points[p1])  # 稳定方向

        s = float(np.dot(edge_dir, np.cross(n1, n2))) * CONVEX_SIGN
        return s > 0.0

    # 5.1 偏移距离：每点 = 2% * (一环最短边)，上限 cap = 0.05 * avg_len
    min_edge_len = np.full(points.shape[0], np.inf, dtype=float)
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])

        lab = float(np.linalg.norm(points[a] - points[b]))
        lbc = float(np.linalg.norm(points[b] - points[c]))
        lca = float(np.linalg.norm(points[c] - points[a]))

        if lab < min_edge_len[a]: min_edge_len[a] = lab
        if lab < min_edge_len[b]: min_edge_len[b] = lab
        if lbc < min_edge_len[b]: min_edge_len[b] = lbc
        if lbc < min_edge_len[c]: min_edge_len[c] = lbc
        if lca < min_edge_len[c]: min_edge_len[c] = lca
        if lca < min_edge_len[a]: min_edge_len[a] = lca

    # 兜底（理论上不会用到）
    min_edge_len[~np.isfinite(min_edge_len)] = float(avg_len)

    inset_ratio = 0.02
    cap = 0.05 * float(avg_len)
    inset_dist_vtx = inset_ratio * min_edge_len
    inset_dist_vtx = np.minimum(inset_dist_vtx, cap)
    inset_dist_vtx = np.maximum(inset_dist_vtx, 1e-12)

    print(
        f"偏移距离(2%最短边, cap=0.05*avg_len): "
        f"min={inset_dist_vtx.min():.3e}, median={np.median(inset_dist_vtx):.3e}, "
        f"max={inset_dist_vtx.max():.3e}, cap={cap:.3e}"
    )

    # 5.2 预计算面片质心（用于兜底方向/向面簇内部校正）
    face_centroids = np.mean(points[faces], axis=1)  # (n_faces, 3)

    # 5.3 对每个“尖锐边上的点 pid”，用尖锐边切断一环面，得到面簇（连通分量）
    #     两个面在 pid 处可连通 <=> 它们共享的边 (pid, q) 不是尖锐边
    pid_to_face_groups = {}     # pid -> [ [fid...], [fid...], ... ]
    pid_to_group_normals = {}   # pid -> [n0, n1, ...]
    split_pids = set()

    for pid in sharp_vertex_ids:
        fl = p2f.get(pid, [])
        if len(fl) <= 1:
            continue

        gpid = int(inv[int(pid)])

        # 建 edge -> faces 映射（只看经过 pid 的两条边）
        edge2faces = collections.defaultdict(list)
        for fid in fl:
            tri = faces[int(fid)]
            for q in tri:
                q = int(q)
                if q == int(pid):
                    continue
                gq = int(inv[q])
                key = (gpid, gq) if gpid < gq else (gq, gpid)
                edge2faces[key].append(int(fid))

        # union-find 在 fl 上做连通分量
        parent = {int(fid): int(fid) for fid in fl}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # 共享同一条“非尖锐边”的 face union 在一起
        for edge_key, fids in edge2faces.items():
            if edge_key in sharp_geo_edges:
                continue  # 尖锐边：切断
            if len(fids) >= 2:
                base = fids[0]
                for other in fids[1:]:
                    union(base, other)

        groups = collections.defaultdict(list)
        for fid in fl:
            groups[find(int(fid))].append(int(fid))

        # 只有真的被“尖锐边切开”成 >=2 个面簇才拆分
        if len(groups) >= 2:
            split_pids.add(int(pid))
            face_groups = list(groups.values())
            pid_to_face_groups[int(pid)] = face_groups

            # 每个面簇算一个“簇法向”（面积加权）
            g_normals = []
            for g in face_groups:
                nsum = np.zeros(3, dtype=float)
                wsum = 0.0
                for fid in g:
                    w = _tri_area(fid)
                    nsum += w * cell_normals[int(fid)]
                    wsum += w
                if wsum < 1e-12:
                    n = _normalize(np.mean(cell_normals[np.array(g, dtype=int)], axis=0))
                else:
                    n = _normalize(nsum)
                g_normals.append(n)
            pid_to_group_normals[int(pid)] = g_normals

    print(f"检测到需要拆分的尖锐边节点数量(按面簇): {len(split_pids)}")

    # 5.4 生成新的偏移节点：每个面簇 1 个节点（不再按每个面生成）
    new_nodes = []
    new_normals = []

    for pid in range(points.shape[0]):
        if pid not in split_pids:
            new_nodes.append(points[pid])
            new_normals.append(normals[pid])
            continue

        v = points[pid]
        inset_dist = float(inset_dist_vtx[pid])  # ★每点偏移距离
        face_groups = pid_to_face_groups[pid]
        group_normals = pid_to_group_normals[pid]
        K = len(face_groups)

        # face -> group index（用于凸/凹投票筛选跨组边）
        face_to_group = {}
        for gi, g in enumerate(face_groups):
            for fid in g:
                face_to_group[int(fid)] = gi

        # 仅在 K==2 的“普通尖锐边”上判凸/凹并决定正负；junction(K>2) 跳过凸凹翻转
        group_is_convex = [None] * K
        if K == 2:
            incident = pid_to_incident_edges.get(pid, [])
            for gi in range(K):
                votes = []
                for e in incident:
                    f1 = int(e['face1'])
                    f2 = int(e['face2'])
                    if (f1 in face_to_group) and (f2 in face_to_group) and (face_to_group[f1] != face_to_group[f2]):
                        if (face_to_group[f1] == gi) or (face_to_group[f2] == gi):
                            votes.append(_edge_is_convex_signed(e))
                if votes:
                    group_is_convex[gi] = (sum(votes) >= (len(votes) / 2.0))

        for i in range(K):
            n_i = group_normals[i]
            others = [group_normals[j] for j in range(K) if j != i]

            # 1) 主方向：切平面投影分离方向
            d = _separate_dir_from_normals(n_i, others)

            # 2) 兜底：若退化，用“指向本组面簇质心方向”并投影到切平面
            if float(np.linalg.norm(d)) < 1e-12:
                cg = np.mean(face_centroids[np.array(face_groups[i], dtype=int)], axis=0)
                t = cg - v
                t = t - float(np.dot(t, n_i)) * n_i
                if float(np.linalg.norm(t)) < 1e-12:
                    d = np.array([1.0, 0.0, 0.0], dtype=float)
                else:
                    d = _normalize(t)

            # 3) 凸/凹决定正负（仅 K==2）：凸边取负方向，凹边取正方向
            if (K == 2) and (group_is_convex[i] is True):
                d = -d

            # 4) 关键守护：确保 d 指向本面簇“内部”（避免偏移跑出扇区）
            cg = np.mean(face_centroids[np.array(face_groups[i], dtype=int)], axis=0)
            t_in = cg - v
            t_in = t_in - float(np.dot(t_in, n_i)) * n_i
            if float(np.linalg.norm(t_in)) > 1e-12:
                t_in = _normalize(t_in)
                if float(np.dot(d, t_in)) < 0.0:
                    d = -d

            p_new = v + inset_dist * d
            new_nodes.append(p_new)
            new_normals.append(n_i)

    new_nodes = np.asarray(new_nodes, dtype=float)
    new_normals = np.asarray(new_normals, dtype=float)


    print(f"原始节点数量: {points.shape[0]}")
    print(f"新节点数量: {new_nodes.shape[0]}")

    # 6. 更新nodes和normals为新生成的节点和法向量
    nodes = new_nodes
    normals = new_normals
    
    # 7. 准备patch中心
    # 尖锐边的每个点都是patch
    sharp_centers = points[sharp_vertex_ids] if sharp_vertex_ids else np.empty((0, 3))
    
    # 平滑区域使用泊松盘采样
    all_point_ids = set(range(points.shape[0]))
    smooth_ids = sorted(list(all_point_ids - set(sharp_vertex_ids)))
    smooth_points = points[smooth_ids] if smooth_ids else np.empty((0, 3))
    
    # 使用泊松盘采样平滑区域
    poisson_radius = 2.0 * avg_len  # 泊松盘采样半径
    sampled_smooth_ids = []
    if smooth_points.shape[0] > 0:
        tree = None
        for idx in smooth_ids:
            p = points[idx]
            if not sampled_smooth_ids:
                sampled_smooth_ids.append(idx)
                tree = cKDTree(points[sampled_smooth_ids])
                continue
            if tree.query(p, k=1)[0] > poisson_radius:
                sampled_smooth_ids.append(idx)
                tree = cKDTree(points[sampled_smooth_ids])
    
    smooth_centers = points[sampled_smooth_ids] if sampled_smooth_ids else np.empty((0, 3))
    
    # 合并所有patch中心
    patches = np.concatenate([sharp_centers, smooth_centers], axis=0) if (sharp_centers.size or smooth_centers.size) else np.empty((0, 3))
    feature_count = sharp_centers.shape[0]
    
    # 8. 半径重新划分：拓扑自适应初值 + 兜底
    radii = np.zeros(patches.shape[0], dtype=float)
    if patches.shape[0] > 0:
        delta = 0.6  # 经验值
        n_min = 40   # 工程建议的n_min值
        k_neighbor = 25  # 经验值，20~30
        
        # 构建节点KD树用于后续查询
        tree_nodes = cKDTree(nodes)
        
        # 8.1 Sharp patch 半径初值：拓扑规则
        if sharp_centers.shape[0] > 0:
            # 构建尖锐顶点的邻接表
            adj = {}
            for e in sharp_edges:
                a = int(e['point1_idx'])
                b = int(e['point2_idx'])
                adj.setdefault(a, []).append(b)
                adj.setdefault(b, []).append(a)
            
            # 计算每个尖锐patch的半径
            for i, v in enumerate(sharp_vertex_ids):
                nbrs = adj.get(v, [])
                if nbrs:
                    # 计算到所有相邻尖锐顶点的距离
                    dists = [np.linalg.norm(points[v] - points[n]) for n in nbrs]
                    # Sharp patch 初值：max(dist) * (1+delta)
                    radii[i] = max(dists) * (1.0 + delta)
                else:
                    radii[i] = avg_len * (1.0 + delta)
        
        # 8.2 Smooth patch 半径初值：拓扑规则
        if smooth_centers.shape[0] > 0:
            for i in range(smooth_centers.shape[0]):
                gi = feature_count + i
                center = smooth_centers[i]
                
                # 计算到k-th neighbor的距离
                dists, _ = tree_nodes.query(center, k=min(k_neighbor, nodes.shape[0]))
                if len(dists) > 0:
                    # Smooth patch 初值：dist(k-th neighbor) * (1+delta)
                    radii[gi] = dists[-1] * (1.0 + delta)
                else:
                    radii[gi] = avg_len * (1.0 + delta)
        
        # 8.3 兜底 1：不足n_min就扩半径
        for i in range(patches.shape[0]):
            center = patches[i]
            current_radius = radii[i]
            
            # 统计当前半径内的节点数量
            node_count = len(tree_nodes.query_ball_point(center, current_radius))
            
            # 如果节点数量不足n_min，持续增大半径
            while node_count < n_min:
                current_radius *= 1.2  # 每次增大20%
                node_count = len(tree_nodes.query_ball_point(center, current_radius))
                # 防止无限循环
                if current_radius > 100 * avg_len:
                    break
            
            radii[i] = current_radius
        
        # 8.4 兜底 2：漏点补覆盖
        node_in_patch = np.zeros(nodes.shape[0], dtype=bool)
        for i in range(patches.shape[0]):
            center = patches[i]
            radius = radii[i]
            id_list = tree_nodes.query_ball_point(center, radius)
            node_in_patch[id_list] = True
        
        # 检查是否有漏点
        missing_ids = np.where(~node_in_patch)[0]
        
        # 创建基于patches数组的KD树，用于查找最近的补丁中心
        patches_tree = cKDTree(patches)
        
        while missing_ids.size > 0:
            # 找到最近的patch center
            # 使用patches_tree而不是tree_nodes来查找最近的补丁中心
            cp_dist, cp_id = patches_tree.query(nodes[missing_ids[0], :], k=1)
            
            # 确保cp_id在patches数组的范围内
            if cp_id < patches.shape[0]:
                # 计算距离
                p_dist = np.linalg.norm(nodes[missing_ids[0], :] - patches[cp_id, :])
                # 增大半径以覆盖漏点
                radii[cp_id] = max(radii[cp_id], 1.01 * p_dist)
                # 更新覆盖情况
                id_list = tree_nodes.query_ball_point(patches[cp_id, :], radii[cp_id])
                node_in_patch[id_list] = True
                # 重新检查漏点
                missing_ids = np.where(~node_in_patch)[0]
            else:
                # 这是一个异常情况，跳过这个漏点
                print(f"警告：找到的补丁中心索引 {cp_id} 超出范围，跳过这个漏点")
                missing_ids = missing_ids[1:]  # 跳过当前漏点，检查下一个
    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(os.path.join(output_dir, 'nodes.txt'), nodes)
    np.savetxt(os.path.join(output_dir, 'normals.txt'), normals)
    np.savetxt(os.path.join(output_dir, 'patches.txt'), patches)
    np.savetxt(os.path.join(output_dir, 'radii.txt'), radii)
    try:
        with open(os.path.join(output_dir, 'feature_count.txt'), 'w') as f:
            f.write(str(feature_count))
    except Exception:
        pass

    # --- B1 bump 修正：导出尖锐边采样（不影响原有流程） ---
    try:
        export_sharp_curve_for_b1(
            output_dir=output_dir,
            input_path=input_path,
            points=points,
            faces=faces,
            cell_normals=cell_normals,
            sharp_edges=sharp_edges,
            sample_step=None,      # 默认 0.25*sharp_Lmin
            tol_ratio=0.01         # 你的误差指标：1% Lmin（可在 main_3 用 --b1_tol_ratio 覆盖）
        )
    except Exception as _e:
        print(f"[B1] export_sharp_curve_for_b1 failed: {_e}")

    return nodes, normals, patches


# =============================================================================
# segment.py
# =============================================================================
def segment_mesh(input_path, output_path, angle_threshold=30.0, edge_split_threshold=None, require_step_face_id_diff=False, angle_turn_threshold=90.0):
    """
    分割网格并生成相关信息，包括区域邻接关系、尖锐边缘和间断点
    
    参数：
        input_path: 输入网格文件路径
        output_path: 输出分割后网格文件路径
        angle_threshold: 尖锐边缘检测的角度阈值
        edge_split_threshold: 边缘分割的角度阈值
        require_step_face_id_diff: 是否需要step_face_id差异
        angle_turn_threshold: 间断点角度阈值
    
    返回：
        adj: 区域邻接关系字典
    """
    mesh = read_mesh(input_path, compute_split_normals=False)
    mapped_labels, cell_normals, edge_to_faces, inv = face_segmentation(mesh, angle_threshold, edge_split_threshold, require_step_face_id_diff)
    mesh.cell_data['RegionId'] = mapped_labels
    base_name = output_path.rsplit('.', 1)[0]
    
    # 保存区域邻接关系
    adj = classify_adjacency(mapped_labels, cell_normals, edge_to_faces)
    mesh.save(output_path)
    
    # 检测尖锐边缘
    sharp_edges, sharp_edge_lines = detect_sharp_edges(mesh, angle_threshold, edge_split_threshold, require_step_face_id_diff)
    
    # 检测尖锐连接点和生成间断点信息
    junctions = detect_sharp_junctions_degree(mesh, sharp_edges)
    segments = build_sharp_segments(sharp_edges, junctions, mesh.points, cell_normals, angle_turn_threshold)
    
    # 计算间断点
    turn_points = set()
    for seg in segments:
        turn_points.update(seg['turn_splits'])
    
    return adj

if __name__ == '__main__':
    print("precompute_single.py - 单文件整合版")
    print("包含模块: kdtree, mesh_io, topology, cfpu_input, segment")
    print("可用函数示例: read_mesh, face_segmentation, detect_sharp_edges, segment_mesh, build_cfpu_input")
