import os
import glob
import json
import numpy as np
import pyvista as pv

def load_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    points = np.array(data['points'])
    faces = np.array(data['faces']) if 'faces' in data else None
    normals = None
    if 'vertex_normals' in data:
        normals = np.array(data['vertex_normals'])
    elif 'corner_normals' in data and faces is not None:
        cn = np.array(data['corner_normals'])
        v_normals = np.zeros_like(points)
        counts = np.zeros(len(points))
        flat_faces = faces.flatten()
        flat_cn = cn.reshape(-1, 3)
        np.add.at(v_normals, flat_faces, flat_cn)
        np.add.at(counts, flat_faces, 1)
        counts[counts==0] = 1
        v_normals /= counts[:, None]
        norms = np.linalg.norm(v_normals, axis=1)
        norms[norms==0] = 1
        v_normals /= norms[:, None]
        normals = v_normals
    return points, normals, faces

def faces_to_pv(faces):
    return np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces.astype(np.int64)]).ravel()

def find_connected_regions(regions_data, tolerance=1e-6):
    connected_pairs = set()
    region_count = len(regions_data)
    
    # For each pair of regions
    for i in range(region_count):
        for j in range(i + 1, region_count):
            region1_pts = regions_data[i]['points']
            region2_pts = regions_data[j]['points']
            
            # Convert to numpy arrays for efficient computation
            pts1 = np.array(region1_pts)
            pts2 = np.array(region2_pts)
            
            # Compute all pairwise distances between points of the two regions
            # This is O(n*m) but for reasonable region sizes it's manageable
            dists = np.sqrt(np.sum((pts1[:, np.newaxis] - pts2[np.newaxis, :])**2, axis=2))
            
            # Check if any pair of points is close enough to be considered connected
            if np.min(dists) < tolerance:
                rid1 = regions_data[i]['id']
                rid2 = regions_data[j]['id']
                connected_pairs.add((min(rid1, rid2), max(rid1, rid2)))
    
    return sorted(list(connected_pairs))

def main():
    base_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data', 'output', 'leftGear_surface_cellnormals_regions'))
    files = sorted(glob.glob(os.path.join(base_dir, 'region_*.json')))
    
    # Load all regions data first for connectivity analysis
    regions_data = []
    for fp in files:
        pts, norms, faces = load_data(fp)
        rid = int(os.path.splitext(os.path.basename(fp))[0].split('_')[1])
        regions_data.append({
            'id': rid,
            'points': pts,
            'normals': norms,
            'faces': faces,
            'file_path': fp
        })
    
    # Find connected regions
    connected_regions = find_connected_regions(regions_data)
    print(f"Connected region pairs:")
    for pair in connected_regions:
        print(f"  Region {pair[0]} ↔ Region {pair[1]}")
    
    # Visualization part remains mostly the same
    plotter = pv.Plotter()
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
    legend = []
    actors = []
    region_ids = [rd['id'] for rd in regions_data]
    
    for i, rd in enumerate(regions_data):
        pts, norms, faces = rd['points'], rd['normals'], rd['faces']
        color = colors[i % len(colors)]
        if faces is not None:
            pv_faces = faces_to_pv(faces)
            mesh = pv.PolyData(pts, pv_faces)
            actor = plotter.add_mesh(mesh, color=color, opacity=0.2)
        else:
            actor = plotter.add_points(pts, color=color, point_size=2, render_points_as_spheres=False, opacity=0.4)
        actors.append(actor)
        legend.append((f"Region {rd['id']}", color))
    
    if legend:
        plotter.add_legend(labels=legend)

    if len(actors) == 0:
        plotter.show()
        return

    cur_idx = 0
    text_actor = plotter.add_text(f"Region {region_ids[cur_idx]}", position='upper_left', font_size=16, color='black')
    # Add connected regions info to text
    connected_info = "Connected: None"
    if connected_regions:
        connected_to_cur = [str(pair[1]) if pair[0] == region_ids[cur_idx] else str(pair[0]) for pair in connected_regions if region_ids[cur_idx] in pair]
        if connected_to_cur:
            connected_info = f"Connected: {', '.join(connected_to_cur)}"
    text_actor2 = plotter.add_text(connected_info, position='upper_right', font_size=14, color='black')

    def highlight(idx):
        for j, a in enumerate(actors):
            if j == idx:
                a.prop.opacity = 0.95
            else:
                a.prop.opacity = 0.15
        text_actor.SetText(2, f"Region {region_ids[idx]}")
        
        # Update connected regions info
        connected_to_cur = [str(pair[1]) if pair[0] == region_ids[idx] else str(pair[0]) for pair in connected_regions if region_ids[idx] in pair]
        if connected_to_cur:
            connected_info = f"Connected: {', '.join(connected_to_cur)}"
        else:
            connected_info = "Connected: None"
        text_actor2.SetText(2, connected_info)

    def on_right():
        nonlocal cur_idx
        cur_idx = (cur_idx + 1) % len(actors)
        highlight(cur_idx)

    def on_left():
        nonlocal cur_idx
        cur_idx = (cur_idx - 1) % len(actors)
        highlight(cur_idx)

    highlight(cur_idx)
    plotter.add_key_event('Right', on_right)
    plotter.add_key_event('Left', on_left)
    plotter.show()

if __name__ == '__main__':
    main()
