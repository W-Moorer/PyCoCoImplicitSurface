import argparse
import os
import numpy as np
import pyvista as pv

def load_cfpu(path):
    nodes = np.loadtxt(os.path.join(path, 'nodes.txt'))
    normals = np.loadtxt(os.path.join(path, 'normals.txt'))
    patches = np.loadtxt(os.path.join(path, 'patches.txt'))
    radii_path = os.path.join(path, 'radii.txt')
    radii = None
    if os.path.exists(radii_path):
        try:
            radii = np.loadtxt(radii_path)
        except Exception:
            radii = None
    feature_count = None
    fc_path = os.path.join(path, 'feature_count.txt')
    if os.path.exists(fc_path):
        try:
            with open(fc_path, 'r') as f:
                feature_count = int(f.read().strip())
        except Exception:
            feature_count = None
    if nodes.ndim == 1:
        nodes = nodes.reshape(1, -1)
    if normals.ndim == 1:
        normals = normals.reshape(1, -1)
    if patches.ndim == 1:
        patches = patches.reshape(1, -1)
    if radii is not None and getattr(radii, 'ndim', 1) == 0:
        radii = radii.reshape(1)
    return nodes, normals, patches, radii, feature_count

def load_sharp_samples(path):
    jf = os.path.join(path, 'sharp_patch_samples.json')
    if not os.path.exists(jf):
        return {}
    import json
    with open(jf, 'r') as f:
        try:
            m = json.load(f)
        except Exception:
            m = {}
    # keys may be strings, normalize to int
    out = {}
    for k, v in m.items():
        try:
            ki = int(k)
        except Exception:
            continue
        out[ki] = list(map(int, v))
    return out

def _add_normals_glyphs(p, nodes, normals, arrow_factor):
    if normals.shape[0] == nodes.shape[0] and normals.ndim == 2 and normals.shape[1] == 3:
        pts_all = pv.PolyData(nodes)
        pts_all['Normals'] = normals
        arrows = pts_all.glyph(orient='Normals', scale=False, factor=arrow_factor, geom=pv.Arrow())
        p.add_mesh(arrows, color='yellow')

def build_scene(nodes, normals, patches, radii, point_size, patch_size, arrow_factor, show_normals, viewrad=False, off_screen=False, sphere_opacity=0.15, show_patches=True):
    p = pv.Plotter(off_screen=off_screen)
    pts_all = pv.PolyData(nodes)
    p.add_points(pts_all, color='white', point_size=point_size, render_points_as_spheres=True, opacity=0.35)
    if show_patches and patches.size > 0:
        p.add_points(patches, color='red', point_size=patch_size, render_points_as_spheres=True)
    if show_normals:
        _add_normals_glyphs(p, nodes, normals, arrow_factor)
    if viewrad and radii is not None and radii.shape[0] == patches.shape[0]:
        patch_poly = pv.PolyData(patches)
        patch_poly['rad'] = radii
        base_sphere = pv.Sphere(radius=1.0, phi_resolution=16, theta_resolution=16)
        glyph = patch_poly.glyph(scale='rad', geom=base_sphere, orient=False)
        p.add_mesh(glyph, color='magenta', style='wireframe', opacity=sphere_opacity)
    p.add_axes()
    p.enable_eye_dome_lighting()
    p.background_color = 'white'
    return p

def add_patch_samples_ui(p, nodes, normals, patches, radii, feature_count, samples_map, init_patch_id=0, point_size=12, color_samples='cyan', color_center='red', sphere_opacity=0.15, show_highlight_normals=True, arrow_factor=0.0001):
    from scipy.spatial import cKDTree
    actors = {'samples': None, 'center': None, 'sphere': None, 'text': None, 'highlight': None, 'highlight_normals': None}
    max_pid = feature_count - 1 if feature_count is not None else patches.shape[0] - 1
    def _update(pid):
        pid = max(0, min(pid, max_pid))
        # remove previous
        for k in ('samples','center','sphere','text','highlight','highlight_normals'):
            a = actors.get(k)
            if a is not None:
                try:
                    p.remove_actor(a)
                except Exception:
                    pass
                actors[k] = None
        center = patches[pid]
        actors['center'] = p.add_points(np.asarray([center]), color=color_center, point_size=point_size+6, render_points_as_spheres=True)
        # radius sphere
        if radii is not None and radii.shape[0] == patches.shape[0]:
            r0 = float(radii[pid])
            if r0 > 0:
                s = pv.Sphere(radius=r0, center=center, phi_resolution=16, theta_resolution=16)
                actors['sphere'] = p.add_mesh(s, color='magenta', style='wireframe', opacity=sphere_opacity)
            # highlight nodes within radius
            tree = cKDTree(nodes)
            idxs = tree.query_ball_point(center, r0)
            if len(idxs) > 0:
                pts = nodes[np.array(idxs, dtype=int)]
                actors['highlight'] = p.add_points(pts, color='orange', point_size=point_size, render_points_as_spheres=True)
                if show_highlight_normals:
                    sel_normals = normals[np.array(idxs, dtype=int)]
                    arrow = pv.Arrow()
                    src = pv.PolyData(pts)
                    src['n'] = sel_normals
                    glyphs = src.glyph(orient='n', scale=False, factor=max(arrow_factor, 1e-8), geom=arrow)
                    actors['highlight_normals'] = p.add_mesh(glyphs, color='blue')
        # samples
        inds = samples_map.get(pid, [])
        pts = nodes[inds] if len(inds)>0 else np.empty((0,3))
        if pts.size>0:
            actors['samples'] = p.add_points(pts, color=color_samples, point_size=point_size, render_points_as_spheres=True)
        txt = f"Patch {pid} | samples: {len(inds)} | radius: {float(radii[pid]) if radii is not None else 0:.4f}"
        actors['text'] = p.add_text(txt, position='upper_left', font_size=12)
        p.render()
        return pid
    state = {'pid': max(0, min(init_patch_id, max_pid))}
    _update(state['pid'])
    def _next():
        state['pid'] = _update(state['pid']+1)
    def _prev():
        state['pid'] = _update(state['pid']-1)
    def _goto():
        try:
            inp = input('输入Patch编号: ').strip()
            if inp:
                state['pid'] = _update(int(inp))
        except Exception:
            pass
    p.add_key_event('Right', lambda: _next())
    p.add_key_event('Left', lambda: _prev())
    p.add_key_event('g', lambda: _goto())
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', required=True)
    ap.add_argument('--point_size', type=int, default=6)
    ap.add_argument('--patch_size', type=int, default=12)
    ap.add_argument('--arrow_factor', type=float, default=0.00001)
    ap.add_argument('--no_normals', action='store_true')
    ap.add_argument('--off_screen', action='store_true')
    ap.add_argument('--sphere_opacity', type=float, default=0.15)
    ap.add_argument('--viewrad', action='store_true')
    ap.add_argument('--sharp_only', action='store_true')
    ap.add_argument('--feature_count', type=int, default=None)
    ap.add_argument('--show_patch_samples', action='store_true')
    ap.add_argument('--patch_id', type=int, default=0)
    ap.add_argument('--screenshot', default=None)
    args = ap.parse_args()
    nodes, normals, patches, radii, feature_count = load_cfpu(args.path)
    if args.sharp_only and radii is not None:
        r = np.asarray(radii).reshape(-1)
        if args.feature_count is not None and 0 <= args.feature_count <= patches.shape[0]:
            patches = patches[:args.feature_count]
            radii = r[:args.feature_count]
        elif feature_count is not None and 0 <= feature_count <= patches.shape[0]:
            patches = patches[:feature_count]
            radii = r[:feature_count]
        else:
            s = np.sort(r)
            if s.size > 1:
                diffs = np.diff(s)
                k = int(np.argmax(diffs))
                thr = 0.5 * (s[k] + s[k+1])
                mask = r <= thr
                patches = patches[mask]
                radii = r[mask]
    viewrad_scene = args.viewrad and not args.show_patch_samples
    show_patches = not args.show_patch_samples
    show_normals_scene = (not args.no_normals) and (not args.show_patch_samples)
    p = build_scene(nodes, normals, patches, radii, args.point_size, args.patch_size, args.arrow_factor, show_normals_scene, viewrad_scene, args.off_screen, args.sphere_opacity, show_patches)
    if args.show_patch_samples:
        samples_map = load_sharp_samples(args.path)
        init_pid = args.patch_id
        if args.sharp_only:
            # constrain to sharp range
            if args.feature_count is not None:
                init_pid = min(init_pid, args.feature_count-1)
            elif feature_count is not None:
                init_pid = min(init_pid, feature_count-1)
        add_patch_samples_ui(p, nodes, normals, patches, radii, feature_count, samples_map, init_patch_id=init_pid, point_size=args.patch_size, sphere_opacity=args.sphere_opacity, show_highlight_normals=not args.no_normals, arrow_factor=args.arrow_factor)
    if args.screenshot:
        p.show(auto_close=False)
        p.screenshot(args.screenshot)
        p.close()
    else:
        p.show()

if __name__ == '__main__':
    main()
