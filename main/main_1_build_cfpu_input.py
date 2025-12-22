import argparse
import os
import sys

# 添加项目根目录到Python路径
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)

def _get_builder():
    try:
        from src.precompute import build_cfpu_input
        return build_cfpu_input
    except Exception:
        from src.precompute import build_cfpu_input
        return build_cfpu_input

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+')
    ap.add_argument('--out_root', default=os.path.join('output', 'cfpu_input'))
    ap.add_argument('--angle_threshold', type=float, default=30.0)
    ap.add_argument('--r_small_factor', type=float, default=0.5)
    ap.add_argument('--r_large_factor', type=float, default=3.0)
    ap.add_argument('--edge_split_threshold', type=float, default=None)
    ap.add_argument('--require_step_face_id_diff', action='store_true')
    ap.add_argument('--sharp_edges_pkl', type=str, default=None)
    args = ap.parse_args()
    inputs = args.inputs
    if not inputs:
        inputs = [
            os.path.join('input', 'smooth_geometry', 'Ellipsoid_surface_cellnormals.vtp'),
            os.path.join('input', 'smooth_geometry', 'Ring_surface_cellnormals.vtp'),
            os.path.join('input', 'smooth_geometry', 'Sphere_surface_cellnormals.vtp'),
            os.path.join('input', 'nonsmooth_geometry', 'Cone_surface_cellnormals.vtp'),
            os.path.join('input', 'nonsmooth_geometry', 'Cylinder_surface_cellnormals.vtp'),
            os.path.join('input', 'nonsmooth_geometry', 'Cube_surface_cellnormals.vtp'),
            os.path.join('input', 'nonsmooth_geometry', 'Prism_surface_cellnormals.vtp'),
            os.path.join('input', 'nonsmooth_geometry', 'TruncatedRing_surface_cellnormals.vtp'),
            os.path.join('input', 'combinatorial_geometry', 'CompositeBody1_surface_cellnormals.vtp'),
            os.path.join('input', 'combinatorial_geometry', 'CompositeBody2_surface_cellnormals.vtp')
        ]
    build_cfpu_input = _get_builder()
    for inp in inputs:
        base = os.path.splitext(os.path.basename(inp))[0]
        out_dir = os.path.join(args.out_root, base + '_cfpu_input')
        build_cfpu_input(inp, out_dir, args.angle_threshold, args.r_small_factor, args.r_large_factor, args.edge_split_threshold, args.require_step_face_id_diff, args.sharp_edges_pkl)

if __name__ == '__main__':
    main()
