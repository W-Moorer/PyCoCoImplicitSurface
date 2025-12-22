import argparse
import os
import numpy as np
import pyvista as pv
import importlib.util
import sys
import types
import time
import logging
from datetime import datetime

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def get_cfpurecon():
    try:
        from src.cfpurecon import cfpurecon
        return cfpurecon
    except Exception:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if root not in sys.path:
            sys.path.insert(0, root)
        from src.cfpurecon import cfpurecon
        return cfpurecon

def get_configure_patch_radii():
    try:
        from src.cfpurecon import configure_patch_radii
        return configure_patch_radii
    except Exception:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if root not in sys.path:
            sys.path.insert(0, root)
        from src.cfpurecon import configure_patch_radii
        return configure_patch_radii

def load_cfpu(path):
    x = np.loadtxt(os.path.join(path, 'nodes.txt'))
    n = np.loadtxt(os.path.join(path, 'normals.txt'))
    y = np.loadtxt(os.path.join(path, 'patches.txt'))
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if n.ndim == 1:
        n = n.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    return x, n, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', required=True)
    ap.add_argument('--grid', type=int, default=400)
    ap.add_argument('--screenshot', default=None)
    ap.add_argument('--off_screen', action='store_true')
    ap.add_argument('--save_iso', default=None)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--parallel', choices=['thread', 'process'], default='process')
    ap.add_argument('--log_name', default=None)
    ap.add_argument('--radius_scale', type=float, default=None)
    ap.add_argument('--feature_idx', default=None)
    ap.add_argument('--feature_scale', type=float, default=0.75)
    ap.add_argument('--radii_path', default=None)
    args = ap.parse_args()
    x, n, y = load_cfpu(args.path)
    os.makedirs('logs', exist_ok=True)
    base = os.path.splitext(os.path.basename(args.path.rstrip(os.sep)))[0]
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = args.log_name if args.log_name else f'logs/render_{base}_{ts}.log'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler(sys.stdout)])
    cfpurecon = get_cfpurecon()
    kernel = {'phi': lambda r: -r, 'eta': lambda r: -r, 'zeta': lambda r: -1.0/np.where(r==0, np.inf, r), 'order': 1}
    reg = {'exactinterp': 1, 'nrmlreg': 1, 'nrmllambda': 1e-4, 'potreg': 1, 'potlambda': 1e-4}
    total = y.shape[0]
    last = [-1]
    def progress(done, total_local):
        if total_local != total:
            return
        if done == last[0]:
            return
        last[0] = done
        w = 30
        filled = int(w * done / max(total, 1))
        bar = '#' * filled + '-' * (w - filled)
        pct = int(100 * done / max(total, 1))
        sys.stdout.write("\rComputing patches: [{}] {}% ({}/{})".format(bar, pct, done, total))
        sys.stdout.flush()
        if done >= total:
            sys.stdout.write("\n")
            sys.stdout.flush()
    def progress_stage(stage, info):
        stages = {
            '构网格-开始': '构网格: 开始',
            '构网格-完成': '构网格: 完成',
            '插值-开始': '插值: 开始',
            '插值-完成': '插值: 完成',
            '加权-开始': '加权: 开始',
            '加权-完成': '加权: 完成'
        }
        msg = stages.get(stage, stage)
        logging.info(msg)
        if '开始' in msg:
            stage_key = msg.split(':')[0]
            stage_times[stage_key] = {'start': time.time()}
        if '完成' in msg:
            stage_key = msg.split(':')[0]
            if stage_key in stage_times and 'start' in stage_times[stage_key]:
                stage_times[stage_key]['end'] = time.time()
                stage_times[stage_key]['dur'] = stage_times[stage_key]['end'] - stage_times[stage_key]['start']
    workers = args.workers if args.workers and args.workers > 0 else (os.cpu_count() or 1)
    os.environ['CFPU_PARALLEL'] = args.parallel
    if args.parallel == 'thread' and os.environ.get('CFPU_BLAS_THREADS') is None:
        os.environ['CFPU_BLAS_THREADS'] = '1'
    logging.info(f'输入: {args.path}')
    logging.info(f'节点数: {x.shape[0]} 法向数: {n.shape[0]} Patch数: {y.shape[0]}')
    logging.info(f'网格尺度: {args.grid} 并行: {args.parallel}')
    logging.info(f'工作线程/进程: {args.workers if args.workers else os.cpu_count()}')
    stage_times = {}
    t0 = time.time()
    rc = None
    if args.radius_scale is not None or args.feature_idx is not None:
        fm = None
        if args.feature_idx:
            ids = []
            for s in str(args.feature_idx).split(','):
                s = s.strip()
                if not s:
                    continue
                try:
                    ids.append(int(s))
                except Exception:
                    pass
            fm = np.zeros(y.shape[0], dtype=bool)
            for i in ids:
                if 0 <= i < y.shape[0]:
                    fm[i] = True
        rc = {'patchRad': args.radius_scale, 'feature_mask': fm, 'feature_scale': args.feature_scale}
        logging.info(f'半径设计: scale={args.radius_scale} feature_count={int(np.sum(fm)) if fm is not None else 0} feature_scale={args.feature_scale}')
    radii = None
    if args.radii_path:
        try:
            radii = np.loadtxt(args.radii_path)
            if radii.ndim == 0:
                radii = radii.reshape(1)
            if radii.shape[0] != y.shape[0]:
                radii = None
                logging.info('半径文件尺寸不匹配')
            else:
                logging.info(f'加载半径: {args.radii_path}')
        except Exception:
            radii = None
            logging.info('半径文件读取失败')
    if rc is not None or radii is not None:
        conf = get_configure_patch_radii()
        pr = None
        fm = None
        fs = 1.0
        if rc is not None:
            pr = rc.get('patchRad', None)
            fm = rc.get('feature_mask', None)
            fs = rc.get('feature_scale', 1.0)
        if radii is not None:
            pr = radii
        idx_list, nn_dist_list, patchRad = conf(x, y, 1.0, pr, fm, fs)
        covered = np.zeros(x.shape[0], dtype=bool)
        for ids in idx_list:
            if ids.size:
                covered[ids] = True
        miss = int(np.sum(~covered))
        logging.info(f'半径覆盖检查: 未覆盖节点数={miss} 最小半径={patchRad.min():.6f} 最大半径={patchRad.max():.6f} 平均半径={patchRad.mean():.6f}')
    try:
        potential, X, Y, Z = cfpurecon(x, n, y, args.grid, kernel, reg, n_jobs=workers, progress=progress, progress_stage=progress_stage)
    except TypeError:
        potential, X, Y, Z = cfpurecon(x, n, y, args.grid, kernel, reg)
    t1 = time.time()
    sg = pv.StructuredGrid(X, Y, Z)
    sg['potential'] = potential.ravel(order='F')
    logging.info('等值面提取: 开始')
    iso = sg.contour(isosurfaces=[0.0])
    logging.info('等值面提取: 完成')
    stage_times['等值面提取'] = stage_times.get('等值面提取', {})
    stage_times['等值面提取']['dur'] = stage_times['等值面提取'].get('dur', 0) + (time.time() - t1)
    total_dur = time.time() - t0
    logging.info(f'总耗时: {total_dur:.3f}s')
    for k in ['构网格', '插值', '加权', '等值面提取']:
        if k in stage_times and 'dur' in stage_times[k]:
            logging.info(f'{k}耗时: {stage_times[k]["dur"]:.3f}s')
    if args.save_iso:
        logging.info(f'保存等值面: {args.save_iso}')
    if args.save_iso:
        iso.save(args.save_iso)
    p = pv.Plotter(shape=(1, 2), off_screen=args.off_screen)
    p.subplot(0, 0)
    p.add_mesh(iso, color='lightgray', specular=0.1, smooth_shading=True, opacity=0.85)
    p.add_axes()
    p.subplot(0, 1)
    p.add_mesh(iso, color='lightgray', specular=0.1, smooth_shading=True, opacity=0.85)
    p.add_points(y, color='red', render_points_as_spheres=True, point_size=10)
    if radii is not None:
        for i in range(y.shape[0]):
            r0 = float(radii[i])
            if r0 <= 0:
                continue
            sph = pv.Sphere(radius=r0, center=y[i], phi_resolution=16, theta_resolution=16)
            p.add_mesh(sph, color='magenta', style='wireframe', opacity=0.15)
    p.add_axes()
    p.link_views()
    if args.screenshot:
        p.show(auto_close=False)
        p.screenshot(args.screenshot)
        p.close()
    else:
        p.show()

if __name__ == '__main__':
    main()
