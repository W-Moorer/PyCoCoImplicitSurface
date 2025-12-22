import numpy as np
# 尝试导入CuPy以支持GPU加速
try:
    import cupy as cp
    GPU_AVAILABLE = True
    # 测试GPU是否真正可用
    try:
        cp.ones(10)
        print("CuPy GPU加速可用")
    except Exception as e:
        print(f"CuPy导入成功，但GPU不可用: {e}")
        cp = np
        GPU_AVAILABLE = False
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    print("CuPy导入失败，将使用CPU计算")

from scipy.spatial import cKDTree
from scipy.linalg import qr, svd, cholesky, solve
from scipy.optimize import fminbound
from scipy.sparse import coo_matrix
from concurrent.futures import ThreadPoolExecutor
try:
    from multiprocessing import get_context
    from multiprocessing.shared_memory import SharedMemory
except Exception:
    get_context = None
    SharedMemory = None
import os
try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

# ===== 来自 util/curlfree_poly.py =====
def curlfree_poly(x, l):
    """计算无旋多项式基函数

    参数:
        x: 输入点坐标数组，形状为 (n, 3)
        l: 多项式阶数，支持 1 或 2

    返回:
        CP: 无旋多项式基函数矩阵, 形状 (3*n, L)
        P:  多项式基函数矩阵,     形状 (n, L)
    """
    n = x.shape[0]
    if l == 1:
        CP = np.zeros((3 * n, 3))
        P = np.zeros((n, 3))
        P[:, 0:3] = x
        CP[:, 0] = np.tile(np.array([1, 0, 0]), n)
        CP[:, 1] = np.tile(np.array([0, 1, 0]), n)
        CP[:, 2] = np.tile(np.array([0, 0, 1]), n)
        return CP, P
    if l == 2:
        CP = np.zeros((3 * n, 9))
        P = np.zeros((n, 9))
        P[:, 0:3] = x
        CP[:, 0] = np.tile(np.array([1, 0, 0]), n)
        CP[:, 1] = np.tile(np.array([0, 1, 0]), n)
        CP[:, 2] = np.tile(np.array([0, 0, 1]), n)
        P[:, 3:6] = 0.5 * x**2
        CP[:, 3] = np.concatenate([x[:, 0], np.zeros(n), np.zeros(n)])
        CP[:, 4] = np.concatenate([np.zeros(n), x[:, 1], np.zeros(n)])
        CP[:, 5] = np.concatenate([np.zeros(n), np.zeros(n), x[:, 2]])
        P[:, 6] = x[:, 1] * x[:, 2]
        CP[:, 6] = np.concatenate([np.zeros(n), x[:, 2], x[:, 1]])
        P[:, 7] = x[:, 0] * x[:, 2]
        CP[:, 7] = np.concatenate([x[:, 2], np.zeros(n), x[:, 0]])
        P[:, 8] = x[:, 0] * x[:, 1]
        CP[:, 8] = np.concatenate([x[:, 1], x[:, 0], np.zeros(n)])
        return CP, P
    raise ValueError('Degree not implemented')

# ===== 来自 util/gcv_cost_function.py =====
def gcv_cost_function(lam, z, d, n):
    """广义交叉验证(GCV)成本函数"""
    lam = np.exp(-lam)
    temp = (n * lam) / (d**2 + n * lam)
    score = n * np.sum((temp * z) ** 2) / (np.sum(temp) ** 2)
    return score

# ===== 来自 util/weight.py =====
def weight(r, delta, k):
    """分区单位(PU)权重函数"""
    r = r / delta
    phi = np.zeros_like(r)
    if k == 0:
        id1 = r <= (1 / 3)
        phi[id1] = 0.75 - 2.25 * r[id1] ** 2
        id2 = (r > 1 / 3) & (r <= 1)
        phi[id2] = 1.125 * (1 - r[id2]) ** 2
        return phi
    if k == 1:
        id1 = r <= (1 / 3)
        phi[id1] = -4.5 / delta**2
        id2 = (r > 1 / 3) & (r <= 1)
        phi[id2] = (-2.25 * (1 - r[id2]) / delta**2) * (1.0 / r[id2])
        return phi
    raise ValueError('PU Weight function error')

_GLOBAL_X = None
_GLOBAL_NRML = None
def _init_proc(shm_name_x, shape_x, dtype_x, shm_name_nrml, shape_nrml, dtype_nrml):
    if SharedMemory is None:
        return
    shm_x = SharedMemory(name=shm_name_x)
    shm_nrml = SharedMemory(name=shm_name_nrml)
    import numpy as _np
    global _GLOBAL_X, _GLOBAL_NRML, _SHM_X, _SHM_NRML
    _GLOBAL_X = _np.ndarray(shape_x, dtype=_np.dtype(dtype_x), buffer=shm_x.buf)
    _GLOBAL_NRML = _np.ndarray(shape_nrml, dtype=_np.dtype(dtype_nrml), buffer=shm_nrml.buf)
    _SHM_X = shm_x
    _SHM_NRML = shm_nrml

def _compute_proc(args):
    k, idk, nn_dist_k, y0, y1, y2, patchRad_k, order_k, exactinterp_k, nrmlreg_k, nrmllambda_k, nrmlschur_k, trbl_local, potreg_k, potlambda_k, startx_k, starty_k, startz_k, griddx_k, mmx_k, mmy_k, mmz_k = args
    if idk.size == 0:
        return (np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float))
    h2 = np.max(nn_dist_k)**2 if nn_dist_k.size else 1.0
    x_local = _GLOBAL_X[idk, :]
    xx_local = x_local[:, 0]
    xy_local = x_local[:, 1]
    xz_local = x_local[:, 2]
    n = x_local.shape[0]
    CFP, P = curlfree_poly(x_local, order_k)
    CFPt = CFP.T
    dx = xx_local.reshape(-1, 1) - xx_local.reshape(1, -1)
    dy = xy_local.reshape(-1, 1) - xy_local.reshape(1, -1)
    dz = xz_local.reshape(-1, 1) - xz_local.reshape(1, -1)
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    ui = _GLOBAL_NRML[idk, :]
    l_local = 3 if order_k == 1 else 9
    b = np.zeros(3*n + l_local)
    b[0:3*n:3] = ui[:, 0]
    b[1:3*n:3] = ui[:, 1]
    b[2:3*n:3] = ui[:, 2]
    A = np.zeros((3*n + l_local, 3*n + l_local))
    if order_k == 1:
        eta_temp = -r
        zeta_temp = -np.divide(1.0, r, where=(r!=0))
    elif order_k == 2:
        eta_temp = r**3
        zeta_temp = 3.0*r
    else:
        raise ValueError('Curl-free polynomial degree not supported')
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
    if nrmlreg_k != 2:
        if nrmlreg_k == 1:
            A[0:3*n, 0:3*n] = A[0:3*n, 0:3*n] + 3*n*nrmllambda_k*np.eye(3*n)
        elif nrmlreg_k == 3:
            if np.any(trbl_local):
                A[0:3*n, 0:3*n] = A[0:3*n, 0:3*n] + 3*n*nrmllambda_k*np.eye(3*n)
        if nrmlschur_k == 0:
            try:
                coeffs = solve(A, b, assume_a='sym', check_finite=False)
            except Exception:
                A[0:3*n, 0:3*n] = A[0:3*n, 0:3*n] + 3*n*1e-6*np.eye(3*n)
                try:
                    coeffs = solve(A, b, assume_a='sym', check_finite=False)
                except Exception:
                    coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
            coeffsp = coeffs[3*n:]
            coeffs = coeffs[:3*n]
        else:
            A0 = A[0:3*n, 0:3*n]
            b0 = b[0:3*n]
            try:
                AinvCFP = solve(A0, CFP, assume_a='sym', check_finite=False)
            except Exception:
                A0 = A0 + 3*n*1e-6*np.eye(3*n)
                AinvCFP = solve(A0, CFP, assume_a='sym', check_finite=False)
            coeffsp = np.linalg.pinv(CFPt @ AinvCFP) @ (CFPt @ solve(A0, b0, assume_a='sym', check_finite=False))
            coeffs = solve(A0, b0 - CFP @ coeffsp, assume_a='sym', check_finite=False)
    else:
        A0 = A[0:3*n, 0:3*n]
        b0 = b[0:3*n]
        Lc = CFP.shape[1]
        F1, G = qr(CFP, mode='economic')
        F2 = F1[:, Lc:]
        F1 = F1[:, :Lc]
        G1 = G[:Lc, :Lc]
        w1 = F1.T @ b0
        w2 = F2.T @ b0
        L = cholesky(F2.T @ A0 @ F2)
        U, D, _ = svd(L.T)
        D = np.diag(D)
        z = U.T @ w2
        lam = fminbound(lambda t: gcv_cost_function(t, z, D, 3.0/h2), -10, 35)
        lam = 3.0/h2 * np.exp(-lam)
        A0 = A0 + lam*np.eye(3*n)
        coeffs = F2 @ (U @ (z / (D**2 + lam)))
        coeffsp = solve(G1, w1 - F1.T @ (A0 @ coeffs), check_finite=False)
    coeffsx = coeffs[0:3*n:3]
    coeffsy = coeffs[1:3*n:3]
    coeffsz = coeffs[2:3*n:3]
    temp_potential_nodes = np.sum(eta_temp * (dx * coeffsx.reshape(1, -1) + dy * coeffsy.reshape(1, -1) + dz * coeffsz.reshape(1, -1)), axis=1) + P @ coeffsp
    if exactinterp_k:
        P0 = np.ones((n, 1))
        A1 = np.ones((n+1, n+1))
        A1[0:n, 0:n] = (-r if order_k == 1 else r**3)
        A1[-1, -1] = 0.0
        b1 = np.concatenate([temp_potential_nodes, np.array([0.0])])
        if potreg_k != 2:
            if potreg_k == 1:
                A1[0:n, 0:n] = A1[0:n, 0:n] + n*potlambda_k*np.eye(n)
            elif potreg_k == 3:
                if np.any(trbl_local):
                    A1[0:n, 0:n] = A1[0:n, 0:n] + n*potlambda_k*np.eye(n)
            coeffs_correction = solve(A1, b1, assume_a='sym', check_finite=False)
        else:
            Lc = P0.shape[1]
            b2 = b1[0:n]
            A2 = A1[0:n, 0:n]
            F1, G = qr(P0, mode='economic')
            F2 = F1[:, Lc:]
            F1 = F1[:, :Lc]
            G1 = G[:Lc, :Lc]
            w1 = F1.T @ b2
            w2 = F2.T @ b2
            L = cholesky(F2.T @ A2 @ F2)
            U, D, _ = svd(L.T)
            D = np.diag(D)
            z2 = U.T @ w2
            lam = fminbound(lambda t: gcv_cost_function(t, z2, D, 1.0/h2), -10, 35)
            lam = (1.0/h2) * np.exp(-lam)
            A2 = A2 + lam*np.eye(n)
            temp = F2 @ (U @ (z2 / (D**2 + lam)))
            coeffs_correction = np.concatenate([temp, solve(G1, w1 - F1.T @ (A2 @ temp), check_finite=False)])
    else:
        P1 = np.hstack([P[:, 0:3], np.ones((n, 1))])
        coeffs_correction = np.linalg.lstsq(P1, temp_potential_nodes, rcond=None)[0]
    coeffs_correction_const = coeffs_correction[-1]
    coeffs_correction_vec = coeffs_correction[:-1]
    ix = int(np.round((y0 - startx_k) / griddx_k)) + 1
    iy = int(np.round((y1 - starty_k) / griddx_k)) + 1
    iz = int(np.round((y2 - startz_k) / griddx_k)) + 1
    factor = int(np.round(patchRad_k / griddx_k))
    ixs = np.arange(max(ix - factor, 1), min(ix + factor, mmx_k) + 1)
    iys = np.arange(max(iy - factor, 1), min(iy + factor, mmy_k) + 1)
    izs = np.arange(max(iz - factor, 1), min(iz + factor, mmz_k) + 1)
    xxg = startx_k + (ixs - 1) * griddx_k
    yyg = starty_k + (iys - 1) * griddx_k
    zzg = startz_k + (izs - 1) * griddx_k
    XX3, YY3, ZZ3 = np.meshgrid(xxg, yyg, zzg, indexing='xy')
    De = (y0 - XX3)**2 + (y1 - YY3)**2 + (y2 - ZZ3)**2
    idmask = De.reshape(-1) < patchRad_k**2
    ixs2 = np.repeat(ixs.reshape(1, -1), len(yyg), axis=0)
    ixs2 = np.repeat(ixs2[:, :, np.newaxis], len(zzg), axis=2)
    iys2 = np.repeat(iys.reshape(-1, 1), len(xxg), axis=1)
    iys2 = np.repeat(iys2[:, :, np.newaxis], len(zzg), axis=2)
    izs2 = np.repeat(izs.reshape(1, 1, -1), len(yyg), axis=0)
    izs2 = np.repeat(izs2, len(xxg), axis=1)
    temp_idg = (iys2 + (ixs2 - 1) * mmy_k) + (izs2 - 1) * (mmx_k * mmy_k)
    temp_idg = temp_idg.reshape(-1)
    temp_idg = temp_idg[idmask] - 1
    De = np.sqrt(De.reshape(-1)[idmask])
    idxe_k = temp_idg.astype(int)
    Psi_k = weight(De, patchRad_k, 0)
    xe_local = np.vstack([XX3.reshape(-1), YY3.reshape(-1), ZZ3.reshape(-1)]).T
    xe_local = xe_local[idmask, :]
    mm = xe_local.shape[0]
    if mm == 0:
        return (idxe_k, np.array([], dtype=int), Psi_k, np.array([], dtype=float))
    batch_sz = int(np.ceil(100**2 / max(n, 1)))
    temp_potential = np.zeros(mm)
    potential_correction = np.zeros(mm)
    for j in range(0, mm, batch_sz):
        idb = slice(j, min(j + batch_sz, mm))
        xe_local_batch = xe_local[idb, :]
        dxb = xe_local_batch[:, 0].reshape(-1, 1) - xx_local.reshape(1, -1)
        dyb = xe_local_batch[:, 1].reshape(-1, 1) - xy_local.reshape(1, -1)
        dzb = xe_local_batch[:, 2].reshape(-1, 1) - xz_local.reshape(1, -1)
        rb = np.sqrt(dxb**2 + dyb**2 + dzb**2)
        _, Pb = curlfree_poly(xe_local_batch, order_k)
        if order_k == 1:
            etab = -rb
            phib = -rb
        else:
            etab = rb**3
            phib = rb**3
        temp_potential[j:j+xe_local_batch.shape[0]] = np.sum(etab * (dxb * coeffsx.reshape(1, -1) + dyb * coeffsy.reshape(1, -1) + dzb * coeffsz.reshape(1, -1)), axis=1) + Pb @ coeffsp
        if exactinterp_k:
            potential_correction[j:j+xe_local_batch.shape[0]] = phib @ coeffs_correction_vec + coeffs_correction_const
        else:
            potential_correction[j:j+xe_local_batch.shape[0]] = Pb[:, 0:3] @ coeffs_correction_vec + coeffs_correction_const
    potential_k = temp_potential - potential_correction
    patch_vec_k = np.full(mm, k + 1)
    return (idxe_k, patch_vec_k, Psi_k, potential_k)

def configure_patch_radii(x, y, delta=1.0, patchRad=None, feature_mask=None, feature_scale=1.0):
    tree_y = cKDTree(y)
    if patchRad is None:
        nn = tree_y.query(y, k=2)[0][:, 1]
        H = np.max(nn) if nn.size else 0.0
        patchRad0 = (1.0 + delta) * H / 2.0
        patchRad = np.full(y.shape[0], patchRad0)
    else:
        if np.isscalar(patchRad):
            nn = tree_y.query(y, k=2)[0][:, 1]
            H = np.max(nn) if nn.size else 0.0
            patchRad0 = (1.0 + delta) * H / 2.0
            patchRad = np.full(y.shape[0], float(patchRad) * patchRad0)
        else:
            patchRad = np.asarray(patchRad, dtype=float)
            if patchRad.shape[0] != y.shape[0]:
                nn = tree_y.query(y, k=2)[0][:, 1]
                H = np.max(nn) if nn.size else 0.0
                patchRad0 = (1.0 + delta) * H / 2.0
                patchRad = np.full(y.shape[0], patchRad0)
    if feature_mask is not None:
        fm = np.asarray(feature_mask, dtype=bool)
        fm = fm[:y.shape[0]]
        patchRad[fm] = patchRad[fm] * feature_scale
    tree_x = cKDTree(x)
    idx = []
    nn_dist_list = []
    for k in range(y.shape[0]):
        id_list = tree_x.query_ball_point(y[k, :], patchRad[k])
        idx.append(np.array(id_list, dtype=int))
        if len(id_list) == 0:
            nn_dist_list.append(np.array([], dtype=float))
        else:
            dists = np.linalg.norm(x[id_list, :] - y[k, :], axis=1)
            nn_dist_list.append(dists)
    nodeInPatch = np.zeros(x.shape[0], dtype=bool)
    for k in range(y.shape[0]):
        nodeInPatch[idx[k]] = True
    missingIds = np.where(~nodeInPatch)[0]
    while missingIds.size > 0:
        cp_id = tree_y.query(x[missingIds[0], :], k=1)[1]
        p_dist = tree_y.query(x[missingIds[0], :], k=1)[0]
        temp_rad = max(patchRad[cp_id], 1.01 * float(p_dist))
        patchRad[cp_id] = temp_rad
        id_list = tree_x.query_ball_point(y[cp_id, :], patchRad[cp_id])
        idx[cp_id] = np.array(id_list, dtype=int)
        if len(id_list) == 0:
            nn_dist_list[cp_id] = np.array([], dtype=float)
        else:
            dists = np.linalg.norm(x[id_list, :] - y[cp_id, :], axis=1)
            nn_dist_list[cp_id] = dists
        nodeInPatch[id_list] = True
        missingIds = np.where(~nodeInPatch)[0]
    return idx, nn_dist_list, patchRad

def cfpurecon(x, nrml, y, gridsize, kernelinfo=None, reginfo=None, n_jobs=None, progress=None, progress_stage=None):
    if kernelinfo is None:
        kernelinfo = {
            'phi': lambda r: -r,
            'eta': lambda r: -r,
            'zeta': lambda r: -1.0/np.where(r==0, np.inf, r),
            'order': 1
        }
    if reginfo is None:
        reginfo = {'exactinterp': 1}
    minxx = np.min(x, axis=0)
    maxxx = np.max(x, axis=0)
    x = x - minxx
    scale = np.max(maxxx - minxx)
    x = x / scale
    y = y - minxx
    y = y / scale
    M = y.shape[0]
    N = x.shape[0]
    tree_y = cKDTree(y)
    nn_dist = tree_y.query(y, k=2)[0][:, 1]
    H = np.max(nn_dist)
    delta = 1.0
    patchRad0 = (1.0 + delta) * H / 2.0
    tree_x = cKDTree(x)
    idx = []
    nn_dist_list = []
    for k in range(M):
        id_list = tree_x.query_ball_point(y[k, :], patchRad0)
        idx.append(np.array(id_list, dtype=int))
        if len(id_list) == 0:
            nn_dist_list.append(np.array([], dtype=float))
        else:
            dists = np.linalg.norm(x[id_list, :] - y[k, :], axis=1)
            nn_dist_list.append(dists)
    patchRad = np.full(M, patchRad0)
    nodeInPatch = np.zeros(N, dtype=bool)
    for k in range(M):
        nodeInPatch[idx[k]] = True
    missingIds = np.where(~nodeInPatch)[0]
    while missingIds.size > 0:
        cp_id = tree_y.query(x[missingIds[0], :], k=1)[1]
        p_dist = tree_y.query(x[missingIds[0], :], k=1)[0]
        temp_rad = 1.01 * p_dist
        id_list = tree_x.query_ball_point(y[cp_id, :], temp_rad)
        dists = np.linalg.norm(x[id_list, :] - y[cp_id, :], axis=1)
        idx[cp_id] = np.array(id_list, dtype=int)
        nn_dist_list[cp_id] = dists
        patchRad[cp_id] = temp_rad
        nodeInPatch[id_list] = True
        missingIds = np.where(~nodeInPatch)[0]
    exactinterp = reginfo.get('exactinterp', 1)
    nrmlreg = reginfo.get('nrmlreg', 0)
    nrmllambda = reginfo.get('nrmllambda', 0)
    nrmlschur = reginfo.get('nrmlschur', 0)
    trbl_id = reginfo.get('trbl_id', np.zeros(N, dtype=bool))
    potreg = reginfo.get('potreg', 0)
    potlambda = reginfo.get('potlambda', 0)
    eta = kernelinfo['eta']
    zeta = kernelinfo['zeta']
    phi = kernelinfo['phi']
    order = kernelinfo['order']
    if order == 1:
        l = 3
    elif order == 2:
        l = 9
    else:
        raise ValueError('Curl-free polynomial degree not supported')
    minx = np.min(x, axis=0)
    maxx = np.max(x, axis=0)
    griddx = np.max((maxx - minx) / gridsize)
    startx = minx[0] - 3 * griddx
    endx = maxx[0] + 3 * griddx
    starty = minx[1] - 3 * griddx
    endy = maxx[1] + 3 * griddx
    startz = minx[2] - 3 * griddx
    endz = maxx[2] + 3 * griddx
    xx = np.arange(startx, endx + griddx/2, griddx)
    yy = np.arange(starty, endy + griddx/2, griddx)
    zz = np.arange(startz, endz + griddx/2, griddx)
    if progress_stage is not None:
        try:
            progress_stage('构网格-开始', None)
        except Exception:
            pass
    X, Y, Z = np.meshgrid(xx, yy, zz, indexing='xy')
    mmy, mmx, mmz = X.shape
    m = mmx * mmy * mmz
    if progress_stage is not None:
        try:
            progress_stage('构网格-完成', None)
        except Exception:
            pass
    idxe_patch = [None] * M
    patch_vec = [None] * M
    Psi = [None] * M
    potential_local = [None] * M
    if progress_stage is not None:
        try:
            progress_stage('插值-开始', M)
        except Exception:
            pass
    def _compute(k):
        idk = idx[k]
        if idk.size == 0:
            return (np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float))
        h2 = np.max(nn_dist_list[k])**2 if nn_dist_list[k].size else 1.0
        x_local = x[idk, :]
        xx_local = x_local[:, 0]
        xy_local = x_local[:, 1]
        xz_local = x_local[:, 2]
        n = x_local.shape[0]
        CFP, P = curlfree_poly(x_local, order)
        CFPt = CFP.T
        
        # 使用CuPy进行GPU加速计算，添加异常处理
        use_gpu = GPU_AVAILABLE
        
        # CPU版本计算作为默认和回退选项
        dx = xx_local.reshape(-1, 1) - xx_local.reshape(1, -1)
        dy = xy_local.reshape(-1, 1) - xy_local.reshape(1, -1)
        dz = xz_local.reshape(-1, 1) - xz_local.reshape(1, -1)
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        ui = nrml[idk, :]
        b = np.zeros(3*n + l)
        b[0:3*n:3] = ui[:, 0]
        b[1:3*n:3] = ui[:, 1]
        b[2:3*n:3] = ui[:, 2]
        A = np.zeros((3*n + l, 3*n + l))
        eta_temp = eta(r)
        zeta_temp = zeta(r)
        np.fill_diagonal(zeta_temp, 0.0)
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
        
        # 后续计算保持CPU版本不变，因为涉及到scipy的特殊函数
        if nrmlreg != 2:
            if nrmlreg == 1:
                A[0:3*n, 0:3*n] = A[0:3*n, 0:3*n] + 3*n*nrmllambda*np.eye(3*n)
            elif nrmlreg == 3:
                if np.any(trbl_id[idk]):
                    A[0:3*n, 0:3*n] = A[0:3*n, 0:3*n] + 3*n*nrmllambda*np.eye(3*n)
            if nrmlschur == 0:
                coeffs = solve(A, b, assume_a='sym', check_finite=False)
                coeffsp = coeffs[3*n:]
                coeffs = coeffs[:3*n]
            else:
                A0 = A[0:3*n, 0:3*n]
                b0 = b[0:3*n]
                AinvCFP = solve(A0, CFP, assume_a='sym', check_finite=False)
                coeffsp = np.linalg.pinv(CFPt @ AinvCFP) @ (CFPt @ solve(A0, b0, assume_a='sym', check_finite=False))
                coeffs = solve(A0, b0 - CFP @ coeffsp, assume_a='sym', check_finite=False)
        else:
            A0 = A[0:3*n, 0:3*n]
            b0 = b[0:3*n]
            Lc = CFP.shape[1]
            F1, G = qr(CFP, mode='economic')
            F2 = F1[:, Lc:]
            F1 = F1[:, :Lc]
            G1 = G[:Lc, :Lc]
            w1 = F1.T @ b0
            w2 = F2.T @ b0
            L = cholesky(F2.T @ A0 @ F2)
            U, D, _ = svd(L.T)
            D = np.diag(D)
            z = U.T @ w2
            lam = fminbound(lambda t: gcv_cost_function(t, z, D, 3.0/h2), -10, 35)
            lam = 3.0/h2 * np.exp(-lam)
            A0 = A0 + lam*np.eye(3*n)
            coeffs = F2 @ (U @ (z / (D**2 + lam)))
            coeffsp = solve(G1, w1 - F1.T @ (A0 @ coeffs), check_finite=False)
        coeffsx = coeffs[0:3*n:3]
        coeffsy = coeffs[1:3*n:3]
        coeffsz = coeffs[2:3*n:3]
        temp_potential_nodes = np.sum(eta_temp * (dx * coeffsx.reshape(1, -1) + dy * coeffsy.reshape(1, -1) + dz * coeffsz.reshape(1, -1)), axis=1) + P @ coeffsp
        if exactinterp:
            P0 = np.ones((n, 1))
            A1 = np.ones((n+1, n+1))
            A1[0:n, 0:n] = phi(r)
            A1[-1, -1] = 0.0
            b1 = np.concatenate([temp_potential_nodes, np.array([0.0])])
            if potreg != 2:
                if potreg == 1:
                    A1[0:n, 0:n] = A1[0:n, 0:n] + n*potlambda*np.eye(n)
                elif potreg == 3:
                    if np.any(trbl_id[idk]):
                        A1[0:n, 0:n] = A1[0:n, 0:n] + n*potlambda*np.eye(n)
                coeffs_correction = np.linalg.solve(A1, b1)
            else:
                Lc = P0.shape[1]
                b2 = b1[0:n]
                A2 = A1[0:n, 0:n]
                F1, G = qr(P0, mode='economic')
                F2 = F1[:, Lc:]
                F1 = F1[:, :Lc]
                G1 = G[:Lc, :Lc]
                w1 = F1.T @ b2
                w2 = F2.T @ b2
                L = cholesky(F2.T @ A2 @ F2)
                U, D, _ = svd(L.T)
                D = np.diag(D)
                z2 = U.T @ w2
                lam = fminbound(lambda t: gcv_cost_function(t, z2, D, 1.0/h2), -10, 35)
                lam = (1.0/h2) * np.exp(-lam)
                A2 = A2 + lam*np.eye(n)
                temp = F2 @ (U @ (z2 / (D**2 + lam)))
                coeffs_correction = np.concatenate([temp, solve(G1, w1 - F1.T @ (A2 @ temp), check_finite=False)])
        else:
            P1 = np.hstack([P[:, 0:3], np.ones((n, 1))])
            coeffs_correction = np.linalg.lstsq(P1, temp_potential_nodes, rcond=None)[0]
        coeffs_correction_const = coeffs_correction[-1]
        coeffs_correction_vec = coeffs_correction[:-1]
        ix = int(np.round((y[k, 0] - startx) / griddx)) + 1
        iy = int(np.round((y[k, 1] - starty) / griddx)) + 1
        iz = int(np.round((y[k, 2] - startz) / griddx)) + 1
        factor = int(np.round(patchRad[k] / griddx))
        ixs = np.arange(max(ix - factor, 1), min(ix + factor, mmx) + 1)
        iys = np.arange(max(iy - factor, 1), min(iy + factor, mmy) + 1)
        izs = np.arange(max(iz - factor, 1), min(iz + factor, mmz) + 1)
        xxg = startx + (ixs - 1) * griddx
        yyg = starty + (iys - 1) * griddx
        zzg = startz + (izs - 1) * griddx
        XX3, YY3, ZZ3 = np.meshgrid(xxg, yyg, zzg, indexing='xy')
        De = (y[k, 0] - XX3)**2 + (y[k, 1] - YY3)**2 + (y[k, 2] - ZZ3)**2
        idmask = De.reshape(-1) < patchRad[k]**2
        ixs2 = np.repeat(ixs.reshape(1, -1), len(yyg), axis=0)
        ixs2 = np.repeat(ixs2[:, :, np.newaxis], len(zzg), axis=2)
        iys2 = np.repeat(iys.reshape(-1, 1), len(xxg), axis=1)
        iys2 = np.repeat(iys2[:, :, np.newaxis], len(zzg), axis=2)
        izs2 = np.repeat(izs.reshape(1, 1, -1), len(yyg), axis=0)
        izs2 = np.repeat(izs2, len(xxg), axis=1)
        temp_idg = (iys2 + (ixs2 - 1) * mmy) + (izs2 - 1) * (mmx * mmy)
        temp_idg = temp_idg.reshape(-1)
        temp_idg = temp_idg[idmask] - 1
        De = np.sqrt(De.reshape(-1)[idmask])
        idxe_k = temp_idg.astype(int)
        Psi_k = weight(De, patchRad[k], 0)
        xe_local = np.vstack([XX3.reshape(-1), YY3.reshape(-1), ZZ3.reshape(-1)]).T
        xe_local = xe_local[idmask, :]
        mm = xe_local.shape[0]
        if mm == 0:
            return (idxe_k, np.array([], dtype=int), Psi_k, np.array([], dtype=float))
        batch_sz = int(np.ceil(100**2 / max(n, 1)))
        temp_potential = np.zeros(mm)
        potential_correction = np.zeros(mm)
        for j in range(0, mm, batch_sz):
            idb = slice(j, min(j + batch_sz, mm))
            xe_local_batch = xe_local[idb, :]
            dxb = xe_local_batch[:, 0].reshape(-1, 1) - xx_local.reshape(1, -1)
            dyb = xe_local_batch[:, 1].reshape(-1, 1) - xy_local.reshape(1, -1)
            dzb = xe_local_batch[:, 2].reshape(-1, 1) - xz_local.reshape(1, -1)
            rb = np.sqrt(dxb**2 + dyb**2 + dzb**2)
            _, Pb = curlfree_poly(xe_local_batch, order)
            temp_potential[j:j+xe_local_batch.shape[0]] = np.sum(eta(rb) * (dxb * coeffsx.reshape(1, -1) + dyb * coeffsy.reshape(1, -1) + dzb * coeffsz.reshape(1, -1)), axis=1) + Pb @ coeffsp
            if exactinterp:
                potential_correction[j:j+xe_local_batch.shape[0]] = phi(rb) @ coeffs_correction_vec + coeffs_correction_const
            else:
                potential_correction[j:j+xe_local_batch.shape[0]] = Pb[:, 0:3] @ coeffs_correction_vec + coeffs_correction_const
        potential_k = temp_potential - potential_correction
        patch_vec_k = np.full(mm, k + 1)
        return (idxe_k, patch_vec_k, Psi_k, potential_k)
    workers = n_jobs if (n_jobs and n_jobs > 0) else min(M, os.cpu_count() or 1)
    mode_env = os.environ.get('CFPU_PARALLEL', 'thread')
    if workers > 1 and mode_env == 'process' and SharedMemory is not None and get_context is not None:
        shm_x = SharedMemory(create=True, size=x.nbytes)
        shm_nrml = SharedMemory(create=True, size=nrml.nbytes)
        np.ndarray(x.shape, dtype=x.dtype, buffer=shm_x.buf)[:] = x
        np.ndarray(nrml.shape, dtype=nrml.dtype, buffer=shm_nrml.buf)[:] = nrml
        try:
            ctx = get_context('spawn')
            global _GLOBAL_X, _GLOBAL_NRML
            _GLOBAL_X = None
            _GLOBAL_NRML = None
            with ctx.Pool(processes=workers, initializer=_init_proc, initargs=(shm_x.name, x.shape, x.dtype.str, shm_nrml.name, nrml.shape, nrml.dtype.str)) as pool:
                arg_iter = ((k, idx[k], nn_dist_list[k], y[k, 0], y[k, 1], y[k, 2], patchRad[k], order, exactinterp, nrmlreg, nrmllambda, nrmlschur, trbl_id[idx[k]], potreg, potlambda, startx, starty, startz, griddx, mmx, mmy, mmz) for k in range(M))
                for k, res in enumerate(pool.imap(_compute_proc, arg_iter)):
                    idxe_patch[k], patch_vec[k], Psi[k], potential_local[k] = res
                    if progress is not None:
                        try:
                            progress(k + 1, M)
                        except Exception:
                            pass
        finally:
            shm_x.close()
            shm_x.unlink()
            shm_nrml.close()
            shm_nrml.unlink()
    elif workers > 1:
        if threadpool_limits is not None:
            blas_threads_env = os.environ.get('CFPU_BLAS_THREADS')
            blas_threads = int(blas_threads_env) if blas_threads_env else 1
            with threadpool_limits(blas_threads, user_api='blas'):
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    for k, res in enumerate(ex.map(_compute, range(M))):
                        idxe_patch[k], patch_vec[k], Psi[k], potential_local[k] = res
                        if progress is not None:
                            try:
                                progress(k + 1, M)
                            except Exception:
                                pass
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for k, res in enumerate(ex.map(_compute, range(M))):
                    idxe_patch[k], patch_vec[k], Psi[k], potential_local[k] = res
                    if progress is not None:
                        try:
                            progress(k + 1, M)
                        except Exception:
                            pass
    else:
        for k in range(M):
            idxe_patch[k], patch_vec[k], Psi[k], potential_local[k] = _compute(k)
            if progress is not None:
                try:
                    progress(k + 1, M)
                except Exception:
                    pass
    if progress_stage is not None:
        try:
            progress_stage('插值-完成', None)
        except Exception:
            pass
    patch_vec_cat = np.concatenate([pv for pv in patch_vec if pv.size > 0]) if any([pv.size > 0 for pv in patch_vec]) else np.array([], dtype=int)
    idxe_vec_cat = np.concatenate([ie for ie in idxe_patch if ie.size > 0]) if any([ie.size > 0 for ie in idxe_patch]) else np.array([], dtype=int)
    Psi_cat = np.concatenate([ps for ps in Psi if ps.size > 0]) if any([ps.size > 0 for ps in Psi]) else np.array([], dtype=float)
    if progress_stage is not None:
        try:
            progress_stage('加权-开始', None)
        except Exception:
            pass
    Psi_sum = np.zeros(m)
    if idxe_vec_cat.size > 0:
        Psi_sum = coo_matrix((Psi_cat, (idxe_vec_cat, patch_vec_cat - 1)), shape=(m, M)).sum(axis=1).A1
    for k in range(M):
        if potential_local[k].size > 0:
            denom = Psi_sum[idxe_patch[k]]
            potential_local[k] = potential_local[k] * (Psi[k] / denom)
    temp = np.zeros(m)
    if idxe_vec_cat.size > 0:
        temp = coo_matrix((np.concatenate([pl for pl in potential_local if pl.size > 0]), (idxe_vec_cat, patch_vec_cat - 1)), shape=(m, M)).sum(axis=1).A1
    i_nonzero = np.where(Psi_sum > 0)[0]
    potential = np.full(m, np.nan)
    potential[i_nonzero] = temp[i_nonzero]
    potential = potential.reshape((mmy, mmx, mmz), order='F')
    X = X * scale + minxx[0]
    Y = Y * scale + minxx[1]
    Z = Z * scale + minxx[2]
    if progress_stage is not None:
        try:
            progress_stage('加权-完成', None)
        except Exception:
            pass
    return potential, X, Y, Z


# ===== 模块导出 =====
__all__ = [
    'GPU_AVAILABLE',
    'curlfree_poly',
    'gcv_cost_function',
    'weight',
    'configure_patch_radii',
    'cfpurecon',
    '_init_proc',
    '_compute_proc',
]


if __name__ == '__main__':
    print("cfpurecon - 无旋RBF分区单位隐式曲面重建模块（单文件合并版）")
    print("包含函数：curlfree_poly, gcv_cost_function, weight, configure_patch_radii, cfpurecon")
