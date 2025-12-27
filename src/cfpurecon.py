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


# ===== GPU/NumPy 通用的多项式基（只返回 P；GPU 路径不需要 CP） =====
def poly_P_xp(x, l, xp):
    """返回与 curlfree_poly 一致的 P（n,L），支持 np/cp。"""
    if l == 1:
        return x  # (n,3)
    if l == 2:
        n = x.shape[0]
        P = xp.zeros((n, 9), dtype=x.dtype)
        P[:, 0:3] = x
        P[:, 3:6] = 0.5 * x**2
        P[:, 6] = x[:, 1] * x[:, 2]
        P[:, 7] = x[:, 0] * x[:, 2]
        P[:, 8] = x[:, 0] * x[:, 1]
        return P
    raise ValueError('Degree not implemented')


def weight_xp(r, delta, k, xp):
    """分区单位(PU)权重函数（np/cp 通用）。"""
    r = r / delta
    phi = xp.zeros_like(r)
    if k == 0:
        id1 = r <= (1.0 / 3.0)
        phi[id1] = 0.75 - 2.25 * r[id1] ** 2
        id2 = (r > 1.0 / 3.0) & (r <= 1.0)
        phi[id2] = 1.125 * (1.0 - r[id2]) ** 2
        return phi
    if k == 1:
        id1 = r <= (1.0 / 3.0)
        phi[id1] = -4.5 / (delta ** 2)
        id2 = (r > 1.0 / 3.0) & (r <= 1.0)
        rr = xp.where(r[id2] == 0, xp.asarray(1.0, dtype=r.dtype), r[id2])
        phi[id2] = (-2.25 * (1.0 - r[id2]) / (delta ** 2)) * (1.0 / rr)
        return phi
    raise ValueError('PU Weight function error')


def _gpu_batch_size(n, mm):
    """根据 n（节点数）和 mm（网格点评估数）选择较安全的 GPU batch。"""
    target = int(os.environ.get('CFPU_GPU_TARGET_ELEMS', '1500000'))
    min_bs = int(os.environ.get('CFPU_GPU_MIN_BATCH', '512'))
    max_bs = int(os.environ.get('CFPU_GPU_MAX_BATCH', '262144'))
    bs = max(1, target // max(int(n), 1))
    bs = max(min_bs, min(bs, int(mm), max_bs))
    return int(bs)

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
_GLOBAL_CURVE = None

def _init_proc(shm_name_x, shape_x, dtype_x,
               shm_name_nrml, shape_nrml, dtype_nrml,
               shm_name_curve=None, shape_curve=None, dtype_curve=None):
    if SharedMemory is None:
        return
    shm_x = SharedMemory(name=shm_name_x)
    shm_nrml = SharedMemory(name=shm_name_nrml)
    import numpy as _np
    global _GLOBAL_X, _GLOBAL_NRML, _GLOBAL_CURVE, _SHM_X, _SHM_NRML, _SHM_CURVE
    _GLOBAL_X = _np.ndarray(shape_x, dtype=_np.dtype(dtype_x), buffer=shm_x.buf)
    _GLOBAL_NRML = _np.ndarray(shape_nrml, dtype=_np.dtype(dtype_nrml), buffer=shm_nrml.buf)
    _SHM_X = shm_x
    _SHM_NRML = shm_nrml

    if shm_name_curve is not None:
        shm_curve = SharedMemory(name=shm_name_curve)
        _GLOBAL_CURVE = _np.ndarray(shape_curve, dtype=_np.dtype(dtype_curve), buffer=shm_curve.buf)
        _SHM_CURVE = shm_curve
    else:
        _GLOBAL_CURVE = None
        _SHM_CURVE = None

def _compute_proc(args):
    # 新增 curve_idx_k：该 patch 对应的曲线点索引（在 _GLOBAL_CURVE 中）
    (k, idk, curve_idx_k, nn_dist_k,
     y0, y1, y2, patchRad_k, order_k,
     exactinterp_k, nrmlreg_k, nrmllambda_k, nrmlschur_k,
     trbl_local, potreg_k, potlambda_k,
     startx_k, starty_k, startz_k, griddx_k, mmx_k, mmy_k, mmz_k) = args

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
        zeta_temp = -np.divide(1.0, r, where=(r != 0))
    elif order_k == 2:
        eta_temp = r**3
        zeta_temp = 3.0 * r
    else:
        raise ValueError('Curl-free polynomial degree not supported')

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

    # 求解 curl-free 系数（保持你原逻辑）
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

    # 节点处势能
    temp_potential_nodes = np.sum(
        eta_temp * (dx * coeffsx.reshape(1, -1) + dy * coeffsy.reshape(1, -1) + dz * coeffsz.reshape(1, -1)),
        axis=1
    ) + P @ coeffsp

    # ===== residual correction：exact interpolation（做法A：节点 + 曲线点）=====
    use_extended_centers = False
    xx_corr = xx_local
    xy_corr = xy_local
    xz_corr = xz_local
    corr_size = n
    coeffs_correction = None

    if exactinterp_k:
        curve_local = np.empty((0, 3), dtype=float)
        if (_GLOBAL_CURVE is not None) and (curve_idx_k is not None) and (getattr(curve_idx_k, "size", 0) > 0):
            curve_local = _GLOBAL_CURVE[np.asarray(curve_idx_k, dtype=int), :]

            # 去掉与节点重合的曲线点，避免重复中心导致奇异
            if curve_local.shape[0] > 0:
                dmin = cKDTree(x_local).query(curve_local, k=1)[0]
                curve_local = curve_local[dmin > 1e-10]

            # 去重
            if curve_local.shape[0] > 1:
                key = np.round(curve_local, decimals=10)
                _, ui = np.unique(key, axis=0, return_index=True)
                curve_local = curve_local[np.sort(ui)]

        if curve_local.shape[0] > 0:
            use_extended_centers = True
            x_corr = np.vstack([x_local, curve_local])
            xx_corr = x_corr[:, 0]
            xy_corr = x_corr[:, 1]
            xz_corr = x_corr[:, 2]
            corr_size = x_corr.shape[0]

            # 计算 s_m 在 x_corr 上的值（中心仍为 x_local）
            dxc = x_corr[:, 0:1] - xx_local.reshape(1, -1)
            dyc = x_corr[:, 1:2] - xy_local.reshape(1, -1)
            dzc = x_corr[:, 2:3] - xz_local.reshape(1, -1)
            rc = np.sqrt(dxc**2 + dyc**2 + dzc**2)
            etac = (-rc if order_k == 1 else rc**3)
            Pc = poly_P_xp(x_corr, order_k, np)
            temp_potential_corr = np.sum(
                etac * (dxc * coeffsx.reshape(1, -1) + dyc * coeffsy.reshape(1, -1) + dzc * coeffsz.reshape(1, -1)),
                axis=1
            ) + Pc @ coeffsp

            # A1 用 x_corr 两两距离
            dx2 = x_corr[:, 0:1] - x_corr[None, :, 0]
            dy2 = x_corr[:, 1:2] - x_corr[None, :, 1]
            dz2 = x_corr[:, 2:3] - x_corr[None, :, 2]
            r2 = np.sqrt(dx2**2 + dy2**2 + dz2**2)
            phi2 = (-r2 if order_k == 1 else r2**3)

            A1 = np.ones((corr_size + 1, corr_size + 1))
            A1[0:corr_size, 0:corr_size] = phi2
            A1[-1, -1] = 0.0
            b1 = np.concatenate([temp_potential_corr, np.array([0.0])])

            if potreg_k != 2:
                if potreg_k == 1:
                    A1[0:corr_size, 0:corr_size] = A1[0:corr_size, 0:corr_size] + corr_size*potlambda_k*np.eye(corr_size)
                elif potreg_k == 3:
                    if np.any(trbl_local):
                        A1[0:corr_size, 0:corr_size] = A1[0:corr_size, 0:corr_size] + corr_size*potlambda_k*np.eye(corr_size)

                # 求解（增加容错，不改变正常情况）
                try:
                    coeffs_correction = solve(A1, b1, assume_a='sym', check_finite=False)
                except Exception:
                    A1[0:corr_size, 0:corr_size] = A1[0:corr_size, 0:corr_size] + corr_size*1e-10*np.eye(corr_size)
                    coeffs_correction = np.linalg.lstsq(A1, b1, rcond=None)[0]
            else:
                # GCV 分支（用 corr_size）
                P0 = np.ones((corr_size, 1))
                Lc = P0.shape[1]
                b2 = b1[0:corr_size]
                A2 = A1[0:corr_size, 0:corr_size]
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
                A2 = A2 + lam*np.eye(corr_size)
                temp = F2 @ (U @ (z2 / (D**2 + lam)))
                coeffs_correction = np.concatenate([temp, solve(G1, w1 - F1.T @ (A2 @ temp), check_finite=False)])
        else:
            # 无曲线点：完全走你原来的 nodes-only exact interpolation
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
        # 线性修正（不变）
        P1 = np.hstack([P[:, 0:3], np.ones((n, 1))])
        coeffs_correction = np.linalg.lstsq(P1, temp_potential_nodes, rcond=None)[0]

    coeffs_correction_const = coeffs_correction[-1]
    coeffs_correction_vec = coeffs_correction[:-1]

    # ===== patch 内网格点构建（不变）=====
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

        # s_m 仍然相对 x_local（n）
        dxb = xe_local_batch[:, 0].reshape(-1, 1) - xx_local.reshape(1, -1)
        dyb = xe_local_batch[:, 1].reshape(-1, 1) - xy_local.reshape(1, -1)
        dzb = xe_local_batch[:, 2].reshape(-1, 1) - xz_local.reshape(1, -1)
        rb = np.sqrt(dxb**2 + dyb**2 + dzb**2)
        _, Pb = curlfree_poly(xe_local_batch, order_k)

        if order_k == 1:
            etab = -rb
        else:
            etab = rb**3

        temp_potential[j:j+xe_local_batch.shape[0]] = np.sum(
            etab * (dxb * coeffsx.reshape(1, -1) + dyb * coeffsy.reshape(1, -1) + dzb * coeffsz.reshape(1, -1)),
            axis=1
        ) + Pb @ coeffsp

        if exactinterp_k:
            if use_extended_centers:
                # correction 相对 x_corr（corr_size）
                dxcg = xe_local_batch[:, 0].reshape(-1, 1) - xx_corr.reshape(1, -1)
                dycg = xe_local_batch[:, 1].reshape(-1, 1) - xy_corr.reshape(1, -1)
                dzcg = xe_local_batch[:, 2].reshape(-1, 1) - xz_corr.reshape(1, -1)
                rcg = np.sqrt(dxcg**2 + dycg**2 + dzcg**2)
                phicg = (-rcg if order_k == 1 else rcg**3)
                potential_correction[j:j+xe_local_batch.shape[0]] = phicg @ coeffs_correction_vec + coeffs_correction_const
            else:
                phib = (-rb if order_k == 1 else rb**3)
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


def load_patch_radii_txt(radii_path, *, expected_len=None):
    """从radii.txt读取每个patch的半径。

    参数:
        radii_path: radii.txt路径
        expected_len: 若给定，则检查长度是否与patch数量一致

    返回:
        radii: shape (M,) 的 float 数组
    """
    if radii_path is None:
        raise ValueError('radii_path is None')
    radii = np.loadtxt(radii_path, dtype=float)
    radii = np.asarray(radii, dtype=float).reshape(-1)
    if expected_len is not None and radii.shape[0] != int(expected_len):
        raise ValueError(f"radii长度({radii.shape[0]})与expected_len({int(expected_len)})不一致: {radii_path}")
    return radii

def cfpurecon(x, nrml, y, gridsize, kernelinfo=None, reginfo=None, n_jobs=None,
              progress=None, progress_stage=None,
              patch_radii=None, patch_radii_file=None,
              patch_radii_in_world_units=True, patch_radii_enforce_coverage=True,
              feature_mask=None, feature_scale=1.0, use_gpu=False,
              curve_points=None, curve_points_in_unit=False,
              curve_patch_map=None, curve_max_points_per_patch=200,
              curve_only_feature_patches=True,
              feature_tube_blend=False, feature_tube_tau=0.0,
              minxx_override=None, scale_override=None, return_transform=False
):
    if kernelinfo is None:
        kernelinfo = {
            'phi': lambda r: -r,
            'eta': lambda r: -r,
            'zeta': lambda r: -1.0/np.where(r==0, np.inf, r),
            'order': 1
        }
    if reginfo is None:
        reginfo = {'exactinterp': 1}
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("x must be (N, 3)")

    # === [修改开始] 支持 Override ===
    if minxx_override is not None:
        minxx = np.asarray(minxx_override, dtype=float)
    else:
        minxx = np.min(x, axis=0)

    if scale_override is not None:
        scale = float(scale_override)
    else:
        maxxx = np.max(x, axis=0)
        scale = float(np.max(maxxx - minxx))
        if scale <= 1e-8: scale = 1.0

    x = (x - minxx) / scale
    y = (y - minxx) / scale
    # === [修改结束] ===
    M = y.shape[0]
    N = x.shape[0]
    delta = 1.0

    # GPU 开关：仅当用户请求且 CuPy/GPU 可用时启用
    use_gpu = bool(use_gpu and GPU_AVAILABLE)

    # -------- Patch 半径/覆盖策略（支持读取 radii.txt） --------
    # 说明：cfpurecon内部会把x/y缩放到单位盒；若radii来自precompute输出（世界坐标），需要按同样scale缩放。
    if patch_radii is None and patch_radii_file is not None:
        patch_radii = load_patch_radii_txt(patch_radii_file, expected_len=M)

    if patch_radii is None:
        # 原始逻辑：统一初值 + 漏点补覆盖（保持原始行为，方便A/B对比）
        tree_y = cKDTree(y)
        nn_dist = tree_y.query(y, k=2)[0][:, 1]
        H = np.max(nn_dist) if nn_dist.size else 0.0
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
        patchRad = np.full(M, patchRad0, dtype=float)
        # feature patch mask（用于PU加权策略）
        fm = np.asarray(feature_mask, dtype=bool)[:M] if feature_mask is not None else None

        nodeInPatch = np.zeros(N, dtype=bool)
        for k in range(M):
            nodeInPatch[idx[k]] = True
        missingIds = np.where(~nodeInPatch)[0]
        while missingIds.size > 0:
            cp_id = tree_y.query(x[missingIds[0], :], k=1)[1]
            p_dist = tree_y.query(x[missingIds[0], :], k=1)[0]
            temp_rad = 1.01 * float(p_dist)
            id_list = tree_x.query_ball_point(y[cp_id, :], temp_rad)
            dists = np.linalg.norm(x[id_list, :] - y[cp_id, :], axis=1) if len(id_list) else np.array([], dtype=float)
            idx[cp_id] = np.array(id_list, dtype=int)
            nn_dist_list[cp_id] = dists
            patchRad[cp_id] = temp_rad
            nodeInPatch[id_list] = True
            missingIds = np.where(~nodeInPatch)[0]
    else:
        # 使用外部半径（数组或文件）
        patch_radii_arr = np.asarray(patch_radii, dtype=float).reshape(-1)
        if patch_radii_arr.size == 1:
            patch_radii_arr = np.full(M, float(patch_radii_arr[0]), dtype=float)
        if patch_radii_arr.shape[0] != M:
            raise ValueError(f"patch_radii长度({patch_radii_arr.shape[0]})与patch数量M({M})不一致")
        if patch_radii_in_world_units:
            if scale == 0:
                raise ValueError('点云尺度(scale)为0，无法缩放radii')
            patch_radii_arr = patch_radii_arr / scale
        patch_radii_arr = np.maximum(patch_radii_arr, 1e-12)

        if patch_radii_enforce_coverage:
            # 仍然执行一次“漏点补覆盖”，避免radii文件有疏漏导致NaN
            idx, nn_dist_list, patchRad = configure_patch_radii(
                x, y, delta=delta, patchRad=patch_radii_arr,
                feature_mask=feature_mask, feature_scale=feature_scale
            )
        else:
            # 严格按给定半径建patch，不做任何半径修改（方便做A/B对比）
            if feature_mask is not None:
                fm = np.asarray(feature_mask, dtype=bool)[:M]
                patch_radii_arr[fm] = patch_radii_arr[fm] * float(feature_scale)
            tree_x = cKDTree(x)
            idx = []
            nn_dist_list = []
            for k in range(M):
                id_list = tree_x.query_ball_point(y[k, :], patch_radii_arr[k])
                idx.append(np.array(id_list, dtype=int))
                if len(id_list) == 0:
                    nn_dist_list.append(np.array([], dtype=float))
                else:
                    dists = np.linalg.norm(x[id_list, :] - y[k, :], axis=1)
                    nn_dist_list.append(dists)
            patchRad = patch_radii_arr
            
    # ===== 做法A：曲线约束（可选）=====
    # curve_points: world坐标(默认) 或 unit box(若 curve_points_in_unit=True)
    # curve_patch_map: dict，可直接用 main_1 输出的 sharp_curve_feature_patch_map.json
    curve_scaled = None
    curve_idx_list = None

    if curve_points is not None:
        curve_points = np.asarray(curve_points, dtype=float)
        if curve_points.ndim == 2 and curve_points.shape[1] == 3 and curve_points.shape[0] > 0:
            if curve_points_in_unit:
                curve_scaled = curve_points
            else:
                curve_scaled = (curve_points - minxx) / scale

            curve_idx_list = [np.empty((0,), dtype=int) for _ in range(M)]

            if isinstance(curve_patch_map, dict) and len(curve_patch_map) > 0:
                # 优先使用预计算映射（更快）
                for kk in range(M):
                    lst = curve_patch_map.get(str(kk), curve_patch_map.get(kk, []))
                    curve_idx_list[kk] = np.asarray(lst, dtype=int)
            else:
                # 无映射时：按 patch 球域分配（默认只对 feature patch 生效）
                tree_curve = cKDTree(curve_scaled)
                fm = np.asarray(feature_mask, dtype=bool)[:M] if feature_mask is not None else None
                cap = int(curve_max_points_per_patch) if (curve_max_points_per_patch is not None) else 0

                for kk in range(M):
                    if curve_only_feature_patches and (fm is not None) and (not fm[kk]):
                        continue
                    ids = tree_curve.query_ball_point(y[kk, :], float(patchRad[kk]))
                    if cap > 0 and len(ids) > cap:
                        ids = np.asarray(ids, dtype=int)
                        d = np.linalg.norm(curve_scaled[ids, :] - y[kk, :], axis=1)
                        ids = ids[np.argsort(d)[:cap]]
                    curve_idx_list[kk] = np.asarray(ids, dtype=int)


    # -------- 生成网格/插值 --------
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
    pad = 1.0 * griddx   # 或 0.5*griddx
    startx = minx[0] - pad
    endx   = maxx[0] + pad
    starty = minx[1] - pad
    endy   = maxx[1] + pad
    startz = minx[2] - pad
    endz   = maxx[2] + pad

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

        # --- 网格点评估：CPU / GPU 两条路径 ---
        if not use_gpu:
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
        else:
            # GPU 版：主要加速 (mm x n) 的距离/核函数评估与求和
            # 为速度与显存折中，默认使用 float32；如需高精度可将环境变量 CFPU_GPU_DTYPE=float64
            dtype_env = os.environ.get('CFPU_GPU_DTYPE', 'float32').lower()
            g_dtype = cp.float64 if dtype_env in ('float64', 'f64', 'double') else cp.float32

            xe_g = cp.asarray(xe_local, dtype=g_dtype)
            xx_g = cp.asarray(xx_local, dtype=g_dtype)
            xy_g = cp.asarray(xy_local, dtype=g_dtype)
            xz_g = cp.asarray(xz_local, dtype=g_dtype)

            cx = cp.asarray(coeffsx, dtype=g_dtype)
            cy = cp.asarray(coeffsy, dtype=g_dtype)
            cz = cp.asarray(coeffsz, dtype=g_dtype)
            csp = cp.asarray(coeffsp, dtype=g_dtype)

            cc_vec = cp.asarray(coeffs_correction_vec, dtype=g_dtype)
            cc_cst = g_dtype(coeffs_correction_const)


            batch_sz = _gpu_batch_size(n, mm)
            temp_potential_g = cp.empty((mm,), dtype=g_dtype)
            potential_corr_g = cp.empty((mm,), dtype=g_dtype)

            for j in range(0, mm, batch_sz):
                xb = xe_g[j:j+batch_sz]  # (b,3)

                dxb = xb[:, 0:1] - xx_g[None, :]
                dyb = xb[:, 1:2] - xy_g[None, :]
                dzb = xb[:, 2:3] - xz_g[None, :]
                rb = cp.sqrt(dxb*dxb + dyb*dyb + dzb*dzb)  # (b,n)

                # 仅支持常用 order=1/2 的核；若自定义 kernelinfo，请走 CPU 路径
                if order == 1:
                    etab = -rb
                    phib = -rb
                else:
                    etab = rb**3
                    phib = rb**3

                Pb = poly_P_xp(xb, order, cp)  # (b,L)

                tmp = cp.sum(etab * (dxb*cx[None, :] + dyb*cy[None, :] + dzb*cz[None, :]), axis=1) + Pb @ csp
                temp_potential_g[j:j+tmp.shape[0]] = tmp

                if exactinterp:
                    potential_corr_g[j:j+tmp.shape[0]] = (phib @ cc_vec) + cc_cst
                else:
                    potential_corr_g[j:j+tmp.shape[0]] = (Pb[:, 0:3] @ cc_vec) + cc_cst

            potential_k = temp_potential_g - potential_corr_g

        patch_vec_k = np.full(mm, k + 1)
        return (idxe_k, patch_vec_k, Psi_k, potential_k)
    
    # 优化并行策略：根据任务数量动态调整并行粒度
    workers = n_jobs if (n_jobs and n_jobs > 0) else min(M, os.cpu_count() or 1)
    
    # 如果任务数量太少，使用单线程避免线程池开销
    if M < workers * 2:
        workers = 1
    
    # GPU 模式：避免 CPU 线程/多进程同时驱动 GPU（会导致 kernel 过碎+争用）
    if use_gpu:
        workers = 1

    mode_env = os.environ.get('CFPU_PARALLEL', 'thread')
    if use_gpu:
        mode_env = 'thread'
    if workers > 1 and mode_env == 'process' and SharedMemory is not None and get_context is not None:
        shm_x = SharedMemory(create=True, size=x.nbytes)
        shm_nrml = SharedMemory(create=True, size=nrml.nbytes)
        np.ndarray(x.shape, dtype=x.dtype, buffer=shm_x.buf)[:] = x
        np.ndarray(nrml.shape, dtype=nrml.dtype, buffer=shm_nrml.buf)[:] = nrml
        try:
            ctx = get_context('spawn')

            if curve_scaled is not None and getattr(curve_scaled, "size", 0) > 0:
                shm_curve = SharedMemory(create=True, size=curve_scaled.nbytes)
                np.ndarray(curve_scaled.shape, dtype=curve_scaled.dtype, buffer=shm_curve.buf)[:] = curve_scaled

            try:
                ctx = get_context('spawn')
                global _GLOBAL_X, _GLOBAL_NRML, _GLOBAL_CURVE
                _GLOBAL_X = None
                _GLOBAL_NRML = None
                _GLOBAL_CURVE = None

                with ctx.Pool(
                    processes=workers,
                    initializer=_init_proc,
                    initargs=(
                        shm_x.name, x.shape, x.dtype.str,
                        shm_nrml.name, nrml.shape, nrml.dtype.str,
                        (shm_curve.name if shm_curve is not None else None),
                        (curve_scaled.shape if shm_curve is not None else None),
                        (curve_scaled.dtype.str if shm_curve is not None else None),
                    )
                ) as pool:
                    arg_iter = (
                        (k, idx[k],
                        (curve_idx_list[k] if curve_idx_list is not None else np.empty((0,), dtype=int)),
                        nn_dist_list[k],
                        y[k, 0], y[k, 1], y[k, 2],
                        patchRad[k], order, exactinterp, nrmlreg, nrmllambda, nrmlschur,
                        trbl_id[idx[k]], potreg, potlambda,
                        startx, starty, startz, griddx, mmx, mmy, mmz)
                        for k in range(M)
                    )

                    for k, res in enumerate(pool.imap(_compute_proc, arg_iter)):
                        idxe_patch[k], patch_vec[k], Psi[k], potential_local[k] = res
                        if progress is not None:
                            try:
                                progress(k + 1, M)
                            except Exception:
                                pass
            finally:
                if shm_curve is not None:
                    shm_curve.close()
                    shm_curve.unlink()

        finally:
            shm_x.close()
            shm_x.unlink()
            shm_nrml.close()
            shm_nrml.unlink()
    elif workers > 1:
        # 优化BLAS线程设置：根据CPU核心数动态调整
        if threadpool_limits is not None:
            blas_threads_env = os.environ.get('CFPU_BLAS_THREADS')
            if blas_threads_env:
                blas_threads = int(blas_threads_env)
            else:
                # 自动设置BLAS线程数为CPU核心数的一半，避免过度竞争
                blas_threads = max(1, os.cpu_count() // 2)
            
            with threadpool_limits(blas_threads, user_api='blas'):
                # 使用更高效的并行策略：批量处理任务
                batch_size = max(1, M // (workers * 4))
                
                def compute_batch(batch_indices):
                    batch_results = []
                    for k in batch_indices:
                        batch_results.append(_compute(k))
                    return batch_results
                
                # 创建批次索引
                batches = [list(range(i, min(i + batch_size, M))) 
                          for i in range(0, M, batch_size)]
                
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    batch_results = list(ex.map(compute_batch, batches))
                    
                    # 展开批次结果
                    result_idx = 0
                    for batch in batch_results:
                        for res in batch:
                            idxe_patch[result_idx], patch_vec[result_idx], Psi[result_idx], potential_local[result_idx] = res
                            if progress is not None:
                                try:
                                    progress(result_idx + 1, M)
                                except Exception:
                                    pass
                            result_idx += 1
        else:
            # 没有threadpool_limits时的回退策略
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for k, res in enumerate(ex.map(_compute, range(M))):
                    idxe_patch[k], patch_vec[k], Psi[k], potential_local[k] = res
                    if progress is not None:
                        try:
                            progress(k + 1, M)
                        except Exception:
                            pass
    else:
        # 单线程执行
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
    if progress_stage is not None:
        try:
            progress_stage('加权-开始', None)
        except Exception:
            pass

    # -------- 加权汇总（CPU: np.add.at；GPU: cp.add.at） --------
    # 说明：原实现使用 coo_matrix 做 (m×M) 稀疏累加，内存和构建开销都很大；
    # 这里改为两遍 scatter-add：先累计 Psi_sum，再累计加权后的 potential。
    xp = cp if use_gpu else np
    if use_gpu:
        dtype_env = os.environ.get('CFPU_GPU_DTYPE', 'float32').lower()
        out_dtype = cp.float64 if dtype_env in ('float64', 'f64', 'double') else cp.float32
    else:
        out_dtype = np.float64
    
    # ====== 加权阶段：全patch Psi_sum + feature-only Psi_sum_feat ======
    Psi_sum = xp.zeros(m, dtype=out_dtype)
    Psi_sum_feat = xp.zeros(m, dtype=out_dtype) if (feature_tube_blend and (fm is not None)) else None

    # 1) 累计 Psi_sum（全patch）和 Psi_sum_feat（仅feature）
    for k in range(M):
        idxk = idxe_patch[k]
        if idxk is None or getattr(idxk, "size", 0) == 0:
            continue
        psk = Psi[k]
        if psk is None or getattr(psk, "size", 0) == 0:
            continue

        if use_gpu:
            # 关键：idxk/psk 必须是 cupy
            if not isinstance(idxk, cp.ndarray):
                idxk = cp.asarray(idxk, dtype=cp.int32)
                idxe_patch[k] = idxk
            if not isinstance(psk, cp.ndarray):
                psk = cp.asarray(psk, dtype=out_dtype)
                Psi[k] = psk
        else:
            idxk = np.asarray(idxk, dtype=np.int64)
            idxe_patch[k] = idxk
            psk = np.asarray(psk, dtype=out_dtype)
            Psi[k] = psk

        xp.add.at(Psi_sum, idxk, psk)
        if Psi_sum_feat is not None and fm[k]:
            xp.add.at(Psi_sum_feat, idxk, psk)

    tube_mask = None
    if Psi_sum_feat is not None:
        tau = out_dtype(feature_tube_tau)
        tube_mask = Psi_sum_feat > tau  # xp bool

    # 2) 累计加权后的 potential
    temp = xp.zeros(m, dtype=out_dtype)
    eps = out_dtype(1e-30)

    for k in range(M):
        idxk = idxe_patch[k]
        if idxk is None or getattr(idxk, "size", 0) == 0:
            continue
        psk = Psi[k]
        if psk is None or getattr(psk, "size", 0) == 0:
            continue
        potk = potential_local[k]
        if potk is None or getattr(potk, "size", 0) == 0:
            continue

        # 关键：potk 也必须跟 xp 一致
        if use_gpu:
            if not isinstance(potk, cp.ndarray):
                potk = cp.asarray(potk, dtype=out_dtype)
                potential_local[k] = potk
        else:
            if not isinstance(potk, np.ndarray):
                potk = np.asarray(potk, dtype=out_dtype)
                potential_local[k] = potk

        if tube_mask is not None:
            tm = tube_mask[idxk]  # xp bool
            if fm[k]:
                # feature patch：tube 内用 Psi_sum_feat 归一化；tube 外仍用 Psi_sum
                denom = xp.where(tm, Psi_sum_feat[idxk], Psi_sum[idxk])
                denom = xp.maximum(denom, eps)
                w = psk / denom
            else:
                # smooth patch：tube 内直接屏蔽（权重=0）
                denom = xp.maximum(Psi_sum[idxk], eps)
                w = psk / denom
                w = w * (~tm).astype(out_dtype)
        else:
            denom = xp.maximum(Psi_sum[idxk], eps)
            w = psk / denom

        xp.add.at(temp, idxk, potk * w)

    potential = xp.full(m, xp.nan, dtype=out_dtype)
    # ===== 全局场组装：temp 已经是 Σ potk*(Psi/Psi_sum) 的结果，不要再除 Psi_sum =====
    mask = Psi_sum > 0

    if xp.any(mask):
        potential = temp.astype(out_dtype, copy=False)

        # outside 幅值：max|.| + 1，确保远离 0
        vals = potential[mask]
        absmax = xp.max(xp.abs(vals))
        A = absmax + out_dtype(1.0)
        A = xp.where(A == 0, out_dtype(1.0), A)

        # outside 符号：优先用边界已覆盖样本的平均符号；若边界无覆盖则用 median 符号
        pot3 = potential.reshape((mmy, mmx, mmz), order='F')
        msk3 = mask.reshape((mmy, mmx, mmz), order='F')

        s = out_dtype(0.0)
        c = out_dtype(0.0)

        def _acc(v_plane, m_plane):
            nonlocal s, c
            mf = m_plane & xp.isfinite(v_plane)
            if xp.any(mf):
                s = s + xp.sum(v_plane[mf], dtype=out_dtype)
                c = c + xp.sum(mf, dtype=out_dtype)

        _acc(pot3[0, :, :],  msk3[0, :, :])
        _acc(pot3[-1, :, :], msk3[-1, :, :])
        _acc(pot3[:, 0, :],  msk3[:, 0, :])
        _acc(pot3[:, -1, :], msk3[:, -1, :])
        _acc(pot3[:, :, 0],  msk3[:, :, 0])
        _acc(pot3[:, :, -1], msk3[:, :, -1])

        # cupy 标量转 float 的兼容
        if use_gpu:
            c_host = float(cp.asnumpy(c))
        else:
            c_host = float(c)

        if c_host > 0.0:
            outside_sign = xp.sign(s)
        else:
            outside_sign = xp.sign(xp.median(vals))

        outside_sign = xp.where(outside_sign == 0, out_dtype(1.0), outside_sign)
        outside_val = outside_sign * A

        potential = potential.copy()
        potential[~mask] = outside_val

        # 关键：对全域做兜底（包括 mask 内部可能来自局部求解的 NaN/Inf）
        potential = xp.nan_to_num(potential, nan=outside_val, posinf=outside_val, neginf=-outside_val)

    else:
        outside_val = out_dtype(1.0)
        potential = xp.full(m, outside_val, dtype=out_dtype)

    potential = potential.reshape((mmy, mmx, mmz), order='F')
    if use_gpu:
        potential = cp.asnumpy(potential)

    X = X * scale + minxx[0]
    Y = Y * scale + minxx[1]
    Z = Z * scale + minxx[2]

    if progress_stage is not None:
        try:
            progress_stage('加权-完成', None)
        except Exception:
            pass


    print("[check] nan:", int(np.isnan(potential).sum()),
        "inf:", int(np.isinf(potential).sum()),
        "finite_ratio:", float(np.isfinite(potential).mean()))

    if return_transform:
        return potential, X, Y, Z, {'minxx': minxx, 'scale': scale}
    return potential, X, Y, Z
# =============================================================================


# ===== 模块导出 =====
__all__ = [
    'GPU_AVAILABLE',
    'curlfree_poly',
    'poly_P_xp',
    'weight_xp',
    '_gpu_batch_size',
    'gcv_cost_function',
    'weight',
    'configure_patch_radii',
    'cfpurecon',
    '_init_proc',
    '_compute_proc',
    'load_patch_radii_txt',
]


if __name__ == '__main__':
    print("cfpurecon - 无旋RBF分区单位隐式曲面重建模块（单文件合并版）")
    print("包含函数：curlfree_poly, gcv_cost_function, weight, configure_patch_radii, cfpurecon")
