"""
Low-level NumPy+Numba helpers for slab intersection and 3D DDA.
"""
import math
import numpy as np
from numba import njit, int64, float64
from typing import Tuple

@njit(fastmath=True, cache=True)
def _ray_aabb_intersect(o: np.ndarray, d: np.ndarray,
                        bmin: np.ndarray, bmax: np.ndarray) -> Tuple[float, float]:
    t0, t1 = -np.inf, np.inf
    for k in range(3):
        inv = 1.0 / d[k] if d[k] != 0.0 else math.inf
        tn = (bmin[k] - o[k]) * inv
        tf = (bmax[k] - o[k]) * inv
        if tn > tf:
            tn, tf = tf, tn
        t0 = tn if tn > t0 else t0
        t1 = tf if tf < t1 else t1
        if t0 > t1:
            return math.inf, -math.inf
    return t0, t1

@njit(cache=True, fastmath=True)
def _dda(o: np.ndarray, d: np.ndarray,
         grid_shape: Tuple[int64, int64, int64],
         voxel_size: np.ndarray,
         grid_origin: np.ndarray,
         t_max: float,
         start_t: float,
         do_intersect: bool,
         out_ix: np.ndarray,
         out_t0: np.ndarray,
         out_t1: np.ndarray) -> int:
    # optional AABB test
    if do_intersect:
        bmin = grid_origin
        bmax = grid_origin + voxel_size * np.array(grid_shape, dtype=np.float64)
        t_ent, t_ext = _ray_aabb_intersect(o, d, bmin, bmax)
        if t_ent > t_ext or t_ext < 0.0:
            return 0
        t0 = t_ent if t_ent > 0.0 else 0.0
        t_exit = t_ext
    else:
        t0 = start_t
        t_exit = math.inf

    # initial voxel index
    p = o + d * t0
    ix = np.empty(3, dtype=int64)
    for k in range(3):
        rel = (p[k] - grid_origin[k]) / voxel_size[k]
        idx = int(rel)
        if idx < 0:
            idx = 0
        elif idx >= grid_shape[k]:
            idx = grid_shape[k] - 1
        ix[k] = idx

    # per-axis stepping info
    step = np.empty(3, dtype=int64)
    tnext = np.empty(3, dtype=float64)
    dt = np.empty(3, dtype=float64)
    for k in range(3):
        if d[k] > 0.0:
            step[k] = 1
            boundary = grid_origin[k] + (ix[k] + 1) * voxel_size[k]
            tnext[k] = (boundary - o[k]) / d[k]
            dt[k] = voxel_size[k] / d[k]
        elif d[k] < 0.0:
            step[k] = -1
            boundary = grid_origin[k] + ix[k] * voxel_size[k]
            tnext[k] = (boundary - o[k]) / d[k]
            dt[k] = -voxel_size[k] / d[k]
        else:
            step[k] = 0
            tnext[k] = math.inf
            dt[k] = math.inf

    # 3D-DDA walk
    count = 0
    while count < out_ix.shape[0]:
        out_ix[count, 0] = ix[0]
        out_ix[count, 1] = ix[1]
        out_ix[count, 2] = ix[2]
        out_t0[count] = t0
        # find next crossing
        tmin = tnext[0]
        if tnext[1] < tmin: tmin = tnext[1]
        if tnext[2] < tmin: tmin = tnext[2]
        out_t1[count] = tmin
        count += 1
        t0 = tmin
        if t0 > t_exit or t0 > t_max:
            break
        # step all axes that match tmin
        for k in range(3):
            if tnext[k] == tmin:
                ix[k] += step[k]
                if ix[k] < 0 or ix[k] >= grid_shape[k]:
                    return count
                tnext[k] += dt[k]
    return count