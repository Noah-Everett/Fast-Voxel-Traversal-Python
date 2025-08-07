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
    """
    Robust slab test returning (t_near, t_far) or (inf, -inf) on miss.
    Handles d==0 by verifying origin inside slab, avoids NaN.
    """
    t0 = -math.inf
    t1 = math.inf
    for k in range(3):
        if d[k] == 0.0:
            # axis-parallel: require origin inside slab
            if o[k] < bmin[k] or o[k] > bmax[k]:
                return math.inf, -math.inf
            tn = -math.inf
            tf = math.inf
        else:
            inv = 1.0 / d[k]
            tn = (bmin[k] - o[k]) * inv
            tf = (bmax[k] - o[k]) * inv
            if tn > tf:
                tn, tf = tf, tn
        if tn > t0:
            t0 = tn
        if tf < t1:
            t1 = tf
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
    """
    Fast 3D DDA stepping. Returns number of hits and fills out_ix, out_t0, out_t1.
    """
    # 1) Optional AABB intersect
    if do_intersect:
        bmin = grid_origin
        bmax = grid_origin + voxel_size * np.array(grid_shape, dtype=np.float64)
        t_ent, t_ext = _ray_aabb_intersect(o, d, bmin, bmax)
        if t_ext < 0.0 or t_ent > t_ext:
            return 0
        t0 = t_ent if t_ent > 0.0 else 0.0
        t_exit = t_ext
    else:
        t0 = start_t
        t_exit = math.inf

    # 2) Initial voxel index
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

    # 3) Setup per-axis stepping
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

    # # 4) March the grid
    # count = 0
    # max_hits = out_ix.shape[0]
    # while count < max_hits:
    #     out_ix[count, 0] = ix[0]
    #     out_ix[count, 1] = ix[1]
    #     out_ix[count, 2] = ix[2]
    #     out_t0[count] = t0

    #     # find next boundary crossing
    #     tmin = tnext[0]
    #     if tnext[1] < tmin:
    #         tmin = tnext[1]
    #     if tnext[2] < tmin:
    #         tmin = tnext[2]
    #     out_t1[count] = tmin

    #     count += 1
    #     t0 = tmin
    #     if t0 > t_exit or t0 > t_max:
    #         break

    #     # step axes that hit simultaneously
    #     for k in range(3):
    #         if tnext[k] == tmin:
    #             ix[k] += step[k]
    #             if ix[k] < 0 or ix[k] >= grid_shape[k]:
    #                 return count
    #             tnext[k] += dt[k]

    # 4) March the grid
    count = 0
    max_hits = out_ix.shape[0]
    while count < max_hits:
        out_ix[count, 0] = ix[0]
        out_ix[count, 1] = ix[1]
        out_ix[count, 2] = ix[2]
        out_t0[count] = t0

        # ---- ❶ choose NEXT boundary exactly like C++ -----------------
        # strict-< comparisons give the same tie-breaking order:
        #   • X wins only when strictly smallest
        #   • Y beats X when X==Y < Z
        #   • Z wins on any remaining ties (X==Y==Z or X<Y==Z)
        if tnext[0] < tnext[1] and tnext[0] < tnext[2]:
            ksel = 0        # advance X
        elif tnext[1] < tnext[2]:
            ksel = 1        # advance Y
        else:
            ksel = 2        # advance Z
        tmin = tnext[ksel]
        # -----------------------------------------------------------------

        out_t1[count] = tmin
        count += 1
        t0 = tmin
        if t0 > t_exit or t0 > t_max:
            break

        # ---- ❷ advance only the selected axis ---------------------------
        ix[ksel] += step[ksel]
        if ix[ksel] < 0 or ix[ksel] >= grid_shape[ksel]:
            return count          # left the grid → finished
        tnext[ksel] += dt[ksel]
        # -----------------------------------------------------------------

    return count