"""
Numba‑accelerated implementation of

    J. Amanatides & A. Woo,
    "A Fast Voxel Traversal Algorithm for Ray Tracing" (Eurographics ’87)

Public API
----------
traverse_ray(origin, direction, grid_shape, ...)
    – generator yielding every visited voxel (ix, iy, iz, t_enter, t_exit)

traverse_until_hit(origin, direction, grid, ...)
    – stop at first occupied grid cell and return its index + travel distance
"""
from __future__ import annotations

import math
from typing import Generator, Iterable, Tuple

import numpy as np
from numba import njit, int64, float64

# --------------------------------------------------------------------------- #
# Helper – axis‑aligned bounding‑box ray test (slab method, numerically stable)
# --------------------------------------------------------------------------- #

@njit(fastmath=True, cache=True)
def _ray_aabb_intersect(o: np.ndarray, d: np.ndarray,
                        box_min: np.ndarray, box_max: np.ndarray
                        ) -> Tuple[float, float]:
    """Return (t_near, t_far) or (np.inf, -np.inf) if the box is missed."""
    t0, t1 = -np.inf, np.inf
    for k in range(3):
        inv = 1.0 / d[k] if d[k] != 0.0 else math.inf
        t_near = (box_min[k] - o[k]) * inv
        t_far  = (box_max[k] - o[k]) * inv
        if t_near > t_far:                        # swap for negative dirs
            t_near, t_far = t_far, t_near
        t0 = t_near if t_near > t0 else t0       # max near
        t1 = t_far  if t_far  < t1 else t1       # min far
        if t0 > t1:                              # early exit – missed
            return np.inf, -np.inf
    return t0, t1


# --------------------------------------------------------------------------- #
# Core 3‑D DDA in 100 % Numba
# --------------------------------------------------------------------------- #

@njit(cache=True, fastmath=True)
def _dda(o: np.ndarray, d: np.ndarray,
         grid_shape: Tuple[int64, int64, int64],
         voxel_size: np.ndarray,
         grid_origin: np.ndarray,
         t_max: float,
         out_ix: np.ndarray,
         out_t0: np.ndarray,
         out_t1: np.ndarray
         ) -> int:
    """
    Fast voxel traversal à la Amanatides–Woo.

    Fills pre‑allocated output buffers (`out_ix`, `out_t0`, `out_t1`)
    and returns the number of valid entries.
    """
    # 1) Intersect ray with the whole grid’s AABB
    bounds_min = grid_origin
    bounds_max = grid_origin + voxel_size * np.array(grid_shape, dtype=np.float64)
    t_enter, t_exit = _ray_aabb_intersect(o, d, bounds_min, bounds_max)
    if t_enter > t_exit or t_exit < 0.0:
        return 0                                 # the ray misses the grid

    if t_enter < 0.0:                            # start inside grid
        t_enter = 0.0

    # 2) Locate the initial voxel
    p = o + d * t_enter                          # entry point
    ix = np.empty(3, dtype=int64)
    for k in range(3):
        rel = (p[k] - grid_origin[k]) / voxel_size[k]
        # Clamp because of possible numeric overshoot
        ix[k] = min(max(int(rel), 0), grid_shape[k] - 1)

    # 3) Pre‑compute per‑axis step + next boundary + delta‑t
    step  = np.empty(3, dtype=int64)
    t_next = np.empty(3, dtype=float64)
    dt     = np.empty(3, dtype=float64)

    for k in range(3):
        if d[k] > 0.0:
            step[k]  = 1
            next_voxel_boundary = grid_origin[k] + (ix[k] + 1.0) * voxel_size[k]
            t_next[k] = (next_voxel_boundary - o[k]) / d[k]
            dt[k] = voxel_size[k] / d[k]
        elif d[k] < 0.0:
            step[k]  = -1
            next_voxel_boundary = grid_origin[k] + ix[k] * voxel_size[k]
            t_next[k] = (next_voxel_boundary - o[k]) / d[k]
            dt[k] = -voxel_size[k] / d[k]
        else:                                    # parallel to axis
            step[k]  = 0
            t_next[k] = math.inf
            dt[k] = math.inf

    # 4) Walk the grid
    n_hits = 0
    while n_hits < out_ix.shape[0]:
        out_ix[n_hits, 0] = ix[0]
        out_ix[n_hits, 1] = ix[1]
        out_ix[n_hits, 2] = ix[2]
        out_t0[n_hits]    = t_enter
        t_shortest = t_next[0]
        axis = 0
        if t_next[1] < t_shortest:
            t_shortest = t_next[1]
            axis = 1
        if t_next[2] < t_shortest:
            t_shortest = t_next[2]
            axis = 2
        out_t1[n_hits] = t_shortest

        n_hits += 1
        t_enter = t_shortest
        if t_enter > t_exit or t_enter > t_max:
            break

        ix[axis] += step[axis]
        if ix[axis] < 0 or ix[axis] >= grid_shape[axis]:
            break                                # left the grid
        t_next[axis] += dt[axis]

    return n_hits


# --------------------------------------------------------------------------- #
# User‑facing wrapper – Pythonic generator
# --------------------------------------------------------------------------- #

def traverse_ray(origin:      Iterable[float],
                 direction:   Iterable[float],
                 grid_shape:  Tuple[int, int, int],
                 *,
                 voxel_size:  float | Tuple[float, float, float] = 1.0,
                 grid_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 t_max: float = math.inf
                 ) -> Generator[Tuple[int, int, int, float, float], None, None]:
    """
    Yield `(ix, iy, iz, t_enter, t_exit)` for every voxel intersected by the ray.

    Parameters
    ----------
    origin, direction : array‑like(3)
    grid_shape        : (nx, ny, nz)
    voxel_size        : scalar or (sx, sy, sz)
    grid_origin       : world coord of voxel (0, 0, 0)
    t_max             : stop traversal after this parametric length
    """
    o = np.asarray(origin,   dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    g_shape = tuple(int(x) for x in grid_shape)
    v_size = (np.asarray(voxel_size, dtype=np.float64)
              if np.ndim(voxel_size) else
              np.full(3, voxel_size, dtype=np.float64))
    g_origin = np.asarray(grid_origin, dtype=np.float64)

    # Worst‑case steps = sum(grid_shape) << ∏(grid_shape), so allocate safely
    max_voxels = g_shape[0] + g_shape[1] + g_shape[2] + 3
    ix_buf  = np.empty((max_voxels, 3), dtype=np.int64)
    t0_buf  = np.empty(max_voxels,     dtype=np.float64)
    t1_buf  = np.empty(max_voxels,     dtype=np.float64)

    hits = _dda(o, d, g_shape, v_size, g_origin, t_max,
                ix_buf, t0_buf, t1_buf)
    for i in range(hits):
        yield (int(ix_buf[i, 0]), int(ix_buf[i, 1]), int(ix_buf[i, 2]),
               float(t0_buf[i]),   float(t1_buf[i]))


# --------------------------------------------------------------------------- #
# Convenience helper that stops at the first occupied voxel
# --------------------------------------------------------------------------- #

def traverse_until_hit(origin:      Iterable[float],
                       direction:   Iterable[float],
                       grid:        np.ndarray,
                       *,
                       voxel_size:  float | Tuple[float, float, float] = 1.0,
                       grid_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                       t_max: float = math.inf
                       ) -> Tuple[int, int, int, float] | None:
    """
    Return `(ix, iy, iz, t_hit)` for the first non‑zero cell in *grid*,
    or *None* if no hit occurs before *t_max*.
    """
    for ix, iy, iz, t0, _ in traverse_ray(origin, direction,
                                          grid.shape, voxel_size=voxel_size,
                                          grid_origin=grid_origin, t_max=t_max):
        if grid[ix, iy, iz]:
            return ix, iy, iz, t0
    return None