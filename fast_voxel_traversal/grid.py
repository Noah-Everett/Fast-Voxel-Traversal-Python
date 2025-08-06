"""
User-facing Grid class: define your grid once, then reuse for multiple rays.
"""
import math
from typing import Iterable, Optional, Sequence, Tuple, Generator, Union

import numpy as np
from ._core import _ray_aabb_intersect, _dda

class Grid:
    def __init__(
        self,
        *,
        grid_shape: Optional[Tuple[int, int, int]] = None,
        voxel_size: Union[float, Tuple[float, float, float]] = 1.0,
        grid_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
        shape: Optional[Tuple[int, int, int]] = None
    ):
        # choose legacy or bounds+shape
        if bounds is not None and shape is not None:
            gmin = np.asarray(bounds[0], dtype=np.float64)
            gmax = np.asarray(bounds[1], dtype=np.float64)
            self.shape = tuple(int(x) for x in shape)
            self.voxel_size = (gmax - gmin) / np.array(self.shape, dtype=np.float64)
            self.origin = gmin
        elif grid_shape is not None:
            self.shape = tuple(int(x) for x in grid_shape)
            if np.ndim(voxel_size):
                self.voxel_size = np.asarray(voxel_size, dtype=np.float64)
            else:
                self.voxel_size = np.full(3, float(voxel_size), dtype=np.float64)
            self.origin = np.asarray(grid_origin, dtype=np.float64)
        else:
            raise ValueError("Must specify either grid_shape or bounds+shape.")

    def entry_time(
        self,
        origin: Iterable[float],
        direction: Iterable[float]
    ) -> float:
        """
        Parametric t where ray first intersects grid AABB, or +inf if missed.
        """
        o = np.asarray(origin, dtype=np.float64)
        d = np.asarray(direction, dtype=np.float64)
        bmin = self.origin
        bmax = self.origin + self.voxel_size * np.array(self.shape, dtype=np.float64)
        t0, _ = _ray_aabb_intersect(o, d, bmin, bmax)
        return float(t0)

    def traverse(
        self,
        origin: Iterable[float],
        direction: Iterable[float],
        start_t: Optional[float] = None,
        t_max: float = math.inf
    ) -> Generator[Tuple[int,int,int,float,float], None, None]:
        """
        Yield (ix,iy,iz,t_enter,t_exit) for each voxel.
        If start_t is None, march from entry point; else begin at start_t.
        """
        o = np.asarray(origin, dtype=np.float64)
        d = np.asarray(direction, dtype=np.float64)
        do_intersect = start_t is None
        t0 = 0.0 if start_t is None else float(start_t)
        max_vox = self.shape[0] + self.shape[1] + self.shape[2] + 3
        buf_ix = np.empty((max_vox, 3), dtype=np.int64)
        buf_t0 = np.empty(max_vox, dtype=np.float64)
        buf_t1 = np.empty(max_vox, dtype=np.float64)
        hits = _dda(o, d,
                    tuple(self.shape),
                    self.voxel_size,
                    self.origin,
                    float(t_max),
                    t0,
                    do_intersect,
                    buf_ix, buf_t0, buf_t1)
        for i in range(hits):
            yield (int(buf_ix[i,0]), int(buf_ix[i,1]), int(buf_ix[i,2]),
                   float(buf_t0[i]), float(buf_t1[i]))

    def traverse_until_hit(
        self,
        origin: Iterable[float],
        direction: Iterable[float],
        grid_array: np.ndarray,
        start_t: Optional[float] = None,
        t_max: float = math.inf
    ) -> Optional[Tuple[int,int,int,float]]:
        """
        Return first non-zero cell (ix,iy,iz,t_hit) or None.
        """
        for ix,iy,iz,t0,_ in self.traverse(origin, direction, start_t, t_max):
            if grid_array[ix,iy,iz]:
                return ix, iy, iz, t0
        return None


def traverse_ray(
    origin, direction,
    *,
    grid_shape=None, voxel_size=1.0, grid_origin=(0.0,0.0,0.0),
    bounds=None, shape=None,
    start_t=None, t_max=math.inf
):
    """Top-level helper: instantiate Grid and call traverse."""
    grid = Grid(
        grid_shape=grid_shape, voxel_size=voxel_size, grid_origin=grid_origin,
        bounds=bounds, shape=shape
    )
    return grid.traverse(origin, direction, start_t=start_t, t_max=t_max)


def traverse_until_hit(
    origin, direction, grid_array,
    *,
    voxel_size=1.0, grid_origin=(0.0,0.0,0.0),
    bounds=None, shape=None,
    start_t=None, t_max=math.inf
):
    """Top-level helper: instantiate Grid and call traverse_until_hit."""
    grid = Grid(
        grid_shape=None, voxel_size=voxel_size, grid_origin=grid_origin,
        bounds=bounds, shape=shape
    )
    return grid.traverse_until_hit(origin, direction, grid_array, start_t=start_t, t_max=t_max)