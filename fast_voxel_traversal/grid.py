"""
User‑facing Grid class and top‑level helpers.
"""
import math
from typing import Iterable, Optional, Sequence, Tuple, Generator, Union

import numpy as np
from ._core import _ray_aabb_intersect, _dda


class Grid:
    """Axis‑aligned 3‑D voxel grid."""

    def __init__(
        self,
        *,
        grid_shape: Optional[Tuple[int, int, int]] = None,
        voxel_size: Union[float, Tuple[float, float, float]] = 1.0,
        grid_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
        shape: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        # Choose representation mode
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

    # ---------------------------------------------------------------------
    # Analytics
    # ---------------------------------------------------------------------
    def entry_time(self, origin: Iterable[float], direction: Iterable[float]) -> float:
        """Return *t_enter* or +inf if the ray misses this grid's AABB."""
        o = np.asarray(origin, dtype=np.float64)
        d = np.asarray(direction, dtype=np.float64)
        bmin = self.origin
        bmax = self.origin + self.voxel_size * np.array(self.shape, dtype=np.float64)
        t0, t1 = _ray_aabb_intersect(o, d, bmin, bmax)
        # no hit if the exit time is before the ray start or infinite entry
        if t1 < 0.0 or t0 == math.inf:
            return math.inf
        return float(t0)

    # ---------------------------------------------------------------------
    # Traversal generators
    # ---------------------------------------------------------------------
    def traverse(
        self,
        origin: Iterable[float],
        direction: Iterable[float],
        start_t: Optional[float] = None,
        t_max: float = math.inf,
    ) -> Generator[Tuple[int, int, int, float, float], None, None]:
        """Yield ``(ix, iy, iz, t_enter, t_exit)`` for each crossed voxel."""
        o = np.asarray(origin, dtype=np.float64)
        d = np.asarray(direction, dtype=np.float64)
        do_intersect = start_t is None
        t0 = 0.0 if start_t is None else float(start_t)

        max_vox = sum(self.shape) + 3  # safe upper bound
        buf_ix = np.empty((max_vox, 3), dtype=np.int64)
        buf_t0 = np.empty(max_vox, dtype=np.float64)
        buf_t1 = np.empty(max_vox, dtype=np.float64)

        hits = _dda(
            o,
            d,
            tuple(self.shape),
            self.voxel_size,
            self.origin,
            float(t_max),
            t0,
            do_intersect,
            buf_ix,
            buf_t0,
            buf_t1,
        )

        for i in range(hits):
            yield (
                int(buf_ix[i, 0]),
                int(buf_ix[i, 1]),
                int(buf_ix[i, 2]),
                float(buf_t0[i]),
                float(buf_t1[i]),
            )

    def traverse_until_hit(
        self,
        origin: Iterable[float],
        direction: Iterable[float],
        grid_array: np.ndarray,
        start_t: Optional[float] = None,
        t_max: float = math.inf,
    ) -> Optional[Tuple[int, int, int, float]]:
        """Return first occupied cell or *None* if no hit before *t_max*."""
        for ix, iy, iz, t0, _ in self.traverse(origin, direction, start_t, t_max):
            if grid_array[ix, iy, iz]:
                return ix, iy, iz, t0
        return None


# -------------------------------------------------------------------------
# Convenience top‑level helpers
# -------------------------------------------------------------------------

def _infer_shape_from_array(arr: np.ndarray) -> Tuple[int, int, int]:
    if arr.ndim != 3:
        raise ValueError("grid_array must be 3‑D")
    return tuple(arr.shape)


def traverse_ray(
    origin,
    direction,
    *,
    grid_shape=None,
    voxel_size=1.0,
    grid_origin=(0.0, 0.0, 0.0),
    bounds=None,
    shape=None,
    start_t=None,
    t_max=math.inf,
):
    grid = Grid(
        grid_shape=grid_shape,
        voxel_size=voxel_size,
        grid_origin=grid_origin,
        bounds=bounds,
        shape=shape,
    )
    return grid.traverse(origin, direction, start_t=start_t, t_max=t_max)


def traverse_until_hit(
    origin,
    direction,
    grid_array,
    *,
    voxel_size=1.0,
    grid_origin=(0.0, 0.0, 0.0),
    bounds=None,
    shape=None,
    start_t=None,
    t_max=math.inf,
):
    # If no explicit bounds/shape, infer grid_shape from the array
    if bounds is None and shape is None:
        grid_shape = _infer_shape_from_array(grid_array)
    else:
        grid_shape = None  # let Grid constructor validate parameters

    grid = Grid(
        grid_shape=grid_shape,
        voxel_size=voxel_size,
        grid_origin=grid_origin,
        bounds=bounds,
        shape=shape,
    )
    return grid.traverse_until_hit(
        origin, direction, grid_array, start_t=start_t, t_max=t_max
    )