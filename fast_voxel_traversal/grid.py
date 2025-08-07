"""
User‑facing Grid class and top‑level helpers.
"""
import math
from typing import Iterable, Optional, Sequence, Tuple, Generator, Union

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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

    def time_grid(self,
                  origin: Iterable[float],
                  direction: Iterable[float],
                  *,
                  use_time: bool = True,
                  start_t: Optional[float] = None,
                  t_max: float = math.inf) -> np.ndarray:
        """
        Rasterize ray traversal into a 3D array of entry times or ones.

        Parameters:
        - origin, direction: ray parameters
        - use_time: if True fill with entry times; if False fill with 1.0
        - start_t, t_max: traversal parameters

        Returns a numpy array shaped like the grid, with NaN for missed voxels.
        """
        times = np.full(self.shape, np.nan, dtype=float)
        for ix, iy, iz, t_enter, _ in self.traverse(origin, direction, start_t=start_t, t_max=t_max):
            times[ix, iy, iz] = t_enter if use_time else 1.0
        return times

    # ---------------------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------------------
    def plot(
        self,
        grid_array: np.ndarray,
        ax: Optional[Axes3D] = None,
        cmap: str = 'viridis',
        edgecolor: str = 'k',
        set_limits: bool = True,
        show: bool = True,
    ) -> Axes3D:
        """Plot occupied voxels in 3D using matplotlib, respecting grid origin and size."""
        mask = grid_array.astype(bool)
        nx, ny, nz = self.shape

        # Compute corner coordinates
        xs = self.origin[0] + np.arange(nx + 1) * self.voxel_size[0]
        ys = self.origin[1] + np.arange(ny + 1) * self.voxel_size[1]
        zs = self.origin[2] + np.arange(nz + 1) * self.voxel_size[2]
        # Create coordinate grids for voxel corners
        xv, yv, zv = np.meshgrid(xs, ys, zs, indexing='ij')

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        # Determine facecolors for occupied voxels
        facecolors = plt.get_cmap(cmap)(grid_array.astype(float))
        # Plot voxels with explicit coordinates
        ax.voxels(xv, yv, zv, mask, facecolors=facecolors, edgecolor=edgecolor)

        # Set limits to grid bounds
        if set_limits:
            ax.set_xlim(self.origin[0], self.origin[0] + nx * self.voxel_size[0])
            ax.set_ylim(self.origin[1], self.origin[1] + ny * self.voxel_size[1])
            ax.set_zlim(self.origin[2], self.origin[2] + nz * self.voxel_size[2])

        if show:
            plt.show()
        return ax




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


def time_grid(
    origin,
    direction,
    grid_array=None,
    *,
    grid_shape=None,
    voxel_size=1.0,
    grid_origin=(0.0, 0.0, 0.0),
    bounds=None,
    shape=None,
    use_time: bool = True,
    start_t=None,
    t_max=math.inf
) -> np.ndarray:
    """
    Rasterize ray traversal into a 3D array of entry times or ones.

    If grid_array is provided, its shape infers grid dimensions; otherwise specify grid_shape or bounds+shape.
    """
    # Determine grid shape
    if grid_array is not None and bounds is None and shape is None:
        grid_shape = _infer_shape_from_array(grid_array)
    # Build grid
    grid = Grid(
        grid_shape=grid_shape,
        voxel_size=voxel_size,
        grid_origin=grid_origin,
        bounds=bounds,
        shape=shape,
    )
    return grid.time_grid(origin, direction, use_time=use_time, start_t=start_t, t_max=t_max)
