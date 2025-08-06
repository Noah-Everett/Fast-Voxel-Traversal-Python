# Fast-Voxel-Traversal-Python

A blazing-fast, pure-Python implementation of the Amanatides–Woo voxel traversal algorithm, accelerated with [Numba](https://numba.pydata.org/). Ideal for ray-marching, volume rendering, collision detection, and any application that needs to walk through a 3D voxel grid in $O(n_x + n_y + n_z)$ time per ray.


---

## Features

- **Amanatides–Woo DDA**  
  Fast, branch-minimal grid walking as described in [Amanatides & Woo '87](https://dl.acm.org/doi/10.5555/37402.37406).
- **Numba JIT**  
  Inner loop compiled to native machine code for C-like performance.
- **Pure-Python API**  
  No C extensions or build steps—just install and import.
- **Flexible parameters**  
  Arbitrary voxel size, grid origin, `t_max`, and scalar or per-axis ray directions.
- **"Stop on hit" helper**  
  `traverse_until_hit()` to quickly find the first occupied voxel in a 3D mask.

---

## Setup

Clone this repository and install dependencies:

```bash
git clone https://github.com/Noah-Everett/Fast-Voxel-Traversal-Python.git
cd Fast-Voxel-Traversal-Python
# Install required packages
pip install numpy numba
```

Dependencies: Python 3.9+, NumPy >=1.23, Numba >=0.59.

---

## Usage

```python
import numpy as np
from fast_voxel_traversal import traverse_ray, traverse_until_hit

# Define a grid of size 128³, unit voxels, origin at (0,0,0)
grid_shape  = (128, 128, 128)
voxel_size  = 1.0
grid_origin = (0.0, 0.0, 0.0)

# Ray parameters
origin    = (-0.5, 0.2,  0.1)
direction = ( 1.0, 0.3,  0.1)

```python
# 1) Iterate through every voxel crossed by the ray
for ix, iy, iz, t_enter, t_exit in traverse_ray(
        origin, direction,
        grid_shape=grid_shape,
        voxel_size=voxel_size,
        grid_origin=grid_origin,
        t_max=100.0
    ):
    # e.g. sample volume density at (ix,iy,iz)
    density = ...  

# 2) Find first occupied voxel in a binary mask
mask = np.zeros(grid_shape, dtype=bool)
mask[50, 60, 70] = True

hit = traverse_until_hit(
    origin, direction, mask,
    voxel_size=voxel_size,
    grid_origin=grid_origin
)

if hit is not None:
    ix, iy, iz, t_hit = hit
    print(f"Hit voxel {(ix,iy,iz)} at t = {t_hit:.3f}")
else:
    print("No hit within t_max")
```

---

## API Reference

### `traverse_ray(origin, direction, *, grid_shape=None, voxel_size=1.0, grid_origin=(0,0,0), bounds=None, shape=None, start_t=None, t_max=∞)`

**Returns:** generator

Yields tuples: `(ix, iy, iz, t_enter, t_exit)`

**Parameters:**
- `origin`: 3-tuple or array of floats. Ray start point in world coordinates.
- `direction`: 3-tuple or array of floats. Ray direction vector.
- `grid_shape`: (nx, ny, nz) dimensions of the voxel grid.
- `voxel_size`: scalar or 3-tuple. Physical size of each voxel along each axis.
- `grid_origin`: 3-tuple. World-space coordinate of voxel (0,0,0).
- `t_max`: float. Parametric ray length limit; stops traversal beyond this t.

---

### `traverse_until_hit(origin, direction, grid_array, *, voxel_size=1.0, grid_origin=(0,0,0), bounds=None, shape=None, start_t=None, t_max=∞)`

**Returns:** `(ix, iy, iz, t_hit)` or `None`

Scans a 3D boolean/bitmask grid and returns the first non-zero voxel 'hit'. Returns None if no occupied cell is found within t_max.


## Class `Grid`

### Constructor
```python
Grid(*, grid_shape=None, voxel_size=1.0, grid_origin=(0,0,0), bounds=None, shape=None)
```
Create a 3D axis-aligned voxel grid.

Parameters:
- `grid_shape`: tuple(int, int, int), number of voxels along each axis (legacy mode).
- `voxel_size`: float or tuple(float, float, float), physical size of each voxel.
- `grid_origin`: tuple(float, float, float), world-space coordinate of voxel (0,0,0).
- `bounds`: (min_pt, max_pt), world-space bounding box of the grid.
- `shape`: tuple(int, int, int), number of voxels when `bounds` is specified.

### Methods
- `entry_time(origin, direction) -> float`
  Compute parametric entry `t` where the ray first intersects the grid AABB, or +∞ if it misses.
  - `origin`: 3-tuple or array of floats.
  - `direction`: 3-tuple or array of floats.
- `traverse(origin, direction, *, start_t=None, t_max=∞) -> generator`
  Iterate over all voxels crossed by the ray.
  Yields `(ix, iy, iz, t_enter, t_exit)` for each voxel.
- `traverse_until_hit(origin, direction, grid_array, *, start_t=None, t_max=∞) -> Optional[tuple]`
  Return first occupied voxel in `grid_array`, or `None` if none before `t_max`.
  - `grid_array`: 3D boolean or integer numpy array serving as occupancy mask.

## Class `Ray`

### Constructor
```python
Ray(origin, direction)
```
Encapsulate ray parameters.

Parameters:
- `origin`: 3-tuple of floats.
- `direction`: 3-tuple of floats.

### Methods
- `entry_time(grid: Grid) -> float`
  Return `t` where this ray first intersects the grid AABB, or +∞ if it misses.
- `traverse(grid: Grid, *, start_t=None, t_max=∞) -> generator`
  Delegate to `grid.traverse`, yields `(ix, iy, iz, t_enter, t_exit)`.
- `traverse_until_hit(grid: Grid, grid_array, *, start_t=None, t_max=∞) -> Optional[tuple]`
  Delegate to `grid.traverse_until_hit`, returns first hit or `None`.

## Testing

Unit tests use pytest.

```bash
# run all tests
pytest -q
```

Example tests cover:
- Diagonal sweep through the main grid diagonal
- Negative-direction rays
- First-hit behavior on a small Boolean mask

---

## Performance Tips

- **Warm up Numba**: The first call to `traverse_ray()` incurs compilation overhead (~10–20 ms). Subsequent calls run at full speed (sub-microsecond per ray).
- **Batching**: For many thousands of rays, consider `@njit(parallel=True)` loops over ray arrays.
- **GPU**: Easily port inner loops to CuPy or Numba's CUDA target for massive parallelism.

---

## License

MIT License

---

## Acknowledgements

- J. Amanatides & A. Woo, "A Fast Voxel Traversal Algorithm for Ray Tracing," Eurographics '87.
- Implementation guidance inspired by modern blog posts and NVIDIA DevBlog.
