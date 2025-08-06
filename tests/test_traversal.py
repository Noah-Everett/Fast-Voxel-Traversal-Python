import numpy as np
import pytest

from fast_voxel_traversal import traverse_ray, traverse_until_hit


def test_simple_diagonal():
    grid_shape = (4, 4, 4)
    voxels = list(traverse_ray((-1, -1, -1), (1, 1, 1), grid_shape=grid_shape))
    expected = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)]
    assert [v[:3] for v in voxels] == expected


def test_negative_direction():
    grid_shape = (3, 3, 3)
    voxels = list(traverse_ray((2.9, 2.9, 2.9), (-1, -1, -1),
                               grid_shape=grid_shape,
                               voxel_size=1.0,
                               grid_origin=(0, 0, 0)))
    assert voxels[0][:3] == (2, 2, 2)
    assert voxels[-1][:3] == (0, 0, 0)


def test_until_hit():
    grid = np.zeros((8, 8, 8), dtype=np.uint8)
    grid[3, 5, 2] = 1
    hit = traverse_until_hit((-10, 5, 2.5), (1, 0, 0), grid)
    assert hit[:3] == (3, 5, 2)
    assert hit[3] == pytest.approx(13.0)


def test_non_unit_grid_and_reversed_direction():
    """
    Non-unit voxels + arbitrary origin, plus the reversed-ray case.
    """
    origin = (1.5, 0.8, -25.0)
    target = (6.0, 1.7, -25.0)
    direction = (target[0] - origin[0],
                 target[1] - origin[1],
                 target[2] - origin[2])

    grid_shape = (4, 2, 1)
    voxel_size = (1.0, 1.0, 35.0)
    grid_origin = (0.0, 0.0, -50.0)

    voxels = list(traverse_ray(origin, direction,
                               grid_shape=grid_shape,
                               voxel_size=voxel_size,
                               grid_origin=grid_origin))
    expected = [(1, 0, 0), (2, 0, 0), (2, 1, 0), (3, 1, 0)]
    assert [v[:3] for v in voxels] == expected

    # reversed-ray should visit the same cells in reverse order—and then exit at (0,0,0)
    origin2 = target
    direction2 = (-direction[0], -direction[1], -direction[2])
    voxels2 = list(traverse_ray(origin2, direction2,
                                grid_shape=grid_shape,
                                voxel_size=voxel_size,
                                grid_origin=grid_origin))
    expected_rev = expected[::-1] + [(0, 0, 0)]
    assert [v[:3] for v in voxels2] == expected_rev


def test_axis_aligned_rays():
    """
    Rays parallel to each principal axis should visit exactly one 1-D “line” of voxels,
    but only if the ray’s other coordinates already lie inside the grid.
    """
    grid_shape = (5, 5, 5)

    # X-axis from inside grid
    voxels_x = list(traverse_ray(
        origin=(0.5, 0.5, 0.5), direction=(1.0, 0.0, 0.0), grid_shape=grid_shape
    ))
    assert [v[:3] for v in voxels_x] == [(i, 0, 0) for i in range(5)]

    # Y-axis from inside grid (x=2.5 in voxel 2, z=3.2 in voxel 3)
    voxels_y = list(traverse_ray(
        origin=(2.5, -1.0, 3.2), direction=(0.0, 1.0, 0.0), grid_shape=grid_shape
    ))
    assert voxels_y[0][:3] == (2, 0, 3)
    assert [v[:3] for v in voxels_y] == [(2, j, 3) for j in range(5)]

    # Z-axis from inside grid (x=0.5 in voxel 0, y=4.0 in voxel 4)
    voxels_z = list(traverse_ray(
        origin=(0.5, 4.0, 0.0), direction=(0.0, 0.0, 1.0), grid_shape=grid_shape
    ))
    assert voxels_z[0][:3] == (0, 4, 0)
    assert [v[:3] for v in voxels_z] == [(0, 4, k) for k in range(5)]


def test_miss_entire_grid():
    """
    A ray that starts and points away from the grid yields no voxels.
    """
    grid_shape = (5, 5, 5)
    voxels = list(traverse_ray(
        origin=(10.0, 10.0, 10.0),
        direction=(1.0, 0.0, 0.0),
        grid_shape=grid_shape
    ))
    assert voxels == []


def test_traverse_until_hit_arbitrary_direction():
    """
    First occupied cell hit in an oblique direction: t_enter = max(tx, ty, tz).
    """
    shape = (5, 5, 5)
    grid = np.zeros(shape, dtype=bool)
    grid[2, 3, 1] = True

    origin = (0.0, 0.0, 0.0)
    direction = (2.5, 4.5, 1.5)
    hit = traverse_until_hit(origin, direction, grid)
    assert hit is not None
    ix, iy, iz, t0 = hit
    assert (ix, iy, iz) == (2, 3, 1)

    tx = 2.0 / direction[0]  # 0.8
    ty = 3.0 / direction[1]  # ~0.6667
    tz = 1.0 / direction[2]  # ~0.6667
    expected_t0 = max(tx, ty, tz)
    assert t0 == pytest.approx(expected_t0)