import math

import numpy as np
import pytest

from fast_voxel_traversal import Grid, traverse_ray, traverse_until_hit


def test_grid_legacy_creation_defaults():
    g = Grid(grid_shape=(2, 2, 2))
    assert g.shape == (2, 2, 2)
    assert np.allclose(g.voxel_size, np.ones(3))
    assert np.allclose(g.origin, np.zeros(3))


def test_grid_legacy_creation_custom_voxel_and_origin():
    g = Grid(grid_shape=(4, 5, 6),
             voxel_size=(2.0, 3.0, 4.0),
             grid_origin=(1.0, 2.0, 3.0))
    assert g.shape == (4, 5, 6)
    assert np.allclose(g.voxel_size, np.array([2.0, 3.0, 4.0]))
    assert np.allclose(g.origin, np.array([1.0, 2.0, 3.0]))


def test_grid_bounds_shape_mode():
    bounds = ((0.0, 0.0, 0.0), (10.0, 20.0, 30.0))
    shape = (5, 4, 3)
    g = Grid(bounds=bounds, shape=shape)
    assert g.shape == shape
    # origin == min corner
    assert np.allclose(g.origin, np.array([0.0, 0.0, 0.0]))
    # voxel sizes = (10/5, 20/4, 30/3) = (2,5,10)
    assert np.allclose(g.voxel_size, np.array([2.0, 5.0, 10.0]))


def test_grid_invalid_constructor():
    with pytest.raises(ValueError):
        Grid()  # must pick legacy or bounds+shape


def test_entry_time_miss_and_hit():
    g = Grid(grid_shape=(2, 2, 2))
    # Miss entirely
    t = g.entry_time((10.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert math.isinf(t)

    # Hit at x=0 face
    t = g.entry_time((-1.0, 0.5, 0.5), (1.0, 0.0, 0.0))
    assert t == pytest.approx(1.0)


def test_traverse_simple_diagonal_legacy():
    g = Grid(grid_shape=(4, 4, 4))
    voxels = list(g.traverse((-1, -1, -1), (1, 1, 1)))
    expected = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)]
    assert [v[:3] for v in voxels] == expected


def test_traverse_simple_diagonal_bounds_shape():
    bounds = ((0.0, 0.0, 0.0), (4.0, 4.0, 4.0))
    shape = (4, 4, 4)
    g = Grid(bounds=bounds, shape=shape)
    voxels = list(g.traverse((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    expected = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)]
    assert [v[:3] for v in voxels] == expected


def test_traverse_start_t_outside_grid():
    g = Grid(grid_shape=(3, 3, 3))
    origin = (10.0, 10.0, 10.0)
    direction = (-1.0, -1.0, -1.0)
    # start_t = 9 brings us to point (1,1,1) inside the grid
    voxels = list(g.traverse(origin, direction, start_t=9.0))
    assert voxels[0][:3] == (1, 1, 1)


def test_traverse_until_hit_legacy_and_bounds():
    arr = np.zeros((5, 5, 5), dtype=int)
    arr[2, 3, 1] = 1

    # Legacy
    g1 = Grid(grid_shape=(5, 5, 5))
    hit1 = g1.traverse_until_hit((0.0, 0.0, 0.0),
                                 (2.5, 4.5, 1.5),
                                 arr)
    assert hit1[:3] == (2, 3, 1)

    # Bounds+shape
    bounds = ((0.0, 0.0, 0.0), (5.0, 5.0, 5.0))
    shape = (5, 5, 5)
    g2 = Grid(bounds=bounds, shape=shape)
    hit2 = g2.traverse_until_hit((0.0, 0.0, 0.0),
                                 (2.5, 4.5, 1.5),
                                 arr)
    assert hit2[:3] == (2, 3, 1)


def test_top_level_helpers_equivalent():
    origin = (0.0, 0.0, 0.0)
    direction = (1.0, 0.0, 0.0)
    arr = np.zeros((10, 10, 10), dtype=int)
    arr[3, 0, 0] = 1

    hit1 = traverse_until_hit(origin,
                              direction,
                              arr,
                              voxel_size=1.0)
    hit2 = traverse_until_hit(origin,
                              direction,
                              arr)  # defaults
    assert hit1 == hit2

    voxels1 = list(traverse_ray(origin,
                                (0.0, 1.0, 0.0),
                                grid_shape=(5, 5, 5)))
    # Should be marching along Y at x=0,z=0
    assert [v[:3] for v in voxels1] == [(0, j, 0) for j in range(5)]

def test_legacy_and_bounds_grid_equivalence():
    # Create a grid with legacy constructor
    g1 = Grid(grid_shape=(3, 3, 3), voxel_size=1.0, grid_origin=(0.0, 0.0, 0.0))
    # Create a grid with bounds and shape
    bounds = ((0.0, 0.0, 0.0), (3.0, 3.0, 3.0))
    shape = (3, 3, 3)
    g2 = Grid(bounds=bounds, shape=shape)

    assert g1.shape == g2.shape
    assert np.allclose(g1.voxel_size, g2.voxel_size)
    assert np.allclose(g1.origin, g2.origin)