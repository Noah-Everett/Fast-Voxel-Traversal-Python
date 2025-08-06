import math

import numpy as np
import pytest

from fast_voxel_traversal import Ray, Grid


def test_ray_entry_and_traverse():
    ray = Ray(origin=(-1.0, -1.0, -1.0),
              direction=(1.0, 1.0, 1.0))
    g = Grid(grid_shape=(3, 3, 3))

    # entry at t=1 → (0,0,0)
    t0 = ray.entry_time(g)
    assert t0 == pytest.approx(1.0)

    voxels = list(ray.traverse(g))
    expected = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
    assert [v[:3] for v in voxels] == expected


def test_ray_traverse_until_hit_and_reversed():
    ray = Ray(origin=(0.0, 0.0, 0.0),
              direction=(1.0, 0.0, 0.0))
    arr = np.zeros((5, 5, 5), dtype=int)
    arr[2, 0, 0] = 1

    g = Grid(grid_shape=(5, 5, 5))
    hit = ray.traverse_until_hit(g, arr)
    assert hit[:3] == (2, 0, 0)

    # reversed ray: start beyond the hit cell, go backward
    ray_rev = Ray(origin=(5.0, 0.0, 0.0),
                  direction=(-1.0, 0.0, 0.0))
    hit_rev = ray_rev.traverse_until_hit(g, arr)
    assert hit_rev[:3] == (2, 0, 0)
    # t for reverse = (5−3)=2
    assert hit_rev[3] == pytest.approx(2.0)


def test_ray_with_start_t_override():
    # if start_t is provided, disable AABB test
    ray = Ray(origin=(10.0, 10.0, 10.0),
              direction=(-1.0, -1.0, -1.0))
    g = Grid(grid_shape=(4, 4, 4))
    # choose start_t so that position = (2,2,2)
    start_t = 8.0
    voxels = list(ray.traverse(g, start_t=start_t))
    assert voxels[0][:3] == (2, 2, 2)