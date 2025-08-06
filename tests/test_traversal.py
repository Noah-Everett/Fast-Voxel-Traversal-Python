import numpy as np
from fast_voxel_traversal import traverse_ray, traverse_until_hit


def test_simple_diagonal():
    grid_shape = (4, 4, 4)
    voxels = list(traverse_ray((-1, -1, -1), (1, 1, 1), grid_shape))
    # Should visit all voxels on main diagonal
    expected = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)]
    assert [v[:3] for v in voxels] == expected


def test_negative_direction():
    grid_shape = (3, 3, 3)
    voxels = list(traverse_ray((2.9, 2.9, 2.9), (-1, -1, -1), grid_shape,
                               voxel_size=1.0, grid_origin=(0, 0, 0)))
    assert voxels[0][:3] == (2, 2, 2)
    assert voxels[-1][:3] == (0, 0, 0)


def test_until_hit():
    grid = np.zeros((8, 8, 8), dtype=np.uint8)
    grid[3, 5, 2] = 1
    hit = traverse_until_hit((-10, 0, 2.5), (1, 0, 0), grid)
    assert hit[:3] == (3, 5, 2)
    # Ray starts outside, travels 13.0 world units to voxel boundary
    assert abs(hit[3] - 13.0) < 1e-6