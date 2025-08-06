"""
Ray class encapsulating origin and direction, with traversal helpers.
"""
from typing import Iterable, Optional
from .grid import Grid

class Ray:
    def __init__(self, origin: Iterable[float], direction: Iterable[float]):
        self.origin = tuple(float(x) for x in origin)
        self.direction = tuple(float(x) for x in direction)

    def entry_time(self, grid: Grid) -> float:
        """Return parametric t where this ray first hits the grid."""
        return grid.entry_time(self.origin, self.direction)

    def traverse(
        self,
        grid: Grid,
        start_t: Optional[float] = None,
        t_max: float = float('inf')
    ):
        """Delegate to Grid.traverse."""
        return grid.traverse(self.origin, self.direction, start_t=start_t, t_max=t_max)

    def traverse_until_hit(
        self,
        grid: Grid,
        grid_array,
        start_t: Optional[float] = None,
        t_max: float = float('inf')
    ):
        """Delegate to Grid.traverse_until_hit."""
        return grid.traverse_until_hit(self.origin, self.direction, grid_array, start_t=start_t, t_max=t_max)