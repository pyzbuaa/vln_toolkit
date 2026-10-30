"""A* in cell space, with world-space output and no diagonal corner cutting."""

import heapq
import math

import numpy as np


class PathNotFound(ValueError):
    pass


def path_length(route) -> float:
    return float(np.linalg.norm(np.diff(np.asarray(route, dtype=float), axis=0), axis=1).sum())


class AStarPlanner:
    def __init__(self, occupancy, *, allow_diagonal=True):
        self.occupancy = occupancy
        self.allow_diagonal = allow_diagonal

    def neighbors(self, cell):
        row, col = cell
        for dr, dc in [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            *([(1, 1), (1, -1), (-1, 1), (-1, -1)] if self.allow_diagonal else []),
        ]:
            target = row + dr, col + dc
            if not self.occupancy.is_free_cell(target):
                continue
            if (
                dr
                and dc
                and (
                    not self.occupancy.is_free_cell((row + dr, col))
                    or not self.occupancy.is_free_cell((row, col + dc))
                )
            ):
                continue
            yield target, math.hypot(dr, dc)

    def plan(self, start, goal, *, min_length=0.0, max_length=math.inf):
        if min_length < 0 or max_length < min_length:
            raise ValueError("Expected 0 <= min_length <= max_length")
        occupancy = self.occupancy
        for position in (start, goal):
            if len(position) not in (2, 3) or not np.isfinite(position).all():
                raise ValueError("Positions must be finite XY or XYZ coordinates")
            if len(position) == 3 and not math.isclose(
                position[2], occupancy.floor_z, abs_tol=1e-6
            ):
                raise ValueError("Position is on a different floor than the 2D occupancy map")
        start_cell, goal_cell = occupancy.world_to_grid(start), occupancy.world_to_grid(goal)
        if not occupancy.is_free_cell(start_cell) or not occupancy.is_free_cell(goal_cell):
            raise PathNotFound("Start and goal must be within free map cells")
        frontier = [(0.0, 0.0, start_cell)]
        costs, parents = {start_cell: 0.0}, {}
        while frontier:
            _, cost, current = heapq.heappop(frontier)
            if cost > costs[current]:
                continue
            if current == goal_cell:
                cells = [current]
                while current in parents:
                    current = parents[current]
                    cells.append(current)
                route = [occupancy.grid_to_world(cell) for cell in reversed(cells)]
                start_point = (float(start[0]), float(start[1]), occupancy.floor_z)
                goal_point = (float(goal[0]), float(goal[1]), occupancy.floor_z)
                if start_cell == goal_cell:
                    route = (
                        [start_point] if start_point == goal_point else [start_point, goal_point]
                    )
                else:
                    if route[0] != start_point:
                        route.insert(0, start_point)
                    if route[-1] != goal_point:
                        route.append(goal_point)
                length = path_length(route)
                if not min_length <= length <= max_length:
                    raise PathNotFound(
                        f"Shortest path length {length:.3f} violates configured limits"
                    )
                return route
            for target, step_cost in self.neighbors(current):
                next_cost = cost + step_cost
                if next_cost < costs.get(target, math.inf):
                    costs[target], parents[target] = next_cost, current
                    heuristic = math.dist(target, goal_cell)
                    heapq.heappush(frontier, (next_cost + heuristic, next_cost, target))
        raise PathNotFound("No reachable path between start and goal")
