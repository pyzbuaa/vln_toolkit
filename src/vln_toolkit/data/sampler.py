import math
import warnings

import numpy as np

from .planner import AStarPlanner, PathNotFound, path_length
from .types import NavigationTask


class RandomTaskSampler:
    def __init__(
        self, occupancy, *, seed=None, min_distance=1.0, max_distance=math.inf, allow_diagonal=True
    ):
        if min_distance < 0 or max_distance < min_distance:
            raise ValueError("Expected 0 <= min_distance <= max_distance")
        self.occupancy, self.seed = occupancy, seed
        self.rng = np.random.default_rng(seed)
        self.min_distance, self.max_distance = min_distance, max_distance
        self.planner = AStarPlanner(occupancy, allow_diagonal=allow_diagonal)

    def sample(self, count=100, *, max_attempts=None) -> list[NavigationTask]:
        if count < 0:
            raise ValueError("count must be nonnegative")
        if count == 0:
            return []
        free_cells = np.argwhere(self.occupancy.grid == 0)
        if len(free_cells) < 2:
            raise ValueError("At least two free map cells are required")
        max_attempts = max_attempts if max_attempts is not None else max(100, count * 100)
        tasks = []
        for _ in range(max_attempts):
            start_idx, goal_idx = self.rng.choice(len(free_cells), 2, replace=False)
            start = self.occupancy.grid_to_world(free_cells[start_idx])
            goal = self.occupancy.grid_to_world(free_cells[goal_idx])
            if math.dist(start, goal) > self.max_distance:
                continue
            try:
                route = self.planner.plan(
                    start, goal, min_length=self.min_distance, max_length=self.max_distance
                )
            except PathNotFound:
                continue
            tasks.append(
                NavigationTask(
                    start,
                    goal,
                    float(self.rng.uniform(-math.pi, math.pi)),
                    f"ep_{len(tasks):06d}",
                    {"seed": self.seed, "path_length": path_length(route)},
                )
            )
            if len(tasks) == count:
                return tasks
        warnings.warn(
            f"Sampled {len(tasks)}/{count} tasks after {max_attempts} attempts; "
            "check connectivity and distance constraints",
            UserWarning,
            stacklevel=2,
        )
        return tasks
