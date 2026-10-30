"""Plain functions connect sampling, planning and templates; no pipeline engine."""

import platform

import numpy as np

from .instruction import TemplateInstructionGenerator
from .planner import AStarPlanner
from .sampler import RandomTaskSampler
from .types import build_episode


def generate_episodes(
    occupancy,
    *,
    scene_id,
    count=100,
    seed=42,
    min_distance=1.0,
    max_distance=20.0,
    language="zh",
    allow_diagonal=True,
    map_id=None,
):
    sampler = RandomTaskSampler(
        occupancy,
        seed=seed,
        min_distance=min_distance,
        max_distance=max_distance,
        allow_diagonal=allow_diagonal,
    )
    tasks = sampler.sample(count)
    if len(tasks) != count:
        raise RuntimeError(f"Only {len(tasks)}/{count} reachable tasks satisfy the constraints")
    planner = AStarPlanner(occupancy, allow_diagonal=allow_diagonal)
    generator = TemplateInstructionGenerator(language=language)
    episodes = []
    for task in tasks:
        route = planner.plan(task.start, task.goal)
        episodes.append(
            build_episode(
                task, route, generator.generate(route, task), scene_id=scene_id, map_id=map_id
            )
        )
    return episodes


def generation_metadata(config, occupancy):
    return {
        "algorithm": "random_reachable_astar_template",
        "config": config,
        "map_metadata": occupancy.metadata,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }
