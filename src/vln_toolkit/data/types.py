from dataclasses import asdict, dataclass, field

import numpy as np

from vln_toolkit.utils.coordinates import yaw_to_xyzw


@dataclass
class NavigationTask:
    start: tuple[float, float, float]
    goal: tuple[float, float, float]
    start_yaw: float = 0.0
    task_id: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class NavigationEpisode:
    episode_id: str
    scene_id: str
    instruction: str
    start_pose: dict
    goal: dict
    reference_path: list
    map_id: str | None = None
    task_type: str = "pointnav"
    route_id: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.episode_id or not self.scene_id or not self.instruction:
            raise ValueError("Episode ID, scene ID and instruction must not be empty")
        route = np.asarray(self.reference_path, dtype=float)
        if route.ndim != 2 or route.shape[1] != 3 or not len(route) or not np.isfinite(route).all():
            raise ValueError("reference_path must be a nonempty finite Nx3 world-space route")
        start = np.asarray(self.start_pose["position"], dtype=float)
        goal = np.asarray(self.goal["position"], dtype=float)
        rotation = np.asarray(self.start_pose["rotation_xyzw"], dtype=float)
        if start.shape != (3,) or goal.shape != (3,):
            raise ValueError("Start and goal must have three coordinates")
        if (
            rotation.shape != (4,)
            or not np.isfinite(rotation).all()
            or not np.isclose(np.linalg.norm(rotation), 1)
        ):
            raise ValueError("start_pose rotation must be a unit xyzw quaternion")
        if not np.allclose(route[0], start) or not np.allclose(route[-1], goal):
            raise ValueError("Reference route endpoints must match the task")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


def build_episode(task, route, instruction, *, scene_id, episode_id=None, map_id=None):
    return NavigationEpisode(
        episode_id=episode_id or task.task_id,
        scene_id=scene_id,
        instruction=instruction,
        start_pose={"position": list(task.start), "rotation_xyzw": yaw_to_xyzw(task.start_yaw)},
        goal={"type": "point", "position": list(task.goal)},
        reference_path=[list(point) for point in route],
        map_id=map_id,
        metadata=task.metadata.copy(),
    )
