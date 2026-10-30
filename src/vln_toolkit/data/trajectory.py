"""Ideal kinematic replay is explicitly separated from physical execution."""

import math
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np

from vln_toolkit.utils.coordinates import yaw_to_xyzw


@dataclass
class Trajectory:
    episode_id: str
    timestamps: list
    poses: list
    actions: list
    observations: dict = field(default_factory=dict)
    mode: str = "kinematic"
    status: str = "completed"
    metadata: dict = field(default_factory=dict)

    def validate(self):
        count = len(self.timestamps)
        if not count or len(self.poses) != count or len(self.actions) != count:
            raise ValueError("Timestamps, poses and actions must have equal nonzero length")
        times = np.asarray(self.timestamps, dtype=float)
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("Timestamps must be finite and strictly increasing")
        if self.mode not in {"kinematic", "physical"}:
            raise ValueError("Trajectory mode must be kinematic or physical")
        for pose in self.poses:
            position = np.asarray(pose["position"], float)
            rotation = np.asarray(pose["rotation_xyzw"], float)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError("Pose position must be finite XYZ")
            if (
                rotation.shape != (4,)
                or not np.isfinite(rotation).all()
                or not np.isclose(np.linalg.norm(rotation), 1)
            ):
                raise ValueError("Pose rotation must be a unit xyzw quaternion")
        if any(len(frames) != count for frames in self.observations.values()):
            raise ValueError("Every observation channel must match the state timestamps")


class KinematicExpert:
    def __init__(self, *, step_size=0.25, dt=0.1):
        if not np.isfinite([step_size, dt]).all() or step_size <= 0 or dt <= 0:
            raise ValueError("step_size and dt must be positive and finite")
        self.step_size, self.dt = step_size, dt

    def execute(self, episode):
        route = np.asarray(episode.reference_path, float)
        points = [route[0]]
        for start, goal in pairwise(route):
            steps = math.ceil(float(np.linalg.norm(goal - start)) / self.step_size)
            points.extend(start + (goal - start) * i / steps for i in range(1, steps + 1))
        poses, actions = [], []
        # Preserve the actual task orientation in the first state. Each ideal move
        # updates both position and heading; no physical turn dynamics are implied.
        rotation = episode.start_pose["rotation_xyzw"]
        for index, point in enumerate(points):
            poses.append({"position": point.tolist(), "rotation_xyzw": list(rotation)})
            if index + 1 < len(points):
                delta = points[index + 1] - point
                rotation = yaw_to_xyzw(math.atan2(delta[1], delta[0]))
                actions.append(
                    {
                        "type": "move_to",
                        "position": points[index + 1].tolist(),
                        "rotation_xyzw": rotation,
                    }
                )
            else:
                actions.append({"type": "stop"})
        result = Trajectory(
            episode.episode_id,
            [i * self.dt for i in range(len(points))],
            poses,
            actions,
            metadata={
                "algorithm": "ideal_reference_replay",
                "step_size": self.step_size,
                "dt": self.dt,
                "sensors_collected": False,
            },
        )
        result.validate()
        return result


class PhysicalExpert:
    def execute(self, episode):
        raise NotImplementedError("Connect a robot controller and sensors through vln_toolkit.sim")
