import numpy as np
import pytest

from vln_toolkit.data import KinematicExpert, NavigationTask, TrajectoryRecorder, build_episode
from vln_toolkit.sim.isaac_runtime import occupancy_from_positions


def test_kinematic_state_action_alignment_and_lossless_depth(tmp_path):
    task = NavigationTask((0, 0, 2), (1, 0, 2), start_yaw=1, task_id="episode")
    episode = build_episode(task, [task.start, task.goal], "前进。", scene_id="room")
    trajectory = KinematicExpert(step_size=0.25, dt=0.1).execute(episode)
    assert trajectory.mode == "kinematic" and len(trajectory.poses) == 5
    assert trajectory.poses[0] == episode.start_pose
    for index, action in enumerate(trajectory.actions[:-1]):
        assert action["position"] == trajectory.poses[index + 1]["position"]
        assert action["rotation_xyzw"] == trajectory.poses[index + 1]["rotation_xyzw"]
    assert trajectory.actions[-1] == {"type": "stop"}
    trajectory.observations["depth"] = np.full((5, 2, 3), 1.234567, dtype=np.float32)
    recorder = TrajectoryRecorder(tmp_path)
    recorder.save(episode.episode_id, trajectory)
    restored = recorder.load(episode.episode_id)
    assert restored.poses == trajectory.poses and restored.actions == trajectory.actions
    np.testing.assert_array_equal(restored.observations["depth"], trajectory.observations["depth"])
    trajectory.observations.clear()
    recorder.save(episode.episode_id, trajectory)
    assert recorder.load(episode.episode_id).observations == {}


def test_trajectory_invalid_timestamps_and_path_ids(tmp_path):
    task = NavigationTask((0, 0, 0), (1, 0, 0), task_id="episode")
    trajectory = KinematicExpert().execute(
        build_episode(task, [task.start, task.goal], "走", scene_id="s")
    )
    with pytest.raises(ValueError, match="episode ID"):
        TrajectoryRecorder(tmp_path).save("../episode", trajectory)
    trajectory.timestamps[1] = trajectory.timestamps[0]
    with pytest.raises(ValueError, match="increasing"):
        trajectory.validate()


def test_native_positions_are_rasterized_without_flipping_axes():
    free = [(1, 2, 0.3), (1.5, 2, 0.3), (1, 3, 0.3)]
    occupied = [(2, 2.5, 0.3)]
    occupancy = occupancy_from_positions(free, occupied, 0.5, 0)
    assert occupancy.origin == (0.75, 1.75)
    assert occupancy.grid.shape == (3, 3)
    assert occupancy.is_free((1, 3))
    assert occupancy.grid[1, 2] == 1 and occupancy.grid[1, 1] == 255
