import math
from itertools import pairwise

import numpy as np
import pytest

from vln_toolkit.data import (
    AStarPlanner,
    NavigationEpisode,
    NavigationTask,
    PathNotFound,
    RandomTaskSampler,
    TemplateInstructionGenerator,
    build_episode,
    load_episodes,
    save_episodes,
)
from vln_toolkit.data.generation import generate_episodes
from vln_toolkit.data.planner import path_length
from vln_toolkit.representations import OccupancyMap
from vln_toolkit.scenes import Scene
from vln_toolkit.utils.coordinates import stage_to_world, transform_points, xyzw_to_yaw, yaw_to_xyzw


def test_map_coordinates_roundtrip_and_metadata(tmp_path):
    grid = np.array([[0, 1, 255], [0, 0, 0]], dtype=np.uint8)
    occupancy = OccupancyMap(grid, 0.25, (-2, 3), floor_z=2.4)
    assert occupancy.world_to_grid((-2.01, 3)) == (0, -1)
    assert not occupancy.is_free((-2.01, 3))
    assert not occupancy.is_free(occupancy.grid_to_world((0, 2)))
    for row, col in np.ndindex(grid.shape):
        assert occupancy.world_to_grid(occupancy.grid_to_world((row, col))) == (row, col)
    occupancy.save(tmp_path)
    restored = OccupancyMap.load(tmp_path)
    np.testing.assert_array_equal(restored.grid, grid)
    assert restored.origin == (-2, 3) and restored.floor_z == 2.4


@pytest.mark.parametrize(
    "grid,resolution", [([[0, 2]], 1), ([[0]], 0), ([[0]], float("nan")), ([], 1)]
)
def test_invalid_maps(grid, resolution):
    with pytest.raises(ValueError):
        OccupancyMap(grid, resolution, (0, 0))


def test_inflation_blocks_unknown_and_boundary():
    grid = np.zeros((7, 7), np.uint8)
    grid[3, 3] = 255
    occupancy = OccupancyMap(grid, 1, (0, 0))
    inflated = occupancy.inflated(1)
    assert inflated.grid[3, 3] == 255
    assert inflated.grid[3, 2] == 1
    assert inflated.grid[0, 1] == 1
    assert inflated.grid[2, 2] == 0
    assert occupancy.grid[3, 2] == 0


def test_astar_detours_and_preserves_elevated_floor():
    grid = np.zeros((6, 6), np.uint8)
    grid[0:5, 3] = 1
    occupancy = OccupancyMap(grid, 0.5, (-1, 2), floor_z=3.25)
    start, goal = occupancy.grid_to_world((0, 0)), occupancy.grid_to_world((0, 5))
    planner = AStarPlanner(occupancy)
    route = planner.plan(start, goal)
    assert route[0] == start and route[-1] == goal
    assert all(point[2] == 3.25 and occupancy.is_free(point) for point in route)
    assert path_length(route) > math.dist(start, goal)
    cells = [occupancy.world_to_grid(point) for point in route]
    for current, target in pairwise(cells):
        assert target in [cell for cell, _ in planner.neighbors(current)]
    with pytest.raises(PathNotFound, match="limits"):
        planner.plan(start, goal, max_length=1)
    with pytest.raises(ValueError, match="floor"):
        planner.plan((start[0], start[1], 0), goal)


def test_astar_no_corner_cutting_or_unknown_traversal():
    occupancy = OccupancyMap([[0, 1], [255, 0]], 1, (0, 0))
    with pytest.raises(PathNotFound):
        AStarPlanner(occupancy).plan((0.5, 0.5), (1.5, 1.5))
    with pytest.raises(PathNotFound):
        AStarPlanner(occupancy).plan((-0.5, 0.5), (0.5, 0.5))


def test_astar_exact_endpoints_in_single_cell():
    planner = AStarPlanner(OccupancyMap([[0]], 1, (0, 0)))
    route = planner.plan((0.1, 0.2), (0.8, 0.7))
    assert route == [(0.1, 0.2, 0), (0.8, 0.7, 0)]


def test_sampler_reproducibility_and_reachable_path_limits():
    grid = np.zeros((8, 8), np.uint8)
    grid[:, 4] = 1
    occupancy = OccupancyMap(grid, 0.5, (0, 0), floor_z=2)
    kwargs = {"seed": 11, "min_distance": 1, "max_distance": 2}
    tasks = RandomTaskSampler(occupancy, **kwargs).sample(20)
    assert tasks == RandomTaskSampler(occupancy, **kwargs).sample(20)
    for task in tasks:
        length = path_length(AStarPlanner(occupancy).plan(task.start, task.goal))
        assert 1 <= length <= 2
        assert task.start[2] == 2


def test_impossible_sampler_stops_without_fabricating_tasks():
    occupancy = OccupancyMap([[0, 1, 0]], 1, (0, 0))
    with pytest.warns(UserWarning, match="0/5"):
        assert RandomTaskSampler(occupancy).sample(5, max_attempts=10) == []


def test_template_initial_orientation_and_turn_geometry():
    route = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0)]
    task = NavigationTask(route[0], route[-1], start_yaw=math.pi / 2)
    text = TemplateInstructionGenerator().generate(route, task)
    assert "向右转90度" in text and "向左转90度" in text
    assert "2.00米" in text and "1.00米" in text
    assert all(word not in text for word in ["沙发", "卧室", "走廊"])


def test_episode_jsonl_roundtrip_and_invalid_endpoints(tmp_path):
    task = NavigationTask((0, 0, 1), (1, 0, 1), task_id="episode")
    episode = build_episode(task, [task.start, task.goal], "前进。", scene_id="room")
    save_episodes([episode], tmp_path, {"seed": 42})
    assert load_episodes(tmp_path) == [episode]
    bad = episode.to_dict()
    bad["reference_path"] = [[3, 0, 1]]
    with pytest.raises(ValueError, match="endpoints"):
        NavigationEpisode.from_dict(bad)
    with pytest.raises(ValueError, match="Duplicate"):
        save_episodes([episode, episode], tmp_path)


def test_generation_loop():
    occupancy = OccupancyMap(np.zeros((10, 10)), 0.5, (0, 0), floor_z=5)
    episodes = generate_episodes(occupancy, scene_id="room", count=5, seed=9)
    assert len(episodes) == 5
    assert all(episode.reference_path[0][2] == 5 for episode in episodes)


def test_axis_transform_is_right_handed_and_quaternion_order():
    transform = stage_to_world("Y", 0.01)
    np.testing.assert_allclose(transform_points([[100, 200, 300]], transform), [[1, -3, 2]])
    assert np.linalg.det(transform[:3, :3]) > 0
    assert xyzw_to_yaw(yaw_to_xyzw(1.2)) == pytest.approx(1.2)


def test_map_load_automatically_detects_changed_usd_dependency(tmp_path):
    source, layer = tmp_path / "scene.usda", tmp_path / "furniture.usda"
    source.write_text("#usda 1.0\n")
    layer.write_text("#usda 1.0\n")
    scene = Scene("room", str(source), {"layers": [str(source), str(layer)]})
    occupancy = OccupancyMap([[0]], 1, (0, 0), metadata={"scene_source": scene.source_record()})
    occupancy.save(tmp_path / "map")
    OccupancyMap.load(tmp_path / "map")
    layer.write_text("#usda 1.0\n# changed dependency\n")
    with pytest.warns(UserWarning, match="revision"):
        OccupancyMap.load(tmp_path / "map")
