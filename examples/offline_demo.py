"""Run a synthetic-map data loop without Isaac Sim or pxr."""

import argparse
from pathlib import Path

import numpy as np

from vln_toolkit.data import KinematicExpert, TrajectoryRecorder, load_episodes, save_episodes
from vln_toolkit.data.generation import generate_episodes, generation_metadata
from vln_toolkit.representations import OccupancyMap, SceneGraphBuilder
from vln_toolkit.utils.visualization import save_map_svg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="workspace/demo")
    args = parser.parse_args()
    output = Path(args.output)
    grid = np.zeros((40, 60), dtype=np.uint8)
    grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1] = 1, 1, 1, 1
    grid[1:30, 30] = 1
    grid[15:21, 10:17] = 1
    grid[1:5, 1:6] = 255
    occupancy = OccupancyMap(
        grid,
        0.2,
        (-6, -4),
        floor_z=1.5,
        metadata={"scene_id": "synthetic_room", "source": "synthetic_demo"},
    ).inflated(0.2)
    occupancy.save(output / "occupancy")
    config = {
        "scene_id": "synthetic_room",
        "count": 10,
        "seed": 42,
        "min_distance": 2.0,
        "max_distance": 20.0,
    }
    episodes = generate_episodes(occupancy, **config)
    save_episodes(episodes, output / "dataset", generation_metadata(config, occupancy))
    recorder = TrajectoryRecorder(output / "dataset")
    for episode in load_episodes(output / "dataset"):
        recorder.save(episode.episode_id, KinematicExpert().execute(episode))
    save_map_svg(occupancy, output / "routes.svg", [ep.reference_path for ep in episodes])
    graph = SceneGraphBuilder().build(
        None,
        {
            "backend": "records",
            "rooms": {"living": {"polygon": [[-6, -4], [6, -4], [6, 4], [-6, 4]]}},
            "objects": [
                {
                    "instance_id": "table",
                    "category": "table",
                    "min_points": [1, 1, 1.5],
                    "max_points": [2, 2, 2.3],
                },
                {
                    "instance_id": "cup",
                    "category": "cup",
                    "min_points": [1.4, 1.4, 2.3],
                    "max_points": [1.6, 1.6, 2.5],
                },
            ],
        },
    )
    graph.save(output / "scene_graph")
    print(
        f"Saved {len(episodes)} episodes, kinematic replays, scene graph and routes.svg to {output}"
    )
    print("This example uses a synthetic map; it does not validate native Isaac Sim mapping.")


if __name__ == "__main__":
    main()
