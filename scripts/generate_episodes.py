import argparse

from vln_toolkit.data import save_episodes
from vln_toolkit.data.generation import generate_episodes, generation_metadata
from vln_toolkit.representations import OccupancyMap
from vln_toolkit.scenes import load_scene
from vln_toolkit.utils.config import load_config
from vln_toolkit.utils.visualization import save_map_svg


def main():
    parser = argparse.ArgumentParser(description="Generate reachable navigation episodes offline")
    parser.add_argument("--map", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--scene", help="Optional USD/scene.json for source-revision verification")
    parser.add_argument("--config", default="configs/generation.yaml")
    parser.add_argument("--count", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    config["scene_id"] = args.scene_id
    config["map_id"] = args.map
    if args.count is not None:
        config["count"] = args.count
    if args.seed is not None:
        config["seed"] = args.seed
    scene = load_scene(args.scene) if args.scene else None
    occupancy = OccupancyMap.load(args.map, scene=scene)
    map_scene_id = occupancy.metadata.get("scene_id")
    if map_scene_id and map_scene_id != args.scene_id:
        raise ValueError(f"Map belongs to {map_scene_id}, not {args.scene_id}")
    episodes = generate_episodes(occupancy, **config)
    save_episodes(episodes, args.output, generation_metadata(config, occupancy))
    save_map_svg(
        occupancy, f"{args.output}/routes.svg", [ep.reference_path for ep in episodes[:20]]
    )
    print(f"Saved {len(episodes)} episodes to {args.output}")


if __name__ == "__main__":
    main()
