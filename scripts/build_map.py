import argparse

from vln_toolkit.representations import OccupancyBuilder
from vln_toolkit.scenes import load_scene
from vln_toolkit.sim.isaac_runtime import IsaacRuntime
from vln_toolkit.utils.config import load_config
from vln_toolkit.utils.visualization import save_map_svg


def main():
    parser = argparse.ArgumentParser(description="Build native Isaac Sim collision occupancy")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--config", default="configs/scene.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    with IsaacRuntime(disable_viewport_updates=True) as runtime:
        scene = load_scene(args.scene)
        runtime.load(scene)
        occupancy = OccupancyBuilder().build(scene, config.get("occupancy", config))
        occupancy.save(args.output)
        save_map_svg(occupancy, f"{args.output}/map.svg")
        print(f"Saved occupancy map to {args.output}", flush=True)


if __name__ == "__main__":
    main()
