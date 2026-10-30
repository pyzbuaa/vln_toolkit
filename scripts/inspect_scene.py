"""Inspect a USD scene and optionally save its reusable scene.json descriptor."""

import argparse

from vln_toolkit.scenes import load_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--scene-id")
    parser.add_argument("--output", help="Output scene.json path; USD assets are not copied")
    args = parser.parse_args()
    scene = load_scene(args.scene, meta={"scene_id": args.scene_id} if args.scene_id else None)
    print(f"Scene ID: {scene.scene_id}")
    print(f"USD: {scene.usd_path}")
    for key in ("up_axis", "meters_per_unit", "has_collision", "has_semantics"):
        print(f"{key}: {scene.metadata[key]}")
    print(f"USD layers: {len(scene.metadata.get('layers', []))}")
    if args.output:
        scene.save(args.output)
        print(f"Saved scene descriptor to {args.output}")


if __name__ == "__main__":
    main()
