"""Extract USD semantics or import existing VLNTube scene graphs."""

import argparse

from vln_toolkit.representations import SceneGraphBuilder
from vln_toolkit.scenes import load_scene
from vln_toolkit.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--scene", help="USD file or scene.json; requires pxr")
    inputs.add_argument("--vlntube", help="Directory with VLNTube's three JSON files")
    inputs.add_argument("--records", help="JSON/YAML config containing explicit object bounds")
    parser.add_argument("--config", help="Scene graph builder configuration")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config) if args.config else {}
    scene = None
    if args.vlntube:
        config.update(backend="vlntube", input_dir=args.vlntube)
    elif args.records:
        config = {**load_config(args.records), **config, "backend": "records"}
    else:
        scene = load_scene(args.scene)
        config["backend"] = "usd"
    graph = SceneGraphBuilder().build(scene, config)
    graph.save(args.output)
    print(
        f"Saved {len(graph.objects)} objects, {len(graph.rooms)} rooms, "
        f"{len(graph.relations)} directed relations to {args.output}"
    )


if __name__ == "__main__":
    main()
