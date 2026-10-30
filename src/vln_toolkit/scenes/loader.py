from pathlib import Path

from vln_toolkit.utils.config import load_config

from .scene import Scene


def load_scene(
    scene_path: str | Path, meta: dict | str | Path | None = None, *, inspect_stage: bool = True
) -> Scene:
    """Load a local USD or scene.json. USD inspection requires pxr, not SimulationApp.

    With inspect_stage=False, source coordinates must be explicitly provided in metadata.
    """
    path = Path(scene_path).expanduser().resolve()
    if path.is_dir():
        path = path / "scene.json"
    metadata = {}
    if path.suffix.lower() == ".json":
        metadata = load_config(path)
        path = (path.parent / metadata["usd_path"]).resolve()
    else:
        sidecar = path.with_suffix(".json")
        if sidecar.is_file():
            metadata = load_config(sidecar)
    if meta is not None:
        metadata.update(meta if isinstance(meta, dict) else load_config(meta))
    if path.suffix.lower() not in {".usd", ".usda", ".usdc", ".usdz"}:
        raise ValueError(f"Unsupported scene format: {path.suffix}; convert it to USD first")
    if not path.is_file():
        raise FileNotFoundError(path)
    if inspect_stage:
        from vln_toolkit.sim.isaac_runtime import inspect_usd

        # Authored stage coordinates are authoritative over sidecar guesses.
        metadata.update(inspect_usd(path))
    elif "up_axis" not in metadata or "meters_per_unit" not in metadata:
        raise ValueError("Without USD inspection, specify up_axis and meters_per_unit")
    from vln_toolkit.utils.coordinates import stage_to_world

    metadata["stage_to_world"] = stage_to_world(
        metadata["up_axis"], metadata["meters_per_unit"]
    ).tolist()
    metadata.setdefault("source", "usd")
    return Scene(metadata.get("scene_id", path.stem), str(path), metadata)
