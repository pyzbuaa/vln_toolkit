"""USD operations and Isaac Sim lifecycle are centralized here.

The USD-only helpers also work with the optional usd-core package. The native
occupancy adapter targets Isaac Sim 5.1's isaacsim.asset.gen.omap Generator API.
"""

import itertools
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from vln_toolkit.utils.coordinates import stage_to_world, transform_points


def open_usd(path):
    try:
        from pxr import Usd
    except ImportError as exc:
        raise RuntimeError(
            "USD operations require Isaac Sim's Python or uv sync --extra usd "
            "(then run with uv run --extra usd)"
        ) from exc
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"Cannot open USD stage: {path}")
    return stage


def _semantic_category(prim, attributes=()) -> str | None:
    # Explicit application-provided attributes and OpenUSD SemanticsLabelsAPI.
    for name in (
        *attributes,
        "semantics:labels:class",
        "semantic:class",
        "primvars:semantic:class",
    ):
        attr = prim.GetAttribute(name)
        value = attr.Get() if attr else None
        if value is not None:
            if isinstance(value, str):
                return value or None
            labels = list(value)
            if labels:
                return str(labels[-1])
    # Isaac Sim SemanticsAPI, including multiple applied semantic instances.
    for attr in prim.GetAttributes():
        name = attr.GetName()
        if name.lower().endswith(":semantictype") and str(attr.Get()).lower() == "class":
            data = prim.GetAttribute(name.rsplit(":", 1)[0] + ":semanticData")
            if data and data.Get():
                return str(data.Get())
    return None


def inspect_usd(path) -> dict:
    stage = open_usd(path)
    from pxr import UsdGeom, UsdPhysics

    return {
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "has_collision": any(
            prim.HasAPI(UsdPhysics.CollisionAPI)
            and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            for prim in stage.Traverse()
        ),
        "has_semantics": any(_semantic_category(prim) is not None for prim in stage.Traverse()),
        "layers": [layer.identifier for layer in stage.GetUsedLayers()],
    }


def extract_semantic_objects(scene, config: dict) -> list[dict]:
    """Extract one world AABB per annotated instance or explicitly selected prim.

    category_roots maps a dataset category path to an explicit label, selecting
    each immediate child as an object. This supports VLNTube's hierarchy without
    inferring categories from arbitrary prim names.
    """
    stage = open_usd(scene.usd_path)
    from pxr import Usd, UsdGeom

    transform = stage_to_world(
        str(UsdGeom.GetStageUpAxis(stage)), float(UsdGeom.GetStageMetersPerUnit(stage))
    )
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
    overrides = dict(config.get("semantic_categories", {}))
    for root_path, label in config.get("category_roots", {}).items():
        root = stage.GetPrimAtPath(root_path)
        if not root:
            raise ValueError(f"Missing category root: {root_path}")
        for child in root.GetChildren():
            overrides[str(child.GetPath())] = label
    explicit = set(config.get("object_paths", [])) | set(overrides)
    for path in explicit:
        if not stage.GetPrimAtPath(path):
            raise ValueError(f"Missing object prim: {path}")
    excluded = tuple(config.get("excluded_paths", []))
    # Select semantic roots first, so their descendant meshes aren't emitted again.
    candidates = []
    selected = set()
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if any(path == p or path.startswith(p + "/") for p in excluded):
            continue
        category = overrides.get(path) or _semantic_category(
            prim, config.get("semantic_attributes", [])
        )
        if (category or path in explicit) and not any(
            path.startswith(parent + "/") for parent in selected
        ):
            candidates.append((prim, category, "mapping" if path in overrides else "annotation"))
            selected.add(path)
    if config.get("include_unlabeled", True):
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if prim.IsA(UsdGeom.Gprim) and not any(
                path == p or path.startswith(p + "/") for p in (*selected, *excluded)
            ):
                candidates.append((prim, None, "missing"))
    records = []
    for prim, category, source in candidates:
        bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if bounds.IsEmpty():
            continue
        low, high = bounds.GetMin(), bounds.GetMax()
        corners = list(itertools.product(*[(low[i], high[i]) for i in range(3)]))
        points = transform_points(corners, transform)
        if not np.isfinite(points).all():
            continue
        records.append(
            {
                "instance_id": str(prim.GetPath()),
                "prim_path": str(prim.GetPath()),
                "category": category,
                "min_points": points.min(axis=0).tolist(),
                "max_points": points.max(axis=0).tolist(),
                "metadata": {"semantic_source": source},
            }
        )
    return records


class IsaacRuntime:
    """Explicit owner of a standalone SimulationApp. Use as a context manager."""

    def __init__(self, *, headless=True, disable_viewport_updates=False):
        self.headless = headless
        self.disable_viewport_updates = disable_viewport_updates
        self.app = None

    def __enter__(self):
        try:
            from isaacsim import SimulationApp
        except ImportError as exc:
            raise RuntimeError("Run this operation using Isaac Sim's Python environment") from exc
        self.app = SimulationApp(
            {"headless": self.headless, "disable_viewport_updates": self.disable_viewport_updates}
        )
        return self

    def load(self, scene):
        import omni.usd

        if not omni.usd.get_context().open_stage(scene.usd_path):
            raise RuntimeError(f"Failed to load scene: {scene.usd_path}")
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            self.app.update()

    def step(self, count=1):
        for _ in range(count):
            self.app.update()

    def __exit__(self, exc_type, exc, traceback):
        if self.app is not None:
            self.app.close()
            self.app = None


def occupancy_from_positions(
    free_positions, occupied_positions, resolution, floor_z, metadata=None
):
    """Rasterize native cell centers by position, avoiding undocumented buffer flips.

    Trim unobserved outer rows/columns. Interior unobserved cells remain unknown.
    """
    from vln_toolkit.representations.occupancy import OccupancyMap

    free = np.asarray(free_positions, dtype=float).reshape(-1, 3)
    occupied = np.asarray(occupied_positions, dtype=float).reshape(-1, 3)
    points = np.concatenate([free, occupied])
    if not len(points):
        raise RuntimeError(
            "Occupancy generator returned no known cells; verify bounds and colliders"
        )
    anchor = points[:, :2].min(axis=0)
    indices = np.rint((points[:, :2] - anchor) / resolution).astype(int)
    width, height = indices.max(axis=0) + 1
    grid = np.full((height, width), 255, dtype=np.uint8)
    for start, stop, code in [(0, len(free), 0), (len(free), len(points), 1)]:
        cells = indices[start:stop]
        grid[cells[:, 1], cells[:, 0]] = code
    return OccupancyMap(
        grid, resolution, tuple(anchor - resolution / 2), floor_z, metadata=metadata or {}
    )


@contextmanager
def mapping_stage(stage, *, freeze_rigid_bodies=True):
    """Temporary physics setup for a map of the authored, stationary scene."""
    from pxr import Sdf, Usd, UsdPhysics

    layer = Sdf.Layer.CreateAnonymous("vln_occupancy")
    session = stage.GetSessionLayer()
    session.subLayerPaths.insert(0, layer.identifier)
    setup = {"frozen_rigid_bodies": 0, "temporary_physics_scene": False}
    try:
        with Usd.EditContext(stage, layer):
            if not any(prim.IsA(UsdPhysics.Scene) for prim in stage.Traverse()):
                UsdPhysics.Scene.Define(stage, "/__VLNToolkitPhysicsScene")
                setup["temporary_physics_scene"] = True
            if freeze_rigid_bodies:
                for prim in stage.Traverse():
                    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        body = UsdPhysics.RigidBodyAPI(prim)
                        if body.GetRigidBodyEnabledAttr().Get():
                            body.CreateRigidBodyEnabledAttr(False)
                            setup["frozen_rigid_bodies"] += 1
        yield setup
    finally:
        session.subLayerPaths.remove(layer.identifier)


def generate_occupancy(scene, config):
    """Native collision-map adapter. Requires an already running IsaacRuntime.

    A single configured height slice is generated; this does not establish full
    robot-height clearance. Use appropriate collision geometry and validate it.
    """
    try:
        import omni.kit.app
        import omni.physx
        import omni.timeline
        import omni.usd
        from pxr import UsdGeom
    except ImportError as exc:
        raise RuntimeError("Occupancy generation requires an active IsaacRuntime") from exc
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.asset.gen.omap")
    from isaacsim.asset.gen.omap.bindings import _omap

    stage = omni.usd.get_context().get_stage()
    if (
        stage is None
        or Path(stage.GetRootLayer().realPath).resolve() != Path(scene.usd_path).resolve()
    ):
        raise ValueError("Load this Scene into IsaacRuntime before building occupancy")
    if str(UsdGeom.GetStageUpAxis(stage)) != "Z" or UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
        raise ValueError("Native occupancy currently requires a Z-up stage authored in metres")
    if not inspect_usd(scene.usd_path)["has_collision"]:
        raise ValueError("Scene has no enabled collision geometry")
    resolution = float(config.get("resolution", 0.05))
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be positive and finite")
    origin, lower, upper = config["origin"], config["lower_bound"], config["upper_bound"]
    timeline = omni.timeline.get_timeline_interface()
    was_playing = timeline.is_playing()
    with mapping_stage(stage, freeze_rigid_bodies=config.get("freeze_rigid_bodies", True)) as setup:
        try:
            timeline.play()
            for _ in range(10):
                omni.kit.app.get_app().update()
            generator = _omap.Generator(
                omni.physx.get_physx_interface(), omni.usd.get_context().get_stage_id()
            )
            generator.update_settings(resolution, 1, 0, 255)
            generator.set_transform(tuple(origin), tuple(lower), tuple(upper))
            generator.generate2d()
            result = occupancy_from_positions(
                generator.get_free_positions(),
                generator.get_occupied_positions(),
                resolution,
                config.get("floor_z", 0.0),
                {
                    "scene_id": scene.scene_id,
                    "scene_source": scene.source_record(),
                    "algorithm": "isaacsim.asset.gen.omap",
                    "config": config,
                    "physics_setup": setup,
                    "clearance_model": "single_height_slice",
                    "outer_unknown_cells": "trimmed",
                },
            )
            return result.inflated(float(config.get("agent_radius", 0.0)))
        finally:
            if not was_playing:
                timeline.stop()
