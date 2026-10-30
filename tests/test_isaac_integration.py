"""Opt-in native integration test; never starts Isaac Sim during ordinary pytest."""

import os
from pathlib import Path

import pytest


@pytest.mark.isaac
@pytest.mark.skipif(
    os.environ.get("VLN_TOOLKIT_ISAAC_TEST") != "1", reason="Requires Isaac Sim runtime"
)
def test_native_occupancy_on_authored_room():
    from vln_toolkit.representations import OccupancyBuilder
    from vln_toolkit.scenes import load_scene
    from vln_toolkit.sim.isaac_runtime import IsaacRuntime
    from vln_toolkit.utils.config import load_config

    root = Path(__file__).parents[1]
    with IsaacRuntime() as runtime:
        scene = load_scene(root / "examples/room.usda")
        runtime.load(scene)
        occupancy = OccupancyBuilder().build(
            scene, load_config(root / "configs/scene.yaml")["occupancy"]
        )
        assert occupancy.is_free((0, 0))
        assert not occupancy.is_free((1, 1))  # Table collider intersects the scan plane.
