from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pxr", reason="Install optional .[usd] for OpenUSD extraction tests")
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from vln_toolkit.representations import OccupancyMap, SceneGraphBuilder
from vln_toolkit.scenes import load_scene
from vln_toolkit.sim.isaac_runtime import mapping_stage


def test_real_usda_fixture_has_semantics_collision_and_support():
    scene = load_scene(Path(__file__).parents[1] / "examples/room.usda")
    assert scene.metadata["has_collision"] and scene.metadata["has_semantics"]
    assert scene.metadata["meters_per_unit"] == 1
    graph = SceneGraphBuilder().build(scene, {"include_unlabeled": False})
    assert len(graph.objects) == 2
    assert graph.find_objects("cup")[0]["instance_id"] == "/World/Cup"
    assert graph.relations_for("/World/Cup", "on")[0]["target"] == "/World/Table"


def test_transforms_units_grouping_and_no_name_inference(tmp_path):
    path = tmp_path / "centimetres.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, "Y")
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    group = UsdGeom.Xform.Define(stage, "/World/Furniture")
    group.AddTranslateOp().Set(Gf.Vec3d(100, 200, 300))
    prim = group.GetPrim()
    prim.CreateAttribute("semantics:Semantics:semanticType", Sdf.ValueTypeNames.String).Set("class")
    prim.CreateAttribute("semantics:Semantics:semanticData", Sdf.ValueTypeNames.String).Set("chair")
    UsdGeom.Cube.Define(stage, "/World/Furniture/Mesh").GetSizeAttr().Set(20)
    UsdGeom.Cube.Define(stage, "/World/SofaWithoutSemantics").GetSizeAttr().Set(10)
    stage.GetRootLayer().Save()
    graph = SceneGraphBuilder().build(load_scene(path))
    assert len(graph.objects) == 2  # Furniture's child mesh is not a duplicate object.
    chair = graph.find_objects("chair")[0]
    np.testing.assert_allclose(chair["position"], [1, -3, 2])
    np.testing.assert_allclose(chair["min_points"], [0.9, -3.1, 1.9])
    assert graph.objects["/World/SofaWithoutSemantics"]["category"] is None
    graph = SceneGraphBuilder().build(
        load_scene(path),
        {
            "semantic_categories": {"/World/SofaWithoutSemantics": "sofa"},
            "include_unlabeled": False,
        },
    )
    assert len(graph.find_objects("sofa")) == 1


def test_scene_revision_change_warns(tmp_path):
    path = tmp_path / "room.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    stage.GetRootLayer().Save()
    scene = load_scene(path)
    occupancy = OccupancyMap([[0]], 1, (0, 0), metadata={"scene_source": scene.source_record()})
    occupancy.check_scene(scene)
    path.write_text(path.read_text() + "\n# changed\n")
    with pytest.warns(UserWarning, match="revision"):
        occupancy.check_scene(scene)


def test_loader_rejects_missing_files_and_incomplete_coordinates(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_scene(tmp_path / "missing.usd")
    path = tmp_path / "file.usda"
    path.write_text("#usda 1.0\n")
    with pytest.raises(ValueError, match="up_axis"):
        load_scene(path, inspect_stage=False)


def test_mapping_physics_setup_is_temporary_even_on_failure():
    stage = Usd.Stage.CreateInMemory()
    cube = UsdGeom.Cube.Define(stage, "/Cube").GetPrim()
    body = UsdPhysics.RigidBodyAPI.Apply(cube)
    body.CreateRigidBodyEnabledAttr(True)
    original = stage.GetRootLayer().ExportToString()
    with pytest.raises(RuntimeError, match="mapping failed"), mapping_stage(stage) as setup:
        assert setup["frozen_rigid_bodies"] == 1
        assert setup["temporary_physics_scene"]
        assert not body.GetRigidBodyEnabledAttr().Get()
        assert stage.GetPrimAtPath("/__VLNToolkitPhysicsScene")
        raise RuntimeError("mapping failed")
    assert body.GetRigidBodyEnabledAttr().Get()
    assert not stage.GetPrimAtPath("/__VLNToolkitPhysicsScene")
    assert stage.GetRootLayer().ExportToString() == original
    assert not stage.GetSessionLayer().subLayerPaths
