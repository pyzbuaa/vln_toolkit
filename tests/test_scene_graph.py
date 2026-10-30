import json

import numpy as np
import pytest

from vln_toolkit.representations import BUILDERS, SceneGraph, SceneGraphBuilder, register
from vln_toolkit.representations.scene_graph import infer_spatial_relationship, load_vlntube_rooms


def obj(key, low, high, category=None):
    return {"instance_id": key, "category": category, "min_points": low, "max_points": high}


def test_support_containment_near_and_inverse_relations(tmp_path):
    table = obj("table", [0, 0, 0], [2, 2, 1], "table")
    cup = obj("cup", [0.8, 0.8, 1], [1.2, 1.2, 1.3], "cup")
    inside = obj("inside", [0.1, 0.1, 0.1], [0.3, 0.3, 0.3])
    adjacent = obj("adjacent", [2.2, 0, 0], [3, 1, 1], "chair")
    assert infer_spatial_relationship(cup, table)[:2] == ("on", "below")
    assert infer_spatial_relationship(inside, table)[:2] == ("in", "contain")
    assert infer_spatial_relationship(table, adjacent)[:2] == ("near", "near")
    graph = SceneGraphBuilder().build(
        None,
        {
            "backend": "records",
            "objects": [table, cup, inside],
            "rooms": {"living": {"polygon": [[-1, -1], [4, -1], [4, 4], [-1, 4]]}},
        },
    )
    assert graph.find_objects("cup")[0]["room"] == "living"
    assert graph.relations_for("cup", "on")[0]["target"] == "table"
    assert graph.objects["inside"]["category"] is None
    assert graph.metadata["unlabeled_object_count"] == 1
    graph.save(tmp_path)
    restored = SceneGraph.load(tmp_path)
    assert restored.objects == graph.objects and restored.relations == graph.relations
    imported = SceneGraph.from_vlntube(tmp_path)
    assert imported.relations_for("cup", "on")[0]["target"] == "table"
    assert imported.relations_for("table", "below")[0]["target"] == "cup"
    assert json.loads((tmp_path / "category_dict.json").read_text())["cup"] == [["cup"]]


def test_object_outside_rooms_and_missing_semantics():
    graph = SceneGraphBuilder().build(
        None,
        {
            "backend": "records",
            "objects": [obj("SofaNameWithoutAnnotation", [10, 10, 0], [11, 11, 1])],
            "rooms": {"room": {"polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}},
        },
    )
    record = graph.objects["SofaNameWithoutAnnotation"]
    assert record["room"] is None and record["category"] is None
    assert graph.metadata["semantics_missing"]
    assert not graph.find_objects("sofa")


def test_room_assignment_maximal_overlap_and_connectivity():
    graph = SceneGraphBuilder().build(
        None,
        {
            "backend": "records",
            "objects": [obj("boundary", [0.8, 0.2, 0], [1.8, 0.8, 1])],
            "rooms": {
                "a": {"polygon": [[0, 0], [1, 0], [1, 1], [0, 1]], "connected_to": ["b"]},
                "b": {"polygon": [[1, 0], [2, 0], [2, 1], [1, 1]]},
            },
        },
    )
    assert graph.objects["boundary"]["room"] == "b"
    assert graph.relations_for("a", "connected_to")[0]["target"] == "b"


def test_objects_in_different_rooms_have_no_inferred_spatial_edges():
    first, second = obj("a", [0, 0, 0], [1, 1, 1]), obj("b", [1.1, 0, 0], [2, 1, 1])
    first["room"], second["room"] = "room_a", "room_b"
    graph = SceneGraphBuilder().build(None, {"backend": "records", "objects": [first, second]})
    assert not graph.relations_for("a", "near")


def test_invalid_and_duplicate_objects_rejected():
    with pytest.raises(ValueError, match="bounding box"):
        SceneGraphBuilder().build(
            None, {"backend": "records", "objects": [obj("a", [1, 0, 0], [0, 1, 1])]}
        )
    record = obj("a", [0, 0, 0], [1, 1, 1])
    with pytest.raises(ValueError, match="Duplicate"):
        SceneGraphBuilder().build(None, {"backend": "records", "objects": [record, record]})


def test_vlntube_pixel_header_conversion_is_explicit(tmp_path):
    grid = np.array([[0, 20, 21], [10, 0, 0], [11, 0, 0]])
    np.save(tmp_path / "freemap.npy", grid)
    (tmp_path / "room_region.json").write_text(json.dumps({"r": [[1, 1], [1, 2], [2, 2], [2, 1]]}))
    rooms = load_vlntube_rooms(tmp_path / "room_region.json", tmp_path / "freemap.npy")
    assert rooms["r"]["polygon"] == [[20, 10], [21, 10], [21, 11], [20, 11]]


def test_registry_builtins_and_duplicate_registration():
    assert set(BUILDERS) >= {"occupancy", "scene_graph", "navmesh", "navgraph"}
    with pytest.raises(ValueError, match="already exists"):
        register("scene_graph")(type("Duplicate", (), {}))
