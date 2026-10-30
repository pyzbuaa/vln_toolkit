"""Semantic scene graphs inspired by VLNTube (see THIRD_PARTY_NOTICES.md).

Relations are directed: source --predicate--> target. Geometry uses deterministic
world-space AABBs, not VLNTube's stochastic surface point clouds. Bounds-based
containment/support are geometric approximations, not semantic facts.
"""

import itertools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon, box

from vln_toolkit.utils.config import load_config, save_json

from .registry import register

INVERSE = {
    "above": "below",
    "below": "above",
    "on": "below",
    "in": "contain",
    "contain": "in",
    "near": "near",
}


def normalize_object(record: dict, object_id: str | None = None) -> dict:
    result = dict(record)
    result.pop("point_cloud", None)
    object_id = str(object_id or result["instance_id"])
    low, high = np.asarray(result["min_points"], float), np.asarray(result["max_points"], float)
    if (
        low.shape != (3,)
        or high.shape != (3,)
        or not np.isfinite([low, high]).all()
        or np.any(high < low)
    ):
        raise ValueError(f"Invalid world-space bounding box for {object_id}")
    result.update(
        instance_id=object_id,
        min_points=low.tolist(),
        max_points=high.tolist(),
        position=((low + high) / 2).tolist(),
    )
    result.setdefault("category", None)
    result.setdefault("room", None)
    result.setdefault("metadata", {})
    result["metadata"] = {**result["metadata"], "semantics_missing": not bool(result["category"])}
    return result


def infer_spatial_relationship(
    a: dict, b: dict, *, near_distance=0.5, touching_distance=0.06
) -> tuple[str | None, str | None, float]:
    """Return (A relative to B, B relative to A, AABB separation in metres)."""
    amin, amax = np.array(a["min_points"]), np.array(a["max_points"])
    bmin, bmax = np.array(b["min_points"]), np.array(b["max_points"])
    distance = float(np.linalg.norm(np.maximum(0, np.maximum(amin - bmax, bmin - amax))))
    if distance > near_distance:
        return None, None, distance
    # Equal boxes are not evidence of containment.
    if np.all(amin >= bmin) and np.all(amax <= bmax) and np.any((amin > bmin) | (amax < bmax)):
        return "in", "contain", distance
    if np.all(bmin >= amin) and np.all(bmax <= amax) and np.any((bmin > amin) | (bmax < amax)):
        return "contain", "in", distance
    overlap = np.maximum(0, np.minimum(amax[:2], bmax[:2]) - np.maximum(amin[:2], bmin[:2]))
    area = float(np.prod(overlap))
    area_a, area_b = float(np.prod(amax[:2] - amin[:2])), float(np.prod(bmax[:2] - bmin[:2]))
    ratio_a, ratio_b = area / max(area_a, 1e-12), area / max(area_b, 1e-12)
    center_delta = (amin[2] + amax[2] - bmin[2] - bmax[2]) / 2
    if area > 0:
        if abs(amin[2] - bmax[2]) <= touching_distance and ratio_a > 0.3 and center_delta > 0.01:
            return "on", "below", distance
        if abs(bmin[2] - amax[2]) <= touching_distance and ratio_b > 0.3 and center_delta < -0.01:
            return "below", "on", distance
        if amin[2] > bmax[2] and max(ratio_a, ratio_b) > 0.2:
            return "above", "below", distance
        if bmin[2] > amax[2] and max(ratio_a, ratio_b) > 0.2:
            return "below", "above", distance
    if area == 0:
        return "near", "near", distance
    return None, None, distance


def assign_room(obj: dict, rooms: dict) -> str | None:
    """Choose maximal footprint overlap; leave outside objects unassigned."""
    low, high = obj["min_points"], obj["max_points"]
    footprint = box(low[0], low[1], high[0], high[1])
    center = Point(obj["position"][:2])
    best, best_score = None, 0.0
    for room_id, room in rooms.items():
        if not room.get("polygon"):
            continue
        polygon = Polygon(room["polygon"])
        if not polygon.is_valid or polygon.area == 0:
            raise ValueError(f"Invalid room polygon: {room_id}")
        score = polygon.intersection(footprint).area
        if score > best_score or (best is None and polygon.covers(center)):
            best, best_score = room_id, score
    return best


def load_vlntube_rooms(room_region_path, freemap_path) -> dict:
    """VLNTube room pixels are (row, col); freemap headers have dataset-specific XY swap.

    This conversion is explicit and only used for VLNTube metadata. Input room
    coordinates must already use the same metre/Z-up world frame as the objects.
    """
    regions = load_config(room_region_path)
    freemap = np.load(freemap_path, allow_pickle=False)
    return {
        name: {"polygon": [[float(freemap[0, col]), float(freemap[row, 0])] for row, col in pixels]}
        for name, pixels in regions.items()
    }


@dataclass
class SceneGraph:
    objects: dict[str, dict] = field(default_factory=dict)
    rooms: dict[str, dict] = field(default_factory=dict)
    relations: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def find_objects(self, category: str) -> list[dict]:
        return [obj for obj in self.objects.values() if obj.get("category") == category]

    def relations_for(self, entity_id: str, predicate: str | None = None) -> list[dict]:
        return [
            rel
            for rel in self.relations
            if rel["source"] == entity_id and (predicate is None or rel["predicate"] == predicate)
        ]

    def save(self, directory: str | Path, *, export_vlntube: bool = True) -> None:
        directory = Path(directory)
        save_json(
            directory / "scene_graph.json",
            {
                "schema_version": 1,
                "objects": self.objects,
                "rooms": self.rooms,
                "relations": self.relations,
                "metadata": self.metadata,
            },
        )
        if export_vlntube:
            self.export_vlntube(directory)

    def export_vlntube(self, directory: str | Path) -> None:
        directory = Path(directory)
        objects = {key: {**value, "nearby_objects": {}} for key, value in self.objects.items()}
        for rel in self.relations:
            source, target = rel["source"], rel["target"]
            if source in objects and target in objects and rel["predicate"] in INVERSE:
                # Upstream summarizer stores the *target's* relation to the source.
                reverse = next(
                    (
                        r["predicate"]
                        for r in self.relations
                        if r["source"] == target and r["target"] == source
                    ),
                    INVERSE[rel["predicate"]],
                )
                objects[source]["nearby_objects"][target] = [reverse, rel.get("distance", 0.0)]
        categories = {}
        for key, obj in objects.items():
            if obj.get("category"):
                categories.setdefault(obj["category"], []).append([key])
        save_json(directory / "object_dict.json", objects)
        save_json(
            directory / "room_dict.json",
            {
                key: [oid for oid, obj in objects.items() if obj.get("room") == key]
                for key in self.rooms
            },
        )
        save_json(directory / "category_dict.json", categories)

    @classmethod
    def load(cls, directory: str | Path) -> "SceneGraph":
        directory = Path(directory)
        path = directory / "scene_graph.json" if directory.is_dir() else directory
        data = load_config(path)
        return cls(
            {key: normalize_object(value, key) for key, value in data["objects"].items()},
            data["rooms"],
            data["relations"],
            data.get("metadata", {}),
        )

    @classmethod
    def from_vlntube(cls, directory: str | Path) -> "SceneGraph":
        directory = Path(directory)
        objects = {
            key: normalize_object(obj, key)
            for key, obj in load_config(directory / "object_dict.json").items()
        }
        room_path = directory / "room_dict.json"
        rooms = {key: {"object_ids": ids} for key, ids in load_config(room_path).items()}
        relations = []
        for key, obj in objects.items():
            for target, (predicate, distance) in obj.pop("nearby_objects", {}).items():
                if target not in objects:
                    raise ValueError(f"Unknown related object: {target}")
                # Invert the edge, preserving non-bijective on/below without guessing.
                relations.append(
                    {
                        "source": target,
                        "target": key,
                        "predicate": predicate,
                        "distance": float(distance),
                    }
                )
            room_id = obj.get("room")
            if room_id is not None:
                rooms.setdefault(room_id, {"object_ids": []})
                relations.append({"source": room_id, "target": key, "predicate": "contains"})
        return cls(
            objects,
            rooms,
            relations,
            {
                "source": "vlntube",
                "frame_id": "world",
                "coordinates": "assumed Z-up metres; convert upstream data before import",
            },
        )


@register("scene_graph")
class SceneGraphBuilder:
    def build(self, scene, config=None) -> SceneGraph:
        config = config or {}
        backend = config.get("backend", "usd")
        if backend == "vlntube":
            graph = SceneGraph.from_vlntube(config["input_dir"])
            if scene is not None:
                graph.metadata["scene_source"] = scene.source_record()
            return graph
        if backend == "usd":
            from vln_toolkit.sim.isaac_runtime import extract_semantic_objects

            records = extract_semantic_objects(scene, config)
        elif backend == "records":
            records = config["objects"]
        else:
            raise ValueError(f"Unknown scene graph backend: {backend}")
        objects = {}
        for record in records:
            obj = normalize_object(record)
            if obj["instance_id"] in objects:
                raise ValueError(f"Duplicate object: {obj['instance_id']}")
            objects[obj["instance_id"]] = obj
        rooms = {key: dict(value) for key, value in config.get("rooms", {}).items()}
        if "room_region_path" in config:
            rooms.update(load_vlntube_rooms(config["room_region_path"], config["freemap_path"]))
        relations = []
        for key, obj in objects.items():
            obj["room"] = obj.get("room") or assign_room(obj, rooms)
            if obj["room"] is not None:
                room = rooms.setdefault(obj["room"], {})
                relations.append({"source": obj["room"], "target": key, "predicate": "contains"})
        for room_id, room in rooms.items():
            room["object_ids"] = [key for key, obj in objects.items() if obj["room"] == room_id]
            for target in room.get("connected_to", []):
                if target not in rooms:
                    raise ValueError(f"Unknown connected room: {target}")
                relations.append({"source": room_id, "target": target, "predicate": "connected_to"})
        for (id_a, a), (id_b, b) in itertools.combinations(objects.items(), 2):
            if a["room"] != b["room"]:
                continue
            rel_a, rel_b, distance = infer_spatial_relationship(
                a,
                b,
                near_distance=config.get("near_distance", 0.5),
                touching_distance=config.get("touching_distance", 0.06),
            )
            if rel_a:
                relations.extend(
                    [
                        {"source": id_a, "target": id_b, "predicate": rel_a, "distance": distance},
                        {"source": id_b, "target": id_a, "predicate": rel_b, "distance": distance},
                    ]
                )
        metadata = {
            "algorithm": "aabb_spatial_relations",
            "frame_id": "world",
            "units": "metres",
            "up_axis": "Z",
            "geometry_approximation": "world_aabb",
            "semantics_missing": not any(o["category"] for o in objects.values()),
            "unlabeled_object_count": sum(not o["category"] for o in objects.values()),
            "config": config,
        }
        if scene is not None:
            metadata.update(scene_id=scene.scene_id, scene_source=scene.source_record())
        return SceneGraph(objects, rooms, relations, metadata)
