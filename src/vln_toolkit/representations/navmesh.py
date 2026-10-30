from dataclasses import dataclass, field

from .registry import register


@dataclass
class NavMesh:
    artifact_path: str | None = None
    metadata: dict = field(default_factory=dict)

    def is_navigable(self, position):
        raise NotImplementedError("Attach an Isaac Sim native NavMesh query backend")

    def find_path(self, start, goal):
        raise NotImplementedError("Attach an Isaac Sim native NavMesh path backend")


@register("navmesh")
class NavMeshBuilder:
    def build(self, scene, config=None):
        raise NotImplementedError(
            "Native NavMesh baking requires a version-specific Isaac Sim adapter"
        )
