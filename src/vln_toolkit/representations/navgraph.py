from dataclasses import dataclass, field

from vln_toolkit.utils.config import load_config, save_json

from .registry import register


@dataclass
class NavGraph:
    nodes: dict = field(default_factory=dict)
    edges: list = field(default_factory=list)

    def neighbors(self, node_id):
        if node_id not in self.nodes:
            raise KeyError(node_id)
        return [
            edge["target"] if edge["source"] == node_id else edge["source"]
            for edge in self.edges
            if node_id in (edge["source"], edge["target"])
        ]

    def save(self, path):
        save_json(path, {"nodes": self.nodes, "edges": self.edges})

    @classmethod
    def load(cls, path):
        return cls(**load_config(path))


@register("navgraph")
class NavGraphBuilder:
    def build(self, occupancy, config=None):
        raise NotImplementedError("NavGraph sampling/connectivity import is reserved for phase two")
