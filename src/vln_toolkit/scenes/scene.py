from dataclasses import dataclass, field
from pathlib import Path

from vln_toolkit.utils.config import save_json


@dataclass
class Scene:
    scene_id: str
    usd_path: str
    metadata: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        save_json(path, {**self.metadata, "scene_id": self.scene_id, "usd_path": self.usd_path})

    def source_record(self) -> dict:
        """Lightweight revision record, not a cache or asset database."""
        path = Path(self.usd_path)
        stat = path.stat()
        layers = []
        for identifier in sorted(self.metadata.get("layers", [])):
            layer = Path(identifier)
            if layer.is_file():
                layer_stat = layer.stat()
                layers.append(
                    {
                        "path": str(layer.resolve()),
                        "size": layer_stat.st_size,
                        "mtime_ns": layer_stat.st_mtime_ns,
                    }
                )
        return {
            "usd_path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "version": self.metadata.get("version"),
            "layers": layers,
        }
