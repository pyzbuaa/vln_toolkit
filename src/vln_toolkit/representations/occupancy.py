"""Row-major maps: grid[row, col], +row = +Y, +col = +X, origin at cell corner."""

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vln_toolkit.utils.config import load_config, save_json

from .registry import register

FREE, OCCUPIED, UNKNOWN = 0, 1, 255


@dataclass
class OccupancyMap:
    grid: np.ndarray
    resolution: float
    origin: tuple[float, float]
    floor_z: float = 0.0
    frame_id: str = "world"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        values = np.asarray(self.grid)
        if values.ndim != 2 or not values.size or not np.isin(values, [0, 1, 255]).all():
            raise ValueError("grid must be a nonempty 2D array encoded as 0/1/255")
        if not math.isfinite(self.resolution) or self.resolution <= 0:
            raise ValueError("resolution must be positive and finite")
        if len(self.origin) != 2 or not np.isfinite([*self.origin, self.floor_z]).all():
            raise ValueError("origin must be finite XY and floor_z must be finite")
        self.grid = values.astype(np.uint8, copy=True)
        self.origin = tuple(float(v) for v in self.origin)

    def world_to_grid(self, position) -> tuple[int, int]:
        x, y = position[:2]
        return (
            math.floor((y - self.origin[1]) / self.resolution),
            math.floor((x - self.origin[0]) / self.resolution),
        )

    def grid_to_world(self, cell) -> tuple[float, float, float]:
        row, col = cell
        return (
            self.origin[0] + (col + 0.5) * self.resolution,
            self.origin[1] + (row + 0.5) * self.resolution,
            self.floor_z,
        )

    def contains_cell(self, cell) -> bool:
        row, col = cell
        return 0 <= row < self.grid.shape[0] and 0 <= col < self.grid.shape[1]

    def is_free_cell(self, cell) -> bool:
        return self.contains_cell(cell) and self.grid[cell] == FREE

    def is_free(self, position) -> bool:
        return self.is_free_cell(self.world_to_grid(position))

    def inflated(self, radius: float) -> "OccupancyMap":
        """Conservative disk dilation. Unknown space and outside bounds block the robot."""
        if not math.isfinite(radius) or radius < 0:
            raise ValueError("radius must be finite and nonnegative")
        grid = self.grid.copy()
        cells = math.ceil(radius / self.resolution)
        blocked = np.pad(self.grid != FREE, cells, constant_values=True)
        height, width = grid.shape
        # Include intersecting cells by conservatively rounding radius up to whole cells.
        for dr in range(-cells, cells + 1):
            for dc in range(-cells, cells + 1):
                if dr * dr + dc * dc <= cells * cells:
                    mask = blocked[
                        cells + dr : cells + dr + height, cells + dc : cells + dc + width
                    ]
                    grid[mask & (self.grid == FREE)] = OCCUPIED
        return OccupancyMap(
            grid,
            self.resolution,
            self.origin,
            self.floor_z,
            self.frame_id,
            {**self.metadata, "inflation_radius": radius},
        )

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "map.npy", self.grid, allow_pickle=False)
        save_json(
            directory / "metadata.json",
            {
                "schema_version": 1,
                "resolution": self.resolution,
                "origin": self.origin,
                "floor_z": self.floor_z,
                "frame_id": self.frame_id,
                "shape": list(self.grid.shape),
                "axes": {"row": "+Y", "col": "+X", "origin": "lower_left_cell_corner"},
                "encoding": {"free": FREE, "occupied": OCCUPIED, "unknown": UNKNOWN},
                "metadata": self.metadata,
            },
        )

    @classmethod
    def load(cls, directory: str | Path, *, scene=None) -> "OccupancyMap":
        directory = Path(directory)
        meta = load_config(directory / "metadata.json")
        if meta.get("axes") != {"row": "+Y", "col": "+X", "origin": "lower_left_cell_corner"}:
            raise ValueError("Unsupported map axes; convert the map explicitly")
        if meta.get("encoding") != {"free": 0, "occupied": 1, "unknown": 255}:
            raise ValueError("Unsupported map encoding")
        grid = np.load(directory / "map.npy", allow_pickle=False)
        if list(grid.shape) != meta["shape"]:
            raise ValueError("Map shape does not match metadata")
        result = cls(
            grid,
            meta["resolution"],
            tuple(meta["origin"]),
            meta["floor_z"],
            meta["frame_id"],
            meta.get("metadata", {}),
        )
        if scene is not None:
            result.check_scene(scene)
        elif previous := result.metadata.get("scene_source"):
            from vln_toolkit.scenes import Scene

            source = Path(previous["usd_path"])
            if source.is_file():
                recorded_scene = Scene(
                    result.metadata.get("scene_id", source.stem),
                    str(source),
                    {
                        "version": previous.get("version"),
                        "layers": [layer["path"] for layer in previous.get("layers", [])],
                    },
                )
                result.check_scene(recorded_scene)
            else:
                warnings.warn(
                    "Recorded scene source is unavailable; map revision cannot be verified",
                    UserWarning,
                    stacklevel=2,
                )
        return result

    def check_scene(self, scene) -> None:
        previous = self.metadata.get("scene_source")
        if previous is None or previous != scene.source_record():
            warnings.warn(
                "Map scene revision is missing or differs from the current scene; "
                "rebuild or verify this map before use",
                UserWarning,
                stacklevel=2,
            )


@register("occupancy")
class OccupancyBuilder:
    def build(self, scene, config=None) -> OccupancyMap:
        from vln_toolkit.sim.isaac_runtime import generate_occupancy

        return generate_occupancy(scene, config or {})


def build_occupancy(scene, config=None) -> OccupancyMap:
    return OccupancyBuilder().build(scene, config)


def load_occupancy(directory, **kwargs) -> OccupancyMap:
    return OccupancyMap.load(directory, **kwargs)


def save_occupancy(occupancy, directory) -> None:
    occupancy.save(directory)
