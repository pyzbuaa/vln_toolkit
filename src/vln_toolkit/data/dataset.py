"""Internal JSONL episodes and lossless NumPy trajectory storage."""

import json
from pathlib import Path

import numpy as np

from vln_toolkit.utils.config import load_config, save_json

from .types import NavigationEpisode


def save_episodes(episodes, output_dir, metadata=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, seen = [], set()
    for episode in episodes:
        if episode.episode_id in seen:
            raise ValueError(f"Duplicate episode ID: {episode.episode_id}")
        seen.add(episode.episode_id)
        records.append(json.dumps(episode.to_dict(), ensure_ascii=False, allow_nan=False))
    (output_dir / "episodes.jsonl").write_text(
        "".join(line + "\n" for line in records), encoding="utf-8"
    )
    save_json(
        output_dir / "metadata.json",
        {
            **(metadata or {}),
            "schema_version": 1,
            "episode_count": len(records),
            "coordinates": {
                "up_axis": "Z",
                "units": "metres",
                "rotation": "xyzw",
                "angles": "radians",
            },
        },
    )


def load_episodes(path):
    path = Path(path)
    if path.is_dir():
        path /= "episodes.jsonl"
    episodes, seen = [], set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            episode = NavigationEpisode.from_dict(json.loads(line))
            if episode.episode_id in seen:
                raise ValueError(f"Duplicate episode ID: {episode.episode_id}")
            seen.add(episode.episode_id)
            episodes.append(episode)
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"Invalid episode at {path}:{number}: {exc}") from exc
    return episodes


class TrajectoryRecorder:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)

    def save(self, episode_id, trajectory):
        trajectory.validate()
        if (
            episode_id != trajectory.episode_id
            or Path(episode_id).name != episode_id
            or episode_id in {"", ".", ".."}
        ):
            raise ValueError("Expected a matching episode ID with no path components")
        for name, frames in trajectory.observations.items():
            if name not in {"rgb", "depth"} or np.asarray(frames).dtype.kind not in "buif":
                raise ValueError(f"Unsupported observation channel or nonnumeric frames: {name}")
        commands = np.asarray(
            [json.dumps(action, allow_nan=False) for action in trajectory.actions]
        )
        directory = self.output_dir / "trajectories" / episode_id
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "states.npz",
            timestamps=trajectory.timestamps,
            positions=[p["position"] for p in trajectory.poses],
            rotations_xyzw=[p["rotation_xyzw"] for p in trajectory.poses],
        )
        np.savez_compressed(directory / "actions.npz", commands=commands)
        for name in {"rgb", "depth"} - trajectory.observations.keys():
            (directory / f"{name}.npz").unlink(missing_ok=True)
        for name, frames in trajectory.observations.items():
            np.savez_compressed(
                directory / f"{name}.npz",
                frames=np.asarray(frames),
                timestamps=trajectory.timestamps,
            )
        save_json(
            directory / "metadata.json",
            {
                "episode_id": episode_id,
                "mode": trajectory.mode,
                "status": trajectory.status,
                "action_observation_alignment": "o_t precedes a_t; final action is stop",
                "depth_units": "metres",
                "rgb_storage": "lossless npz",
                "metadata": trajectory.metadata,
            },
        )

    def load(self, episode_id):
        from .trajectory import Trajectory

        if Path(episode_id).name != episode_id or episode_id in {"", ".", ".."}:
            raise ValueError("Expected an episode ID with no path components")
        directory = self.output_dir / "trajectories" / episode_id
        meta = load_config(directory / "metadata.json")
        with np.load(directory / "states.npz", allow_pickle=False) as states:
            timestamps = states["timestamps"].tolist()
            poses = [
                {"position": p.tolist(), "rotation_xyzw": r.tolist()}
                for p, r in zip(states["positions"], states["rotations_xyzw"])
            ]
        with np.load(directory / "actions.npz", allow_pickle=False) as actions:
            commands = [json.loads(value) for value in actions["commands"]]
        observations = {}
        for name in ("rgb", "depth"):
            path = directory / f"{name}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as archive:
                    if not np.array_equal(archive["timestamps"], timestamps):
                        raise ValueError("Observation timestamps do not match states")
                    observations[name] = archive["frames"]
        result = Trajectory(
            episode_id,
            timestamps,
            poses,
            commands,
            observations,
            meta["mode"],
            meta["status"],
            meta.get("metadata", {}),
        )
        result.validate()
        return result
