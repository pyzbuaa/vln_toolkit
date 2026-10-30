"""Small JSON/YAML configuration helpers."""

import json
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        result = json.load(stream) if path.suffix == ".json" else yaml.safe_load(stream)
    if not isinstance(result, dict):
        raise TypeError(f"Expected a configuration mapping: {path}")
    return result


def save_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
