from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_value: str | Path, base_dir: str | Path | None = None) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    if base_dir is None:
        base_dir = get_project_root()

    return Path(base_dir) / path


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_text_file(path: str | Path) -> str:
    path = Path(path)

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return f.read()