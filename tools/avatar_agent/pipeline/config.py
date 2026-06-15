from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


DEFAULT_PROJECT_ROOT = Path("/scratch/e1554543/avatar_system_full")


def project_root() -> Path:
    return Path(os.environ.get("AVATAR_SYSTEM_ROOT", str(DEFAULT_PROJECT_ROOT))).expanduser().resolve()


def project_path(*parts: str) -> Path:
    return project_root().joinpath(*parts)


def _expand_value(value: Any, root: Path) -> Any:
    if isinstance(value, dict):
        return {key: _expand_value(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_value(item, root) for item in value]
    if isinstance(value, str):
        return (
            value.replace("${PROJECT_ROOT}", str(root))
            .replace("$PROJECT_ROOT", str(root))
            .replace("${AVATAR_SYSTEM_ROOT}", str(root))
            .replace("$AVATAR_SYSTEM_ROOT", str(root))
        )
    return value


def load_pipeline_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return _expand_value(config, project_root())
