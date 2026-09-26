from __future__ import annotations

import secrets
import string
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def generate_run_id(current_time: datetime | None = None) -> str:
    current_time = datetime.now() if current_time is None else current_time
    random_suffix = "".join(secrets.choice(string.ascii_uppercase) for _ in range(6))
    return f"{current_time.strftime('%Y-%m-%d_%H-%M-%S')}_{random_suffix}"


def make_run_dir(run_group: str, run_id: str) -> Path:
    _validate_run_path_part(run_group, name="run_group")
    _validate_run_path_part(run_id, name="run_id")
    return REPO_ROOT / "runs" / run_group / run_id


def get_run_id_from_checkpoint_path(load_path: str | Path) -> str:
    checkpoint_path = Path(load_path)
    models_dir = next(
        (parent for parent in checkpoint_path.parents if parent.name == "models"),
        None,
    )
    if models_dir is None:
        return checkpoint_path.stem
    return models_dir.parent.name


def _validate_run_path_part(value: str, *, name: str) -> None:
    path = Path(value)
    if not value or path.is_absolute() or path.name != value or value in {".", ".."}:
        raise ValueError(f"{name} must be a single relative path name, got {value!r}")
