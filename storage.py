"""Atomic JSON persistence and workspace synchronization helpers."""
from __future__ import annotations
import json
import os
import threading
import uuid
from pathlib import Path

WORKSPACE_LOCK = threading.RLock()


def read_json(path: Path, default):
    try:
        raw = Path(path).read_text(encoding="utf-8")
        value = json.loads(raw)
        return value if isinstance(value, type(default)) else default
    except (OSError, ValueError, TypeError):
        return default


def write_json_atomic(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = json.dumps(value, ensure_ascii=False, indent=2)
    try:
        with open(temporary, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        temporary.replace(path)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass
