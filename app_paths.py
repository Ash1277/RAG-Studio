"""Centralized, platform-aware RAG Studio paths and safe Windows legacy migration.

Development keeps the historical ./data directory unless overridden by
RAG_STUDIO_DATA_DIR or RAG_STUDIO_HOME. Frozen desktop builds use the OS
user-data directory and safely migrate legacy RAGDocumentAssistant/data by copy
so old installations remain untouched.
"""
from __future__ import annotations
import logging
import os
import shutil
import sys
from pathlib import Path

logger = logging.getLogger("rag_studio.paths")

APP_ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def _migrate_legacy_windows_data(legacy_dir: Path, target_dir: Path) -> None:
    """Copy legacy Windows data to the new target directory once without deleting the original."""
    marker = target_dir / ".migrated_from_legacy"
    if not legacy_dir.exists() or not legacy_dir.is_dir():
        return
    if marker.exists():
        return
    uploads_nonempty = (target_dir / "uploads").exists() and any((target_dir / "uploads").iterdir())
    sessions_nonempty = (target_dir / "sessions").exists() and any((target_dir / "sessions").glob("*.json"))
    has_existing = (target_dir / "library.json").exists() or uploads_nonempty or sessions_nonempty
    if has_existing:
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            marker.write_text(f"Existing data preserved; skipped copy from {legacy_dir}\n", encoding="utf-8")
        except OSError:
            pass
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        for item in ("uploads", "sessions", "chroma", "versions", "library.json", "settings.json"):
            src = legacy_dir / item
            dst = target_dir / item
            if not src.exists():
                continue
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            elif src.is_file() and not dst.exists():
                shutil.copy2(src, dst)
        marker.write_text(f"Migrated by copy from {legacy_dir}\n", encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not complete legacy Windows data migration from %s to %s: %s", legacy_dir, target_dir, exc)


def resolve_data_dir(
    env: dict | None = None,
    platform: str | None = None,
    frozen: bool | None = None,
    home: Path | None = None,
    app_dir: Path | None = None,
) -> Path:
    """Resolve the active RAG Studio data directory with override and migration support."""
    env_map = os.environ if env is None else env
    plat = sys.platform if platform is None else platform
    is_frozen = getattr(sys, "frozen", False) if frozen is None else frozen
    home_dir = Path.home() if home is None else Path(home)
    base_app_dir = Path(__file__).resolve().parent if app_dir is None else Path(app_dir)

    override = (env_map.get("RAG_STUDIO_DATA_DIR") or env_map.get("RAG_STUDIO_HOME") or "").strip()
    if override:
        return Path(override).expanduser().resolve()

    if not is_frozen:
        return base_app_dir / "data"

    if plat == "win32":
        local_appdata = Path(env_map.get("LOCALAPPDATA") or (home_dir / "AppData" / "Local"))
        legacy = local_appdata / "RAGDocumentAssistant" / "data"
        target = local_appdata / "RAG Studio" / "data"
        _migrate_legacy_windows_data(legacy, target)
        return target

    if plat == "darwin":
        return home_dir / "Library" / "Application Support" / "RAG Studio"

    xdg_base = Path(env_map.get("XDG_DATA_HOME") or (home_dir / ".local" / "share"))
    return xdg_base / "rag-studio"


def user_data_dir() -> Path:
    return resolve_data_dir()


def ensure_data_subdirs(data_dir: Path | None = None) -> dict[str, Path]:
    base = resolve_data_dir() if data_dir is None else Path(data_dir)
    paths = {
        "data": base,
        "uploads": base / "uploads",
        "sessions": base / "sessions",
        "chroma": base / "chroma",
        "hf_cache": base / "hf_cache",
        "versions": base / "versions",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


_INITIAL = ensure_data_subdirs()
DATA_DIR = _INITIAL["data"]
UPLOADS_DIR = _INITIAL["uploads"]
SESSIONS_DIR = _INITIAL["sessions"]
CHROMA_DIR = _INITIAL["chroma"]
HF_CACHE_DIR = _INITIAL["hf_cache"]
VERSIONS_DIR = _INITIAL["versions"]
