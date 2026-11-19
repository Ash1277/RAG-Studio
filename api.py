import difflib
import hashlib
import io
import json
import logging
import os
import re
import shutil
import tempfile
import time
import traceback
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app_paths import DATA_DIR, UPLOADS_DIR, SESSIONS_DIR, CHROMA_DIR, VERSIONS_DIR
from storage import read_json, write_json_atomic, WORKSPACE_LOCK
import security
import rag_core as rag
from quiz_service import parse_quiz_response, generate_quiz_batches

ROOT = Path(__file__).resolve().parent
DATA = DATA_DIR
UPLOADS = UPLOADS_DIR
SESSIONS = SESSIONS_DIR
CHROMA = CHROMA_DIR
VERSIONS = VERSIONS_DIR
LIBRARY = DATA / "library.json"
SETTINGS = DATA / "settings.json"

for _p in (DATA, UPLOADS, SESSIONS, CHROMA, VERSIONS):
    _p.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="RAG Studio API", version="1.0")
logger = logging.getLogger("rag_studio.api")
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


@app.middleware("http")
async def _disable_static_cache(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    path = getattr(request, "url", None)
    path_str = getattr(path, "path", "") if path else ""
    if path_str == "/" or path_str.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


def _shutdown_cleanup():
    try:
        if hasattr(rag, "close_store"):
            rag.close_store()
    except Exception:
        pass


if hasattr(app, "add_event_handler"):
    app.add_event_handler("shutdown", _shutdown_cleanup)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    session_id: Optional[str] = None
    use_reranker: bool = False
    top_k: int = Field(default=5, ge=1, le=12)
    source_ids: Optional[list[str]] = None


class ResearchRequest(BaseModel):
    task: str = Field(min_length=3, max_length=3000)
    sources: list[str] = Field(default_factory=list)
    use_reranker: bool = False


class StudyRequest(BaseModel):
    source: str = "all"
    kind: str = "quiz"
    count: int = Field(default=5, ge=1, le=15)


class KeyRequest(BaseModel):
    api_key: str = Field(min_length=20, max_length=512)


class SourceSelectionRequest(BaseModel):
    source_ids: list[str] = Field(default_factory=list)


class SessionRenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class PreferencesRequest(BaseModel):
    theme: str = "ocean"
    default_top_k: int = Field(default=5, ge=1, le=12)
    use_reranker: bool = False


def session_path(sid: str) -> Path:
    if not isinstance(sid, str) or not re.fullmatch(r"[A-Za-z0-9-]{1,80}", sid):
        raise HTTPException(400, "Invalid session ID")
    return SESSIONS / f"{sid}.json"


def read_session(sid: str) -> dict:
    p = session_path(sid)
    if p.exists():
        value = read_json(p, {})
        if value and isinstance(value, dict):
            if not isinstance(value.get("messages"), list):
                value["messages"] = []
            if not isinstance(value.get("source_ids"), list):
                value["source_ids"] = []
            return value
    return {
        "id": sid,
        "title": "New conversation",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "messages": [],
        "source_ids": [],
    }


def save_session(obj: dict) -> None:
    write_json_atomic(session_path(obj["id"]), obj)


def clean_name(name: str) -> str:
    cleaned = Path(str(name or "upload.txt").replace("\\", "/")).name.strip()
    return cleaned or "upload.txt"


def read_library() -> dict:
    data = read_json(LIBRARY, {"files": {}, "urls": {}, "versions": {}})
    if not isinstance(data.get("files"), dict):
        data["files"] = {}
    if not isinstance(data.get("urls"), dict):
        data["urls"] = {}
    if not isinstance(data.get("versions"), dict):
        data["versions"] = {}
    return data


def write_library(data: dict) -> None:
    write_json_atomic(LIBRARY, data)


def read_preferences() -> dict:
    defaults = {"theme": "ocean", "default_top_k": 5, "use_reranker": False}
    data = read_json(SETTINGS, defaults)
    if not isinstance(data, dict):
        return defaults
    allowed_themes = {"ocean", "forest", "lavender", "sand", "midnight", "ghost", "ghost-dark", "dark"}
    theme = str(data.get("theme") or "ocean").lower()
    if theme not in allowed_themes:
        theme = "ocean"
    top_k = data.get("default_top_k", 5)
    if not isinstance(top_k, int) or not (1 <= top_k <= 12):
        top_k = 5
    return {
        "theme": theme,
        "default_top_k": top_k,
        "use_reranker": bool(data.get("use_reranker", False)),
    }


def write_preferences(prefs: dict) -> dict:
    clean = {
        "theme": prefs.get("theme", "ocean"),
        "default_top_k": int(prefs.get("default_top_k", 5)),
        "use_reranker": bool(prefs.get("use_reranker", False)),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json_atomic(SETTINGS, clean)
    return clean


def normalize_url(url: str) -> str:
    from urllib.parse import urlsplit, urlunsplit
    parts = urlsplit(url.strip())
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), parts.query, "")
    )


@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health():
    sources = rag.list_sources()
    key_info = security.get_api_key_info()
    return {
        "ok": True,
        "version": "1.0",
        "groq_configured": bool(key_info["configured"]),
        "key_source": key_info["source"],
        "model": rag.GROQ_MODEL,
        "embedding_model": rag.EMBEDDING_MODEL,
        "documents": len(sources),
        "data_dir": str(DATA),
        "preferences": read_preferences(),
    }


@app.get("/api/settings/preferences")
def get_preferences():
    with WORKSPACE_LOCK:
        return read_preferences()


@app.put("/api/settings/preferences")
def update_preferences(req: PreferencesRequest):
    allowed_themes = {"ocean", "forest", "lavender", "sand", "midnight", "ghost", "ghost-dark", "dark"}
    theme = req.theme.strip().lower()
    if theme not in allowed_themes:
        raise HTTPException(400, "Unsupported theme name.")
    with WORKSPACE_LOCK:
        saved = write_preferences(
            {"theme": theme, "default_top_k": req.default_top_k, "use_reranker": req.use_reranker}
        )
        return saved


@app.get("/api/settings/key")
def key_status():
    info = security.get_api_key_info()
    return {
        "configured": bool(info["configured"]),
        "source": info["source"],
    }


@app.put("/api/settings/key")
def save_key(req: KeyRequest):
    try:
        security.validate_api_key(req.api_key)
        backend = security.set_api_key(req.api_key)
        return {
            "configured": True,
            "storage": backend,
            "message": "Groq API key validated and saved in the operating-system credential store.",
        }
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except security.CredentialStoreError as exc:
        raise HTTPException(503, str(exc))


@app.delete("/api/settings/key")
def remove_key():
    try:
        security.delete_api_key()
        info = security.get_api_key_info()
        configured = bool(info["configured"])
        return {
            "configured": configured,
            "source": info["source"],
            "message": (
                "Credential-store key removed."
                if not configured
                else "Credential-store key removed; a developer .env fallback is still active."
            ),
        }
    except security.CredentialStoreError as exc:
        raise HTTPException(503, str(exc))


@app.get("/api/documents")
def documents():
    with WORKSPACE_LOCK:
        indexed = {x["id"]: dict(x) for x in rag.list_sources()}
        registry = read_library()
        versions_map = registry.get("versions", {})
        meta_by_id = {}
        for group, source_type in (("files", "document"), ("urls", "web")):
            for digest, item in registry.get(group, {}).items():
                if not isinstance(item, dict):
                    continue
                did = item.get("id")
                if not did:
                    continue
                meta_by_id[did] = item
                if did not in indexed:
                    indexed[did] = {
                        "id": did,
                        "doc_id": did,
                        "name": item.get("name", "Unknown source"),
                        "chunks": item.get("chunks", 0),
                        "type": source_type,
                        "registry_only": True,
                    }
        files = []
        for did, item in indexed.items():
            local = next((p for p in UPLOADS.glob(f"{did}__*") if p.is_file()), None)
            is_indexed = not item.get("registry_only", False) and item.get("chunks", 0) > 0
            reg_meta = meta_by_id.get(did, {})
            ver_history = versions_map.get(did, []) if isinstance(versions_map, dict) else []
            current_ver = int(reg_meta.get("version") or (len(ver_history) + 1))
            files.append(
                {
                    **item,
                    "size": local.stat().st_size if local else None,
                    "indexed": is_indexed,
                    "version": current_ver,
                    "versions_count": len(ver_history) + 1,
                    "has_versions": len(ver_history) > 0,
                    "modified": (
                        datetime.fromtimestamp(local.stat().st_mtime).isoformat(timespec="minutes")
                        if local
                        else None
                    ),
                }
            )
        return {"documents": sorted(files, key=lambda x: x["name"].lower()), "count": len(files)}


@app.post("/api/upload")
def upload(files: list[UploadFile] = File(...)):
    output = []
    allowed = {
        ".pdf", ".txt", ".md", ".docx", ".xlsx", ".xls", ".pptx", ".csv",
        ".png", ".jpg", ".jpeg", ".webp",
    }
    if len(files) > 20:
        raise HTTPException(
            400, "A batch can contain at most 20 files. Please split the selection into smaller batches."
        )
    with WORKSPACE_LOCK:
        library = read_library()
        current_sources = rag.list_sources()
        indexed_by_id = {x.get("id"): x for x in current_sources if x.get("id")}

        for file in files:
            name = clean_name(file.filename or "upload.txt")
            suffix = Path(name).suffix.lower()
            did = uuid.uuid4().hex
            if suffix not in allowed:
                output.append(
                    {
                        "id": did,
                        "name": name,
                        "status": "error",
                        "error": f"Unsupported file type: {suffix}. Supported: PDF, TXT, MD, DOCX, XLSX, XLS, PPTX, CSV, PNG, JPG, WEBP.",
                    }
                )
                continue
            content = file.file.read()
            if len(content) > 40 * 1024 * 1024:
                output.append({"id": did, "name": name, "status": "error", "error": "File exceeds 40 MB limit."})
                continue
            if not content:
                output.append({"id": did, "name": name, "status": "error", "error": "The uploaded file is empty."})
                continue
            digest = hashlib.sha256(content).hexdigest()

            if digest not in library["files"]:
                for prior in UPLOADS.glob("*__*"):
                    try:
                        prior_id = prior.name.split("__", 1)[0]
                        if (
                            hashlib.sha256(prior.read_bytes()).hexdigest() == digest
                            and prior_id in indexed_by_id
                        ):
                            prior_name = prior.name.split("__", 1)[1]
                            prior_source = indexed_by_id.get(prior_id, {})
                            library["files"][digest] = {
                                "id": prior_id,
                                "name": prior_source.get("name", prior_name),
                                "chunks": prior_source.get("chunks", 0),
                                "version": 1,
                            }
                            write_library(library)
                            break
                    except OSError:
                        continue

            existing = library["files"].get(digest)
            if existing and existing.get("id") in indexed_by_id:
                output.append(
                    {
                        "id": existing["id"],
                        "name": existing.get("name", name),
                        "status": "duplicate",
                        "chunks": existing.get("chunks", 0),
                        "version": existing.get("version", 1),
                        "message": "This exact file is already in your document library; existing index reused.",
                    }
                )
                continue

            # Phase 11.2: Check if a document with the same filename already exists with a different content hash
            same_name_digest = None
            same_name_entry = None
            for old_digest, entry in list(library["files"].items()):
                if (
                    isinstance(entry, dict)
                    and str(entry.get("name", "")).lower() == name.lower()
                    and old_digest != digest
                ):
                    same_name_digest = old_digest
                    same_name_entry = entry
                    break

            if same_name_entry and same_name_entry.get("id"):
                did = same_name_entry["id"]
                prev_ver = int(same_name_entry.get("version") or 1)
                new_ver = prev_ver + 1
                old_upload = next((p for p in UPLOADS.glob(f"{did}__*") if p.is_file()), None)
                ver_dir = VERSIONS / did
                ver_dir.mkdir(parents=True, exist_ok=True)
                archived_filename = f"v{prev_ver}_{same_name_digest[:8]}__{name}"
                archived_path = ver_dir / archived_filename
                if old_upload and old_upload.exists():
                    shutil.copy2(old_upload, archived_path)

                tmp_new_target = UPLOADS / f".tmp_{did}__{name}"
                tmp_new_target.write_bytes(content)
                try:
                    docs = rag.load_file(tmp_new_target, name, did)
                    # Remove old vectors and index new version under the same doc_id
                    rag.remove_doc_id(did)
                    stats = rag.index_docs(docs)
                    if stats["chunks"] == 0:
                        raise ValueError("No extractable text was found; updated version was not indexed.")
                    if old_upload and old_upload.exists() and old_upload != (UPLOADS / f"{did}__{name}"):
                        old_upload.unlink(missing_ok=True)
                    final_target = UPLOADS / f"{did}__{name}"
                    tmp_new_target.replace(final_target)

                    # Record version archive metadata
                    ver_list = library.setdefault("versions", {}).setdefault(did, [])
                    ver_list.append(
                        {
                            "version": prev_ver,
                            "digest": same_name_digest,
                            "name": same_name_entry.get("name", name),
                            "chunks": same_name_entry.get("chunks", 0),
                            "archived_at": datetime.now().isoformat(timespec="seconds"),
                            "archive_file": archived_filename,
                        }
                    )
                    library["files"].pop(same_name_digest, None)
                    library["files"][digest] = {
                        "id": did,
                        "name": name,
                        "chunks": stats["chunks"],
                        "version": new_ver,
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                    }
                    write_library(library)
                    indexed_by_id[did] = {"id": did, "name": name, "chunks": stats["chunks"]}
                    output.append(
                        {
                            "id": did,
                            "name": name,
                            "status": "updated",
                            "version": new_ver,
                            "previous_version": prev_ver,
                            "chunks": stats["chunks"],
                            "pages": stats["pages"],
                            "message": f"Updated {name} to v{new_ver} (v{prev_ver} archived).",
                        }
                    )
                except Exception as exc:
                    tmp_new_target.unlink(missing_ok=True)
                    # Restore previous vectors if old_upload was preserved
                    if old_upload and old_upload.exists():
                        try:
                            rag.remove_doc_id(did)
                            old_docs = rag.load_file(old_upload, same_name_entry.get("name", name), did)
                            rag.index_docs(old_docs)
                        except Exception as rb_exc:
                            logger.error("Failed to re-index previous version after update failure for %s: %s", did, rb_exc)
                    output.append({"id": did, "name": name, "status": "error", "error": str(exc)})
                continue

            target = UPLOADS / f"{did}__{name}"
            target.write_bytes(content)
            indexed_called = False
            try:
                docs = rag.load_file(target, name, did)
                indexed_called = True
                stats = rag.index_docs(docs)
                if stats["chunks"] == 0:
                    raise ValueError("No extractable text was found; this file was not indexed.")
                library["files"][digest] = {
                    "id": did,
                    "name": name,
                    "chunks": stats["chunks"],
                    "version": 1,
                    "added_at": datetime.now().isoformat(timespec="seconds"),
                }
                write_library(library)
                indexed_by_id[did] = {"id": did, "name": name, "chunks": stats["chunks"]}
                output.append(
                    {
                        "id": did,
                        "name": name,
                        "status": "indexed",
                        "version": 1,
                        "chunks": stats["chunks"],
                        "pages": stats["pages"],
                    }
                )
            except Exception as exc:
                target.unlink(missing_ok=True)
                if indexed_called:
                    try:
                        rag.remove_doc_id(did)
                    except Exception as rb_exc:
                        logger.error("Rollback of partially indexed vectors failed for %s: %s", did, rb_exc)
                output.append({"id": did, "name": name, "status": "error", "error": str(exc)})
        return {"results": output, "documents": documents()["documents"]}


def _extract_full_text_for_diff(path: Path, name: str, did: str) -> str:
    try:
        docs = rag.load_file(path, name, did)
        return "\n\n".join(str(getattr(d, "page_content", "")).strip() for d in docs if getattr(d, "page_content", "")).strip()
    except Exception:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""


@app.get("/api/documents/{doc_id}/versions")
def get_document_versions(doc_id: str):
    if not doc_id or "/" in doc_id or "\\" in doc_id or ".." in doc_id:
        raise HTTPException(400, "Invalid document ID.")
    with WORKSPACE_LOCK:
        library = read_library()
        current_digest = None
        current_entry = None
        for dg, item in library.get("files", {}).items():
            if isinstance(item, dict) and item.get("id") == doc_id:
                current_digest = dg
                current_entry = item
                break
        if not current_entry:
            raise HTTPException(404, "Document not found in library.")
        archived = list(library.get("versions", {}).get(doc_id, []))
        current_ver = int(current_entry.get("version") or (len(archived) + 1))

        diff_summary = None
        if archived:
            latest_arch = archived[-1]
            arch_path = VERSIONS / doc_id / str(latest_arch.get("archive_file", ""))
            live_path = next((p for p in UPLOADS.glob(f"{doc_id}__*") if p.is_file()), None)
            if arch_path.exists() and live_path and live_path.exists():
                old_txt = _extract_full_text_for_diff(arch_path, latest_arch.get("name", ""), doc_id).splitlines()
                new_txt = _extract_full_text_for_diff(live_path, current_entry.get("name", ""), doc_id).splitlines()
                udiff = list(
                    difflib.unified_diff(
                        old_txt,
                        new_txt,
                        fromfile=f"v{latest_arch.get('version')} ({latest_arch.get('name')})",
                        tofile=f"v{current_ver} ({current_entry.get('name')})",
                        lineterm="",
                        n=2,
                    )
                )
                added = sum(1 for ln in udiff if ln.startswith("+") and not ln.startswith("+++"))
                removed = sum(1 for ln in udiff if ln.startswith("-") and not ln.startswith("---"))
                diff_summary = {
                    "from_version": latest_arch.get("version"),
                    "to_version": current_ver,
                    "lines_added": added,
                    "lines_removed": removed,
                    "diff_preview": "\n".join(udiff[:80]) if udiff else "No text line changes detected.",
                }

        return {
            "id": doc_id,
            "name": current_entry.get("name"),
            "current": {
                "version": current_ver,
                "digest": current_digest,
                "chunks": current_entry.get("chunks", 0),
                "updated_at": current_entry.get("updated_at") or current_entry.get("added_at"),
            },
            "archived_versions": list(reversed(archived)),
            "latest_diff": diff_summary,
        }


@app.post("/api/documents/{doc_id}/versions/{version_num}/restore")
def restore_document_version(doc_id: str, version_num: int):
    if not doc_id or "/" in doc_id or "\\" in doc_id or ".." in doc_id:
        raise HTTPException(400, "Invalid document ID.")
    with WORKSPACE_LOCK:
        library = read_library()
        current_digest = None
        current_entry = None
        for dg, item in library.get("files", {}).items():
            if isinstance(item, dict) and item.get("id") == doc_id:
                current_digest = dg
                current_entry = item
                break
        if not current_entry:
            raise HTTPException(404, "Document not found in library.")

        archived_list = library.get("versions", {}).get(doc_id, [])
        target_arch = next((v for v in archived_list if int(v.get("version", -1)) == int(version_num)), None)
        if not target_arch:
            raise HTTPException(404, f"Archived version v{version_num} not found.")

        arch_path = VERSIONS / doc_id / str(target_arch.get("archive_file", ""))
        if not arch_path.exists():
            raise HTTPException(404, "Archived version file is missing on disk.")

        live_path = next((p for p in UPLOADS.glob(f"{doc_id}__*") if p.is_file()), None)
        current_ver = int(current_entry.get("version") or (len(archived_list) + 1))
        name = current_entry.get("name") or target_arch.get("name") or "restored.txt"

        # Archive the current live file before restoring the older version
        ver_dir = VERSIONS / doc_id
        ver_dir.mkdir(parents=True, exist_ok=True)
        backup_live_name = f"v{current_ver}_{(current_digest or 'live')[:8]}__{name}"
        if live_path and live_path.exists():
            shutil.copy2(live_path, ver_dir / backup_live_name)

        restored_bytes = arch_path.read_bytes()
        restored_digest = hashlib.sha256(restored_bytes).hexdigest()
        new_live_path = UPLOADS / f"{doc_id}__{name}"
        new_live_path.write_bytes(restored_bytes)

        docs = rag.load_file(new_live_path, name, doc_id)
        rag.remove_doc_id(doc_id)
        stats = rag.index_docs(docs)

        new_ver_num = current_ver + 1
        archived_list.append(
            {
                "version": current_ver,
                "digest": current_digest,
                "name": name,
                "chunks": current_entry.get("chunks", 0),
                "archived_at": datetime.now().isoformat(timespec="seconds"),
                "archive_file": backup_live_name,
            }
        )
        if current_digest:
            library["files"].pop(current_digest, None)
        library["files"][restored_digest] = {
            "id": doc_id,
            "name": name,
            "chunks": stats["chunks"],
            "version": new_ver_num,
            "restored_from": version_num,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        write_library(library)
        return {
            "restored": True,
            "id": doc_id,
            "name": name,
            "restored_from_version": version_num,
            "new_version": new_ver_num,
            "chunks": stats["chunks"],
            "documents": documents()["documents"],
        }


def _validate_public_url(url: str) -> str:
    """Block SSRF attempts targeting localhost, loopback, private LANs, or link-local metadata endpoints."""
    import ipaddress
    from urllib.parse import urlsplit
    cleaned = str(url or "").strip()
    parts = urlsplit(cleaned)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise HTTPException(400, "Enter a complete public http:// or https:// URL.")
    host = parts.hostname.strip().lower().rstrip(".")
    blocked_hosts = {"localhost", "0.0.0.0", "127.0.0.1", "::1", "metadata.google.internal"}
    if host in blocked_hosts or host.endswith(".local") or host.endswith(".internal"):
        raise HTTPException(400, "Local or private network URLs are blocked for security.")
    try:
        ip_obj = ipaddress.ip_address(host)
        if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_reserved or ip_obj.is_multicast:
            raise HTTPException(400, "Private, loopback, or link-local IP addresses are blocked for security.")
    except ValueError:
        pass
    return cleaned


@app.post("/api/url")
def add_url(url: str = Form(...)):
    url = _validate_public_url(url)
    with WORKSPACE_LOCK:
        library = read_library()
        key = normalize_url(url)
        existing = library["urls"].get(key)
        indexed_ids = {x.get("id") for x in rag.list_sources()}
        if existing and existing.get("id") in indexed_ids:
            return {
                "status": "duplicate",
                "id": existing["id"],
                "name": existing.get("name", url),
                "chunks": existing.get("chunks", 0),
                "message": "This URL is already in your website library; existing index reused.",
                "documents": documents()["documents"],
            }
        did = uuid.uuid4().hex
        indexed_called = False
        try:
            docs = rag.load_url(url, did)
            indexed_called = True
            stats = rag.index_docs(docs)
            if not stats["chunks"]:
                raise ValueError("The page contained no extractable text.")
            library["urls"][key] = {
                "id": did,
                "name": url,
                "chunks": stats["chunks"],
                "added_at": datetime.now().isoformat(timespec="seconds"),
            }
            write_library(library)
            return {
                "status": "indexed",
                "id": did,
                "name": url,
                "chunks": stats["chunks"],
                "documents": documents()["documents"],
            }
        except Exception as exc:
            if indexed_called:
                try:
                    rag.remove_doc_id(did)
                except Exception as rb_exc:
                    logger.error("Rollback of partially indexed URL vectors failed for %s: %s", did, rb_exc)
            raise HTTPException(400, f"Could not fetch/index URL: {exc}")


@app.delete("/api/documents")
def clear_documents():
    with WORKSPACE_LOCK:
        docs_before = documents()["documents"]
        sources_count = len(docs_before)
        try:
            if hasattr(rag, "clear_all_documents"):
                chunks_removed = rag.clear_all_documents()
            else:
                chunks_removed = 0
                for item in rag.list_sources():
                    did = item.get("id")
                    if did:
                        chunks_removed += rag.remove_doc_id(did)
        except Exception as exc:
            logger.exception("Vector clearing failed; preserving local files and library metadata")
            raise HTTPException(
                500,
                f"Could not clear vector chunks from Chroma. Local files and library entries were preserved: {exc}",
            )

        files_removed = 0
        if UPLOADS.exists():
            for p in UPLOADS.iterdir():
                if p.is_file():
                    p.unlink(missing_ok=True)
                    files_removed += 1
        if VERSIONS.exists():
            for p in VERSIONS.iterdir():
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                elif p.is_file():
                    p.unlink(missing_ok=True)

        write_library({"files": {}, "urls": {}, "versions": {}})

        for sp in SESSIONS.glob("*.json"):
            try:
                sess = read_json(sp, {})
                if isinstance(sess, dict) and sess.get("source_ids"):
                    sess["source_ids"] = []
                    write_json_atomic(sp, sess)
            except Exception:
                pass

        return {
            "cleared": True,
            "sources_removed": sources_count,
            "files_removed": files_removed,
            "chunks_removed": chunks_removed,
            "documents": [],
            "count": 0,
        }


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    if not doc_id or "/" in doc_id or "\\" in doc_id or ".." in doc_id:
        raise HTTPException(400, "Invalid document ID.")
    with WORKSPACE_LOCK:
        try:
            count = rag.remove_doc_id(doc_id)
        except Exception as exc:
            logger.exception("Vector deletion failed for document %s; preserving files and metadata", doc_id)
            raise HTTPException(
                500,
                f"Could not remove vector chunks for source {doc_id}. Local file and library entry were preserved: {exc}",
            )
        for p in UPLOADS.glob(f"{doc_id}__*"):
            if p.is_file():
                p.unlink(missing_ok=True)
        ver_dir = VERSIONS / doc_id
        if ver_dir.exists():
            shutil.rmtree(ver_dir, ignore_errors=True)
        library = read_library()
        for group in ("files", "urls"):
            library[group] = {k: v for k, v in library[group].items() if v.get("id") != doc_id}
        if isinstance(library.get("versions"), dict):
            library["versions"].pop(doc_id, None)
        write_library(library)
        for sp in SESSIONS.glob("*.json"):
            try:
                sess = read_json(sp, {})
                if isinstance(sess, dict) and isinstance(sess.get("source_ids"), list) and doc_id in sess["source_ids"]:
                    sess["source_ids"] = [s for s in sess["source_ids"] if s != doc_id]
                    write_json_atomic(sp, sess)
            except Exception:
                pass
        return {"deleted": doc_id, "chunks_removed": count, "documents": documents()["documents"]}


@app.get("/api/sessions")
def sessions():
    with WORKSPACE_LOCK:
        result = []
        for p in SESSIONS.glob("*.json"):
            try:
                s = read_json(p, {})
                if s and "id" in s:
                    result.append(
                        {
                            "id": s["id"],
                            "title": s.get("title", "Conversation"),
                            "created_at": s.get("created_at"),
                            "updated_at": s.get("updated_at"),
                            "source_ids": s.get("source_ids", []),
                        }
                    )
            except Exception:
                pass
        return {
            "sessions": sorted(
                result, key=lambda x: x.get("updated_at") or x.get("created_at") or "", reverse=True
            )
        }


@app.delete("/api/sessions")
def clear_sessions():
    with WORKSPACE_LOCK:
        removed = 0
        for p in SESSIONS.glob("*.json"):
            if p.is_file():
                p.unlink(missing_ok=True)
                removed += 1
        return {"deleted_count": removed}


@app.get("/api/sessions/{sid}")
def get_session(sid: str):
    with WORKSPACE_LOCK:
        return read_session(sid)


@app.patch("/api/sessions/{sid}")
@app.put("/api/sessions/{sid}/title")
def rename_session(sid: str, req: SessionRenameRequest):
    title = req.title.strip()
    if not title:
        raise HTTPException(400, "Session title cannot be empty.")
    with WORKSPACE_LOCK:
        sess = read_session(sid)
        sess["title"] = title[:120]
        sess["updated_at"] = datetime.now().isoformat(timespec="seconds")
        save_session(sess)
        return {"session_id": sid, "title": sess["title"], "session": sess}


@app.delete("/api/sessions/{sid}")
def delete_session(sid: str):
    with WORKSPACE_LOCK:
        p = session_path(sid)
        if p.exists():
            p.unlink()
        return {"deleted": sid}


@app.put("/api/sessions/{sid}/sources")
def update_session_sources(sid: str, req: SourceSelectionRequest):
    with WORKSPACE_LOCK:
        p = session_path(sid)
        valid_ids = {x.get("id") for x in rag.list_sources()}
        cleaned_ids = [str(s).strip() for s in req.source_ids if str(s).strip()]
        if set(cleaned_ids) - valid_ids:
            raise HTTPException(400, "One or more selected sources are no longer indexed.")
        sess = read_session(sid)
        sess["source_ids"] = list(dict.fromkeys(cleaned_ids))
        sess["updated_at"] = datetime.now().isoformat(timespec="seconds")
        save_session(sess)
        return {"session_id": sid, "source_ids": sess["source_ids"]}


@app.post("/api/chat")
def chat(req: ChatRequest):
    sid = req.session_id or uuid.uuid4().hex
    with WORKSPACE_LOCK:
        sess = read_session(sid)
        history = list(sess.get("messages", []))
        valid_ids = {x.get("id") for x in rag.list_sources()}
        if req.source_ids is not None:
            cleaned = [str(s).strip() for s in req.source_ids if str(s).strip()]
            unknown = set(cleaned) - valid_ids
            if unknown:
                raise HTTPException(
                    400, "One or more selected sources are no longer in the library. Refresh the source list."
                )
            sess["source_ids"] = list(dict.fromkeys(cleaned))
        selected_ids = [s for s in (sess.get("source_ids") or []) if s in valid_ids]
        sess["source_ids"] = selected_ids
        if not selected_ids:
            raise HTTPException(
                400, "Select at least one document or website in the Knowledge Base before sending a message."
            )

    started = time.time()
    try:
        answer, sources = rag.answer_question(
            req.question, history[-8:], req.use_reranker, req.top_k, selected_ids
        )
        if not str(answer or "").strip():
            answer = (
                "The model processed your selected sources but returned an empty reply. "
                "Please ask your question again."
            )
    except Exception as exc:
        raise HTTPException(500, str(exc))

    grounding = (
        rag.verify_answer_grounding(answer, sources)
        if hasattr(rag, "verify_answer_grounding")
        else None
    )

    with WORKSPACE_LOCK:
        usermsg = {
            "role": "user",
            "content": req.question,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        botmsg = {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "grounding": grounding,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "latency_seconds": round(time.time() - started, 2),
        }
        sess["messages"].extend([usermsg, botmsg])
        if sess.get("title") == "New conversation":
            sess["title"] = req.question[:48] + ("…" if len(req.question) > 48 else "")
        sess["updated_at"] = datetime.now().isoformat(timespec="seconds")
        save_session(sess)
        return {
            "session_id": sid,
            "answer": answer,
            "sources": sources,
            "grounding": grounding,
            "latency_seconds": botmsg["latency_seconds"],
            "session": sess,
        }


@app.post("/api/research")
def research(req: ResearchRequest):
    query = req.task.strip()
    chosen_sources = [str(s).strip() for s in (req.sources or []) if str(s).strip()]
    if len(chosen_sources) > 5:
        raise HTTPException(400, "Choose at most five documents to compare.")

    coverage = []
    if len(chosen_sources) >= 2:
        per_doc_k = max(2, min(4, 12 // len(chosen_sources)))
        if hasattr(rag, "retrieve_for_comparison"):
            candidates, coverage = rag.retrieve_for_comparison(
                query,
                source_ids=chosen_sources,
                per_source_k=per_doc_k,
                use_reranker=req.use_reranker,
            )
        else:
            candidates = []
            for sid in dict.fromkeys(chosen_sources):
                hits = rag.retrieve(query, top_k=per_doc_k, use_reranker=req.use_reranker, source_ids=[sid])
                candidates.extend(hits)
                coverage.append({"id": sid, "name": sid, "passages_retrieved": len(hits), "covered": len(hits) > 0})
    else:
        candidates = rag.retrieve(
            query, top_k=8, use_reranker=req.use_reranker, source_ids=chosen_sources or None
        )
        if chosen_sources:
            coverage = [
                {
                    "id": chosen_sources[0],
                    "name": chosen_sources[0],
                    "passages_retrieved": len(candidates),
                    "covered": len(candidates) > 0,
                }
            ]

    # Phase 9: Compute cell-level tabular diff summary if comparing 2+ Excel/CSV files
    tabular_diff = ""
    if len(chosen_sources) >= 2 and hasattr(rag, "compute_tabular_comparison_summary"):
        paths_with_names = []
        for sid in dict.fromkeys(chosen_sources):
            local_file = next((p for p in UPLOADS.glob(f"{sid}__*") if p.is_file()), None)
            if local_file and local_file.suffix.lower() in (".csv", ".xlsx", ".xls"):
                display = local_file.name.split("__", 1)[-1]
                paths_with_names.append((local_file, display))
        if len(paths_with_names) >= 2:
            tabular_diff = rag.compute_tabular_comparison_summary(paths_with_names)

    missing_docs = [c["name"] for c in coverage if not c.get("covered")]
    coverage_note = ""
    if coverage:
        summary_parts = [f"{c['name']} ({c['passages_retrieved']} passages)" for c in coverage]
        coverage_note = "Retrieved evidence coverage: " + ", ".join(summary_parts) + "."
        if missing_docs:
            coverage_note += " No matching passages were found in: " + ", ".join(missing_docs) + "."

    instruction = (
        "You are a careful document research analyst. Complete the requested task using ONLY the supplied evidence. "
        "Structure your response with clear Markdown headings: "
        "1) **Overview & Coverage**, 2) **Key Findings by Source**, 3) **Similarities & Agreements**, "
        "4) **Differences & Contrasts**, and 5) **Evidence Gaps & Limitations**. "
        "Cite every major finding inline as [Source N]. "
        "Do not claim a contradiction unless the sources address the exact same condition or metric with conflicting claims. "
        "If any selected document has zero retrieved passages or insufficient evidence on a point, state that explicitly rather than guessing."
    )
    prompt_parts = [query]
    if coverage_note:
        prompt_parts.append(coverage_note)
    if tabular_diff:
        prompt_parts.append(tabular_diff)
    prompt_query = "\n\n".join(prompt_parts).strip()

    try:
        result = rag.answer_with_context(instruction, prompt_query, candidates, max_tokens=2400)
    except Exception as exc:
        raise HTTPException(500, str(exc))

    grounding = (
        rag.verify_answer_grounding(result, candidates)
        if hasattr(rag, "verify_answer_grounding")
        else None
    )
    return {
        "result": result,
        "sources": candidates,
        "coverage": coverage,
        "tabular_diff": tabular_diff or None,
        "grounding": grounding,
    }


@app.post("/api/study")
def study(req: StudyRequest):
    docs = rag.retrieve(
        "key concepts definitions important facts core principles detailed explanations",
        top_k=12,
        source_ids=None if req.source == "all" else [req.source],
    )
    if not docs:
        raise HTTPException(400, "No matching indexed passages found for the selected source.")
    if req.kind not in ("quiz", "flashcards", "summary", "questions", "notes", "definitions"):
        raise HTTPException(400, "Unsupported study mode")

    if req.kind == "quiz":
        try:
            try:
                questions, warning = generate_quiz_batches(docs, req.count, allow_partial=True)
            except TypeError:
                questions = generate_quiz_batches(docs, req.count)
                warning = None
        except Exception as exc:
            logger.exception("Quiz generation failed after batch validation")
            raise HTTPException(
                502,
                {
                    "message": "The model could not produce a valid quiz batch.",
                    "reason": str(exc),
                    "model": rag.GROQ_MODEL,
                    "requested_questions": req.count,
                    "suggestion": "Try a smaller question count or a different source if this document has very little extractable text.",
                },
            )
        payload = {
            "result": warning or "Quiz generated successfully.",
            "quiz": questions,
            "sources": docs,
        }
        if warning:
            payload["warning"] = warning
        return payload

    if req.kind == "flashcards":
        instruction = (
            f"Create {req.count} concise flashcards from the evidence. "
            "Output readable Markdown with a clear **Front** and **Back** for each card. "
            "Cite factual claims with [Source N]. Do not invent facts."
        )
        token_budget = 2000
        prompt_question = f"Create {req.count} flashcards from the selected documents."
    elif req.kind == "questions":
        instruction = (
            f"Create {req.count} high-value study and review prompts from the supplied evidence. "
            "Frame them strictly as useful study prompts and self-check review questions—never claim they are guaranteed exam predictions. "
            "For each prompt, include a concise evidence-backed model answer outline and cite [Source N]. Do not invent facts."
        )
        token_budget = 2200
        prompt_question = f"Create {req.count} important study prompts and answer outlines from the selected documents."
    elif req.kind == "notes":
        instruction = (
            f"Create {req.count} comprehensive, structured study note sections from the supplied evidence. "
            "Organize each section with a clear heading, detailed bullet-point explanations, examples or mechanisms mentioned in the text, "
            "and inline [Source N] citations. Do not invent facts."
        )
        token_budget = 2600
        prompt_question = f"Create {req.count} detailed study note sections covering the selected documents."
    elif req.kind == "definitions":
        instruction = (
            f"Extract the {req.count} most important terms, concepts, and definitions from the supplied evidence. "
            "Format as a clean Markdown glossary (**Term** — Exact definition and context) with inline [Source N] citations. "
            "Do not invent definitions not supported by the text."
        )
        token_budget = 2000
        prompt_question = f"Extract {req.count} key definitions and terminology items from the selected documents."
    else:
        instruction = (
            "Create a structured revision summary from the evidence, organized into **Core Concepts**, "
            "**Key Definitions**, **Important Details**, and a **Quick Recap**. "
            "Cite factual claims with [Source N]. Do not invent facts."
        )
        token_budget = 2000
        prompt_question = "Create a structured revision summary from the selected documents."

    try:
        result = rag.answer_with_context(instruction, prompt_question, docs, max_tokens=token_budget)
    except Exception as exc:
        logger.exception("Study generation failed before response validation")
        raise HTTPException(502, f"Study generation failed: {type(exc).__name__}: {exc}")
    return {"result": result, "sources": docs}


@app.get("/api/export/session/{sid}")
def export_session(sid: str):
    with WORKSPACE_LOCK:
        p = session_path(sid)
        if not p.exists():
            raise HTTPException(404, "Session not found")
        return FileResponse(p, media_type="application/json", filename=f"rag-session-{sid}.json")


@app.get("/api/backup")
def backup():
    with WORKSPACE_LOCK:
        target = DATA / "rag-backup.zip"
        tmp_target = DATA / f".rag-backup.{uuid.uuid4().hex}.zip.tmp"
        try:
            with zipfile.ZipFile(tmp_target, "w", zipfile.ZIP_DEFLATED) as z:
                for folder_name, folder_path in (
                    ("sessions", SESSIONS),
                    ("uploads", UPLOADS),
                    ("chroma", CHROMA),
                    ("versions", VERSIONS),
                ):
                    if folder_path.exists():
                        for p in folder_path.rglob("*"):
                            if p.is_file():
                                rel = Path(folder_name) / p.relative_to(folder_path)
                                z.write(p, rel.as_posix())
                if LIBRARY.exists():
                    z.write(LIBRARY, "library.json")
                if SETTINGS.exists():
                    z.write(SETTINGS, "settings.json")
                manifest = {
                    "app": "RAG Studio",
                    "version": "1.0",
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                }
                z.writestr("backup_manifest.json", json.dumps(manifest, indent=2))
            tmp_target.replace(target)
        finally:
            if tmp_target.exists():
                tmp_target.unlink(missing_ok=True)
        return FileResponse(
            target, media_type="application/zip", filename="rag-studio-backup.zip"
        )


def _safe_extract_zip(zf: zipfile.ZipFile, dest: Path) -> list[str]:
    """Validate and extract archive entries into dest, blocking path traversal and symlinks."""
    dest = dest.resolve()
    extracted = []
    allowed_roots = {
        "sessions", "uploads", "chroma", "versions",
        "library.json", "settings.json", "backup_manifest.json", "data",
    }
    for info in zf.infolist():
        raw_name = info.filename.replace("\\", "/")
        if not raw_name or raw_name.startswith("/") or ":" in raw_name:
            raise ValueError(f"Unsafe archive entry path: {info.filename}")
        norm_parts = [part for part in raw_name.split("/") if part and part != "."]
        if not norm_parts or any(part == ".." for part in norm_parts):
            raise ValueError(f"Path traversal detected in archive entry: {info.filename}")
        if norm_parts[0] == "data" and len(norm_parts) > 1:
            norm_parts = norm_parts[1:]
        if norm_parts[0] not in allowed_roots:
            raise ValueError(f"Unexpected top-level archive item: {norm_parts[0]}")
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError(f"Symbolic links are not permitted in backup archives: {info.filename}")
        rel_path = Path(*norm_parts)
        target_path = (dest / rel_path).resolve()
        if dest != target_path and dest not in target_path.parents:
            raise ValueError(f"Archive entry escapes staging directory: {info.filename}")
        if info.is_dir() or raw_name.endswith("/"):
            target_path.mkdir(parents=True, exist_ok=True)
        else:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src_fh, open(target_path, "wb") as dst_fh:
                shutil.copyfileobj(src_fh, dst_fh)
            extracted.append(rel_path.as_posix())
    if not extracted:
        raise ValueError("The backup archive contains no restorable files.")
    return extracted


def _replace_workspace_item(src: Path, dst: Path, is_dir: bool) -> None:
    if is_dir:
        if dst.exists():
            shutil.rmtree(dst)
        if src.exists():
            shutil.copytree(src, dst)
        else:
            dst.mkdir(parents=True, exist_ok=True)
    else:
        if dst.exists():
            dst.unlink()
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


@app.post("/api/restore")
def restore_backup(file: UploadFile = File(...)):
    raw_bytes = file.file.read()
    if not raw_bytes:
        raise HTTPException(400, "Uploaded backup archive is empty.")
    if len(raw_bytes) > 500 * 1024 * 1024:
        raise HTTPException(400, "Backup archive exceeds the 500 MB restore limit.")

    with WORKSPACE_LOCK:
        with tempfile.TemporaryDirectory(prefix="rag_restore_") as tmp_root_str:
            tmp_root = Path(tmp_root_str)
            stage_dir = tmp_root / "stage"
            rollback_dir = tmp_root / "rollback"
            stage_dir.mkdir(parents=True, exist_ok=True)
            rollback_dir.mkdir(parents=True, exist_ok=True)

            try:
                with zipfile.ZipFile(io.BytesIO(raw_bytes), "r") as zf:
                    extracted = _safe_extract_zip(zf, stage_dir)
            except zipfile.BadZipFile:
                raise HTTPException(400, "Uploaded file is not a valid ZIP archive.")
            except ValueError as exc:
                raise HTTPException(400, str(exc))

            for json_name in ("library.json", "settings.json"):
                staged_json = stage_dir / json_name
                if staged_json.exists():
                    try:
                        parsed_obj = json.loads(staged_json.read_text(encoding="utf-8"))
                        if not isinstance(parsed_obj, dict):
                            raise ValueError(f"{json_name} must be a JSON object.")
                    except Exception as exc:
                        raise HTTPException(400, f"Backup contains invalid {json_name}: {exc}")

            staged_sessions = stage_dir / "sessions"
            if staged_sessions.exists():
                for sp in staged_sessions.glob("*.json"):
                    try:
                        s_obj = json.loads(sp.read_text(encoding="utf-8"))
                        if not isinstance(s_obj, dict) or "id" not in s_obj:
                            raise ValueError(f"Invalid session file: {sp.name}")
                    except Exception as exc:
                        raise HTTPException(400, f"Backup contains corrupted session JSON ({sp.name}): {exc}")

            managed_items = (
                ("sessions", SESSIONS, True),
                ("uploads", UPLOADS, True),
                ("chroma", CHROMA, True),
                ("versions", VERSIONS, True),
                ("library.json", LIBRARY, False),
                ("settings.json", SETTINGS, False),
            )
            for name, live_path, is_dir in managed_items:
                rb_target = rollback_dir / name
                if live_path.exists():
                    if is_dir:
                        shutil.copytree(live_path, rb_target)
                    else:
                        shutil.copy2(live_path, rb_target)

            if hasattr(rag, "close_store"):
                rag.close_store()

            try:
                for name, live_path, is_dir in managed_items:
                    _replace_workspace_item(stage_dir / name, live_path, is_dir)

                if hasattr(rag, "reopen_store"):
                    rag.reopen_store()
            except Exception as exc:
                logger.exception("Restore failed during live swap/reopen; rolling back workspace")
                try:
                    if hasattr(rag, "close_store"):
                        rag.close_store()
                    for name, live_path, is_dir in managed_items:
                        _replace_workspace_item(rollback_dir / name, live_path, is_dir)
                    if hasattr(rag, "reopen_store"):
                        rag.reopen_store()
                except Exception as rb_exc:
                    logger.critical("Workspace rollback after failed restore encountered an error: %s", rb_exc)
                raise HTTPException(
                    500,
                    f"Restore failed and previous workspace state was rolled back: {exc}",
                )

            docs_info = documents()
            sess_info = sessions()
            return {
                "restored": True,
                "files_restored": len(extracted),
                "documents_count": docs_info["count"],
                "sessions_count": len(sess_info["sessions"]),
                "preferences": read_preferences(),
                "message": "Workspace backup restored successfully.",
            }
