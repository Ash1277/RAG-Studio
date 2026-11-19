"""Comprehensive tests for app_paths migration, backup/restore rollback, Chroma lifecycle, and multi-doc research."""
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _test_shims import install_fastapi_shims
install_fastapi_shims()
from starlette.datastructures import UploadFile

import app_paths
import rag_core
import api
import storage


class AppPathsMigrationTests(unittest.TestCase):
    def test_env_override_rag_studio_data_dir_and_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            custom = Path(tmp) / "custom_data"
            resolved = app_paths.resolve_data_dir(env={"RAG_STUDIO_DATA_DIR": str(custom)}, frozen=True)
            self.assertEqual(resolved, custom.resolve())
            resolved_home = app_paths.resolve_data_dir(env={"RAG_STUDIO_HOME": str(custom)}, frozen=False)
            self.assertEqual(resolved_home, custom.resolve())

    def test_windows_legacy_data_migrated_by_copy_without_deleting_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            appdata = Path(tmp) / "LocalAppData"
            legacy = appdata / "RAGDocumentAssistant" / "data"
            (legacy / "uploads").mkdir(parents=True)
            (legacy / "sessions").mkdir(parents=True)
            (legacy / "uploads" / "doc1__test.pdf").write_bytes(b"pdf-bytes")
            (legacy / "library.json").write_text('{"files":{"h":{"id":"doc1"}},"urls":{}}', encoding="utf-8")

            target = app_paths.resolve_data_dir(
                env={"LOCALAPPDATA": str(appdata)},
                platform="win32",
                frozen=True,
                home=Path(tmp),
            )
            self.assertEqual(target, appdata / "RAG Studio" / "data")
            self.assertTrue((target / "uploads" / "doc1__test.pdf").exists())
            self.assertTrue((target / "library.json").exists())
            # Legacy directory must remain untouched
            self.assertTrue((legacy / "uploads" / "doc1__test.pdf").exists())
            self.assertTrue((legacy / "library.json").exists())

    def test_macos_and_linux_frozen_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            mac_dir = app_paths.resolve_data_dir(env={}, platform="darwin", frozen=True, home=home)
            self.assertEqual(mac_dir, home / "Library" / "Application Support" / "RAG Studio")
            linux_dir = app_paths.resolve_data_dir(env={}, platform="linux", frozen=True, home=home)
            self.assertEqual(linux_dir, home / ".local" / "share" / "rag-studio")


class BackupRestoreAndRollbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        api.DATA = root
        api.UPLOADS = root / "uploads"
        api.SESSIONS = root / "sessions"
        api.CHROMA = root / "chroma"
        api.LIBRARY = root / "library.json"
        for p in (api.UPLOADS, api.SESSIONS, api.CHROMA):
            p.mkdir(parents=True, exist_ok=True)
        # Seed live workspace
        (api.UPLOADS / "live1__notes.txt").write_text("live upload", encoding="utf-8")
        (api.SESSIONS / "sess-live.json").write_text(
            json.dumps({"id": "sess-live", "title": "Live Chat", "messages": [], "source_ids": ["live1"]}),
            encoding="utf-8",
        )
        (api.CHROMA / "chroma.sqlite3").write_bytes(b"live-chroma-db")
        api.LIBRARY.write_text(
            json.dumps({"files": {"h1": {"id": "live1", "name": "notes.txt", "chunks": 2}}, "urls": {}}),
            encoding="utf-8",
        )
        self.old_close = getattr(rag_core, "close_store", None)
        self.old_reopen = getattr(rag_core, "reopen_store", None)
        self.old_list = rag_core.list_sources
        self.lifecycle = []
        rag_core.close_store = lambda: self.lifecycle.append("close")
        rag_core.reopen_store = lambda: self.lifecycle.append("reopen")
        rag_core.list_sources = lambda: [{"id": "live1", "name": "notes.txt", "chunks": 2}]

    def tearDown(self):
        if self.old_close is not None:
            rag_core.close_store = self.old_close
        if self.old_reopen is not None:
            rag_core.reopen_store = self.old_reopen
        rag_core.list_sources = self.old_list
        self.temp.cleanup()

    def test_backup_creates_complete_zip_and_restore_succeeds(self):
        resp = api.backup()
        backup_zip = Path(resp.path)
        self.assertTrue(backup_zip.exists())
        raw_zip = backup_zip.read_bytes()

        # Mutate live state before restoring
        (api.UPLOADS / "live1__notes.txt").unlink()
        api.LIBRARY.write_text('{"files":{},"urls":{}}', encoding="utf-8")

        res = api.restore_backup(UploadFile(filename="backup.zip", file=io.BytesIO(raw_zip)))
        self.assertTrue(res["restored"])
        self.assertTrue((api.UPLOADS / "live1__notes.txt").exists())
        self.assertIn("close", self.lifecycle)
        self.assertIn("reopen", self.lifecycle)

    def test_restore_blocks_zip_path_traversal(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("../evil.txt", "pwned")
        buf.seek(0)
        with self.assertRaises(Exception) as ctx:
            api.restore_backup(UploadFile(filename="evil.zip", file=buf))
        self.assertIn("traversal", str(ctx.exception).lower())
        # Live workspace remains untouched
        self.assertTrue((api.UPLOADS / "live1__notes.txt").exists())

    def test_restore_rolls_back_when_chroma_reopen_fails(self):
        # Create valid replacement archive
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("library.json", json.dumps({"files": {"h2": {"id": "newdoc", "name": "new.pdf", "chunks": 1}}, "urls": {}}))
            zf.writestr("uploads/newdoc__new.pdf", "new content")
            zf.writestr("sessions/sess-new.json", json.dumps({"id": "sess-new", "title": "New Chat", "messages": []}))
            zf.writestr("chroma/chroma.sqlite3", "corrupt-sqlite-bytes")
        buf.seek(0)

        def fail_once_reopen():
            self.lifecycle.append("reopen_attempt")
            if self.lifecycle.count("reopen_attempt") == 1:
                raise RuntimeError("Simulated Chroma SQLite corruption on reopen")

        rag_core.reopen_store = fail_once_reopen
        with self.assertRaises(Exception) as ctx:
            api.restore_backup(UploadFile(filename="bad_chroma.zip", file=buf))
        self.assertIn("rolled back", str(ctx.exception).lower())

        # Verify rollback restored original live files, sessions, chroma, and library.json
        self.assertTrue((api.UPLOADS / "live1__notes.txt").exists())
        self.assertFalse((api.UPLOADS / "newdoc__new.pdf").exists())
        self.assertTrue((api.SESSIONS / "sess-live.json").exists())
        self.assertEqual((api.CHROMA / "chroma.sqlite3").read_bytes(), b"live-chroma-db")
        restored_lib = json.loads(api.LIBRARY.read_text(encoding="utf-8"))
        self.assertIn("h1", restored_lib["files"])


class ResearchComparisonCoverageTests(unittest.TestCase):
    def setUp(self):
        self.old_list = rag_core.list_sources
        self.old_retrieve = rag_core.retrieve
        self.old_answer = rag_core.answer_with_context
        rag_core.list_sources = lambda: [
            {"id": "doc-1", "name": "Alpha.pdf", "chunks": 5},
            {"id": "doc-2", "name": "Beta.pdf", "chunks": 5},
            {"id": "doc-3", "name": "Gamma.pdf", "chunks": 5},
        ]

    def tearDown(self):
        rag_core.list_sources = self.old_list
        rag_core.retrieve = self.old_retrieve
        rag_core.answer_with_context = self.old_answer

    def test_multi_document_comparison_retrieves_per_source_and_reports_coverage(self):
        per_source_calls = []
        def fake_retrieve(question, top_k=None, use_reranker=False, source_ids=None):
            per_source_calls.append(list(source_ids or []))
            sid = source_ids[0]
            if sid == "doc-3":
                return []  # Simulate zero matching passages in Gamma.pdf
            return [{"source": f"{sid}.pdf — page 1", "source_id": f"{sid}.pdf", "doc_id": sid, "content": f"Content from {sid}"}]

        captured_prompt = {}
        def fake_answer(instruction, question, contexts, history=None, max_tokens=1600):
            captured_prompt["instruction"] = instruction
            captured_prompt["question"] = question
            captured_prompt["contexts"] = contexts
            return "## Overview & Coverage\nCompared Alpha, Beta, Gamma."

        rag_core.retrieve = fake_retrieve
        rag_core.answer_with_context = fake_answer

        res = api.research(
            api.ResearchRequest(
                task="Compare methodology across selected documents",
                sources=["doc-1", "doc-2", "doc-3"],
                use_reranker=False,
            )
        )
        # Must have queried each source separately rather than a single pooled call
        self.assertEqual(per_source_calls, [["doc-1"], ["doc-2"], ["doc-3"]])
        self.assertEqual(len(res["coverage"]), 3)
        cov_by_id = {c["id"]: c for c in res["coverage"]}
        self.assertTrue(cov_by_id["doc-1"]["covered"])
        self.assertTrue(cov_by_id["doc-2"]["covered"])
        self.assertFalse(cov_by_id["doc-3"]["covered"])
        self.assertIn("Gamma.pdf", captured_prompt["question"])


class RagCoreInternalsTests(unittest.TestCase):
    def test_build_source_filter_single_and_multi(self):
        self.assertIsNone(rag_core._build_source_filter(None))
        self.assertIsNone(rag_core._build_source_filter([]))
        self.assertEqual(rag_core._build_source_filter(["doc-1"]), {"doc_id": "doc-1"})
        self.assertEqual(
            rag_core._build_source_filter(["doc-1", "doc-2", "doc-1"]),
            {"doc_id": {"$in": ["doc-1", "doc-2"]}},
        )

    def test_remove_doc_id_propagates_store_error(self):
        old_store = rag_core._store
        class BrokenStore:
            def get(self, **kwargs):
                raise RuntimeError("Chroma backend disconnected")
        rag_core._store = BrokenStore()
        try:
            with self.assertRaisesRegex(RuntimeError, "Chroma backend disconnected"):
                rag_core.remove_doc_id("doc-1")
        finally:
            rag_core._store = old_store

    def test_close_store_stops_client_system_and_clears_reference(self):
        old_store = rag_core._store
        stopped = []
        class DummySystem:
            def stop(self):
                stopped.append("stopped")
        class DummyClient:
            _system = DummySystem()
            def clear_system_cache(self):
                stopped.append("cache_cleared")
        class DummyStore:
            _client = DummyClient()
        rag_core._store = DummyStore()
        rag_core.close_store()
        self.assertIsNone(rag_core._store)
        self.assertEqual(stopped, ["stopped", "cache_cleared"])
        rag_core._store = old_store


class FrontendIntegrityTests(unittest.TestCase):
    def test_html_contains_all_required_ids_and_scroll_region(self):
        root = Path(__file__).resolve().parent.parent
        html = (root / "static" / "index.html").read_text(encoding="utf-8")
        js = (root / "static" / "app.js").read_text(encoding="utf-8")
        css = (root / "static" / "styles.css").read_text(encoding="utf-8")
        required_ids = [
            "chatScrollRegion",
            "welcome",
            "chatMessages",
            "questionInput",
            "sendButton",
            "selectAllSources",
            "clearSourceSelection",
            "groqApiKey",
            "saveGroqKey",
            "removeGroqKey",
            "keyStatus",
            "restoreBackupBtn",
            "restoreFileInput",
            "compareDocs",
            "compareDocuments",
            "studyKind",
            "quizSource",
        ]
        for rid in required_ids:
            self.assertIn(f'id="{rid}"', html, f"Missing ID in index.html: {rid}")
        self.assertIn(".chat-scroll-region", css)
        self.assertIn(".result-coverage", css)
        self.assertIn("/api/restore", js)
        self.assertIn("/api/sessions/", js)




class AdditionalEdgeCaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        api.DATA = root
        api.UPLOADS = root / "uploads"
        api.SESSIONS = root / "sessions"
        api.CHROMA = root / "chroma"
        api.LIBRARY = root / "library.json"
        for p in (api.UPLOADS, api.SESSIONS, api.CHROMA):
            p.mkdir(parents=True, exist_ok=True)
        self.old_list = rag_core.list_sources
        self.old_load_url = rag_core.load_url
        self.old_index = rag_core.index_docs
        self.old_remove = rag_core.remove_doc_id
        self.removed_ids = []
        rag_core.list_sources = lambda: []
        rag_core.remove_doc_id = lambda did: self.removed_ids.append(did) or 1

    def tearDown(self):
        rag_core.list_sources = self.old_list
        rag_core.load_url = self.old_load_url
        rag_core.index_docs = self.old_index
        rag_core.remove_doc_id = self.old_remove
        self.temp.cleanup()

    def test_url_ingestion_error_rolls_back_partially_indexed_vectors(self):
        rag_core.load_url = lambda url, did: [object()]
        rag_core.index_docs = lambda docs: {"chunks": 0, "pages": 1}
        with self.assertRaises(Exception):
            api.add_url("https://example.com/empty")
        self.assertEqual(len(self.removed_ids), 1)

    def test_restore_rejects_corrupted_session_json_without_touching_live_data(self):
        (api.UPLOADS / "keep__file.txt").write_text("safe", encoding="utf-8")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("sessions/broken.json", "{not_valid_json")
        buf.seek(0)
        with self.assertRaises(Exception) as ctx:
            api.restore_backup(UploadFile(filename="corrupt.zip", file=buf))
        self.assertIn("corrupted session json", str(ctx.exception).lower())
        self.assertTrue((api.UPLOADS / "keep__file.txt").exists())

    def test_restore_rejects_symlink_entries_in_zip(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            info = zipfile.ZipInfo("uploads/symlink_attack.txt")
            info.external_attr = (0o120777 << 16)
            zf.writestr(info, "/etc/passwd")
        buf.seek(0)
        with self.assertRaises(Exception) as ctx:
            api.restore_backup(UploadFile(filename="symlink.zip", file=buf))
        self.assertIn("symbolic links", str(ctx.exception).lower())

    def test_atomic_json_write_leaves_no_temp_files_and_handles_invalid_read(self):
        target = api.DATA / "atomic_test.json"
        storage.write_json_atomic(target, {"hello": "world", "count": 42})
        self.assertEqual(storage.read_json(target, {}), {"hello": "world", "count": 42})
        tmp_files = list(api.DATA.glob("*.tmp"))
        self.assertEqual(len(tmp_files), 0)
        target.write_text("corrupted{{{", encoding="utf-8")
        self.assertEqual(storage.read_json(target, {"fallback": True}), {"fallback": True})

    def test_session_id_validation_blocks_path_traversal(self):
        with self.assertRaises(Exception):
            api.session_path("../escape")
        with self.assertRaises(Exception):
            api.session_path("bad/slash")

    def test_health_and_key_settings_never_return_secret_key(self):
        import os
        old_env = os.environ.get("GROQ_API_KEY")
        secret = "gsk_SUPER_SECRET_KEY_VALUE_1234567890"
        os.environ["GROQ_API_KEY"] = secret
        try:
            h = api.health()
            ks = api.key_status()
            self.assertTrue(h["groq_configured"])
            self.assertTrue(ks["configured"])
            self.assertNotIn(secret, json.dumps(h))
            self.assertNotIn(secret, json.dumps(ks))
        finally:
            if old_env is None:
                os.environ.pop("GROQ_API_KEY", None)
            else:
                os.environ["GROQ_API_KEY"] = old_env



    def test_ocr_pdf_fallback_recovers_text_when_pypdf_loader_returns_empty(self):
        old_loader = rag_core.PyPDFLoader
        old_ocr = rag_core._ocr_pdf_fallback
        class EmptyLoader:
            def __init__(self, path):
                self.path = path
            def load(self):
                return [rag_core.Document(page_content="   ", metadata={"page": 0})]
        rag_core.PyPDFLoader = EmptyLoader
        rag_core._ocr_pdf_fallback = lambda p: [rag_core.Document(page_content="Scanned OCR text recovered", metadata={"page": 0, "ocr": True})]
        try:
            docs = rag_core.load_file(Path("scanned.pdf"), "scanned.pdf", "doc-ocr-1")
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0].page_content, "Scanned OCR text recovered")
            self.assertEqual(docs[0].metadata["doc_id"], "doc-ocr-1")
            self.assertTrue(docs[0].metadata["ocr"])
        finally:
            rag_core.PyPDFLoader = old_loader
            rag_core._ocr_pdf_fallback = old_ocr

    def test_ocr_pdf_fallback_raises_clear_error_when_scanned_and_no_ocr_text(self):
        old_loader = rag_core.PyPDFLoader
        old_ocr = rag_core._ocr_pdf_fallback
        class EmptyLoader:
            def __init__(self, path):
                self.path = path
            def load(self):
                return []
        rag_core.PyPDFLoader = EmptyLoader
        rag_core._ocr_pdf_fallback = lambda p: []
        try:
            with self.assertRaisesRegex(ValueError, "scanned image PDF"):
                rag_core.load_file(Path("blank_scan.pdf"), "blank_scan.pdf", "doc-ocr-2")
        finally:
            rag_core.PyPDFLoader = old_loader
            rag_core._ocr_pdf_fallback = old_ocr



    def test_packaging_scripts_include_all_core_hidden_imports(self):
        root = Path(__file__).resolve().parent.parent
        if not (root / "packaging").exists():
            return
        required_modules = ["api", "rag_core", "app_paths", "paths", "security", "storage", "quiz_service"]
        for script_name in ("build_exe.bat", "build_linux.sh", "build_macos.sh", "build_unix.sh"):
            text = (root / "packaging" / script_name).read_text(encoding="utf-8")
            for mod in required_modules:
                self.assertIn(f"--hidden-import {mod}", text, f"{script_name} missing hidden-import {mod}")



    def test_themes_include_redesigned_palettes_and_preserve_absolute_dark(self):
        root = Path(__file__).resolve().parent.parent
        css = (root / "static" / "styles.css").read_text(encoding="utf-8")
        for theme in ("ocean", "forest", "lavender", "sand", "midnight", "ghost", "ghost-dark", "dark"):
            self.assertIn(f'body[data-theme="{theme}"]', css)
        # Verify Absolute Dark preserves its exact baseline palette
        self.assertIn('body[data-theme="dark"]{--green:#a6b8ff;--green-dark:#8499ef;--mint:#252b39;--ink:#f4f5fa;--line:#303541;--canvas:#090b10;--paper:#10131b;--muted:#c0c5d2}', css)



    def test_chat_scroll_chain_and_themed_buttons_enforced_in_css(self):
        root = Path(__file__).resolve().parent.parent
        css = (root / "static" / "styles.css").read_text(encoding="utf-8")
        html = (root / "static" / "index.html").read_text(encoding="utf-8")
        normalized_css = "".join(css.split())
        self.assertIn(".chat-messages:empty{display:none!important", normalized_css)
        self.assertIn(".chat-messages{flex:00auto!important", normalized_css)
        self.assertIn(".chat-scroll-region{flex:110%!important;min-height:0!important;overflow-y:auto!important", normalized_css)
        self.assertIn("grid-template-rows:minmax(0,1fr)", normalized_css)
        self.assertIn("body[data-theme] .icon-button", css)
        self.assertIn("body[data-theme] .outline-button", css)
        self.assertIn("body[data-theme] .text-button", css)
        self.assertIn('id="toggleDrawerBtn"', html)
        self.assertIn('id="selectedSourcesBadge"', html)



    def test_clean_model_text_strips_think_tags_and_handles_lists(self):
        raw = "<think>Internal reasoning about Samsung SSDs...</think>\nHere are the Samsung SSDs: 990 PRO [Source 1]."
        self.assertEqual(
            rag_core._clean_model_text(raw),
            "Here are the Samsung SSDs: 990 PRO [Source 1].",
        )
        list_raw = [{"type": "text", "text": "Samsung 990 PRO"}]
        self.assertEqual(rag_core._clean_model_text(list_raw), "Samsung 990 PRO")

    def test_answer_with_context_retries_when_model_returns_empty_content(self):
        old_llm = rag_core.llm
        old_prompt = rag_core.ChatPromptTemplate
        calls = []

        class FakeMsg:
            def __init__(self, content):
                self.content = content
                self.additional_kwargs = {}

        class FakeChain:
            def __init__(self, budget):
                self.budget = budget
            def invoke(self, payload):
                calls.append(self.budget)
                if len(calls) == 1:
                    return FakeMsg("   ")  # Simulate empty first response from reasoning token exhaustion
                return FakeMsg("Samsung 990 PRO and 9100 PRO [Source 1].")

        class FakePrompt:
            def __or__(self, other_budget):
                return FakeChain(other_budget)

        class FakePromptTemplate:
            @staticmethod
            def from_messages(msgs):
                return FakePrompt()

        rag_core.llm = lambda temperature=0.1, max_tokens=2400: max_tokens
        rag_core.ChatPromptTemplate = FakePromptTemplate
        try:
            ans = rag_core.answer_with_context(
                "System instruction",
                "list ssd made by samsung",
                [{"source": "specs.txt", "content": "Samsung 990 PRO"}],
            )
            self.assertEqual(ans, "Samsung 990 PRO and 9100 PRO [Source 1].")
            self.assertEqual(len(calls), 2)
            self.assertGreaterEqual(calls[1], 4096)
        finally:
            rag_core.llm = old_llm
            rag_core.ChatPromptTemplate = old_prompt




class Phase6To11AdvancedFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        api.DATA = root
        api.UPLOADS = root / "uploads"
        api.SESSIONS = root / "sessions"
        api.CHROMA = root / "chroma"
        api.VERSIONS = root / "versions"
        api.LIBRARY = root / "library.json"
        api.SETTINGS = root / "settings.json"
        for p in (api.UPLOADS, api.SESSIONS, api.CHROMA, api.VERSIONS):
            p.mkdir(parents=True, exist_ok=True)
        self.old_list = rag_core.list_sources
        self.old_load_file = rag_core.load_file
        self.old_index = rag_core.index_docs
        self.old_remove = rag_core.remove_doc_id
        self.old_retrieve = rag_core.retrieve
        self.old_answer_ctx = rag_core.answer_with_context
        self.old_close = getattr(rag_core, "close_store", None)
        self.old_reopen = getattr(rag_core, "reopen_store", None)
        rag_core.close_store = lambda: None
        rag_core.reopen_store = lambda: None
        self._sources = []
        rag_core.list_sources = lambda: list(self._sources)

        def fake_load_file(path, name, did):
            txt = Path(path).read_text(encoding="utf-8", errors="replace")
            self._pending = {"id": did, "name": name, "chunks": 2}
            return [rag_core.Document(page_content=txt, metadata={"source": name, "doc_id": did})]

        def fake_index_docs(docs):
            did = docs[0].metadata.get("doc_id")
            name = docs[0].metadata.get("source")
            self._sources = [s for s in self._sources if s.get("id") != did]
            self._sources.append({"id": did, "name": name, "chunks": 2})
            return {"chunks": 2, "pages": 1}

        def fake_remove(did):
            self._sources = [s for s in self._sources if s.get("id") != did]
            return 2

        def fake_clear_all():
            count = len(self._sources) * 2
            self._sources = []
            return count

        self.old_clear_all = getattr(rag_core, "clear_all_documents", None)
        rag_core.load_file = fake_load_file
        rag_core.index_docs = fake_index_docs
        rag_core.remove_doc_id = fake_remove
        rag_core.clear_all_documents = fake_clear_all

    def tearDown(self):
        rag_core.list_sources = self.old_list
        rag_core.load_file = self.old_load_file
        rag_core.index_docs = self.old_index
        rag_core.remove_doc_id = self.old_remove
        if self.old_clear_all is not None:
            rag_core.clear_all_documents = self.old_clear_all
        rag_core.retrieve = self.old_retrieve
        rag_core.answer_with_context = self.old_answer_ctx
        if self.old_close is not None:
            rag_core.close_store = self.old_close
        if self.old_reopen is not None:
            rag_core.reopen_store = self.old_reopen
        self.temp.cleanup()

    def test_document_version_control_tracks_updates_computes_diff_and_restores_older_version(self):
        v1 = api.upload([UploadFile(filename="syllabus.txt", file=io.BytesIO(b"Line 1: Intro\nLine 2: Midterm in Week 5"))])
        self.assertEqual(v1["results"][0]["status"], "indexed")
        self.assertEqual(v1["results"][0]["version"], 1)
        doc_id = v1["results"][0]["id"]

        # Upload modified content under the same filename -> should update to v2 and archive v1
        v2 = api.upload([UploadFile(filename="syllabus.txt", file=io.BytesIO(b"Line 1: Intro\nLine 2: Midterm in Week 7\nLine 3: Project"))])
        self.assertEqual(v2["results"][0]["status"], "updated")
        self.assertEqual(v2["results"][0]["id"], doc_id)
        self.assertEqual(v2["results"][0]["version"], 2)

        ver_info = api.get_document_versions(doc_id)
        self.assertEqual(ver_info["current"]["version"], 2)
        self.assertEqual(len(ver_info["archived_versions"]), 1)
        self.assertEqual(ver_info["archived_versions"][0]["version"], 1)
        self.assertIsNotNone(ver_info["latest_diff"])
        self.assertGreaterEqual(ver_info["latest_diff"]["lines_added"], 1)

        # Restore v1 -> becomes active v3 while archiving v2
        restored = api.restore_document_version(doc_id, 1)
        self.assertTrue(restored["restored"])
        self.assertEqual(restored["new_version"], 3)
        live_file = next(api.UPLOADS.glob(f"{doc_id}__*"))
        self.assertIn("Week 5", live_file.read_text(encoding="utf-8"))

    def test_session_rename_endpoint_updates_title(self):
        api.save_session({"id": "sess-ren", "title": "Old title", "messages": [], "source_ids": []})
        res = api.rename_session("sess-ren", api.SessionRenameRequest(title="Machine Learning Notes"))
        self.assertEqual(res["title"], "Machine Learning Notes")
        loaded = api.get_session("sess-ren")
        self.assertEqual(loaded["title"], "Machine Learning Notes")

    def test_study_notes_and_definitions_modes(self):
        rag_core.retrieve = lambda *a, **k: [{"source": "ML.pdf", "content": "Gradient descent optimizes parameters."}]
        captured = []
        rag_core.answer_with_context = lambda inst, q, docs, max_tokens=2000: (captured.append(inst) or "Study output")
        notes_res = api.study(api.StudyRequest(source="all", kind="notes", count=5))
        defs_res = api.study(api.StudyRequest(source="all", kind="definitions", count=5))
        self.assertEqual(notes_res["result"], "Study output")
        self.assertEqual(defs_res["result"], "Study output")
        self.assertIn("study note sections", captured[0])
        self.assertIn("glossary", captured[1])

    def test_tabular_csv_cell_level_comparison_summary(self):
        f1 = api.UPLOADS / "d1__q1.csv"
        f2 = api.UPLOADS / "d2__q2.csv"
        f1.write_text("Model,Capacity,Price\n990 PRO,1TB,109\n9100 PRO,1TB,179\n", encoding="utf-8")
        f2.write_text("Model,Capacity,Price\n990 PRO,1TB,99\n9100 PRO,2TB,279\n", encoding="utf-8")
        summary = rag_core.compute_tabular_comparison_summary([(f1, "q1.csv"), (f2, "q2.csv")])
        self.assertIn("Shared columns (3)", summary)
        self.assertIn("990 PRO", summary)
        self.assertIn("Price", summary)

    def test_verify_answer_grounding_high_and_low_cases(self):
        sources = [
            {"source": "ssd.txt", "content": "Samsung manufactures the 990 PRO NVMe SSD and the 9100 PRO PCIe 5.0 SSD."}
        ]
        good = "Samsung manufactures the 990 PRO NVMe SSD and the 9100 PRO PCIe 5.0 SSD [Source 1]."
        g_high = rag_core.verify_answer_grounding(good, sources)
        self.assertEqual(g_high["level"], "high")
        self.assertEqual(g_high["citations_used"], [1])

        bad = "Quantum photonic crystal drives were invented on Mars in 2099 without any citations."
        g_low = rag_core.verify_answer_grounding(bad, sources)
        self.assertEqual(g_low["level"], "low")
        self.assertIsNotNone(g_low["warning"])

    def test_preferences_persisted_and_included_in_backup_restore(self):
        api.update_preferences(api.PreferencesRequest(theme="midnight", default_top_k=8, use_reranker=True))
        self.assertEqual(api.get_preferences()["theme"], "midnight")
        resp = api.backup()
        raw_zip = Path(resp.path).read_bytes()
        api.update_preferences(api.PreferencesRequest(theme="ocean", default_top_k=5, use_reranker=False))
        res = api.restore_backup(UploadFile(filename="b.zip", file=io.BytesIO(raw_zip)))
        self.assertEqual(res["preferences"]["theme"], "midnight")

    def test_image_upload_ocr_pipeline(self):
        rag_core.load_file = self.old_load_file
        old_ocr_bytes = rag_core._ocr_image_bytes
        rag_core._ocr_image_bytes = lambda b: "Extracted diagram text: Neural Network Architecture"
        try:
            res = api.upload([UploadFile(filename="diagram.png", file=io.BytesIO(b"\x89PNG\r\n\x1a\nfakeimage"))])
            self.assertEqual(res["results"][0]["status"], "indexed")
        finally:
            rag_core._ocr_image_bytes = old_ocr_bytes



    def test_clear_all_documents_removes_vectors_files_versions_library_and_session_selections(self):
        api.upload([
            UploadFile(filename="a.txt", file=io.BytesIO(b"First document text")),
            UploadFile(filename="b.txt", file=io.BytesIO(b"Second document text")),
        ])
        docs_before = api.documents()["documents"]
        self.assertEqual(len(docs_before), 2)
        ids_before = [d["id"] for d in docs_before]
        api.save_session({"id": "sess-1", "title": "Chat", "messages": [], "source_ids": ids_before})

        res = api.clear_documents()
        self.assertTrue(res["cleared"])
        self.assertEqual(res["sources_removed"], 2)
        self.assertEqual(api.documents()["count"], 0)
        self.assertEqual(len(list(api.UPLOADS.glob("*"))), 0)
        saved_lib = json.loads(api.LIBRARY.read_text(encoding="utf-8"))
        self.assertEqual(saved_lib["files"], {})
        self.assertEqual(saved_lib["urls"], {})
        self.assertEqual(saved_lib["versions"], {})
        sess_after = api.get_session("sess-1")
        self.assertEqual(sess_after["source_ids"], [])

    def test_clear_all_documents_preserves_files_when_chroma_clear_fails(self):
        api.upload([UploadFile(filename="preserve.txt", file=io.BytesIO(b"Keep on failure"))])
        old_clear = getattr(rag_core, "clear_all_documents", None)
        def fail_clear():
            raise RuntimeError("Chroma lock error during clear_all")
        rag_core.clear_all_documents = fail_clear
        try:
            with self.assertRaises(Exception):
                api.clear_documents()
        finally:
            if old_clear is not None:
                rag_core.clear_all_documents = old_clear
            else:
                delattr(rag_core, "clear_all_documents")
        # Local file and library entry must remain intact
        self.assertEqual(len(list(api.UPLOADS.glob("*"))), 1)
        saved_lib = json.loads(api.LIBRARY.read_text(encoding="utf-8"))
        self.assertEqual(len(saved_lib["files"]), 1)



    def test_url_ssrf_protection_blocks_localhost_and_private_ips(self):
        for bad_url in (
            "http://localhost:8765/api/health",
            "http://127.0.0.1:8000/secret",
            "http://169.254.169.254/latest/meta-data/",
            "http://192.168.1.1/admin",
            "file:///etc/passwd",
        ):
            with self.assertRaises(Exception):
                api.add_url(bad_url)


if __name__ == "__main__":
    unittest.main()
