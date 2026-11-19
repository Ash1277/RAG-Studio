"""Local persistence, deduplication, deletion safety, rollback, and session source isolation tests."""
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _test_shims import install_fastapi_shims
install_fastapi_shims()
from starlette.datastructures import UploadFile

import rag_core as stub
import api


class DocumentLibraryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        api.UPLOADS = root / "uploads"
        api.SESSIONS = root / "sessions"
        api.LIBRARY = root / "library.json"
        api.UPLOADS.mkdir()
        api.SESSIONS.mkdir()
        self._sources = []
        self.original_list_sources = stub.list_sources
        self.original_load_file = stub.load_file
        self.original_index_docs = stub.index_docs
        self.original_remove_doc_id = stub.remove_doc_id
        self.calls = {"index": 0, "removed": []}

        stub.list_sources = lambda: list(self._sources)

        def load_file(path, name, did):
            self._pending = {"id": did, "name": name, "chunks": 2}
            return [object()]

        def index_docs(docs):
            self.calls["index"] += 1
            self._sources.append(self._pending)
            return {"chunks": 2, "pages": 1}

        def remove_doc_id(did):
            self.calls["removed"].append(did)
            self._sources = [s for s in self._sources if s.get("id") != did]
            return 2

        stub.load_file = load_file
        stub.index_docs = index_docs
        stub.remove_doc_id = remove_doc_id

    def tearDown(self):
        stub.list_sources = self.original_list_sources
        stub.load_file = self.original_load_file
        stub.index_docs = self.original_index_docs
        stub.remove_doc_id = self.original_remove_doc_id
        self.temp.cleanup()

    def test_identical_file_is_indexed_once_and_registry_persists(self):
        content = b"unique college notes for content hash test"
        first = api.upload([UploadFile(filename="notes.txt", file=io.BytesIO(content))])
        second = api.upload([UploadFile(filename="renamed.txt", file=io.BytesIO(content))])
        self.assertEqual(first["results"][0]["status"], "indexed")
        self.assertEqual(second["results"][0]["status"], "duplicate")
        self.assertEqual(self.calls["index"], 1)
        saved = json.loads(api.LIBRARY.read_text(encoding="utf-8"))
        self.assertEqual(len(saved["files"]), 1)

    def test_url_normalization_ignores_case_and_trailing_slash(self):
        self.assertEqual(api.normalize_url("HTTPS://Example.com/page/"), "https://example.com/page")

    def test_document_endpoint_keeps_registry_sources_visible(self):
        api.LIBRARY.write_text(
            json.dumps({"files": {"hash": {"id": "persisted-id", "name": "Old lecture.pdf", "chunks": 4}}, "urls": {}}),
            encoding="utf-8",
        )
        self._sources = []
        result = api.documents()
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["documents"][0]["name"], "Old lecture.pdf")
        self.assertFalse(result["documents"][0]["indexed"])

    def test_delete_document_preserves_file_and_metadata_when_chroma_delete_fails(self):
        content = b"important document that must not orphan"
        res = api.upload([UploadFile(filename="keepme.txt", file=io.BytesIO(content))])
        did = res["results"][0]["id"]

        def failing_remove(doc_id):
            raise RuntimeError("Chroma SQLite database locked")

        stub.remove_doc_id = failing_remove
        with self.assertRaises(Exception):
            api.delete_document(did)

        uploaded_files = list(api.UPLOADS.glob(f"{did}__*"))
        self.assertEqual(len(uploaded_files), 1)
        saved = json.loads(api.LIBRARY.read_text(encoding="utf-8"))
        self.assertEqual(len(saved["files"]), 1)

    def test_upload_error_after_index_rolls_back_partially_indexed_vectors(self):
        def zero_chunk_index(docs):
            return {"chunks": 0, "pages": 1}

        stub.index_docs = zero_chunk_index
        res = api.upload([UploadFile(filename="empty_extract.pdf", file=io.BytesIO(b"%PDF-1.4 dummy"))])
        self.assertEqual(res["results"][0]["status"], "error")
        did = res["results"][0]["id"]
        self.assertIn(did, self.calls["removed"])
        self.assertEqual(len(list(api.UPLOADS.glob(f"{did}__*"))), 0)


class SessionSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        api.UPLOADS = root / "uploads"
        api.SESSIONS = root / "sessions"
        api.LIBRARY = root / "library.json"
        api.UPLOADS.mkdir()
        api.SESSIONS.mkdir()
        self.old_list_sources = stub.list_sources
        self.old_answer = stub.answer_question
        self._sources = [
            {"id": "doc-a", "name": "A.pdf", "chunks": 3},
            {"id": "doc-b", "name": "B.pdf", "chunks": 4},
        ]
        stub.list_sources = lambda: list(self._sources)
        self.calls = []
        stub.answer_question = lambda question, history, rerank, top_k, source_ids: (
            self.calls.append(list(source_ids)) or "answer",
            [],
        )

    def tearDown(self):
        stub.list_sources = self.old_list_sources
        stub.answer_question = self.old_answer
        self.temp.cleanup()

    def test_chat_saves_and_reuses_session_source_selection(self):
        req = api.ChatRequest(question="Explain this", session_id="session-a", source_ids=["doc-a"])
        result = api.chat(req)
        self.assertEqual(result["session"]["source_ids"], ["doc-a"])
        self.assertEqual(self.calls, [["doc-a"]])
        api.chat(api.ChatRequest(question="Continue", session_id="session-a"))
        self.assertEqual(self.calls[-1], ["doc-a"])

    def test_per_session_source_isolation_between_conversations(self):
        api.chat(api.ChatRequest(question="Q1", session_id="session-1", source_ids=["doc-a"]))
        api.chat(api.ChatRequest(question="Q2", session_id="session-2", source_ids=["doc-b"]))
        s1 = api.get_session("session-1")
        s2 = api.get_session("session-2")
        self.assertEqual(s1["source_ids"], ["doc-a"])
        self.assertEqual(s2["source_ids"], ["doc-b"])

    def test_new_session_requires_explicit_source_selection(self):
        with self.assertRaises(Exception):
            api.chat(api.ChatRequest(question="Explain this", session_id="session-empty", source_ids=[]))

    def test_chat_rejects_unindexed_source_id(self):
        with self.assertRaises(Exception):
            api.chat(api.ChatRequest(question="Explain", session_id="session-x", source_ids=["non-existent"]))


class SessionSourceEndpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        api.SESSIONS = root / "sessions"
        api.SESSIONS.mkdir()
        self.old_list_sources = stub.list_sources
        stub.list_sources = lambda: [{"id": "doc-1", "name": "one.pdf", "chunks": 2}]

    def tearDown(self):
        stub.list_sources = self.old_list_sources
        self.temp.cleanup()

    def test_source_selection_endpoint_persists_selection_without_chat_message(self):
        result = api.update_session_sources("session-b", api.SourceSelectionRequest(source_ids=["doc-1"]))
        self.assertEqual(result["source_ids"], ["doc-1"])
        saved = api.get_session("session-b")
        self.assertEqual(saved["source_ids"], ["doc-1"])
        self.assertEqual(saved["messages"], [])


if __name__ == "__main__":
    unittest.main()
