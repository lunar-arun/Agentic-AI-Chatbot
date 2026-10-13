import os
import shutil
import tempfile
import unittest

import _stubs


class ChunkingTests(unittest.TestCase):
    def setUp(self):
        self.fake = _stubs.install()
        import rag
        self.rag = rag

    def test_short_doc_is_single_chunk_keeping_structure(self):
        text = "# Title\n\n## A\n\nPara one.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
        chunks = self.rag._chunk_text(text, "x.md")
        self.assertEqual(len(chunks), 1)
        self.assertIn("\n| a | b |\n|---|---|\n| 1 | 2 |", chunks[0]["text"])
        self.assertIn("## A", chunks[0]["text"])
        self.assertEqual(chunks[0]["source"], "x.md")
        self.assertEqual(chunks[0]["chunk_index"], 0)

    def test_long_doc_splits_and_later_chunks_get_section_prefix(self):
        rag = self.rag
        para = " ".join(["word"] * 150)
        text = "# Doc\n\n## Section One\n\n" + "\n\n".join([para] * 6)
        chunks = rag._chunk_text(text, "long.md")
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(
                len(c["text"].split()), rag.CHUNK_SIZE_WORDS + 20
            )
        self.assertTrue(chunks[1]["text"].startswith("[Section: Doc > Section One]"))
        self.assertEqual([c["chunk_index"] for c in chunks], list(range(len(chunks))))

    def test_oversized_block_is_windowed(self):
        rag = self.rag
        text = " ".join(f"w{i}" for i in range(1000))
        chunks = rag._chunk_text(text, "big.md")
        self.assertGreaterEqual(len(chunks), 3)
        # overlap: last words of chunk 0 reappear at start of chunk 1
        tail = chunks[0]["text"].split()[-rag.CHUNK_OVERLAP_WORDS:]
        head = chunks[1]["text"].split()[:rag.CHUNK_OVERLAP_WORDS]
        self.assertEqual(tail, head)

    def test_empty_and_crlf(self):
        self.assertEqual(self.rag._chunk_text("", "e.md"), [])
        self.assertEqual(self.rag._chunk_text("  \n\n  ", "e.md"), [])
        c = self.rag._chunk_text("# T\r\n\r\nline1\r\nline2\r\n", "w.md")
        self.assertNotIn("\r", c[0]["text"])


class InitAndRetrieveTests(unittest.TestCase):
    def setUp(self):
        self.fake = _stubs.install()
        self.tmp = tempfile.mkdtemp()
        import rag
        self.rag = rag
        rag.KNOWLEDGE_BASE_DIR = os.path.join(self.tmp, "kb")
        rag.CHROMA_DB_DIR = os.path.join(self.tmp, "chroma")
        rag._MANIFEST_PATH = os.path.join(rag.CHROMA_DB_DIR, "_file_manifest.json")
        rag._client = None
        os.makedirs(rag.KNOWLEDGE_BASE_DIR)
        self._write("a.md", "# A\n\nAlpha policy text.\n")
        self._write("b.txt", "Beta text\n")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, name, text):
        with open(os.path.join(self.rag.KNOWLEDGE_BASE_DIR, name), "w",
                  encoding="utf-8") as f:
            f.write(text)

    def test_no_files(self):
        os.remove(os.path.join(self.rag.KNOWLEDGE_BASE_DIR, "a.md"))
        os.remove(os.path.join(self.rag.KNOWLEDGE_BASE_DIR, "b.txt"))
        ok, msg = self.rag.initialize_knowledge_base()
        self.assertFalse(ok)
        self.assertIn("No .txt or .md", msg)

    def test_builds_then_is_up_to_date_then_rebuilds_on_change(self):
        ok, msg = self.rag.initialize_knowledge_base()
        self.assertTrue(ok)
        self.assertIn("Embedded 2 chunk(s)", msg)
        calls = len(self.fake.embed_calls)

        ok, msg = self.rag.initialize_knowledge_base()
        self.assertTrue(ok)
        self.assertIn("up-to-date", msg)
        self.assertEqual(len(self.fake.embed_calls), calls)  # nothing re-embedded

        self._write("b.txt", "Beta text changed\n")
        ok, msg = self.rag.initialize_knowledge_base()
        self.assertIn("Embedded 2 chunk(s)", msg)

    def test_rebuilds_when_index_missing_but_manifest_present(self):
        """Regression: manifest said 'up-to-date' while the index was gone."""
        self.rag.initialize_knowledge_base()
        _stubs.FakeClient._stores.clear()
        self.rag._client = None
        ok, msg = self.rag.initialize_knowledge_base()
        self.assertIn("Embedded", msg)
        self.assertEqual(len(self.rag.retrieve("anything")), 2)

    def test_failed_embedding_keeps_old_index_and_manifest(self):
        """Regression: old collection was deleted before embedding."""
        self.rag.initialize_knowledge_base()
        self._write("a.md", "# A\n\nChanged.\n")
        self.fake.fail_embed = True
        with self.assertRaises(ConnectionError):
            self.rag.initialize_knowledge_base()
        self.fake.fail_embed = False
        self.assertEqual(len(self.rag.retrieve("alpha")), 2)  # old index intact
        ok, msg = self.rag.initialize_knowledge_base()        # retried, not skipped
        self.assertIn("Embedded", msg)

    def test_config_change_triggers_rebuild(self):
        self.rag.initialize_knowledge_base()
        self.rag.EMBEDDING_MODEL = "other-model"
        ok, msg = self.rag.initialize_knowledge_base()
        self.assertIn("Embedded", msg)

    def test_bom_file_is_read_cleanly(self):
        with open(os.path.join(self.rag.KNOWLEDGE_BASE_DIR, "c.md"), "w",
                  encoding="utf-8-sig") as f:
            f.write("# C\n\nBom text\n")
        text = self.rag._read_file(
            os.path.join(self.rag.KNOWLEDGE_BASE_DIR, "c.md"))
        self.assertTrue(text.startswith("# C"))

    def test_retrieve_scores_and_edge_cases(self):
        self.assertEqual(self.rag.retrieve("x"), [])       # no index yet
        self.rag.initialize_knowledge_base()
        self.assertEqual(self.rag.retrieve("   "), [])     # blank query
        res = self.rag.retrieve("alpha", top_k=10)         # top_k > count
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0]["score"], 0.9)             # 1 - 0.1
        self.assertGreaterEqual(res[0]["score"], res[1]["score"])
        self.assertIn(res[0]["source"], {"a.md", "b.txt"})


if __name__ == "__main__":
    unittest.main()
