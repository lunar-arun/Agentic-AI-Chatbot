"""
Lightweight fakes for `ollama` and `chromadb` so the logic in rag.py and
agents.py can be unit-tested without a running Ollama server or the real
libraries.  Import this module BEFORE importing rag / agents.
"""

import os
import sys
import types
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ---------------------------------------------------------------- ollama ---
class FakeOllama(types.ModuleType):
    def __init__(self):
        super().__init__("ollama")
        self.chat_replies = []   # scripted replies (str) for non-stream chat
        self.chat_calls = []
        self.embed_calls = []
        self.fail_embed = False
        self.dim = 4

    def reset(self):
        self.chat_replies.clear()
        self.chat_calls.clear()
        self.embed_calls.clear()
        self.fail_embed = False

    def chat(self, model, messages, stream=False):
        self.chat_calls.append(messages)
        if stream:
            return iter(self.chat_replies.pop(0))
        reply = self.chat_replies.pop(0)
        return SimpleNamespace(message=SimpleNamespace(content=reply))

    def embed(self, model, input):
        if self.fail_embed:
            raise ConnectionError("ollama down")
        items = [input] if isinstance(input, str) else list(input)
        self.embed_calls.append(items)
        vecs = [[float(len(t) % 7 + 1)] * self.dim for t in items]
        return SimpleNamespace(embeddings=vecs)


# -------------------------------------------------------------- chromadb ---
class FakeCollection:
    def __init__(self):
        self.docs = {}

    def add(self, ids, embeddings, documents, metadatas):
        for i, e, d, m in zip(ids, embeddings, documents, metadatas):
            assert i not in self.docs, f"duplicate id {i}"
            self.docs[i] = (e, d, m)

    def count(self):
        return len(self.docs)

    def query(self, query_embeddings, n_results, include):
        items = list(self.docs.values())[:n_results]
        return {
            "documents": [[d for _, d, _ in items]],
            "metadatas": [[m for _, _, m in items]],
            "distances": [[0.1 * (k + 1) for k in range(len(items))]],
        }


class FakeClient:
    _stores = {}   # path -> {name: FakeCollection}  (simulates persistence)

    def __init__(self, path):
        self.store = FakeClient._stores.setdefault(path, {})

    def delete_collection(self, name):
        if name not in self.store:
            raise ValueError("no such collection")
        del self.store[name]

    def get_or_create_collection(self, name, metadata=None):
        return self.store.setdefault(name, FakeCollection())

    def get_collection(self, name):
        if name not in self.store:
            raise ValueError("no such collection")
        return self.store[name]


def install():
    fake_ollama = FakeOllama()
    fake_chroma = types.ModuleType("chromadb")
    fake_chroma.PersistentClient = FakeClient
    sys.modules["ollama"] = fake_ollama
    sys.modules["chromadb"] = fake_chroma
    for name in ("rag", "agents"):
        sys.modules.pop(name, None)
    FakeClient._stores.clear()
    return fake_ollama
