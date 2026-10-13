"""
rag.py — RAG pipeline: document loading, chunking, embedding, and retrieval.

Uses ChromaDB (file-based, COSINE distance explicitly set) for the vector
store and Ollama's nomic-embed-text model for local embeddings.
No cloud calls, no API keys.

Scoring note:
    ChromaDB with cosine space returns cosine DISTANCE (0 = identical, 2 = opposite).
    We convert to cosine SIMILARITY:  similarity = 1 - distance
    Scores are rounded to 2 decimal places.
"""

import os
import re
import hashlib
import json
import chromadb
import ollama

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_DIR = os.path.join(_BASE_DIR, "knowledge_base")
CHROMA_DB_DIR = os.path.join(_BASE_DIR, "chroma_db")
COLLECTION_NAME = "customer_service_kb"

# Embedding model served by Ollama.  nomic-embed-text is lightweight and
# runs well locally; swap if you prefer a different model (env override
# EMBEDDING_MODEL).  Changing it triggers an automatic re-index.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# Chunking parameters (approximate word counts)
CHUNK_SIZE_WORDS = 400       # target ~300-500 words per chunk
CHUNK_OVERLAP_WORDS = 50     # trailing block carried into the next chunk

# Retrieval
TOP_K = 4                    # number of chunks to retrieve per query

# Embedding / insert batch sizes
_EMBED_BATCH = 32
_ADD_BATCH = 500

# Bump when the chunking/index format changes so existing indexes rebuild.
_INDEX_FORMAT_VERSION = 2

# File where we persist a manifest of content hashes so we only re-embed
# when source files (or index settings) actually change.
_MANIFEST_PATH = os.path.join(CHROMA_DB_DIR, "_file_manifest.json")

_client = None


def _get_client():
    """Lazily create (and reuse) the persistent ChromaDB client."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    return _client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_hash(filepath: str) -> str:
    """Return the SHA-256 hex digest of a file's contents."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def _config_signature() -> str:
    """Settings that, when changed, require re-embedding everything."""
    return (
        f"v{_INDEX_FORMAT_VERSION}|{EMBEDDING_MODEL}|"
        f"{CHUNK_SIZE_WORDS}|{CHUNK_OVERLAP_WORDS}"
    )


def _load_manifest() -> dict:
    """Load the saved manifest ({"config": str, "files": {name: hash}})."""
    try:
        with open(_MANIFEST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_manifest(manifest: dict) -> None:
    """Persist the manifest to disk."""
    os.makedirs(os.path.dirname(_MANIFEST_PATH), exist_ok=True)
    with open(_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def _list_source_files() -> list[str]:
    """Return absolute paths of all .txt and .md files in knowledge_base/."""
    if not os.path.isdir(KNOWLEDGE_BASE_DIR):
        return []
    exts = {".txt", ".md"}
    return sorted(
        os.path.join(KNOWLEDGE_BASE_DIR, fn)
        for fn in os.listdir(KNOWLEDGE_BASE_DIR)
        if os.path.splitext(fn)[1].lower() in exts
    )


def _read_file(filepath: str) -> str:
    # utf-8-sig transparently drops the BOM that Windows editors often add.
    with open(filepath, "r", encoding="utf-8-sig") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _split_oversized(block: str) -> list[str]:
    """Split a single huge block into overlapping word windows."""
    words = block.split()
    step = max(1, CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS)
    return [
        " ".join(words[i:i + CHUNK_SIZE_WORDS])
        for i in range(0, len(words), step)
    ]


def _chunk_text(text: str, source_name: str) -> list[dict]:
    """
    Split *text* into chunks of roughly CHUNK_SIZE_WORDS words.

    Chunks follow the document structure: blocks (paragraphs, lists, tables)
    are kept intact with their newlines, headings stay with their content,
    and a chunk that does not start with a heading is prefixed with its
    section path so it remains understandable on its own.

    Returns a list of dicts:
        {"text": str, "source": str, "chunk_index": int}
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_blocks = [b.strip("\n") for b in re.split(r"\n\s*\n", text) if b.strip()]

    # (block_text, section_path_at_block_start, word_count)
    units: list[tuple[str, str, int]] = []
    stack: list[tuple[int, str]] = []
    for block in raw_blocks:
        section = " > ".join(t for _, t in stack)
        first_line = block.split("\n", 1)[0]
        m = _HEADING_RE.match(first_line)
        if m:
            level = len(m.group(1))
            stack = [h for h in stack if h[0] < level] + [(level, m.group(2))]
        pieces = (
            _split_oversized(block)
            if len(block.split()) > CHUNK_SIZE_WORDS
            else [block]
        )
        for piece in pieces:
            units.append((piece, section, len(piece.split())))

    chunks: list[dict] = []

    def flush(group: list[tuple[str, str, int]]) -> None:
        if not group:
            return
        body = "\n\n".join(u[0] for u in group)
        first_block, section, _ = group[0]
        if section and not _HEADING_RE.match(first_block.split("\n", 1)[0]):
            body = f"[Section: {section}]\n\n{body}"
        chunks.append({
            "text": body,
            "source": source_name,
            "chunk_index": len(chunks),
        })

    group: list[tuple[str, str, int]] = []
    total = 0
    for unit in units:
        if group and total + unit[2] > CHUNK_SIZE_WORDS:
            flush(group)
            last = group[-1]
            if last[2] <= CHUNK_OVERLAP_WORDS:
                group, total = [last], last[2]
            else:
                group, total = [], 0
        group.append(unit)
        total += unit[2]
    flush(group)
    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embed_text(text: str) -> list[float]:
    """Generate an embedding vector using Ollama's local embedding model."""
    response = ollama.embed(model=EMBEDDING_MODEL, input=text)
    return response.embeddings[0]


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed a list of texts."""
    out: list[list[float]] = []
    for i in range(0, len(texts), _EMBED_BATCH):
        response = ollama.embed(
            model=EMBEDDING_MODEL, input=texts[i:i + _EMBED_BATCH]
        )
        out.extend(response.embeddings)
    if len(out) != len(texts):
        raise RuntimeError(
            f"Embedding model returned {len(out)} vectors for {len(texts)} texts."
        )
    return out


# ---------------------------------------------------------------------------
# Public API — Initialization
# ---------------------------------------------------------------------------

def _index_is_populated() -> bool:
    try:
        return _get_client().get_collection(name=COLLECTION_NAME).count() > 0
    except Exception:
        return False


def initialize_knowledge_base() -> tuple[bool, str]:
    """
    Ensure the ChromaDB collection is populated with up-to-date embeddings
    from knowledge_base/.  Re-embeds only when a source file or an index
    setting (embedding model, chunking) has changed, or the index is missing.

    The old index is only replaced after the new embeddings were computed
    successfully, so a failed rebuild (e.g. Ollama down) never destroys it.

    Returns (success: bool, message: str).  Raises if embedding fails.
    """
    source_files = _list_source_files()

    if not source_files:
        return False, (
            "⚠️ No .txt or .md files found in `knowledge_base/`.  "
            "Add your FAQ/policy documents there and restart."
        )

    old = _load_manifest()
    new_files = {os.path.basename(fp): _file_hash(fp) for fp in source_files}
    signature = _config_signature()

    if (
        old.get("config") == signature
        and old.get("files") == new_files
        and _index_is_populated()
    ):
        return True, f"Knowledge base up-to-date ({len(source_files)} file(s))."

    # ------ Chunk + embed everything BEFORE touching the existing index ------
    all_chunks: list[dict] = []
    for fp in source_files:
        all_chunks.extend(_chunk_text(_read_file(fp), os.path.basename(fp)))

    if not all_chunks:
        return False, "Documents are empty - nothing to embed."

    texts = [c["text"] for c in all_chunks]
    embeddings = _embed_texts(texts)

    # ------ Replace the collection ------
    client = _get_client()
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass  # collection did not exist yet

    # IMPORTANT: Explicitly set cosine distance space.
    # Do NOT rely on ChromaDB's default (which is L2/euclidean).
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"{c['source']}__chunk_{c['chunk_index']}" for c in all_chunks]
    metadatas = [
        {"source": c["source"], "chunk_index": c["chunk_index"]}
        for c in all_chunks
    ]
    for i in range(0, len(ids), _ADD_BATCH):
        sl = slice(i, i + _ADD_BATCH)
        collection.add(
            ids=ids[sl],
            embeddings=embeddings[sl],
            documents=texts[sl],
            metadatas=metadatas[sl],
        )

    _save_manifest({"config": signature, "files": new_files})

    return True, (
        f"Embedded {len(all_chunks)} chunk(s) from "
        f"{len(source_files)} file(s)."
    )


# ---------------------------------------------------------------------------
# Public API — Retrieval (used by the agentic RAG loop in agents.py)
# ---------------------------------------------------------------------------

def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Retrieve the top-k most relevant knowledge-base chunks for *query*.

    Returns a list of dicts sorted by score descending:
        {"text": str, "source": str, "score": float}

    Score calculation:
        ChromaDB with hnsw:space="cosine" returns cosine DISTANCE.
        We convert:  similarity = 1 - distance
        Then round to 2 decimal places.
    """
    query = (query or "").strip()
    if not query:
        return []

    try:
        collection = _get_client().get_collection(name=COLLECTION_NAME)
        count = collection.count()
    except Exception:
        return []  # collection doesn't exist yet
    if count == 0:
        return []

    query_embedding = _embed_text(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    # ChromaDB returns lists-of-lists; unpack the first (only) query
    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    dists = results["distances"][0] if results["distances"] else []

    chunks = []
    for doc, meta, dist in zip(docs, metas, dists):
        # distance = 1 - similarity, so similarity = 1 - distance.
        similarity = round(1 - dist, 2)
        chunks.append({
            "text": doc,
            "source": (meta or {}).get("source", "unknown"),
            "score": similarity,
        })

    # Sort by score descending (most relevant first)
    chunks.sort(key=lambda c: c["score"], reverse=True)
    return chunks
