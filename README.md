# Agentic RAG chatbot

A fully local customer-support chatbot (no cloud calls, no API keys).

- **UI:** Streamlit (`app.py`)
- **Agent 1 (frontline):** agentic RAG loop — decide → retrieve → score-filter → LLM-judge → refine → answer / hand off (`agents.py`)
- **Agent 2 (resolution):** handles escalations. It has **no live order/billing backend**, so it never claims to have performed an action; it gathers the needed details and points the customer to the real support channels.
- **RAG:** ChromaDB (cosine) + Ollama `nomic-embed-text` over the markdown/text files in `knowledge_base/` (`rag.py`)

## Setup

1. Install [Ollama](https://ollama.com) and pull the models:
   ```
   ollama pull qwen3:8b
   ollama pull nomic-embed-text
   ```
2. Python 3.10+ (developed on 3.11):
   ```
   python -m venv .venv
   .venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Run (Ollama must be running):
   ```
   streamlit run app.py
   ```
   The index in `chroma_db/` is built on first start and rebuilt automatically when files in `knowledge_base/` (or the embedding/chunking settings) change. Press **Re-scan knowledge base** in the sidebar after adding files.

## Configuration (optional environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `GENERATION_MODEL` | `qwen3:8b` | Ollama chat model |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model (changing it re-indexes) |
| `RELEVANCE_THRESHOLD` | `0.6` | Min cosine similarity for a chunk to count |
| `MAX_RETRIEVAL_ITERATIONS` | `4` | Searches before forced hand-off |
| `MAX_HISTORY_MESSAGES` | `12` | Chat messages sent to the model |
| `SUPPORT_CONTACT` | text from the KB | Contact details Agent 2 may quote |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Read by the `ollama` library |

## Tests

```
python -m unittest discover -s tests
```
The tests use in-memory fakes for Ollama and ChromaDB, so they need neither a running Ollama nor the real libraries.

## Project layout

```
app.py            Streamlit UI
agents.py         Agent 1 loop + Agent 2
rag.py            chunking, embedding, indexing, retrieval
validation.py     input filter + HTML-escaping helpers
knowledge_base/   .md / .txt source documents
.streamlit/       dark theme config
tests/            unit tests
```

## Known limitations

- Agent 2 is advisory only; real order lookups, cancellations or refunds need a backend integration.
- Some knowledge-base files contradict each other (see below). The bot can only be as consistent as its documents.
- Answer quality depends on the local 8B model; unusual phrasing may trigger an unnecessary hand-off.

## Knowledge-base inconsistencies to resolve

- International delivery: `order_tracking.md` says 7–12 (Canada) / 10–18 (UK/EU) business days; `order_tracking_faq.md` says 7–21.
- Tracking not updating: 5 business days (`order_tracking.md`) vs 3 (`order_tracking_faq.md`).
- Overnight cutoff: 2 PM CT vs 2 PM local time.
- Order status names differ between the two files (e.g. "Packed", "Delayed" vs "Order Placed", "In Transit", "Exception").
