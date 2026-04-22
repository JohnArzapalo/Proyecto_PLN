# Knowledge Base

This folder contains the ChromaDB vector database generated at runtime.

## How it works

When the server starts, `RAGComponent` connects to (or creates) the ChromaDB
database stored here. Documents ingested via `rag.ingest_documents()` are
persisted in this folder and survive server restarts.

## Adding documents

Use the ingestion script (coming soon) or call directly from Python:

```python
from rag import RAGComponent

rag = RAGComponent()
rag.ingest_documents(
    documents=["Hannah is a 360M parameter model..."],
    metadatas=[{"source": "architecture.pdf"}],
    ids=["doc_001"]
)
```

## Contents

- `vectordb/` — ChromaDB persistent storage (auto-generated, do not edit manually)

> The `vectordb/` folder is excluded from Git via `.gitignore`.
> Each deployment must ingest its own documents.
