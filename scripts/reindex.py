"""Rebuild the ChromaDB vectors from SQLite with the current EMBED_MODEL.

Run from the project root after changing EMBED_MODEL, with the server stopped:  python scripts/reindex.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.label_store import COLLECTION  # noqa: E402
from app.llm import OllamaClient  # noqa: E402
from app.storage import Storage  # noqa: E402

BATCH = 64


def main() -> None:
    settings = get_settings()
    settings.ensure_dirs()
    rows = Storage(settings.db_path).query("SELECT * FROM labels ORDER BY id")
    llm = OllamaClient(settings)

    # Embed everything first, so a failure leaves the old collection untouched.
    embeddings = []
    for i in range(0, len(rows), BATCH):
        embeddings += llm.embed([r["original"] for r in rows[i : i + BATCH]])

    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass  # no collection yet
    col = client.create_collection(
        COLLECTION, embedding_function=None,
        metadata={"hnsw:space": "cosine", "embed_model": settings.embed_model},
    )
    for i in range(0, len(rows), BATCH):
        chunk = rows[i : i + BATCH]
        col.add(
            ids=[str(r["id"]) for r in chunk],
            embeddings=embeddings[i : i + BATCH],
            documents=[r["original"] for r in chunk],
            metadatas=[{"normalized": r["normalized"], "field": r["field"], "section": r["section"],
                        "source": r["source"], "embed_model": settings.embed_model} for r in chunk],
        )
    print(f"Reindexed {len(rows)} labels with {settings.embed_model}")


if __name__ == "__main__":
    main()
