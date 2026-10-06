"""The label library: an in-memory dictionary (exact + fuzzy) backed by SQLite, plus ChromaDB vectors.

SQLite is the source of truth; ChromaDB only holds the vectors and can be rebuilt with scripts/reindex.py.
"""
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Callable

from rapidfuzz import fuzz, process

from .config import Settings
from .labels import normalize_label
from .storage import Storage, now_iso

log = logging.getLogger(__name__)
COLLECTION = "labels"
FUZZY_MIN_LENGTH = 4  # very short labels ("dt", "vat") must match exactly


class EmbedModelMismatch(RuntimeError):
    pass


@dataclass
class LabelRecord:
    id: int
    normalized: str
    original: str
    field: str
    section: str
    pattern: str | None
    count: int
    source: str
    vendors: list[str] = field(default_factory=list)


def open_collection(settings: Settings, create: bool = True):
    """Open the ChromaDB labels collection and check it was built with EMBED_MODEL."""
    import chromadb

    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    try:
        col = client.get_collection(COLLECTION, embedding_function=None)
    except Exception:
        if not create:
            raise
        col = client.create_collection(
            COLLECTION,
            embedding_function=None,
            metadata={"hnsw:space": "cosine", "embed_model": settings.embed_model},
        )
    stored = (col.metadata or {}).get("embed_model")
    if stored != settings.embed_model:
        raise EmbedModelMismatch(
            f"The label library was built with embedding model '{stored}', but EMBED_MODEL is "
            f"'{settings.embed_model}'. Run: python scripts/reindex.py"
        )
    return col


class LabelStore:
    def __init__(self, db: Storage, settings: Settings,
                 embed: Callable[[list[str]], list[list[float]]] | None = None, collection=None):
        self.db = db
        self.settings = settings
        self.embed = embed
        self.collection = collection
        self._lock = threading.Lock()
        self._dict: dict[str, list[LabelRecord]] = {}
        self._by_id: dict[int, LabelRecord] = {}
        self.load()

    # --- dictionary ----------------------------------------------------------
    def load(self) -> None:
        with self._lock:
            self._dict.clear()
            self._by_id.clear()
            for row in self.db.query("SELECT * FROM labels"):
                self._index(LabelRecord(
                    id=row["id"], normalized=row["normalized"], original=row["original"],
                    field=row["field"], section=row["section"], pattern=row["pattern"],
                    count=row["count"], source=row["source"], vendors=json.loads(row["vendors"]),
                ))

    def _index(self, rec: LabelRecord) -> None:
        self._dict.setdefault(rec.normalized, []).append(rec)
        self._by_id[rec.id] = rec

    def __len__(self) -> int:
        return len(self._by_id)

    def all(self) -> list[LabelRecord]:
        return list(self._by_id.values())

    def _usable(self, recs: list[LabelRecord], sources: set[str] | None) -> list[LabelRecord]:
        return [
            r for r in recs
            if (sources is None or r.source in sources)
            and (r.source == "seed" or r.count >= self.settings.min_label_count)
        ]

    def lookup(self, label: str, sources: set[str] | None = None) -> tuple[list[LabelRecord], str | None]:
        """Dictionary lookup. Returns (records, 'exact' | 'fuzzy' | None)."""
        norm = normalize_label(label)
        if not norm:
            return [], None
        recs = self._usable(self._dict.get(norm, []), sources)
        if recs:
            return recs, "exact"
        if len(norm) < FUZZY_MIN_LENGTH:
            return [], None
        keys = [k for k, v in self._dict.items() if len(k) >= FUZZY_MIN_LENGTH and self._usable(v, sources)]
        best = process.extractOne(norm, keys, scorer=fuzz.ratio, score_cutoff=self.settings.fuzzy_match_min)
        if best:
            return self._usable(self._dict[best[0]], sources), "fuzzy"
        return [], None

    # --- vectors -------------------------------------------------------------
    def vector_lookup(self, labels: list[str], sources: set[str] | None = None) -> list[list[LabelRecord]]:
        """One batch embedding call, then a vector search per label. Empty list = no match."""
        empty = [[] for _ in labels]
        if not labels or self.collection is None or self.embed is None or self.collection.count() == 0:
            return empty
        embeddings = self.embed(labels)
        kwargs = {}
        if sources is not None:
            kwargs["where"] = {"source": {"$in": sorted(sources)}}
        res = self.collection.query(
            query_embeddings=embeddings, n_results=self.settings.vector_top_k,
            include=["distances"], **kwargs,
        )
        out = []
        for ids, dists in zip(res["ids"], res["distances"]):
            recs = []
            for id_, dist in zip(ids, dists):
                rec = self._by_id.get(int(id_))
                if rec and 1.0 - dist >= self.settings.vector_sim_threshold:
                    recs.append(rec)
            out.append(self._usable(recs, sources))
        return out

    # --- writes --------------------------------------------------------------
    def add_labels(self, items: list[dict], source: str, vendor: str | None = None,
                   increment: bool = True) -> tuple[int, int]:
        """items: [{'label', 'field', 'section', 'pattern'}]. Returns (added, updated)."""
        added = updated = 0
        new: dict[tuple, dict] = {}
        with self._lock:
            for item in items:
                norm = normalize_label(item["label"])
                if not norm:
                    continue
                key = (norm, item["field"], item["section"])
                existing = next(
                    (r for r in self._dict.get(norm, []) if (r.normalized, r.field, r.section) == key), None
                )
                if existing is None:
                    new.setdefault(key, item)
                    continue
                if not increment:
                    continue
                existing.count += 1
                if vendor and vendor not in existing.vendors:
                    existing.vendors.append(vendor)
                if not existing.pattern and item.get("pattern"):
                    existing.pattern = item["pattern"]
                self.db.execute(
                    "UPDATE labels SET count = ?, vendors = ?, pattern = ? WHERE id = ?",
                    (existing.count, json.dumps(existing.vendors), existing.pattern, existing.id),
                )
                updated += 1

            if not new:
                return added, updated
            originals = [item["label"].strip() for item in new.values()]
            embeddings = self.embed(originals) if self.embed and self.collection is not None else None
            count = 0 if source == "seed" else 1
            vendors = [vendor] if vendor else []
            recs = []
            for (norm, fld, section), item in new.items():
                label_id = self.db.execute(
                    "INSERT INTO labels (normalized, original, field, section, pattern, count, vendors,"
                    " source, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (norm, item["label"].strip(), fld, section, item.get("pattern"), count,
                     json.dumps(vendors), source, now_iso()),
                )
                rec = LabelRecord(label_id, norm, item["label"].strip(), fld, section,
                                  item.get("pattern"), count, source, list(vendors))
                self._index(rec)
                recs.append(rec)
                added += 1
            if embeddings is not None:
                self.collection.add(
                    ids=[str(r.id) for r in recs],
                    embeddings=embeddings,
                    documents=[r.original for r in recs],
                    metadatas=[{"normalized": r.normalized, "field": r.field, "section": r.section,
                                "source": r.source, "embed_model": self.settings.embed_model} for r in recs],
                )
        return added, updated

    def delete(self, label_id: int) -> bool:
        with self._lock:
            rec = self._by_id.pop(label_id, None)
            if rec is None:
                return False
            self._dict[rec.normalized] = [r for r in self._dict[rec.normalized] if r.id != label_id]
            if not self._dict[rec.normalized]:
                del self._dict[rec.normalized]
            self.db.execute("DELETE FROM labels WHERE id = ?", (label_id,))
            if self.collection is not None:
                self.collection.delete(ids=[str(label_id)])
        return True
