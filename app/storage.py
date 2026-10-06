"""SQLite storage (pending records, labels, vendors, file hashes) and output files."""
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pending (
    doc_id TEXT PRIMARY KEY,
    file_hash TEXT, filename TEXT, raw_text TEXT,
    extracted_json TEXT, warnings_json TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    was_corrected INTEGER NOT NULL DEFAULT 0,
    llm_model TEXT, embed_model TEXT, prompt_version TEXT,
    created_at TEXT, confirmed_at TEXT
);
CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    normalized TEXT NOT NULL, original TEXT NOT NULL,
    field TEXT NOT NULL, section TEXT NOT NULL,
    pattern TEXT, count INTEGER NOT NULL DEFAULT 0,
    vendors TEXT NOT NULL DEFAULT '[]',
    source TEXT NOT NULL, created_at TEXT,
    UNIQUE (normalized, field, section)
);
CREATE TABLE IF NOT EXISTS vendors (
    key TEXT PRIMARY KEY,
    name TEXT, vat_number TEXT, currency TEXT, vat_rate REAL,
    bank_hash TEXT, bank_last4 TEXT,
    count INTEGER NOT NULL DEFAULT 0,
    min_total REAL, max_total REAL, avg_total REAL,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS confirmed_invoices (
    vendor_key TEXT NOT NULL, invoice_number TEXT NOT NULL, doc_id TEXT NOT NULL,
    PRIMARY KEY (vendor_key, invoice_number)
);
CREATE TABLE IF NOT EXISTS files (
    file_hash TEXT NOT NULL, doc_id TEXT NOT NULL, endpoint TEXT NOT NULL, created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_files_hash ON files (file_hash);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Storage:
    def __init__(self, db_path: Path | str):
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def execute(self, sql: str, params: tuple = ()) -> int:
        """Run a write statement; returns lastrowid."""
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur.lastrowid

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def one(self, sql: str, params: tuple = ()) -> dict | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    # --- files -------------------------------------------------------------
    def seen_file(self, file_hash: str) -> dict | None:
        return self.one("SELECT * FROM files WHERE file_hash = ? ORDER BY created_at LIMIT 1", (file_hash,))

    def record_file(self, file_hash: str, doc_id: str, endpoint: str) -> None:
        self.execute(
            "INSERT INTO files (file_hash, doc_id, endpoint, created_at) VALUES (?, ?, ?, ?)",
            (file_hash, doc_id, endpoint, now_iso()),
        )

    # --- pending -----------------------------------------------------------
    def save_pending(self, doc_id, file_hash, filename, raw_text, extracted: dict, warnings: list[str],
                     llm_model, embed_model, prompt_version) -> None:
        self.execute(
            "INSERT INTO pending (doc_id, file_hash, filename, raw_text, extracted_json, warnings_json,"
            " status, llm_model, embed_model, prompt_version, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?)",
            (doc_id, file_hash, filename, raw_text, json.dumps(extracted), json.dumps(warnings),
             llm_model, embed_model, prompt_version, now_iso()),
        )

    def get_pending(self, doc_id: str) -> dict | None:
        row = self.one("SELECT * FROM pending WHERE doc_id = ?", (doc_id,))
        if row:
            row["extracted"] = json.loads(row.pop("extracted_json") or "{}")
            row["warnings"] = json.loads(row.pop("warnings_json") or "[]")
        return row

    def list_pending(self, status: str = "pending") -> list[dict]:
        return self.query(
            "SELECT doc_id, filename, status, was_corrected, created_at, confirmed_at"
            " FROM pending WHERE status = ? ORDER BY created_at",
            (status,),
        )

    def set_pending_status(self, doc_id: str, status: str, was_corrected: bool = False,
                           extracted: dict | None = None) -> None:
        if extracted is not None:
            self.execute("UPDATE pending SET extracted_json = ? WHERE doc_id = ?", (json.dumps(extracted), doc_id))
        self.execute(
            "UPDATE pending SET status = ?, was_corrected = ?, confirmed_at = ? WHERE doc_id = ?",
            (status, int(was_corrected), now_iso(), doc_id),
        )

    # --- confirmed invoices ------------------------------------------------
    def find_confirmed(self, vendor_key: str, invoice_number: str) -> dict | None:
        return self.one(
            "SELECT * FROM confirmed_invoices WHERE vendor_key = ? AND invoice_number = ?",
            (vendor_key, invoice_number),
        )

    def save_confirmed(self, vendor_key: str, invoice_number: str, doc_id: str) -> None:
        self.execute(
            "INSERT OR REPLACE INTO confirmed_invoices (vendor_key, invoice_number, doc_id) VALUES (?, ?, ?)",
            (vendor_key, invoice_number, doc_id),
        )


def save_output(outputs_dir: Path, doc_id: str, result: dict) -> None:
    (outputs_dir / f"{doc_id}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
