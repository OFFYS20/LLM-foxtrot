"""SQLite metadata store.

Large artefacts (weights, indexes, corpora) live on the filesystem; everything
you need to *find* them lives here. Connections are per-thread because Gradio
handlers, the training worker and the monitor all touch the database
concurrently.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ai_studio.core.config import get_config
from ai_studio.core.errors import NotFoundError

SCHEMA_VERSION = 1

#: Columns stored as JSON text, decoded transparently on read.
JSON_COLUMNS: dict[str, set[str]] = {
    "documents": {"meta"},
    "datasets": {"config", "source_document_ids", "stats"},
    "tokenizers": {"special_tokens", "trained_on", "stats"},
    "models": {"config", "meta", "metrics"},
    "experiments": {"hyperparameters", "benchmark_summary", "metrics"},
    "checkpoints": {"meta", "metrics"},
    "conversations": {"params", "meta"},
    "messages": {"stats"},
    "benchmark_runs": {"config", "category_scores"},
    "benchmark_items": {"meta"},
    "rag_indexes": {"document_ids", "config", "stats"},
    "logs": {"context"},
    "settings": {"value"},
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    title TEXT,
    author TEXT,
    doc_type TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'file',
    original_path TEXT,
    cleaned_path TEXT,
    size_bytes INTEGER DEFAULT 0,
    char_count INTEGER DEFAULT 0,
    word_count INTEGER DEFAULT 0,
    token_estimate INTEGER DEFAULT 0,
    sha256 TEXT,
    status TEXT NOT NULL DEFAULT 'imported',
    error TEXT,
    meta TEXT DEFAULT '{}',
    imported_at REAL NOT NULL,
    cleaned_at REAL
);
CREATE INDEX IF NOT EXISTS idx_documents_imported ON documents(imported_at DESC);

CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    mode TEXT NOT NULL,
    description TEXT,
    path TEXT,
    tokenizer_id TEXT,
    rows INTEGER DEFAULT 0,
    train_rows INTEGER DEFAULT 0,
    validation_rows INTEGER DEFAULT 0,
    test_rows INTEGER DEFAULT 0,
    token_count INTEGER DEFAULT 0,
    token_method TEXT DEFAULT 'estimated',
    is_synthetic INTEGER DEFAULT 0,
    content_hash TEXT,
    source_document_ids TEXT DEFAULT '[]',
    config TEXT DEFAULT '{}',
    stats TEXT DEFAULT '{}',
    notes TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS tokenizers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    vocab_size INTEGER DEFAULT 0,
    path TEXT NOT NULL,
    special_tokens TEXT DEFAULT '{}',
    trained_on TEXT DEFAULT '[]',
    stats TEXT DEFAULT '{}',
    source TEXT DEFAULT 'trained',
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    architecture TEXT,
    path TEXT,
    repo_id TEXT,
    revision TEXT,
    tokenizer_id TEXT,
    parameters INTEGER,
    context_length INTEGER,
    precision TEXT,
    quantization TEXT,
    size_bytes INTEGER DEFAULT 0,
    license TEXT,
    status TEXT NOT NULL DEFAULT 'ready',
    parent_model_id TEXT,
    source_checkpoint_id TEXT,
    error TEXT,
    config TEXT DEFAULT '{}',
    metrics TEXT DEFAULT '{}',
    meta TEXT DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    method TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    model_id TEXT,
    base_model_id TEXT,
    dataset_id TEXT,
    dataset_hash TEXT,
    tokenizer_id TEXT,
    output_model_id TEXT,
    hyperparameters TEXT DEFAULT '{}',
    metrics TEXT DEFAULT '{}',
    benchmark_summary TEXT DEFAULT '{}',
    final_train_loss REAL,
    best_val_loss REAL,
    best_checkpoint_id TEXT,
    total_steps INTEGER DEFAULT 0,
    current_step INTEGER DEFAULT 0,
    current_epoch REAL DEFAULT 0,
    tokens_processed INTEGER DEFAULT 0,
    started_at REAL,
    ended_at REAL,
    duration_seconds REAL,
    error TEXT,
    notes TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_experiments_created ON experiments(created_at DESC);

CREATE TABLE IF NOT EXISTS training_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    epoch REAL,
    ts REAL NOT NULL,
    loss REAL,
    val_loss REAL,
    learning_rate REAL,
    grad_norm REAL,
    tokens_per_sec REAL,
    samples_per_sec REAL,
    gpu_util REAL,
    vram_mb REAL,
    ram_mb REAL
);
CREATE INDEX IF NOT EXISTS idx_metrics_exp_step ON training_metrics(experiment_id, step);

CREATE TABLE IF NOT EXISTS checkpoints (
    id TEXT PRIMARY KEY,
    experiment_id TEXT,
    model_id TEXT,
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    step INTEGER DEFAULT 0,
    epoch REAL DEFAULT 0,
    train_loss REAL,
    val_loss REAL,
    benchmark_score REAL,
    size_bytes INTEGER DEFAULT 0,
    is_best INTEGER DEFAULT 0,
    has_optimizer_state INTEGER DEFAULT 0,
    metrics TEXT DEFAULT '{}',
    meta TEXT DEFAULT '{}',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_checkpoints_exp ON checkpoints(experiment_id, step DESC);

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT 'New conversation',
    model_id TEXT,
    system_prompt TEXT DEFAULT '',
    rag_index_id TEXT,
    params TEXT DEFAULT '{}',
    meta TEXT DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    stats TEXT DEFAULT '{}',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS benchmark_runs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    suite TEXT NOT NULL,
    model_id TEXT NOT NULL,
    checkpoint_id TEXT,
    dataset_id TEXT,
    status TEXT NOT NULL DEFAULT 'queued',
    config TEXT DEFAULT '{}',
    total_items INTEGER DEFAULT 0,
    completed_items INTEGER DEFAULT 0,
    accuracy REAL,
    score REAL,
    avg_latency_ms REAL,
    tokens_per_sec REAL,
    total_tokens INTEGER DEFAULT 0,
    runtime_seconds REAL,
    category_scores TEXT DEFAULT '{}',
    error TEXT,
    started_at REAL,
    ended_at REAL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS benchmark_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    category TEXT,
    prompt TEXT,
    question TEXT,
    expected TEXT,
    response TEXT,
    correct INTEGER DEFAULT 0,
    score REAL DEFAULT 0,
    method TEXT,
    latency_ms REAL DEFAULT 0,
    tokens INTEGER DEFAULT 0,
    meta TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_bench_items_run ON benchmark_items(run_id, idx);

CREATE TABLE IF NOT EXISTS rag_indexes (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    embedding_model TEXT NOT NULL,
    path TEXT NOT NULL,
    dimension INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0,
    chunk_size INTEGER DEFAULT 0,
    chunk_overlap INTEGER DEFAULT 0,
    document_ids TEXT DEFAULT '[]',
    config TEXT DEFAULT '{}',
    stats TEXT DEFAULT '{}',
    status TEXT DEFAULT 'ready',
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    level TEXT NOT NULL,
    source TEXT NOT NULL,
    message TEXT NOT NULL,
    context TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_logs_ts ON logs(ts DESC);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS schema_info (
    version INTEGER PRIMARY KEY,
    applied_at REAL NOT NULL
);
"""


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Database:
    """Thin, thread-safe repository over SQLite."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else get_config().database_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.RLock()
        self._initialise()

    # ------------------------------------------------------------ plumbing
    @property
    def connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def _initialise(self) -> None:
        with self._write_lock:
            self.connection.executescript(SCHEMA)
            self.connection.execute(
                "INSERT OR IGNORE INTO schema_info (version, applied_at) VALUES (?, ?)",
                (SCHEMA_VERSION, time.time()),
            )
            self.connection.commit()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._write_lock:
            conn = self.connection
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # --------------------------------------------------------------- CRUD
    def insert(self, table: str, values: dict[str, Any]) -> str:
        payload = _encode(table, values)
        columns = ", ".join(payload)
        placeholders = ", ".join("?" for _ in payload)
        with self.transaction() as conn:
            cursor = conn.execute(
                f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",  # noqa: S608
                tuple(payload.values()),
            )
            return str(values.get("id") or cursor.lastrowid)

    def update(self, table: str, row_id: Any, values: dict[str, Any], key: str = "id") -> None:
        if not values:
            return
        payload = _encode(table, values)
        assignments = ", ".join(f"{column} = ?" for column in payload)
        with self.transaction() as conn:
            conn.execute(
                f"UPDATE {table} SET {assignments} WHERE {key} = ?",  # noqa: S608
                (*payload.values(), row_id),
            )

    def get(self, table: str, row_id: Any, key: str = "id") -> dict[str, Any] | None:
        row = self.connection.execute(
            f"SELECT * FROM {table} WHERE {key} = ?", (row_id,)  # noqa: S608
        ).fetchone()
        return _decode(table, row)

    def require(self, table: str, row_id: Any, key: str = "id") -> dict[str, Any]:
        record = self.get(table, row_id, key)
        if record is None:
            raise NotFoundError(f"{table[:-1] if table.endswith('s') else table} {row_id} not found")
        return record

    def delete(self, table: str, row_id: Any, key: str = "id") -> None:
        with self.transaction() as conn:
            conn.execute(f"DELETE FROM {table} WHERE {key} = ?", (row_id,))  # noqa: S608

    def query(self, sql: str, params: Iterable[Any] = (), *, table: str | None = None) -> list[dict[str, Any]]:
        rows = self.connection.execute(sql, tuple(params)).fetchall()
        inferred = table or _table_from_sql(sql)
        return [_decode(inferred, row) for row in rows if row is not None]  # type: ignore[misc]

    def scalar(self, sql: str, params: Iterable[Any] = ()) -> Any:
        row = self.connection.execute(sql, tuple(params)).fetchone()
        return row[0] if row else None

    def list(
        self,
        table: str,
        *,
        where: str | None = None,
        params: Iterable[Any] = (),
        order_by: str = "rowid DESC",
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        sql = f"SELECT * FROM {table}"  # noqa: S608
        if where:
            sql += f" WHERE {where}"
        sql += f" ORDER BY {order_by}"
        if limit is not None:
            sql += f" LIMIT {int(limit)} OFFSET {int(offset)}"
        return self.query(sql, params, table=table)

    def count(self, table: str, where: str | None = None, params: Iterable[Any] = ()) -> int:
        sql = f"SELECT COUNT(*) FROM {table}"  # noqa: S608
        if where:
            sql += f" WHERE {where}"
        return int(self.scalar(sql, params) or 0)

    # ----------------------------------------------------------- settings
    def get_setting(self, key: str, default: Any = None) -> Any:
        row = self.get("settings", key, key="key")
        return row["value"] if row else default

    def set_setting(self, key: str, value: Any) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                (key, json.dumps(value), time.time()),
            )


def _encode(table: str, values: dict[str, Any]) -> dict[str, Any]:
    json_columns = JSON_COLUMNS.get(table, set())
    encoded: dict[str, Any] = {}
    for column, value in values.items():
        if column in json_columns and not isinstance(value, (str, type(None))):
            encoded[column] = json.dumps(value)
        elif isinstance(value, bool):
            encoded[column] = int(value)
        elif isinstance(value, Path):
            encoded[column] = str(value)
        else:
            encoded[column] = value
    return encoded


def _decode(table: str | None, row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    record = dict(row)
    for column in JSON_COLUMNS.get(table or "", set()):
        raw = record.get(column)
        if isinstance(raw, str):
            try:
                record[column] = json.loads(raw)
            except json.JSONDecodeError:
                record[column] = {}
    return record


def _table_from_sql(sql: str) -> str | None:
    lowered = sql.lower()
    if " from " not in lowered:
        return None
    after = lowered.split(" from ", 1)[1].strip()
    return after.split()[0].strip("`\"'") if after else None


_db: Database | None = None
_db_lock = threading.Lock()


def get_db() -> Database:
    global _db
    if _db is None:
        with _db_lock:
            if _db is None:
                _db = Database()
    return _db


def reset_db(path: Path | str | None = None) -> Database:
    """Used by tests to point the singleton at a temporary file."""
    global _db
    with _db_lock:
        if _db is not None:
            _db.close()
        _db = Database(path)
    return _db
