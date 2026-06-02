"""HalluScope - Persistent message store.

SQLite-backed key/value store for analysed messages. Replaces the
process-local dict that previously lived in api/chat.py so that
metrics links survive a backend restart.

Pydantic objects (MetricsDetail, TokenDetail) are serialised to plain
dicts before write so json.dumps does not need a custom encoder.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

# Repo root: backend/app/storage.py -> backend/app -> backend -> repo root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = ROOT_DIR / "halluscope.db"


def _connect() -> sqlite3.Connection:
    # check_same_thread=False is safe here because every helper opens, uses,
    # and closes its own connection — we never share one across threads.
    return sqlite3.connect(str(DB_PATH), check_same_thread=False)


def init_db() -> None:
    """Create the messages table if it does not yet exist."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                message_id   TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def _to_jsonable(value: Any) -> Any:
    """Recursively convert pydantic models / dataclasses into plain JSON types."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    return value


def save_message(message_id: str, payload: dict) -> None:
    """Insert (or replace) a stored message payload."""
    serialisable = {k: _to_jsonable(v) for k, v in payload.items()}
    blob = json.dumps(serialisable, allow_nan=False)
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO messages (message_id, payload_json) "
            "VALUES (?, ?)",
            (message_id, blob),
        )
        conn.commit()


def get_message(message_id: str) -> Optional[dict]:
    """Return the stored payload for message_id, or None if not found."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT payload_json FROM messages WHERE message_id = ?",
            (message_id,),
        ).fetchone()
    if row is None:
        return None
    return json.loads(row[0])
