"""Small, persistent per-user state store for the Slack Home tab."""

import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def state_db_path() -> Path:
    return Path(os.environ.get("WAVELENGTH_STATE_DB", "data/wavelength-state.sqlite3"))


class StateStore:
    def __init__(self, path: Optional[str | Path] = None):
        self.path = Path(path) if path else state_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _migrate(self) -> None:
        with self._connect() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version < 1:
                db.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS recent_queries (
                        id TEXT PRIMARY KEY,
                        team_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        question TEXT NOT NULL,
                        response_channel_id TEXT,
                        response_ts TEXT,
                        created_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS saved_sources (
                        team_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        source_id TEXT NOT NULL,
                        content_type TEXT NOT NULL,
                        title TEXT NOT NULL,
                        canonical_url TEXT,
                        saved_at TEXT NOT NULL,
                        PRIMARY KEY (team_id, user_id, source_id)
                    );
                    CREATE INDEX IF NOT EXISTS recent_queries_user_idx
                        ON recent_queries(team_id, user_id, created_at DESC);
                    PRAGMA user_version = 1;
                    """
                )

    def record_query(self, query_id: str, team_id: str, user_id: str, question: str,
                     channel_id: Optional[str] = None, response_ts: Optional[str] = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        with self._connect() as db:
            db.execute("DELETE FROM recent_queries WHERE created_at < ?", (cutoff,))
            db.execute(
                "INSERT OR REPLACE INTO recent_queries "
                "(id, team_id, user_id, question, response_channel_id, response_ts, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (query_id, team_id, user_id, question.strip(), channel_id, response_ts, now),
            )
            ids = db.execute(
                "SELECT id FROM recent_queries WHERE team_id=? AND user_id=? "
                "ORDER BY created_at DESC", (team_id, user_id)
            ).fetchall()
            for row in ids[10:]:
                db.execute("DELETE FROM recent_queries WHERE id=?", (row["id"],))

    def recent_queries(self, team_id: str, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM recent_queries WHERE team_id=? AND user_id=? "
                "ORDER BY created_at DESC LIMIT ?", (team_id, user_id, min(limit, 10))
            ).fetchall()
        return [dict(row) for row in rows]

    def save_source(self, team_id: str, user_id: str, source: Dict[str, Any]) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO saved_sources "
                "(team_id,user_id,source_id,content_type,title,canonical_url,saved_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (team_id, user_id, str(source["source_id"]), source.get("content_type", "article"),
                 source.get("title", "Untitled source"), source.get("canonical_url"),
                 datetime.now(timezone.utc).isoformat()),
            )

    def unsave_source(self, team_id: str, user_id: str, source_id: str) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM saved_sources WHERE team_id=? AND user_id=? AND source_id=?",
                       (team_id, user_id, source_id))

    def saved_sources(self, team_id: str, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM saved_sources WHERE team_id=? AND user_id=? "
                "ORDER BY saved_at DESC LIMIT ?", (team_id, user_id, min(limit, 10))
            ).fetchall()
        return [dict(row) for row in rows]
