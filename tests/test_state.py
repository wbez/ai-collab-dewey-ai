from datetime import datetime, timedelta, timezone

import app.state as state_module
from app.state import StateStore


class AdvancingDatetime:
    current = datetime(2026, 1, 1, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):
        cls.current += timedelta(seconds=1)
        if tz is None:
            return cls.current.replace(tzinfo=None)
        return cls.current.astimezone(tz)


def test_record_query_keeps_ten_most_recent_per_user(tmp_path, monkeypatch):
    AdvancingDatetime.current = datetime(2026, 1, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(state_module, "datetime", AdvancingDatetime)
    store = StateStore(tmp_path / "state.sqlite3")

    for index in range(12):
        store.record_query(
            f"query-{index}",
            "T1",
            "U1",
            f"Question {index}",
            channel_id="C1",
            response_ts=f"1710000000.{index:06d}",
        )
    store.record_query("other-user-query", "T1", "U2", "Other user question")

    recent = store.recent_queries("T1", "U1", limit=20)

    assert len(recent) == 10
    assert [row["id"] for row in recent[:3]] == ["query-11", "query-10", "query-9"]
    assert "query-0" not in {row["id"] for row in recent}
    assert store.recent_queries("T1", "U2")[0]["id"] == "other-user-query"


def test_save_source_replaces_existing_source_for_user(tmp_path):
    store = StateStore(tmp_path / "state.sqlite3")

    store.save_source(
        "T1",
        "U1",
        {
            "source_id": "source-1",
            "content_type": "article",
            "title": "Original title",
            "canonical_url": "https://example.com/original",
        },
    )
    store.save_source(
        "T1",
        "U1",
        {
            "source_id": "source-1",
            "content_type": "transcript",
            "title": "Updated title",
            "canonical_url": "https://example.com/updated",
        },
    )

    saved = store.saved_sources("T1", "U1")

    assert len(saved) == 1
    assert saved[0]["source_id"] == "source-1"
    assert saved[0]["content_type"] == "transcript"
    assert saved[0]["title"] == "Updated title"
    assert saved[0]["canonical_url"] == "https://example.com/updated"


def test_unsave_source_is_scoped_to_team_and_user(tmp_path):
    store = StateStore(tmp_path / "state.sqlite3")
    source = {"source_id": "source-1", "content_type": "article", "title": "Story"}
    store.save_source("T1", "U1", source)
    store.save_source("T1", "U2", source)

    store.unsave_source("T1", "U1", "source-1")

    assert store.saved_sources("T1", "U1") == []
    assert store.saved_sources("T1", "U2")[0]["source_id"] == "source-1"
