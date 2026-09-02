import json

import pytest

from scripts_content import (
    build_script_chunks,
    load_script_document,
    normalize_azure_datetime,
)


def test_normalize_azure_datetime_converts_offsets_to_utc():
    assert normalize_azure_datetime("2026-01-02T03:04:05-06:00") == "2026-01-02T09:04:05Z"


def test_normalize_azure_datetime_treats_naive_values_as_utc():
    assert normalize_azure_datetime("2026-01-02T03:04:05") == "2026-01-02T03:04:05Z"


@pytest.mark.parametrize("value", ["", None, "not-a-date"])
def test_normalize_azure_datetime_returns_none_for_blank_or_invalid_values(value):
    assert normalize_azure_datetime(value) is None


def test_load_script_document_requires_script_content_type(tmp_path):
    path = tmp_path / "article.json"
    path.write_text(
        json.dumps({"content_type": "article", "id": "1", "title": "Title", "content": "Body"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not a script document"):
        load_script_document(path)


def test_load_script_document_reports_missing_required_fields(tmp_path):
    path = tmp_path / "script.json"
    path.write_text(json.dumps({"content_type": "script", "id": "script-1"}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required fields: title, content"):
        load_script_document(path)


def test_build_script_chunks_sets_search_and_citation_fields():
    chunks = build_script_chunks(
        {
            "content_type": "script",
            "id": "script-1",
            "title": "Morning Rundown",
            "description": "Host copy and rundowns",
            "program": "Morning Show",
            "occurrence_id": 123,
            "recording_date": "2026-01-02T03:04:05-06:00",
            "url": "https://example.com/script",
            "content": "First sentence. Second sentence.",
        }
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["chunk_id"].startswith("script-1-0000-")
    assert chunk["parent_id"] == "script-1"
    assert chunk["occurrence_id"] == "123"
    assert chunk["content_type"] == "script"
    assert chunk["title"] == "Morning Rundown"
    assert chunk["program"] == "Morning Show"
    assert chunk["citation_url"] == "https://example.com/script"
    assert chunk["publish_date"] == "2026-01-02T09:04:05Z"
    assert chunk["chunk_text"] == "First sentence. Second sentence."
    assert "Title: Morning Rundown" in chunk["search_text"]
    assert "Script excerpt: First sentence. Second sentence." in chunk["search_text"]
