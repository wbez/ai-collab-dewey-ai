"""Chunk and load occurrence scripts for direct Azure Search indexing."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from transcripts import TARGET_CHUNK_TOKENS, build_name_search_text, count_tokens, split_sentences


def normalize_azure_datetime(value: Any) -> str | None:
    """Return an Azure Edm.DateTimeOffset-compatible UTC value, if parseable."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def load_script_document(path: Path) -> Dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("content_type") != "script":
        raise ValueError("not a script document")
    required = ("id", "title", "content")
    missing = [key for key in required if not str(document.get(key) or "").strip()]
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")
    return document


def build_script_chunks(document: Dict[str, Any]) -> List[Dict[str, Any]]:
    content = str(document["content"]).strip()
    sentences = split_sentences(content) or [content]
    parts: List[str] = []
    current: List[str] = []
    current_tokens = 0
    for sentence in sentences:
        tokens = count_tokens(sentence)
        if current and current_tokens + tokens > TARGET_CHUNK_TOKENS:
            parts.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(sentence)
        current_tokens += tokens
    if current:
        parts.append(" ".join(current))

    recording_date = normalize_azure_datetime(document.get("recording_date"))
    chunks = []
    for index, text in enumerate(parts):
        fingerprint = hashlib.md5(f"{document['id']}:{index}:{text}".encode("utf-8")).hexdigest()[:12]
        search_text = "\n".join(
            value for value in (
                f"Title: {document['title']}",
                f"Description: {document.get('description')}" if document.get("description") else "",
                f"Program: {document.get('program')}" if document.get("program") else "",
                f"Script excerpt: {text}",
            ) if value
        )
        chunks.append({
            "chunk_id": f"{document['id']}-{index:04d}-{fingerprint}",
            "parent_id": str(document["id"]),
            "occurrence_id": str(document.get("occurrence_id") or "") or None,
            "content_type": "script",
            "title": document["title"],
            "description": document.get("description"),
            "program": document.get("program"),
            "guests": [], "speakers": [], "authors": [],
            "speaker_search_text": "", "author_search_text": build_name_search_text([]),
            "transcript_url": None, "transcript_name": path_name(document),
            "citation_url": document.get("url"), "recording_urls": [],
            "publish_date": recording_date, "recording_date": recording_date,
            "chunk_text": text, "search_text": search_text,
        })
    return chunks


def path_name(document: Dict[str, Any]) -> str:
    return f"{document['id']}.json"
