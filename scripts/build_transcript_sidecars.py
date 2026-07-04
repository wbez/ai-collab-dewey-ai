#!/usr/bin/env python3
"""Create sidecar JSON files for local VTT transcripts."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from tqdm import tqdm

try:
    import boto3
except ModuleNotFoundError:  # pragma: no cover
    boto3 = None  # type: ignore[assignment]

import dateparser
try:
    import pymysql
except ModuleNotFoundError:  # pragma: no cover
    pymysql = None  # type: ignore[assignment]
from azure.storage.blob import BlobServiceClient


ENV_CONNECTION_STRING = "CANADA_STORAGE_AZURE_TABLE_CONN"
DEFAULT_TRANSCRIPT_CONTAINER = "final-transcripts"
DEV_CA_CERT = Path(__file__).resolve().parents[1] / "certs" / "global-bundle.pem"
_NOTE_LINE_RE = re.compile(r"^NOTE\s+(?P<key>[A-Za-z0-9_]+(?:\[\d+\])?)\s*:\s*(?P<value>.*)$")
_INDEXED_KEY_RE = re.compile(r"^(?P<key>[A-Za-z0-9_]+)\[(?P<index>\d+)\]$")
_FILE_PART_RE = re.compile(r"([A-Za-z0-9_]+)=((?:\"[^\"]*\")|(?:[^;]+))")
_VOICE_TAG_RE = re.compile(r"<v\s+([^>]+)>", re.IGNORECASE)

_NAME_PREFIXES = (
    "dr.",
    "dr",
    "mr.",
    "mr",
    "mrs.",
    "mrs",
    "ms.",
    "ms",
    "prof.",
    "prof",
    "professor",
    "rev.",
    "rev",
    "reverend",
    "fr.",
    "fr",
    "father",
)

_AUDIO_SUFFIXES = (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".ac3", ".mp4", ".aif")
_LABEL_EXTENSION_MAP = {
    "MP3": ".mp3",
    "WAV": ".wav",
    "AC3": ".ac3",
    "MPEG-4": ".mp4",
    "audio/wav": ".wav",
    "audio/mp3": ".mp3",
    "MP4": ".mp4",
    "AIFF": ".aif",
}


@dataclass(frozen=True)
class MatchResult:
    payload: Dict[str, Any]
    matched_by: str


@dataclass(frozen=True)
class NoteHeader:
    occurrence_ids: Tuple[str, ...]
    object_ids: Tuple[str, ...]
    date_values: Tuple[str, ...]
    source_urls: Tuple[str, ...]
    filenames: Tuple[str, ...]
    file_entries: Tuple[Dict[str, Any], ...]


@dataclass
class _FileEntry:
    occurrence_id: str
    object_id: str
    attr_id: str
    name_root: str
    ext: str
    url: str
    filename: str
    md5: str | None
    size: str | None
    type_label: str | None
    name: str | None


def _require_connection_string() -> str:
    connection_string = os.environ.get(ENV_CONNECTION_STRING)
    if connection_string:
        return connection_string
    raise RuntimeError(f"Environment variable {ENV_CONNECTION_STRING} is not set")


def _blob_service_client() -> BlobServiceClient:
    return BlobServiceClient.from_connection_string(_require_connection_string())


def _run_query(sql: str, params: Optional[Tuple[Any, ...]] = None) -> List[Dict[str, Any]]:
    if boto3 is None or pymysql is None:
        raise ModuleNotFoundError("boto3 and PyMySQL are required to fetch CollectiveAccess rows.")

    db_user = os.getenv("AWS_DB_USER", "svc_trnscrpt")
    db_host = os.getenv(
        "AWS_DB_HOST",
        "collective-access-db.cluster-ro-cuhegju9o0up.us-east-1.rds.amazonaws.com",
    )
    db_port = int(os.getenv("AWS_DB_PORT", 3306))
    db_name = os.getenv("AWS_DB_NAME", "collectiveassets")
    region = os.getenv("AWS_REGION", "us-east-1")
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    session_kwargs: Dict[str, Any] = {"region_name": region}
    if aws_access_key_id:
        session_kwargs["aws_access_key_id"] = aws_access_key_id
    if aws_secret_access_key:
        session_kwargs["aws_secret_access_key"] = aws_secret_access_key

    session = boto3.Session(**session_kwargs)
    token = session.client("rds").generate_db_auth_token(
        DBHostname=db_host,
        Port=db_port,
        DBUsername=db_user,
        Region=region,
    )

    connection = pymysql.connect(
        host=db_host,
        user=db_user,
        password=token,
        database=db_name,
        port=db_port,
        ssl_ca=str(DEV_CA_CERT),
        connect_timeout=10,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute("SET SESSION group_concat_max_len = 1000000")
            if params is None:
                cursor.execute(sql)
            else:
                cursor.execute(sql, params)
            rows = cursor.fetchall()
    finally:
        connection.close()
    return rows or []


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _split_displayname(value: str) -> Tuple[str, str]:
    text = (value or "").strip()
    if text.endswith(")") and "(" in text:
        idx = text.rfind("(")
        inner = text[idx + 1 : -1].strip()
        base = text[:idx].strip()
        if base:
            return base, inner
    return text, ""


def _strip_name_prefix(value: str) -> str:
    text = " ".join((value or "").split()).strip()
    if not text:
        return ""
    parts = text.split()
    while parts and parts[0].lower() in _NAME_PREFIXES:
        parts = parts[1:]
    return " ".join(parts).strip()


def _assemble_name(entry: Dict[str, Any]) -> str:
    parts = [
        _clean_text(entry.get("forename")),
        _clean_text(entry.get("other_forenames")),
        _clean_text(entry.get("middlename")),
        _clean_text(entry.get("surname")),
    ]
    return " ".join(part for part in parts if part)


def _normalize_person_entry(entry: Dict[str, Any]) -> Optional[List[str]]:
    relation = _clean_text(entry.get("type_code") or entry.get("typename_reverse"))
    if relation.lower() == "publisher":
        return None

    base_name = _assemble_name(entry)
    if not base_name:
        displayname = _clean_text(entry.get("displayname"))
        base_name, _comment = _split_displayname(displayname)
    if not base_name and not relation:
        return None
    return [base_name or relation, "", relation]


def _fetch_occurrence_people(rows: Iterable[Dict[str, Any]]) -> Dict[str, List[List[str]]]:
    occurrence_ids = [
        str(row.get("occurrence_id"))
        for row in rows
        if row.get("occurrence_id") is not None
    ]
    if not occurrence_ids:
        return {}

    placeholders = ",".join(["%s"] * len(occurrence_ids))
    sql = f"""
SELECT
    x.occurrence_id,
    x.entity_id,
    labels.displayname,
    labels.forename,
    labels.other_forenames,
    labels.middlename,
    labels.surname,
    labels.prefix,
    labels.suffix,
    rel_labels.typename_reverse,
    rel.type_code
FROM ca_entities_x_occurrences AS x
JOIN ca_entity_labels AS labels
    ON labels.entity_id = x.entity_id
    AND labels.locale_id = 1
    AND labels.is_preferred = 1
LEFT JOIN ca_relationship_types AS rel
    ON rel.type_id = x.type_id
LEFT JOIN ca_relationship_type_labels AS rel_labels
    ON rel_labels.type_id = x.type_id
    AND rel_labels.locale_id = 1
WHERE x.occurrence_id IN ({placeholders})
"""
    entries = _run_query(sql, tuple(occurrence_ids))
    people_map: Dict[str, List[List[str]]] = {}
    for entry in entries or []:
        normalized = _normalize_person_entry(entry)
        if not normalized:
            continue
        occ_id = str(entry.get("occurrence_id")) if entry.get("occurrence_id") is not None else ""
        people = people_map.setdefault(occ_id, [])
        if normalized not in people:
            people.append(normalized)
    return people_map


def _normalize_description(raw: Any) -> str:
    if not raw:
        return ""
    parts = [segment.strip() for segment in str(raw).split(" || ") if segment.strip()]
    return "\n".join(parts)


def _parse_occurrence_date(raw: Any) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    parsed = dateparser.parse(text)
    if not parsed:
        return {"text": text}
    return {
        "text": text,
        "iso": parsed.date().isoformat(),
        "year": parsed.year,
        "month": parsed.month,
        "day": parsed.day,
    }


def _split_name(link: str) -> Tuple[str, str]:
    basename = os.path.basename(link).split("?")[0]
    if not basename:
        return "", ""
    return os.path.splitext(basename)


def _normalize_row(row: Dict[str, Any]) -> _FileEntry | None:
    link = row.get("link")
    if not link:
        return None

    occurrence_id = str(row.get("occurrence_id"))
    object_id = str(row.get("object_id"))
    attr_id_raw = row.get("attr_id")
    attr_id = str(attr_id_raw) if attr_id_raw is not None else ""

    name_root, ext = _split_name(link)
    type_label = row.get("type_label")
    if (not ext) and type_label:
        ext = _LABEL_EXTENSION_MAP.get(type_label, "")
    ext = (ext or "").lower()
    if ext and not ext.startswith("."):
        ext = f".{ext}"

    filename = row.get("filename") or ""
    if not filename:
        if name_root or ext:
            filename = f"{name_root}{ext}"
        else:
            filename = os.path.basename(link).split("?")[0]
    if ext and filename and not filename.lower().endswith(ext):
        filename = f"{filename}{ext}"
    if not filename:
        filename = name_root or f"attr-{attr_id}"

    resolved_root = name_root or (filename.rsplit(".", 1)[0] if filename else f"attr-{attr_id}")
    return _FileEntry(
        occurrence_id=occurrence_id,
        object_id=object_id,
        attr_id=attr_id,
        name_root=resolved_root,
        ext=ext or "",
        url=link,
        filename=filename,
        md5=row.get("md5"),
        size=row.get("size"),
        type_label=type_label,
        name=row.get("name"),
    )


def _prefer_new_file(existing_ext: str, new_ext: str) -> bool:
    existing_ext = (existing_ext or "").lower()
    new_ext = (new_ext or "").lower()
    if existing_ext == ".mp3":
        return False
    if new_ext == ".mp3":
        return True
    return False


def _hydrate_collective_access_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized_rows = rows or []
    people_map = _fetch_occurrence_people(rows)
    for row in normalized_rows:
        occ_id = str(row.get("occurrence_id")) if row.get("occurrence_id") is not None else ""
        row["occurrence_people"] = people_map.get(occ_id, [])
        row["occurrence_description_normalized"] = _normalize_description(row.get("occurrence_descriptions"))
        row["occurrence_program_normalized"] = str(row.get("occurrence_program") or "").strip() or None
        row["occurrence_date_normalized"] = _parse_occurrence_date(row.get("occurrence_date_raw"))
    return normalized_rows


def _build_payloads(rows: Iterable[Dict[str, Any]], *, final_step: str) -> List[Dict[str, Any]]:
    occurrences: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        entry = _normalize_row(row)
        if entry is None:
            continue

        occ = occurrences.setdefault(
            entry.occurrence_id,
            {
                "file_map": {},
                "object_ids": set(),
                "name": None,
                "description": "",
                "program": None,
                "people": [],
                "date_raw": None,
                "date_parts": None,
            },
        )

        if entry.name and not occ["name"]:
            occ["name"] = entry.name

        normalized_description = row.get("occurrence_description_normalized") or ""
        if normalized_description and not occ["description"]:
            occ["description"] = normalized_description

        normalized_program = row.get("occurrence_program_normalized")
        if normalized_program and not occ["program"]:
            occ["program"] = normalized_program

        for person in row.get("occurrence_people", []) or []:
            if person not in occ["people"]:
                occ["people"].append(person)

        if not occ["date_raw"]:
            date_raw = (row.get("occurrence_date_raw") or "").strip()
            if date_raw:
                occ["date_raw"] = date_raw
                occ["date_parts"] = row.get("occurrence_date_normalized")

        key = (entry.object_id, entry.name_root)
        stored = occ["file_map"].get(key)
        if stored is None or _prefer_new_file(stored.ext, entry.ext):
            occ["file_map"][key] = entry

        occ["object_ids"].add(entry.object_id)

    payloads: List[Dict[str, Any]] = []
    for occurrence_id, occ_data in occurrences.items():
        files = list(occ_data["file_map"].values())
        if not files:
            continue

        object_ids = sorted(occ_data["object_ids"])
        primary_object_id = object_ids[0] if object_ids else ""
        sorted_files = [
            entry
            for entry in sorted(files, key=lambda item: item.filename.lower())
            if entry.url
        ]
        if not sorted_files:
            continue

        file_entries = [
            {
                "link": file_entry.url,
                "md5": file_entry.md5,
                "size": file_entry.size,
                "object_id": file_entry.object_id,
                "occurrence_id": file_entry.occurrence_id,
                "attr_id": file_entry.attr_id,
                "filename": file_entry.filename,
            }
            for file_entry in sorted_files
        ]

        description_text = occ_data.get("description", "")
        program = occ_data.get("program")
        people = list(occ_data.get("people") or [])
        date_raw = occ_data.get("date_raw")
        date_parts = occ_data.get("date_parts")
        occurrence_name = (
            occ_data.get("name")
            or next((item.name for item in sorted_files if item.name), None)
            or next((item.name_root for item in sorted_files if item.name_root), None)
            or next(
                (
                    item.filename.rsplit(".", 1)[0]
                    if "." in item.filename
                    else item.filename
                )
                for item in sorted_files
                if item.filename
            )
            or next((item.filename for item in sorted_files if item.filename), None)
        )

        payloads.append(
            {
                "source": "collective_access",
                "source_ref": {
                    "occurrence_id": occurrence_id,
                    "object_id": primary_object_id,
                    "files": file_entries,
                    "name": occurrence_name,
                    "prompt": None,
                    "description": description_text or None,
                    "program": program,
                    "people": people,
                    "date": date_parts or date_raw,
                    "date_raw": date_raw,
                },
                "final_step": final_step,
                "collective_access_people": people,
                "collective_access_description": description_text,
                "collective_access_program": program,
                "collective_access_date": date_parts,
                "collective_access_date_raw": date_raw,
                "collective_access_date_iso": date_parts.get("iso") if isinstance(date_parts, dict) else None,
            }
        )
    return payloads


def _sidecar_rows_sql(occurrence_selector_sql: str) -> str:
    return f"""
SELECT
    occ.occurrence_id AS occurrence_id,
    o.object_id AS object_id,
    vals.attribute_id AS attr_id,
    name_vals.value_longtext1 AS filename,
    vals.value_longtext1 AS link,
    md5_vals.value_longtext1 AS md5,
    size_vals.value_longtext1 AS size,
    type_label.name_singular AS type_label,
    (
        SELECT GROUP_CONCAT(desc_vals.value_longtext1 SEPARATOR ' || ')
        FROM ca_attributes AS desc_attrs
        JOIN ca_attribute_values AS desc_vals
            ON desc_vals.attribute_id = desc_attrs.attribute_id
            AND desc_vals.element_id = 56
        WHERE desc_attrs.row_id = occ.occurrence_id
            AND desc_attrs.table_num = 67
    ) AS occurrence_descriptions,
    (
        SELECT date_vals.value_longtext1
        FROM ca_attributes AS date_attrs
        JOIN ca_attribute_values AS date_vals
            ON date_vals.attribute_id = date_attrs.attribute_id
            AND date_vals.element_id = 175
        WHERE date_attrs.row_id = occ.occurrence_id
            AND date_attrs.table_num = 67
        LIMIT 1
    ) AS occurrence_date_raw,
    (
        SELECT program_vals.value_longtext1
        FROM ca_attributes AS program_attrs
        JOIN ca_attribute_values AS program_vals
            ON program_vals.attribute_id = program_attrs.attribute_id
            AND program_vals.element_id = 137
        WHERE program_attrs.row_id = occ.occurrence_id
            AND program_attrs.table_num = 67
        LIMIT 1
    ) AS occurrence_program,
    labels.name AS name
FROM (
{occurrence_selector_sql.strip()}
) AS occ
JOIN ca_objects_x_occurrences AS x ON occ.occurrence_id = x.occurrence_id
JOIN ca_occurrence_labels AS labels
    ON occ.occurrence_id = labels.occurrence_id
    AND labels.is_preferred = 1
JOIN ca_objects AS o ON x.object_id = o.object_id
JOIN ca_attributes AS attr_vals
    ON o.object_id = attr_vals.row_id AND attr_vals.table_num = 57
JOIN ca_attribute_values AS vals
    ON vals.attribute_id = attr_vals.attribute_id AND vals.element_id = 160
JOIN ca_attribute_values AS md5_vals
    ON md5_vals.attribute_id = attr_vals.attribute_id AND md5_vals.element_id = 162
JOIN ca_attribute_values AS size_vals
    ON size_vals.attribute_id = attr_vals.attribute_id AND size_vals.element_id = 158
LEFT JOIN ca_attributes AS attr_type
    ON o.object_id = attr_type.row_id AND attr_type.table_num = 57 AND attr_type.element_id = 71
LEFT JOIN ca_attribute_values AS type_vals
    ON type_vals.attribute_id = attr_type.attribute_id AND type_vals.element_id = 71
LEFT JOIN ca_attributes AS attr_name
    ON o.object_id = attr_name.row_id AND attr_name.table_num = 57 AND attr_name.element_id = 156
LEFT JOIN ca_attribute_values AS name_vals
    ON name_vals.attribute_id = attr_vals.attribute_id AND name_vals.element_id = 156
LEFT JOIN ca_list_item_labels AS type_label
    ON type_vals.value_longtext1 = type_label.item_id
WHERE vals.value_longtext1 NOT LIKE 'https://en.wikipedia.org%%'
  AND NOT o.deleted
ORDER BY occ.occurrence_id, o.object_id, attr_vals.attribute_id
"""


def _dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _parse_file_note(value: str) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for key, raw_value in _FILE_PART_RE.findall(value):
        item = raw_value.strip()
        if item.startswith('"') and item.endswith('"'):
            item = item[1:-1]
        normalized_key = key.strip()
        if normalized_key in {"duration_seconds"}:
            try:
                parsed[normalized_key] = float(item)
                continue
            except ValueError:
                pass
        parsed[normalized_key] = item
    return parsed


def _notes_from_vtt_text(text: str) -> NoteHeader:
    occurrence_ids: List[str] = []
    object_ids: List[str] = []
    date_values: List[str] = []
    source_urls: List[str] = []
    filenames: List[str] = []
    file_entries: List[Dict[str, Any]] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "-->" in line:
            break
        match = _NOTE_LINE_RE.match(line)
        if not match:
            continue
        note_key = match.group("key").strip()
        value = match.group("value").strip()
        indexed = _INDEXED_KEY_RE.match(note_key)
        base_key = indexed.group("key") if indexed else note_key

        if base_key == "occurrence_id" and value:
            occurrence_ids.append(value)
        elif base_key == "object_id" and value:
            object_ids.append(value)
        elif base_key == "date" and value:
            date_values.append(value)
        elif base_key == "source_url" and value:
            source_urls.append(value)
        elif base_key == "filename" and value:
            filenames.append(value)
        elif base_key == "file" and value:
            entry = _parse_file_note(value)
            if entry:
                file_entries.append(entry)
                source_url = str(entry.get("source_url") or "").strip()
                filename = str(entry.get("filename") or "").strip()
                if source_url:
                    source_urls.append(source_url)
                if filename:
                    filenames.append(filename)

    return NoteHeader(
        occurrence_ids=tuple(_dedupe_preserve_order(occurrence_ids)),
        object_ids=tuple(_dedupe_preserve_order(object_ids)),
        date_values=tuple(_dedupe_preserve_order(date_values)),
        source_urls=tuple(_dedupe_preserve_order(source_urls)),
        filenames=tuple(_dedupe_preserve_order(filenames)),
        file_entries=tuple(file_entries),
    )


def _augment_notes_from_path(notes: NoteHeader, transcript_path: Path) -> NoteHeader:
    if notes.occurrence_ids or notes.object_ids or notes.source_urls or notes.filenames:
        return notes
    stem = transcript_path.stem.strip()
    if not stem:
        return notes
    guessed_filenames = tuple(f"{stem}{suffix}" for suffix in _AUDIO_SUFFIXES)
    return NoteHeader(
        occurrence_ids=notes.occurrence_ids,
        object_ids=notes.object_ids,
        date_values=notes.date_values,
        source_urls=notes.source_urls,
        filenames=guessed_filenames,
        file_entries=notes.file_entries,
    )


def _download_vtts(
    directory: Path,
    *,
    container_name: str = DEFAULT_TRANSCRIPT_CONTAINER,
    limit: Optional[int] = None,
    from_index: int = 0,
) -> List[Path]:
    if from_index < 0:
        raise ValueError("--from must be 0 or greater")
    if limit is not None and limit < 0:
        raise ValueError("--limit must be 0 or greater")

    container = _blob_service_client().get_container_client(container_name)
    blob_names = sorted(
        blob.name
        for blob in container.list_blobs()
        if str(getattr(blob, "name", "")).lower().endswith(".vtt")
    )
    selected = blob_names[from_index:]
    if limit is not None:
        selected = selected[:limit]

    directory.mkdir(parents=True, exist_ok=True)
    downloaded: List[Path] = []
    for blob_name in tqdm(selected):
        payload = container.get_blob_client(blob_name).download_blob().readall()
        destination = directory / Path(blob_name).name
        destination.write_bytes(payload)
        downloaded.append(destination)
    return downloaded


def _fetch_payloads(selector_sql: str, params: Sequence[Any]) -> List[Dict[str, Any]]:
    rows = _run_query(_sidecar_rows_sql(selector_sql), tuple(params))
    hydrated = _hydrate_collective_access_rows(rows)
    return _build_payloads(hydrated, final_step="azure")


def _payloads_for_occurrence_ids(occurrence_ids: Sequence[str]) -> List[Dict[str, Any]]:
    if not occurrence_ids:
        return []
    placeholders = ",".join(["%s"] * len(occurrence_ids))
    selector_sql = f"""
SELECT occurrence_id
FROM ca_occurrences
WHERE deleted = 0
  AND occurrence_id IN ({placeholders})
ORDER BY occurrence_id
"""
    return _fetch_payloads(selector_sql, occurrence_ids)


def _payloads_for_object_ids(object_ids: Sequence[str]) -> List[Dict[str, Any]]:
    if not object_ids:
        return []
    placeholders = ",".join(["%s"] * len(object_ids))
    selector_sql = f"""
SELECT DISTINCT occ.occurrence_id
FROM ca_occurrences AS occ
JOIN ca_objects_x_occurrences AS x ON occ.occurrence_id = x.occurrence_id
WHERE occ.deleted = 0
  AND x.object_id IN ({placeholders})
ORDER BY occ.occurrence_id
"""
    return _fetch_payloads(selector_sql, object_ids)


def _payloads_for_file_refs(source_urls: Sequence[str], filenames: Sequence[str]) -> List[Dict[str, Any]]:
    if not source_urls and not filenames:
        return []
    filters: List[str] = []
    params: List[Any] = []
    if source_urls:
        placeholders = ",".join(["%s"] * len(source_urls))
        filters.append(f"vals.value_longtext1 IN ({placeholders})")
        params.extend(source_urls)
    if filenames:
        placeholders = ",".join(["%s"] * len(filenames))
        filters.append(f"name_vals.value_longtext1 IN ({placeholders})")
        params.extend(filenames)
    selector_sql = f"""
SELECT DISTINCT occ.occurrence_id
FROM ca_occurrences AS occ
JOIN ca_objects_x_occurrences AS x ON occ.occurrence_id = x.occurrence_id
JOIN ca_objects AS o ON x.object_id = o.object_id
JOIN ca_attributes AS attr_vals
    ON o.object_id = attr_vals.row_id AND attr_vals.table_num = 57
JOIN ca_attribute_values AS vals
    ON vals.attribute_id = attr_vals.attribute_id AND vals.element_id = 160
LEFT JOIN ca_attributes AS attr_name
    ON o.object_id = attr_name.row_id AND attr_name.table_num = 57 AND attr_name.element_id = 156
LEFT JOIN ca_attribute_values AS name_vals
    ON name_vals.attribute_id = attr_vals.attribute_id AND name_vals.element_id = 156
WHERE occ.deleted = 0
  AND NOT o.deleted
  AND ({' OR '.join(filters)})
ORDER BY occ.occurrence_id
"""
    return _fetch_payloads(selector_sql, params)


def _score_payload(payload: Dict[str, Any], notes: NoteHeader) -> int:
    source_ref = payload.get("source_ref") if isinstance(payload.get("source_ref"), dict) else {}
    if not isinstance(source_ref, dict):
        source_ref = {}
    score = 0
    occurrence_id = str(source_ref.get("occurrence_id") or "").strip()
    object_id = str(source_ref.get("object_id") or "").strip()
    if occurrence_id and occurrence_id in notes.occurrence_ids:
        score += 1000
    if object_id and object_id in notes.object_ids:
        score += 500

    files = source_ref.get("files") or []
    for entry in files:
        if not isinstance(entry, dict):
            continue
        source_url = str(entry.get("link") or entry.get("source_url") or "").strip()
        filename = str(entry.get("filename") or "").strip()
        attr_id = str(entry.get("attr_id") or "").strip()
        if source_url and source_url in notes.source_urls:
            score += 100
        if filename and filename in notes.filenames:
            score += 50
        if attr_id and any(str(item.get("attr_id") or "").strip() == attr_id for item in notes.file_entries):
            score += 25
    return score


def _select_payload(notes: NoteHeader) -> MatchResult:
    payloads_with_reason: List[Tuple[str, Dict[str, Any]]] = []
    if notes.occurrence_ids:
        payloads_with_reason.extend(
            ("occurrence_id", payload)
            for payload in _payloads_for_occurrence_ids(notes.occurrence_ids)
        )
    if not payloads_with_reason and notes.object_ids:
        payloads_with_reason.extend(
            ("object_id", payload)
            for payload in _payloads_for_object_ids(notes.object_ids)
        )
    if not payloads_with_reason and (notes.source_urls or notes.filenames):
        payloads_with_reason.extend(
            ("file_reference", payload)
            for payload in _payloads_for_file_refs(notes.source_urls, notes.filenames)
        )

    if not payloads_with_reason:
        raise ValueError("No CollectiveAccess match found from VTT NOTE metadata.")

    ranked = sorted(
        payloads_with_reason,
        key=lambda item: (
            _score_payload(item[1], notes),
            str((item[1].get("source_ref") or {}).get("occurrence_id") or ""),
        ),
        reverse=True,
    )
    matched_by, payload = ranked[0]
    return MatchResult(payload=payload, matched_by=matched_by)


def _parse_datetime_to_utc(value: str) -> Optional[datetime]:
    parsed = dateparser.parse(value)
    if not parsed:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _recording_date(notes: NoteHeader, payload: Dict[str, Any]) -> Optional[str]:
    candidates = list(notes.date_values)
    source_ref = payload.get("source_ref") if isinstance(payload.get("source_ref"), dict) else {}
    if isinstance(source_ref, dict):
        date_value = source_ref.get("date")
        date_raw = source_ref.get("date_raw")
        if isinstance(date_value, dict):
            text = str(date_value.get("text") or "").strip()
            if text:
                candidates.append(text)
        elif isinstance(date_value, str) and date_value.strip():
            candidates.append(date_value.strip())
        if isinstance(date_raw, str) and date_raw.strip():
            candidates.append(date_raw.strip())

    best: Optional[datetime] = None
    best_score = -1
    for value in candidates:
        parsed = _parse_datetime_to_utc(value)
        if parsed is None:
            continue
        score = 1
        lowered = value.lower()
        if any(token in lowered for token in ("am", "pm", ":")):
            score = 2
        if score > best_score:
            best = parsed
            best_score = score
    if best is None:
        return None
    return best.isoformat().replace("+00:00", "Z")


def _filter_guest_names(people: Sequence[Sequence[str]]) -> List[str]:
    filtered: List[str] = []
    for entry in people:
        if not entry:
            continue
        name = str(entry[0] if len(entry) > 0 else "").strip()
        if not name:
            continue
        if name.lower().startswith("unknown"):
            continue
        if name not in filtered:
            filtered.append(name)
    return filtered


def _extract_vtt_guest_labels(vtt_text: str) -> List[str]:
    guests: List[str] = []
    for match in _VOICE_TAG_RE.finditer(vtt_text or ""):
        raw_label = " ".join(match.group(1).split()).strip()
        if not raw_label:
            continue
        name, _comment = _split_displayname(raw_label)
        normalized = _strip_name_prefix((name or raw_label).strip())
        lowered = normalized.lower()
        if not normalized or lowered.startswith("unknown") or lowered.startswith("speaker_"):
            continue
        if normalized not in guests:
            guests.append(normalized)
    return guests


def _matching_note_file(file_entry: Dict[str, Any], notes: NoteHeader) -> Optional[Dict[str, Any]]:
    attr_id = str(file_entry.get("attr_id") or "").strip()
    filename = str(file_entry.get("filename") or "").strip()
    url = str(file_entry.get("link") or file_entry.get("source_url") or "").strip()
    for note_entry in notes.file_entries:
        if attr_id and attr_id == str(note_entry.get("attr_id") or "").strip():
            return note_entry
        if filename and filename == str(note_entry.get("filename") or "").strip():
            return note_entry
        if url and url == str(note_entry.get("source_url") or "").strip():
            return note_entry
    return None


def _recording_files(payload: Dict[str, Any], notes: NoteHeader) -> List[Dict[str, Any]]:
    source_ref = payload.get("source_ref") if isinstance(payload.get("source_ref"), dict) else {}
    files = list(source_ref.get("files") or []) if isinstance(source_ref, dict) else []
    if not files:
        return []

    if notes.file_entries or notes.source_urls or notes.filenames:
        filtered = []
        for entry in files:
            if not isinstance(entry, dict):
                continue
            filename = str(entry.get("filename") or "").strip()
            source_url = str(entry.get("link") or entry.get("source_url") or "").strip()
            attr_id = str(entry.get("attr_id") or "").strip()
            note_match = _matching_note_file(entry, notes)
            if note_match is not None:
                filtered.append(entry)
                continue
            if filename and filename in notes.filenames:
                filtered.append(entry)
                continue
            if source_url and source_url in notes.source_urls:
                filtered.append(entry)
                continue
            if attr_id and any(str(item.get("attr_id") or "").strip() == attr_id for item in notes.file_entries):
                filtered.append(entry)
        if filtered:
            files = filtered

    recording_urls: List[Dict[str, Any]] = []
    for entry in files:
        if not isinstance(entry, dict):
            continue
        source_url = str(entry.get("link") or entry.get("source_url") or "").strip()
        if not source_url:
            continue
        note_entry = _matching_note_file(entry, notes)
        duration = note_entry.get("duration_seconds") if note_entry is not None else None
        size = (
            note_entry.get("size_bytes")
            if note_entry is not None and note_entry.get("size_bytes") not in (None, "")
            else entry.get("size")
        )
        recording_urls.append({"url": source_url, "length": duration, "size": size})
    return recording_urls


def _program_value(payload: Dict[str, Any]) -> Optional[str]:
    source_ref = payload.get("source_ref")
    if not isinstance(source_ref, dict):
        return None
    for key in ("program", "series", "show"):
        value = source_ref.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def build_sidecar_dict(
    transcript_path: Path,
    payload: Dict[str, Any],
    notes: NoteHeader,
    *,
    matched_by: str,
    transcript_text: str = "",
) -> Dict[str, Any]:
    source_ref = payload.get("source_ref") if isinstance(payload.get("source_ref"), dict) else {}
    if not isinstance(source_ref, dict):
        source_ref = {}
    people = source_ref.get("people") or payload.get("collective_access_people") or []
    title = str(source_ref.get("name") or transcript_path.stem).strip()
    description = str(
        source_ref.get("description") or payload.get("collective_access_description") or ""
    ).strip()

    return {
        "transcript_name": transcript_path.name,
        "recording_urls": _recording_files(payload, notes),
        "title": title,
        "description": description,
        "program": _program_value(payload),
        "recording_date": _recording_date(notes, payload),
        "guests": _extract_vtt_guest_labels(transcript_text),
        "collective_access_metadata": {
            "object_id": str(source_ref.get("object_id") or "").strip() or None,
            "occurrence_id": str(source_ref.get("occurrence_id") or "").strip() or None,
            "matched_by": matched_by,
        },
    }


def build_sidecar_for_vtt(transcript_path: Path) -> Path:
    text = transcript_path.read_text(encoding="utf-8")
    notes = _augment_notes_from_path(_notes_from_vtt_text(text), transcript_path)
    match = _select_payload(notes)
    sidecar = build_sidecar_dict(
        transcript_path,
        match.payload,
        notes,
        matched_by=match.matched_by,
        transcript_text=text,
    )
    output_path = transcript_path.with_suffix(".json")
    output_path.write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def _occurrence_id_from_vtt_path(transcript_path: Path) -> Optional[str]:
    try:
        text = transcript_path.read_text(encoding="utf-8")
    except Exception:
        return None
    notes = _notes_from_vtt_text(text)
    if not notes.occurrence_ids:
        return None
    value = str(notes.occurrence_ids[0]).strip()
    return value or None


def _dedupe_vtt_paths_by_occurrence_id(
    transcript_paths: Sequence[Path],
) -> Tuple[List[Path], List[Tuple[str, Path]]]:
    kept: List[Path] = []
    skipped: List[Tuple[str, Path]] = []
    seen_occurrence_ids: set[str] = set()

    for transcript_path in sorted(transcript_paths):
        occurrence_id = _occurrence_id_from_vtt_path(transcript_path)
        if not occurrence_id:
            kept.append(transcript_path)
            continue
        if occurrence_id in seen_occurrence_ids:
            skipped.append((occurrence_id, transcript_path))
            continue
        seen_occurrence_ids.add(occurrence_id)
        kept.append(transcript_path)

    return kept, skipped


def build_sidecars_for_paths(transcript_paths: Sequence[Path]) -> List[Path]:
    created: List[Path] = []
    for transcript_path in tqdm(sorted(transcript_paths)):
        created.append(build_sidecar_for_vtt(transcript_path))
    return created


def build_sidecars_in_directory(directory: Path) -> List[Path]:
    return build_sidecars_for_paths(sorted(directory.glob("*.vtt")))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create sidecar JSON files for VTTs in a directory."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory to scan for .vtt files. Defaults to the current directory.",
    )
    parser.add_argument(
        "--get-vtts",
        action="store_true",
        help="Download .vtt files from Azure blob storage before generating sidecars.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of VTT blobs to download when --get-vtts is used.",
    )
    parser.add_argument(
        "--from",
        dest="from_index",
        type=int,
        default=0,
        help="Zero-based starting offset into the sorted VTT blob list when --get-vtts is used.",
    )
    args = parser.parse_args(argv)

    directory = Path(args.directory).resolve()
    transcript_paths: List[Path]
    if args.get_vtts:
        downloaded = _download_vtts(
            directory,
            limit=args.limit,
            from_index=args.from_index,
        )
        if downloaded:
            for path in downloaded:
                print(f"Downloaded {path}")
        else:
            print(f"No VTT blobs downloaded into {directory}")
        transcript_paths, skipped = _dedupe_vtt_paths_by_occurrence_id(downloaded)
        for occurrence_id, path in skipped:
            print(f"Skipping duplicate occurrence_id {occurrence_id} for {path}")
    else:
        transcript_paths = sorted(directory.glob("*.vtt"))

    created = build_sidecars_for_paths(transcript_paths)
    if not created:
        print(f"No .vtt files found in {directory}")
        return 0

    for path in created:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
