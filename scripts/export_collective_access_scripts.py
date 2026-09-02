#!/usr/bin/env python3
"""Export CollectiveAccess occurrence scripts into Dewey search documents."""

from __future__ import annotations

import argparse
import json
from datetime import timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import dateparser

from build_transcript_sidecars import _run_query


DEFAULT_OUTPUT_DIR = Path("data") / "scripts"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export CollectiveAccess occurrence scripts for Dewey search."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def build_query(limit: Optional[int] = None) -> str:
    limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""
    return f"""
SELECT
    occ.occurrence_id,
    labels.name AS title,
    MAX(CASE WHEN values_.element_id = 182 THEN values_.value_longtext1 END) AS content,
    MAX(CASE WHEN values_.element_id = 178 THEN values_.value_longtext1 END) AS url,
    MAX(CASE WHEN values_.element_id = 56 THEN values_.value_longtext1 END) AS description,
    MAX(CASE WHEN values_.element_id = 175 THEN values_.value_longtext1 END) AS recording_date,
    MAX(CASE WHEN values_.element_id = 137 THEN values_.value_longtext1 END) AS program
FROM ca_occurrences AS occ
JOIN ca_occurrence_labels AS labels
    ON labels.occurrence_id = occ.occurrence_id
    AND labels.is_preferred = 1
JOIN ca_attributes AS attrs
    ON attrs.row_id = occ.occurrence_id
    AND attrs.table_num = 67
    AND attrs.element_id IN (56, 137, 175, 178, 182)
JOIN ca_attribute_values AS values_
    ON values_.attribute_id = attrs.attribute_id
    AND values_.element_id IN (56, 137, 175, 178, 182)
WHERE occ.deleted = 0
GROUP BY occ.occurrence_id, labels.name
HAVING content IS NOT NULL AND TRIM(content) <> ''
ORDER BY occ.occurrence_id
{limit_clause}
"""


def document_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    occurrence_id = str(row["occurrence_id"])
    return {
        "id": f"script-{occurrence_id}",
        "occurrence_id": occurrence_id,
        "content_type": "script",
        "title": str(row.get("title") or f"Occurrence {occurrence_id}").strip(),
        "content": str(row["content"]).strip(),
        "url": str(row.get("url") or "").strip() or None,
        "description": str(row.get("description") or "").strip() or None,
        "recording_date": normalize_date(row.get("recording_date")),
        "program": str(row.get("program") or "").strip() or None,
    }


def normalize_date(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    parsed = dateparser.parse(text)
    if not parsed:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than zero")
    rows = _run_query(build_query(args.limit))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        document = document_from_row(row)
        output_path = args.output_dir / f"{document['id']}.json"
        output_path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} script document(s) to {args.output_dir.resolve()}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
