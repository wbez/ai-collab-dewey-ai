#!/usr/bin/env python3
"""Export de-duplicated contentdb_v2 articles by year."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import psycopg2
from psycopg2.extras import DictCursor


DEFAULT_OUTPUT_DIR = Path("data") / "dewey-data"
DEFAULT_BASE_URL = "https://chicago.suntimes.com"
URL_RE = re.compile(r"^https?://", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export de-duplicated contentdb_v2 articles into Dewey JSON files grouped by year."
        )
    )
    parser.add_argument("--database-url", help="Optional libpq connection string or DSN.")
    parser.add_argument("--db-name", default="contentdb_v2", help="Postgres database name.")
    parser.add_argument("--db-user", default="cpm_readonly", help="Postgres user.")
    parser.add_argument("--db-host", help="Postgres host. Omit to use the local socket.")
    parser.add_argument("--db-port", type=int, help="Postgres port.")
    parser.add_argument("--db-password", help="Postgres password.")
    parser.add_argument(
        "--db-password-env",
        default="PGPASSWORD",
        help="Environment variable to read for the password when --db-password is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where per-year JSON files will be written.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL used when rss_guid is missing but a slug exists.",
    )
    parser.add_argument(
        "--unknown-author-label",
        default="Unknown",
        help="Fallback author label when no author name can be resolved.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        help="Only export articles with derived publish years greater than or equal to this value.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        help="Only export articles with derived publish years less than or equal to this value.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional cap on exported articles, useful for smoke tests.",
    )
    parser.add_argument(
        "--fetch-size",
        type=int,
        default=2000,
        help="Server-side cursor batch size.",
    )
    args = parser.parse_args()
    if args.start_year and args.end_year and args.start_year > args.end_year:
        parser.error("--start-year must be less than or equal to --end-year")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than 0")
    if args.fetch_size <= 0:
        parser.error("--fetch-size must be greater than 0")
    return args


def connect(args: argparse.Namespace):
    if args.database_url:
        return psycopg2.connect(args.database_url)

    kwargs: Dict[str, Any] = {
        "dbname": args.db_name,
        "user": args.db_user,
    }
    password = args.db_password or os.getenv(args.db_password_env)
    if password:
        kwargs["password"] = password
    if args.db_host:
        kwargs["host"] = args.db_host
    if args.db_port:
        kwargs["port"] = args.db_port
    return psycopg2.connect(**kwargs)


def strip_html(raw_html: Optional[str]) -> str:
    text = raw_html or ""
    text = re.sub(r"(?is)<bsp-[^>\s/]+\b[^>]*>.*?</bsp-[^>]+>", " ", text)
    text = re.sub(r"(?is)<bsp-[^>]+/?>", " ", text)
    text = re.sub(r"(?is)<(script|style)\b[^>]*>.*?</\1>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(p|div|section|article|blockquote|h[1-6]|tr)>", "\n", text)
    text = re.sub(r"(?i)<li\b[^>]*>", "\n- ", text)
    text = re.sub(r"(?i)</(ul|ol)>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(value: Optional[str]) -> str:
    return WHITESPACE_RE.sub(" ", (value or "").strip()).strip()


def clean_block_text(value: Optional[str]) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def isoformat_utc(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def resolve_publish_datetime(row: Dict[str, Any]) -> datetime:
    return (
        row["original_published_date"]
        or row["publish_date"]
        or row["create_date"]
        or row["observed_at"]
    )


def load_source_payload(source_payload_json: Optional[str]) -> Dict[str, Any]:
    if not source_payload_json:
        return {}
    try:
        parsed = json.loads(source_payload_json)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def resolve_authors(row: Dict[str, Any], source_payload: Dict[str, Any], unknown_author_label: str) -> list[str]:
    joined_authors = [
        clean_text(author_name)
        for author_name in (row["authors"] or [])
        if clean_text(author_name)
    ]
    if joined_authors:
        return joined_authors

    payload_authors = (
        source_payload.get("brightspot_author_HasAuthorsWithFieldData", {}).get("authors") or []
    )
    resolved_payload_authors = []
    for author in payload_authors:
        if not isinstance(author, dict):
            continue
        name = clean_text(author.get("name"))
        if name:
            resolved_payload_authors.append(name)
    if resolved_payload_authors:
        return resolved_payload_authors

    return [unknown_author_label]


def resolve_content(row: Dict[str, Any]) -> str:
    subheadline = clean_block_text(row["subheadline_text"]) or strip_html(row["subheadline_raw_html"])
    body = clean_block_text(row["body_text"]) or strip_html(row["body_raw_html"])
    content = "\n\n".join(part for part in (subheadline, body) if part)
    if content:
        return content
    return clean_text(row["headline"])


def resolve_url(row: Dict[str, Any], base_url: str) -> Optional[str]:
    rss_guid = clean_text(row["rss_guid"])
    if rss_guid and URL_RE.match(rss_guid):
        return rss_guid

    slug = clean_text(row["slug"])
    if slug:
        return f"{base_url.rstrip('/')}/{slug.lstrip('/')}"

    return None


class YearWriter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._current_year: Optional[int] = None
        self._current_path: Optional[Path] = None
        self._handle = None
        self._items_in_file = 0
        self.counts: Dict[int, int] = {}

    def write(self, year: int, article: Dict[str, Any]) -> None:
        if self._current_year != year:
            self._open_year(year)

        assert self._handle is not None
        if self._items_in_file:
            self._handle.write(",\n")
        else:
            self._handle.write("[\n")
        self._handle.write(json.dumps(article, ensure_ascii=False, indent=2))
        self._items_in_file += 1
        self.counts[year] = self.counts.get(year, 0) + 1

    def close(self) -> None:
        if self._handle is None:
            return
        if self._items_in_file:
            self._handle.write("\n]\n")
        else:
            self._handle.write("[]\n")
        self._handle.close()
        self._handle = None
        self._current_year = None
        self._current_path = None
        self._items_in_file = 0

    def _open_year(self, year: int) -> None:
        self.close()
        self._current_year = year
        self._current_path = self.output_dir / f"{year}.json"
        self._handle = self._current_path.open("w", encoding="utf-8")


def build_query(args: argparse.Namespace) -> str:
    conditions = []
    version_conditions = []
    if args.start_year is not None:
        year = int(args.start_year)
        conditions.append(f"extract(year from fv.year_source) >= {year}")
        version_conditions.append(
            "COALESCE(av.original_published_date, av.publish_date, av.create_date, av.observed_at) "
            f">= make_date({year}, 1, 1)"
        )
    if args.end_year is not None:
        year = int(args.end_year)
        conditions.append(f"extract(year from fv.year_source) <= {year}")
        version_conditions.append(
            "COALESCE(av.original_published_date, av.publish_date, av.create_date, av.observed_at) "
            f"< make_date({year + 1}, 1, 1)"
        )
    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)
    version_where_clause = ""
    if version_conditions:
        version_where_clause = "WHERE " + " AND ".join(version_conditions)

    limit_clause = f"LIMIT {int(args.limit)}" if args.limit is not None else ""

    return f"""
WITH ranked_versions AS (
    SELECT
        av.id,
        av.article_id,
        av.source_id,
        av.root_id,
        av.headline,
        av.subheadline_text,
        av.subheadline_raw_html,
        av.body_text,
        av.body_raw_html,
        av.slug,
        av.rss_guid,
        av.content_hash,
        av.publish_date,
        av.original_published_date,
        av.create_date,
        av.observed_at,
        av.source_payload::text AS source_payload_json,
        COALESCE(av.original_published_date, av.publish_date, av.create_date, av.observed_at) AS publication_sort_date,
        COALESCE(av.original_published_date, av.publish_date, av.create_date, av.observed_at) AS year_source,
        COALESCE(NULLIF(av.source_id, av.article_id), av.article_id) AS canonical_article_key,
        av.content_hash AS content_md5
    FROM article_versions AS av
    {version_where_clause}
),
content_deduped AS (
    SELECT DISTINCT ON (rv.content_md5)
        rv.*
    FROM ranked_versions AS rv
    ORDER BY
        rv.content_md5,
        rv.publication_sort_date ASC,
        rv.observed_at ASC,
        rv.id ASC
),
first_versions AS (
    SELECT DISTINCT ON (cd.canonical_article_key)
        cd.*
    FROM content_deduped AS cd
    ORDER BY
        cd.canonical_article_key,
        cd.publication_sort_date ASC,
        cd.observed_at ASC,
        cd.id ASC
),
author_names AS (
    SELECT
        aa.article_id,
        ARRAY_AGG(a.name ORDER BY aa.ordinal)
            FILTER (WHERE a.name IS NOT NULL AND BTRIM(a.name) <> '') AS authors
    FROM article_authors AS aa
    JOIN authors AS a ON a.id = aa.author_id
    GROUP BY aa.article_id
),
publisher_names AS (
    SELECT
        ar.id AS article_id,
        s.name AS publisher
    FROM articles AS ar
    LEFT JOIN sites AS s ON s.id = ar.site_id
)
SELECT
    fv.article_id AS id,
    fv.headline,
    fv.subheadline_text,
    fv.subheadline_raw_html,
    fv.body_text,
    fv.body_raw_html,
    fv.source_id,
    fv.root_id,
    fv.slug,
    fv.rss_guid,
    fv.publish_date,
    fv.original_published_date,
    fv.create_date,
    fv.observed_at,
    fv.year_source,
    fv.source_payload_json,
    an.authors,
    pn.publisher
FROM first_versions AS fv
LEFT JOIN author_names AS an ON an.article_id = fv.article_id
LEFT JOIN publisher_names AS pn ON pn.article_id = fv.article_id
{where_clause}
ORDER BY
    EXTRACT(YEAR FROM fv.year_source) ASC,
    fv.year_source ASC,
    fv.article_id ASC
{limit_clause}
"""


def row_to_article(row: Dict[str, Any], args: argparse.Namespace) -> Optional[tuple[int, Dict[str, Any]]]:
    headline = clean_text(row["headline"])
    if not headline:
        return None

    publish_datetime = resolve_publish_datetime(row)
    source_payload = load_source_payload(row["source_payload_json"])
    url = resolve_url(row, args.base_url)
    if not url:
        return None

    content = resolve_content(row)
    if not content:
        return None

    article = {
        "id": row["id"],
        "headline": headline,
        "content": content,
        "url": url,
        "authors": resolve_authors(row, source_payload, args.unknown_author_label),
        "publisher": clean_text(row["publisher"]) or None,
        "publish_date": isoformat_utc(publish_datetime),
    }
    return publish_datetime.year, article


def main() -> None:
    args = parse_args()
    query = build_query(args)

    connection = connect(args)
    writer = YearWriter(args.output_dir)
    exported = 0
    skipped_missing_url = 0
    skipped_missing_headline = 0
    skipped_other = 0

    try:
        with connection:
            with connection.cursor(name="contentdb_original_export", cursor_factory=DictCursor) as cursor:
                cursor.itersize = args.fetch_size
                cursor.execute(query)
                for row in cursor:
                    row_dict = dict(row)
                    if not clean_text(row_dict["headline"]):
                        skipped_missing_headline += 1
                        continue
                    converted = row_to_article(row_dict, args)
                    if converted is None:
                        if resolve_url(row_dict, args.base_url) is None:
                            skipped_missing_url += 1
                        else:
                            skipped_other += 1
                        continue
                    year, article = converted
                    writer.write(year, article)
                    exported += 1
    finally:
        writer.close()
        connection.close()

    print(f"Wrote {exported} article(s) across {len(writer.counts)} year file(s) in {args.output_dir.resolve()}.")
    for year in sorted(writer.counts):
        print(f"  {year}: {writer.counts[year]}")
    if skipped_missing_headline:
        print(f"Skipped {skipped_missing_headline} article(s) with no headline.")
    if skipped_missing_url:
        print(f"Skipped {skipped_missing_url} article(s) with no URL or slug fallback.")
    if skipped_other:
        print(f"Skipped {skipped_other} article(s) for other validation reasons.")


if __name__ == "__main__":
    main()
