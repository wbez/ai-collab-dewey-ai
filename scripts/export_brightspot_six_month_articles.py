#!/usr/bin/env python3
"""Export Brightspot articles in six-month windows, preserving site ids."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv


DEFAULT_OUTPUT_DIR = Path("data") / "brightspot_six_month_exports"
PROD_HIGHER_ENDPOINT = "https://cms.chicago.suntimes.com/graphql/management/all-articles-read-only-higher"
PROD_FALLBACK_ENDPOINT = "https://cms.chicago.suntimes.com/graphql/management/all-content-read-only"
STAGING_ENDPOINT = "https://debug:926698f2b6f59fc0ebf9c570b0ddeb64@cms.cst-uat.lower.chorus.brightspot.cloud/graphql/management/horoscope-upload-api"
URL_RE = re.compile(r"^https?://", re.IGNORECASE)
SITE_ID_TO_NAME = {
    "0000018e-3d78-db4c-a5ae-bd7bedb70000": "WBEZ",
    "0000017e-bbdb-d3b1-a7fe-fbdff9ca0000": "Chicago Sun-Times",
    "0000017e-bbde-d3b1-a7fe-fbdeb6f20000": "The Straight Dope",
    "0000019d-8de8-d560-a79d-efec28550000": "Vocalo",
    "0000018f-2b50-d106-adcf-af5c41950000": "STNG Wire",
}


class ConfigurationError(Exception):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Brightspot articles in six-month JSON windows starting two years ago."
    )
    parser.add_argument(
        "--months-back",
        type=int,
        default=24,
        help="How far back from now to start exporting. Defaults to 24 months.",
    )
    parser.add_argument(
        "--interval-months",
        type=int,
        default=6,
        help="How wide each export window should be. Defaults to 6 months.",
    )
    parser.add_argument("--page-size", type=int, default=200, help="GraphQL page size for pagination.")
    parser.add_argument(
        "--env",
        choices=("prod", "staging"),
        default="prod",
        help="Which Brightspot environment to use when --graphql-url is omitted.",
    )
    parser.add_argument("--graphql-url", help="Override the Brightspot GraphQL management endpoint.")
    parser.add_argument("--api-key", help="Override the Brightspot X-API-Key directly.")
    parser.add_argument(
        "--api-key-env",
        default="BRIGHTSPOT_API_KEY",
        help="Environment variable to read when --graphql-url is provided without --api-key.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for the exported windowed JSON files. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--unknown-author-label",
        default="Unknown",
        help="Fallback author label when Brightspot does not return byline names.",
    )
    args = parser.parse_args()
    if args.months_back < 0:
        parser.error("--months-back must be greater than or equal to 0")
    if args.interval_months <= 0:
        parser.error("--interval-months must be greater than 0")
    if args.page_size <= 0:
        parser.error("--page-size must be greater than 0")
    return args


def load_environment() -> None:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def resolve_endpoint_and_key(args: argparse.Namespace) -> Tuple[str, str]:
    if args.graphql_url:
        api_key = args.api_key or os.getenv(args.api_key_env)
        if not api_key:
            raise ConfigurationError(f"Missing API key. Provide --api-key or set {args.api_key_env}.")
        return args.graphql_url, api_key

    if args.env == "prod":
        api_key = (
            args.api_key
            or os.getenv("HIGHER_MAX_KEY_PROD")
            or os.getenv("READ_ALL_KEY_PROD")
            or os.getenv("BRIGHTSPOT_API_KEY")
        )
        if not api_key:
            raise ConfigurationError(
                "Missing Brightspot production API key. Set HIGHER_MAX_KEY_PROD, READ_ALL_KEY_PROD, or BRIGHTSPOT_API_KEY."
            )
        endpoint = PROD_HIGHER_ENDPOINT if os.getenv("HIGHER_MAX_KEY_PROD") else PROD_FALLBACK_ENDPOINT
        return endpoint, api_key

    api_key = args.api_key or os.getenv("READ_ALL_KEY") or os.getenv("BRIGHTSPOT_API_KEY")
    if not api_key:
        raise ConfigurationError("Missing Brightspot staging API key. Set READ_ALL_KEY or BRIGHTSPOT_API_KEY.")
    return STAGING_ENDPOINT, api_key


def graphql_post(
    session: requests.Session,
    endpoint: str,
    api_key: str,
    query: str,
    timeout: int,
) -> Dict[str, Any]:
    response = session.post(
        endpoint,
        headers={"X-API-Key": api_key},
        json={"query": query},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError(json.dumps(payload["errors"], indent=2))
    return payload.get("data") or {}


def build_articles_query(page_size: int, offset: int, min_publish_ms: int, max_publish_ms: Optional[int]) -> str:
    predicate = f"cms.content.publishDate >= {min_publish_ms}"
    if max_publish_ms is not None:
        predicate = f"{predicate} and cms.content.publishDate < {max_publish_ms}"
    return f"""
    query {{
        brightspot_article_ArticleQuery(
            limit: {page_size},
            offset: {offset},
            where: {{ predicate: {json.dumps(predicate)} }}
        ) {{
            items {{
                _id
                headline
                subheadline
                body
                brightspot_author_HasAuthorsData {{
                    getAuthorNames
                    getAuthors {{ _id }}
                }}
                brightspot_rss_feed_RssFeedItemData {{
                    getRssFeedItemGuid
                }}
                _globals {{
                    com_psddev_cms_db_Content_ObjectModification {{
                        publishDate
                    }}
                    com_psddev_cms_db_Site_ObjectModification {{
                        owner {{
                            _id
                        }}
                    }}
                }}
            }}
        }}
    }}
    """


def build_author_query(author_id: str) -> str:
    return f"""
    query {{
        brightspot_author_PersonAuthorQuery(id: {json.dumps(author_id)}) {{
            items {{
                _id
                name
            }}
        }}
    }}
    """


def fetch_articles(
    session: requests.Session,
    endpoint: str,
    api_key: str,
    timeout: int,
    page_size: int,
    min_publish_ms: int,
    max_publish_ms: Optional[int],
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    offset = 0
    while True:
        data = graphql_post(
            session=session,
            endpoint=endpoint,
            api_key=api_key,
            query=build_articles_query(page_size, offset, min_publish_ms, max_publish_ms),
            timeout=timeout,
        )
        page_items = data.get("brightspot_article_ArticleQuery", {}).get("items") or []
        if not page_items:
            break
        items.extend(page_items)
        print(f"Fetched {len(page_items)} article(s) at offset {offset}.")
        if len(page_items) < page_size:
            break
        offset += page_size
    return items


def get_author_names(author_data: Optional[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    for name in (author_data or {}).get("getAuthorNames") or []:
        if isinstance(name, str):
            cleaned = name.strip()
            if cleaned:
                names.append(cleaned)
    return names


def fill_missing_authors(
    articles: List[Dict[str, Any]],
    session: requests.Session,
    endpoint: str,
    api_key: str,
    timeout: int,
) -> Dict[str, str]:
    missing_ids = set()
    for article in articles:
        author_data = article.get("brightspot_author_HasAuthorsData") or {}
        if get_author_names(author_data):
            continue
        for author in author_data.get("getAuthors") or []:
            author_id = author.get("_id")
            if author_id:
                missing_ids.add(author_id)

    resolved: Dict[str, str] = {}
    for author_id in sorted(missing_ids):
        data = graphql_post(
            session=session,
            endpoint=endpoint,
            api_key=api_key,
            query=build_author_query(author_id),
            timeout=timeout,
        )
        items = data.get("brightspot_author_PersonAuthorQuery", {}).get("items") or []
        if items and items[0].get("name"):
            resolved[author_id] = items[0]["name"].strip()
    return resolved


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


def isoformat_from_millis(value: Any) -> Optional[str]:
    if value is None:
        return None
    timestamp = int(value) / 1000
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def resolve_url(article: Dict[str, Any]) -> Optional[str]:
    rss_guid = (article.get("brightspot_rss_feed_RssFeedItemData") or {}).get("getRssFeedItemGuid")
    if isinstance(rss_guid, str) and URL_RE.match(rss_guid.strip()):
        return rss_guid.strip()
    return None


def resolve_site_id(article: Dict[str, Any]) -> Optional[str]:
    globals_data = article.get("_globals") or {}
    site_data = globals_data.get("com_psddev_cms_db_Site_ObjectModification") or {}
    owner = site_data.get("owner") or {}
    site_id = owner.get("_id")
    return str(site_id).strip() if site_id else None


def resolve_site_name(site_id: Optional[str]) -> Optional[str]:
    if not site_id:
        return None
    return SITE_ID_TO_NAME.get(site_id)




def normalize_article(
    article: Dict[str, Any], author_lookup: Dict[str, str], unknown_author_label: str
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    headline = (article.get("headline") or "").strip()
    subheadline = strip_html(article.get("subheadline"))
    body = strip_html(article.get("body"))
    publish_date_ms = (
        (article.get("_globals") or {})
        .get("com_psddev_cms_db_Content_ObjectModification", {})
        .get("publishDate")
    )
    url = resolve_url(article)
    site_id = resolve_site_id(article)
    site_name = resolve_site_name(site_id)

    author_data = article.get("brightspot_author_HasAuthorsData") or {}
    authors = get_author_names(author_data)
    if not authors:
        for author in author_data.get("getAuthors") or []:
            author_id = author.get("_id")
            if author_id and author_lookup.get(author_id):
                authors.append(author_lookup[author_id])
    if not authors:
        authors = [unknown_author_label]

    content_parts = [part for part in (headline, subheadline, body) if part]
    content = "\n\n".join(content_parts)

    if not headline:
        return None, "missing headline"
    if not content:
        return None, "missing content"
    if not url:
        return None, "missing url"
    if not publish_date_ms:
        return None, "missing publish_date"

    publish_date_ms_int = int(publish_date_ms)
    return {
        "id": article.get("_id"),
        "headline": headline,
        "content": content,
        "url": url,
        "authors": authors,
        "publish_date": isoformat_from_millis(publish_date_ms_int),
        "site_id": site_id,
        "site_name": site_name,
        "_publish_date_ms": publish_date_ms_int,
    }, None


def sort_key(item: Dict[str, Any]) -> Tuple[int, str]:
    return int(item.get("_publish_date_ms") or 0), str(item.get("id") or "")


def serialize_article(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": item["id"],
        "headline": item["headline"],
        "content": item["content"],
        "url": item["url"],
        "authors": item["authors"],
        "publish_date": item["publish_date"],
        "site_id": item["site_id"],
        "site_name": item["site_name"],
    }


def write_article_list(output_path: Path, items: List[Dict[str, Any]]) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([serialize_article(item) for item in items], ensure_ascii=False, indent=2) + "\n")
    return len(items)


def main() -> None:
    args = parse_args()
    load_environment()
    try:
        endpoint, api_key = resolve_endpoint_and_key(args)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}")
        raise SystemExit(1) from exc

    now = datetime.now(timezone.utc)
    cursor = now - relativedelta(months=args.months_back)

    session = requests.Session()
    try:
        while True:
            window_end = cursor
            window_start = cursor - relativedelta(months=args.interval_months)
            min_publish_ms = int(window_start.timestamp() * 1000)
            max_publish_ms = int(window_end.timestamp() * 1000)
            raw_articles = fetch_articles(
                session=session,
                endpoint=endpoint,
                api_key=api_key,
                timeout=args.timeout,
                page_size=args.page_size,
                min_publish_ms=min_publish_ms,
                max_publish_ms=max_publish_ms,
            )
            if not raw_articles:
                print(
                    f"No articles found for window {window_start.date()} to {window_end.date()}; stopping."
                )
                break

            author_lookup = fill_missing_authors(
                articles=raw_articles,
                session=session,
                endpoint=endpoint,
                api_key=api_key,
                timeout=args.timeout,
            )

            exported = []
            skipped = []
            for article in raw_articles:
                normalized, skip_reason = normalize_article(article, author_lookup, args.unknown_author_label)
                if normalized:
                    exported.append(normalized)
                else:
                    skipped.append((article.get("_id") or "unknown", skip_reason or "unknown"))

            exported.sort(key=sort_key)
            window_name = f"{window_start:%Y%m%d}_{window_end:%Y%m%d}"
            output_path = args.output_dir / f"brightspot_articles_{window_name}.json"
            written = write_article_list(output_path, exported)
            site_pairs = sorted({(item["site_id"], item["site_name"]) for item in exported if item.get("site_id")})
            print(
                f"Wrote {written} article(s) to {output_path.resolve()} "
                f"for window {window_start.date()} to {window_end.date()}."
            )
            if site_pairs:
                print(
                    "  Sites: "
                    + ", ".join(
                        f"{site_name or 'Unknown'} ({site_id})" for site_id, site_name in site_pairs
                    )
                )
            if skipped:
                print("  Skipped articles:")
                for item_id, reason in skipped:
                    print(f"    - {item_id}: {reason}")

            cursor = window_start
    except requests.RequestException as exc:
        print(f"Network error while talking to Brightspot: {exc}")
        raise SystemExit(1) from exc
    except Exception as exc:
        print(f"Brightspot query failed: {exc}")
        raise SystemExit(1) from exc
    finally:
        session.close()


if __name__ == "__main__":
    main()
