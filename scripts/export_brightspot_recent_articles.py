import argparse
import hashlib
import html
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import editdistance
import requests
from datasketch import MinHash, MinHashLSH
from dotenv import load_dotenv
from tqdm.auto import tqdm


DEFAULT_OUTPUT_DIR = Path("data")
DEFAULT_OUTPUT_FILENAME = "brightspot_articles.json"
PROD_HIGHER_ENDPOINT = "https://cms.chicago.suntimes.com/graphql/management/all-articles-read-only-higher"
PROD_FALLBACK_ENDPOINT = "https://cms.chicago.suntimes.com/graphql/management/all-content-read-only"
STAGING_ENDPOINT = "https://debug:926698f2b6f59fc0ebf9c570b0ddeb64@cms.cst-uat.lower.chorus.brightspot.cloud/graphql/management/horoscope-upload-api"
URL_RE = re.compile(r"^https?://", re.IGNORECASE)

LSH_CHAR_NGRAM_SIZE = 5
LSH_NUM_PERM = 128
LSH_THRESHOLD = 0.9
LSH_MIN_ESTIMATED_SIMILARITY = 0.9


class ConfigurationError(Exception):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch recent Brightspot articles and write them in Dewey's expected JSON format."
    )
    parser.add_argument("--days", type=int, default=30, help="How many days back to pull.")
    parser.add_argument(
        "--min-days",
        "--minimum-days",
        dest="min_days",
        type=int,
        default=0,
        help="Minimum age in days for articles to pull. Defaults to 0, meaning up to now.",
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
        help=(
            "Directory for the default aggregate article JSON file when --output is omitted. "
            f"Defaults to {DEFAULT_OUTPUT_DIR / DEFAULT_OUTPUT_FILENAME}."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Aggregate JSON file path. Writes the exported articles as one JSON list of objects.",
    )
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--unknown-author-label",
        default="Unknown",
        help="Fallback author label when Brightspot does not return byline names.",
    )
    args = parser.parse_args()
    if args.days < 0:
        parser.error("--days must be greater than or equal to 0")
    if args.min_days < 0:
        parser.error("--min-days must be greater than or equal to 0")
    if args.min_days > args.days:
        parser.error("--min-days must be less than or equal to --days")
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


def build_articles_query(page_size: int, offset: int, min_publish_ms: int, max_publish_ms: Optional[int] = None) -> str:
    predicate = f"cms.content.publishDate >= {min_publish_ms}"
    if max_publish_ms is not None:
        predicate = f"{predicate} and cms.content.publishDate <= {max_publish_ms}"
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
    max_publish_ms: Optional[int] = None,
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
    names = []
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

    resolved = {}
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
    comparison_text = "\n\n".join(part for part in (headline, subheadline, body) if part)
    comparison_md5 = md5_text(comparison_text)
    return {
        "id": article.get("_id"),
        "headline": headline,
        "content": content,
        "url": url,
        "authors": authors,
        "publish_date": isoformat_from_millis(publish_date_ms_int),
        "_comparison_text": comparison_text,
        "_comparison_md5": comparison_md5,
        "_publish_date_ms": publish_date_ms_int,
    }, None


def dedupe_by_id(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: Dict[str, Dict[str, Any]] = {}
    for item in items:
        item_id = item.get("_id")
        if not item_id:
            continue
        existing = deduped.get(item_id)
        current_publish_date = int(
            ((item.get("_globals") or {}).get("com_psddev_cms_db_Content_ObjectModification") or {}).get(
                "publishDate", 0
            )
            or 0
        )
        existing_publish_date = 0
        if existing:
            existing_publish_date = int(
                ((existing.get("_globals") or {}).get("com_psddev_cms_db_Content_ObjectModification") or {}).get(
                    "publishDate", 0
                )
                or 0
            )
        if not existing or current_publish_date > existing_publish_date:
            deduped[item_id] = item
    return list(deduped.values())


def md5_text(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()


def sort_key_earliest(item: Dict[str, Any]) -> Tuple[int, str]:
    return int(item.get("_publish_date_ms") or 0), str(item.get("id") or "")


def sort_key_latest(item: Dict[str, Any]) -> Tuple[int, str]:
    return int(item.get("_publish_date_ms") or 0), str(item.get("id") or "")


def max_similarity_distance(left_length: int, right_length: int) -> int:
    max_length = max(left_length, right_length)
    if max_length == 0:
        return 0
    return max(1, min(20, int(max_length * 0.03)))


def normalize_similarity_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower().strip()


def lsh_shingles(text: str, ngram_size: int) -> Iterable[str]:
    normalized = normalize_similarity_text(text)
    if len(normalized) <= ngram_size:
        yield normalized
        return
    for start in range(len(normalized) - ngram_size + 1):
        yield normalized[start : start + ngram_size]


def build_minhash(text: str, *, num_perm: int = LSH_NUM_PERM) -> MinHash:
    minhash = MinHash(num_perm=num_perm)
    for shingle in lsh_shingles(text, LSH_CHAR_NGRAM_SIZE):
        minhash.update(shingle.encode("utf-8"))
    return minhash


def are_similar_articles(left_text: str, right_text: str) -> bool:
    max_distance = max_similarity_distance(len(left_text), len(right_text))
    if abs(len(left_text) - len(right_text)) > max_distance:
        return False
    return editdistance.eval(left_text, right_text) <= max_distance


def progress_bar(iterable: Iterable[Any], *, total: Optional[int], desc: str, unit: str) -> Iterable[Any]:
    return tqdm(iterable, total=total, desc=desc, unit=unit, disable=not sys.stderr.isatty())


def dedupe_normalized_articles(items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int, int]:
    exact_groups: Dict[str, List[Dict[str, Any]]] = {}
    exact_removed = 0
    for item in progress_bar(items, total=len(items), desc="MD5 dedupe", unit="article"):
        comparison_text = (item.get("_comparison_text") or "").strip()
        group_hash = item.get("_comparison_md5") or md5_text(comparison_text)
        exact_groups.setdefault(group_hash, []).append(item)

    groups = []
    for group in exact_groups.values():
        group.sort(key=sort_key_earliest)
        exact_removed += max(0, len(group) - 1)
        item = group[0]
        text = item.get("_comparison_text") or ""
        groups.append(
            {
                "hash": item.get("_comparison_md5") or md5_text(text),
                "text": text,
                "items": group,
                "length": len(text),
                "minhash": build_minhash(text),
            }
        )

    if len(groups) <= 1:
        return [group["items"][0] for group in groups], exact_removed, 0

    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=LSH_NUM_PERM)
    parent = list(range(len(groups)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left_index: int, right_index: int) -> None:
        left_root = find(left_index)
        right_root = find(right_index)
        if left_root != right_root:
            parent[right_root] = left_root

    for right_index, right_group in enumerate(progress_bar(groups, total=len(groups), desc="Levenshtein dedupe", unit="group")):
        candidate_indices = []
        for candidate_id in lsh.query(right_group["minhash"]):
            left_index = int(candidate_id)
            left_group = groups[left_index]
            if abs(left_group["length"] - right_group["length"]) > max_similarity_distance(
                left_group["length"], right_group["length"]
            ):
                continue
            estimated_similarity = left_group["minhash"].jaccard(right_group["minhash"])
            if estimated_similarity >= LSH_MIN_ESTIMATED_SIMILARITY:
                candidate_indices.append(left_index)
        for left_index in sorted(candidate_indices):
            if are_similar_articles(groups[left_index]["text"], right_group["text"]):
                union(left_index, right_index)
        lsh.insert(str(right_index), right_group["minhash"])

    components: Dict[int, List[int]] = {}
    for index in range(len(groups)):
        components.setdefault(find(index), []).append(index)

    deduped: List[Dict[str, Any]] = []
    similar_removed = 0
    for component_groups in components.values():
        component_items = []
        for index in component_groups:
            component_items.extend(groups[index]["items"])
        chosen = max(component_items, key=sort_key_latest)
        similar_removed += len(component_items) - 1
        deduped.append(chosen)

    deduped.sort(key=sort_key_earliest)
    return deduped, exact_removed, similar_removed - exact_removed


def serialize_article(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": item["id"],
        "headline": item["headline"],
        "content": item["content"],
        "url": item["url"],
        "authors": item["authors"],
        "publish_date": item["publish_date"],
    }


def resolve_output_path(args: argparse.Namespace) -> Path:
    return args.output or args.output_dir / DEFAULT_OUTPUT_FILENAME


def write_article_list(output_path: Path, items: List[Dict[str, Any]]) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([serialize_article(item) for item in items], ensure_ascii=False, indent=2) + "\n"
    )
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
    cutoff = now - timedelta(days=args.days)
    min_publish_ms = int(cutoff.timestamp() * 1000)
    max_publish_ms = int((now - timedelta(days=args.min_days)).timestamp() * 1000)

    session = requests.Session()
    try:
        raw_articles = fetch_articles(
            session=session,
            endpoint=endpoint,
            api_key=api_key,
            timeout=args.timeout,
            page_size=args.page_size,
            min_publish_ms=min_publish_ms,
            max_publish_ms=max_publish_ms,
        )
        raw_articles = dedupe_by_id(raw_articles)
        author_lookup = fill_missing_authors(
            articles=raw_articles,
            session=session,
            endpoint=endpoint,
            api_key=api_key,
            timeout=args.timeout,
        )
    except requests.RequestException as exc:
        print(f"Network error while talking to Brightspot: {exc}")
        raise SystemExit(1) from exc
    except Exception as exc:
        print(f"Brightspot query failed: {exc}")
        raise SystemExit(1) from exc
    finally:
        session.close()

    exported = []
    skipped = []
    for article in raw_articles:
        normalized, skip_reason = normalize_article(article, author_lookup, args.unknown_author_label)
        if normalized:
            exported.append(normalized)
        else:
            skipped.append((article.get("_id") or "unknown", skip_reason or "unknown"))

    deduped_exported, exact_removed, similar_removed = dedupe_normalized_articles(exported)
    output_path = resolve_output_path(args)
    written = write_article_list(output_path, deduped_exported)
    print(f"Wrote {written} deduped article(s) to aggregate JSON at {output_path.resolve()}.")

    print(f"Deduped {exact_removed} exact duplicate(s) and {similar_removed} similar duplicate(s).")
    if skipped:
        print("Skipped articles:")
        for item_id, reason in skipped:
            print(f"  - {item_id}: {reason}")


if __name__ == "__main__":
    main()