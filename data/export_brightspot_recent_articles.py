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
PROD_HIGHER_ENDPOINT = (
    "https://cms.chicago.suntimes.com/graphql/management/all-articles-read-only-higher"
)
PROD_FALLBACK_ENDPOINT = (
    "https://cms.chicago.suntimes.com/graphql/management/all-content-read-only"
)
STAGING_ENDPOINT = (
    "https://debug:926698f2b6f59fc0ebf9c570b0ddeb64@"
    "cms.cst-uat.lower.chorus.brightspot.cloud/graphql/management/horoscope-upload-api"
)
URL_RE = re.compile(r"^https?://", re.IGNORECASE)


class ConfigurationError(Exception):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch recent Brightspot articles and write them in Dewey's expected JSON format."
        )
    )
    parser.add_argument("--days", type=int, default=30, help="How many days to pull.")
    parser.add_argument(
        "--page-size",
        type=int,
        default=200,
        help="GraphQL page size for pagination.",
    )
    parser.add_argument(
        "--env",
        choices=("prod", "staging"),
        default="prod",
        help="Which Brightspot environment to use when --graphql-url is omitted.",
    )
    parser.add_argument(
        "--graphql-url",
        help="Override the Brightspot GraphQL management endpoint.",
    )
    parser.add_argument(
        "--api-key",
        help="Override the Brightspot X-API-Key directly.",
    )
    parser.add_argument(
        "--api-key-env",
        default="BRIGHTSPOT_API_KEY",
        help="Environment variable to read when --graphql-url is provided without --api-key.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for individual article JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional aggregate JSON file path.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--unknown-author-label",
        default="Unknown",
        help="Fallback author label when Brightspot does not return byline names.",
    )
    return parser.parse_args()


def load_environment() -> None:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def resolve_endpoint_and_key(args: argparse.Namespace) -> Tuple[str, str]:
    if args.graphql_url:
        api_key = args.api_key or os.getenv(args.api_key_env)
        if not api_key:
            raise ConfigurationError(
                f"Missing API key. Provide --api-key or set {args.api_key_env}."
            )
        return args.graphql_url, api_key

    if args.api_key:
        if args.env == "prod":
            return PROD_HIGHER_ENDPOINT, args.api_key
        return STAGING_ENDPOINT, args.api_key

    if args.env == "prod":
        if os.getenv("HIGHER_MAX_KEY_PROD"):
            return PROD_HIGHER_ENDPOINT, os.getenv("HIGHER_MAX_KEY_PROD", "")
        if os.getenv("READ_ALL_KEY_PROD"):
            return PROD_FALLBACK_ENDPOINT, os.getenv("READ_ALL_KEY_PROD", "")
        if os.getenv("BRIGHTSPOT_API_KEY"):
            return PROD_FALLBACK_ENDPOINT, os.getenv("BRIGHTSPOT_API_KEY", "")
        raise ConfigurationError(
            "Missing Brightspot production API key. Set HIGHER_MAX_KEY_PROD, "
            "READ_ALL_KEY_PROD, or BRIGHTSPOT_API_KEY."
        )

    if os.getenv("READ_ALL_KEY"):
        return STAGING_ENDPOINT, os.getenv("READ_ALL_KEY", "")
    if os.getenv("BRIGHTSPOT_API_KEY"):
        return STAGING_ENDPOINT, os.getenv("BRIGHTSPOT_API_KEY", "")
    raise ConfigurationError(
        "Missing Brightspot staging API key. Set READ_ALL_KEY or BRIGHTSPOT_API_KEY."
    )


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
    return payload["data"]


def build_articles_query(page_size: int, offset: int, min_publish_ms: int) -> str:
    predicate = f"cms.content.publishDate >= {min_publish_ms}"
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
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    offset = 0

    while True:
        data = graphql_post(
            session=session,
            endpoint=endpoint,
            api_key=api_key,
            query=build_articles_query(page_size, offset, min_publish_ms),
            timeout=timeout,
        )
        page_items = data["brightspot_article_ArticleQuery"]["items"]
        if not page_items:
            break
        items.extend(page_items)
        print(f"Fetched {len(page_items)} article(s) at offset {offset}.")
        if len(page_items) < page_size:
            break
        offset += page_size

    return items


def get_author_names(author_data: Dict[str, Any]) -> List[str]:
    names = author_data.get("getAuthorNames")
    if isinstance(names, str):
        return [names] if names.strip() else []
    if isinstance(names, list):
        cleaned = [name.strip() for name in names if isinstance(name, str) and name.strip()]
        return cleaned
    return []


def fill_missing_authors(
    articles: Iterable[Dict[str, Any]],
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
        items = data["brightspot_author_PersonAuthorQuery"]["items"]
        if items and items[0].get("name"):
            resolved[author_id] = items[0]["name"].strip()

    return resolved


def strip_html(raw_html: Optional[str]) -> str:
    if not raw_html:
        return ""

    text = re.sub(r"(?is)<bsp-[^>\s/]+\b[^>]*>.*?</bsp-[^>]+>", " ", raw_html)
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


def isoformat_from_millis(value: Any) -> str:
    timestamp = int(value) / 1000
    return (
        datetime.fromtimestamp(timestamp, tz=timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def resolve_url(article: Dict[str, Any]) -> str:
    rss_guid = (
        article.get("brightspot_rss_feed_RssFeedItemData") or {}
    ).get("getRssFeedItemGuid")
    if isinstance(rss_guid, str) and URL_RE.match(rss_guid.strip()):
        return rss_guid.strip()
    return ""


def normalize_article(
    article: Dict[str, Any],
    author_lookup: Dict[str, str],
    unknown_author_label: str,
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
        authors = [
            author_lookup[author["_id"]]
            for author in author_data.get("getAuthors") or []
            if author.get("_id") in author_lookup
        ]
    if not authors and unknown_author_label:
        authors = [unknown_author_label]

    content_parts = [part for part in (subheadline, body) if part]
    content = "\n\n".join(content_parts).strip()

    if not headline:
        return None, "missing headline"
    if not content:
        return None, "missing content"
    if not url:
        return None, "missing url"
    if not publish_date_ms:
        return None, "missing publish_date"

    publish_date_ms_int = int(publish_date_ms)
    comparison_text = "\n\n".join(part for part in (headline, content) if part).strip()
    comparison_md5 = md5_text(comparison_text)

    return (
        {
            "id": article["_id"],
            "headline": headline,
            "content": content,
            "url": url,
            "authors": authors,
            "publish_date": isoformat_from_millis(publish_date_ms_int),
            "_comparison_text": comparison_text,
            "_comparison_md5": comparison_md5,
            "_publish_date_ms": publish_date_ms_int,
        },
        None,
    )


def dedupe_by_id(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: Dict[str, Dict[str, Any]] = {}
    for item in items:
        item_id = item.get("_id")
        if item_id:
            existing = deduped.get(item_id)
            if not existing:
                deduped[item_id] = item
                continue

            existing_publish_date = (
                (existing.get("_globals") or {})
                .get("com_psddev_cms_db_Content_ObjectModification", {})
                .get("publishDate")
                or 0
            )
            current_publish_date = (
                (item.get("_globals") or {})
                .get("com_psddev_cms_db_Content_ObjectModification", {})
                .get("publishDate")
                or 0
            )
            if int(current_publish_date) >= int(existing_publish_date):
                deduped[item_id] = item
    return list(deduped.values())


def md5_text(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()


def sort_key_earliest(item: Dict[str, Any]) -> Tuple[int, str]:
    return (int(item["_publish_date_ms"]), item["id"])


def sort_key_latest(item: Dict[str, Any]) -> Tuple[int, str]:
    return (int(item["_publish_date_ms"]), item["id"])


def max_similarity_distance(left_length: int, right_length: int) -> int:
    max_length = max(left_length, right_length)
    return max(0, (max_length - 1) // 20)


LSH_CHAR_NGRAM_SIZE = 5
LSH_NUM_PERM = 128
LSH_THRESHOLD = 0.9
LSH_MIN_ESTIMATED_SIMILARITY = 0.9


def normalize_similarity_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def lsh_shingles(text: str, ngram_size: int = LSH_CHAR_NGRAM_SIZE) -> List[str]:
    normalized = normalize_similarity_text(text)
    if not normalized:
        return []

    if len(normalized) <= ngram_size:
        return [normalized]

    return [
        normalized[start : start + ngram_size]
        for start in range(len(normalized) - ngram_size + 1)
    ]


def build_minhash(text: str, *, num_perm: int = LSH_NUM_PERM) -> MinHash:
    minhash = MinHash(num_perm=num_perm)
    for shingle in lsh_shingles(text):
        minhash.update(shingle.encode("utf-8"))
    return minhash


def are_similar_articles(left_text: str, right_text: str) -> bool:
    max_distance = max_similarity_distance(len(left_text), len(right_text))
    if max_distance == 0:
        return False
    if abs(len(left_text) - len(right_text)) > max_distance:
        return False
    return editdistance.eval(left_text, right_text) <= max_distance


def progress_bar(
    iterable: Iterable[Any],
    *,
    total: Optional[int] = None,
    desc: str,
    unit: str,
) -> Iterable[Any]:
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        disable=not sys.stderr.isatty(),
    )


def dedupe_normalized_articles(
    items: Iterable[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int, int]:
    items = list(items)
    exact_groups: Dict[str, Dict[str, Any]] = {}
    exact_removed = 0

    for item in progress_bar(
        items,
        total=len(items),
        desc="MD5 dedupe",
        unit="article",
    ):
        comparison_text = item["_comparison_text"].strip()
        group_hash = item.get("_comparison_md5") or md5_text(comparison_text)
        group = exact_groups.get(group_hash)
        if not group:
            exact_groups[group_hash] = {
                "hash": group_hash,
                "text": comparison_text,
                "items": [item],
                "length": len(comparison_text),
                "minhash": build_minhash(comparison_text),
            }
            continue
        group["items"].append(item)
        exact_removed += 1

    groups = list(exact_groups.values())
    groups.sort(key=lambda group: (group["length"], group["text"]))
    parent = list(range(len(groups)))
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=LSH_NUM_PERM)

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

    for right_index, right_group in enumerate(
        progress_bar(
            groups,
            total=len(groups),
            desc="Levenshtein dedupe",
            unit="group",
        )
    ):
        candidate_indices = {
            int(candidate_id)
            for candidate_id in lsh.query(right_group["minhash"])
            if int(candidate_id) < right_index
        }
        for left_index in sorted(candidate_indices):
            left_group = groups[left_index]
            if (
                abs(right_group["length"] - left_group["length"])
                > max_similarity_distance(right_group["length"], left_group["length"])
            ):
                continue
            estimated_similarity = right_group["minhash"].jaccard(left_group["minhash"])
            if estimated_similarity < LSH_MIN_ESTIMATED_SIMILARITY:
                continue
            if are_similar_articles(left_group["text"], right_group["text"]):
                union(left_index, right_index)

        lsh.insert(str(right_index), right_group["minhash"])

    components: Dict[int, List[Dict[str, Any]]] = {}
    for index, group in enumerate(groups):
        components.setdefault(find(index), []).append(group)

    deduped: List[Dict[str, Any]] = []
    similar_removed = 0
    for component_groups in components.values():
        component_items = [
            item
            for group in component_groups
            for item in group["items"]
        ]
        if len(component_groups) == 1:
            chosen = min(component_items, key=sort_key_earliest)
        else:
            chosen = max(component_items, key=sort_key_latest)
            similar_removed += len(component_groups) - 1
        deduped.append(chosen)

    deduped.sort(key=lambda item: (item["_publish_date_ms"], item["id"]))
    return deduped, exact_removed, similar_removed


def serialize_article(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": item["id"],
        "headline": item["headline"],
        "content": item["content"],
        "url": item["url"],
        "authors": item["authors"],
        "publish_date": item["publish_date"],
    }


def write_individual_articles(output_dir: Path, items: Iterable[Dict[str, Any]]) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for item in items:
        article_path = output_dir / f"{item['id']}.json"
        article_path.write_text(
            json.dumps(serialize_article(item), ensure_ascii=False) + "\n"
        )
        written += 1
    return written


def main() -> int:
    args = parse_args()
    load_environment()

    try:
        endpoint, api_key = resolve_endpoint_and_key(args)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}")
        return 1

    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
    min_publish_ms = int(cutoff.timestamp() * 1000)

    session = requests.Session()
    try:
        raw_articles = fetch_articles(
            session=session,
            endpoint=endpoint,
            api_key=api_key,
            timeout=args.timeout,
            page_size=args.page_size,
            min_publish_ms=min_publish_ms,
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
        return 1
    except Exception as exc:
        print(f"Brightspot query failed: {exc}")
        return 1
    finally:
        session.close()

    exported: List[Dict[str, Any]] = []
    skipped: Dict[str, int] = {}
    for article in raw_articles:
        normalized, skip_reason = normalize_article(
            article=article,
            author_lookup=author_lookup,
            unknown_author_label=args.unknown_author_label,
        )
        if normalized:
            exported.append(normalized)
            continue
        skipped[skip_reason or "unknown"] = skipped.get(skip_reason or "unknown", 0) + 1

    deduped_exported, exact_removed, similar_removed = dedupe_normalized_articles(exported)
    written = write_individual_articles(args.output_dir, deduped_exported)

    print(
        f"Wrote {written} article JSON file(s) to {args.output_dir.resolve()}."
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                [serialize_article(item) for item in deduped_exported],
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        print(f"Wrote aggregate JSON to {args.output.resolve()}.")
    print(
        f"Deduped {exact_removed} exact duplicate(s) and {similar_removed} similar duplicate(s)."
    )
    if skipped:
        print("Skipped articles:")
        for reason, count in sorted(skipped.items()):
            print(f"  - {reason}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
