import html
import json
import os, re
from hashlib import sha256
from datetime import datetime, date
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote


SOURCE_MARKER_PATTERN = re.compile(r"\[SRC(\d+)\]")
FOOTNOTE_PATTERN = re.compile(r"\[(?P<num>\d+)(?P<suffix>[a-z])?\]")
WBEZ_WORK_OBJECT_ICON_URL = "https://mchonofsky-test-bucket.s3.us-east-2.amazonaws.com/wbez.jpg"
CST_WORK_OBJECT_ICON_URL = "https://mchonofsky-test-bucket.s3.us-east-2.amazonaws.com/cst.jpg"
# When we previously emitted `<url|[1]>` mrkdwn, normalize back to `[1]` so we
# can render a Work Object mention element.
SLACK_LINKED_FOOTNOTE_PATTERN = re.compile(r"<[^>|]+\|(?P<label>\[(?:\d+)(?:[a-z])?\])>")
HTML_CITATION_PATTERN = re.compile(
    r'<a\s+href="([^"]+)"[^>]*>(\[\d+\])</a>',
    re.IGNORECASE,
)
MARKDOWN_LINK_PATTERN = re.compile(r"(?<!!)\[([^\]\n]+)\]\((https?://[^)\s]+)\)")
MARKDOWN_BOLD_PATTERN = re.compile(r"\*\*([^*\n][\s\S]*?[^*\n])\*\*")
SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def slack_link_url(url: str) -> str:
    return quote(str(url), safe=":/?#[]@!$&'()*+,;=%")


def replace_source_markers(
    text: str,
    source_urls: Dict[int, Optional[str]],
    *,
    link_urls: bool = True,
) -> str:
    display_numbers: Dict[str, int] = {}

    def replace(match):
        source_number = int(match.group(1))
        source_url = source_urls.get(source_number)
        footnote_key = source_url or f"src:{source_number}"
        display_number = display_numbers.setdefault(
            footnote_key,
            len(display_numbers) + 1,
        )
        label = f"[{display_number}]"
        if not source_url or not link_urls:
            return label
        return f"<{slack_link_url(source_url)}|{label}>"

    return SOURCE_MARKER_PATTERN.sub(replace, text)


def html_citations_to_mrkdwn(text: str) -> str:
    def replace(match):
        url = html.unescape(match.group(1))
        label = html.unescape(match.group(2))
        return f"<{slack_link_url(url)}|{label}>"

    return HTML_CITATION_PATTERN.sub(replace, text)


def markdown_to_slack_mrkdwn(text: str) -> str:
    def replace_link(match):
        label = match.group(1)
        url = match.group(2)
        return f"<{slack_link_url(url)}|{label}>"

    text = MARKDOWN_LINK_PATTERN.sub(replace_link, text)
    return MARKDOWN_BOLD_PATTERN.sub(r"*\1*", text)


def truncate_text(text: Any, limit: int) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    if limit <= 3:
        return value[: max(0, limit)].rstrip()
    return value[: max(0, limit - 3)].rstrip() + "..."


def _paragraphize_flat_text(text: str, target_chars: int = 850) -> str:
    sentences = SENTENCE_BOUNDARY_PATTERN.split(text.strip())
    if len(sentences) < 4:
        return text.strip()

    paragraphs: List[str] = []
    current: List[str] = []
    current_len = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        current.append(sentence)
        current_len += len(sentence) + 1
        if current_len >= target_chars:
            paragraphs.append(" ".join(current).strip())
            current = []
            current_len = 0
    if current:
        paragraphs.append(" ".join(current).strip())
    return "\n\n".join(paragraphs)


def work_object_description_text(text: Any, limit: int = 12000) -> str:
    value = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not value:
        return ""
    lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in value.split("\n")]
    value = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    if "\n" not in value and len(value) > 600:
        value = _paragraphize_flat_text(value)
    return truncate_text(value, limit)

def _parse_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    raw = str(value).strip()
    if not raw:
        return None
    # Common case: YYYY-MM-DD
    try:
        return datetime.strptime(raw[:10], "%Y-%m-%d").date()
    except ValueError:
        pass
    # ISO-ish fallback (supports trailing Z)
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def format_human_date(value: Any) -> Optional[str]:
    parsed = _parse_date(value)
    if not parsed:
        return None
    day = str(int(parsed.strftime("%d")))
    return parsed.strftime(f"%B {day}, %Y")


def _bool_env(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _wavelength_static_url(path: str) -> Optional[str]:
    base = (os.environ.get("WAVELENGTH_PUBLIC_BASE_URL") or "").rstrip("/")
    if not base:
        return None
    url = f"{base}/{path.lstrip('/')}"
    version = (os.environ.get("WAVELENGTH_STATIC_VERSION") or "").strip()
    if version:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}v={quote(version, safe='')}"
    return url


def _is_transcript(source: Dict[str, Any]) -> bool:
    return str(source.get("content_type") or "").lower() == "transcript"


def _source_authors(source: Dict[str, Any]) -> List[str]:
    authors = source.get("authors") or source.get("author") or []
    if isinstance(authors, str):
        return [authors.strip()] if authors.strip() else []
    if isinstance(authors, list):
        return [str(item).strip() for item in authors if str(item).strip()]
    return []


def _source_guests(source: Dict[str, Any]) -> List[str]:
    guests = source.get("guests") or []
    if isinstance(guests, str):
        guests = [guests]
    if not isinstance(guests, list):
        return []
    filtered = []
    for guest in guests:
        name = str(guest or "").strip()
        if not name:
            continue
        lower = name.lower()
        if "unknown" in lower:
            continue
        if "speaker_" in lower:
            continue
        filtered.append(name)
    return filtered


def _collectiveaccess_links(source: Dict[str, Any]) -> Optional[str]:
    occurrence_id = str(source.get("occurrence_id") or "").strip()
    object_id = str(
        source.get("object_id")
        or source.get("parent_id")
        or source.get("archive_object_id")
        or ""
    ).strip()
    if not occurrence_id and not object_id:
        return None
    parts = []
    if occurrence_id:
        parts.append(
            "Occurrence: "
            + f"https://archives.wbez.org/index.php/editor/occurrences/OccurrenceEditor/Edit/occurrence_id/{occurrence_id}"
        )
    if object_id:
        parts.append(
            "Object: "
            + f"https://archives.wbez.org/index.php/editor/objects/ObjectEditor/Edit/object_id/{object_id}"
        )
    return " / ".join(parts) if parts else None


def source_title(source: Dict[str, Any]) -> str:
    if source.get("content_type") == "transcript":
        for key in ("program", "program_name", "show", "series", "transcript_name"):
            value = source.get(key)
            if value:
                return str(value)
    for key in ("headline", "title", "transcript_name", "url", "citation_url"):
        value = source.get(key)
        if value:
            return str(value)
    return "Untitled source"


def source_url(source: Dict[str, Any]) -> Optional[str]:
    for key in ("citation_url", "transcript_url", "url"):
        value = source.get(key)
        if value:
            return str(value)
    recording_urls = source.get("recording_urls") or []
    if isinstance(recording_urls, list):
        for value in recording_urls:
            if value:
                return str(value)
    return None


def source_listen_url(source: Dict[str, Any]) -> Optional[str]:
    recording_urls = source.get("recording_urls") or []
    if isinstance(recording_urls, list):
        for value in recording_urls:
            if value:
                return str(value)
    for key in ("recording_url", "citation_url", "transcript_url", "url"):
        if source.get(key):
            return str(source[key])
    return None


def source_excerpt(source: Dict[str, Any], limit: int = 16) -> str:
    content = source.get("chunk_text") if source.get("content_type") == "transcript" else source.get("content")
    if not content and source.get("passages"):
        passages = source.get("passages") or []
        if isinstance(passages, list) and passages and isinstance(passages[0], dict):
            content = passages[0].get("text")
    words = str(content or "").replace("\n", " ").split()
    if not words:
        return ""
    return " ".join(words[:limit]) + "…"


def source_full_text(source: Dict[str, Any]) -> str:
    content = source.get("chunk_text") if source.get("content_type") == "transcript" else source.get("content")
    if content:
        return str(content)
    passages = source.get("passages") or []
    if not isinstance(passages, list):
        return ""
    chunks = []
    for passage in passages:
        if not isinstance(passage, dict):
            continue
        text = passage.get("raw_vtt_excerpt") or passage.get("text")
        if text:
            chunks.append(str(text).strip())
    return "\n\n".join(value for value in chunks if value)


def source_reference(source: Dict[str, Any], index: int) -> Dict[str, Any]:
    publish_date = source.get("publish_date") or source.get("published_at") or source.get("publication_date") or source.get("date")
    if hasattr(publish_date, "isoformat"):
        publish_date = publish_date.isoformat()
    # Prefer IDs that the archive can look up again (used by Work Objects and
    # "Open" actions). In particular, avoid defaulting to a URL-like "source_id"
    # when a stable archive identifier is available.
    stable_id_candidates = [
        source.get("parent_id"),
        source.get("chunk_id"),
        source.get("occurrence_id"),
        source.get("source_id"),
        source.get("id"),
        index,
    ]
    stable_id: str = ""
    for candidate in stable_id_candidates:
        value = str(candidate or "").strip()
        if value:
            stable_id = value
            break
    if not stable_id:
        stable_id = str(index)
    # If the only identifier we have is a URL, convert it to an opaque stable
    # ID. Slack Work Objects expect external_ref.id to uniquely identify the
    # resource without embedding other information (like the full URL).
    if stable_id.startswith("http://") or stable_id.startswith("https://"):
        url = source_url(source) or stable_id
        digest = sha256(url.encode("utf-8", errors="ignore")).hexdigest()[:20]
        stable_id = f"src_{digest}"
    return {
        "id": stable_id,
        "source_id": stable_id,
        "number": index,
        "title": source_title(source),
        "url": source_url(source),
        "listen_url": source_listen_url(source),
        "publish_date": publish_date,
        "content_type": source.get("content_type", "article"),
        "excerpt": source_excerpt(source),
        "full_text": source_full_text(source),
        "authors": source.get("authors") or [],
        "program": source.get("program"),
        "guests": source.get("guests") or [],
        "occurrence_id": source.get("occurrence_id"),
        "parent_id": source.get("parent_id"),
        "passages": source.get("passages") or [],
    }


def source_references(results: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [source_reference(source, index) for index, source in enumerate(results, 1)]


def cited_source_references(answer: str, sources: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return only cited sources, numbered by their first appearance in an answer."""
    by_number = {int(source["number"]): source for source in sources if source.get("number") is not None}
    cited: List[Dict[str, Any]] = []
    displayed_keys = set()
    for match in SOURCE_MARKER_PATTERN.finditer(answer or ""):
        source = by_number.get(int(match.group(1)))
        if not source:
            continue
        key = source.get("url") or f"src:{source['number']}"
        if key in displayed_keys:
            continue
        displayed_keys.add(key)
        cited.append({**source, "number": len(cited) + 1})
    return cited


def split_mrkdwn(text: str, limit: int = 2900) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    remaining = text
    while len(remaining) > limit:
        split_at = remaining.rfind("\n\n", 0, limit)
        if split_at < limit // 2:
            split_at = remaining.rfind(" ", 0, limit)
        if split_at < limit // 2:
            split_at = limit
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks

def _rich_text_elements_for_chunk(
    chunk: str,
    *,
    sources_by_number: Dict[int, Dict[str, Any]],
    work_object_app_id: str,
) -> List[Dict[str, Any]]:
    elements: List[Dict[str, Any]] = []
    cursor = 0
    for match in FOOTNOTE_PATTERN.finditer(chunk or ""):
        start, end = match.span()
        if start > cursor:
            elements.append({"type": "text", "text": chunk[cursor:start]})
        label = match.group(0)
        number = int(match.group("num"))
        source = sources_by_number.get(number)
        if source and source.get("url") and (source.get("source_id") or source.get("id")):
            mention: Dict[str, Any] = {
                "type": "work_object_mention",
                "entity_id": str(source.get("source_id") or source.get("id")),
                "app_id": str(work_object_app_id),
                "text": label,
                "url": slack_link_url(str(source["url"])),
                "icon_url": WBEZ_WORK_OBJECT_ICON_URL if _is_wbez(source) else CST_WORK_OBJECT_ICON_URL 
            }
            elements.append(mention)
        else:
            elements.append({"type": "text", "text": label})
        cursor = end
    if cursor < len(chunk):
        elements.append({"type": "text", "text": chunk[cursor:]})
    return elements

def _is_wbez(source):
    return _is_transcript(source) or "wbez.org" in str(source.get("url") or "")

def answer_blocks(
    answer: str,
    sources: List[Dict[str, Any]],
    *,
    work_object_app_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    sources_by_number = {
        int(source.get("number")): source
        for source in sources
        if isinstance(source, dict)
        and str(source.get("number") or "").isdigit()
    }
    use_work_object_mentions = bool(work_object_app_id) and bool(sources_by_number)
    for chunk in split_mrkdwn(answer):
        if use_work_object_mentions:
            chunk = SLACK_LINKED_FOOTNOTE_PATTERN.sub(lambda m: m.group("label"), chunk)
            blocks.append(
                {
                    "type": "rich_text",
                    "elements": [
                        {
                            "type": "rich_text_section",
                            "elements": _rich_text_elements_for_chunk(
                                chunk,
                                sources_by_number=sources_by_number,
                                work_object_app_id=str(work_object_app_id),
                            ),
                        }
                    ],
                }
            )
        else:
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": chunk}})

    return blocks[:50]

def work_object_entities(
    sources: List[Dict[str, Any]],
    *,
    include_full_text: bool = False,
    include_excerpt: bool = False,
    include_collectiveaccess: bool = False,
) -> List[Dict[str, Any]]:
    """Build Content Item Work Object metadata for cited archive sources."""
    entities = []
    for source in sources:
        external_id = str(source.get("source_id") or source.get("id"))
        url = source.get("url")
        if not external_id or not url:
            continue
        transcript = _is_transcript(source)
        title_text = str(source.get("title") or "Untitled source").strip()
        program_name = str(source.get("program") or "").strip() or title_text
        human_date = format_human_date(source.get("publish_date"))
        if transcript:
            header_title = (program_name + (f" {human_date}" if human_date else "")).strip()
        else:
            header_title = title_text

        date_value = _parse_date(source.get("publish_date"))
        icon_url = WBEZ_WORK_OBJECT_ICON_URL if transcript else CST_WORK_OBJECT_ICON_URL
        product_icon = {"url": icon_url, "alt_text": "WBEZ" if transcript else "CST"} if icon_url else None
        custom_fields: List[Dict[str, Any]] = [
            {
                "key": "source_url",
                "label": "Source",
                "value": url,
                "type": "slack#/types/link",
            }
        ]
        if date_value:
            custom_fields.append(
                {
                    "key": "date",
                    "label": "Date",
                    "value": date_value.strftime("%Y-%m-%d"),
                    "type": "slack#/types/date",
                }
            )
        if transcript:
            guests = _source_guests(source)
            if guests:
                shortlist = guests[:3]
                guests_text = ", ".join(shortlist) + ("..." if len(guests) > 3 else "")
                custom_fields.append(
                    {
                        "key": "guests",
                        "label": "Guests",
                        "value": guests_text,
                        "type": "string",
                    }
                )
            if include_collectiveaccess:
                ca_links = _collectiveaccess_links(source)
                if ca_links:
                    custom_fields.append(
                        {
                            "key": "collectiveaccess",
                            "label": "CollectiveAccess",
                            "value": ca_links,
                            "type": "string",
                        }
                    )
        else:
            authors = _source_authors(source)
            if authors:
                shortlist = authors[:3]
                author_text = ", ".join(shortlist) + ("..." if len(authors) > 3 else "")
                custom_fields.append(
                    {
                        "key": "author",
                        "label": "Author",
                        "value": author_text,
                        "type": "string",
                    }
                )
        if include_excerpt:
            excerpt_value = truncate_text(source.get("excerpt") or "", 200)
            if excerpt_value:
                custom_fields.append(
                    {
                        "key": "excerpt",
                        "label": "Excerpt",
                        "value": excerpt_value,
                        "type": "string",
                    }
                )
        if include_full_text and source.get("full_text"):
            body_text = work_object_description_text(source.get("full_text"))
        else:
            body_text = ""
        display_order = []
        if body_text:
            display_order.append("description")
        display_order.extend(["source_url", "date", "author" if not transcript else "guests"])
        if transcript and include_collectiveaccess:
            display_order.append("collectiveaccess")
        if include_excerpt:
            display_order.append("excerpt")

        entities.append({
            "entity_type": "slack#/entities/content_item",
            "external_ref": {"id": external_id},
            "app_unfurl_url": url,
            "url": url,
            "entity_payload": {
                "attributes": {
                    "title": {"text": header_title},
                    "display_type": "Transcript" if transcript else "Article",
                    "display_id": str(source.get("number", "")),
                    **({"product_icon": product_icon} if product_icon else {}),
                },
                "fields": {
                    **({
                        "description": {
                            "value": body_text,
                            "format": "markdown",
                        }
                    } if body_text else {}),
                },
                "custom_fields": custom_fields,
                "display_order": display_order,
            },
        })
    return entities


def search_blocks(question: str, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "Wavelength archive results"},
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Search*\n{truncate_text(question, 500)}"},
        },
    ]
    if not sources:
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "No matching archive sources were found."},
            }
        )
        return blocks

    for source in sources[:10]:
        title = truncate_text(source["title"], 120)
        if source.get("url"):
            title = f"<{slack_link_url(source['url'])}|{title}>"
        fields = [
            {"type": "mrkdwn", "text": f"*Source*\n[{source['number']}] {title}"},
            {
                "type": "mrkdwn",
                "text": f"*Type/date*\n{source.get('content_type', 'source')} - {source.get('publish_date') or 'unknown date'}",
            },
        ]
        blocks.append({"type": "section", "fields": fields})
        if source.get("excerpt"):
            blocks.append(
                {
                    "type": "context",
                    "elements": [{"type": "mrkdwn", "text": source["excerpt"]}],
                }
            )
    return blocks[:50]


def block_kit_tool_result(
    fallback_text: str,
    blocks: List[Dict[str, Any]],
    structured_content: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "content": [{"type": "text", "text": fallback_text}],
        "structuredContent": structured_content,
        "_meta": {
            "slack": {
                "blocks": blocks,
            }
        },
    }


def interaction_value(arguments: Dict[str, Any]) -> str:
    return json.dumps(arguments, separators=(",", ":"))
