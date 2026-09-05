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
# can render a source attachment mention element.
SLACK_LINKED_FOOTNOTE_PATTERN = re.compile(r"<[^>|]+\|(?P<label>\[(?:\d+)(?:[a-z])?\])>")
HTML_CITATION_PATTERN = re.compile(
    r'<a\s+href="([^"]+)"[^>]*>(\[\d+\])</a>',
    re.IGNORECASE,
)
MARKDOWN_LINK_PATTERN = re.compile(r"(?<!!)\[([^\]\n]+)\]\((https?://[^)\s]+)\)")
MARKDOWN_BOLD_PATTERN = re.compile(r"\*\*([^*\n][\s\S]*?[^*\n])\*\*")
SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
ORDERED_LIST_LINE_PATTERN = re.compile(r"^\s*\d+[.)]\s+(.+)$")
UNORDERED_LIST_LINE_PATTERN = re.compile(r"^\s*[-*+]\s+(.+)$")
VTT_TIMING_PATTERN = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2}\.\d{3})"
)
VOICE_TAG_PATTERN = re.compile(r"^<v\s+([^>]+)>(.*?)(?:</v>)?$", re.IGNORECASE | re.DOTALL)
SPEAKER_PREFIX_PATTERN = re.compile(r"^([^:\n]{1,80}):\s*(.+)$", re.DOTALL)


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


def _filtered_names(values: Any) -> List[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    filtered = []
    for value in values:
        name = str(value or "").strip()
        if not name:
            continue
        lower = name.lower()
        if "unknown" in lower:
            continue
        if "speaker_" in lower:
            continue
        filtered.append(name)
    return filtered


def _source_speakers(source: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    names.extend(_filtered_names(source.get("speakers") or []))
    for passage in source.get("passages") or []:
        if isinstance(passage, dict):
            names.extend(_filtered_names(passage.get("speakers") or []))
    return list(dict.fromkeys(names))


def _collectiveaccess_link(source: Dict[str, Any]) -> Optional[str]:
    occurrence_id = str(source.get("occurrence_id") or "").strip()
    if not occurrence_id:
        return None
    return (
        "https://archives.wbez.org/index.php/editor/occurrences/"
        f"OccurrenceEditor/Summary/occurrence_id/{occurrence_id}"
    )


def _format_mmss(seconds: Any) -> Optional[str]:
    if not isinstance(seconds, (int, float)):
        return None
    total_seconds = max(0, int(round(float(seconds))))
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


def source_time(source: Dict[str, Any]) -> Optional[str]:
    starts: List[float] = []
    ends: List[float] = []
    for key, values in (("start_seconds", starts), ("end_seconds", ends)):
        value = source.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    for passage in source.get("passages") or []:
        if not isinstance(passage, dict):
            continue
        start = passage.get("start_seconds")
        end = passage.get("end_seconds")
        if isinstance(start, (int, float)):
            starts.append(float(start))
        if isinstance(end, (int, float)):
            ends.append(float(end))
    if not starts and not ends:
        return None
    start_text = _format_mmss(min(starts)) if starts else None
    end_text = _format_mmss(max(ends)) if ends else None
    if start_text and end_text:
        return f"{start_text}-{end_text}"
    return start_text or end_text


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
    if _is_transcript(source):
        for key in ("transcript_url", "url"):
            value = source.get(key)
            if value:
                return str(value)
        recording_urls = source.get("recording_urls") or []
        if isinstance(recording_urls, list):
            for value in recording_urls:
                if value:
                    return str(value)
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


def _parse_vtt_timestamp(value: str) -> Optional[float]:
    try:
        hours, minutes, seconds = value.split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except (TypeError, ValueError):
        return None


def _clean_transcript_text(value: str) -> tuple[Optional[str], str]:
    value = value.strip()
    voice = VOICE_TAG_PATTERN.match(value)
    if voice:
        return (
            re.sub(r"\s+", " ", voice.group(1)).strip(),
            re.sub(r"\s+", " ", re.sub(r"</v>", "", voice.group(2))).strip(),
        )
    value = re.sub(r"</?v[^>]*>", "", value).strip()
    prefix = SPEAKER_PREFIX_PATTERN.match(value)
    if prefix:
        return re.sub(r"\s+", " ", prefix.group(1)).strip(), re.sub(r"\s+", " ", prefix.group(2)).strip()
    return None, re.sub(r"\s+", " ", value).strip()


def format_transcript_excerpt_text(text: Any) -> str:
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return ""
    blocks = re.split(r"\n{2,}", raw)
    formatted: List[str] = []
    for block in blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        timing = VTT_TIMING_PATTERN.match(lines[0])
        if not timing:
            formatted.append(work_object_description_text(block, 2000))
            continue
        start = _parse_vtt_timestamp(timing.group("start"))
        end = _parse_vtt_timestamp(timing.group("end"))
        time_text = ""
        if start is not None and end is not None:
            time_text = f"{_format_mmss(start)}-{_format_mmss(end)}"
        speaker, cue_text = _clean_transcript_text(" ".join(lines[1:]))
        if speaker and cue_text:
            formatted.append(f"{time_text} {speaker}: {cue_text}".strip())
        elif cue_text:
            formatted.append(f"{time_text} {cue_text}".strip())
    return truncate_text("\n\n".join(formatted), 12000)


def source_full_text(source: Dict[str, Any]) -> str:
    if _is_transcript(source):
        raw = source.get("raw_vtt_excerpt")
        if raw:
            return format_transcript_excerpt_text(raw)
        passages = source.get("passages") or []
        if isinstance(passages, list):
            chunks = [
                format_transcript_excerpt_text(passage.get("raw_vtt_excerpt"))
                for passage in passages
                if isinstance(passage, dict) and passage.get("raw_vtt_excerpt")
            ]
            text = "\n\n".join(chunk for chunk in chunks if chunk)
            if text:
                return text
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


def work_object_external_id(source: Dict[str, Any], index: Optional[int] = None) -> str:
    # Prefer IDs that the archive can look up again (used by Work Objects and
    # "Open" actions). In particular, avoid defaulting to a URL-like "source_id"
    # when a stable archive identifier is available.
    stable_id_candidates = [
        source.get("shared_source_id"),
        source.get("parent_id"),
        source.get("chunk_id"),
        source.get("occurrence_id"),
        source.get("source_id"),
        source.get("id"),
        index,
        source_url(source),
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
    return stable_id


def source_reference(source: Dict[str, Any], index: int) -> Dict[str, Any]:
    publish_date = source.get("publish_date") or source.get("published_at") or source.get("publication_date") or source.get("date")
    if hasattr(publish_date, "isoformat"):
        publish_date = publish_date.isoformat()
    stable_id = work_object_external_id(source, index)
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
        "speakers": source.get("speakers") or [],
        "program": source.get("program"),
        "guests": source.get("guests") or [],
        "time": source_time(source),
        "occurrence_id": source.get("occurrence_id"),
        "parent_id": source.get("parent_id"),
        "passages": source.get("passages") or [],
    }


def source_references(results: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = [source for source in results if isinstance(source, dict)]
    transcript_groups: Dict[str, List[Dict[str, Any]]] = {}
    for source in rows:
        if not _is_transcript(source):
            continue
        recording_id = str(
            source.get("parent_id")
            or source.get("source_id")
            or source.get("occurrence_id")
            or source.get("transcript_name")
            or ""
        ).strip()
        chunk_id = str(source.get("chunk_id") or "").strip()
        if recording_id and chunk_id:
            transcript_groups.setdefault(recording_id, []).append(source)

    shared_ids: Dict[int, str] = {}
    for recording_id, sources in transcript_groups.items():
        if len(sources) < 2:
            continue
        chunk_parts: List[str] = []
        for source in sources:
            chunk_id = str(source.get("chunk_id") or "").strip()
            prefix = f"{recording_id}-"
            chunk_parts.append(chunk_id[len(prefix):] if chunk_id.startswith(prefix) else chunk_id)
        shared_id = f"{recording_id}-{','.join(dict.fromkeys(chunk_parts))}"
        for source in sources:
            shared_ids[id(source)] = shared_id

    return [
        source_reference(
            {**source, "shared_source_id": shared_ids[id(source)]}
            if id(source) in shared_ids
            else source,
            index,
        )
        for index, source in enumerate(rows, 1)
    ]


def cited_source_references(answer: str, sources: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return only cited sources, numbered by their first appearance in an answer."""
    by_number = {int(source["number"]): source for source in sources if source.get("number") is not None}
    cited: List[Dict[str, Any]] = []
    displayed_keys = set()
    for match in SOURCE_MARKER_PATTERN.finditer(answer or ""):
        source = by_number.get(int(match.group(1)))
        if not source:
            continue
        key = source.get("source_id") or source.get("url") or f"src:{source['number']}"
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

def _merge_styles(*styles: Optional[Dict[str, bool]]) -> Dict[str, bool]:
    merged: Dict[str, bool] = {}
    for style in styles:
        if style:
            merged.update(style)
    return merged


def _text_element(text: str, style: Optional[Dict[str, bool]] = None) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    element: Dict[str, Any] = {"type": "text", "text": text}
    if style:
        element["style"] = style
    return element


def _append_text_element(
    elements: List[Dict[str, Any]],
    text: str,
    style: Optional[Dict[str, bool]] = None,
) -> None:
    element = _text_element(text, style)
    if not element:
        return
    if (
        elements
        and elements[-1].get("type") == "text"
        and elements[-1].get("style") == element.get("style")
    ):
        elements[-1]["text"] = str(elements[-1].get("text") or "") + text
        return
    elements.append(element)


def _attachment_mention_element(
    label: str,
    source: Dict[str, Any],
    *,
    work_object_app_id: Optional[str],
    attachment_locations: Optional[Dict[str, Dict[str, str]]] = None,
) -> Optional[Dict[str, Any]]:
    if not work_object_app_id:
        return None
    entity_id = work_object_external_id(source)
    url = source.get("url")
    if not entity_id or not url:
        return None
    location = (attachment_locations or {}).get(entity_id) or {}
    if not location.get("channel_id") or not location.get("ts"):
        return None
    mention = {
        "type": "attachment_mention",
        "entity_id": entity_id,
        "app_id": str(work_object_app_id),
        "text": label,
        "url": slack_link_url(str(url)),
        "icon_url": WBEZ_WORK_OBJECT_ICON_URL if _is_wbez(source) else CST_WORK_OBJECT_ICON_URL,
        "channel_id": str(location["channel_id"]),
        "ts": str(location["ts"]),
    }
    return mention


def _find_next_inline_token(text: str, start: int) -> Optional[tuple[int, int, str, Any]]:
    candidates: List[tuple[int, int, str, Any]] = []

    for delimiter, token_type in (("**", "bold"), ("__", "underline"), ("_", "italic"), ("*", "italic")):
        token_start = text.find(delimiter, start)
        if token_start == -1:
            continue
        if delimiter in {"_", "*"}:
            if token_start > 0 and text[token_start - 1].isalnum():
                continue
            if token_start + 1 < len(text) and text[token_start + 1].isspace():
                continue
        token_end = text.find(delimiter, token_start + len(delimiter))
        if token_end == -1 or token_end == token_start + len(delimiter):
            continue
        if delimiter in {"_", "*"} and token_end + 1 < len(text) and text[token_end + 1].isalnum():
            continue
        candidates.append((token_start, token_end + len(delimiter), token_type, (delimiter, token_end)))

    for opener, closer, token_type in (("<u>", "</u>", "underline"),):
        token_start = text.find(opener, start)
        if token_start == -1:
            continue
        token_end = text.find(closer, token_start + len(opener))
        if token_end == -1 or token_end == token_start + len(opener):
            continue
        candidates.append((token_start, token_end + len(closer), token_type, (opener, closer, token_end)))

    link_match = MARKDOWN_LINK_PATTERN.search(text, start)
    if link_match:
        candidates.append((link_match.start(), link_match.end(), "link", link_match))

    slack_link_match = re.search(r"<(https?://[^>|]+)\|([^>\n]+)>", text[start:])
    if slack_link_match:
        candidates.append((
            start + slack_link_match.start(),
            start + slack_link_match.end(),
            "slack_link",
            slack_link_match,
        ))

    footnote_match = FOOTNOTE_PATTERN.search(text, start)
    if footnote_match:
        candidates.append((footnote_match.start(), footnote_match.end(), "footnote", footnote_match))

    if not candidates:
        return None
    return min(candidates, key=lambda candidate: (candidate[0], candidate[1]))


def _rich_text_inline_elements(
    text: str,
    *,
    sources_by_number: Dict[int, Dict[str, Any]],
    work_object_app_id: Optional[str],
    attachment_locations: Optional[Dict[str, Dict[str, str]]] = None,
    style: Optional[Dict[str, bool]] = None,
) -> List[Dict[str, Any]]:
    elements: List[Dict[str, Any]] = []
    cursor = 0
    while cursor < len(text):
        token = _find_next_inline_token(text, cursor)
        if not token:
            _append_text_element(elements, text[cursor:], style)
            break
        start, end, token_type, data = token
        if start > cursor:
            _append_text_element(elements, text[cursor:start], style)

        if token_type in {"bold", "italic", "underline"}:
            if token_type == "underline" and data[0] == "<u>":
                inner_start = start + len(data[0])
                inner_end = data[2]
            else:
                delimiter, inner_end = data
                inner_start = start + len(delimiter)
            nested_style = _merge_styles(style, {token_type: True})
            elements.extend(
                _rich_text_inline_elements(
                    text[inner_start:inner_end],
                    sources_by_number=sources_by_number,
                    work_object_app_id=work_object_app_id,
                    attachment_locations=attachment_locations,
                    style=nested_style,
                )
            )
        elif token_type == "link":
            match = data
            label = html.unescape(match.group(1))
            url = html.unescape(match.group(2))
            element: Dict[str, Any] = {
                "type": "link",
                "url": slack_link_url(url),
                "text": label,
            }
            if style:
                element["style"] = style
            elements.append(element)
        elif token_type == "slack_link":
            match = data
            url = html.unescape(match.group(1))
            label = html.unescape(match.group(2))
            element = {
                "type": "link",
                "url": slack_link_url(url),
                "text": label,
            }
            if style:
                element["style"] = style
            elements.append(element)
        elif token_type == "footnote":
            match = data
            label = match.group(0)
            source = sources_by_number.get(int(match.group("num")))
            mention = _attachment_mention_element(
                label,
                source,
                work_object_app_id=work_object_app_id,
                attachment_locations=attachment_locations,
            ) if source else None
            if mention:
                elements.append(mention)
            else:
                _append_text_element(elements, label, style)
        cursor = end
    return elements

def _is_wbez(source):
    return _is_transcript(source) or "wbez.org" in str(source.get("url") or "")


def _rich_text_section(
    text: str,
    sources_by_number: Dict[int, Dict[str, Any]],
    work_object_app_id: Optional[str],
    attachment_locations: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, Any]:
    return {
        "type": "rich_text_section",
        "elements": _rich_text_inline_elements(
            text,
            sources_by_number=sources_by_number,
            work_object_app_id=work_object_app_id,
            attachment_locations=attachment_locations,
        ) or [{"type": "text", "text": ""}],
    }


def _markdown_rich_text_elements(
    text: str,
    *,
    sources_by_number: Dict[int, Dict[str, Any]],
    work_object_app_id: Optional[str],
    attachment_locations: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[Dict[str, Any]]:
    rich_elements: List[Dict[str, Any]] = []
    pending_list_style: Optional[str] = None
    pending_list_items: List[Dict[str, Any]] = []

    def flush_list() -> None:
        nonlocal pending_list_style, pending_list_items
        if pending_list_style and pending_list_items:
            rich_elements.append({
                "type": "rich_text_list",
                "style": pending_list_style,
                "elements": pending_list_items,
            })
        pending_list_style = None
        pending_list_items = []

    for line in (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        unordered_match = UNORDERED_LIST_LINE_PATTERN.match(line)
        ordered_match = ORDERED_LIST_LINE_PATTERN.match(line)
        if unordered_match or ordered_match:
            list_style = "bullet" if unordered_match else "ordered"
            item_text = (unordered_match or ordered_match).group(1)
            if pending_list_style != list_style:
                flush_list()
                pending_list_style = list_style
            pending_list_items.append(
                _rich_text_section(item_text, sources_by_number, work_object_app_id, attachment_locations)
            )
            continue

        flush_list()
        if not line.strip():
            if rich_elements:
                rich_elements.append(_rich_text_section("\n", {}, None))
            continue
        rich_elements.append(_rich_text_section(line, sources_by_number, work_object_app_id, attachment_locations))

    flush_list()
    return rich_elements

def answer_blocks(
    answer: str,
    sources: List[Dict[str, Any]],
    *,
    work_object_app_id: Optional[str] = None,
    attachment_locations: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    sources_by_number = {
        int(source.get("number")): source
        for source in sources
        if isinstance(source, dict)
        and str(source.get("number") or "").isdigit()
    }
    for chunk in split_mrkdwn(answer):
        chunk = SLACK_LINKED_FOOTNOTE_PATTERN.sub(lambda m: m.group("label"), chunk)
        blocks.append({
            "type": "rich_text",
            "elements": _markdown_rich_text_elements(
                chunk,
                sources_by_number=sources_by_number,
                work_object_app_id=work_object_app_id,
                attachment_locations=attachment_locations,
            ),
        })

    return blocks[:50]


def source_attachment_blocks(
    sources: List[Dict[str, Any]],
    *,
    work_object_app_id: Optional[str],
) -> Optional[List[Dict[str, Any]]]:
    if not sources or not work_object_app_id:
        return None

    elements: List[Dict[str, Any]] = [
        {"type": "text", "text": "Sources:", "style": {"bold": True}},
    ]
    for source in sources:
        if not isinstance(source, dict):
            continue
        entity_id = work_object_external_id(source)
        url = source.get("url")
        if not entity_id or not url:
            continue
        title = str(source.get("title") or "Archive source").strip() or "Archive source"
        display_id = str(source.get("number") or "").strip()
        label = f"[{display_id}] " if display_id else ""
        elements.extend(
            [
                {"type": "text", "text": "\n"},
                {"type": "text", "text": label},
                {
                    "type": "attachment_mention",
                    "entity_id": entity_id,
                    "app_id": str(work_object_app_id),
                    "text": title,
                    "url": slack_link_url(str(url)),
                    "icon_url": WBEZ_WORK_OBJECT_ICON_URL if _is_wbez(source) else CST_WORK_OBJECT_ICON_URL,
                },
            ]
        )

    if len(elements) == 1:
        return None
    return [
        {
            "type": "rich_text",
            "elements": [
                {
                    "type": "rich_text_section",
                    "elements": elements,
                }
            ],
        }
    ]

def work_object_entities(
    sources: List[Dict[str, Any]],
    *,
    include_full_text: bool = False,
    include_excerpt: bool = False,
    include_collectiveaccess: bool = False,
) -> List[Dict[str, Any]]:
    """Build Item Work Object metadata for cited archive sources."""
    entities = []
    for source in sources:
        external_id = work_object_external_id(source)
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
        icon_url = WBEZ_WORK_OBJECT_ICON_URL if _is_wbez(source) else CST_WORK_OBJECT_ICON_URL
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
            time_text = source_time(source)
            if time_text:
                custom_fields.append(
                    {
                        "key": "time",
                        "label": "Time",
                        "value": time_text,
                        "type": "string",
                    }
                )
            speakers = _source_speakers(source)
            if speakers:
                shortlist = speakers[:5]
                speakers_text = ", ".join(shortlist) + ("..." if len(speakers) > 5 else "")
                custom_fields.append(
                    {
                        "key": "speakers",
                        "label": "Speakers",
                        "value": speakers_text,
                        "type": "string",
                    }
                )
            if include_collectiveaccess:
                ca_link = _collectiveaccess_link(source)
                if ca_link:
                    custom_fields.append(
                        {
                            "key": "collectiveaccess",
                            "label": "CollectiveAccess",
                            "value": ca_link,
                            "type": "slack#/types/link",
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
        excerpt_value = ""
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
        elif excerpt_value:
            body_text = work_object_description_text(excerpt_value, 500)
        else:
            body_text = ""
        display_order = []
        if body_text:
            custom_fields.append(
                {
                    "key": "description",
                    "label": "Description",
                    "value": body_text,
                    "type": "string",
                }
            )
            display_order.append("description")
        display_order.extend(["source_url", "date"])
        display_order.extend(["time", "speakers"] if transcript else ["author"])
        if transcript and include_collectiveaccess:
            display_order.append("collectiveaccess")
        if include_excerpt:
            display_order.append("excerpt")

        entities.append({
            "entity_type": "slack#/entities/item",
            "external_ref": {"id": external_id},
            "url": url,
            "entity_payload": {
                "attributes": {
                    "title": {"text": header_title},
                    "display_type": "Transcript" if transcript else "Article",
                    "display_id": str(source.get("number", "")),
                    **({"product_icon": product_icon} if product_icon else {}),
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
