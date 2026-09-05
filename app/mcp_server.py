import hmac
import json
import logging
import os
import re
import sys
import time
import uuid
from collections import Counter, OrderedDict
from contextlib import asynccontextmanager
from contextvars import ContextVar
from functools import lru_cache, partial
from pathlib import Path
from urllib.parse import parse_qs
from hashlib import sha256
from typing import Any, Dict, Iterable, List, Optional, Set

import requests
from starlette.concurrency import run_in_threadpool
from starlette.responses import PlainTextResponse
from starlette.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

BLOCK_KIT_META = {"slack": {"supportsBlockKit": True}}
SLACK_API_BASE_URL = "https://slack.com/api"
SLACK_POST_MESSAGE_URL = "https://slack.com/api/chat.postMessage"
SLACK_THREAD_HISTORY_LIMIT = 200
SLACK_THREAD_CONTEXT_MAX_CHARS = 30000
SLACK_PERSISTED_THREAD_MAX_TURNS = 40
SLACK_DM_WINDOW_CONTEXT_TS = "__dm_window__"
SLACK_GET_METHODS = {"conversations.replies", "conversations.history", "users.list"}
SLACK_WORK_OBJECT_ENTITIES_PER_MESSAGE = 50
WAVELENGTH_ALPHA_NOTICE = (
    "Wavelength is in early beta testing. "
    "Hallucinations, errors, and unexpected outages are likely. Please report "
    "all issues and commentary, good and bad, to <@U08G63C7H5Y>."
)
WAVELENGTH_ACCESS_DENIED_TEXT = (
    "Wavelength is currently limited to approved alpha testers."
)

_SLACK_THREAD_HISTORY: Dict[tuple[str, str], List[Dict[str, str]]] = {}
_MCP_SLACK_DM_CONTEXT: ContextVar[Optional[Dict[str, str]]] = ContextVar(
    "mcp_slack_dm_context", default=None
)
MCP_SERVER_CODE_VERSION = "slack-home-v1"

WORK_OBJECT_CACHE_MAX = 2000
WORK_OBJECT_CACHE_TTL_SECONDS = 60 * 60 * 24
SLACK_WORK_OBJECT_METADATA_MAX_BYTES = 3950
_WORK_OBJECT_ENTITY_CACHE: "OrderedDict[str, tuple[float, Dict[str, Any]]]" = OrderedDict()
_WORK_OBJECT_SOURCE_CACHE: "OrderedDict[str, tuple[float, Dict[str, Any]]]" = OrderedDict()

_TEAM_API_APP_ID: Dict[str, str] = {}


def _cache_team_api_app_id(team_id: Optional[str], api_app_id: Optional[str]) -> None:
    if not team_id or not api_app_id:
        return
    _TEAM_API_APP_ID[str(team_id)] = str(api_app_id)


def _team_api_app_id(team_id: Optional[str]) -> Optional[str]:
    if not team_id:
        return None
    return _TEAM_API_APP_ID.get(str(team_id))


def _verbose_logging_enabled() -> bool:
    return os.environ.get("WAVELENGTH_VERBOSE_LOGS") == "true"


def _configure_console_logging_from_argv() -> None:
    script_name = os.path.basename(sys.argv[0]) if sys.argv else ""
    if script_name == "mcp_server.py" and "--log-sources" in sys.argv[1:]:
        os.environ["WAVELENGTH_LOG_SOURCES"] = "true"
        logging.basicConfig(level=logging.INFO)
        sys.argv[:] = [sys.argv[0], *(arg for arg in sys.argv[1:] if arg != "--log-sources")]
    if script_name == "mcp_server.py" and any(arg in {"-v", "--verbose"} for arg in sys.argv[1:]):
        os.environ["WAVELENGTH_VERBOSE_LOGS"] = "true"
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
        sys.argv[:] = [sys.argv[0], *(arg for arg in sys.argv[1:] if arg not in {"-v", "--verbose"})]


def _log_verbose(label: str, value: Any) -> None:
    if not _verbose_logging_enabled():
        return
    logger.debug(
        "%s:\n%s",
        label,
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
    )


def _cache_work_object_entities(entities: List[Dict[str, Any]]) -> None:
    if not entities:
        return
    now = time.time()
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        external_ref = entity.get("external_ref")
        external_id = external_ref.get("id") if isinstance(external_ref, dict) else None
        if external_id is None:
            continue
        external_id = str(external_id)
        cached = {
            "entity_type": entity.get("entity_type"),
            "url": entity.get("url"),
            "app_unfurl_url": entity.get("app_unfurl_url"),
            "external_ref": external_ref,
            "entity_payload": entity.get("entity_payload"),
        }
        _WORK_OBJECT_ENTITY_CACHE.pop(external_id, None)
        _WORK_OBJECT_ENTITY_CACHE[external_id] = (now, cached)

    # Prune oldest entries and expired entries.
    cutoff = now - WORK_OBJECT_CACHE_TTL_SECONDS
    while _WORK_OBJECT_ENTITY_CACHE:
        oldest_id, (ts, _) = next(iter(_WORK_OBJECT_ENTITY_CACHE.items()))
        if len(_WORK_OBJECT_ENTITY_CACHE) > WORK_OBJECT_CACHE_MAX or ts < cutoff:
            _WORK_OBJECT_ENTITY_CACHE.pop(oldest_id, None)
            continue
        break


def _cached_work_object_entity(external_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not external_id:
        return None
    external_id = str(external_id)
    hit = _WORK_OBJECT_ENTITY_CACHE.get(external_id)
    if not hit:
        return None
    ts, entity = hit
    if (time.time() - ts) > WORK_OBJECT_CACHE_TTL_SECONDS:
        _WORK_OBJECT_ENTITY_CACHE.pop(external_id, None)
        return None
    # Mark as most recently used.
    _WORK_OBJECT_ENTITY_CACHE.move_to_end(external_id)
    return dict(entity)


def _cache_work_object_sources(sources: List[Dict[str, Any]]) -> None:
    from slack_format import work_object_external_id

    if not sources:
        return
    now = time.time()
    for source in sources:
        if not isinstance(source, dict):
            continue
        external_id = work_object_external_id(source)
        if not external_id:
            continue
        _WORK_OBJECT_SOURCE_CACHE.pop(external_id, None)
        _WORK_OBJECT_SOURCE_CACHE[external_id] = (now, dict(source))

    cutoff = now - WORK_OBJECT_CACHE_TTL_SECONDS
    while _WORK_OBJECT_SOURCE_CACHE:
        oldest_id, (ts, _) = next(iter(_WORK_OBJECT_SOURCE_CACHE.items()))
        if len(_WORK_OBJECT_SOURCE_CACHE) > WORK_OBJECT_CACHE_MAX or ts < cutoff:
            _WORK_OBJECT_SOURCE_CACHE.pop(oldest_id, None)
            continue
        break


def _cached_work_object_source(external_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not external_id:
        return None
    external_id = str(external_id)
    hit = _WORK_OBJECT_SOURCE_CACHE.get(external_id)
    if not hit:
        return None
    ts, source = hit
    if (time.time() - ts) > WORK_OBJECT_CACHE_TTL_SECONDS:
        _WORK_OBJECT_SOURCE_CACHE.pop(external_id, None)
        return None
    _WORK_OBJECT_SOURCE_CACHE.move_to_end(external_id)
    return dict(source)


def _work_object_source_id_from_event(event: Dict[str, Any]) -> Optional[str]:
    for key in ("external_ref", "entity_id", "external_id"):
        value = event.get(key)
        if isinstance(value, dict) and value.get("id") is not None:
            return str(value["id"])
        if value is not None and not isinstance(value, dict):
            return str(value)
    candidates = [event.get("external_ref"), event.get("entity"), event.get("link")]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        nested = candidate.get("external_ref")
        if isinstance(nested, dict) and nested.get("id") is not None:
            return str(nested["id"])
        for key in ("id", "entity_id", "external_id"):
            if candidate.get(key) is not None:
                return str(candidate[key])
    return None


def _verify_slack_signature(headers: Dict[str, str], body: bytes, signing_secret: str) -> bool:
    timestamp = headers.get("x-slack-request-timestamp")
    signature = headers.get("x-slack-signature")
    if not timestamp or not signature:
        return False
    try:
        request_time = int(timestamp)
    except ValueError:
        return False
    if abs(time.time() - request_time) > 60 * 5:
        return False
    base_string = b"v0:" + timestamp.encode("utf-8") + b":" + body
    expected = "v0=" + hmac.new(
        signing_secret.encode("utf-8"),
        base_string,
        sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


def _split_env_list(value: Optional[str]) -> Set[str]:
    return {item.strip() for item in (value or "").split(",") if item.strip()}


def _wavelength_allowed_slack_user_ids() -> Set[str]:
    return _split_env_list(os.environ.get("WAVELENGTH_ALLOWED_SLACK_USER_IDS"))


def _is_wavelength_allowed_slack_user(user_id: Optional[str]) -> bool:
    allowed_user_ids = _wavelength_allowed_slack_user_ids()
    return bool(user_id and user_id in allowed_user_ids)


def _slack_payload_user_id(value: Any) -> Optional[str]:
    if isinstance(value, dict):
        nested = value.get("id") or value.get("user")
        return str(nested) if nested else None
    return str(value) if value else None


def _headers_from_scope(scope) -> Dict[str, str]:
    return {
        key.decode("latin1").lower(): value.decode("latin1")
        for key, value in scope.get("headers", [])
    }


def _extract_bearer_token(headers: Dict[str, str]) -> Optional[str]:
    authorization = headers.get("authorization", "")
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() == "bearer" and value.strip():
        return value.strip()
    fallback = headers.get("x-wavelength-mcp-token")
    return fallback.strip() if fallback else None


def _header_value(headers: Dict[str, str], names: Iterable[str]) -> Optional[str]:
    for name in names:
        value = headers.get(name)
        if value:
            return value.strip()
    return None


def _slack_team_id(headers: Dict[str, str]) -> Optional[str]:
    return _header_value(
        headers,
        (
            "x-slack-team-id",
            "x-slack-context-team-id",
            "x-slack-workspace-id",
        ),
    )


def _slack_user_id(headers: Dict[str, str]) -> Optional[str]:
    return _header_value(
        headers,
        (
            "x-slack-user-id",
            "x-slack-context-user-id",
        ),
    )


def _mcp_slack_dm_context(headers: Dict[str, str]) -> Optional[Dict[str, str]]:
    channel = headers.get("x-wavelength-slack-channel")
    event_ts = headers.get("x-wavelength-slack-event-ts")
    user_id = headers.get("x-wavelength-slack-user-id")
    if not channel or not event_ts or not user_id:
        return None
    return {"channel": channel, "event_ts": event_ts, "user_id": user_id}


def _strip_bot_mentions(text: str) -> str:
    cleaned = re.sub(r"<@[A-Z0-9]+>\s*", "", text or "").strip()
    return cleaned or text.strip()


def _tool_payload_text(payload: Dict[str, Any]) -> str:
    content = payload.get("content") or []
    text = "\n".join(item.get("text", "") for item in content if item.get("type") == "text")
    return text or "Wavelength finished processing the request."


def _tool_payload_answer_markdown(payload: Dict[str, Any]) -> str:
    structured = payload.get("structuredContent") if isinstance(payload.get("structuredContent"), dict) else {}
    answer = structured.get("answer_markdown") if isinstance(structured, dict) else None
    return str(answer) if answer else _tool_payload_text(payload)


def _tool_payload_context_text(payload: Dict[str, Any]) -> str:
    text = _tool_payload_text(payload)
    sources = (payload.get("structuredContent") or {}).get("sources") or []
    if not isinstance(sources, list) or not sources:
        return text

    source_lines = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        number = source.get("number")
        title = source.get("title") or "Untitled source"
        content_type = source.get("content_type") or "source"
        publish_date = source.get("publish_date") or "unknown date"
        excerpt = source.get("excerpt")
        line = f"[{number}] {title}\n{content_type} - {publish_date}"
        if excerpt:
            line += f"\nExcerpt: {excerpt}"
        source_lines.append(line)
    if not source_lines:
        return text
    return text + "\n\nSources\n" + "\n\n".join(source_lines)


def _tool_payload_blocks(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    slack_meta = (payload.get("_meta") or {}).get("slack") or {}
    blocks = slack_meta.get("blocks") or []
    return blocks if isinstance(blocks, list) else []


def _metadata_size(metadata: Dict[str, Any]) -> int:
    return len(json.dumps(metadata, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def _rich_text_mention_summaries(blocks: Any) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            element_type = value.get("type")
            if element_type in {"attachment_mention", "work_object_mention"}:
                summaries.append({
                    "type": element_type,
                    "entity_id": value.get("entity_id"),
                    "app_id": value.get("app_id"),
                    "url": value.get("url"),
                    "text": value.get("text"),
                    "keys": sorted(value.keys()),
                })
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(blocks)
    return summaries


def _work_object_entity_summaries(entities: Any) -> List[Dict[str, Any]]:
    if not isinstance(entities, list):
        return []
    summaries: List[Dict[str, Any]] = []
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        external_ref = entity.get("external_ref") if isinstance(entity.get("external_ref"), dict) else {}
        payload = entity.get("entity_payload") if isinstance(entity.get("entity_payload"), dict) else {}
        attributes = payload.get("attributes") if isinstance(payload.get("attributes"), dict) else {}
        summaries.append({
            "entity_type": entity.get("entity_type"),
            "external_ref": external_ref,
            "url": entity.get("url"),
            "app_unfurl_url": entity.get("app_unfurl_url"),
            "title": (
                attributes.get("title", {}).get("text")
                if isinstance(attributes.get("title"), dict)
                else None
            ),
            "keys": sorted(entity.keys()),
        })
    return summaries


def _trim_work_object_entity(entity: Dict[str, Any]) -> Dict[str, Any]:
    payload = entity.get("entity_payload") if isinstance(entity.get("entity_payload"), dict) else {}
    attributes = payload.get("attributes") if isinstance(payload.get("attributes"), dict) else {}
    title = ""
    if isinstance(attributes.get("title"), dict):
        title = str(attributes["title"].get("text") or "")
    product_icon = attributes.get("product_icon") if isinstance(attributes, dict) else None
    display_type = attributes.get("display_type") if isinstance(attributes, dict) else None
    custom_fields = payload.get("custom_fields") if isinstance(payload.get("custom_fields"), list) else []
    minimal_fields = [
        field
        for field in custom_fields
        if isinstance(field, dict)
        and field.get("key") in {
            "description",
            "source_url",
            "date",
            "time",
            "author",
            "speakers",
            "excerpt",
            "collectiveaccess",
        }
    ]
    display_order = [
        field.get("key")
        for field in minimal_fields
        if isinstance(field, dict) and field.get("key")
    ]
    return {
        **entity,
        "entity_payload": {
            "attributes": {
                "title": {"text": title},
                **({"display_type": display_type} if display_type else {}),
                "display_id": str(attributes.get("display_id") or ""),
                **({"product_icon": product_icon} if product_icon else {}),
            },
            "custom_fields": minimal_fields,
            "display_order": display_order,
        },
    }


def _work_object_metadata_batches(
    entities: List[Dict[str, Any]],
    *,
    max_entities_per_batch: int = SLACK_WORK_OBJECT_ENTITIES_PER_MESSAGE,
) -> List[Dict[str, Any]]:
    batches: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []
    max_entities_per_batch = max(1, max_entities_per_batch)

    for entity in entities:
        if not isinstance(entity, dict):
            continue
        candidate_entity = entity
        if _metadata_size({"entities": [candidate_entity]}) > SLACK_WORK_OBJECT_METADATA_MAX_BYTES:
            candidate_entity = _trim_work_object_entity(entity)
        if _metadata_size({"entities": [candidate_entity]}) > SLACK_WORK_OBJECT_METADATA_MAX_BYTES:
            candidate_entity = _trim_work_object_entity({
                **entity,
                "entity_payload": {
                    "attributes": {"title": {"text": ""}},
                    "custom_fields": [],
                },
            })

        candidate = {"entities": [*current, candidate_entity]}
        if current and (
            len(current) >= max_entities_per_batch
            or _metadata_size(candidate) > SLACK_WORK_OBJECT_METADATA_MAX_BYTES
        ):
            batches.append({"entities": current})
            current = [candidate_entity]
        else:
            current.append(candidate_entity)

    if current:
        batches.append({"entities": current})
    return batches


def _tool_payload_work_object_metadata(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    batches = _tool_payload_work_object_metadata_batches(payload)
    return batches[0] if batches else None


def _tool_payload_work_object_metadata_batches(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    from slack_format import work_object_entities

    sources = (payload.get("structuredContent") or {}).get("sources") or []
    entities = (
        work_object_entities(sources, include_excerpt=True, include_collectiveaccess=False)
        if isinstance(sources, list)
        else []
    )
    if isinstance(sources, list):
        _cache_work_object_sources(sources)
    if not entities:
        return []
    # For Work Objects, `chat.postMessage` expects the same `metadata` schema as
    # `chat.unfurl`: a top-level object with an `entities` array.
    #
    # Do not mix Work Object metadata with "message metadata" (event_type /
    # event_payload), or Slack may silently drop the metadata and the unfurls.
    metadata = {"entities": entities}
    trimmed = False
    split = False
    try:
        if (
            len(entities) > SLACK_WORK_OBJECT_ENTITIES_PER_MESSAGE
            or _metadata_size(metadata) > SLACK_WORK_OBJECT_METADATA_MAX_BYTES
        ):
            metadata_batches = _work_object_metadata_batches(entities)
            split = len(metadata_batches) > 1
            trimmed = any(
                _metadata_size({"entities": [entity]}) > SLACK_WORK_OBJECT_METADATA_MAX_BYTES
                for entity in entities
                if isinstance(entity, dict)
            )
        else:
            metadata_batches = [metadata]
        # Log a small, safe summary for diagnosing missing icons/unfurls.
        entity_count = sum(
            len(batch.get("entities") or [])
            for batch in metadata_batches
            if isinstance(batch.get("entities"), list)
        )
        first_product_icon_url = None
        first_batch = metadata_batches[0] if metadata_batches else {}
        if entity_count and first_batch.get("entities"):
            first = first_batch["entities"][0]
            payload = first.get("entity_payload") if isinstance(first, dict) else None
            attrs = payload.get("attributes") if isinstance(payload, dict) else None
            product_icon = attrs.get("product_icon") if isinstance(attrs, dict) else None
            first_product_icon_url = product_icon.get("url") if isinstance(product_icon, dict) else None
        logger.info(
            "Slack Work Object metadata prepared: entities=%d batches=%d bytes=%s split=%s trimmed=%s product_icon_https=%s",
            entity_count,
            len(metadata_batches),
            [_metadata_size(batch) for batch in metadata_batches],
            split,
            trimmed,
            bool(first_product_icon_url and str(first_product_icon_url).startswith("https://")),
        )
        return metadata_batches
    except Exception:
        logger.exception("Could not size-trim Slack Work Object metadata; sending as-is.")
    return [metadata]


def _slack_api_url(method: str) -> str:
    base_url = os.environ.get("SLACK_API_URL", SLACK_API_BASE_URL).rstrip("/")
    return f"{base_url}/{method}"


def _without_none_values(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _log_slack_api_warnings(method: str, data: Any) -> None:
    """Log Slack warning fields without dumping full payloads."""
    if not isinstance(data, dict):
        return
    warning = data.get("warning")
    if isinstance(warning, str) and warning.strip():
        logger.warning("Slack %s warning: %s", method, warning)
    warnings = data.get("warnings")
    if isinstance(warnings, list) and warnings:
        logger.warning(
            "Slack %s warnings: %s",
            method,
            [str(item) for item in warnings if item is not None],
        )
    response_metadata = data.get("response_metadata")
    if not isinstance(response_metadata, dict):
        return
    messages = response_metadata.get("messages")
    if isinstance(messages, list) and messages:
        logger.warning(
            "Slack %s response_metadata.messages: %s",
            method,
            [str(item) for item in messages if item is not None],
        )


def _epoch_seconds_age_ms(value: Any, now: Optional[float] = None) -> Optional[int]:
    if value is None:
        return None
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return None
    if timestamp <= 0:
        return None
    return int(((now or time.time()) - timestamp) * 1000)


def _call_slack_api(
    *,
    token: str,
    method: str,
    payload: Dict[str, Any],
    timeout: int = 30,
) -> Dict[str, Any]:
    clean_payload = _without_none_values(payload)
    _log_verbose(f"Slack API request {method}", clean_payload)
    if method in SLACK_GET_METHODS:
        logger.info(
            "Calling Slack %s: channel=%s ts=%s latest=%s cursor_present=%s",
            method,
            clean_payload.get("channel"),
            clean_payload.get("ts"),
            clean_payload.get("latest"),
            bool(clean_payload.get("cursor")),
        )
        response = requests.get(
            _slack_api_url(method),
            headers={"Authorization": f"Bearer {token}"},
            params=clean_payload,
            timeout=timeout,
        )
    else:
        response = requests.post(
            _slack_api_url(method),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8",
            },
            json=clean_payload,
            timeout=timeout,
        )
    response.raise_for_status()
    data = response.json()
    _log_verbose(f"Slack API response {method}", data)
    if not _verbose_logging_enabled():
        _log_slack_api_warnings(method, data)
    if not data.get("ok"):
        raise RuntimeError(f"Slack {method} failed: {data}")
    return data


async def _call_slack_api_async(
    *,
    token: str,
    method: str,
    payload: Dict[str, Any],
    timeout: int = 30,
) -> Dict[str, Any]:
    return await run_in_threadpool(partial(
        _call_slack_api,
        token=token,
        method=method,
        payload=payload,
        timeout=timeout,
    ))


def _default_data_dir() -> Path:
    return Path(os.environ.get("WAVELENGTH_DATA_DIR") or Path(__file__).resolve().parent.parent / "data")


def _coverage_year_from_vtt(vtt_path: Path) -> Optional[str]:
    try:
        with vtt_path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.startswith("NOTE "):
                    if "-->" in line:
                        break
                    continue
                match = re.match(r"NOTE date:\s*(\d{4})-\d{2}-\d{2}\s*$", line.strip())
                if match:
                    return match.group(1)
    except OSError:
        return None
    return None


def _coverage_year_from_transcript_json(json_path: Path) -> Optional[str]:
    try:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(metadata, dict):
        return None
    recording_date = str(metadata.get("recording_date") or "")
    match = re.match(r"(\d{4})-\d{2}-\d{2}", recording_date)
    if match:
        return match.group(1)
    transcript_name = metadata.get("transcript_name")
    if isinstance(transcript_name, str) and transcript_name.endswith(".vtt"):
        return _coverage_year_from_vtt(json_path.with_name(transcript_name))
    return _coverage_year_from_vtt(json_path.with_suffix(".vtt"))


def _is_displayable_coverage_year(year: str) -> bool:
    try:
        year_number = int(year)
    except ValueError:
        return False
    return 1900 <= year_number <= time.gmtime().tm_year


def _count_article_ids(year_json_path: Path) -> int:
    try:
        with year_json_path.open(encoding="utf-8", errors="replace") as handle:
            return sum(1 for line in handle if '"id"' in line)
    except OSError:
        return 0


def _archive_coverage_by_year(data_dir: Optional[Path] = None) -> Dict[str, Dict[str, int]]:
    data_dir = data_dir or _default_data_dir()
    article_counts = {
        path.stem: _count_article_ids(path)
        for path in sorted(data_dir.glob("[0-9][0-9][0-9][0-9].json"))
    }

    transcript_counts: Counter[str] = Counter()
    for path in sorted(data_dir.glob("*.json")):
        if re.fullmatch(r"\d{4}", path.stem):
            continue
        year = _coverage_year_from_transcript_json(path)
        if year and _is_displayable_coverage_year(year):
            transcript_counts[year] += 1

    coverage: Dict[str, Dict[str, int]] = {}
    years = {
        year
        for year in set(article_counts) | set(transcript_counts)
        if _is_displayable_coverage_year(year)
    }
    for year in sorted(years):
        articles = article_counts.get(year, 0)
        transcripts = transcript_counts.get(year, 0)
        coverage[year] = {
            "articles": articles,
            "transcripts": transcripts,
            "total": articles + transcripts,
        }
    return coverage


def _number_text(value: int) -> str:
    return f"{value:,}"


def _coverage_summary(coverage: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    if not coverage:
        return {
            "year_range": "No local coverage data found",
            "articles": 0,
            "transcripts": 0,
            "total": 0,
        }
    years = sorted(coverage)
    articles = sum(row["articles"] for row in coverage.values())
    transcripts = sum(row["transcripts"] for row in coverage.values())
    return {
        "year_range": f"{years[0]}-{years[-1]}",
        "articles": articles,
        "transcripts": transcripts,
        "total": articles + transcripts,
    }


def _coverage_table_rows(coverage: Dict[str, Dict[str, int]]) -> List[List[Dict[str, Any]]]:
    rows: List[List[Dict[str, Any]]] = [
        [
            {"type": "raw_text", "text": "Year"},
            {"type": "raw_text", "text": "Articles"},
            {"type": "raw_text", "text": "Transcripts"},
            {"type": "raw_text", "text": "Total"},
        ]
    ]
    for year, counts in sorted(coverage.items(), reverse=True):
        rows.append(
            [
                {"type": "raw_text", "text": year},
                {
                    "type": "raw_number",
                    "value": counts["articles"],
                    "text": _number_text(counts["articles"]),
                },
                {
                    "type": "raw_number",
                    "value": counts["transcripts"],
                    "text": _number_text(counts["transcripts"]),
                },
                {
                    "type": "raw_number",
                    "value": counts["total"],
                    "text": _number_text(counts["total"]),
                },
            ]
        )
    return rows


def _wavelength_home_view(data_dir: Optional[Path] = None, user_id: str = "U_UNKNOWN", team_id: str = "unknown") -> Dict[str, Any]:
    from state import StateStore
    store = StateStore()
    blocks: List[Dict[str, Any]] = [
        {"type": "header", "text": {"type": "plain_text", "text": "Wavelength archive assistant"}},
        {"type": "section", "text": {"type": "mrkdwn", "text": f"Welcome, <@{user_id}>. Wavelength searches newsroom articles and timestamped transcripts. Answers include citations pointing to the evidence used."}},
        {"type": "actions", "elements": [
            {"type": "button", "text": {"type": "plain_text", "text": "Topic + date"}, "action_id": "wavelength_example:topic"},
            {"type": "button", "text": {"type": "plain_text", "text": "Compare over time"}, "action_id": "wavelength_example:compare"},
            {"type": "button", "text": {"type": "plain_text", "text": "Guest or speaker"}, "action_id": "wavelength_example:guest"},
        ]}, {"type": "divider"},
        {"type": "header", "text": {"type": "plain_text", "text": "Recent questions"}},
    ]
    for item in store.recent_queries(team_id, user_id):
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": item["question"]}, "accessory": {"type": "button", "text": {"type": "plain_text", "text": "Ask again"}, "action_id": f"wavelength_repeat:{item['id']}"}})
    blocks.append({"type": "header", "text": {"type": "plain_text", "text": "Saved sources"}})
    for item in store.saved_sources(team_id, user_id):
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*{item['title']}*\n{item['content_type']}"}})
    blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": "Ask in a DM, mention Wavelength in a channel, or use `/wavelength`."}]})
    return {"type": "home", "callback_id": "wavelength_home", "blocks": blocks}


@lru_cache(maxsize=1)
def _default_wavelength_home_view() -> Dict[str, Any]:
    return _wavelength_home_view()


def _publish_wavelength_home_tab(*, token: str, user_id: str, team_id: Optional[str] = None) -> None:
    if not _is_wavelength_allowed_slack_user(user_id):
        logger.info("Skipping Wavelength Home tab publish for non-allowed user %s.", user_id)
        return
    _call_slack_api(
        token=token,
        method="views.publish",
        payload={
            "user_id": user_id,
            "view": _wavelength_home_view(user_id=user_id, team_id=team_id or os.environ.get("WAVELENGTH_TEAM_ID", "unknown")),
        },
    )


def _slack_home_user_ids(token: str) -> List[str]:
    allowed_user_ids = _wavelength_allowed_slack_user_ids()
    if not allowed_user_ids:
        return []

    configured_user_ids = sorted(_split_env_list(os.environ.get("WAVELENGTH_HOME_USER_IDS")))
    if configured_user_ids:
        return [user_id for user_id in configured_user_ids if user_id in allowed_user_ids]

    user_ids: List[str] = []
    cursor = None
    while True:
        response = _call_slack_api(
            token=token,
            method="users.list",
            payload={"limit": 200, "cursor": cursor},
        )
        for member in response.get("members") or []:
            if not isinstance(member, dict):
                continue
            user_id = str(member.get("id") or "")
            if (
                user_id
                and user_id in allowed_user_ids
                and not member.get("deleted")
                and not member.get("is_bot")
                and user_id != "USLACKBOT"
            ):
                user_ids.append(user_id)
        cursor = (response.get("response_metadata") or {}).get("next_cursor")
        if not cursor:
            break
    return user_ids


def _publish_wavelength_home_tabs_on_startup() -> None:
    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        logger.warning("Skipping Wavelength Home tab startup publish; SLACK_BOT_TOKEN is not configured.")
        return
    try:
        user_ids = _slack_home_user_ids(token)
    except Exception as exc:
        logger.warning("Could not list Slack users for Home tab startup publish: %s", exc, exc_info=True)
        return
    for user_id in user_ids:
        try:
            _publish_wavelength_home_tab(token=token, user_id=user_id)
        except Exception as exc:
            logger.warning("Could not publish Wavelength Home tab for user %s: %s", user_id, exc, exc_info=True)
    logger.info("Published Wavelength Home tab on startup for %d Slack user(s).", len(user_ids))


def _set_slack_status(
    *,
    token: str,
    channel: str,
    thread_ts: str,
    status: str,
    loading_messages: Optional[List[str]] = None,
) -> None:
    _call_slack_api(
        token=token,
        method="assistant.threads.setStatus",
        payload={
            "channel_id": channel,
            "thread_ts": thread_ts,
            "status": status,
            "loading_messages": loading_messages,
        },
    )


def _task_stream_chunk(event: Dict[str, Any]) -> Dict[str, Any]:
    return _without_none_values(
        {
            "type": "task_update",
            "id": event.get("id"),
            "title": event.get("title"),
            "status": event.get("status"),
            "details": event.get("details"),
            "output": event.get("output"),
        }
    )


def _task_init_chunks(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for step in event.get("steps") or []:
        if not isinstance(step, dict):
            continue
        chunks.append(_task_stream_chunk(step))
    return chunks


def _message_ts_value(message: Dict[str, Any]) -> float:
    try:
        return float(message.get("ts", "0"))
    except (TypeError, ValueError):
        return 0.0


def _message_text(message: Dict[str, Any]) -> str:
    text_parts = []
    text = str(message.get("text") or "").strip()
    if text:
        text_parts.append(text)
    attachments = message.get("attachments") or []
    if isinstance(attachments, list):
        attachment_text = "\n".join(
            str(attachment.get("text") or "").strip()
            for attachment in attachments
            if isinstance(attachment, dict) and str(attachment.get("text") or "").strip()
        )
        if attachment_text:
            text_parts.append(attachment_text)
    blocks = message.get("blocks") or []
    if isinstance(blocks, list):
        for block in blocks:
            if not isinstance(block, dict):
                continue
            text_obj = block.get("text")
            if isinstance(text_obj, dict) and text_obj.get("text"):
                text_parts.append(str(text_obj["text"]).strip())

    unique_parts = []
    for part in text_parts:
        if part and part not in unique_parts:
            unique_parts.append(part)
    return "\n\n".join(unique_parts)


def _is_progress_message(text: str) -> bool:
    normalized = " ".join(text.split()).lower()
    return normalized in {
        "wavelength is searching the archive.",
        "received archive question",
        "wavelength source unfurls.",
    }


def _slack_thread_history_key(channel: Optional[str], thread_ts: Optional[str]) -> Optional[tuple[str, str]]:
    if not channel or not thread_ts:
        return None
    return (str(channel), str(thread_ts))


def _persist_slack_thread_turn(
    *,
    channel: Optional[str],
    thread_ts: Optional[str],
    role: str,
    content: str,
) -> None:
    key = _slack_thread_history_key(channel, thread_ts)
    content = str(content or "").strip()
    if key is None or role not in {"user", "assistant"} or not content:
        return
    if _is_progress_message(content):
        return

    history = _SLACK_THREAD_HISTORY.setdefault(key, [])
    turn = {"role": role, "content": content}
    if history and history[-1] == turn:
        return
    history.append(turn)
    if len(history) > SLACK_PERSISTED_THREAD_MAX_TURNS:
        del history[: len(history) - SLACK_PERSISTED_THREAD_MAX_TURNS]


def _persisted_slack_thread_context(
    *,
    channel: Optional[str],
    thread_ts: Optional[str],
) -> List[Dict[str, str]]:
    key = _slack_thread_history_key(channel, thread_ts)
    if key is None:
        return []
    context = [dict(turn) for turn in _SLACK_THREAD_HISTORY.get(key, [])]
    total_chars = sum(len(turn.get("content") or "") for turn in context)
    if total_chars <= SLACK_THREAD_CONTEXT_MAX_CHARS:
        return context
    trimmed: List[Dict[str, str]] = []
    remaining = SLACK_THREAD_CONTEXT_MAX_CHARS
    for turn in reversed(context):
        content = turn.get("content") or ""
        if len(content) > remaining:
            break
        trimmed.append(turn)
        remaining -= len(content)
    trimmed.reverse()
    trimmed.insert(
        0,
        {
            "role": "assistant",
            "content": "Earlier Slack conversation history was too long and was truncated before this point.",
        },
    )
    return trimmed


def _slack_thread_messages(
    *,
    token: str,
    channel: str,
    thread_ts: str,
    latest_ts: Optional[str] = None,
) -> List[Dict[str, Any]]:
    messages = []
    cursor = None
    while True:
        payload = {
            "channel": channel,
            "ts": thread_ts,
            "limit": SLACK_THREAD_HISTORY_LIMIT,
            "cursor": cursor,
            "latest": latest_ts,
            "inclusive": True if latest_ts else None,
        }
        response = _call_slack_api(
            token=token,
            method="conversations.replies",
            payload=payload,
        )
        page_messages = response.get("messages") or []
        if isinstance(page_messages, list):
            messages.extend(
                message for message in page_messages if isinstance(message, dict)
            )
        cursor = (response.get("response_metadata") or {}).get("next_cursor")
        if not cursor:
            break
    return sorted(messages, key=_message_ts_value)


def _slack_channel_messages(
    *,
    token: str,
    channel: str,
    latest_ts: Optional[str] = None,
) -> List[Dict[str, Any]]:
    messages = []
    cursor = None
    while True:
        payload = {
            "channel": channel,
            "limit": SLACK_THREAD_HISTORY_LIMIT,
            "cursor": cursor,
            "latest": latest_ts,
            "inclusive": False if latest_ts else None,
        }
        response = _call_slack_api(
            token=token,
            method="conversations.history",
            payload=payload,
        )
        page_messages = response.get("messages") or []
        if isinstance(page_messages, list):
            messages.extend(
                message for message in page_messages if isinstance(message, dict)
            )
        cursor = (response.get("response_metadata") or {}).get("next_cursor")
        if not cursor:
            break
    return sorted(messages, key=_message_ts_value)


def _slack_messages_context(
    *,
    messages: List[Dict[str, Any]],
    current_event_ts: Optional[str],
    current_user_id: Optional[str],
    current_question: str,
) -> List[Dict[str, str]]:
    context: List[Dict[str, str]] = []
    total_chars = 0
    current_event_value = _message_ts_value({"ts": current_event_ts})
    current_question = (current_question or "").strip()

    for message in messages:
        message_ts = _message_ts_value(message)
        if current_event_value and message_ts >= current_event_value:
            continue
        subtype = message.get("subtype")
        if subtype in {"message_changed", "message_deleted"}:
            continue
        text = _strip_bot_mentions(_message_text(message))
        if not text or _is_progress_message(text):
            continue

        role = "assistant" if message.get("bot_id") else "user"
        if (
            role == "user"
            and message.get("user") == current_user_id
            and current_question
            and text == current_question
        ):
            continue

        total_chars += len(text)
        context.append({"role": role, "content": text})

    if total_chars <= SLACK_THREAD_CONTEXT_MAX_CHARS:
        return context

    trimmed: List[Dict[str, str]] = []
    remaining = SLACK_THREAD_CONTEXT_MAX_CHARS
    for turn in reversed(context):
        content = turn["content"]
        if len(content) > remaining:
            break
        trimmed.append(turn)
        remaining -= len(content)
    trimmed.reverse()
    trimmed.insert(
        0,
        {
            "role": "assistant",
            "content": "Earlier Slack conversation history was too long and was truncated before this point.",
        },
    )
    return trimmed


def _slack_thread_context(
    *,
    token: str,
    channel: str,
    thread_ts: Optional[str],
    current_event_ts: Optional[str],
    current_user_id: Optional[str],
    current_question: str,
) -> List[Dict[str, str]]:
    if not thread_ts:
        return []

    messages = _slack_thread_messages(
        token=token,
        channel=channel,
        thread_ts=thread_ts,
        latest_ts=current_event_ts,
    )
    return _slack_messages_context(
        messages=messages,
        current_event_ts=current_event_ts,
        current_user_id=current_user_id,
        current_question=current_question,
    )


def _slack_dm_window_context(
    *,
    token: str,
    channel: str,
    current_event_ts: Optional[str],
    current_user_id: Optional[str],
    current_question: str,
) -> List[Dict[str, str]]:
    messages = _slack_channel_messages(
        token=token,
        channel=channel,
        latest_ts=current_event_ts,
    )
    return _slack_messages_context(
        messages=messages,
        current_event_ts=current_event_ts,
        current_user_id=current_user_id,
        current_question=current_question,
    )


def _slack_thread_context_or_empty(
    *,
    token: str,
    channel: Optional[str],
    thread_ts: Optional[str],
    current_event_ts: Optional[str],
    current_user_id: Optional[str],
    current_question: str,
) -> List[Dict[str, str]]:
    if not channel or not thread_ts:
        persisted_context = _persisted_slack_thread_context(
            channel=channel,
            thread_ts=thread_ts,
        )
        logger.warning(
            "Skipping Slack conversations.replies; missing required thread identifiers: "
            "channel=%s thread_ts=%s event_ts=%s user_id=%s persisted_turns=%d",
            channel,
            thread_ts,
            current_event_ts,
            current_user_id,
            len(persisted_context),
        )
        return persisted_context

    try:
        context = _slack_thread_context(
            token=token,
            channel=channel,
            thread_ts=thread_ts,
            current_event_ts=current_event_ts,
            current_user_id=current_user_id,
            current_question=current_question,
        )
        if context:
            logger.info(
                "Slack thread history loaded: channel=%s thread_ts=%s event_ts=%s turns=%d",
                channel,
                thread_ts,
                current_event_ts,
                len(context),
            )
            return context

        persisted_context = _persisted_slack_thread_context(
            channel=channel,
            thread_ts=thread_ts,
        )
        logger.warning(
            "Slack thread history was empty; using persisted fallback: "
            "channel=%s thread_ts=%s event_ts=%s persisted_turns=%d",
            channel,
            thread_ts,
            current_event_ts,
            len(persisted_context),
        )
        return persisted_context
    except Exception as exc:
        persisted_context = _persisted_slack_thread_context(
            channel=channel,
            thread_ts=thread_ts,
        )
        logger.warning(
            "Slack conversations.replies failed; using persisted fallback: "
            "channel=%s thread_ts=%s event_ts=%s user_id=%s persisted_turns=%d error=%s",
            channel,
            thread_ts,
            current_event_ts,
            current_user_id,
            len(persisted_context),
            exc,
            exc_info=True,
        )
        return persisted_context


def _slack_dm_window_context_or_empty(
    *,
    token: str,
    channel: Optional[str],
    current_event_ts: Optional[str],
    current_user_id: Optional[str],
    current_question: str,
) -> List[Dict[str, str]]:
    persisted_context = _persisted_slack_thread_context(
        channel=channel,
        thread_ts=SLACK_DM_WINDOW_CONTEXT_TS,
    )
    if not channel:
        logger.warning(
            "Skipping Slack conversations.history; missing DM channel: "
            "event_ts=%s user_id=%s persisted_turns=%d",
            current_event_ts,
            current_user_id,
            len(persisted_context),
        )
        return persisted_context

    try:
        context = _slack_dm_window_context(
            token=token,
            channel=channel,
            current_event_ts=current_event_ts,
            current_user_id=current_user_id,
            current_question=current_question,
        )
        if context:
            logger.info(
                "Slack DM window history loaded: channel=%s event_ts=%s turns=%d",
                channel,
                current_event_ts,
                len(context),
            )
            return context
        logger.warning(
            "Slack DM window history was empty; using persisted fallback: "
            "channel=%s event_ts=%s persisted_turns=%d",
            channel,
            current_event_ts,
            len(persisted_context),
        )
        return persisted_context
    except Exception as exc:
        logger.warning(
            "Slack conversations.history failed; using persisted fallback: "
            "channel=%s event_ts=%s user_id=%s persisted_turns=%d error=%s",
            channel,
            current_event_ts,
            current_user_id,
            len(persisted_context),
            exc,
            exc_info=True,
        )
        return persisted_context


def _start_slack_stream(
    *,
    token: str,
    channel: str,
    thread_ts: Optional[str],
    user_id: Optional[str],
    team_id: Optional[str],
    chunks: Optional[List[Dict[str, Any]]] = None,
    task_display_mode: Optional[str] = None,
) -> str:
    payload = _call_slack_api(
        token=token,
        method="chat.startStream",
        payload={
            "channel": channel,
            "thread_ts": thread_ts,
            "recipient_user_id": user_id,
            "recipient_team_id": team_id,
            "chunks": chunks,
            "task_display_mode": task_display_mode,
        },
    )
    stream_ts = payload.get("ts")
    if not stream_ts:
        raise RuntimeError(f"Slack chat.startStream did not return ts: {payload}")
    return str(stream_ts)


def _append_slack_stream_text(
    *,
    token: str,
    channel: str,
    stream_ts: str,
    text: str,
) -> None:
    if not text:
        return
    _call_slack_api(
        token=token,
        method="chat.appendStream",
        payload={
            "channel": channel,
            "ts": stream_ts,
            "markdown_text": text,
        },
    )


def _append_slack_stream_chunks(
    *,
    token: str,
    channel: str,
    stream_ts: str,
    chunks: List[Dict[str, Any]],
) -> None:
    if not chunks:
        return
    _call_slack_api(
        token=token,
        method="chat.appendStream",
        payload={
            "channel": channel,
            "ts": stream_ts,
            "chunks": chunks,
        },
    )


def _stop_slack_stream(
    *,
    token: str,
    channel: str,
    stream_ts: str,
    text: Optional[str] = None,
    blocks: Optional[List[Dict[str, Any]]] = None,
    chunks: Optional[List[Dict[str, Any]]] = None,
) -> None:
    _call_slack_api(
        token=token,
        method="chat.stopStream",
        payload={
            "channel": channel,
            "ts": stream_ts,
            "markdown_text": text,
            "blocks": blocks,
            "chunks": chunks,
        },
    )


def _stop_slack_stream_with_block_fallback(
    *,
    token: str,
    channel: str,
    stream_ts: str,
    text: Optional[str] = None,
    blocks: Optional[List[Dict[str, Any]]] = None,
) -> None:
    try:
        _stop_slack_stream(
            token=token,
            channel=channel,
            stream_ts=stream_ts,
            text=text,
            blocks=blocks,
        )
    except RuntimeError as exc:
        if not blocks or "streaming_mode_mismatch" not in str(exc):
            if "msg_too_long" in str(exc):
                _stop_slack_stream(
                    token=token, channel=channel, stream_ts=stream_ts,
                    text=(text or "")[:3000], blocks=None,
                )
                return
            raise
        _stop_slack_stream(
            token=token,
            channel=channel,
            stream_ts=stream_ts,
            text=text,
            blocks=None,
        )


def _post_slack_message(
    *,
    token: str,
    channel: str,
    text: str,
    blocks: Optional[List[Dict[str, Any]]] = None,
    thread_ts: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    unfurl_links: Optional[bool] = None,
    unfurl_media: Optional[bool] = None,
    fetch_after_post: bool = False,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "channel": channel,
        "text": text,
    }
    if blocks:
        body["blocks"] = blocks
    if thread_ts:
        body["thread_ts"] = thread_ts
    if metadata:
        body["metadata"] = metadata
    if unfurl_links is not None:
        body["unfurl_links"] = unfurl_links
    if unfurl_media is not None:
        body["unfurl_media"] = unfurl_media

    # This is deliberately an INFO log, rather than a verbose payload dump: a
    # Work Object can fail to render while its accompanying Block Kit cards
    # render normally.  The summary makes that distinction diagnosable without
    # putting source URLs or article text in application logs.
    entities = metadata.get("entities") if isinstance(metadata, dict) else []
    entities = entities if isinstance(entities, list) else []
    _cache_work_object_entities([entity for entity in entities if isinstance(entity, dict)])
    entity_types = sorted({
        str(entity.get("entity_type"))
        for entity in entities
        if isinstance(entity, dict) and entity.get("entity_type")
    })
    entity_ids = [
        str((entity.get("external_ref") or {}).get("id"))
        for entity in entities
        if isinstance(entity, dict)
        and isinstance(entity.get("external_ref"), dict)
        and (entity.get("external_ref") or {}).get("id") is not None
    ]
    logger.info(
        "Posting Slack message: channel=%s thread_ts=%s blocks=%d "
        "work_object_entities=%d entity_types=%s entity_ids=%s",
        channel,
        thread_ts,
        len(blocks or []),
        len(entities),
        entity_types,
        entity_ids,
    )
    try:
        response = _call_slack_api(token=token, method="chat.postMessage", payload=body)
    except Exception:
        logger.exception(
            "Slack message post failed: channel=%s thread_ts=%s "
            "work_object_entities=%d entity_types=%s entity_ids=%s",
            channel,
            thread_ts,
            len(entities),
            entity_types,
            entity_ids,
        )
        raise

    if isinstance(response, dict) and response.get("warning"):
        logger.warning(
            "Slack chat.postMessage warning: channel=%s thread_ts=%s warning=%s response_metadata=%s",
            channel,
            thread_ts,
            response.get("warning"),
            response.get("response_metadata"),
        )

    message = response.get("message") if isinstance(response, dict) else {}
    if isinstance(message, dict):
        _log_verbose("Slack chat.postMessage returned message", message)
    returned_metadata = message.get("metadata") if isinstance(message, dict) else None
    returned_entities = (
        returned_metadata.get("entities")
        if isinstance(returned_metadata, dict)
        and isinstance(returned_metadata.get("entities"), list)
        else []
    )
    returned_blocks = message.get("blocks") if isinstance(message, dict) else []
    returned_mentions = _rich_text_mention_summaries(returned_blocks)
    returned_entity_summaries = _work_object_entity_summaries(returned_entities)
    logger.info(
        "Slack message posted: channel=%s ts=%s work_object_entities_sent=%d "
        "work_object_entities_returned=%d metadata_returned=%s "
        "returned_work_object_entities=%s returned_mentions=%s",
        channel,
        message.get("ts") if isinstance(message, dict) else None,
        len(entities),
        len(returned_entities),
        isinstance(returned_metadata, dict),
        returned_entity_summaries,
        returned_mentions,
    )
    if fetch_after_post and isinstance(message, dict) and message.get("ts"):
        thread_root_ts = thread_ts or str(message["ts"])
        try:
            replies = _call_slack_api(
                token=token,
                method="conversations.replies",
                payload={
                    "channel": channel,
                    "ts": thread_root_ts,
                    "include_all_metadata":True,
                    "limit": 15,
                },
            )
            messages = replies.get("messages") if isinstance(replies, dict) else []
            posted = None
            if isinstance(messages, list):
                for candidate in messages:
                    if isinstance(candidate, dict) and candidate.get("ts") == message.get("ts"):
                        posted = candidate
                        break
            if isinstance(posted, dict):
                _log_verbose("Slack conversations.replies returned posted message", posted)
                history_metadata = posted.get("metadata")
                history_entities = (
                    history_metadata.get("entities")
                    if isinstance(history_metadata, dict)
                    and isinstance(history_metadata.get("entities"), list)
                    else []
                )
                logger.info(
                    "Slack conversations.replies posted message: channel=%s ts=%s "
                    "metadata_returned=%s work_object_entities_returned=%d "
                    "returned_work_object_entities=%s returned_mentions=%s attachments=%s",
                    channel,
                    message.get("ts"),
                    isinstance(history_metadata, dict),
                    len(history_entities),
                    _work_object_entity_summaries(history_entities),
                    _rich_text_mention_summaries(posted.get("blocks")),
                    posted.get("attachments"),
                )
            else:
                logger.info(
                    "Slack conversations.replies did not include posted message: channel=%s "
                    "thread_ts=%s posted_ts=%s message_count=%d",
                    channel,
                    thread_root_ts,
                    message.get("ts"),
                    len(messages) if isinstance(messages, list) else 0,
                )
        except Exception:
            logger.exception(
                "Could not fetch Slack posted message after post: channel=%s ts=%s",
                channel,
                message.get("ts"),
            )
    return message if isinstance(message, dict) else {}


def _update_slack_message(
    *,
    token: str,
    channel: str,
    ts: str,
    text: str,
    blocks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "channel": channel,
        "ts": ts,
        "text": text,
    }
    if blocks:
        body["blocks"] = blocks
    logger.info(
        "Updating Slack message: channel=%s ts=%s blocks=%d",
        channel,
        ts,
        len(blocks or []),
    )
    response = _call_slack_api(token=token, method="chat.update", payload=body)
    message = response.get("message") if isinstance(response, dict) else {}
    if isinstance(message, dict):
        _log_verbose("Slack chat.update returned message", message)
    return message if isinstance(message, dict) else {}


def _posted_message_from_replies(
    *,
    token: str,
    channel: str,
    thread_ts: str,
    posted_ts: str,
) -> Optional[Dict[str, Any]]:
    replies = _call_slack_api(
        token=token,
        method="conversations.replies",
        payload={
            "channel": channel,
            "ts": thread_ts,
            "include_all_metadata": True,
            "limit": 100,
        },
    )
    messages = replies.get("messages") if isinstance(replies, dict) else []
    if not isinstance(messages, list):
        return None
    for candidate in messages:
        if isinstance(candidate, dict) and str(candidate.get("ts") or "") == str(posted_ts):
            return candidate
    return None


def _message_has_attachment(message: Optional[Dict[str, Any]]) -> bool:
    attachments = message.get("attachments") if isinstance(message, dict) else None
    return isinstance(attachments, list) and bool(attachments)


def _wait_for_posted_message_attachment(
    *,
    token: str,
    channel: str,
    thread_ts: Optional[str],
    posted_ts: str,
    attempts: int = 4,
    delay_seconds: float = 0.35,
) -> Optional[Dict[str, Any]]:
    if not posted_ts:
        return None
    root_ts = thread_ts or posted_ts
    last_message: Optional[Dict[str, Any]] = None
    for attempt in range(max(1, attempts)):
        try:
            last_message = _posted_message_from_replies(
                token=token,
                channel=channel,
                thread_ts=root_ts,
                posted_ts=posted_ts,
            )
        except Exception:
            logger.exception(
                "Could not fetch Slack message while waiting for attachment: channel=%s thread_ts=%s posted_ts=%s",
                channel,
                root_ts,
                posted_ts,
            )
            return None
        if _message_has_attachment(last_message):
            return last_message
        if attempt < attempts - 1:
            time.sleep(delay_seconds)
    logger.warning(
        "Slack message attachment was not visible before inline mention update: channel=%s thread_ts=%s posted_ts=%s attempts=%d",
        channel,
        root_ts,
        posted_ts,
        attempts,
    )
    return last_message


def _work_object_registration_blocks(
    entities: List[Dict[str, Any]],
    *,
    work_object_app_id: Optional[str],
    include_header: bool = True,
) -> Optional[List[Dict[str, Any]]]:
    from slack_format import slack_link_url

    if not entities or not work_object_app_id:
        return None
    elements: List[Dict[str, Any]] = []
    if include_header:
        elements.append({"type": "text", "text": "Sources:", "style": {"bold": True}})
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        external_ref = entity.get("external_ref") if isinstance(entity.get("external_ref"), dict) else {}
        entity_id = str(external_ref.get("id") or "").strip()
        url = str(entity.get("url") or entity.get("app_unfurl_url") or "").strip()
        payload = entity.get("entity_payload") if isinstance(entity.get("entity_payload"), dict) else {}
        attributes = payload.get("attributes") if isinstance(payload.get("attributes"), dict) else {}
        display_id = str(attributes.get("display_id") or entity_id).strip()
        title = ""
        if isinstance(attributes.get("title"), dict):
            title = str(attributes["title"].get("text") or "").strip()
        title = title or display_id or "Archive source"
        product_icon = attributes.get("product_icon") if isinstance(attributes.get("product_icon"), dict) else {}
        icon_url = product_icon.get("url") if isinstance(product_icon, dict) else None
        if not entity_id or not url:
            continue
        label = f"[{display_id}] " if display_id.isdigit() else ""
        if elements:
            elements.append({"type": "text", "text": "\n"})
        if label:
            elements.append({"type": "text", "text": label})
        elements.append({
            "type": "attachment_mention",
            "entity_id": entity_id,
            "app_id": str(work_object_app_id),
            "text": title,
            "url": slack_link_url(url),
            **({"icon_url": slack_link_url(str(icon_url))} if icon_url else {}),
        })
    if not any(isinstance(element, dict) and element.get("type") == "attachment_mention" for element in elements):
        return None
    return [{
        "type": "rich_text",
        "elements": [
            {
                "type": "rich_text_section",
                "elements": elements,
            }
        ],
    }]


def _payload_sources(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    sources = (payload.get("structuredContent") or {}).get("sources") or []
    return [source for source in sources if isinstance(source, dict)] if isinstance(sources, list) else []


def _citation_numbers(text: str) -> List[int]:
    numbers: List[int] = []
    seen: Set[int] = set()
    for match in re.finditer(r"\[(\d+)(?:[a-z])?\]", text or ""):
        number = int(match.group(1))
        if number in seen:
            continue
        seen.add(number)
        numbers.append(number)
    return numbers


def _answer_message_units(answer: str) -> List[str]:
    units: List[str] = []
    current: List[str] = []

    def flush_current() -> None:
        nonlocal current
        if current:
            units.append("\n".join(current).strip())
            current = []

    for line in (answer or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if not line.strip():
            flush_current()
            continue
        if re.match(r"^\s*(?:[-*+]|\d+[.)])\s+", line):
            flush_current()
            units.append(line.strip())
            continue
        current.append(line)

    flush_current()
    return units or ([answer.strip()] if (answer or "").strip() else [])


def _answer_source_segments(
    answer: str,
    sources: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    from slack_format import work_object_external_id

    sources_by_number = {
        int(source.get("number")): source
        for source in sources
        if str(source.get("number") or "").isdigit()
    }
    attached_entity_ids: Set[str] = set()
    pending_units: List[str] = []
    segments: List[Dict[str, Any]] = []

    def flush(new_sources: List[Dict[str, Any]]) -> None:
        nonlocal pending_units
        text = "\n".join(unit for unit in pending_units if unit).strip()
        if text:
            segments.append({"text": text, "new_sources": new_sources})
        pending_units = []

    for unit in _answer_message_units(answer):
        pending_units.append(unit)
        new_sources: List[Dict[str, Any]] = []
        for number in _citation_numbers(unit):
            source = sources_by_number.get(number)
            if not source:
                continue
            entity_id = work_object_external_id(source)
            if not entity_id or entity_id in attached_entity_ids:
                continue
            attached_entity_ids.add(entity_id)
            new_sources.append(source)
        if new_sources:
            flush(new_sources)

    if pending_units:
        flush([])

    return segments


def _work_object_metadata_batches_for_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from slack_format import work_object_entities

    entities = work_object_entities(sources, include_excerpt=True, include_collectiveaccess=False)
    if sources:
        _cache_work_object_sources(sources)
    return _work_object_metadata_batches(entities) if entities else []


def _record_attachment_locations(
    attachment_locations: Dict[str, Dict[str, str]],
    *,
    entities: List[Dict[str, Any]],
    channel: str,
    message_ts: Optional[str],
) -> None:
    if not message_ts:
        return
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        external_ref = entity.get("external_ref") if isinstance(entity.get("external_ref"), dict) else {}
        entity_id = str(external_ref.get("id") or "").strip()
        if entity_id:
            attachment_locations.setdefault(entity_id, {"channel_id": channel, "ts": str(message_ts)})


def _post_slack_message_with_work_objects(
    *,
    token: str,
    channel: str,
    text: str,
    blocks: Optional[List[Dict[str, Any]]] = None,
    thread_ts: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
    team_id: Optional[str] = None,
) -> None:
    app_id = _team_api_app_id(team_id)
    payload = payload or {}
    sources = _payload_sources(payload)
    if not sources or not app_id:
        _post_slack_message(
            token=token,
            channel=channel,
            text=text,
            blocks=blocks,
            thread_ts=thread_ts,
        )
        return

    from slack_format import answer_blocks, markdown_to_slack_mrkdwn

    answer_markdown = _tool_payload_answer_markdown(payload)
    attachment_locations: Dict[str, Dict[str, str]] = {}
    segments = _answer_source_segments(answer_markdown, sources)
    if not segments:
        segments = [{"text": answer_markdown or text, "new_sources": []}]

    for segment in segments:
        segment_text = str(segment.get("text") or "")
        new_sources = segment.get("new_sources") if isinstance(segment.get("new_sources"), list) else []
        batches = _work_object_metadata_batches_for_sources(new_sources)
        first_batch = batches[0] if batches else None
        segment_blocks = answer_blocks(
            segment_text,
            sources,
            work_object_app_id=None if batches else app_id,
            attachment_locations=None if batches else attachment_locations,
        )
        posted = _post_slack_message(
            token=token,
            channel=channel,
            text=markdown_to_slack_mrkdwn(segment_text) or text,
            blocks=segment_blocks,
            thread_ts=thread_ts,
            metadata=first_batch,
        )
        posted_ts = str(posted.get("ts") or "")
        refreshed = None
        if first_batch:
            refreshed = _wait_for_posted_message_attachment(
                token=token,
                channel=channel,
                thread_ts=thread_ts,
                posted_ts=posted_ts,
            )
            if _message_has_attachment(refreshed):
                _record_attachment_locations(
                    attachment_locations,
                    entities=(first_batch or {}).get("entities") or [],
                    channel=channel,
                    message_ts=str(refreshed.get("ts") or posted_ts),
                )

        for batch in batches[1:]:
            continuation = _post_slack_message(
                token=token,
                channel=channel,
                text="Additional source attachments.",
                blocks=_work_object_registration_blocks(
                    batch.get("entities") or [],
                    work_object_app_id=app_id,
                    include_header=False,
                ),
                thread_ts=thread_ts,
                metadata=batch,
            )
            continuation_ts = str(continuation.get("ts") or "")
            refreshed_continuation = _wait_for_posted_message_attachment(
                token=token,
                channel=channel,
                thread_ts=thread_ts,
                posted_ts=continuation_ts,
            )
            if _message_has_attachment(refreshed_continuation):
                _record_attachment_locations(
                    attachment_locations,
                    entities=batch.get("entities") or [],
                    channel=channel,
                    message_ts=str(refreshed_continuation.get("ts") or continuation_ts),
                )

        if posted_ts and any(
            (entity.get("external_ref") or {}).get("id") in attachment_locations
            for batch in batches
            for entity in (batch.get("entities") or [])
            if isinstance(entity, dict) and isinstance(entity.get("external_ref"), dict)
        ):
            _update_slack_message(
                token=token,
                channel=channel,
                ts=posted_ts,
                text=markdown_to_slack_mrkdwn(segment_text) or text,
                blocks=answer_blocks(
                    segment_text,
                    sources,
                    work_object_app_id=app_id,
                    attachment_locations=attachment_locations,
                ),
            )


def _logo_test_payload(work_object_app_id: Optional[str]) -> Dict[str, Any]:
    from slack_format import CST_WORK_OBJECT_ICON_URL, WBEZ_WORK_OBJECT_ICON_URL, slack_link_url

    nonce = uuid.uuid4().hex[:8]
    wbez_entity_id = f"logo_test_wbez_s3_{nonce}"
    cst_entity_id = f"logo_test_cst_s3_{nonce}"
    wbez_icon = WBEZ_WORK_OBJECT_ICON_URL
    cst_icon = CST_WORK_OBJECT_ICON_URL
    wbez_url = f"https://www.wbez.org/?k={nonce}a"
    cst_url = f"https://chicago.suntimes.com/?k={nonce}b"
    text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit [1]. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua [2]."

    if work_object_app_id:
        blocks = [
            {
                "type": "rich_text",
                "elements": [
                    {
                        "type": "rich_text_section",
                        "elements": [
                            {"type": "text", "text": "Lorem ipsum dolor sit amet, consectetur adipiscing elit "},
                            {
                                "type": "work_object_mention",
                                "entity_id": wbez_entity_id,
                                "app_id": str(work_object_app_id),
                                "text": "[1]",
                                "url": slack_link_url(wbez_url),
                                **({"icon_url": slack_link_url(wbez_icon)} if wbez_icon else {}),
                            },
                            {"type": "text", "text": ". Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua "},
                            {
                                "type": "work_object_mention",
                                "entity_id": cst_entity_id,
                                "app_id": str(work_object_app_id),
                                "text": "[2]",
                                "url": slack_link_url(cst_url),
                                **({"icon_url": slack_link_url(cst_icon)} if cst_icon else {}),
                            },
                            {"type": "text", "text": "."},
                        ],
                    }
                ],
            }
        ]
    else:
        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": text}}]

    def entity(entity_id: str, title: str, url: str, icon_url: Optional[str], alt_text: str) -> Dict[str, Any]:
        return {
            "entity_type": "slack#/entities/item",
            "external_ref": {"id": entity_id},
            "url": url,
            "entity_payload": {
                "attributes": {
                    "title": {"text": title},
                    "display_id": entity_id,
                    **({"product_icon": {"url": icon_url, "alt_text": alt_text}} if icon_url else {}),
                },
                "custom_fields": [],
            },
        }

    return {
        "text": text,
        "blocks": blocks,
        "metadata": {
            "entities": [
                entity(wbez_entity_id, "[1] Lorem ipsum WBEZ S3", wbez_url, wbez_icon, "WBEZ"),
                entity(cst_entity_id, "[2] Lorem ipsum CST S3", cst_url, cst_icon, "CST"),
            ]
        },
        "selected_icons": {
            "wbez": {"source": "s3", "url": wbez_icon, "entity_id": wbez_entity_id},
            "cst": {"source": "s3", "url": cst_icon, "entity_id": cst_entity_id},
        },
    }


LOGO_TEST_PROBE_ID = "citation-probe-8df1ebd3"
LOGO_TEST_PROBE_URL = "https://www.wbez.org/?k=8df1ebd3a"


def _logo_test_registration_probe_payload() -> Dict[str, Any]:
    return {
        "text": "Work Object registration probe",
        "metadata": {
            "entities": [
                {
                    "entity_type": "slack#/entities/item",
                    "external_ref": {
                        "id": LOGO_TEST_PROBE_ID,
                        "type": "citation",
                    },
                    "url": LOGO_TEST_PROBE_URL,
                    "entity_payload": {
                        "attributes": {
                            "title": {"text": "WBEZ citation probe"},
                            "display_id": LOGO_TEST_PROBE_ID,
                        },
                        "custom_fields": [],
                    },
                }
            ],
        },
    }


def _logo_test_work_object_mention_probe_payload(work_object_app_id: Optional[str], ref: Optional[str] = None) -> Dict[str, Any]:
    from slack_format import slack_link_url

    text = "Work Object mention probe [1]"
    if not work_object_app_id:
        return {"text": text, "blocks": None}
    return {
        "text": text,
        "blocks": [
            {
                "type": "rich_text",
                "elements": [
                    {
                        "type": "rich_text_section",
                        "elements": [
                            {"type": "text", "text": "Work Object mention probe "},
                            {
                                "type": "work_object_mention",
                                "entity_id": ref or LOGO_TEST_PROBE_ID,
                                "app_id": str(work_object_app_id),
                                "text": "[1]",
                                "url": slack_link_url(LOGO_TEST_PROBE_URL),
                            },
                        ],
                    }
                ],
            }
        ],
    }


def _post_slash_response(
    *,
    response_url: str,
    text: str,
    blocks: Optional[List[Dict[str, Any]]] = None,
    response_type: str = "ephemeral",
) -> None:
    body: Dict[str, Any] = {
        "response_type": response_type,
        "text": text,
    }
    if blocks:
        body["blocks"] = blocks
    response = requests.post(response_url, json=body, timeout=30)
    response.raise_for_status()


def _answer_for_slack(
    service,
    question: str,
    user_id: Optional[str] = None,
    conversation_context: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    context = list(conversation_context or [])
    if user_id:
        context.append({"role": "user", "content": f"Slack user: <@{user_id}>"})
    return service.ask_archive(question=question, conversation_context=context)


def _stream_answer_to_slack(
    *,
    service,
    bot_token: str,
    question: str,
    channel: str,
    user_id: Optional[str],
    team_id: Optional[str],
    thread_ts: Optional[str],
    persist_thread_ts: Optional[str] = None,
    conversation_context: Optional[List[Dict[str, str]]] = None,
    event_ts: Optional[str] = None,
) -> None:
    persist_thread_ts = persist_thread_ts if persist_thread_ts is not None else thread_ts
    thinking_stream_ts = None
    thinking_stream_unavailable = False
    pending_initial_chunks: Optional[List[Dict[str, Any]]] = None

    def append_thinking_step(event: Dict[str, Any]) -> None:
        nonlocal thinking_stream_ts, thinking_stream_unavailable, pending_initial_chunks
        if thinking_stream_unavailable:
            return
        chunk = _task_stream_chunk(event)
        try:
            if thinking_stream_ts is None:
                thinking_stream_ts = _start_slack_stream(
                    token=bot_token,
                    channel=channel,
                    thread_ts=thread_ts,
                    user_id=user_id,
                    team_id=team_id,
                    chunks=pending_initial_chunks,
                    task_display_mode="plan",
                )
                pending_initial_chunks = None
            _append_slack_stream_chunks(
                token=bot_token,
                channel=channel,
                stream_ts=thinking_stream_ts,
                chunks=[chunk],
            )
        except Exception as exc:
            # Slack can finalize a Thinking Steps stream before a late task
            # update arrives. The archive request itself remains valid.
            logger.warning("Could not update Slack Thinking Steps; continuing: %s", exc)
            thinking_stream_ts = None
            thinking_stream_unavailable = True

    try:
        final_payload = None

        for event in service.stream_archive(
            question=question,
            conversation_context=[
                *(conversation_context or []),
                *(
                    [{"role": "user", "content": f"Slack user: <@{user_id}>"}]
                    if user_id
                    else []
                ),
            ],
            slack_dm_context=(
                {"channel": channel, "event_ts": event_ts, "user_id": user_id}
                if persist_thread_ts == SLACK_DM_WINDOW_CONTEXT_TS and event_ts and user_id
                else None
            ),
        ):
            event_type = event.get("type")
            if event_type == "tasks_init":
                pending_initial_chunks = _task_init_chunks(event)
                continue
            if event_type == "task":
                append_thinking_step(event)
                continue
            elif event_type == "final":
                final_payload = event.get("payload") or {}

        final_text = _tool_payload_text(final_payload or {})
        if thinking_stream_ts:
            try:
                _stop_slack_stream(
                    token=bot_token,
                    channel=channel,
                    stream_ts=thinking_stream_ts,
                    text=None,
                )
            except Exception:
                logger.warning("Could not close Slack generation indicator", exc_info=True)
        blocks = _tool_payload_blocks(final_payload or {})
        sources = (final_payload.get("structuredContent") or {}).get("sources") or []
        if isinstance(sources, list) and sources:
            try:
                from slack_format import answer_blocks

                blocks = answer_blocks(
                    _tool_payload_answer_markdown(final_payload or {}),
                    sources,
                    work_object_app_id=_team_api_app_id(team_id),
                )
            except Exception:
                logger.exception("Could not render Work Object citation mentions; using fallback blocks.")
        _post_slack_message_with_work_objects(
            token=bot_token,
            channel=channel,
            text=final_text,
            blocks=blocks,
            thread_ts=thread_ts,
            payload=final_payload or {},
            team_id=team_id,
        )
        _persist_slack_thread_turn(
            channel=channel,
            thread_ts=persist_thread_ts,
            role="user",
            content=question,
        )
        _persist_slack_thread_turn(
            channel=channel,
            thread_ts=persist_thread_ts,
            role="assistant",
            content=_tool_payload_context_text(final_payload or {}),
        )
        if team_id and user_id:
            from state import StateStore
            StateStore().record_query(uuid.uuid4().hex, team_id, user_id, question, channel, thinking_stream_ts)
    except Exception as exc:
        if thinking_stream_ts:
            try:
                _stop_slack_stream(
                    token=bot_token,
                    channel=channel,
                    stream_ts=thinking_stream_ts,
                    text=None,
                )
            except Exception:
                logger.warning("Could not close Slack Thinking Steps after an error", exc_info=True)
        raise


def _handle_event_question(
    *,
    service,
    bot_token: str,
    question: str,
    channel: Optional[str],
    user_id: Optional[str],
    team_id: Optional[str],
    thread_ts: Optional[str],
    event_ts: Optional[str] = None,
    conversation_context: Optional[List[Dict[str, str]]] = None,
    persist_thread_ts: Optional[str] = None,
    use_dm_window_context: bool = False,
    include_alpha_notice: bool = False,
) -> None:
    try:
        logger.info(
            "Handling Slack event question: channel=%s thread_ts=%s event_ts=%s user_id=%s team_id=%s",
            channel,
            thread_ts,
            event_ts,
            user_id,
            team_id,
        )
        if include_alpha_notice:
            _post_slack_message(
                token=bot_token,
                channel=channel,
                text=WAVELENGTH_ALPHA_NOTICE,
                thread_ts=thread_ts,
            )
        if conversation_context is None:
            if use_dm_window_context:
                if os.environ.get("WAVELENGTH_RESPONSES_MCP_ENABLED") == "true":
                    conversation_context = []
                else:
                    conversation_context = _slack_dm_window_context_or_empty(
                        token=bot_token,
                        channel=channel,
                        current_event_ts=event_ts,
                        current_user_id=user_id,
                        current_question=question,
                    )
            else:
                conversation_context = _slack_thread_context_or_empty(
                    token=bot_token,
                    channel=channel,
                    thread_ts=thread_ts,
                    current_event_ts=event_ts,
                    current_user_id=user_id,
                    current_question=question,
                )
        if not hasattr(service, "stream_archive"):
            payload = _answer_for_slack(
                service,
                question,
                user_id=user_id,
                conversation_context=conversation_context,
            )
            blocks = _tool_payload_blocks(payload)
            sources = (payload.get("structuredContent") or {}).get("sources") or []
            if isinstance(sources, list) and sources:
                try:
                    from slack_format import answer_blocks

                    blocks = answer_blocks(
                        _tool_payload_answer_markdown(payload),
                        sources,
                        work_object_app_id=_team_api_app_id(team_id),
                    )
                except Exception:
                    logger.exception("Could not render Work Object citation mentions; using fallback blocks.")
            _post_slack_message_with_work_objects(
                token=bot_token,
                channel=channel,
                text=_tool_payload_text(payload),
                blocks=blocks,
                thread_ts=thread_ts,
                payload=payload,
                team_id=team_id,
            )
            _persist_slack_thread_turn(
                channel=channel,
                thread_ts=persist_thread_ts if persist_thread_ts is not None else thread_ts,
                role="user",
                content=question,
            )
            _persist_slack_thread_turn(
                channel=channel,
                thread_ts=persist_thread_ts if persist_thread_ts is not None else thread_ts,
                role="assistant",
                content=_tool_payload_context_text(payload),
            )
            return

        _stream_answer_to_slack(
            service=service,
            bot_token=bot_token,
            question=question,
            channel=channel,
            user_id=user_id,
            team_id=team_id,
            thread_ts=thread_ts,
            persist_thread_ts=persist_thread_ts,
            conversation_context=conversation_context,
            event_ts=event_ts,
        )
    except Exception as exc:
        _post_slack_message(
            token=bot_token,
            channel=channel,
            text=f"Wavelength could not complete that request: {exc}",
            thread_ts=thread_ts,
        )


def _handle_slash_question(
    *,
    service,
    question: str,
    response_url: str,
    bot_token: Optional[str],
    channel: Optional[str],
    user_id: Optional[str],
    team_id: Optional[str],
    response_type: str,
) -> None:
    try:
        if bot_token and channel and hasattr(service, "stream_archive"):
            try:
                _stream_answer_to_slack(
                    service=service,
                    bot_token=bot_token,
                    question=question,
                    channel=channel,
                    user_id=user_id,
                    team_id=team_id,
                    thread_ts=None,
                )
                return
            except Exception:
                pass

        payload = _answer_for_slack(service, question, user_id=user_id)
        blocks = _tool_payload_blocks(payload)
        sources = (payload.get("structuredContent") or {}).get("sources") or []
        if isinstance(sources, list) and sources:
            try:
                from slack_format import answer_blocks

                blocks = answer_blocks(
                    _tool_payload_answer_markdown(payload),
                    sources,
                    work_object_app_id=_team_api_app_id(team_id),
                )
            except Exception:
                logger.exception("Could not render Work Object citation mentions; using fallback blocks.")
        _post_slash_response(
            response_url=response_url,
            text=_tool_payload_text(payload),
            blocks=blocks,
            response_type=response_type,
        )
    except Exception as exc:
        _post_slash_response(
            response_url=response_url,
            text=f"Wavelength could not complete that request: {exc}",
            response_type="ephemeral",
        )


class MCPAuthMiddleware:
    def __init__(
        self,
        app,
        signing_secret: Optional[str],
        auth_token: Optional[str],
        allowed_team_ids: Optional[Set[str]] = None,
        allowed_user_ids: Optional[Set[str]] = None,
        skip_auth: bool = False,
    ):
        self.app = app
        self.signing_secret = signing_secret
        self.auth_token = auth_token
        self.allowed_team_ids = allowed_team_ids or set()
        self.allowed_user_ids = allowed_user_ids or set()
        self.skip_auth = skip_auth

    async def __call__(self, scope, receive, send):
        path = str(scope.get("path") or "")
        is_search_mcp = path.startswith("/mcp/search")
        is_slack_mcp = path.startswith("/mcp/slack")
        is_legacy_mcp = path == "/mcp"
        if scope["type"] != "http" or not (is_search_mcp or is_slack_mcp or is_legacy_mcp):
            await self.app(scope, receive, send)
            return

        if self.skip_auth:
            if is_slack_mcp:
                context_token = _MCP_SLACK_DM_CONTEXT.set(
                    _mcp_slack_dm_context(_headers_from_scope(scope))
                )
                try:
                    await self.app(scope, receive, send)
                finally:
                    _MCP_SLACK_DM_CONTEXT.reset(context_token)
            else:
                await self.app(scope, receive, send)
            return

        body = b""
        more_body = True
        while more_body:
            message = await receive()
            body += message.get("body", b"")
            more_body = message.get("more_body", False)

        headers = _headers_from_scope(scope)
        # MCP is a separate public interface. Slack signatures are accepted only
        # by the Slack routes below, never as MCP authentication.
        authenticated = self._is_bearer_token_valid(headers)
        if not authenticated:
            if not self.auth_token and not self.signing_secret:
                await self._reject(send, 500, b"MCP authentication is not configured.")
                return
            await self._reject(send, 401, b"Unauthorized MCP request.")
            return

        if not self._is_allowed_slack_identity(headers):
            await self._reject(send, 403, b"Slack identity is not allowed.")
            return

        consumed = False

        async def replay_receive():
            nonlocal consumed
            if consumed:
                return {"type": "http.request", "body": b"", "more_body": False}
            consumed = True
            return {"type": "http.request", "body": body, "more_body": False}

        if is_slack_mcp:
            context_token = _MCP_SLACK_DM_CONTEXT.set(_mcp_slack_dm_context(headers))
            try:
                await self.app(scope, replay_receive, send)
            finally:
                _MCP_SLACK_DM_CONTEXT.reset(context_token)
        else:
            await self.app(scope, replay_receive, send)

    def _is_bearer_token_valid(self, headers: Dict[str, str]) -> bool:
        if not self.auth_token:
            return False
        candidate = _extract_bearer_token(headers)
        return bool(candidate) and hmac.compare_digest(candidate, self.auth_token)

    def _is_slack_signature_valid(self, scope, headers: Dict[str, str], body: bytes) -> bool:
        if not self.signing_secret or scope.get("method") != "POST":
            return False
        return _verify_slack_signature(headers, body, self.signing_secret)

    def _is_allowed_slack_identity(self, headers: Dict[str, str]) -> bool:
        if self.allowed_team_ids:
            team_id = _slack_team_id(headers)
            # Azure's remote MCP client is not a Slack request and therefore
            # has no Slack identity headers. Bearer authentication already
            # protects that path; apply the Slack allowlist only when an
            # identity was supplied.
            if team_id is not None and team_id not in self.allowed_team_ids:
                return False
        if self.allowed_user_ids:
            user_id = _slack_user_id(headers)
            if user_id is not None and user_id not in self.allowed_user_ids:
                return False
        return True

    async def _reject(self, send, status: int, body: bytes):
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"text/plain; charset=utf-8")],
            }
        )
        await send({"type": "http.response.body", "body": body})


def _call_tool_result(payload: Dict[str, Any]):
    from mcp.types import CallToolResult, TextContent

    content = payload.get("content") or []
    text = "\n".join(item.get("text", "") for item in content if item.get("type") == "text")
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content=payload.get("structuredContent"),
        _meta=payload.get("_meta"),
    )


def create_mcp_server(service: Optional[Any] = None):
    from mcp.server.mcpserver import MCPServer
    from pydantic import Field
    from wavelength_service import WavelengthService
    from typing import Annotated, Literal

    service = service or WavelengthService.from_environment()
    mcp = MCPServer("Wavelength Archive")

    from mcp.types import ToolAnnotations
    readonly = ToolAnnotations(read_only_hint=True)

    @mcp.tool(
        name="search_archive",
        title="Search Wavelength articles, transcripts, and scripts",
        description=(
            "Search the Chicago Public Media Wavelength archive. Start here for every "
            "archive question. Use a complete semantic query, and put dates, people, "
            "program, content scope, and ordering in their dedicated parameters. Never "
            "use web-search syntax or domains in query text. content_types is limited to "
            "the literal values article, transcript, and script. Returned passages include "
            "short citation_key handles for inline answer citations."
        ),
        annotations=readonly,
    )
    def search_archive(
        query: Annotated[str, Field(description="A natural-language semantic archive query. Never include domains or web-search operators such as site:. Put filter constraints in their dedicated parameters.")],
        start_date: Annotated[Optional[str], Field(description="Optional inclusive ISO date (YYYY-MM-DD). Use for the requested time range.")] = None,
        end_date: Annotated[Optional[str], Field(description="Optional inclusive ISO date (YYYY-MM-DD). Use for the requested time range.")] = None,
        content_types: Annotated[Optional[List[Literal["article", "transcript", "script"]]], Field(description="Optional scope. Values may be article, transcript, and/or script; omit to search all.")] = None,
        authors: Annotated[Optional[List[str]], Field(description="Full names of article byline authors, when requested.")] = None,
        speakers: Annotated[Optional[List[str]], Field(description="Full names of people speaking in recordings/transcripts, when requested.")] = None,
        guests: Annotated[Optional[List[str]], Field(description="Full names of named interview guests, when requested.")] = None,
        program: Annotated[Optional[str], Field(description="Exact or likely show/program name, when requested.")] = None,
        sort: Annotated[Literal["relevance", "newest", "oldest"], Field(description="Use newest for latest/recent requests, oldest for earliest/first requests, and relevance otherwise.")] = "relevance",
        search_top: Annotated[int, Field(ge=1, le=100, description="Number of candidate logical sources to retrieve before returning results, from 1 to 100. Increase for breadth.")] = 10,
        limit: Annotated[int, Field(ge=1, le=50, description="Number of logical sources to return, from 1 to 50.")] = 8,
    ):
        return service.search_archive_data(query=query, start_date=start_date, end_date=end_date,
                                           content_types=content_types, authors=authors,
                                           speakers=speakers, guests=guests, program=program,
                                           limit=limit, sort=sort, search_top=search_top)

    @mcp.tool(
        name="keyword_search_archive",
        title="Keyword search Wavelength archive",
        description=(
            "Search the archive with lexical keyword matching. Use when exact words, "
            "titles, names, or phrases matter, or when semantic search misses likely "
            "literal matches. Put filters in dedicated parameters. Returned passages "
            "include short citation_key handles for inline answer citations."
        ),
        annotations=readonly,
    )
    def keyword_search_archive(
        query: Annotated[str, Field(description="Keyword query text. Use exact terms or phrases the archive record should contain.")],
        start_date: Annotated[Optional[str], Field(description="Optional inclusive ISO date (YYYY-MM-DD). Use for the requested time range.")] = None,
        end_date: Annotated[Optional[str], Field(description="Optional inclusive ISO date (YYYY-MM-DD). Use for the requested time range.")] = None,
        content_types: Annotated[Optional[List[Literal["article", "transcript", "script"]]], Field(description="Optional scope. Values may be article, transcript, and/or script; omit to search all.")] = None,
        authors: Annotated[Optional[List[str]], Field(description="Full names of article byline authors, when requested.")] = None,
        speakers: Annotated[Optional[List[str]], Field(description="Full names of people speaking in recordings/transcripts, when requested.")] = None,
        guests: Annotated[Optional[List[str]], Field(description="Full names of named interview guests, when requested.")] = None,
        program: Annotated[Optional[str], Field(description="Exact or likely show/program name, when requested.")] = None,
        sort: Annotated[Literal["relevance", "newest", "oldest"], Field(description="Use newest for latest/recent requests, oldest for earliest/first requests, and relevance otherwise.")] = "relevance",
        search_top: Annotated[int, Field(ge=1, le=100, description="Number of candidate logical sources to retrieve before returning results, from 1 to 100.")] = 10,
        limit: Annotated[int, Field(ge=1, le=50, description="Number of logical sources to return, from 1 to 50.")] = 8,
    ):
        return service.search_archive_data(query=query, start_date=start_date, end_date=end_date,
                                           content_types=content_types, authors=authors,
                                           speakers=speakers, guests=guests, program=program,
                                           limit=limit, sort=sort, search_top=search_top,
                                           search_mode="keyword")

    @mcp.tool(
        name="sample_archive",
        title="Sample Wavelength archive results",
        description=(
            "Retrieve a broader candidate set and return a random sample. Use for "
            "breadth, representative examples, or to avoid overfitting to the first "
            "relevance-ranked results. Returned passages include short citation_key "
            "handles for inline answer citations."
        ),
        annotations=readonly,
    )
    def sample_archive(
        query: Annotated[str, Field(description="A natural-language semantic archive query.")],
        start_date: Annotated[Optional[str], Field(description="Optional inclusive ISO date (YYYY-MM-DD).")] = None,
        end_date: Annotated[Optional[str], Field(description="Optional inclusive ISO date (YYYY-MM-DD).")] = None,
        content_types: Annotated[Optional[List[Literal["article", "transcript", "script"]]], Field(description="Optional scope. Values may be article, transcript, and/or script; omit to search all.")] = None,
        authors: Annotated[Optional[List[str]], Field(description="Full names of article byline authors, when requested.")] = None,
        speakers: Annotated[Optional[List[str]], Field(description="Full names of people speaking in recordings/transcripts, when requested.")] = None,
        guests: Annotated[Optional[List[str]], Field(description="Full names of named interview guests, when requested.")] = None,
        program: Annotated[Optional[str], Field(description="Exact or likely show/program name, when requested.")] = None,
        search_top: Annotated[int, Field(ge=1, le=100, description="Candidate logical sources to retrieve before downsampling, from 1 to 100.")] = 50,
        limit: Annotated[int, Field(ge=1, le=50, description="Random sample size to return, from 1 to 50.")] = 8,
        random_seed: Annotated[Optional[int], Field(description="Optional deterministic seed for repeatable sampling.")] = None,
    ):
        return service.search_archive_data(query=query, start_date=start_date, end_date=end_date,
                                           content_types=content_types, authors=authors,
                                           speakers=speakers, guests=guests, program=program,
                                           limit=limit, search_top=search_top,
                                           sample=True, random_seed=random_seed)

    @mcp.tool(
        name="get_full_article",
        title="Read a full Wavelength article",
        description="Retrieve the full text of an article returned by search_archive. Use when search passages are insufficient evidence; follow cursor when has_more is true. The response includes a short citation_key for the returned text segment.",
        annotations=readonly,
    )
    def get_full_article(
        source_id: Annotated[str, Field(description="source_id of an article returned by search_archive")],
        cursor: Annotated[Optional[str], Field(description="Cursor returned by an earlier call, if more text is needed")] = None,
        max_chars: Annotated[int, Field(ge=1, le=20000, description="Maximum characters to return, from 1 to 20,000")] = 12000,
    ):
        return service.get_full_article_data(source_id, cursor, max_chars)

    @mcp.tool(name="get_full_transcript", title="Get full transcript",
              description="Retrieve ordered cues from a transcript returned by search_archive. Use time bounds for a requested moment and follow cursor when has_more is true. Returned cues include short citation_key handles for inline answer citations.", annotations=readonly)
    def get_full_transcript(
        source_id: Annotated[str, Field(description="source_id of a transcript returned by search_archive")],
        cursor: Annotated[Optional[str], Field(description="Cursor returned by an earlier call, if more cues are needed")] = None,
        max_chars: Annotated[int, Field(ge=1, le=20000, description="Maximum characters to return, from 1 to 20,000")] = 12000,
        start_seconds: Annotated[Optional[float], Field(ge=0, description="Optional start of the requested transcript interval, in seconds")] = None,
        end_seconds: Annotated[Optional[float], Field(ge=0, description="Optional end of the requested transcript interval, in seconds")] = None,
    ):
        return service.get_full_transcript_data(source_id, cursor, max_chars, start_seconds, end_seconds)

    @mcp.tool(name="get_full_script", title="Get full script",
              description="Retrieve the full text of an occurrence script returned by search_archive. The response includes a short citation_key for the returned text segment.", annotations=readonly)
    def get_full_script(
        source_id: Annotated[str, Field(description="source_id of a script returned by search_archive")],
        cursor: Annotated[Optional[str], Field(description="Cursor returned by an earlier call, if more text is needed")] = None,
        max_chars: Annotated[int, Field(ge=1, le=20000, description="Maximum characters to return, from 1 to 20,000")] = 12000,
    ):
        return service.get_full_script_data(source_id, cursor, max_chars)

    return mcp


def create_slack_mcp_server():
    from mcp.server.mcpserver import MCPServer

    mcp = MCPServer("Wavelength Slack Context")

    from mcp.types import ToolAnnotations
    readonly = ToolAnnotations(read_only_hint=True)

    @mcp.tool(
        name="get_prior_slack_dms",
        title="Get earlier messages from this Slack DM",
        description=(
            "Retrieve recent direct-message context before the current top-level DM, "
            "only when needed to resolve a follow-up or reference. This tool has no "
            "archive access and takes no arguments."
        ),
        annotations=readonly,
    )
    def get_prior_slack_dms():
        context = _MCP_SLACK_DM_CONTEXT.get()
        if not context:
            return {
                "messages": [],
                "available": False,
                "note": "Prior Slack DM context is unavailable for this request.",
            }
        bot_token = os.environ.get("SLACK_BOT_TOKEN")
        if not bot_token:
            return {
                "messages": [],
                "available": False,
                "note": "Slack history is not configured.",
            }
        messages = _slack_dm_window_context(
            token=bot_token,
            channel=context["channel"],
            current_event_ts=context["event_ts"],
            current_user_id=context["user_id"],
            current_question="",
        )
        return {"messages": messages, "available": True}

    return mcp


def _verify_slack_route_request(headers: Dict[str, str], body: bytes) -> bool:
    if os.environ.get("WAVELENGTH_SKIP_SLACK_REQUEST_AUTH") == "true":
        return True
    signing_secret = os.environ.get("SLACK_SIGNING_SECRET")
    return bool(signing_secret) and _verify_slack_signature(headers, body, signing_secret)


async def _slack_request_body_and_headers(request):
    body = await request.body()
    headers = {key.lower(): value for key, value in request.headers.items()}
    return body, headers


def _slack_event_channel(payload: Dict[str, Any]) -> Optional[str]:
    channel = payload.get("channel")
    if isinstance(channel, dict):
        channel = channel.get("id")
    channel = channel or payload.get("channel_id")
    if channel:
        return str(channel)
    for key in ("event", "item", "assistant_thread", "message", "previous_message"):
        value = payload.get(key)
        if isinstance(value, dict):
            channel = _slack_event_channel(value)
            if channel:
                return channel
    container = payload.get("container") or {}
    if isinstance(container, dict) and container.get("channel_id"):
        return str(container["channel_id"])
    return None


def _slack_event_ts(payload: Dict[str, Any]) -> Optional[str]:
    ts = payload.get("ts") or payload.get("event_ts")
    if ts:
        return str(ts)
    for key in ("event", "message", "previous_message"):
        value = payload.get(key)
        if isinstance(value, dict):
            ts = _slack_event_ts(value)
            if ts:
                return ts
    return str(ts) if ts else None


def _slack_event_thread_ts(payload: Dict[str, Any]) -> Optional[str]:
    thread_ts = payload.get("thread_ts")
    if thread_ts:
        return str(thread_ts)
    for key in ("event", "assistant_thread", "message", "previous_message"):
        value = payload.get(key)
        if isinstance(value, dict):
            thread_ts = _slack_event_thread_ts(value)
            if thread_ts:
                return thread_ts
    return _slack_event_ts(payload)


def _add_slack_routes(app, service) -> None:
    from starlette.background import BackgroundTask
    from starlette.responses import JSONResponse, PlainTextResponse

    async def slack_health(request):
        checks = {
            "sqlite": bool(os.environ.get("WAVELENGTH_STATE_DB", "data/wavelength-state.sqlite3")),
            "azure_search": bool(os.environ.get("AZURE_SEARCH_ENDPOINT") and os.environ.get("AZURE_SEARCH_INDEX_NAME")),
            "azure_openai": bool(os.environ.get("AZURE_OPENAI_ENDPOINT") and os.environ.get("CHATGPT_DEPLOYMENT_NAME")),
            "mcp_url": bool(os.environ.get("WAVELENGTH_PUBLIC_BASE_URL")),
        }
        return JSONResponse(
            {
                "ok": all(checks.values()),
                "service": "wavelength-slack-agent",
                "code_version": MCP_SERVER_CODE_VERSION,
                "ready": checks,
            }
        )

    async def slack_events(request):
        request_started_at = time.time()
        body, headers = await _slack_request_body_and_headers(request)
        body_read_at = time.time()
        payload = json.loads(body.decode("utf-8") or "{}")
        _log_verbose("Slack Events inbound payload", payload)
        if not _verify_slack_route_request(headers, body):
            return PlainTextResponse("Invalid Slack signature.", status_code=401)
        _cache_team_api_app_id(payload.get("team_id"), payload.get("api_app_id"))

        if payload.get("type") == "url_verification":
            return PlainTextResponse(str(payload.get("challenge", "")))

        if payload.get("type") != "event_callback":
            return JSONResponse({"ok": True})

        event = payload.get("event") or {}
        event_type = event.get("type")
        event_message = event.get("message") if isinstance(event.get("message"), dict) else event
        event_metadata = event_message.get("metadata") if isinstance(event_message, dict) else None
        event_metadata_entities = (
            event_metadata.get("entities")
            if isinstance(event_metadata, dict) and isinstance(event_metadata.get("entities"), list)
            else []
        )
        logger.info(
            "Received Slack event: code_version=%s event_type=%s subtype=%s "
            "team_id=%s api_app_id=%s event_id=%s event_time=%s "
            "slack_request_timestamp=%s slack_request_age_ms=%s "
            "event_time_age_ms=%s body_read_ms=%d "
            "message_metadata=%s work_object_entities=%d payload_keys=%s event_keys=%s",
            MCP_SERVER_CODE_VERSION,
            event_type,
            event.get("subtype"),
            payload.get("team_id"),
            payload.get("api_app_id"),
            payload.get("event_id"),
            payload.get("event_time"),
            headers.get("x-slack-request-timestamp"),
            _epoch_seconds_age_ms(headers.get("x-slack-request-timestamp"), body_read_at),
            _epoch_seconds_age_ms(payload.get("event_time"), body_read_at),
            int((body_read_at - request_started_at) * 1000),
            isinstance(event_metadata, dict),
            len(event_metadata_entities),
            sorted(payload.keys()),
            sorted(event.keys()) if isinstance(event, dict) else [],
        )
        if event_type == "app_home_opened":
            bot_token = os.environ.get("SLACK_BOT_TOKEN")
            if not bot_token:
                return PlainTextResponse("SLACK_BOT_TOKEN is not configured.", status_code=500)
            user_id = _slack_payload_user_id(event.get("user"))
            if not user_id:
                return JSONResponse({"ok": True})
            if not _is_wavelength_allowed_slack_user(user_id):
                logger.info("Ignoring Wavelength Home open for non-allowed user %s.", user_id)
                return JSONResponse({"ok": True})
            return JSONResponse(
                {"ok": True},
                background=BackgroundTask(
                    _publish_wavelength_home_tab,
                    token=bot_token,
                    user_id=user_id,
                    team_id=str(event.get("team_id") or payload.get("team_id") or "unknown"),
                ),
            )

        # Work Object flexpane requests can include bot context; do not short
        # circuit these based on `bot_id` or `subtype`.
        if event_type == "entity_details_requested":
            user_id = _slack_payload_user_id(event.get("user"))
            if not _is_wavelength_allowed_slack_user(user_id):
                logger.info(
                    "Ignoring Work Object details request for non-allowed user %s.",
                    user_id,
                )
                return JSONResponse({"ok": True})
            # Present details quickly: trigger_id expires fast. Prefer cached
            # entity metadata from the message we posted, and fall back to a
            # minimal view derived from the event payload.
            logger.info(
                "Work Object details requested: team_id=%s api_app_id=%s event_id=%s "
                "event_ts=%s trigger_id_present=%s trigger_id_prefix=%s "
                "external_ref=%s entity_url=%s app_unfurl_url=%s link=%s "
                "channel=%s message_ts=%s thread_ts=%s user=%s user_locale=%s authorizations=%s",
                payload.get("team_id"),
                payload.get("api_app_id"),
                payload.get("event_id"),
                event.get("event_ts"),
                bool(event.get("trigger_id")),
                str(event.get("trigger_id") or "")[:18],
                event.get("external_ref"),
                event.get("entity_url"),
                event.get("app_unfurl_url"),
                event.get("link"),
                event.get("channel"),
                event.get("message_ts"),
                event.get("thread_ts"),
                event.get("user"),
                event.get("user_locale"),
                payload.get("authorizations"),
            )
            bot_token = os.environ.get("SLACK_BOT_TOKEN")
            if bot_token:
                start_time = time.time()
                source_id = _work_object_source_id_from_event(event)
                cached = _cached_work_object_entity(str(source_id) if source_id else None)
                work_object = cached or {}
                cache_hit = bool(cached)
                logger.info(
                    "Work Object cache lookup: source_id=%s cache_hit=%s cache_size=%d "
                    "request_age_ms=%s event_time_age_ms=%s pre_present_ms=%d",
                    source_id,
                    cache_hit,
                    len(_WORK_OBJECT_ENTITY_CACHE),
                    _epoch_seconds_age_ms(headers.get("x-slack-request-timestamp"), time.time()),
                    _epoch_seconds_age_ms(payload.get("event_time"), time.time()),
                    int((time.time() - request_started_at) * 1000),
                )

                entity_url = event.get("entity_url") or (event.get("link") or {}).get("url")
                source_detail = _cached_work_object_source(str(source_id) if source_id else None)
                if not source_detail:
                    raw_source = None
                    if source_id:
                        try:
                            raw_source = service.engine.get_source(str(source_id))
                        except Exception:
                            logger.exception(
                                "Could not fetch Work Object source by id: source_id=%s",
                                source_id,
                            )
                    if not raw_source and entity_url:
                        try:
                            raw_source = service.engine.get_source_by_url(str(entity_url))
                        except Exception:
                            logger.exception(
                                "Could not fetch Work Object source by URL: source_id=%s entity_url=%s",
                                source_id,
                                entity_url,
                            )
                    if raw_source:
                        from slack_format import source_reference

                        source_detail = source_reference(raw_source, 1)
                        _cache_work_object_sources([source_detail])
                        logger.info(
                            "Reconstituted Work Object source detail: source_id=%s "
                            "entity_url_present=%s content_type=%s title=%s",
                            source_id,
                            bool(entity_url),
                            source_detail.get("content_type"),
                            source_detail.get("title"),
                        )

                if not work_object:
                    work_object = {
                        "entity_type": "slack#/entities/item",
                        "url": entity_url,
                        "external_ref": {"id": str(source_id)} if source_id is not None else {},
                        "entity_payload": {
                            "attributes": {
                                "title": {"text": f"Archive source {source_id}"},
                                "display_id": str(source_id),
                            },
                            "custom_fields": [
                                *([
                                    {
                                        "key": "source_url",
                                        "label": "Source",
                                        "value": entity_url,
                                        "type": "slack#/types/link",
                                    }
                                ] if entity_url else [])
                            ],
                        },
                    }
                if source_detail:
                    from slack_format import work_object_entities

                    detail_objects = work_object_entities(
                        [source_detail],
                        include_full_text=True,
                        include_excerpt=True,
                        include_collectiveaccess=True,
                    )
                    if detail_objects:
                        work_object = detail_objects[0]

                if source_detail and source_detail.get("full_text"):
                    entity_payload = (
                        work_object.get("entity_payload")
                        if isinstance(work_object.get("entity_payload"), dict)
                        else {}
                    )
                    custom_fields = list(entity_payload.get("custom_fields") or [])
                    custom_fields = [
                        field
                        for field in custom_fields
                        if not (isinstance(field, dict) and field.get("key") == "description")
                    ]
                    from slack_format import work_object_description_text

                    custom_fields.insert(0, {
                        "key": "description",
                        "label": "Description",
                        "value": work_object_description_text(source_detail.get("full_text")),
                        "type": "string",
                        "format": "markdown"
                    })
                    display_order = list(entity_payload.get("display_order") or [])
                    display_order = [key for key in display_order if key != "description"]
                    work_object = {
                        **work_object,
                        "entity_payload": {
                            **entity_payload,
                            "custom_fields": custom_fields,
                            "display_order": ["description", *display_order],
                        },
                    }

                present_payload = {
                    "trigger_id": event.get("trigger_id"),
                    "metadata": {
                        "entity_type": work_object.get("entity_type"),
                        "url": work_object.get("url"),
                        "app_unfurl_url": work_object.get("app_unfurl_url") or work_object.get("url"),
                        "external_ref": work_object.get("external_ref"),
                        "entity_payload": work_object.get("entity_payload"),
                    },
                }
                metadata = present_payload["metadata"]
                entity_payload = metadata.get("entity_payload") if isinstance(metadata, dict) else {}
                attributes = entity_payload.get("attributes") if isinstance(entity_payload, dict) else {}
                custom_fields = entity_payload.get("custom_fields") if isinstance(entity_payload, dict) else []
                logger.info(
                    "Calling entity.presentDetails: source_id=%s cache_hit=%s "
                    "metadata_entity_type=%s metadata_url=%s metadata_external_ref=%s "
                    "title=%s display_id=%s custom_field_keys=%s request_age_ms=%s pre_call_ms=%d",
                    source_id,
                    cache_hit,
                    metadata.get("entity_type") if isinstance(metadata, dict) else None,
                    metadata.get("url") if isinstance(metadata, dict) else None,
                    metadata.get("external_ref") if isinstance(metadata, dict) else None,
                    (attributes.get("title") or {}).get("text") if isinstance(attributes.get("title"), dict) else None,
                    attributes.get("display_id") if isinstance(attributes, dict) else None,
                    [
                        field.get("key")
                        for field in custom_fields
                        if isinstance(field, dict)
                    ] if isinstance(custom_fields, list) else None,
                    _epoch_seconds_age_ms(headers.get("x-slack-request-timestamp"), time.time()),
                    int((time.time() - request_started_at) * 1000),
                )
                try:
                    await _call_slack_api_async(token=bot_token, method="entity.presentDetails", payload=present_payload)
                    logger.info(
                        "Presented Work Object details: source_id=%s cache_hit=%s duration_ms=%d total_ms=%d",
                        source_id,
                        cache_hit,
                        int((time.time() - start_time) * 1000),
                        int((time.time() - request_started_at) * 1000),
                    )
                except Exception as exc:
                    if "invalid_trigger_id" in str(exc):
                        try:
                            auth = await _call_slack_api_async(token=bot_token, method="auth.test", payload={})
                            logger.error(
                                "Slack bot token identity after invalid_trigger_id: "
                                "incoming_team_id=%s incoming_api_app_id=%s auth_team_id=%s "
                                "auth_user_id=%s auth_bot_id=%s auth_app_id=%s auth_team=%s auth_url=%s",
                                payload.get("team_id"),
                                payload.get("api_app_id"),
                                auth.get("team_id"),
                                auth.get("user_id"),
                                auth.get("bot_id"),
                                auth.get("app_id"),
                                auth.get("team"),
                                auth.get("url"),
                            )
                        except Exception:
                            logger.exception("Could not run Slack auth.test after invalid_trigger_id.")
                    # A trigger is single-use. A 500 would make Slack retry
                    # the event with the same, now-expired trigger ID.
                    logger.exception(
                        "Could not present Work Object details: source_id=%s cache_hit=%s "
                        "duration_ms=%d total_ms=%d request_age_ms=%s",
                        source_id,
                        cache_hit,
                        int((time.time() - start_time) * 1000),
                        int((time.time() - request_started_at) * 1000),
                        _epoch_seconds_age_ms(headers.get("x-slack-request-timestamp"), time.time()),
                    )
            else:
                logger.error(
                    "Cannot present Work Object details: SLACK_BOT_TOKEN is not configured "
                    "team_id=%s api_app_id=%s event_id=%s",
                    payload.get("team_id"),
                    payload.get("api_app_id"),
                    payload.get("event_id"),
                )
            return JSONResponse({"ok": True})

        if event.get("bot_id") or event.get("subtype") == "bot_message":
            return JSONResponse({"ok": True})

        if event_type not in {"app_mention", "message"}:
            return JSONResponse({"ok": True})
        if event_type == "message" and event.get("channel_type") != "im":
            return JSONResponse({"ok": True})

        user_id = _slack_payload_user_id(event.get("user"))
        if not user_id:
            logger.info(
                "Ignoring Wavelength Slack event without a user id: event_type=%s text=%r.",
                event_type,
                str(event.get("text") or "")[:500],
            )
            return JSONResponse({"ok": True})
        if not _is_wavelength_allowed_slack_user(user_id):
            logger.info(
                "Ignoring Wavelength Slack event for non-allowed user %s: event_type=%s.",
                user_id,
                event_type,
            )
            return JSONResponse({"ok": True})

        bot_token = os.environ.get("SLACK_BOT_TOKEN")
        if not bot_token:
            return PlainTextResponse("SLACK_BOT_TOKEN is not configured.", status_code=500)

        question = _strip_bot_mentions(event.get("text", ""))
        if not question:
            return JSONResponse({"ok": True})

        channel = _slack_event_channel(payload)
        if not channel:
            logger.warning(
                "Ignoring Slack event without channel identifier: event_type=%s event_ts=%s keys=%s",
                event_type,
                event.get("event_ts") or event.get("ts"),
                sorted(event.keys()),
            )
            return JSONResponse({"ok": True})

        event_ts = _slack_event_ts(event)
        incoming_thread_ts = _slack_event_thread_ts(payload)
        if question == "logo test 1":
            logo_payload = _logo_test_registration_probe_payload()
            logger.info(
                "Posting logo test 1 registration probe: team_id=%s api_app_id=%s "
                "thread_ts=%s probe_id=%s metadata_entities=%d",
                payload.get("team_id"),
                payload.get("api_app_id"),
                incoming_thread_ts,
                LOGO_TEST_PROBE_ID,
                len((logo_payload.get("metadata") or {}).get("entities") or []),
            )
            return JSONResponse(
                {"ok": True},
                background=BackgroundTask(
                    _post_slack_message,
                    token=bot_token,
                    channel=channel,
                    text=logo_payload["text"],
                    thread_ts=incoming_thread_ts,
                    metadata=logo_payload["metadata"],
                    unfurl_links=False,
                    unfurl_media=False,
                    fetch_after_post=True,
                ),
            )
        if question == "logo test 2":
            work_object_app_id = payload.get("api_app_id") or _team_api_app_id(payload.get("team_id"))
            logo_payload = _logo_test_work_object_mention_probe_payload(work_object_app_id)
            logger.info(
                "Posting logo test 2 Work Object mention probe: team_id=%s api_app_id=%s "
                "work_object_app_id=%s thread_ts=%s probe_id=%s blocks=%d",
                payload.get("team_id"),
                payload.get("api_app_id"),
                work_object_app_id,
                incoming_thread_ts,
                LOGO_TEST_PROBE_ID,
                len(logo_payload.get("blocks") or []),
            )
            return JSONResponse(
                {"ok": True},
                background=BackgroundTask(
                    _post_slack_message,
                    token=bot_token,
                    channel=channel,
                    text=logo_payload["text"],
                    blocks=logo_payload.get("blocks"),
                    thread_ts=incoming_thread_ts,
                    unfurl_links=False,
                    unfurl_media=False,
                ),
            )
        if question == "logo test 3":
            work_object_app_id = payload.get("api_app_id") or _team_api_app_id(payload.get("team_id"))
            logo_payload = _logo_test_work_object_mention_probe_payload(work_object_app_id, ref='Ev0BULNW89S6')
            logger.info(
                "Posting logo test 2 Work Object mention probe: team_id=%s api_app_id=%s "
                "work_object_app_id=%s thread_ts=%s probe_id=%s blocks=%d",
                payload.get("team_id"),
                payload.get("api_app_id"),
                work_object_app_id,
                incoming_thread_ts,
                LOGO_TEST_PROBE_ID,
                len(logo_payload.get("blocks") or []),
            )
            return JSONResponse(
                {"ok": True},
                background=BackgroundTask(
                    _post_slack_message,
                    token=bot_token,
                    channel=channel,
                    text=logo_payload["text"],
                    blocks=logo_payload.get("blocks"),
                    thread_ts=incoming_thread_ts,
                    unfurl_links=False,
                    unfurl_media=False,
                ),
            )
        if question == "logo test":
            logo_payload = _logo_test_payload(payload.get("api_app_id") or _team_api_app_id(payload.get("team_id")))
            logger.info(
                "Posting logo test response: team_id=%s api_app_id=%s work_object_app_id=%s "
                "thread_ts=%s metadata_entities=%d selected_icons=%s",
                payload.get("team_id"),
                payload.get("api_app_id"),
                payload.get("api_app_id") or _team_api_app_id(payload.get("team_id")),
                incoming_thread_ts,
                len((logo_payload.get("metadata") or {}).get("entities") or []),
                logo_payload.get("selected_icons"),
            )
            return JSONResponse(
                {"ok": True},
                background=BackgroundTask(
                    _post_slack_message,
                    token=bot_token,
                    channel=channel,
                    text=logo_payload["text"],
                    blocks=logo_payload["blocks"],
                    thread_ts=incoming_thread_ts,
                    metadata=logo_payload["metadata"],
                ),
            )
        assistant_thread = event.get("assistant_thread") if isinstance(event.get("assistant_thread"), dict) else {}
        explicit_thread_ts = event.get("thread_ts") or assistant_thread.get("thread_ts")
        is_top_level_dm = event_type == "message" and event.get("channel_type") == "im" and not explicit_thread_ts
        is_new_mention_thread = event_type == "app_mention" and not explicit_thread_ts
        # Thinking Steps streams require a concrete parent message. In a DM,
        # use the incoming message as that parent and keep the exchange in its
        # reply thread.
        response_thread_ts = incoming_thread_ts
        persist_thread_ts = SLACK_DM_WINDOW_CONTEXT_TS if is_top_level_dm else incoming_thread_ts
        logger.info(
            "Resolved Slack event identifiers: code_version=%s channel=%s thread_ts=%s event_ts=%s dm_window=%s",
            MCP_SERVER_CODE_VERSION,
            channel,
            incoming_thread_ts,
            event_ts,
            is_top_level_dm,
        )

        return JSONResponse(
            {"ok": True},
            background=BackgroundTask(
                _handle_event_question,
                service=service,
                bot_token=bot_token,
                question=question,
                channel=channel,
                user_id=user_id,
                team_id=event.get("team") or payload.get("team_id"),
                thread_ts=response_thread_ts,
                event_ts=event_ts,
                persist_thread_ts=persist_thread_ts,
                use_dm_window_context=is_top_level_dm,
                include_alpha_notice=is_top_level_dm or is_new_mention_thread,
            ),
        )

    async def wavelength_command(request):
        body, headers = await _slack_request_body_and_headers(request)
        if not _verify_slack_route_request(headers, body):
            return PlainTextResponse("Invalid Slack signature.", status_code=401)

        form = {
            key: values[0] if values else ""
            for key, values in parse_qs(body.decode("utf-8")).items()
        }
        _cache_team_api_app_id(form.get("team_id"), form.get("api_app_id"))
        _log_verbose(
            "Slack slash command inbound form",
            {key: ("REDACTED" if key == "response_url" else value) for key, value in form.items()},
        )
        user_id = form.get("user_id")
        if not _is_wavelength_allowed_slack_user(user_id):
            logger.info("Rejecting Wavelength slash command for non-allowed user %s.", user_id)
            return JSONResponse(
                {
                    "response_type": "ephemeral",
                    "text": WAVELENGTH_ACCESS_DENIED_TEXT,
                }
            )

        question = (form.get("text") or "").strip()
        if not question:
            return JSONResponse(
                {
                    "response_type": "ephemeral",
                    "text": WAVELENGTH_ALPHA_NOTICE,
                }
            )

        response_url = form.get("response_url")
        if not response_url:
            return PlainTextResponse("Slack response_url is missing.", status_code=400)

        response_type = os.environ.get("WAVELENGTH_SLASH_RESPONSE_TYPE", "ephemeral")
        if response_type not in {"ephemeral", "in_channel"}:
            response_type = "ephemeral"

        bot_token = os.environ.get("SLACK_BOT_TOKEN")

        return JSONResponse(
            {
                "response_type": "ephemeral",
                "text": WAVELENGTH_ALPHA_NOTICE,
            },
            background=BackgroundTask(
                _handle_slash_question,
                service=service,
                question=question,
                response_url=response_url,
                bot_token=bot_token,
                channel=form.get("channel_id"),
                user_id=user_id,
                team_id=form.get("team_id"),
                response_type=response_type,
            ),
        )

    async def slack_interactions(request):
        body, headers = await _slack_request_body_and_headers(request)
        if not _verify_slack_route_request(headers, body):
            return PlainTextResponse("Invalid Slack signature.", status_code=401)
        form = parse_qs(body.decode("utf-8"))
        raw = (form.get("payload") or ["{}"]) [0]
        payload = json.loads(raw)
        team = payload.get("team") or {}
        _cache_team_api_app_id(team.get("id") or payload.get("team_id"), payload.get("api_app_id"))
        user = payload.get("user") or {}
        user_id = _slack_payload_user_id(user)
        if not _is_wavelength_allowed_slack_user(user_id):
            logger.info("Ignoring Wavelength interaction for non-allowed user %s.", user_id)
            return JSONResponse({"ok": True, "text": WAVELENGTH_ACCESS_DENIED_TEXT})
        action = (payload.get("actions") or [{}])[0]
        action_id = str(action.get("action_id") or "")
        if action_id.startswith("wavelength_example:"):
            examples = {"topic": "Find coverage of a topic and date", "compare": "Compare coverage over time", "guest": "Find where a guest or speaker discussed a topic"}
            return JSONResponse({"text": examples.get(action_id.split(":", 1)[1], "Ask a Wavelength archive question.")})
        if action_id.startswith("wavelength_save:") or action_id.startswith("wavelength_unsave:"):
            from state import StateStore
            source_id = action_id.split(":", 1)[1]
            store = StateStore()
            args = (str(team.get("id") or "unknown"), str(user.get("id") or "unknown"), source_id)
            if action_id.startswith("wavelength_save:"):
                store.save_source(*args, {"source_id": source_id, "title": action.get("value") or source_id})
            else:
                store.unsave_source(*args)
            token = os.environ.get("SLACK_BOT_TOKEN")
            if token and user.get("id"):
                _publish_wavelength_home_tab(token=token, user_id=str(user["id"]))
        if action_id.startswith("open_source:"):
            source_id = action_id.split(":", 1)[1]
            source = service.engine.get_source(source_id)
            token = os.environ.get("SLACK_BOT_TOKEN")
            if token and source:
                from slack_format import source_reference, work_object_entities

                work_objects = work_object_entities(
                    [source_reference(source, 1)],
                    include_full_text=True,
                    include_excerpt=True,
                    include_collectiveaccess=True,
                )
                if work_objects:
                    work_object = work_objects[0]
                    try:
                        await _call_slack_api_async(token=token, method="entity.presentDetails", payload={
                            "trigger_id": payload.get("trigger_id"),
                            "metadata": {
                                "entity_type": work_object["entity_type"],
                                "url": work_object["url"],
                                "external_ref": work_object["external_ref"],
                                "entity_payload": work_object["entity_payload"],
                            },
                        })
                    except Exception:
                        # This generic card action is not guaranteed to provide a
                        # Work Object trigger. Do not make Slack retry it.
                        logger.exception(
                            "Could not present card source as Work Object: source_id=%s",
                            source_id,
                        )
        return JSONResponse({"ok": True})

    app.add_route("/slack/health", slack_health, methods=["GET"])
    app.add_route("/slack/events", slack_events, methods=["POST"])
    app.add_route("/slack/interactions", slack_interactions, methods=["POST"])
    app.add_route("/slack/commands/wavelength", wavelength_command, methods=["POST"])

    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
    # Home is published on app_home_opened and state-changing interactions only.


def _add_startup_handler(app, handler) -> None:
    if hasattr(app, "add_event_handler"):
        app.add_event_handler("startup", handler)
        return

    router = getattr(app, "router", None)
    lifespan_context = getattr(router, "lifespan_context", None)
    if router is None or lifespan_context is None:
        logger.warning("Could not register startup handler; app has no compatible lifespan API.")
        return

    @asynccontextmanager
    async def lifespan_with_startup(current_app):
        async with lifespan_context(current_app) as state:
            handler()
            yield state

    router.lifespan_context = lifespan_with_startup


def create_app():
    from wavelength_service import WavelengthService
    from starlette.middleware.base import BaseHTTPMiddleware

    logger.warning(
        "Starting Wavelength Slack app: code_version=%s module_file=%s",
        MCP_SERVER_CODE_VERSION,
        __file__,
    )
    service = WavelengthService.from_environment()
    search_mcp = create_mcp_server(service)
    slack_mcp = create_slack_mcp_server()
    app = search_mcp.streamable_http_app(
        streamable_http_path="/mcp/search",
        json_response=True,
        stateless_http=True,
        host=os.environ.get("WAVELENGTH_MCP_HOST", "127.0.0.1"),
    )
    slack_mcp_app = slack_mcp.streamable_http_app(
        streamable_http_path="/mcp/slack",
        json_response=True,
        stateless_http=True,
        host=os.environ.get("WAVELENGTH_MCP_HOST", "127.0.0.1"),
    )
    app.router.routes.extend(slack_mcp_app.router.routes)
    search_lifespan_context = app.router.lifespan_context
    slack_lifespan_context = slack_mcp_app.router.lifespan_context

    @asynccontextmanager
    async def combined_mcp_lifespan(current_app):
        async with search_lifespan_context(current_app) as state:
            async with slack_lifespan_context(slack_mcp_app):
                yield state

    app.router.lifespan_context = combined_mcp_lifespan
    app.add_route(
        "/mcp",
        lambda request: PlainTextResponse("Wavelength MCP moved to /mcp/search and /mcp/slack.", status_code=404),
        methods=["GET", "POST"],
    )
    _add_slack_routes(app, service)

    class StaticLogMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            start_time = time.time()
            response = await call_next(request)
            try:
                path = getattr(request, "url", None).path if getattr(request, "url", None) else ""
                if isinstance(path, str) and path.startswith("/static/"):
                    logger.info(
                        "Static request: path=%s status=%s duration_ms=%d content_type=%s content_length=%s",
                        path,
                        getattr(response, "status_code", None),
                        int((time.time() - start_time) * 1000),
                        (getattr(response, "headers", {}) or {}).get("content-type"),
                        (getattr(response, "headers", {}) or {}).get("content-length"),
                    )
            except Exception:
                logger.exception("Static request logging failed")
            return response

    app.add_middleware(StaticLogMiddleware)
    app.add_middleware(
        MCPAuthMiddleware,
        signing_secret=os.environ.get("SLACK_SIGNING_SECRET"),
        auth_token=os.environ.get("WAVELENGTH_MCP_AUTH_TOKEN"),
        allowed_team_ids=_split_env_list(os.environ.get("WAVELENGTH_ALLOWED_SLACK_TEAM_IDS")),
        allowed_user_ids=_split_env_list(os.environ.get("WAVELENGTH_ALLOWED_SLACK_USER_IDS")),
        skip_auth=(
            os.environ.get("WAVELENGTH_SKIP_MCP_AUTH") == "true"
            or os.environ.get("WAVELENGTH_SKIP_SLACK_SIGNATURE_VERIFY") == "true"
        ),
    )
    return app


_configure_console_logging_from_argv()

try:
    app = create_app()
except ModuleNotFoundError as exc:
    optional_runtime_modules = {
        "azure",
        "dotenv",
        "mcp",
        "openai",
        "wavelength_service",
    }
    if exc.name not in optional_runtime_modules:
        raise
    app = None
except KeyError as exc:
    logger.warning("Skipping Wavelength app startup; missing environment variable %s.", exc)
    app = None


if __name__ == "__main__":
    if app is None:
        raise RuntimeError('The "mcp" package is required. Install requirements.txt first.')
    import uvicorn

    uvicorn.run(
        "mcp_server:app",
        host=os.environ.get("WAVELENGTH_HTTP_HOST", "127.0.0.1"),
        port=int(os.environ.get("WAVELENGTH_HTTP_PORT", "8000")),
        reload=False,
    )
