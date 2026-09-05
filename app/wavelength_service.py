from datetime import datetime
import json
import logging
import os
import re
import random
from hashlib import sha256
from typing import Any, Dict, Iterator, List, Optional
from pathlib import Path
from urllib.parse import urlencode

from dotenv import load_dotenv

from dewey import Dewey
from models import AzureOpenAIConfig, AzureSearchConfig
from models.core import resolve_embedding_dimensions
from slack_format import (
    answer_blocks,
    block_kit_tool_result,
    cited_source_references,
    markdown_to_slack_mrkdwn,
    replace_source_markers,
    search_blocks,
    slack_link_url,
    source_references,
    truncate_text,
)
from search_plan import (
    DEFAULT_SEARCH_STEPS,
    archive_search_instructions,
    search_details,
    search_signature,
)
from tools import load_answer_prompt


logger = logging.getLogger(__name__)
CITATION_MARKER_PATTERN = re.compile(r"\{\{cite:(?P<key>c_[A-Za-z0-9]+)\}\}")


def _verbose_logging_enabled() -> bool:
    return os.environ.get("WAVELENGTH_VERBOSE_LOGS") == "true"


def _source_logging_enabled() -> bool:
    return (
        os.environ.get("WAVELENGTH_LOG_SOURCES") == "true"
        or _verbose_logging_enabled()
    )


def _log_verbose(label: str, value: Any) -> None:
    if not _verbose_logging_enabled():
        return
    logger.debug(
        "%s:\n%s",
        label,
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
    )


def _source_summary(source: Dict[str, Any]) -> Dict[str, Any]:
    passages = []
    for passage in source.get("passages") or []:
        if not isinstance(passage, dict):
            continue
        passages.append({
            "passage_id": passage.get("passage_id"),
            "passage_ids": passage.get("passage_ids") or [passage.get("passage_id")],
            "start_seconds": passage.get("start_seconds"),
            "end_seconds": passage.get("end_seconds"),
            "speakers": passage.get("speakers") or [],
        })
    return {
        "source_id": source.get("source_id") or source.get("id"),
        "content_type": source.get("content_type"),
        "title": source.get("title") or source.get("headline") or source.get("transcript_name"),
        "published_at": source.get("published_at") or source.get("publish_date"),
        "url": source.get("url") or source.get("canonical_url") or source.get("transcript_url"),
        "program": source.get("program"),
        "authors": source.get("authors") or [],
        "speakers": source.get("speakers") or [],
        "guests": source.get("guests") or [],
        "passages": passages,
    }


def _log_sources(label: str, value: Any, *, level: int = logging.INFO) -> None:
    if not _source_logging_enabled():
        return
    logger.log(
        level,
        "%s:\n%s",
        label,
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
    )


def _citation_key(source_id: Any, passage_id: Any) -> str:
    raw = f"{source_id or ''}\0{passage_id or ''}"
    return "c_" + sha256(raw.encode("utf-8", errors="ignore")).hexdigest()[:8]


def _citation_map_for_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    seen = set()
    for source in sources:
        if not isinstance(source, dict):
            continue
        source_id = source.get("source_id") or source.get("id")
        for passage in source.get("passages") or []:
            if not isinstance(passage, dict):
                continue
            citation_key = passage.get("citation_key") or _citation_key(
                source_id,
                passage.get("passage_id"),
            )
            if citation_key in seen:
                continue
            seen.add(citation_key)
            entries.append({
                "citation_key": citation_key,
                "source_id": source_id,
                "passage_id": passage.get("passage_id"),
                "passage_ids": passage.get("passage_ids") or [passage.get("passage_id")],
                "content_type": source.get("content_type"),
                "title": source.get("title") or source.get("headline") or source.get("transcript_name"),
                "published_at": source.get("published_at") or source.get("publish_date"),
                "url": source.get("url") or source.get("canonical_url") or source.get("transcript_url"),
                "listen_url": passage.get("timestamp_url") or source.get("listen_url") or source.get("transcript_url"),
                "program": source.get("program"),
                "parent_id": source.get("parent_id"),
                "occurrence_id": source.get("occurrence_id"),
                "authors": source.get("authors") or [],
                "speakers": passage.get("speakers") or source.get("speakers") or [],
                "guests": source.get("guests") or [],
                "start_seconds": passage.get("start_seconds"),
                "end_seconds": passage.get("end_seconds"),
                "excerpt": truncate_text(passage.get("text") or "", 1200),
                "raw_vtt_excerpt": passage.get("raw_vtt_excerpt"),
            })
    return entries


def _segment_citation_entry(
    source: Dict[str, Any],
    citation_key: str,
    passage_id: str,
) -> Dict[str, Any]:
    return {
        "citation_key": citation_key,
        "source_id": source.get("source_id") or source.get("id"),
        "passage_id": passage_id,
        "passage_ids": [passage_id],
        "content_type": source.get("content_type"),
        "title": source.get("title") or source.get("headline") or source.get("transcript_name"),
        "published_at": source.get("published_at") or source.get("publish_date"),
        "url": source.get("url") or source.get("canonical_url") or source.get("transcript_url"),
        "listen_url": source.get("listen_url") or source.get("transcript_url"),
        "program": source.get("program"),
        "parent_id": source.get("parent_id"),
        "occurrence_id": source.get("occurrence_id"),
        "authors": source.get("authors") or [],
        "speakers": source.get("speakers") or [],
        "guests": source.get("guests") or [],
        "start_seconds": None,
        "end_seconds": None,
        "excerpt": truncate_text(source.get("content") or source.get("excerpt") or "", 1200),
        "raw_vtt_excerpt": source.get("raw_vtt_excerpt"),
    }


def _source_from_citation_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source_id": entry.get("source_id"),
        "content_type": entry.get("content_type"),
        "title": entry.get("title") or "Untitled source",
        "published_at": entry.get("published_at"),
        "url": entry.get("url") or entry.get("listen_url"),
        "listen_url": entry.get("listen_url"),
        "program": entry.get("program"),
        "parent_id": entry.get("parent_id"),
        "occurrence_id": entry.get("occurrence_id"),
        "authors": entry.get("authors") or [],
        "speakers": entry.get("speakers") or [],
        "guests": entry.get("guests") or [],
        "content": entry.get("excerpt") or "",
        "excerpt": entry.get("excerpt") or "",
        "raw_vtt_excerpt": entry.get("raw_vtt_excerpt"),
        "passages": [
            {
                "passage_id": entry.get("passage_id"),
                "passage_ids": entry.get("passage_ids") or [entry.get("passage_id")],
                "text": entry.get("excerpt") or "",
                "raw_vtt_excerpt": entry.get("raw_vtt_excerpt"),
                "start_seconds": entry.get("start_seconds"),
                "end_seconds": entry.get("end_seconds"),
                "timestamp_url": entry.get("listen_url"),
            }
        ],
    }


def _citation_map_summary(citation_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "citation_key": key,
            "source_id": entry.get("source_id"),
            "passage_id": entry.get("passage_id"),
            "passage_ids": entry.get("passage_ids") or [],
            "title": entry.get("title"),
            "published_at": entry.get("published_at"),
            "url": entry.get("url"),
            "program": entry.get("program"),
            "start_seconds": entry.get("start_seconds"),
            "end_seconds": entry.get("end_seconds"),
        }
        for key, entry in citation_map.items()
    ]


def _answer_text_from_structured_answer(answer: Any) -> str:
    if isinstance(answer, str):
        return answer.strip()
    if not isinstance(answer, dict):
        return ""
    text = answer.get("answer")
    if isinstance(text, str) and text.strip():
        return text.strip()
    # Backward-compatible fallback for earlier claim-shaped responses.
    claims = answer.get("claims")
    if isinstance(claims, list):
        return "\n\n".join(
            str(claim.get("text")).strip()
            for claim in claims
            if isinstance(claim, dict) and str(claim.get("text") or "").strip()
        )
    intro = answer.get("intro")
    return str(intro).strip() if isinstance(intro, str) else ""


def _uncited_claim_texts(answer: Any) -> List[str]:
    if not isinstance(answer, dict) or not isinstance(answer.get("claims"), list):
        return []
    uncited = []
    for claim in answer["claims"]:
        if not isinstance(claim, dict):
            continue
        text = str(claim.get("text") or "").strip()
        citations = claim.get("citations")
        if text and isinstance(citations, list) and not citations:
            uncited.append(text)
    return uncited


def resolve_inline_citations(
    answer_text: str,
    citation_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    source_refs: List[Dict[str, Any]] = []
    number_by_key: Dict[str, int] = {}
    number_by_source: Dict[str, int] = {}
    unresolved_keys = []

    def replace(match):
        key = match.group("key")
        entry = citation_map.get(key)
        if not entry:
            unresolved_keys.append(key)
            return ""
        source = _source_from_citation_entry(entry)
        source_ref = source_references([source])[0]
        source_identity = str(source_ref.get("source_id") or source_ref.get("url") or key)
        number = number_by_source.get(source_identity)
        if not number:
            number = len(source_refs) + 1
            number_by_source[source_identity] = number
            source_refs.append({**source_ref, "number": number})
        number_by_key[key] = number
        return f"[{number}]"

    rendered = CITATION_MARKER_PATTERN.sub(replace, answer_text)
    rendered = re.sub(r"[ \t]{2,}", " ", rendered).strip()
    rendered = re.sub(r"\s+([,.;:!?])", r"\1", rendered)
    warnings = []
    if unresolved_keys:
        warnings.append("Some citation markers could not be resolved; answer text was preserved.")
    return {
        "answer": rendered,
        "sources": source_refs,
        "warnings": warnings,
        "unresolved_citation_keys": sorted(set(unresolved_keys)),
        "resolved_citation_keys": number_by_key,
    }


def _named_entities(values: Optional[List[Any]]) -> List[Dict[str, str]]:
    entities = []
    for value in values or []:
        if isinstance(value, dict):
            name = str(value.get("name", "")).strip()
        else:
            name = str(value or "").strip()
        if name:
            entities.append({"name": name})
    return entities


def build_metadata(
    question: str,
    *,
    date_range: Optional[Dict[str, Optional[str]]] = None,
    authors: Optional[List[Any]] = None,
    speakers: Optional[List[Any]] = None,
    guests: Optional[List[Any]] = None,
    content_types: Optional[List[str]] = None,
    program: Optional[str] = None,
    sort: str = "relevance",
) -> Dict[str, Any]:
    date_range = date_range or {}
    return {
        "question": question,
        "date_range": {
            "start_date": date_range.get("start_date"),
            "end_date": date_range.get("end_date"),
        },
        "authors": _named_entities(authors),
        "speakers": _named_entities(speakers),
        "guests": _named_entities(guests),
        "content_types": [
            value for value in (content_types or []) if value in {"article", "transcript", "script"}
        ],
        "program": program,
        "sort": sort if sort in {"relevance", "newest", "oldest"} else "relevance",
    }


def _apply_overrides(
    metadata: Dict[str, Any],
    *,
    content_types: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    metadata = dict(metadata)
    metadata.setdefault("date_range", {"start_date": None, "end_date": None})
    if start_date is not None:
        metadata["date_range"]["start_date"] = start_date
    if end_date is not None:
        metadata["date_range"]["end_date"] = end_date
    if content_types:
        metadata["content_types"] = [
            value for value in content_types if value in {"article", "transcript", "script"}
        ]
    return metadata


def _normalize_history(conversation_context: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    history = []
    for turn in conversation_context or []:
        role = turn.get("role")
        content = turn.get("content")
        if role in {"user", "assistant"} and content:
            history.append({"role": role, "content": str(content)})
    return history


def _validate_date(value: Optional[str], field: str) -> Optional[str]:
    if value is None:
        return None
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be YYYY-MM-DD") from exc
    return parsed.date().isoformat()


def validate_search_arguments(query: str, start_date=None, end_date=None,
                              content_types=None, limit=8, sort="relevance") -> Dict[str, Any]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query is required")
    start = _validate_date(start_date, "start_date")
    end = _validate_date(end_date, "end_date")
    if start and end and start > end:
        raise ValueError("start_date must not be after end_date")
    types = list(content_types or ["article", "transcript", "script"])
    if any(value not in {"article", "transcript", "script"} for value in types):
        raise ValueError("content_types must contain only article, transcript, or script")
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 50:
        raise ValueError("limit must be between 1 and 50")
    if sort not in {"relevance", "newest", "oldest"}:
        raise ValueError("sort must be relevance, newest, or oldest")
    return {"query": query.strip(), "start_date": start, "end_date": end,
            "content_types": types, "limit": limit, "sort": sort}


def validate_citations(answer: Dict[str, Any], sources: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    claims = answer.get("claims") if isinstance(answer, dict) else None
    if not isinstance(claims, list):
        raise ValueError("structured answer must contain claims")
    valid_claims = []
    rejected_citations = []
    for claim in claims:
        if not isinstance(claim, dict) or not isinstance(claim.get("text"), str):
            continue
        citations = []
        for citation in claim.get("citations") or []:
            if not isinstance(citation, dict):
                continue
            source_id = str(citation.get("source_id"))
            source = sources.get(source_id)
            passage_ids = [str(value) for value in citation.get("passage_ids") or []]
            known = set()
            for passage in (source or {}).get("passages", []):
                if not isinstance(passage, dict):
                    continue
                passage_id = passage.get("passage_id")
                if passage_id is not None:
                    known.add(str(passage_id))
                for merged_id in passage.get("passage_ids") or []:
                    known.add(str(merged_id))
            if source and passage_ids and set(passage_ids) <= known:
                citations.append({"source_id": source_id, "passage_ids": passage_ids})
            else:
                if not source:
                    reason = "unknown_source_id"
                elif not passage_ids:
                    reason = "missing_passage_ids"
                else:
                    reason = "unknown_passage_ids"
                rejected_citations.append({
                    "claim_text": claim["text"],
                    "source_id": source_id,
                    "passage_ids": passage_ids,
                    "known_passage_ids": sorted(known),
                    "reason": reason,
                })
        if citations:
            valid_claims.append({"text": claim["text"], "citations": citations})
    return {
        "intro": answer.get("intro") if isinstance(answer, dict) else None,
        "claims": valid_claims,
        "_rejected_citations": rejected_citations,
    }


def _split_source_marker_safe_prefix(text: str) -> tuple[str, str]:
    match = re.search(r"\[S?R?C?\d*$", text)
    if not match:
        return text, ""
    return text[: match.start()], text[match.start():]


def _replace_source_markers_with_source_numbers(
    text: str,
    source_urls: Dict[int, Optional[str]],
) -> str:
    def replace(match):
        source_number = int(match.group(1))
        label = f"[{source_number}]"
        source_url = source_urls.get(source_number)
        if not source_url:
            return label
        return f"<{slack_link_url(source_url)}|{label}>"

    return markdown_to_slack_mrkdwn(re.sub(r"\[SRC(\d+)\]", replace, text))


def _mcp_search_details(arguments: Any) -> str:
    """Render the user-relevant constraints of an MCP search call."""
    return search_details(arguments)


class WavelengthService:
    def __init__(self, engine: Dewey):
        self.engine = engine
        self._search_cache: Dict[tuple[str, str], Dict[str, Any]] = {}
        self._source_cache: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def responses_mcp_tool_config(
        slack_dm_context: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        base = os.environ.get("WAVELENGTH_PUBLIC_BASE_URL", "").rstrip("/")
        token = os.environ.get("WAVELENGTH_MCP_AUTH_TOKEN", "")
        if not base or not token:
            raise ValueError("WAVELENGTH_PUBLIC_BASE_URL and WAVELENGTH_MCP_AUTH_TOKEN are required")
        headers = {"Authorization": f"Bearer {token}"}
        search_config = {
            "type": "mcp", "server_label": "wavelength_archive",
            "server_description": "Read-only search and retrieval for the Wavelength newsroom archive.",
            "server_url": f"{base}/mcp/search", "headers": headers,
            "allowed_tools": [
                "search_archive",
                "keyword_search_archive",
                "sample_archive",
                "get_full_article",
                "get_full_transcript",
                "get_full_script",
            ],
            "require_approval": "never",
        }
        configs = [search_config]
        if slack_dm_context and all(
            slack_dm_context.get(key) for key in ("channel", "event_ts", "user_id")
        ):
            configs.append({
                "type": "mcp",
                "server_label": "wavelength_slack_context",
                "server_description": "Request-scoped Slack direct-message context for Wavelength.",
                "server_url": f"{base}/mcp/slack",
                "headers": {
                    **headers,
                    "X-Wavelength-Slack-Channel": slack_dm_context["channel"],
                    "X-Wavelength-Slack-Event-Ts": slack_dm_context["event_ts"],
                    "X-Wavelength-Slack-User-Id": slack_dm_context["user_id"],
                },
                "allowed_tools": ["get_prior_slack_dms"],
                "require_approval": "never",
            })
        return configs

    @staticmethod
    def structured_answer_schema() -> Dict[str, Any]:
        return {"type": "json_schema", "name": "wavelength_answer", "strict": True,
                "schema": {"type": "object", "additionalProperties": False,
                           "properties": {"answer": {"type": "string"}},
                           "required": ["answer"]}}

    @staticmethod
    def responses_mcp_instructions(
        current_date: str,
    ) -> str:
        """Instructions for the model that plans and answers through remote MCP."""
        return archive_search_instructions(current_date)

    def ask_archive_via_responses(self, question: str, conversation_context=None, response=None,
                                  slack_dm_context: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        if response is None:
            from openai import OpenAI
            client = OpenAI(api_key=self.engine.openai_config.api_key,
                            base_url=f"{os.environ['AZURE_OPENAI_ENDPOINT'].rstrip('/')}/openai/v1/")
            mcp_config = self.responses_mcp_tool_config(slack_dm_context)
            logger.info(
                "Starting Responses remote-MCP orchestration: tools=%s",
                {
                    config.get("server_label"): config.get("allowed_tools")
                    for config in mcp_config
                    if isinstance(config, dict)
                },
            )
            response = client.responses.create(model=self.engine.openai_config.chat_deployment,
                input=[*_normalize_history(conversation_context), {"role": "user", "content": question}],
                tools=mcp_config,
                text={"format": self.structured_answer_schema()},
                instructions=self.responses_mcp_instructions(
                    datetime.now().strftime("%A, %B %d, %Y"),
                ))
        logger.info("Responses remote-MCP orchestration completed: output_items=%d", len(getattr(response, "output", []) or []))
        raw = None
        for item in getattr(response, "output", []):
            if getattr(item, "type", "") == "message":
                content = getattr(item, "content", []) or []
                for part in content:
                    raw = getattr(part, "text", None) or (part.get("text") if isinstance(part, dict) else None)
                    if raw:
                        break
            raw = raw or getattr(item, "text", None) or getattr(item, "output_text", None)
            if raw:
                break
        if not raw:
            raw = getattr(response, "output_text", "{\"answer\":\"\"}")
        answer = json.loads(raw) if isinstance(raw, str) else raw
        _log_sources("Responses structured answer output", answer)
        citation_map: Dict[str, Dict[str, Any]] = {}
        mcp_calls = []
        for item in getattr(response, "output", []):
            if getattr(item, "type", "") == "mcp_call":
                tool_name = (
                    getattr(item, "name", None)
                    or getattr(item, "tool_name", None)
                    or getattr(item, "method", None)
                    or "archive tool"
                )
                mcp_calls.append({
                    "name": str(tool_name),
                    "arguments": getattr(item, "arguments", None),
                })
                output = getattr(item, "output", None)
                if isinstance(output, str):
                    try:
                        output = json.loads(output)
                    except json.JSONDecodeError:
                        output = None
                if isinstance(output, dict):
                    for entry in output.get("citation_map", []) or []:
                        if not isinstance(entry, dict) or not entry.get("citation_key"):
                            continue
                        citation_map[str(entry["citation_key"])] = entry
                    # Backward-compatible fallback for older MCP results that
                    # have not yet emitted citation_map.
                    result_sources = [
                        row for row in output.get("results", []) or []
                        if isinstance(row, dict)
                    ]
                    for entry in _citation_map_for_sources(result_sources):
                        citation_map.setdefault(str(entry["citation_key"]), entry)
        answer_text = _answer_text_from_structured_answer(answer)
        uncited_claims = _uncited_claim_texts(answer)
        resolved = resolve_inline_citations(answer_text, citation_map)
        _log_sources(
            "Responses citation map",
            {
                "citation_count": len(citation_map),
                "citations": _citation_map_summary(citation_map),
                "mcp_calls": mcp_calls,
            },
        )
        if resolved["unresolved_citation_keys"]:
            logger.warning(
                "Could not resolve %d inline citation marker(s). "
                "Set WAVELENGTH_LOG_SOURCES=true or start with --log-sources for details.",
                len(resolved["unresolved_citation_keys"]),
            )
            _log_sources(
                "Unresolved inline citation markers",
                {
                    "unresolved_citation_keys": resolved["unresolved_citation_keys"],
                    "known_citation_keys": sorted(citation_map.keys()),
                    "answer": answer_text,
                },
                level=logging.WARNING,
            )
        if uncited_claims:
            logger.warning(
                "Structured answer included %d uncited claim(s); preserving text.",
                len(uncited_claims),
            )
            _log_sources(
                "Uncited structured claims",
                uncited_claims,
                level=logging.WARNING,
            )
        rendered = resolved["answer"]
        if not rendered:
            rendered = "Here are the top archive sources I found:"
        warnings = list(resolved["warnings"])
        if uncited_claims:
            warnings.append("Some structured claims had no citation markers; answer text was preserved.")
        if warnings:
            rendered = rendered + "\n\nWarning: " + " ".join(warnings)
        slack_rendered = markdown_to_slack_mrkdwn(rendered)
        return block_kit_tool_result(
            slack_rendered,
            answer_blocks(rendered, resolved["sources"]),
            {
                "answer": slack_rendered,
                "answer_markdown": rendered,
                "sources": resolved["sources"],
                "mcp_calls": mcp_calls,
                "warnings": warnings,
                "unresolved_citation_keys": resolved["unresolved_citation_keys"],
                "uncited_claim_count": len(uncited_claims),
            },
        )

    @classmethod
    def from_environment(cls) -> "WavelengthService":
        load_dotenv()
        openai_config = AzureOpenAIConfig(
            endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            embedding_deployment=os.environ["EMBEDDING_DEPLOYMENT_NAME"],
            embedding_model=os.environ["EMBEDDING_MODEL_NAME"],
            chat_deployment=os.environ["CHATGPT_DEPLOYMENT_NAME"],
            chat_model=os.environ["CHATGPT_MODEL_NAME"],
            embedding_dimensions=resolve_embedding_dimensions(
                os.environ["EMBEDDING_MODEL_NAME"],
                os.environ.get("EMBEDDING_DIMENSIONS"),
            ),
        )
        search_config = AzureSearchConfig(
            service_endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
            index_name=os.environ["AZURE_SEARCH_INDEX_NAME"],
            key=os.environ["AZURE_SEARCH_API_KEY"],
        )
        return cls(Dewey(openai_config, search_config))

    def ask_archive(
        self,
        question: str,
        conversation_context: Optional[List[Dict[str, str]]] = None,
        content_types: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        if os.environ.get("WAVELENGTH_RESPONSES_MCP_ENABLED") == "true":
            return self.ask_archive_via_responses(question, conversation_context)
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        messages = _normalize_history(conversation_context)
        messages.append({"role": "user", "content": question})
        _log_verbose("ask_archive metadata model input", messages)

        metadata = self.engine.generate_metadata(
            messages,
            current_date,
            assistant_name="Wavelength",
        )
        metadata = _apply_overrides(
            metadata,
            content_types=content_types,
            start_date=start_date,
            end_date=end_date,
        )
        _log_verbose("ask_archive search metadata", metadata)
        results = self.engine.retrieve_documents(metadata)
        _log_verbose("ask_archive raw sources returned", results)
        formatted_sources = self.engine.format_sources(results)
        _log_verbose("ask_archive formatted sources for answer model", formatted_sources)
        source_urls = self.engine.build_source_url_map(results)

        answer_messages = list(messages)
        answer_messages.append(
            {
                "role": "user",
                "content": f"{question}\n\n## Sources\n{chr(10).join(formatted_sources)}",
            }
        )
        _log_verbose("ask_archive answer model input", answer_messages)

        response = self.engine.oai_client.responses.create(
            model=self.engine.openai_config.chat_deployment,
            instructions=load_answer_prompt(current_date, assistant_name="Wavelength"),
            input=answer_messages,
            stream=True,
        )

        answer = ""
        for chunk in response:
            if chunk.type == "response.output_text.delta" and chunk.delta:
                answer += chunk.delta
        _log_verbose("ask_archive raw answer output", answer)

        all_sources = source_references(results)
        sources = cited_source_references(answer, all_sources)
        markdown_answer = replace_source_markers(answer, source_urls, link_urls=False)
        slack_answer = markdown_to_slack_mrkdwn(markdown_answer)
        fallback = slack_answer or "Wavelength could not generate an answer from the retrieved sources."
        structured = {
            "answer": slack_answer,
            "answer_markdown": markdown_answer,
            "sources": sources,
            "needs_clarification": "what time period" in slack_answer.lower(),
        }
        payload = block_kit_tool_result(
            fallback,
            answer_blocks(markdown_answer or fallback, sources),
            structured,
        )
        _log_verbose("ask_archive final Slack markdown", fallback)
        return payload

    def stream_archive(
        self,
        question: str,
        conversation_context: Optional[List[Dict[str, str]]] = None,
        content_types: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        slack_dm_context: Optional[Dict[str, str]] = None,
    ) -> Iterator[Dict[str, Any]]:
        if os.environ.get("WAVELENGTH_RESPONSES_MCP_ENABLED") == "true":
            # Remote MCP orchestration is intentionally non-streaming: the
            # complete structured response must be citation-validated before
            # Slack renders it or attaches Work Object metadata.
            yield {"type": "plan", "title": "Searching the archive", "steps": DEFAULT_SEARCH_STEPS}
            yield {"type": "task", "id": "planner", "title": "Planning Search", "status": "in_progress"}
            from openai import OpenAI
            client = OpenAI(api_key=self.engine.openai_config.api_key,
                            base_url=f"{os.environ['AZURE_OPENAI_ENDPOINT'].rstrip('/')}/openai/v1/")
            mcp_config = self.responses_mcp_tool_config(slack_dm_context)
            stream = client.responses.create(
                model=self.engine.openai_config.chat_deployment,
                input=[*_normalize_history(conversation_context), {"role": "user", "content": question}],
                tools=mcp_config,
                text={"format": self.structured_answer_schema()},
                instructions=self.responses_mcp_instructions(
                    datetime.now().strftime("%A, %B %d, %Y"),
                ),
                stream=True,
            )
            yield {"type": "task", "id": "planner", "title": "Planning Search", "status": "complete"}
            search_details = []
            seen_searches = set()
            call_names = {}
            final_response = None
            drafting = False
            for event in stream:
                event_type = getattr(event, "type", "")
                item = getattr(event, "item", None)
                if event_type == "response.output_item.added" and getattr(item, "type", "") == "mcp_call":
                    call_names[getattr(item, "id", "")] = getattr(item, "name", "archive tool")
                if event_type == "response.mcp_call_arguments.done":
                    call_id = getattr(event, "item_id", "search")
                    name = call_names.get(call_id, "search_archive")
                    if name not in {"search_archive", "keyword_search_archive", "sample_archive"}:
                        continue
                    signature = search_signature(name, getattr(event, "arguments", None))
                    if signature in seen_searches:
                        continue
                    if seen_searches:
                        yield {"type": "task", "id": "expand", "title": "Expanding search", "status": "in_progress"}
                    seen_searches.add(signature)
                    detail = _mcp_search_details(getattr(event, "arguments", None))
                    search_details.append(detail)
                    yield {"type": "task", "id": "search", "title": "Searching", "status": "in_progress", "details": "\n\n".join(search_details)}
                elif event_type == "response.mcp_call.completed":
                    yield {"type": "task", "id": "search", "title": "Searching", "status": "in_progress", "details": "\n\n".join(search_details) or "Retrieved archive sources"}
                elif event_type == "response.output_text.delta" and not drafting:
                    drafting = True
                    if len(seen_searches) > 1:
                        yield {"type": "task", "id": "expand", "title": "Expanding search", "status": "complete"}
                    yield {"type": "task", "id": "search", "title": "Searching", "status": "complete", "details": ""}
                    yield {"type": "task", "id": "draft", "title": "Drafting an answer", "status": "in_progress"}
                elif event_type == "response.completed":
                    final_response = getattr(event, "response", None)
            if not final_response and hasattr(stream, "get_final_response"):
                final_response = stream.get_final_response()
            if drafting:
                yield {"type": "task", "id": "draft", "title": "Drafting an answer", "status": "complete"}
            yield {"type": "final", "payload": self.ask_archive_via_responses(question, conversation_context, response=final_response, slack_dm_context=slack_dm_context)}
            return
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        messages = _normalize_history(conversation_context)
        messages.append({"role": "user", "content": question})
        _log_verbose("stream_archive metadata model input", messages)

        yield {
            "type": "task",
            "id": "plan",
            "title": "Planning search",
            "status": "in_progress",
            "details": "Planning archive filters.",
        }
        metadata = self.engine.generate_metadata(
            messages,
            current_date,
            assistant_name="Wavelength",
        )
        metadata = _apply_overrides(
            metadata,
            content_types=content_types,
            start_date=start_date,
            end_date=end_date,
        )
        _log_verbose("stream_archive search metadata", metadata)
        yield {
            "type": "task",
            "id": "plan",
            "title": "Planning search",
            "status": "complete",
        }

        yield {
            "type": "task",
            "id": "search",
            "title": "Searching",
            "status": "in_progress",
            "details": _mcp_search_details(metadata),
        }
        results = self.engine.retrieve_documents(metadata)
        _log_verbose("stream_archive raw sources returned", results)
        formatted_sources = self.engine.format_sources(results)
        _log_verbose("stream_archive formatted sources for answer model", formatted_sources)
        source_urls = self.engine.build_source_url_map(results)
        all_sources = source_references(results)
        yield {
            "type": "task",
            "id": "search",
            "title": "Searching",
            "status": "complete",
            "details": f"Found {len(all_sources)} source(s).",
        }

        answer_messages = list(messages)
        answer_messages.append(
            {
                "role": "user",
                "content": f"{question}\n\n## Sources\n{chr(10).join(formatted_sources)}",
            }
        )
        _log_verbose("stream_archive answer model input", answer_messages)
        yield {
            "type": "task",
            "id": "draft",
            "title": "Drafting an answer",
            "status": "in_progress",
        }

        response = self.engine.oai_client.responses.create(
            model=self.engine.openai_config.chat_deployment,
            instructions=load_answer_prompt(current_date, assistant_name="Wavelength"),
            input=answer_messages,
            stream=True,
        )

        answer = ""
        raw_buffer = ""
        for chunk in response:
            if chunk.type == "response.output_text.delta" and chunk.delta:
                answer += chunk.delta
                raw_buffer += chunk.delta
                safe_text, raw_buffer = _split_source_marker_safe_prefix(raw_buffer)
                delta = _replace_source_markers_with_source_numbers(safe_text, source_urls)
                if delta:
                    yield {"type": "text_delta", "text": delta}
        if raw_buffer:
            yield {
                "type": "text_delta",
                "text": _replace_source_markers_with_source_numbers(raw_buffer, source_urls),
            }
        _log_verbose("stream_archive raw answer output", answer)
        yield {
            "type": "task",
            "id": "draft",
            "title": "Drafting an answer",
            "status": "complete",
        }

        sources = cited_source_references(answer, all_sources)
        markdown_answer = replace_source_markers(answer, source_urls, link_urls=False)
        slack_answer = markdown_to_slack_mrkdwn(markdown_answer)
        fallback = slack_answer or "Wavelength could not generate an answer from the retrieved sources."
        structured = {
            "answer": slack_answer,
            "answer_markdown": markdown_answer,
            "sources": sources,
            "needs_clarification": "what time period" in slack_answer.lower(),
        }
        payload = block_kit_tool_result(
            fallback,
            answer_blocks(markdown_answer or fallback, sources),
            structured,
        )
        _log_verbose("stream_archive final Slack markdown", fallback)
        yield {
            "type": "final",
            "payload": payload,
        }

    def _logical_sources(self, results: List[Dict[str, Any]], limit: int = 8,
                         sort: str = "relevance") -> List[Dict[str, Any]]:
        grouped: Dict[str, Dict[str, Any]] = {}
        for row in results:
            content_type = row.get("content_type", "article")
            source_id_candidates = (
                [
                    row.get("parent_id"),
                    row.get("source_id"),
                    row.get("occurrence_id"),
                    row.get("sourcepage"),
                    row.get("url"),
                ]
                if content_type == "transcript"
                else [
                    row.get("source_id"),
                    row.get("parent_id"),
                    row.get("chunk_id"),
                    row.get("occurrence_id"),
                    row.get("sourcepage"),
                    row.get("url"),
                ]
            )
            source_id = str(
                next((candidate for candidate in source_id_candidates if candidate), "")
            )
            if not source_id:
                continue
            # Work Objects/citations need a stable, opaque identifier.
            #
            # Some CST article rows use URL-shaped IDs or very long composite
            # IDs that embed URLs (e.g., base64-encoded blob URLs). Slack tends
            # to drop Work Object entities for those. Normalize them to an
            # opaque ID so:
            # - the model cites a compact `source_id`
            # - Slack can retain Work Object entity metadata
            is_urlish = source_id.startswith(("http://", "https://"))
            is_too_long = len(source_id) > 80
            embeds_encoded_url = "_aHR0c" in source_id  # base64("http")
            if is_urlish or is_too_long or embeds_encoded_url:
                digest = sha256(source_id.encode("utf-8", errors="ignore")).hexdigest()[:20]
                source_id = f"src_{digest}"
            source = grouped.setdefault(source_id, {
                "source_id": source_id, "content_type": content_type,
                "title": row.get("headline") or row.get("title") or row.get("transcript_name") or "Untitled source",
                "canonical_url": row.get("citation_url") or row.get("url") or row.get("transcript_url"),
                "url": row.get("citation_url") or row.get("url") or row.get("transcript_url"),
                "published_at": row.get("publish_date"), "authors": row.get("authors") or [],
                "program": row.get("program"),
                "guests": row.get("guests") or [],
                "occurrence_id": row.get("occurrence_id"),
                "parent_id": row.get("parent_id"),
                "passages": [],
                "_score": row.get("@search.score", 0) or 0,
            })
            text = row.get("chunk_text") if content_type in {"transcript", "script"} else row.get("content")
            passage_id = str(row.get("passage_id") or row.get("occurrence_id") or row.get("chunk_id") or len(source["passages"]) + 1)
            passage = {"passage_id": passage_id, "text": str(text or ""), "speakers": row.get("speakers") or [],
                       "raw_vtt_excerpt": row.get("raw_vtt_excerpt"),
                       "start_seconds": row.get("start_seconds"), "end_seconds": row.get("end_seconds"),
                       "timestamp_url": row.get("timestamp_url") or row.get("transcript_url") or row.get("recording_url"),
                       "score": row.get("@search.score")}
            if not any(p["passage_id"] == passage_id for p in source["passages"]):
                source["passages"].append(passage)
            source["_score"] = max(source["_score"], row.get("@search.score", 0) or 0)
        if sort in {"newest", "oldest"}:
            sources = sorted(
                grouped.values(),
                key=lambda value: str(value.get("published_at") or ""),
                reverse=sort == "newest",
            )[:limit]
        else:
            sources = sorted(grouped.values(), key=lambda value: value["_score"], reverse=True)[:limit]
        for source in sources:
            ordered = sorted(source["passages"], key=lambda p: (p.get("start_seconds") is None, p.get("start_seconds") or 0))
            merged = []
            for passage in ordered:
                previous = merged[-1] if merged else None
                overlaps = (source["content_type"] == "transcript" and previous and
                            previous.get("end_seconds") is not None and passage.get("start_seconds") is not None and
                            passage["start_seconds"] <= previous["end_seconds"] + 1)
                if overlaps:
                    if passage["text"] not in previous["text"]:
                        previous["text"] = f"{previous['text']} {passage['text']}".strip()
                    if passage.get("raw_vtt_excerpt") and passage["raw_vtt_excerpt"] not in str(previous.get("raw_vtt_excerpt") or ""):
                        previous["raw_vtt_excerpt"] = "\n\n".join(
                            value for value in [previous.get("raw_vtt_excerpt"), passage.get("raw_vtt_excerpt")] if value
                        )
                    previous["end_seconds"] = max(previous.get("end_seconds") or 0, passage.get("end_seconds") or 0)
                    previous.setdefault("passage_ids", [previous["passage_id"]]).append(passage["passage_id"])
                else:
                    passage.setdefault("passage_ids", [passage["passage_id"]])
                    merged.append(passage)
            source["passages"] = merged
            source.pop("_score", None)
        return sources

    def _remember_sources(self, sources: List[Dict[str, Any]]) -> None:
        for source in sources:
            if not isinstance(source, dict):
                continue
            source_id = str(source.get("source_id") or source.get("id") or "").strip()
            if source_id:
                self._source_cache[source_id] = dict(source)

    def _agent_source(self, source: Dict[str, Any]) -> Dict[str, Any]:
        source_id = source.get("source_id") or source.get("id")
        passages = []
        for passage in source.get("passages") or []:
            if not isinstance(passage, dict):
                continue
            passage_id = passage.get("passage_id")
            passages.append({
                "citation_key": _citation_key(source_id, passage_id),
                "passage_id": passage_id,
                "passage_ids": passage.get("passage_ids") or [passage_id],
                "text": passage.get("text"),
                "raw_vtt_excerpt": passage.get("raw_vtt_excerpt"),
                "speakers": passage.get("speakers") or [],
                "start_seconds": passage.get("start_seconds"),
                "end_seconds": passage.get("end_seconds"),
                "timestamp_url": passage.get("timestamp_url"),
            })
        return {
            "source_id": source_id,
            "content_type": source.get("content_type"),
            "title": source.get("title") or source.get("headline") or source.get("transcript_name"),
            "url": source.get("url") or source.get("canonical_url") or source.get("transcript_url"),
            "listen_url": source.get("listen_url") or source.get("transcript_url"),
            "published_at": source.get("published_at") or source.get("publish_date"),
            "authors": source.get("authors") or [],
            "speakers": source.get("speakers") or [],
            "guests": source.get("guests") or [],
            "program": source.get("program"),
            "parent_id": source.get("parent_id"),
            "occurrence_id": source.get("occurrence_id"),
            "passages": passages,
        }

    def _display_source(self, source: Dict[str, Any]) -> Dict[str, Any]:
        source_id = str(source.get("source_id") or source.get("id") or "").strip()
        cached = self._source_cache.get(source_id)
        if cached:
            display = dict(cached)
        else:
            display = {}
            if source_id:
                try:
                    raw_source = self.engine.get_source(source_id)
                except Exception:
                    raw_source = None
                if raw_source:
                    logical = self._logical_sources([raw_source], 1)
                    display = logical[0] if logical else dict(raw_source)
                    self._remember_sources([display])
        if not display:
            return source
        if source.get("passages"):
            display["passages"] = source["passages"]
        return display

    def search_archive_data(self, *, query: str, start_date=None, end_date=None,
                            content_types=None, authors=None, speakers=None, guests=None,
                            program=None, limit: int = 8, sort: str = "relevance",
                            search_top: Optional[int] = None,
                            search_mode: str = "semantic",
                            sample: bool = False,
                            random_seed: Optional[int] = None) -> Dict[str, Any]:
        args = validate_search_arguments(query, start_date, end_date, content_types, limit, sort)
        if search_mode not in {"semantic", "keyword"}:
            raise ValueError("search_mode must be semantic or keyword")
        if not isinstance(search_top, int) or isinstance(search_top, bool):
            search_top = max(args["limit"], 10)
        search_top = max(args["limit"], min(search_top, 100))
        metadata = build_metadata(query, date_range={"start_date": args["start_date"], "end_date": args["end_date"]},
                                   authors=authors, speakers=speakers, guests=guests,
                                   content_types=args["content_types"], program=program,
                                   sort=args["sort"])
        metadata["search_top"] = search_top
        metadata["search_mode"] = search_mode
        cache_key = search_signature("search_archive_data", {
            "query": args["query"],
            "start_date": args["start_date"],
            "end_date": args["end_date"],
            "content_types": args["content_types"],
            "authors": authors,
            "speakers": speakers,
            "guests": guests,
            "program": program,
            "limit": args["limit"],
            "sort": args["sort"],
            "search_top": search_top,
            "search_mode": search_mode,
            "sample": bool(sample),
            "random_seed": random_seed,
        })
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]
        candidates = self._logical_sources(
            self.engine.retrieve_documents(metadata), search_top, args["sort"]
        )
        self._remember_sources(candidates)
        if sample and candidates:
            rng = random.Random(random_seed)
            results = rng.sample(candidates, min(args["limit"], len(candidates)))
        else:
            results = candidates[:args["limit"]]
        agent_sources = [self._agent_source(source) for source in results]
        payload = {
            "query": args["query"],
            "search_mode": search_mode,
            "search_top": search_top,
            "sampled": bool(sample),
            "results": agent_sources,
            "citation_map": _citation_map_for_sources(agent_sources),
        }
        self._search_cache[cache_key] = payload
        return payload

    def _source_file(self, source_id: str, suffix: str) -> Optional[Path]:
        data_dir = Path(os.environ.get("WAVELENGTH_DATA_DIR", "data"))
        candidate = Path(source_id)
        if candidate.is_absolute() or ".." in candidate.parts:
            return None
        path = data_dir / candidate
        if path.suffix != suffix:
            path = path.with_suffix(suffix)
        return path if path.is_file() else None

    def get_full_article_data(self, source_id: str, cursor: Optional[str] = None, max_chars: int = 12000) -> Dict[str, Any]:
        if not isinstance(max_chars, int) or not 1 <= max_chars <= 20000:
            raise ValueError("max_chars must be between 1 and 20000")
        source = self.engine.get_source(source_id)
        if not source or source.get("content_type") != "article":
            raise LookupError("source_content_unavailable")
        content = source.get("content")
        sourcepage = source.get("sourcepage") or source.get("source_id") or source_id
        path = self._source_file(str(sourcepage), ".json")
        if path:
            try:
                content = json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False)
            except (OSError, json.JSONDecodeError):
                pass
        content = str(content or "")
        offset = int(cursor or 0)
        segment = content[offset:offset + max_chars]
        next_offset = offset + len(segment)
        logical = self._logical_sources([source], 1)
        self._remember_sources(logical)
        source_payload = self._agent_source(logical[0]) if logical else {}
        passage_id = f"content:{offset}:{next_offset}"
        citation_key = _citation_key(source_payload.get("source_id") or source_id, passage_id)
        return {"source": source_payload, "content": segment,
                "citation_key": citation_key,
                "citation_map": [_segment_citation_entry(source_payload, citation_key, passage_id)],
                "cursor": str(next_offset) if next_offset < len(content) else None,
                "has_more": next_offset < len(content)}

    def get_full_transcript_data(self, source_id: str, cursor: Optional[str] = None, max_chars: int = 12000,
                                 start_seconds=None, end_seconds=None) -> Dict[str, Any]:
        if not isinstance(max_chars, int) or not 1 <= max_chars <= 20000:
            raise ValueError("max_chars must be between 1 and 20000")
        source = self.engine.get_source(source_id)
        if not source or source.get("content_type") != "transcript":
            raise LookupError("source_content_unavailable")
        vtt = self._source_file(str(source.get("sourcepage") or source.get("transcript_name") or source_id), ".vtt")
        if not vtt:
            raise LookupError("source_content_unavailable")
        from transcripts import load_vtt
        _, parsed_cues = load_vtt(vtt)
        cues = [{"start_seconds": cue.start_seconds, "end_seconds": cue.end_seconds,
                 "speaker": cue.speaker, "text": cue.text,
                 "timestamp_url": source.get("transcript_url")} for cue in parsed_cues]
        offset = int(cursor or 0)
        selected, size = [], 0
        for cue in cues[offset:]:
            if start_seconds is not None and cue["end_seconds"] < start_seconds: continue
            if end_seconds is not None and cue["start_seconds"] > end_seconds: continue
            if selected and size + len(cue["text"]) > max_chars: break
            selected.append(cue); size += len(cue["text"])
        next_offset = offset + len(selected)
        logical = self._logical_sources([source], 1)
        self._remember_sources(logical)
        source_payload = self._agent_source(logical[0]) if logical else {}
        source_id_for_keys = source_payload.get("source_id") or source_id
        keyed_cues = []
        citation_map = []
        for cue_index, cue in enumerate(selected, offset):
            passage_id = f"cue:{cue_index}:{cue.get('start_seconds')}:{cue.get('end_seconds')}"
            citation_key = _citation_key(source_id_for_keys, passage_id)
            keyed_cues.append({**cue, "citation_key": citation_key})
            citation_map.append({
                **_segment_citation_entry(source_payload, citation_key, passage_id),
                "listen_url": cue.get("timestamp_url") or source_payload.get("listen_url"),
                "speakers": [cue.get("speaker")] if cue.get("speaker") else [],
                "start_seconds": cue.get("start_seconds"),
                "end_seconds": cue.get("end_seconds"),
            })
        return {"source": source_payload, "cues": keyed_cues,
                "citation_map": citation_map,
                "cursor": str(next_offset) if next_offset < len(cues) else None,
                "has_more": next_offset < len(cues)}

    def get_full_script_data(self, source_id: str, cursor: Optional[str] = None, max_chars: int = 12000) -> Dict[str, Any]:
        if not isinstance(max_chars, int) or not 1 <= max_chars <= 20000:
            raise ValueError("max_chars must be between 1 and 20000")
        source = self.engine.get_source(source_id)
        if not source or source.get("content_type") != "script":
            raise LookupError("source_content_unavailable")
        data_dir = Path(os.environ.get("WAVELENGTH_DATA_DIR", "data"))
        path = data_dir / "scripts" / f"{source_id}.json"
        if not path.is_file():
            raise LookupError("source_content_unavailable")
        try:
            content = str(json.loads(path.read_text(encoding="utf-8"))["content"])
        except (OSError, json.JSONDecodeError, KeyError):
            raise LookupError("source_content_unavailable")
        offset = int(cursor or 0)
        segment = content[offset:offset + max_chars]
        next_offset = offset + len(segment)
        logical = self._logical_sources([source], 1)
        self._remember_sources(logical)
        source_payload = self._agent_source(logical[0]) if logical else {}
        passage_id = f"content:{offset}:{next_offset}"
        citation_key = _citation_key(source_payload.get("source_id") or source_id, passage_id)
        return {"source": source_payload, "content": segment,
                "citation_key": citation_key,
                "citation_map": [_segment_citation_entry(source_payload, citation_key, passage_id)],
                "cursor": str(next_offset) if next_offset < len(content) else None,
                "has_more": next_offset < len(content)}

    def search_archive(
        self,
        question: str,
        date_range: Optional[Dict[str, Optional[str]]] = None,
        authors: Optional[List[Any]] = None,
        speakers: Optional[List[Any]] = None,
        guests: Optional[List[Any]] = None,
        content_types: Optional[List[str]] = None,
        program: Optional[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        metadata = build_metadata(
            question,
            date_range=date_range,
            authors=authors,
            speakers=speakers,
            guests=guests,
            content_types=content_types,
            program=program,
        )
        _log_verbose("search_archive search metadata", metadata)
        results = self.engine.retrieve_documents(metadata)[: max(1, min(limit, 10))]
        _log_verbose("search_archive raw sources returned", results)
        sources = source_references(results)
        if sources:
            fallback = f"Wavelength found {len(sources)} archive source(s) for: {question}"
        else:
            fallback = f"Wavelength found no archive sources for: {question}"
        structured = {"question": question, "sources": sources}
        payload = block_kit_tool_result(
            fallback,
            search_blocks(question, sources),
            structured,
        )
        _log_verbose("search_archive final Slack markdown", fallback)
        return payload

    def get_source(self, source_id: str) -> Dict[str, Any]:
        source = self.engine.get_source(source_id)
        if not source:
            fallback = f"Wavelength could not find archive source {source_id}."
            return block_kit_tool_result(
                fallback,
                search_blocks(f"Source {source_id}", []),
                {"source": None},
            )

        sources = source_references([source])
        fallback = f"Wavelength found archive source {source_id}."
        return block_kit_tool_result(
            fallback,
            search_blocks(f"Source {source_id}", sources),
            {"source": sources[0]},
        )
