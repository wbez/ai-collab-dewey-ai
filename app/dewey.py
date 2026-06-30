import json
import re
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Optional

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType, VectorQuery, VectorizedQuery
from dateutil.parser import parse
from openai import AzureOpenAI

from models import AzureOpenAIConfig, AzureSearchConfig
from tips import TipFormatter
from tools import load_answer_prompt, load_search_prompt, load_search_tool
from transcripts import normalize_name


LUCENE_RESERVED_PATTERN = re.compile(r'([+\-&|!(){}\[\]^"~*?:\\/])')
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "how",
    "in",
    "of",
    "on",
    "the",
    "to",
    "what",
    "when",
    "where",
    "who",
    "with",
}


class Dewey:
    def __init__(self, openai_config: AzureOpenAIConfig, search_config: AzureSearchConfig):
        self.oai_client = AzureOpenAI(
            api_key=openai_config.api_key,
            azure_endpoint=openai_config.endpoint,
            api_version="2025-03-01-preview",
        )
        self.search_client = SearchClient(
            search_config.service_endpoint,
            search_config.index_name,
            AzureKeyCredential(search_config.key),
        )
        self.openai_config = openai_config
        self.sessions = {}
        self.tip_formatter = TipFormatter

    @contextmanager
    def step(self, title, show_steps=True):
        if not show_steps:
            yield lambda: None
            return

        step = {"title": title, "status": "pending"}
        if not hasattr(self, "_current_steps"):
            self._current_steps = []

        self._current_steps.append(step)

        class StepYielder:
            def __init__(self, step, steps_list):
                self.step = step
                self.steps_list = steps_list
                self.has_started = False

            def start(self, content=""):
                if not self.has_started:
                    if content:
                        self.step["content"] = content
                    self.has_started = True
                    return "", self.steps_list.copy()
                return None

            def complete(self, content=""):
                self.step["status"] = "done"
                if content:
                    self.step["content"] = content
                return "", self.steps_list.copy()

        yielder = StepYielder(step, self._current_steps)
        try:
            yield yielder
        finally:
            step["status"] = "done"

    def generate_metadata(self, messages, current_date: str):
        response = self.oai_client.responses.create(
            model=self.openai_config.chat_deployment,
            input=messages,
            instructions=load_search_prompt(current_date),
            tools=[load_search_tool()],
            tool_choice={"type": "function", "name": "search_archive"},
        )
        return json.loads(response.output[-1].arguments)

    def build_filter(self, metadata: Dict[str, object], include_exact_names: bool = True) -> Optional[str]:
        filters: List[str] = []
        date_range = metadata.get("date_range", {})

        start_date = date_range.get("start_date") if isinstance(date_range, dict) else None
        end_date = date_range.get("end_date") if isinstance(date_range, dict) else None
        if start_date:
            filters.append(f"publish_date ge {start_date}T00:00:00Z")
        if end_date:
            filters.append(f"publish_date le {end_date}T23:59:59Z")

        content_types = metadata.get("content_types") or []
        content_type_filters = [
            f"content_type eq '{self._escape_odata_string(value)}'"
            for value in content_types
            if value in {"article", "transcript"}
        ]
        if content_type_filters:
            filters.append(f"({' or '.join(content_type_filters)})")

        program = str(metadata.get("program") or "").strip()
        if program:
            filters.append(f"program eq '{self._escape_odata_string(program)}'")

        guest_filters = self._build_name_filter("guests", metadata.get("guests", []))
        if guest_filters:
            filters.append(guest_filters)

        if include_exact_names:
            author_filters = self._build_name_filter("authors", metadata.get("authors", []))
            if author_filters:
                filters.append(author_filters)

            speaker_filters = self._build_name_filter("speakers", metadata.get("speakers", []))
            if speaker_filters:
                filters.append(speaker_filters)

        return " and ".join(filters) if filters else None

    def _build_name_filter(self, field_name: str, values: object) -> Optional[str]:
        names = [self._coerce_entity_name(value) for value in values or []]
        normalized_names = []
        for name in names:
            compact = re.sub(r"\s+", " ", name).strip()
            if compact:
                normalized_names.append(self._escape_odata_string(compact.lower()))

        if not normalized_names:
            return None

        clauses = [f"tolower(a) eq '{name}'" for name in normalized_names]
        return f"{field_name}/any(a: {' or '.join(clauses)})"

    def _search_documents(
        self,
        metadata: Dict[str, object],
        *,
        search_text: str,
        filter_text: Optional[str],
        vector_queries: Optional[List[VectorQuery]],
        query_type=None,
        semantic_query: Optional[str] = None,
        search_fields: Optional[List[str]] = None,
    ):
        search_kwargs = {
            "search_text": search_text,
            "filter": filter_text,
            "top": 10,
            "select": [
                "url",
                "headline",
                "publish_date",
                "content",
                "authors",
                "content_type",
                "transcript_url",
                "recording_urls",
                "timestamp_label",
                "start_seconds",
                "end_seconds",
                "speakers",
                "guests",
                "program",
                "chunk_id",
            ],
        }
        if vector_queries:
            search_kwargs["vector_queries"] = vector_queries
        if query_type is not None:
            search_kwargs["query_type"] = query_type
        if semantic_query:
            search_kwargs["semantic_configuration_name"] = "default"
            search_kwargs["semantic_query"] = semantic_query
        if search_fields:
            search_kwargs["search_fields"] = search_fields
        return list(self.search_client.search(**search_kwargs))

    def _build_vector_query(self, metadata: Dict[str, object]) -> List[VectorQuery]:
        embedding = self.oai_client.embeddings.create(
            model=self.openai_config.embedding_deployment,
            input=metadata["question"],
            dimensions=self.openai_config.embedding_dimensions,
        )
        return [
            VectorizedQuery(
                vector=embedding.data[0].embedding,
                k_nearest_neighbors=50,
                fields="content_vector",
            )
        ]

    def _build_fuzzy_query(self, metadata: Dict[str, object]) -> Optional[str]:
        clauses: List[str] = []
        keyword_clause = self._build_keyword_clause(metadata.get("question", ""))
        if keyword_clause:
            clauses.append(keyword_clause)

        author_clause = self._build_fuzzy_name_clause(
            "author_search_text",
            metadata.get("authors", []),
        )
        if author_clause:
            clauses.append(author_clause)

        speaker_clause = self._build_fuzzy_name_clause(
            "speaker_search_text",
            metadata.get("speakers", []),
        )
        if speaker_clause:
            clauses.append(speaker_clause)

        return " AND ".join(f"({clause})" for clause in clauses if clause) or None

    def _build_keyword_clause(self, question: object) -> Optional[str]:
        words = re.findall(r"[a-z0-9']+", str(question).lower())
        tokens = [word for word in words if len(word) > 2 and word not in STOP_WORDS]
        if not tokens:
            return None
        return " AND ".join(f"search_text:{self._escape_lucene_term(token)}" for token in tokens)

    def _build_fuzzy_name_clause(self, field_name: str, entities: object) -> Optional[str]:
        entity_clauses: List[str] = []
        for entity in entities or []:
            name = self._coerce_entity_name(entity)
            normalized_tokens = [token for token in normalize_name(name).split() if token]
            if not normalized_tokens:
                continue
            fuzzy_terms = [
                f"{field_name}:{self._escape_lucene_term(token)}~"
                for token in normalized_tokens
            ]
            entity_clauses.append(" AND ".join(fuzzy_terms))
        if not entity_clauses:
            return None
        return " OR ".join(f"({clause})" for clause in entity_clauses)

    def _coerce_entity_name(self, value: object) -> str:
        if isinstance(value, dict):
            return str(value.get("name", "")).strip()
        return str(value or "").strip()

    def _escape_odata_string(self, value: str) -> str:
        return value.replace("'", "''")

    def _escape_lucene_term(self, value: str) -> str:
        return LUCENE_RESERVED_PATTERN.sub(r"\\\1", value)

    def _format_sources(self, results):
        sources = []
        for page in results:
            publish_date = page.get("publish_date")
            source_payload = {
                "content_type": page.get("content_type", "article"),
                "url": page.get("url"),
                "transcript_url": page.get("transcript_url"),
                "recording_urls": page.get("recording_urls") or [],
                "publish_date": (
                    parse(publish_date).date().isoformat() if publish_date else None
                ),
                "authors": page.get("authors") or [],
                "speakers": page.get("speakers") or [],
                "guests": page.get("guests") or [],
                "program": page.get("program"),
                "headline": page.get("headline"),
                "chunk_id": page.get("chunk_id"),
                "timestamp_label": page.get("timestamp_label"),
                "start_seconds": page.get("start_seconds"),
                "end_seconds": page.get("end_seconds"),
                "content": (page.get("content") or "").replace("\n", " ").replace("\r", " "),
            }
            sources.append(json.dumps(source_payload))
        return sources

    def retrieve_articles(self, metadata):
        vectors = self._build_vector_query(metadata)
        exact_filter = self.build_filter(metadata, include_exact_names=True)
        results = self._search_documents(
            metadata,
            search_text=metadata["question"],
            filter_text=exact_filter,
            vector_queries=vectors,
            query_type=QueryType.SEMANTIC,
            semantic_query=metadata["question"],
        )

        has_fuzzy_names = bool(metadata.get("authors") or metadata.get("speakers"))
        if not results and has_fuzzy_names:
            fuzzy_query = self._build_fuzzy_query(metadata)
            if fuzzy_query:
                fuzzy_filter = self.build_filter(metadata, include_exact_names=False)
                results = self._search_documents(
                    metadata,
                    search_text=fuzzy_query,
                    filter_text=fuzzy_filter,
                    vector_queries=None,
                    query_type=QueryType.FULL,
                    search_fields=["search_text", "author_search_text", "speaker_search_text"],
                )

        if not results and metadata.get("authors") and not metadata.get("speakers"):
            fallback_filter = self.build_filter(metadata, include_exact_names=True)
            results = self._search_documents(
                metadata,
                search_text="*",
                filter_text=fallback_filter,
                vector_queries=None,
            )

        return self._format_sources(results)

    def process(self, message: str, history: List, show_steps: bool = True):
        self._current_steps = []
        formatted_date_today = datetime.now().strftime("%A, %B %d, %Y")

        messages = [{"role": turn["role"], "content": turn["content"]} for turn in history]
        messages.append({"role": "user", "content": message})

        with self.step("Generating metadata", show_steps) as step:
            if result := step.start("🔍 I'm planning my approach."):
                yield result
            metadata = self.generate_metadata(messages, formatted_date_today)
            metadata_tip = self.tip_formatter.tip_metadata(metadata)
            if result := step.complete(metadata_tip):
                yield result

        with self.step("Searching articles", show_steps) as step:
            if result := step.start("🔍 Digging through the archives"):
                yield result
            sources = self.retrieve_articles(metadata)
            sources_tip = self.tip_formatter.tip_search(sources)
            if result := step.complete(sources_tip):
                yield result

        stacked_sources = "\n\n".join(sources)
        messages.append({"role": "user", "content": f"{message}\n\n## Sources\n{stacked_sources}"})

        source_urls = {}
        for i, source_json in enumerate(sources, 1):
            source_data = json.loads(source_json)
            source_urls[i] = source_data.get("transcript_url") or source_data.get("url")

        response = self.oai_client.responses.create(
            model=self.openai_config.chat_deployment,
            instructions=load_answer_prompt(formatted_date_today),
            input=messages,
            stream=True,
        )

        partial = ""
        for chunk in response:
            if chunk.type == "response.output_text.delta" and chunk.delta:
                partial += chunk.delta
                processed_partial = re.sub(
                    r"\[SRC(\d+)\]",
                    lambda match: f"[[{match.group(1)}]]({source_urls.get(int(match.group(1)), '#')})",
                    partial,
                )
                yield processed_partial, self._current_steps.copy()
