import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace


APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


def install_sdk_stubs():
    azure_module = types.ModuleType("azure")
    azure_core_module = types.ModuleType("azure.core")
    azure_credentials_module = types.ModuleType("azure.core.credentials")
    azure_search_module = types.ModuleType("azure.search")
    azure_search_documents_module = types.ModuleType("azure.search.documents")
    azure_search_models_module = types.ModuleType("azure.search.documents.models")
    openai_module = types.ModuleType("openai")

    class AzureKeyCredential:
        def __init__(self, key):
            self.key = key

    class SearchClient:
        def __init__(self, *args, **kwargs):
            pass

    class VectorizedQuery:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class QueryType:
        SEMANTIC = "semantic"
        FULL = "full"

    class AzureOpenAI:
        def __init__(self, *args, **kwargs):
            self.responses = None
            self.embeddings = None

    azure_credentials_module.AzureKeyCredential = AzureKeyCredential
    azure_search_documents_module.SearchClient = SearchClient
    azure_search_models_module.VectorizedQuery = VectorizedQuery
    azure_search_models_module.VectorQuery = object
    azure_search_models_module.QueryType = QueryType
    openai_module.AzureOpenAI = AzureOpenAI

    sys.modules.setdefault("azure", azure_module)
    sys.modules.setdefault("azure.core", azure_core_module)
    sys.modules["azure.core.credentials"] = azure_credentials_module
    sys.modules.setdefault("azure.search", azure_search_module)
    sys.modules["azure.search.documents"] = azure_search_documents_module
    sys.modules["azure.search.documents.models"] = azure_search_models_module
    sys.modules["openai"] = openai_module


install_sdk_stubs()

MODULE_PATH = APP_DIR / "wavelength_service.py"
SPEC = importlib.util.spec_from_file_location("wavelength_service", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return [
                SimpleNamespace(
                    type="response.output_text.delta",
                    delta="**Reported fact** [SRC1].",
                )
            ]
        return SimpleNamespace(
            output=[
                SimpleNamespace(
                    arguments=(
                        '{"question":"parks","date_range":{"start_date":null,"end_date":null},'
                        '"authors":[],"speakers":[],"guests":[],"content_types":["article"],'
                        '"program":null}'
                    )
                )
            ]
        )


class FakeEngine:
    def __init__(self):
        self.oai_client = SimpleNamespace(responses=FakeResponses())
        self.openai_config = SimpleNamespace(chat_deployment="chat-test")

    def generate_metadata(self, messages, current_date, assistant_name="Dewey"):
        assert assistant_name == "Wavelength"
        return {
            "question": "parks",
            "date_range": {"start_date": None, "end_date": None},
            "authors": [],
            "speakers": [],
            "guests": [],
            "content_types": ["article"],
            "program": None,
        }

    def retrieve_documents(self, metadata):
        assert metadata["content_types"] == ["article"]
        return [
            {
                "url": "https://example.com/story",
                "publish_date": "2026-01-02",
                "authors": ["Reporter"],
                "headline": "Story",
                "content": "Article body",
                "content_type": "article",
            }
        ]

    def format_sources(self, results):
        return ['{"content_type":"article","publish_date":"2026-01-02","content":"Article body"}']

    def build_source_url_map(self, results):
        return {1: "https://example.com/story"}

    def get_source(self, source_id):
        if source_id != "chunk-1":
            return None
        return {
            "chunk_id": "chunk-1",
            "url": "https://example.com/story",
            "publish_date": "2026-01-02",
            "headline": "Story",
            "content": "Article body",
            "content_type": "article",
        }


def test_ask_archive_returns_block_kit_tool_result():
    service = MODULE.WavelengthService(FakeEngine())

    result = service.ask_archive("What happened?", content_types=["article"])

    assert result["structuredContent"]["answer"] == "*Reported fact* [1]."
    assert result["_meta"]["slack"]["blocks"][0]["type"] == "section"
    assert result["_meta"]["slack"]["blocks"][0]["text"]["text"].startswith("*Reported fact*")


def test_stream_archive_yields_tasks_text_and_final_payload():
    service = MODULE.WavelengthService(FakeEngine())

    events = list(service.stream_archive("What happened?", content_types=["article"]))

    assert events[0] == {
        "type": "task",
        "id": "plan",
        "title": "Planning search",
        "status": "in_progress",
        "details": "Planning archive filters.",
    }
    streamed_text = "".join(
        event["text"] for event in events if event.get("type") == "text_delta"
    )
    assert streamed_text == "*Reported fact* <https://example.com/story|[1]>."
    assert events[-1]["type"] == "final"
    assert events[-1]["payload"]["structuredContent"]["answer"] == "*Reported fact* [1]."


def test_build_metadata_normalizes_filters():
    metadata = MODULE.build_metadata(
        "question",
        date_range={"start_date": "2026-01-01", "end_date": None},
        authors=["Reporter"],
        content_types=["article", "bad"],
    )

    assert metadata["date_range"] == {"start_date": "2026-01-01", "end_date": None}
    assert metadata["authors"] == [{"name": "Reporter"}]
    assert metadata["content_types"] == ["article"]


def test_get_source_returns_block_kit_tool_result():
    service = MODULE.WavelengthService(FakeEngine())

    result = service.get_source("chunk-1")

    assert result["structuredContent"]["source"]["id"] == "chunk-1"
    assert result["_meta"]["slack"]["blocks"][0]["type"] == "header"


def test_ask_archive_via_responses_validates_mcp_citations():
    service = MODULE.WavelengthService(FakeEngine())
    source = {
        "source_id": "article-1",
        "url": "https://example.com/story",
        "headline": "Story",
        "publish_date": "2026-01-02",
        "content_type": "article",
        "content": "Full article body",
        "passages": [{"passage_id": "passage-1", "text": "Evidence text"}],
    }
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="mcp_call",
                name="search_archive",
                arguments=json.dumps({"query": "parks", "content_types": ["article"]}),
                output=json.dumps({"results": [source]}),
            ),
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(
                        text=json.dumps(
                            {
                                "intro": "This uncited intro should not render.",
                                "claims": [
                                    {
                                        "text": "The archive supports this claim.",
                                        "citations": [
                                            {"source_id": "article-1", "passage_ids": ["passage-1"]},
                                            {"source_id": "article-1", "passage_ids": ["missing"]},
                                        ],
                                    },
                                    {
                                        "text": "This claim has no valid citation.",
                                        "citations": [
                                            {"source_id": "missing-source", "passage_ids": ["passage-1"]}
                                        ],
                                    },
                                ],
                            }
                        )
                    )
                ],
            ),
        ]
    )

    result = service.ask_archive_via_responses("What happened?", response=response)

    assert result["structuredContent"]["intro"] is None
    assert result["structuredContent"]["claims"] == [
        {
            "text": "The archive supports this claim.",
            "citations": [{"source_id": "article-1", "passage_ids": ["passage-1"]}],
        }
    ]
    assert result["content"] == [{"type": "text", "text": "The archive supports this claim. [1]"}]
    assert result["structuredContent"]["sources"][0]["source_id"] == "article-1"
    assert result["structuredContent"]["mcp_calls"] == [
        {"name": "search_archive", "arguments": '{"query": "parks", "content_types": ["article"]}'}
    ]


def test_responses_mcp_tool_config_adds_request_scoped_slack_dm_tool(monkeypatch):
    monkeypatch.setenv("WAVELENGTH_PUBLIC_BASE_URL", "https://wavelength.example")
    monkeypatch.setenv("WAVELENGTH_MCP_AUTH_TOKEN", "secret-token")

    configs = MODULE.WavelengthService.responses_mcp_tool_config(
        {"channel": "D1", "event_ts": "123.456", "user_id": "U1"}
    )

    assert len(configs) == 2
    search_config, slack_config = configs
    assert search_config["type"] == "mcp"
    assert search_config["server_url"] == "https://wavelength.example/mcp/search"
    assert search_config["headers"]["Authorization"] == "Bearer secret-token"
    assert "search_archive" in search_config["allowed_tools"]
    assert "get_prior_slack_dms" not in search_config["allowed_tools"]

    assert slack_config["type"] == "mcp"
    assert slack_config["server_url"] == "https://wavelength.example/mcp/slack"
    assert slack_config["headers"]["Authorization"] == "Bearer secret-token"
    assert slack_config["headers"]["X-Wavelength-Slack-Channel"] == "D1"
    assert slack_config["headers"]["X-Wavelength-Slack-Event-Ts"] == "123.456"
    assert slack_config["headers"]["X-Wavelength-Slack-User-Id"] == "U1"
    assert slack_config["allowed_tools"] == ["get_prior_slack_dms"]
