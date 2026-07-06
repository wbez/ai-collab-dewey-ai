import importlib.util
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

MODULE_PATH = APP_DIR / "dewey.py"
SPEC = importlib.util.spec_from_file_location("dewey", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeEmbeddingsClient:
    def create(self, **kwargs):
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])


class FakeOpenAIClient:
    def __init__(self):
        self.embeddings = FakeEmbeddingsClient()


class FakeSearchClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def make_dewey(search_responses):
    dewey = object.__new__(MODULE.Dewey)
    dewey.oai_client = FakeOpenAIClient()
    dewey.search_client = FakeSearchClient(search_responses)
    dewey.openai_config = SimpleNamespace(
        embedding_deployment="embedding-test",
        embedding_dimensions=3,
    )
    return dewey


def test_build_filter_normalizes_and_escapes_author_names():
    dewey = make_dewey([])

    metadata = {
        "question": "What did these authors write?",
        "date_range": {"start_date": None, "end_date": None},
        "authors": [{"name": " Maureen  O'Connor "}, {"name": "WILL BUNCH"}],
        "speakers": [],
        "guests": [],
        "content_types": [],
        "program": None,
    }

    assert dewey.build_filter(metadata) == (
        "authors/any(a: a eq 'Maureen O''Connor' or a eq 'WILL BUNCH')"
    )


def test_build_filter_supports_transcript_fields():
    dewey = make_dewey([])
    metadata = {
        "question": "What did John say on Radio Times?",
        "date_range": {"start_date": "2024-01-01", "end_date": "2024-12-31"},
        "authors": [],
        "speakers": [{"name": "John Doe"}],
        "guests": [{"name": "Jane Guest"}],
        "content_types": ["transcript"],
        "program": "Radio Times",
    }

    assert dewey.build_filter(metadata) == (
        "publish_date ge 2024-01-01T00:00:00Z and "
        "publish_date le 2024-12-31T23:59:59Z and "
        "(content_type eq 'transcript') and "
        "program eq 'Radio Times' and "
        "guests/any(a: a eq 'Jane Guest') and "
        "speakers/any(a: a eq 'John Doe')"
    )


def test_retrieve_articles_uses_fuzzy_fallback_for_author_queries():
    fuzzy_page = {
        "url": "https://example.com/story",
        "publish_date": "2026-01-02T12:00:00Z",
        "authors": ["Will Bnch"],
        "headline": "Story",
        "content": "Article body",
        "content_type": "article",
    }
    dewey = make_dewey([[], [fuzzy_page]])

    metadata = {
        "question": "septa strike",
        "date_range": {"start_date": None, "end_date": None},
        "authors": [{"name": "Will Bunch"}],
        "speakers": [],
        "guests": [],
        "content_types": ["article"],
        "program": None,
    }

    sources = dewey.retrieve_articles(metadata)

    assert len(sources) == 1
    assert len(dewey.search_client.calls) == 2

    first_call, second_call = dewey.search_client.calls
    assert first_call["search_text"] == metadata["question"]
    assert first_call["query_type"] == MODULE.QueryType.SEMANTIC
    assert "vector_queries" in first_call
    assert "authors/any" in first_call["filter"]

    assert second_call["query_type"] == MODULE.QueryType.FULL
    assert second_call["search_fields"] == ["search_text", "author_search_text", "speaker_search_text"]
    assert "author_search_text:will~" in second_call["search_text"]
    assert "author_search_text:bunch~" in second_call["search_text"]
    assert "search_text:septa" in second_call["search_text"]
    assert "vector_queries" not in second_call


def test_retrieve_articles_formats_transcript_sources():
    transcript_page = {
        "citation_url": "https://example.com/audio.mp3#t=1",
        "transcript_url": "https://example.com/transcript",
        "recording_urls": ["https://example.com/audio.mp3"],
        "publish_date": "2026-01-02T12:00:00Z",
        "authors": [],
        "speakers": ["Host", "Guest"],
        "guests": ["Guest"],
        "program": "Radio Times",
        "title": "Episode Title",
        "chunk_text": "Transcript body",
        "content_type": "transcript",
        "chunk_id": "chunk-1",
        "start_seconds": 1.0,
        "end_seconds": 7.0,
    }
    dewey = make_dewey([[transcript_page]])

    metadata = {
        "question": "What did the guest say?",
        "date_range": {"start_date": None, "end_date": None},
        "authors": [],
        "speakers": [],
        "guests": [],
        "content_types": ["transcript"],
        "program": None,
    }

    sources = dewey.retrieve_articles(metadata)

    payload = MODULE.json.loads(sources[0])
    assert payload == {
        "content_type": "transcript",
        "publish_date": "2026-01-02",
        "speakers": ["Host", "Guest"],
        "start_time": "00:00:01.000",
        "end_time": "00:00:07.000",
        "content": "Transcript body",
    }


def test_retrieve_articles_formats_article_sources():
    article_page = {
        "url": "https://example.com/story",
        "publish_date": "2026-01-02T12:00:00Z",
        "authors": ["Reema Saleh"],
        "headline": "Story",
        "content": "Article body",
        "content_type": "article",
    }
    dewey = make_dewey([[article_page]])

    metadata = {
        "question": "What changed?",
        "date_range": {"start_date": None, "end_date": None},
        "authors": [],
        "speakers": [],
        "guests": [],
        "content_types": ["article"],
        "program": None,
    }

    sources = dewey.retrieve_articles(metadata)

    payload = MODULE.json.loads(sources[0])
    assert payload == {
        "content_type": "article",
        "publish_date": "2026-01-02",
        "authors": ["Reema Saleh"],
        "headline": "Story",
        "content": "Article body",
    }


def test_build_source_url_map_prefers_transcript_url_for_transcripts():
    dewey = make_dewey([])

    results = [
        {
            "content_type": "transcript",
            "transcript_url": "https://example.com/transcript",
            "recording_urls": ["https://example.com/audio.mp3"],
        }
    ]

    assert dewey._build_source_url_map(results) == {1: "https://example.com/transcript"}


def test_build_source_url_map_prefers_citation_url_for_transcripts():
    dewey = make_dewey([])

    results = [
        {
            "content_type": "transcript",
            "citation_url": "https://example.com/part-3.mp3#t=69",
            "transcript_url": "https://example.com/transcript",
            "recording_urls": [
                "https://example.com/part-1.mp3",
                "https://example.com/part-2.mp3",
                "https://example.com/part-3.mp3",
            ],
        }
    ]

    assert dewey._build_source_url_map(results) == {1: "https://example.com/part-3.mp3#t=69"}


def test_build_source_url_map_falls_back_to_recording_url_for_transcripts():
    dewey = make_dewey([])

    results = [
        {
            "content_type": "transcript",
            "url": None,
            "transcript_url": None,
            "recording_urls": ["https://example.com/audio.mp3"],
        }
    ]

    assert dewey._build_source_url_map(results) == {1: "https://example.com/audio.mp3"}


def test_replace_source_markers_renders_clickable_citation_links():
    dewey = make_dewey([])

    rendered = dewey._replace_source_markers(
        "Transcript match [SRC1] and article match [SRC2].",
        {
            1: "https://example.com/transcript",
            2: "https://example.com/story",
        },
    )

    assert rendered == (
        'Transcript match <a href="https://example.com/transcript" target="_blank" '
        'rel="noopener noreferrer">[1]</a> and article match '
        '<a href="https://example.com/story" target="_blank" rel="noopener noreferrer">[2]</a>.'
    )


def test_replace_source_markers_leaves_plain_label_without_url():
    dewey = make_dewey([])

    rendered = dewey._replace_source_markers("No link [SRC1].", {1: None})

    assert rendered == "No link [1]."


def test_replace_source_markers_renumbers_by_first_appearance_in_answer():
    dewey = make_dewey([])

    rendered = dewey._replace_source_markers(
        "Second source first [SRC2], then first source [SRC1], then second again [SRC2].",
        {
            1: "https://example.com/first",
            2: "https://example.com/second",
        },
    )

    assert rendered == (
        'Second source first <a href="https://example.com/second" target="_blank" '
        'rel="noopener noreferrer">[1]</a>, then first source '
        '<a href="https://example.com/first" target="_blank" rel="noopener noreferrer">[2]</a>, '
        'then second again <a href="https://example.com/second" target="_blank" '
        'rel="noopener noreferrer">[1]</a>.'
    )


def test_replace_source_markers_reuses_number_for_duplicate_footnote_targets():
    dewey = make_dewey([])

    rendered = dewey._replace_source_markers(
        "First duplicate [SRC2], unique [SRC1], duplicate again [SRC3].",
        {
            1: "https://example.com/unique",
            2: "https://example.com/shared",
            3: "https://example.com/shared",
        },
    )

    assert rendered == (
        'First duplicate <a href="https://example.com/shared" target="_blank" '
        'rel="noopener noreferrer">[1]</a>, unique '
        '<a href="https://example.com/unique" target="_blank" rel="noopener noreferrer">[2]</a>, '
        'duplicate again <a href="https://example.com/shared" target="_blank" '
        'rel="noopener noreferrer">[1]</a>.'
    )
