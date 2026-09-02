import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

azure_module = sys.modules.setdefault("azure", types.ModuleType("azure"))
azure_core_module = sys.modules.setdefault("azure.core", types.ModuleType("azure.core"))
azure_credentials_module = sys.modules.setdefault(
    "azure.core.credentials",
    types.ModuleType("azure.core.credentials"),
)
azure_storage_module = sys.modules.setdefault("azure.storage", types.ModuleType("azure.storage"))
azure_storage_blob_module = sys.modules.setdefault(
    "azure.storage.blob",
    types.ModuleType("azure.storage.blob"),
)
azure_storage_blob_aio_module = sys.modules.setdefault(
    "azure.storage.blob.aio",
    types.ModuleType("azure.storage.blob.aio"),
)

class AzureKeyCredential:
    def __init__(self, key):
        self.key = key

class BlobServiceClient:
    @classmethod
    def from_connection_string(cls, connection_string):
        raise RuntimeError("BlobServiceClient should not be used in this test")

azure_credentials_module.AzureKeyCredential = AzureKeyCredential
azure_storage_blob_aio_module.BlobServiceClient = BlobServiceClient

azure_module.core = azure_core_module
azure_module.storage = azure_storage_module
azure_core_module.credentials = azure_credentials_module
azure_storage_module.blob = azure_storage_blob_module
azure_storage_blob_module.aio = azure_storage_blob_aio_module

if "dotenv" not in sys.modules:
    dotenv_module = types.ModuleType("dotenv")
    dotenv_module.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv_module

if "tqdm" not in sys.modules:
    tqdm_module = types.ModuleType("tqdm")

    class DummyTqdm:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            if self.iterable is None:
                return iter(())
            return iter(self.iterable)

        def update(self, value):
            return None

        def set_postfix(self, **kwargs):
            return None

        def close(self):
            return None

    tqdm_module.tqdm = DummyTqdm
    sys.modules["tqdm"] = tqdm_module

if "setup" not in sys.modules:
    setup_package = types.ModuleType("setup")
    setup_package.EmbeddingService = type("EmbeddingService", (), {})
    setup_package.SearchInfo = type("SearchInfo", (), {})
    setup_package.SearchManager = type("SearchManager", (), {})
    sys.modules["setup"] = setup_package

if "models" not in sys.modules:
    models_package = types.ModuleType("models")
    models_core_module = types.ModuleType("models.core")
    models_core_module.resolve_embedding_dimensions = lambda model_name, dimensions: dimensions or 1536
    sys.modules["models"] = models_package
    sys.modules["models.core"] = models_core_module

SETUP_MODULE_PATH = APP_DIR / "setup.py"
SPEC = importlib.util.spec_from_file_location("dewey_setup_script", SETUP_MODULE_PATH)
SETUP_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SETUP_MODULE)
SetupManager = SETUP_MODULE.SetupManager


class FailingSearchManager:
    def __init__(self):
        self.deleted_parent_ids = []
        self.uploaded_parent_ids = []

    async def delete_transcript_chunks(self, parent_id: str) -> int:
        self.deleted_parent_ids.append(parent_id)
        return 1

    async def upload_transcript_chunks(self, chunks, progress_callback=None):
        parent_id = chunks[0]["parent_id"]
        self.uploaded_parent_ids.append(parent_id)
        if progress_callback is not None:
            progress_callback(len(chunks))
        if parent_id == "source-second":
            raise RuntimeError("boom")


def test_upload_transcript_chunks_streams_and_checkpoints(tmp_path):
    manager = SetupManager()
    manager.transcript_state_path = tmp_path / ".ingest-state.json"
    search_manager = FailingSearchManager()

    loaded = []
    bundles = {
        "first.vtt": {
            "document": {"id": "first"},
            "chunks": [{"parent_id": "source-first"}],
            "source_hash": "source-first",
            "ingest_hash": "hash-first",
        },
        "second.vtt": {
            "document": {"id": "second"},
            "chunks": [{"parent_id": "source-second"}],
            "source_hash": "source-second",
            "ingest_hash": "hash-second",
        },
        "third.vtt": {
            "document": {"id": "third"},
            "chunks": [{"parent_id": "source-third"}],
            "source_hash": "source-third",
            "ingest_hash": "hash-third",
        },
    }

    def fake_load_transcript_bundle(vtt_path, sidecar_path):
        loaded.append(vtt_path.name)
        return bundles[vtt_path.name]

    manager.load_transcript_bundle = fake_load_transcript_bundle

    transcript_pairs = [
        (tmp_path / "first.vtt", tmp_path / "first.json"),
        (tmp_path / "second.vtt", tmp_path / "second.json"),
        (tmp_path / "third.vtt", tmp_path / "third.json"),
    ]

    try:
        asyncio.run(
            manager.upload_transcript_chunks(search_manager, transcript_pairs, {"transcripts": {}})
        )
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("expected upload_transcript_chunks to raise")

    assert loaded == ["first.vtt", "second.vtt"]
    assert search_manager.deleted_parent_ids == ["source-first", "source-second"]
    assert search_manager.uploaded_parent_ids == ["source-first", "source-second"]
    assert json.loads(manager.transcript_state_path.read_text(encoding="utf-8")) == {
        "transcripts": {"first": {"ingest_hash": "hash-first", "parent_id": "source-first"}}
    }


def test_compute_file_md5_reads_source_bytes(tmp_path):
    manager = SetupManager()
    source_path = tmp_path / "episode.vtt"
    source_path.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHi\n", encoding="utf-8")

    assert manager.compute_file_md5(source_path) == "262d4a11c477765132390c8b4c505acf"


def test_iter_article_documents_adds_stable_source_id(tmp_path):
    manager = SetupManager()
    article_path = tmp_path / "article.json"
    article_path.write_text(
        json.dumps(
            {
                "id": "article-1",
                "headline": "Story",
                "content": "Body",
                "url": "https://example.com/story",
                "authors": ["Reporter"],
                "publish_date": "2026-01-02T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    document = next(manager.iter_article_documents([article_path]))
    source_id = document.pop("source_id")

    assert source_id == manager.compute_stable_hash(document)


def test_iter_article_documents_streams_top_level_array(tmp_path, monkeypatch):
    manager = SetupManager()
    article_path = tmp_path / "articles.json"
    article_path.write_text(
        json.dumps(
            [
                {
                    "id": "article-1",
                    "headline": "Story 1",
                    "content": "Body 1",
                    "url": "https://example.com/story-1",
                    "authors": ["Reporter"],
                    "publish_date": "2026-01-02T00:00:00Z",
                },
                {
                    "id": "article-2",
                    "headline": "Story 2",
                    "content": "Body 2",
                    "url": "https://example.com/story-2",
                    "authors": ["Reporter"],
                    "publish_date": "2026-01-03T00:00:00Z",
                },
            ]
        ),
        encoding="utf-8",
    )

    def fail_read_text(self, *args, **kwargs):
        raise AssertionError("article exports should not be read into memory")

    monkeypatch.setattr(Path, "read_text", fail_read_text)

    documents = list(manager.iter_article_documents([article_path]))

    assert [document["id"] for document in documents] == ["article-1", "article-2"]
    assert all(document["source_id"] for document in documents)


class FakeBlob:
    def __init__(self, name, metadata=None):
        self.name = name
        self.metadata = metadata or {}


class FakeBlobClient:
    def __init__(self, uploads, container, blob):
        self.uploads = uploads
        self.container = container
        self.blob = blob

    async def upload_blob(self, data, overwrite=False, metadata=None):
        self.uploads.append(
            {
                "container": self.container,
                "blob": self.blob,
                "data": data,
                "overwrite": overwrite,
                "metadata": metadata or {},
            }
        )


class FakeContainerClient:
    def __init__(self, blobs):
        self.blobs = blobs

    async def create_container(self):
        return None

    async def list_blobs(self, include=None):
        for blob in self.blobs:
            yield blob


class FakeBlobServiceClient:
    def __init__(self, blobs):
        self.container_client = FakeContainerClient(blobs)
        self.uploads = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def get_container_client(self, container_name):
        return self.container_client

    def get_blob_client(self, container, blob):
        return FakeBlobClient(self.uploads, container, blob)


def test_upload_documents_to_blob_uses_hash_tables_for_rejection(monkeypatch):
    manager = SetupManager()
    existing_doc = {
        "id": "existing",
        "headline": "Existing",
        "content": "Already uploaded",
        "url": "https://example.com/existing",
        "authors": ["A"],
        "publish_date": "2026-01-01",
    }
    duplicate_first = {
        "headline": "Duplicate",
        "content": "Same body",
        "url": "https://example.com/duplicate",
        "authors": ["B"],
        "publish_date": "2026-01-02",
    }
    duplicate_second = dict(duplicate_first)

    existing_hash = manager.compute_stable_hash(existing_doc)
    fake_service_client = FakeBlobServiceClient(
        [FakeBlob("doc_existing.json", {"content_hash": existing_hash})]
    )

    class FakeBlobServiceClientFactory:
        @classmethod
        def from_connection_string(cls, connection_string):
            assert connection_string == "UseDevelopmentStorage=true"
            return fake_service_client

    monkeypatch.setattr(SETUP_MODULE, "BlobServiceClient", FakeBlobServiceClientFactory)

    result = asyncio.run(
        manager.upload_documents_to_blob(
            "UseDevelopmentStorage=true",
            "articles",
            [existing_doc, duplicate_first, duplicate_second],
        )
    )

    assert result == {"uploaded": 1, "skipped": 2, "failed": 0}
    assert [upload["blob"] for upload in fake_service_client.uploads] == ["doc_1.json"]
