import unittest
import sys
import types


def _install_azure_test_stubs() -> None:
    azure = types.ModuleType("azure")
    azure_core = types.ModuleType("azure.core")
    azure_core_credentials = types.ModuleType("azure.core.credentials")
    azure_core_exceptions = types.ModuleType("azure.core.exceptions")
    azure_search = types.ModuleType("azure.search")
    azure_search_documents = types.ModuleType("azure.search.documents")
    azure_search_documents_models = types.ModuleType("azure.search.documents.models")

    class AzureKeyCredential:
        def __init__(self, *args, **kwargs):
            pass

    class SearchClient:
        def __init__(self, *args, **kwargs):
            pass

    class HttpResponseError(Exception):
        pass

    class ServiceRequestError(Exception):
        pass

    class QueryType:
        FULL = "full"
        SEMANTIC = "semantic"

    class VectorQuery:
        pass

    class VectorizedQuery:
        def __init__(self, *args, **kwargs):
            pass

    azure_core_credentials.AzureKeyCredential = AzureKeyCredential
    azure_core_exceptions.HttpResponseError = HttpResponseError
    azure_core_exceptions.ServiceRequestError = ServiceRequestError
    azure_search_documents.SearchClient = SearchClient
    azure_search_documents_models.QueryType = QueryType
    azure_search_documents_models.VectorQuery = VectorQuery
    azure_search_documents_models.VectorizedQuery = VectorizedQuery

    sys.modules.setdefault("azure", azure)
    sys.modules.setdefault("azure.core", azure_core)
    sys.modules.setdefault("azure.core.credentials", azure_core_credentials)
    sys.modules.setdefault("azure.core.exceptions", azure_core_exceptions)
    sys.modules.setdefault("azure.search", azure_search)
    sys.modules.setdefault("azure.search.documents", azure_search_documents)
    sys.modules.setdefault("azure.search.documents.models", azure_search_documents_models)


_install_azure_test_stubs()

from wavelength_service import validate_citations


class TestValidateCitations(unittest.TestCase):
    def test_accepts_merged_transcript_passage_ids(self) -> None:
        answer = {
            "intro": None,
            "claims": [
                {
                    "text": "The transcript discusses the source material.",
                    "citations": [
                        {
                            "source_id": "transcript_123",
                            "passage_ids": ["chunk-1", "chunk-2"],
                        }
                    ],
                }
            ],
        }
        sources = {
            "transcript_123": {
                "source_id": "transcript_123",
                "content_type": "transcript",
                "passages": [
                    {
                        "passage_id": "chunk-1",
                        "passage_ids": ["chunk-1", "chunk-2"],
                        "text": "Merged transcript passage.",
                    }
                ],
            }
        }

        validated = validate_citations(answer, sources)

        self.assertEqual(validated["claims"], answer["claims"])

    def test_rejects_unknown_passage_ids(self) -> None:
        answer = {
            "intro": None,
            "claims": [
                {
                    "text": "The transcript discusses the source material.",
                    "citations": [
                        {
                            "source_id": "transcript_123",
                            "passage_ids": ["chunk-1", "not-returned"],
                        }
                    ],
                }
            ],
        }
        sources = {
            "transcript_123": {
                "source_id": "transcript_123",
                "content_type": "transcript",
                "passages": [
                    {
                        "passage_id": "chunk-1",
                        "passage_ids": ["chunk-1", "chunk-2"],
                        "text": "Merged transcript passage.",
                    }
                ],
            }
        }

        validated = validate_citations(answer, sources)

        self.assertEqual(validated["claims"], [])


if __name__ == "__main__":
    unittest.main()
