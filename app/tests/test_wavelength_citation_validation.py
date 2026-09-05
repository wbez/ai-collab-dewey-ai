import sys
import types
import unittest
from types import SimpleNamespace


if "dotenv" not in sys.modules:
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv

if "dewey" not in sys.modules:
    dewey = types.ModuleType("dewey")

    class Dewey:
        pass

    dewey.Dewey = Dewey
    sys.modules["dewey"] = dewey

if "models" not in sys.modules:
    models = types.ModuleType("models")

    class AzureOpenAIConfig:
        pass

    class AzureSearchConfig:
        pass

    models.AzureOpenAIConfig = AzureOpenAIConfig
    models.AzureSearchConfig = AzureSearchConfig
    sys.modules["models"] = models

if "models.core" not in sys.modules:
    models_core = types.ModuleType("models.core")
    models_core.resolve_embedding_dimensions = lambda *args, **kwargs: None
    sys.modules["models.core"] = models_core

import wavelength_service


class TestCitationValidation(unittest.TestCase):
    def test_structured_answer_schema_uses_inline_answer_text(self) -> None:
        schema = wavelength_service.WavelengthService.structured_answer_schema()["schema"]

        self.assertEqual(schema["required"], ["answer"])
        self.assertIn("answer", schema["properties"])
        self.assertNotIn("claims", schema["properties"])

    def test_citation_map_for_sources_uses_compact_keys(self) -> None:
        sources = [
            {
                "source_id": "source_1",
                "content_type": "article",
                "title": "Article",
                "published_at": "2024-01-01",
                "url": "https://example.com/story",
                "passages": [
                    {
                        "passage_id": "very-long-passage-id",
                        "passage_ids": ["very-long-passage-id"],
                        "text": "Long context text.",
                    }
                ],
            }
        ]

        citation_map = wavelength_service._citation_map_for_sources(sources)

        self.assertEqual(len(citation_map), 1)
        self.assertRegex(citation_map[0]["citation_key"], r"^c_[0-9a-f]{8}$")
        self.assertEqual(citation_map[0]["source_id"], "source_1")
        self.assertNotIn("text", citation_map[0])

    def test_resolve_inline_citations_preserves_uncited_text(self) -> None:
        resolved = wavelength_service.resolve_inline_citations(
            "This sentence has no citation.",
            {},
        )

        self.assertEqual(resolved["answer"], "This sentence has no citation.")
        self.assertEqual(resolved["sources"], [])
        self.assertEqual(resolved["warnings"], [])

    def test_resolve_inline_citations_replaces_valid_marker(self) -> None:
        citation_map = {
            "c_1234abcd": {
                "citation_key": "c_1234abcd",
                "source_id": "source_1",
                "passage_id": "p1",
                "passage_ids": ["p1"],
                "content_type": "article",
                "title": "Article",
                "published_at": "2024-01-01",
                "url": "https://example.com/story",
            }
        }

        resolved = wavelength_service.resolve_inline_citations(
            "The archive has this {{cite:c_1234abcd}} in context.",
            citation_map,
        )

        self.assertEqual(resolved["answer"], "The archive has this [1] in context.")
        self.assertEqual(len(resolved["sources"]), 1)
        self.assertEqual(resolved["sources"][0]["number"], 1)
        self.assertEqual(resolved["warnings"], [])

    def test_resolve_inline_citations_warns_without_dropping_text(self) -> None:
        resolved = wavelength_service.resolve_inline_citations(
            "The archive has this {{cite:c_missing}}.",
            {},
        )

        self.assertEqual(resolved["answer"], "The archive has this.")
        self.assertEqual(
            resolved["warnings"],
            ["Some citation markers could not be resolved; answer text was preserved."],
        )
        self.assertEqual(resolved["unresolved_citation_keys"], ["c_missing"])

    def test_responses_assembly_preserves_uncited_text_and_resolves_handles(self) -> None:
        service = wavelength_service.WavelengthService(engine=object())
        response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="mcp_call",
                    name="search_archive",
                    arguments='{"query":"test"}',
                    output={
                        "citation_map": [
                            {
                                "citation_key": "c_1234abcd",
                                "source_id": "source_1",
                                "passage_id": "p1",
                                "passage_ids": ["p1"],
                                "content_type": "article",
                                "title": "Article",
                                "published_at": "2024-01-01",
                                "url": "https://example.com/story",
                            }
                        ]
                    },
                ),
                SimpleNamespace(
                    type="message",
                    content=[
                        {
                            "text": (
                                '{"answer":"Cited text {{cite:c_1234abcd}}. '
                                'Uncited but appropriate context."}'
                            )
                        }
                    ],
                ),
            ]
        )

        payload = service.ask_archive_via_responses(
            "question",
            conversation_context=[],
            response=response,
        )

        text = payload["content"][0]["text"]
        self.assertIn("Cited text [1].", text)
        self.assertIn("Uncited but appropriate context.", text)
        self.assertEqual(payload["structuredContent"]["warnings"], [])
        self.assertEqual(len(payload["structuredContent"]["sources"]), 1)

    def test_responses_assembly_converts_markdown_to_slack_mrkdwn(self) -> None:
        service = wavelength_service.WavelengthService(engine=object())
        response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="mcp_call",
                    name="search_archive",
                    arguments='{"query":"test"}',
                    output={
                        "citation_map": [
                            {
                                "citation_key": "c_1234abcd",
                                "source_id": "source_1",
                                "passage_id": "p1",
                                "passage_ids": ["p1"],
                                "content_type": "article",
                                "title": "Mideast to Become Medill Training Ground?",
                                "published_at": "2007-04-06",
                                "url": "https://example.com/story",
                            }
                        ]
                    },
                ),
                SimpleNamespace(
                    type="message",
                    content=[
                        {
                            "text": (
                                '{"answer":"- **Mideast to Become Medill Training Ground?** '
                                '{{cite:c_1234abcd}}"}'
                            )
                        }
                    ],
                ),
            ]
        )

        payload = service.ask_archive_via_responses(
            "question",
            conversation_context=[],
            response=response,
        )

        text = payload["content"][0]["text"]
        self.assertIn("- *Mideast to Become Medill Training Ground?* [1]", text)
        self.assertNotIn("**Mideast", text)
        block = payload["_meta"]["slack"]["blocks"][0]
        self.assertEqual(block["type"], "rich_text")
        rich_list = block["elements"][0]
        self.assertEqual(rich_list["type"], "rich_text_list")
        self.assertEqual(rich_list["style"], "bullet")
        item_elements = rich_list["elements"][0]["elements"]
        self.assertEqual(
            item_elements[0],
            {
                "type": "text",
                "text": "Mideast to Become Medill Training Ground?",
                "style": {"bold": True},
            },
        )
        self.assertEqual(item_elements[1]["text"], " [1]")

    def test_responses_assembly_warns_on_uncited_claims_without_dropping(self) -> None:
        service = wavelength_service.WavelengthService(engine=object())
        response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="message",
                    content=[
                        {
                            "text": (
                                '{"intro":null,"claims":[{"text":"Useful uncited context.",'
                                '"citations":[]}]}'
                            )
                        }
                    ],
                ),
            ]
        )

        payload = service.ask_archive_via_responses(
            "question",
            conversation_context=[],
            response=response,
        )

        self.assertIn("Useful uncited context.", payload["content"][0]["text"])
        self.assertEqual(payload["structuredContent"]["uncited_claim_count"], 1)
        self.assertEqual(
            payload["structuredContent"]["warnings"],
            ["Some structured claims had no citation markers; answer text was preserved."],
        )

    def test_validate_citations_reports_unknown_passage_ids(self) -> None:
        answer = {
            "intro": None,
            "claims": [
                {
                    "text": "A claim.",
                    "citations": [
                        {"source_id": "src_1", "passage_ids": ["p_missing"]},
                    ],
                }
            ],
        }
        sources = {
            "src_1": {
                "source_id": "src_1",
                "passages": [{"passage_id": "p1", "passage_ids": ["p1", "p2"]}],
            }
        }

        validated = wavelength_service.validate_citations(answer, sources)

        self.assertEqual(validated["claims"], [])
        self.assertEqual(len(validated["_rejected_citations"]), 1)
        self.assertEqual(
            validated["_rejected_citations"][0]["reason"],
            "unknown_passage_ids",
        )
        self.assertEqual(
            validated["_rejected_citations"][0]["known_passage_ids"],
            ["p1", "p2"],
        )

    def test_validate_citations_reports_unknown_source_ids(self) -> None:
        answer = {
            "intro": None,
            "claims": [
                {
                    "text": "A claim.",
                    "citations": [
                        {"source_id": "src_missing", "passage_ids": ["p1"]},
                    ],
                }
            ],
        }

        validated = wavelength_service.validate_citations(answer, {})

        self.assertEqual(validated["claims"], [])
        self.assertEqual(
            validated["_rejected_citations"][0]["reason"],
            "unknown_source_id",
        )

    def test_source_summary_drops_full_text_fields(self) -> None:
        summary = wavelength_service._source_summary(
            {
                "source_id": "src_1",
                "title": "Title",
                "content": "Very long article body",
                "full_text": "Very long full text",
                "passages": [
                    {
                        "passage_id": "p1",
                        "passage_ids": ["p1"],
                        "text": "Very long passage text",
                        "raw_vtt_excerpt": "Very long VTT",
                    }
                ],
            }
        )

        self.assertNotIn("content", summary)
        self.assertNotIn("full_text", summary)
        self.assertNotIn("text", summary["passages"][0])
        self.assertNotIn("raw_vtt_excerpt", summary["passages"][0])


if __name__ == "__main__":
    unittest.main()
