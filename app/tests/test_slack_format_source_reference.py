import unittest
from hashlib import sha256

from slack_format import source_reference, work_object_description_text, work_object_entities


class TestSourceReference(unittest.TestCase):
    def test_prefers_parent_id_over_url_source_id(self) -> None:
        source = {
            "source_id": "https://example.com/story",
            "parent_id": "parent_123",
            "url": "https://example.com/story",
            "title": "Story",
        }
        ref = source_reference(source, 1)
        self.assertEqual(ref["id"], "parent_123")
        self.assertEqual(ref["source_id"], "parent_123")

    def test_prefers_chunk_id_over_url_source_id(self) -> None:
        source = {
            "source_id": "https://example.com/story",
            "chunk_id": "chunk_456",
            "url": "https://example.com/story",
            "title": "Story",
        }
        ref = source_reference(source, 1)
        self.assertEqual(ref["id"], "chunk_456")
        self.assertEqual(ref["source_id"], "chunk_456")

    def test_hashes_url_only_ids(self) -> None:
        url = "https://example.com/story"
        source = {
            "source_id": url,
            "url": url,
            "title": "Story",
        }
        ref = source_reference(source, 1)
        self.assertTrue(ref["id"].startswith("src_"))
        expected = "src_" + sha256(url.encode("utf-8", errors="ignore")).hexdigest()[:20]
        self.assertEqual(ref["id"], expected)

    def test_keeps_non_url_source_id(self) -> None:
        source = {
            "source_id": "occ_999",
            "url": "https://example.com/story",
            "title": "Story",
        }
        ref = source_reference(source, 1)
        self.assertEqual(ref["id"], "occ_999")
        self.assertEqual(ref["source_id"], "occ_999")

    def test_work_object_description_preserves_existing_paragraphs(self) -> None:
        text = "First paragraph.\n\nSecond paragraph."

        self.assertEqual(work_object_description_text(text), text)

    def test_work_object_description_paragraphizes_flat_article_text(self) -> None:
        text = " ".join(f"Sentence {index} has enough article text." for index in range(1, 80))

        formatted = work_object_description_text(text)

        self.assertIn("\n\n", formatted)

    def test_work_object_full_text_uses_native_description_field(self) -> None:
        entities = work_object_entities(
            [
                {
                    "source_id": "src_123",
                    "url": "https://example.com/story",
                    "title": "Story",
                    "number": 1,
                    "content_type": "article",
                    "full_text": "Lead paragraph.\n\nSecond paragraph.",
                }
            ],
            include_full_text=True,
        )

        payload = entities[0]["entity_payload"]
        self.assertEqual(
            payload["fields"]["description"],
            {"value": "Lead paragraph.\n\nSecond paragraph.", "format": "markdown"},
        )
        self.assertEqual(payload["display_order"][0], "description")
        self.assertFalse(
            any(field.get("key") == "full_text" for field in payload["custom_fields"])
        )


if __name__ == "__main__":
    unittest.main()
