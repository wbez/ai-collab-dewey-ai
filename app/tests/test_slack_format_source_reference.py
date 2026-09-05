import unittest
from hashlib import sha256

from slack_format import (
    answer_blocks,
    format_transcript_excerpt_text,
    source_reference,
    source_attachment_blocks,
    source_references,
    work_object_description_text,
    work_object_entities,
)


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

    def test_work_object_full_text_uses_custom_description_field(self) -> None:
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

        self.assertEqual(entities[0]["entity_type"], "slack#/entities/item")
        self.assertNotIn("app_unfurl_url", entities[0])
        payload = entities[0]["entity_payload"]
        self.assertNotIn("fields", payload)
        self.assertEqual(payload["display_order"][0], "description")
        self.assertEqual(
            payload["custom_fields"][1],
            {
                "key": "description",
                "label": "Description",
                "value": "Lead paragraph.\n\nSecond paragraph.",
                "type": "string",
            },
        )
        self.assertFalse(
            any(field.get("key") == "full_text" for field in payload["custom_fields"])
        )

    def test_work_object_unfurl_keeps_full_title_and_visible_fields(self) -> None:
        long_title = "This is a complete archive headline that should not be shortened in Work Object metadata"

        entities = work_object_entities(
            [
                {
                    "source_id": "src_456",
                    "url": "https://example.com/story",
                    "title": long_title,
                    "number": 2,
                    "content_type": "article",
                    "publish_date": "2024-05-17",
                    "authors": ["Reporter One"],
                    "excerpt": "A concise source passage that should be visible on the unfurl.",
                }
            ],
            include_excerpt=True,
        )

        payload = entities[0]["entity_payload"]
        self.assertEqual(entities[0]["entity_type"], "slack#/entities/item")
        self.assertNotIn("app_unfurl_url", entities[0])
        self.assertEqual(payload["attributes"]["title"]["text"], long_title)
        self.assertEqual(
            payload["display_order"],
            ["description", "source_url", "date", "author", "excerpt"],
        )
        self.assertNotIn("fields", payload)
        self.assertEqual(
            [(field["key"], field["type"]) for field in payload["custom_fields"]],
            [
                ("source_url", "slack#/types/link"),
                ("date", "slack#/types/date"),
                ("author", "string"),
                ("excerpt", "string"),
                ("description", "string"),
            ],
        )

    def test_transcript_unfurl_uses_time_and_speakers_without_collectiveaccess(self) -> None:
        entities = work_object_entities(
            [
                {
                    "source_id": "transcript_1",
                    "parent_id": "transcript_1",
                    "chunk_id": "transcript_1-4",
                    "url": "https://example.com/transcript",
                    "title": "Broadcast",
                    "number": 1,
                    "content_type": "transcript",
                    "publish_date": "2024-05-17",
                    "speakers": ["Host One", "unknown speaker", "Speaker_2", "Guest Two"],
                    "start_seconds": 65.2,
                    "end_seconds": 130.8,
                    "occurrence_id": "12345",
                    "raw_vtt_excerpt": (
                        "00:01:05.000 --> 00:01:10.000\n"
                        "<v Host One>Hello there.</v>"
                    ),
                }
            ],
            include_excerpt=True,
            include_collectiveaccess=False,
        )

        payload = entities[0]["entity_payload"]
        fields = {field["key"]: field for field in payload["custom_fields"]}
        self.assertEqual(fields["time"]["value"], "01:05-02:11")
        self.assertEqual(fields["speakers"]["label"], "Speakers")
        self.assertEqual(fields["speakers"]["value"], "Host One, Guest Two")
        self.assertNotIn("guests", fields)
        self.assertNotIn("collectiveaccess", fields)
        self.assertIn("time", payload["display_order"])
        self.assertIn("speakers", payload["display_order"])

    def test_transcript_flexpane_includes_collectiveaccess_summary_link(self) -> None:
        entities = work_object_entities(
            [
                {
                    "source_id": "transcript_1",
                    "url": "https://example.com/transcript",
                    "title": "Broadcast",
                    "number": 1,
                    "content_type": "transcript",
                    "occurrence_id": "12345",
                }
            ],
            include_collectiveaccess=True,
        )

        payload = entities[0]["entity_payload"]
        fields = {field["key"]: field for field in payload["custom_fields"]}
        self.assertEqual(fields["collectiveaccess"]["label"], "CollectiveAccess")
        self.assertEqual(fields["collectiveaccess"]["type"], "slack#/types/link")
        self.assertEqual(
            fields["collectiveaccess"]["value"],
            "https://archives.wbez.org/index.php/editor/occurrences/"
            "OccurrenceEditor/Summary/occurrence_id/12345",
        )

    def test_transcript_description_prefers_raw_timestamped_excerpt(self) -> None:
        formatted = format_transcript_excerpt_text(
            "00:01:05.000 --> 00:01:10.000\n"
            "<v Host One>Hello there.</v>\n\n"
            "00:01:11.000 --> 00:01:14.000\n"
            "Guest Two: Good morning."
        )

        self.assertEqual(
            formatted,
            "01:05-01:10 Host One: Hello there.\n\n"
            "01:11-01:14 Guest Two: Good morning.",
        )

    def test_transcript_sources_from_same_recording_share_representation(self) -> None:
        sources = source_references(
            [
                {
                    "source_id": "transcript_1-4",
                    "parent_id": "transcript_1",
                    "chunk_id": "transcript_1-4",
                    "url": "https://example.com/transcript",
                    "title": "Broadcast",
                    "content_type": "transcript",
                },
                {
                    "source_id": "transcript_1-8",
                    "parent_id": "transcript_1",
                    "chunk_id": "transcript_1-8",
                    "url": "https://example.com/transcript",
                    "title": "Broadcast",
                    "content_type": "transcript",
                },
            ]
        )

        self.assertEqual(sources[0]["source_id"], "transcript_1-4,8")
        self.assertEqual(sources[1]["source_id"], "transcript_1-4,8")

    def test_answer_blocks_convert_markdown_to_rich_text_with_inline_attachment_mentions(self) -> None:
        source = {
            "source_id": "https://example.com/story",
            "parent_id": "parent_123",
            "url": "https://example.com/story",
            "title": "Story",
            "number": 1,
        }

        blocks = answer_blocks(
            "A few examples:\n- **Story** _italic_ __underlined__ [site](https://example.com) [1].",
            [source],
            work_object_app_id="A123",
            attachment_locations={"parent_123": {"channel_id": "C123", "ts": "123.456"}},
        )

        self.assertEqual(blocks[0]["type"], "rich_text")
        rich_elements = blocks[0]["elements"]
        self.assertEqual(rich_elements[0]["type"], "rich_text_section")
        self.assertEqual(rich_elements[0]["elements"][0]["text"], "A few examples:")
        self.assertEqual(rich_elements[1]["type"], "rich_text_list")
        self.assertEqual(rich_elements[1]["style"], "bullet")

        item_elements = rich_elements[1]["elements"][0]["elements"]
        self.assertEqual(item_elements[0], {"type": "text", "text": "Story", "style": {"bold": True}})
        self.assertEqual(item_elements[2]["style"], {"italic": True})
        self.assertEqual(item_elements[4]["style"], {"underline": True})
        self.assertEqual(item_elements[6]["type"], "link")
        self.assertEqual(item_elements[6]["text"], "site")
        mention = item_elements[8]
        self.assertEqual(mention["type"], "attachment_mention")
        self.assertEqual(mention["entity_id"], "parent_123")
        self.assertEqual(mention["text"], "[1]")

    def test_answer_blocks_convert_ordered_lists_to_rich_text_lists(self) -> None:
        blocks = answer_blocks("1. First item\n2. Second item", [], work_object_app_id=None)

        self.assertEqual(blocks[0]["type"], "rich_text")
        rich_list = blocks[0]["elements"][0]
        self.assertEqual(rich_list["type"], "rich_text_list")
        self.assertEqual(rich_list["style"], "ordered")
        self.assertEqual(
            [item["elements"][0]["text"] for item in rich_list["elements"]],
            ["First item", "Second item"],
        )

    def test_answer_blocks_support_markdown_asterisk_italic(self) -> None:
        blocks = answer_blocks("This is *italic* text.", [], work_object_app_id=None)

        elements = blocks[0]["elements"][0]["elements"]
        self.assertEqual(elements[1], {"type": "text", "text": "italic", "style": {"italic": True}})

    def test_attachment_mentions_use_same_external_id_as_unfurls(self) -> None:
        source = {
            "source_id": "https://example.com/story",
            "parent_id": "parent_123",
            "url": "https://example.com/story",
            "title": "Story",
            "number": 1,
        }

        blocks = source_attachment_blocks([source], work_object_app_id="A123")
        entities = work_object_entities([source])

        self.assertIsNotNone(blocks)
        section_elements = blocks[0]["elements"][0]["elements"]
        self.assertEqual(section_elements[0], {"type": "text", "text": "Sources:", "style": {"bold": True}})
        mention = section_elements[3]
        self.assertEqual(mention["type"], "attachment_mention")
        self.assertEqual(mention["entity_id"], "parent_123")
        self.assertEqual(entities[0]["external_ref"]["id"], "parent_123")
        self.assertEqual(mention["entity_id"], entities[0]["external_ref"]["id"])

    def test_attachment_mentions_hash_url_ids_like_unfurls(self) -> None:
        url = "https://example.com/story"
        expected = "src_" + sha256(url.encode("utf-8", errors="ignore")).hexdigest()[:20]
        source = {
            "source_id": url,
            "url": url,
            "title": "Story",
            "number": 1,
        }

        blocks = source_attachment_blocks([source], work_object_app_id="A123")
        entities = work_object_entities([source])

        self.assertIsNotNone(blocks)
        mention = blocks[0]["elements"][0]["elements"][3]
        self.assertEqual(mention["entity_id"], expected)
        self.assertEqual(entities[0]["external_ref"]["id"], expected)

    def test_attachment_mentions_hash_url_only_sources_like_unfurls(self) -> None:
        url = "https://example.com/story"
        expected = "src_" + sha256(url.encode("utf-8", errors="ignore")).hexdigest()[:20]
        source = {
            "url": url,
            "title": "Story",
            "number": 1,
        }

        blocks = source_attachment_blocks([source], work_object_app_id="A123")
        entities = work_object_entities([source])

        self.assertIsNotNone(blocks)
        mention = blocks[0]["elements"][0]["elements"][3]
        self.assertEqual(mention["entity_id"], expected)
        self.assertEqual(entities[0]["external_ref"]["id"], expected)

    def test_inline_attachment_mentions_use_channel_id_back_reference(self) -> None:
        source = {
            "source_id": "source_1",
            "url": "https://example.com/story",
            "title": "Story",
            "number": 1,
        }

        blocks = answer_blocks(
            "See [1].",
            [source],
            work_object_app_id="A123",
            attachment_locations={"source_1": {"channel_id": "C123", "ts": "123.456"}},
        )

        mention = blocks[0]["elements"][0]["elements"][1]
        self.assertEqual(mention["type"], "attachment_mention")
        self.assertEqual(mention["entity_id"], "source_1")
        self.assertEqual(mention["app_id"], "A123")
        self.assertEqual(mention["text"], "[1]")
        self.assertEqual(mention["channel_id"], "C123")
        self.assertEqual(mention["ts"], "123.456")
        self.assertNotIn("channel", mention)

    def test_inline_attachment_mentions_remain_text_without_confirmed_location(self) -> None:
        source = {
            "source_id": "source_1",
            "url": "https://example.com/story",
            "title": "Story",
            "number": 1,
        }

        blocks = answer_blocks("See [1].", [source], work_object_app_id="A123")

        elements = blocks[0]["elements"][0]["elements"]
        self.assertFalse(any(element.get("type") == "attachment_mention" for element in elements))
        self.assertEqual(elements[0]["text"], "See [1].")


if __name__ == "__main__":
    unittest.main()
