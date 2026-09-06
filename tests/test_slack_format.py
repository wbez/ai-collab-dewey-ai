import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from slack_format import (
    SLACK_WORK_OBJECT_EMBED_MIME_TYPE,
    answer_blocks,
    block_kit_tool_result,
    cited_source_references,
    markdown_to_slack_mrkdwn,
    replace_source_markers,
    search_blocks,
    work_object_entities,
)


def test_replace_source_markers_renders_slack_links_and_renumbers():
    rendered = replace_source_markers(
        "Second [SRC2], first [SRC1], second again [SRC2], missing [SRC3].",
        {
            1: "https://example.com/first",
            2: "https://example.com/second",
            3: None,
        },
    )

    assert rendered == (
        "Second <https://example.com/second|[1]>, "
        "first <https://example.com/first|[2]>, "
        "second again <https://example.com/second|[1]>, missing [3]."
    )


def test_answer_blocks_renders_plain_answer_sections_without_work_objects():
    sources = [
        {
            "number": 1,
            "source_id": "story-1",
            "title": "Archive Story",
            "url": "https://example.com/story",
            "publish_date": "2026-01-02",
            "content_type": "article",
        }
    ]

    blocks = answer_blocks("Answer <https://example.com/story|[1]>", sources)

    assert blocks[0]["type"] == "section"
    assert blocks[0]["text"]["text"] == "Answer <https://example.com/story|[1]>"
    assert len(blocks) == 1


def test_answer_blocks_renders_work_object_mentions_when_app_id_is_available():
    blocks = answer_blocks(
        "Answer <https://example.com/story|[1]>",
        [
            {
                "number": 1,
                "source_id": "story-1",
                "title": "Archive Story",
                "url": "https://example.com/story",
                "publish_date": "2026-01-02",
                "content_type": "article",
            }
        ],
        work_object_app_id="A123",
    )

    assert blocks[0]["type"] == "rich_text"
    elements = blocks[0]["elements"][0]["elements"]
    assert elements == [
        {"type": "text", "text": "Answer "},
        {
            "type": "work_object_mention",
            "entity_id": "story-1",
            "app_id": "A123",
            "text": "[1]",
            "url": "https://example.com/story",
            "icon_url": "https://mchonofsky-test-bucket.s3.us-east-2.amazonaws.com/cst.jpg",
        },
    ]


def test_slack_links_encode_spaces():
    url = "https://s3.amazonaws.com/cpm-archives-project/16406/OD-990218/Web Copy/OD-990218_02.mp3#t=1219"
    rendered = replace_source_markers("See [SRC8].", {8: url})
    sources = [
        {
            "number": 8,
            "title": "Archive audio",
            "url": url,
            "publish_date": "1999-02-18",
            "content_type": "transcript",
        }
    ]
    blocks = answer_blocks("See [8].", sources)

    expected_url = "https://s3.amazonaws.com/cpm-archives-project/16406/OD-990218/Web%20Copy/OD-990218_02.mp3#t=1219"
    assert rendered == f"See <{expected_url}|[1]>."
    assert blocks == [{"type": "section", "text": {"type": "mrkdwn", "text": "See [8]."}}]


def test_transcript_answer_blocks_can_render_work_object_mentions():
    sources = [{
        "number": 1, "source_id": "t1", "title": "Reset", "content_type": "transcript",
        "publish_date": "2026-01-02", "excerpt": "One two three…",
        "url": "https://example.com/audio.mp3",
    }]

    blocks = answer_blocks("Answer [1]", sources, work_object_app_id="A123")

    assert blocks[0]["type"] == "rich_text"
    mention = blocks[0]["elements"][0]["elements"][1]
    assert mention["type"] == "work_object_mention"
    assert mention["entity_id"] == "t1"
    assert mention["url"] == "https://example.com/audio.mp3"


def test_work_object_entities_add_display_type_and_article_fields():
    entities = work_object_entities([
        {
            "number": 1,
            "source_id": "story-1",
            "title": "Archive Story",
            "url": "https://example.com/story",
            "publish_date": "2026-01-02",
            "content_type": "article",
            "authors": ["Reporter One", "Reporter Two"],
        },
        {
            "number": 2,
            "source_id": "transcript-1",
            "title": "Reset",
            "program": "Reset",
            "url": "https://example.com/audio.mp3",
            "publish_date": "2026-01-03",
            "content_type": "transcript",
        },
    ])

    article_payload = entities[0]["entity_payload"]
    transcript_payload = entities[1]["entity_payload"]
    assert article_payload["attributes"]["display_type"] == "Article"
    assert transcript_payload["attributes"]["display_type"] == "Transcript"
    article_fields = {
        field["key"]: field
        for field in article_payload["custom_fields"]
    }
    assert article_fields["date"]["value"] == "2026-01-02"
    assert article_fields["author"]["value"] == "Reporter One, Reporter Two"


def test_work_object_entities_can_declare_embed_support_without_preview_url():
    entities = work_object_entities(
        [
            {
                "number": 1,
                "source_id": "story-1",
                "title": "Archive Story",
                "url": "https://example.com/story",
                "publish_date": "2026-01-02",
                "content_type": "article",
            }
        ],
        include_embed=True,
    )

    full_size_preview = entities[0]["entity_payload"]["attributes"]["full_size_preview"]
    assert full_size_preview == {
        "is_supported": True,
        "mime_type": SLACK_WORK_OBJECT_EMBED_MIME_TYPE,
    }
    assert "preview_url" not in full_size_preview


def test_cited_source_references_excludes_unused_sources_and_renumbers():
    sources = [
        {"number": 1, "source_id": "one", "url": "https://example.com/one"},
        {"number": 2, "source_id": "two", "url": "https://example.com/two"},
        {"number": 9, "source_id": "nine", "url": "https://example.com/nine"},
    ]

    cited = cited_source_references("Claim [SRC9], then [SRC1], and [SRC9].", sources)

    assert [(source["source_id"], source["number"]) for source in cited] == [
        ("nine", 1),
        ("one", 2),
    ]


def test_markdown_to_slack_mrkdwn_converts_common_answer_formatting():
    rendered = markdown_to_slack_mrkdwn(
        "- **March 13-14, 2024:** Coverage linked [story](https://example.com/story)."
    )

    assert rendered == (
        "- *March 13-14, 2024:* Coverage linked <https://example.com/story|story>."
    )


def test_search_blocks_render_empty_state():
    blocks = search_blocks("missing story", [])

    assert blocks[0]["type"] == "header"
    assert "No matching archive sources" in str(blocks)


def test_block_kit_tool_result_uses_slack_meta_extension():
    result = block_kit_tool_result(
        "fallback",
        [{"type": "section", "text": {"type": "mrkdwn", "text": "fallback"}}],
        {"answer": "fallback"},
    )

    assert result["content"] == [{"type": "text", "text": "fallback"}]
    assert result["structuredContent"] == {"answer": "fallback"}
    assert result["_meta"]["slack"]["blocks"][0]["type"] == "section"
