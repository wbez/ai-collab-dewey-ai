from transcripts import (
    append_timestamp_fragment,
    build_citation_url,
    build_name_search_text,
    format_timestamp,
    normalize_recording_files,
    parse_note_file_entry,
    parse_timestamp,
)


def test_timestamp_helpers_round_trip_milliseconds():
    assert parse_timestamp("01:02:03.456") == 3723.456
    assert format_timestamp(3723.456) == "01:02:03.456"


def test_append_timestamp_fragment_preserves_existing_fragment_separator():
    assert append_timestamp_fragment("https://example.com/audio.mp3", 12.9) == (
        "https://example.com/audio.mp3#t=12"
    )
    assert append_timestamp_fragment("https://example.com/audio.mp3#track=1", 12.9) == (
        "https://example.com/audio.mp3#track=1&t=12"
    )


def test_parse_note_file_entry_coerces_numbers_and_quoted_strings():
    parsed = parse_note_file_entry('source_url="https://example.com/a.mp3"; length=120.5; size=1024')

    assert parsed == {
        "source_url": "https://example.com/a.mp3",
        "length": 120.5,
        "size": 1024,
    }


def test_normalize_recording_files_accepts_dicts_strings_and_note_fallbacks():
    assert normalize_recording_files(
        {
            "recording_urls": [
                {"url": "https://example.com/One%20File.mp3", "length": 60, "size": "2048"},
                "https://example.com/two.mp3",
                {"url": ""},
            ]
        }
    ) == [
        {
            "url": "https://example.com/One%20File.mp3",
            "filename": "One File.mp3",
            "length_seconds": 60.0,
            "size_bytes": 2048,
        },
        {"url": "https://example.com/two.mp3", "filename": "two.mp3"},
    ]

    assert normalize_recording_files(
        {
            "source_url": "https://example.com/source.mp3",
            "source_url[1]": "https://example.com/source-1.mp3",
        }
    ) == [
        {"url": "https://example.com/source.mp3", "filename": "source.mp3"},
        {"url": "https://example.com/source-1.mp3", "filename": "source-1.mp3"},
    ]


def test_build_citation_url_offsets_across_multiple_recordings():
    files = [
        {"url": "https://example.com/part-1.mp3", "length_seconds": 60},
        {"url": "https://example.com/part-2.mp3", "length_seconds": 90},
    ]

    assert build_citation_url(files, 45) == "https://example.com/part-1.mp3#t=45"
    assert build_citation_url(files, 75) == "https://example.com/part-2.mp3#t=15"
    assert build_citation_url(files, 200) == "https://example.com/part-2.mp3#t=50"


def test_build_name_search_text_includes_full_normalized_and_part_tokens_once():
    assert build_name_search_text(["  Jane Q. Public  ", "Jane Public"]) == (
        "jane jane q public jane q. public public q jane public"
    )
