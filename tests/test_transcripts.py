import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from transcripts import build_transcript_chunks, load_transcript_document, load_vtt


def test_load_vtt_parses_notes_and_speakers(tmp_path):
    vtt_path = tmp_path / "episode.vtt"
    vtt_path.write_text(
        "\n".join(
            [
                "WEBVTT",
                "NOTE object_id: 22563",
                "NOTE source_url: https://example.com/audio.mp3",
                "",
                "00:00:01.000 --> 00:00:03.000",
                "Host: Welcome back.",
                "",
                "00:00:03.000 --> 00:00:06.000",
                "<v Guest>Thanks for having me.</v>",
                "",
            ]
        ),
        encoding="utf-8",
    )

    notes, cues = load_vtt(vtt_path)

    assert notes["object_id"] == "22563"
    assert notes["source_url"] == "https://example.com/audio.mp3"
    assert cues[0].speaker == "Host"
    assert cues[1].speaker == "Guest"
    assert cues[1].text == "Thanks for having me."


def test_build_transcript_chunks_merges_short_turns(tmp_path):
    vtt_path = tmp_path / "episode.vtt"
    vtt_path.write_text(
        "\n".join(
            [
                "WEBVTT",
                "NOTE object_id: 22563",
                "NOTE source_url: https://example.com/audio.mp3",
                "NOTE filename: episode.mp3",
                "",
                "00:00:01.000 --> 00:00:03.000",
                "Host: Hello there.",
                "",
                "00:00:03.000 --> 00:00:05.000",
                "Guest: General Kenobi.",
                "",
                "00:00:05.000 --> 00:00:40.000",
                "Guest: "
                + " ".join(["Longer discussion sentence."] * 40),
                "",
            ]
        ),
        encoding="utf-8",
    )
    metadata_path = tmp_path / "episode.json"
    metadata_path.write_text(
        """
        {
          "transcript_name": "episode.vtt",
          "recording_urls": [
            {
              "url": "https://example.com/audio.mp3",
              "length": 120.0,
              "size": "1024"
            }
          ],
          "program": "Radio Times",
          "guests": ["Guest Name"],
          "collective_access_metadata": {
            "object_id": "22563",
            "occurrence_id": "1299",
            "matched_by": "occurrence_id"
          }
        }
        """,
        encoding="utf-8",
    )

    document = load_transcript_document(vtt_path, metadata_path)
    chunks = build_transcript_chunks(document)

    assert chunks
    assert chunks[0]["content_type"] == "transcript"
    assert chunks[0]["occurrence_id"] == "1299"
    assert chunks[0]["title"] == "episode.mp3"
    assert chunks[0]["transcript_url"] is None
    assert chunks[0]["transcript_name"] == "episode.vtt"
    assert chunks[0]["citation_url"] == "https://example.com/audio.mp3#t=1"
    assert chunks[0]["recording_urls"] == ["https://example.com/audio.mp3"]
    assert chunks[0]["chunk_text"]
    assert "Host" in chunks[0]["speakers"]
    assert "Guest" in chunks[0]["speakers"]
    assert chunks[0]["program"] == "Radio Times"
    assert chunks[0]["speaker_search_text"]
    assert "Transcript excerpt:" in chunks[0]["search_text"]
    assert document["id"] == "22563"
    assert document["occurrence_id"] == "1299"
    assert document["transcript_name"] == "episode.vtt"
    assert document["recording_files"] == [
        {
            "url": "https://example.com/audio.mp3",
            "filename": "audio.mp3",
            "length_seconds": 120.0,
            "size_bytes": 1024,
        }
    ]
