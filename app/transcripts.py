import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import tiktoken
except ImportError:  # pragma: no cover - local fallback for environments without deps
    tiktoken = None


MIN_CHUNK_TOKENS = 120
MAX_CHUNK_TOKENS = 512
TARGET_CHUNK_TOKENS = 500

NOTE_PATTERN = re.compile(r"^NOTE\s+([^:]+):\s*(.+?)\s*$")
TIMESTAMP_PATTERN = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2}\.\d{3})"
)
VOICE_TAG_PATTERN = re.compile(r"^<v\s+([^>]+)>(.*)$", re.IGNORECASE)
SPEAKER_PREFIX_PATTERN = re.compile(r"^([^:\n]{1,80}):\s*(.+)$", re.DOTALL)
WHITESPACE_PATTERN = re.compile(r"\s+")

KNOWN_TRANSCRIPT_FIELDS = {
    "title",
    "description",
    "program",
    "recording_date",
    "transcript_url",
    "recording_urls",
    "guests",
    "extra_metadata",
}


@dataclass
class TranscriptCue:
    start_seconds: float
    end_seconds: float
    speaker: Optional[str]
    text: str


def normalize_whitespace(value: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", value).strip()


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def parse_timestamp(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def format_timestamp(seconds: float) -> str:
    total_millis = int(round(seconds * 1000))
    hours, remainder = divmod(total_millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def count_tokens(value: str) -> int:
    if not value:
        return 0
    if tiktoken is not None:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(value))
    return len(re.findall(r"\w+|[^\w\s]", value))


def split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", normalize_whitespace(text))
    return [sentence for sentence in sentences if sentence]


def load_vtt(path: Path) -> tuple[Dict[str, str], List[TranscriptCue]]:
    notes: Dict[str, str] = {}
    cues: List[TranscriptCue] = []
    current_timing: Optional[re.Match[str]] = None
    text_lines: List[str] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip("\ufeff")
        note_match = NOTE_PATTERN.match(line.strip())
        timing_match = TIMESTAMP_PATTERN.match(line.strip())

        if note_match and current_timing is None and not text_lines:
            notes[note_match.group(1).strip()] = note_match.group(2).strip()
            continue

        if timing_match:
            if current_timing and text_lines:
                cues.append(_build_cue(current_timing, text_lines))
            current_timing = timing_match
            text_lines = []
            continue

        if current_timing is not None:
            if line.strip():
                text_lines.append(line.strip())
            elif text_lines:
                cues.append(_build_cue(current_timing, text_lines))
                current_timing = None
                text_lines = []

    if current_timing and text_lines:
        cues.append(_build_cue(current_timing, text_lines))

    return notes, cues


def _build_cue(timing_match: re.Match[str], text_lines: List[str]) -> TranscriptCue:
    raw_text = normalize_whitespace(" ".join(text_lines))
    speaker, text = extract_speaker(raw_text)
    return TranscriptCue(
        start_seconds=parse_timestamp(timing_match.group("start")),
        end_seconds=parse_timestamp(timing_match.group("end")),
        speaker=speaker,
        text=text,
    )


def extract_speaker(raw_text: str) -> tuple[Optional[str], str]:
    voice_match = VOICE_TAG_PATTERN.match(raw_text)
    if voice_match:
        speaker = normalize_whitespace(voice_match.group(1))
        text = normalize_whitespace(re.sub(r"</v>", "", voice_match.group(2)))
        return speaker, text

    prefix_match = SPEAKER_PREFIX_PATTERN.match(raw_text)
    if prefix_match:
        speaker = normalize_whitespace(prefix_match.group(1))
        text = normalize_whitespace(prefix_match.group(2))
        return speaker, text

    return None, normalize_whitespace(re.sub(r"</?v[^>]*>", "", raw_text))


def parse_note_file_entry(value: str) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for part in value.split(";"):
        item = part.strip()
        if not item or "=" not in item:
            continue
        key, raw = item.split("=", 1)
        raw = raw.strip().strip('"')
        if raw.replace(".", "", 1).isdigit():
            parsed[key.strip()] = float(raw) if "." in raw else int(raw)
        else:
            parsed[key.strip()] = raw
    return parsed


def build_name_search_text(names: Iterable[str]) -> str:
    tokens: List[str] = []
    for name in names:
        cleaned = normalize_whitespace(name)
        if not cleaned:
            continue
        normalized = normalize_name(cleaned)
        parts = [part for part in normalized.split() if part]
        variants = {cleaned.lower(), normalized, *parts}
        tokens.extend(sorted(filter(None, variants)))
    return " ".join(dict.fromkeys(tokens))


def build_transcript_search_text(
    headline: str,
    description: Optional[str],
    program: Optional[str],
    guests: List[str],
    speakers: List[str],
    content: str,
) -> str:
    blocks = [
        f"Title: {headline}" if headline else "",
        f"Description: {description}" if description else "",
        f"Program: {program}" if program else "",
        f"Guests: {'; '.join(guests)}" if guests else "",
        f"Speakers: {'; '.join(speakers)}" if speakers else "",
        f"Transcript excerpt: {content}" if content else "",
    ]
    return "\n".join(block for block in blocks if block)


def load_transcript_document(vtt_path: Path, metadata_path: Path) -> Dict[str, Any]:
    notes, cues = load_vtt(vtt_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    merged_metadata: Dict[str, Any] = {}
    merged_metadata.update(_metadata_from_notes(notes))
    merged_metadata.update(metadata)

    extra_metadata = {
        key: value
        for key, value in merged_metadata.items()
        if key not in KNOWN_TRANSCRIPT_FIELDS
    }
    if notes:
        extra_metadata.setdefault("vtt_notes", notes)

    recording_urls = merged_metadata.get("recording_urls") or []
    if isinstance(recording_urls, str):
        recording_urls = [recording_urls]
    if not recording_urls and merged_metadata.get("source_url"):
        recording_urls = [merged_metadata["source_url"]]

    guests = merged_metadata.get("guests") or []
    if isinstance(guests, str):
        guests = [guests]

    title = (
        merged_metadata.get("title")
        or merged_metadata.get("filename")
        or merged_metadata.get("object_id")
        or vtt_path.stem
    )

    recording_date = (
        merged_metadata.get("recording_date")
        or merged_metadata.get("date")
        or merged_metadata.get("date[1]")
    )

    transcript_url = merged_metadata.get("transcript_url")
    if not transcript_url:
        raise ValueError(f"Transcript sidecar {metadata_path.name} is missing transcript_url.")

    return {
        "id": str(
            merged_metadata.get("object_id")
            or merged_metadata.get("occurrence_id")
            or vtt_path.stem
        ),
        "title": title,
        "description": merged_metadata.get("description"),
        "program": merged_metadata.get("program"),
        "recording_date": recording_date,
        "transcript_url": transcript_url,
        "recording_urls": recording_urls,
        "guests": [normalize_whitespace(guest) for guest in guests if normalize_whitespace(guest)],
        "extra_metadata": extra_metadata,
        "cues": cues,
    }


def _metadata_from_notes(notes: Dict[str, str]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for key, value in notes.items():
        if key.startswith("file["):
            metadata.setdefault("recording_urls", [])
            parsed_file = parse_note_file_entry(value)
            if parsed_file.get("source_url"):
                metadata["recording_urls"].append(parsed_file["source_url"])
            metadata.setdefault("extra_metadata", {})
            metadata["extra_metadata"].setdefault("files", []).append(parsed_file)
            continue
        metadata[key] = value
    return metadata


def build_transcript_chunks(document: Dict[str, Any]) -> List[Dict[str, Any]]:
    turns = _merge_cues_into_turns(document["cues"])
    chunks = _merge_turns_into_chunks(turns)
    chunk_documents: List[Dict[str, Any]] = []

    for index, chunk in enumerate(chunks):
        speakers = [speaker for speaker in chunk["speakers"] if speaker]
        search_text = build_transcript_search_text(
            headline=document["title"],
            description=document.get("description"),
            program=document.get("program"),
            guests=document.get("guests", []),
            speakers=speakers,
            content=chunk["content"],
        )
        fingerprint = hashlib.md5(
            f"{document['id']}:{chunk['start_seconds']}:{chunk['end_seconds']}:{chunk['content']}".encode(
                "utf-8"
            )
        ).hexdigest()[:12]
        chunk_documents.append(
            {
                "chunk_id": f"{document['id']}-{index:04d}-{fingerprint}",
                "parent_id": document["id"],
                "content_type": "transcript",
                "headline": document["title"],
                "description": document.get("description"),
                "program": document.get("program"),
                "guests": document.get("guests", []),
                "speakers": speakers,
                "speaker_search_text": build_name_search_text(speakers),
                "author_search_text": "",
                "transcript_url": document["transcript_url"],
                "recording_urls": document.get("recording_urls", []),
                "timestamp_label": f"{format_timestamp(chunk['start_seconds'])} - {format_timestamp(chunk['end_seconds'])}",
                "start_seconds": chunk["start_seconds"],
                "end_seconds": chunk["end_seconds"],
                "publish_date": document.get("recording_date"),
                "recording_date": document.get("recording_date"),
                "content": chunk["content"],
                "search_text": search_text,
                "raw_metadata_json": json.dumps(document.get("extra_metadata", {}), ensure_ascii=False),
                "url": document["transcript_url"],
                "authors": [],
            }
        )

    return chunk_documents


def _merge_cues_into_turns(cues: List[TranscriptCue]) -> List[Dict[str, Any]]:
    turns: List[Dict[str, Any]] = []
    for cue in cues:
        if (
            turns
            and turns[-1]["speaker"] == cue.speaker
            and cue.speaker is not None
        ):
            turns[-1]["end_seconds"] = cue.end_seconds
            turns[-1]["texts"].append(cue.text)
            turns[-1]["cue_starts"].append(cue.start_seconds)
            continue
        turns.append(
            {
                "speaker": cue.speaker,
                "start_seconds": cue.start_seconds,
                "end_seconds": cue.end_seconds,
                "texts": [cue.text],
                "cue_starts": [cue.start_seconds],
            }
        )

    for turn in turns:
        turn["content"] = normalize_whitespace(" ".join(turn.pop("texts")))
        turn["token_count"] = count_tokens(turn["content"])
        turn["speakers"] = [turn["speaker"]] if turn["speaker"] else []
    return turns


def _merge_turns_into_chunks(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    index = 0
    while index < len(turns):
        current = _clone_turn(turns[index])
        index += 1

        while current["token_count"] < MIN_CHUNK_TOKENS and index < len(turns):
            current = _combine_units(current, turns[index])
            index += 1

        if current["token_count"] > MAX_CHUNK_TOKENS:
            chunks.extend(_split_oversized_unit(current))
            continue

        if current["token_count"] < TARGET_CHUNK_TOKENS and index < len(turns):
            lookahead = turns[index]
            combined = _combine_units(current, lookahead)
            if combined["token_count"] <= MAX_CHUNK_TOKENS:
                current = combined
                index += 1

        chunks.append(current)

    return chunks


def _clone_turn(turn: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "start_seconds": turn["start_seconds"],
        "end_seconds": turn["end_seconds"],
        "content": turn["content"],
        "speakers": list(turn["speakers"]),
        "cue_starts": list(turn["cue_starts"]),
        "token_count": turn["token_count"],
    }


def _combine_units(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    content = normalize_whitespace(f"{left['content']} {right['content']}")
    speakers = list(dict.fromkeys(left["speakers"] + right["speakers"]))
    return {
        "start_seconds": left["start_seconds"],
        "end_seconds": right["end_seconds"],
        "content": content,
        "speakers": speakers,
        "cue_starts": left["cue_starts"] + right["cue_starts"],
        "token_count": count_tokens(content),
    }


def _split_oversized_unit(unit: Dict[str, Any]) -> List[Dict[str, Any]]:
    sentences = split_sentences(unit["content"])
    if len(sentences) <= 1:
        return [_trim_unit(unit)]

    split_units: List[Dict[str, Any]] = []
    buffer: List[str] = []
    for sentence in sentences:
        candidate = normalize_whitespace(" ".join(buffer + [sentence]))
        if buffer and count_tokens(candidate) > MAX_CHUNK_TOKENS:
            split_units.append(
                {
                    **unit,
                    "content": normalize_whitespace(" ".join(buffer)),
                    "token_count": count_tokens(normalize_whitespace(" ".join(buffer))),
                }
            )
            buffer = [sentence]
        else:
            buffer.append(sentence)

    if buffer:
        split_units.append(
            {
                **unit,
                "content": normalize_whitespace(" ".join(buffer)),
                "token_count": count_tokens(normalize_whitespace(" ".join(buffer))),
            }
        )

    return [_trim_unit(split_unit) for split_unit in split_units]


def _trim_unit(unit: Dict[str, Any]) -> Dict[str, Any]:
    text = unit["content"]
    if count_tokens(text) <= MAX_CHUNK_TOKENS:
        return {**unit, "content": text, "token_count": count_tokens(text)}

    words = text.split()
    trimmed: List[str] = []
    for word in words:
        candidate = normalize_whitespace(" ".join(trimmed + [word]))
        if trimmed and count_tokens(candidate) > MAX_CHUNK_TOKENS:
            break
        trimmed.append(word)
    content = normalize_whitespace(" ".join(trimmed))
    return {**unit, "content": content, "token_count": count_tokens(content)}
