class TipFormatter:
    @staticmethod
    def tip_metadata(metadata: dict):
        search_text = f'🔍 Searched "{metadata["question"]}"'

        start_date = metadata["date_range"]["start_date"]
        end_date = metadata["date_range"]["end_date"]

        date_text = ""
        if start_date and end_date:
            date_text = f"⏳ From {start_date} to {end_date}."
        elif start_date:
            date_text = f"⏳ After {start_date}."
        elif end_date:
            date_text = f"⏳ Until {end_date}"

        author_names = [author["name"] for author in metadata.get("authors", [])]
        speaker_names = [speaker["name"] for speaker in metadata.get("speakers", [])]
        content_types = metadata.get("content_types", [])

        author_text = ""
        if len(author_names) == 1:
            author_text = f"🖊️ Written by {author_names[0]}"
        elif len(author_names) > 1:
            author_text = (
                f"🖊️ Written by {', '.join(author_names[:-1])}"
                f"{',' if len(author_names) > 2 else ''} and {author_names[-1]}"
            )

        speaker_text = ""
        if len(speaker_names) == 1:
            speaker_text = f"🎙️ Speaker {speaker_names[0]}"
        elif len(speaker_names) > 1:
            speaker_text = (
                f"🎙️ Speakers {', '.join(speaker_names[:-1])}"
                f"{',' if len(speaker_names) > 2 else ''} and {speaker_names[-1]}"
            )

        scope_text = ""
        if len(content_types) == 1:
            scope_text = f"📚 Scope: {content_types[0]}"

        text_blocks = filter(
            lambda x: len(x) > 0,
            [search_text, date_text, author_text, speaker_text, scope_text],
        )
        return "\n".join(text_blocks)

    @staticmethod
    def tip_search(sources: list):
        return f"🔍 Retrieved {len(sources)} result(s)."
