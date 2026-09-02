from textwrap import dedent
from typing import Any, Dict


def load_search_prompt(current_date: str, assistant_name: str = "Dewey") -> str:
    return dedent(
        f"""
        The assistant is {assistant_name}, created by Chicago Public Media.

        The current date is {current_date}.

        {assistant_name} helps journalists search a corpus that includes written articles, timestamped transcripts, and scripts.
        The corpus is searchable with a full-sentence search question and filterable by dates, content type,
        authors, speakers, guests, and program name.

        When a user asks a question, always generate:
        - a full-sentence search question based on the user's request and conversation history
        - filter metadata for dates and named people
        - optional content type scope when the user clearly asks for only articles, transcripts, or scripts

        Keep filter criteria out of the search question itself.
        If a user asks for certain time periods, include them in filters.
        If a user asks for articles written by specific people, include them as authors.
        If a user asks for transcript participants or quoted people in a recording, include them as speakers.
        If a user mentions named guests or a show/program, include them in those fields.
        If the user does not clearly scope the request, search articles, transcripts, and scripts.
        """
    ).strip()


def _named_entity_array(description: str) -> Dict[str, Any]:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": description},
            },
            "additionalProperties": False,
            "required": ["name"],
        },
    }


def load_search_tool() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "search_archive",
        "description": "Retrieves archive content using a search question and structured metadata.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The semantic search query text to find relevant archive material.",
                },
                "date_range": {
                    "type": "object",
                    "properties": {
                        "start_date": {
                            "type": ["string", "null"],
                            "description": "The start date of the user's query in ISO format (YYYY-MM-DD)",
                        },
                        "end_date": {
                            "type": ["string", "null"],
                            "description": "The end date of the user's query in ISO format (YYYY-MM-DD)",
                        },
                    },
                    "required": ["start_date", "end_date"],
                    "additionalProperties": False,
                },
                "authors": _named_entity_array("Full journalist name"),
                "speakers": _named_entity_array("Full speaker name"),
                "guests": _named_entity_array("Full guest name"),
                "content_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["article", "transcript", "script"],
                    },
                },
                "program": {
                    "type": ["string", "null"],
                    "description": "Show or program name when requested by the user.",
                },
            },
            "required": [
                "question",
                "date_range",
                "authors",
                "speakers",
                "guests",
                "content_types",
                "program",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    }
