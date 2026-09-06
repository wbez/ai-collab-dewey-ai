#!/usr/bin/env python3
"""POC Slack stream using the repo's authorized bot transport.

Dry run:
    PYTHONPATH=app python scripts/poc_stream_lorem_slack.py --dry-run

Real Slack stream:
    PYTHONPATH=app SLACK_BOT_TOKEN=xoxb-... \
    python scripts/poc_stream_lorem_slack.py --channel D123 --user U123 --team T123
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import mcp_server  # noqa: E402


LOREM = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "Integer nec odio. Praesent libero. Sed cursus ante dapibus diam. "
    "Sed nisi. Nulla quis sem at nibh elementum imperdiet. Duis sagittis ipsum. "
    "Praesent mauris. Fusce nec tellus sed augue semper porta."
)


class LoremStreamingService:
    def __init__(self, *, delay_seconds: float, chunk_words: int, include_sources: bool = False) -> None:
        self.delay_seconds = delay_seconds
        self.chunk_words = max(1, chunk_words)
        self.include_sources = include_sources

    def stream_archive(
        self,
        question: str,
        conversation_context: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        yield {
            "type": "task",
            "id": "poc",
            "title": "Starting lorem ipsum stream",
            "status": "complete",
        }

        words = LOREM.split()
        for index in range(0, len(words), self.chunk_words):
            yield {"type": "text_delta", "text": " ".join(words[index : index + self.chunk_words]) + " "}
            if self.delay_seconds:
                time.sleep(self.delay_seconds)

        answer = f"{LOREM} [1]" if self.include_sources else LOREM
        sources = [
            {
                "number": 1,
                "source_id": "poc-lorem-source",
                "title": "Lorem ipsum prototype source",
                "url": "https://www.wbez.org/?wavelength-poc=lorem",
                "publish_date": "2026-09-05",
                "content_type": "article",
                "excerpt": LOREM[:180],
                "full_text": LOREM,
                "authors": ["Wavelength Prototype"],
            }
        ] if self.include_sources else []
        yield {
            "type": "final",
            "payload": {
                "content": [{"type": "text", "text": answer}],
                "_meta": {"slack": {"blocks": []}},
                "structuredContent": {
                    "answer": answer,
                    "answer_markdown": answer,
                    "sources": sources,
                    "needs_clarification": False,
                },
            },
        }


def install_dry_run_slack_api() -> None:
    def fake_call_slack_api(**kwargs: Any) -> Dict[str, Any]:
        method = kwargs["method"]
        payload = kwargs["payload"]
        print(f"{method}: {payload}")
        if method == "chat.postMessage":
            return {"ok": True, "ts": "888.000"}
        if method == "chat.startStream":
            return {"ok": True, "ts": "999.000"}
        return {"ok": True}

    mcp_server._call_slack_api = fake_call_slack_api


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stream lorem ipsum through the existing Slack bot streaming path."
    )
    parser.add_argument("--channel", help="Slack channel/DM id, for example D123 or C123.")
    parser.add_argument("--user", help="Recipient Slack user id, for example U123.")
    parser.add_argument("--team", help="Recipient Slack team id, for example T123.")
    parser.add_argument("--app-id", help="Slack api_app_id for Work Object mention rendering in this direct prototype.")
    parser.add_argument("--thread-ts", "--thread_ts", dest="thread_ts", help="Optional Slack thread timestamp.")
    parser.add_argument(
        "--seed-message",
        default=None,
        help="Post a normal Slack message first and stream into its returned thread timestamp.",
    )
    parser.add_argument("--token", default=os.environ.get("SLACK_BOT_TOKEN"))
    parser.add_argument("--dry-run", action="store_true", help="Print Slack API calls instead of sending them.")
    parser.add_argument("--embeds", action="store_true", help="Attach a cited Work Object that declares embed support.")
    parser.add_argument(
        "--public-base-url",
        default=os.environ.get("WAVELENGTH_PUBLIC_BASE_URL"),
        help="Public HTTPS base URL used for signed Work Object embed preview URLs.",
    )
    parser.add_argument("--delay", type=float, default=0.15, help="Delay between text chunks.")
    parser.add_argument("--chunk-words", type=int, default=4, help="Words per streamed delta.")
    args = parser.parse_args()

    if args.dry_run:
        install_dry_run_slack_api()
        args.channel = args.channel or "DLOCAL"
        args.user = args.user or "ULOCAL"
        args.team = args.team or "TLOCAL"
        args.app_id = args.app_id or "ALOCAL"
        args.token = args.token or "xoxb-dry-run"
        args.public_base_url = args.public_base_url or "https://wavelength.example.test"

    if args.embeds:
        os.environ["WAVELENGTH_WORK_OBJECT_EMBEDS"] = "true"
        if args.public_base_url:
            os.environ["WAVELENGTH_PUBLIC_BASE_URL"] = args.public_base_url
        if args.dry_run:
            os.environ.setdefault("WAVELENGTH_EMBED_SIGNING_SECRET", "dry-run-embed-secret")
        if not (os.environ.get("WAVELENGTH_PUBLIC_BASE_URL") or "").startswith("https://"):
            parser.error("--embeds requires --public-base-url or WAVELENGTH_PUBLIC_BASE_URL with an HTTPS URL.")
        if not os.environ.get("WAVELENGTH_EMBED_SIGNING_SECRET"):
            parser.error("--embeds requires WAVELENGTH_EMBED_SIGNING_SECRET.")
        if not args.team or not args.app_id:
            parser.error("--embeds requires --team and --app-id so Work Object mentions can be rendered.")
        mcp_server._cache_team_api_app_id(args.team, args.app_id)

    if not args.token:
        parser.error("Provide --token or SLACK_BOT_TOKEN, or use --dry-run.")
    if not args.channel:
        parser.error("Provide --channel, or use --dry-run.")
    if args.channel.startswith("D") and not args.thread_ts and not args.user and not args.seed_message:
        parser.error(
            "Top-level Slack DM streams usually need --user. "
            "Alternatively provide --thread-ts to stream in an existing thread, "
            "or --seed-message to create one first."
        )
    if args.thread_ts == "0":
        parser.error("--thread-ts must be a real Slack message timestamp; 0 is treated as omitted.")

    thread_ts = args.thread_ts
    if args.seed_message:
        seed = mcp_server._call_slack_api(
            token=args.token,
            method="chat.postMessage",
            payload={
                "channel": args.channel,
                "text": args.seed_message,
            },
        )
        thread_ts = str(seed.get("ts") or "")
        if not thread_ts:
            raise RuntimeError(f"Slack chat.postMessage did not return ts: {seed}")
        print(f"Seed message posted with ts={thread_ts}")

    mcp_server._stream_answer_to_slack(
        service=LoremStreamingService(
            delay_seconds=args.delay,
            chunk_words=args.chunk_words,
            include_sources=args.embeds,
        ),
        bot_token=args.token,
        question="POC lorem ipsum stream",
        channel=args.channel,
        user_id=args.user,
        team_id=args.team,
        thread_ts=thread_ts,
    )
    print("Lorem ipsum stream request completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
