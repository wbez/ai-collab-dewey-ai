#!/usr/bin/env python3
"""Minimal Slack SDK ChatStream POC.

Examples:
    SLACK_BOT_TOKEN=xoxb-... .venv/bin/python scripts/poc_chatstream.py \
        --channel D123 --seed-message "POC seed"

    SLACK_BOT_TOKEN=xoxb-... .venv/bin/python scripts/poc_chatstream.py \
        --channel D123 --thread-ts 1234567890.123456
"""

from __future__ import annotations

import argparse
import os
import time

from slack_sdk.web import WebClient


LOREM = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "Integer nec odio. Praesent libero. Sed cursus ante dapibus diam."
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream lorem ipsum with Slack SDK ChatStream.")
    parser.add_argument("--channel", required=True, help="Slack channel, DM, or MPIM id.")
    parser.add_argument("--thread-ts", "--thread_ts", dest="thread_ts", help="Existing Slack message ts.")
    parser.add_argument("--seed-message", help="Post this message first, then stream into its ts.")
    parser.add_argument("--user", help="Recipient Slack user id.")
    parser.add_argument("--team", help="Recipient Slack team id.")
    parser.add_argument("--token", default=os.environ.get("SLACK_BOT_TOKEN"))
    parser.add_argument("--delay", type=float, default=0.15)
    args = parser.parse_args()

    if not args.token:
        parser.error("Provide --token or SLACK_BOT_TOKEN.")
    if args.thread_ts == "0":
        parser.error("--thread-ts must be a real Slack message timestamp; 0 is not valid.")
    if not args.thread_ts and not args.seed_message:
        parser.error("ChatStream requires --thread-ts, or use --seed-message to create one.")

    client = WebClient(token=args.token)

    thread_ts = args.thread_ts
    if args.seed_message:
        response = client.chat_postMessage(channel=args.channel, text=args.seed_message)
        thread_ts = str(response["ts"])
        print(f"Seed message posted with ts={thread_ts}")

    streamer = client.chat_stream(
        channel=args.channel,
        thread_ts=thread_ts,
        recipient_user_id=args.user,
        recipient_team_id=args.team,
        buffer_size=1,
    )

    for word in LOREM.split():
        streamer.append(markdown_text=f"{word} ")
        if args.delay:
            time.sleep(args.delay)

    streamer.stop()
    print("ChatStream completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
