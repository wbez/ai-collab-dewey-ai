import os
import sys
import types
import unittest
from unittest.mock import patch


if "starlette" not in sys.modules:
    starlette = types.ModuleType("starlette")
    background = types.ModuleType("starlette.background")
    concurrency = types.ModuleType("starlette.concurrency")
    middleware = types.ModuleType("starlette.middleware")
    middleware_base = types.ModuleType("starlette.middleware.base")
    responses = types.ModuleType("starlette.responses")
    staticfiles = types.ModuleType("starlette.staticfiles")

    async def run_in_threadpool(func, *args, **kwargs):
        return func(*args, **kwargs)

    class BackgroundTask:
        def __init__(self, func, *args, **kwargs):
            self.func = func
            self.args = args
            self.kwargs = kwargs

    class BaseHTTPMiddleware:
        def __init__(self, app, *args, **kwargs):
            self.app = app

    class PlainTextResponse:
        def __init__(self, content="", status_code=200, **kwargs):
            self.content = content
            self.status_code = status_code

    class JSONResponse(PlainTextResponse):
        pass

    class StaticFiles:
        def __init__(self, *args, **kwargs):
            pass

    background.BackgroundTask = BackgroundTask
    concurrency.run_in_threadpool = run_in_threadpool
    middleware_base.BaseHTTPMiddleware = BaseHTTPMiddleware
    responses.JSONResponse = JSONResponse
    responses.PlainTextResponse = PlainTextResponse
    staticfiles.StaticFiles = StaticFiles
    sys.modules["starlette"] = starlette
    sys.modules["starlette.background"] = background
    sys.modules["starlette.concurrency"] = concurrency
    sys.modules["starlette.middleware"] = middleware
    sys.modules["starlette.middleware.base"] = middleware_base
    sys.modules["starlette.responses"] = responses
    sys.modules["starlette.staticfiles"] = staticfiles

import mcp_server


class FakeStreamingService:
    def stream_archive(self, **kwargs):
        yield {
            "type": "final",
            "payload": {
                "content": [{"type": "text", "text": "Archive answer."}],
                "structuredContent": {"sources": []},
            },
        }


class TestWavelengthSlackAccess(unittest.TestCase):
    def test_slack_allowlist_is_fail_closed(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(mcp_server._is_wavelength_allowed_slack_user("U_ALLOWED"))

    def test_slack_allowlist_accepts_configured_user(self) -> None:
        with patch.dict(
            os.environ,
            {"WAVELENGTH_ALLOWED_SLACK_USER_IDS": "U_ALLOWED, U_OTHER"},
            clear=True,
        ):
            self.assertTrue(mcp_server._is_wavelength_allowed_slack_user("U_ALLOWED"))
            self.assertFalse(mcp_server._is_wavelength_allowed_slack_user("U_DENIED"))

    def test_home_user_ids_are_intersection_of_home_and_allowed_lists(self) -> None:
        with patch.dict(
            os.environ,
            {
                "WAVELENGTH_ALLOWED_SLACK_USER_IDS": "U_ALLOWED,U_OTHER",
                "WAVELENGTH_HOME_USER_IDS": "U_ALLOWED,U_DENIED",
            },
            clear=True,
        ):
            self.assertEqual(
                mcp_server._slack_home_user_ids("token"),
                ["U_ALLOWED"],
            )

    def test_home_user_listing_is_limited_to_allowed_users(self) -> None:
        with patch.dict(
            os.environ,
            {"WAVELENGTH_ALLOWED_SLACK_USER_IDS": "U_ALLOWED,U_OTHER"},
            clear=True,
        ):
            with patch.object(
                mcp_server,
                "_call_slack_api",
                return_value={
                    "members": [
                        {"id": "U_ALLOWED"},
                        {"id": "U_DENIED"},
                        {"id": "U_OTHER", "deleted": True},
                        {"id": "U_BOT", "is_bot": True},
                    ]
                },
            ):
                self.assertEqual(mcp_server._slack_home_user_ids("token"), ["U_ALLOWED"])

    def test_event_handler_posts_alpha_notice_before_answer_when_requested(self) -> None:
        calls = []

        def post_message(**kwargs):
            calls.append(("notice", kwargs["text"]))

        def post_answer(**kwargs):
            calls.append(("answer", kwargs["text"]))

        with patch.object(mcp_server, "_post_slack_message", side_effect=post_message):
            with patch.object(
                mcp_server,
                "_post_slack_message_with_work_objects",
                side_effect=post_answer,
            ):
                mcp_server._handle_event_question(
                    service=FakeStreamingService(),
                    bot_token="token",
                    question="question",
                    channel="C123",
                    user_id="U_ALLOWED",
                    team_id=None,
                    thread_ts="123.456",
                    event_ts="123.456",
                    conversation_context=[],
                    include_alpha_notice=True,
                )

        self.assertEqual(
            calls,
            [
                ("notice", mcp_server.WAVELENGTH_ALPHA_NOTICE),
                ("answer", "Archive answer."),
            ],
        )

    def test_event_handler_omits_alpha_notice_when_not_requested(self) -> None:
        calls = []

        def post_message(**kwargs):
            calls.append(("notice", kwargs["text"]))

        def post_answer(**kwargs):
            calls.append(("answer", kwargs["text"]))

        with patch.object(mcp_server, "_post_slack_message", side_effect=post_message):
            with patch.object(
                mcp_server,
                "_post_slack_message_with_work_objects",
                side_effect=post_answer,
            ):
                mcp_server._handle_event_question(
                    service=FakeStreamingService(),
                    bot_token="token",
                    question="question",
                    channel="C123",
                    user_id="U_ALLOWED",
                    team_id=None,
                    thread_ts="123.456",
                    event_ts="123.456",
                    conversation_context=[],
                    include_alpha_notice=False,
                )

        self.assertEqual(calls, [("answer", "Archive answer.")])

    def test_sources_attach_to_first_message_that_mentions_them(self) -> None:
        calls = []
        posted_messages = {}

        def call_slack_api(**kwargs):
            method = kwargs["method"]
            payload = kwargs["payload"]
            calls.append((method, payload))
            if method == "conversations.replies":
                return {
                    "ok": True,
                    "messages": [
                        posted_messages[ts]
                        for ts in sorted(posted_messages)
                    ],
                }
            ts = payload.get("ts") or str(sum(1 for call_method, _ in calls if call_method == "chat.postMessage"))
            if method == "chat.postMessage":
                posted_messages[ts] = {
                    "ts": ts,
                    "blocks": payload.get("blocks") or [],
                    "metadata": payload.get("metadata"),
                    "attachments": ([{"id": "attachment"}] if payload.get("metadata") else []),
                }
            elif method == "chat.update" and ts in posted_messages:
                posted_messages[ts] = {
                    **posted_messages[ts],
                    "blocks": payload.get("blocks") or [],
                }
            return {
                "ok": True,
                "message": {
                    "ts": ts,
                    "blocks": payload.get("blocks") or [],
                    "metadata": payload.get("metadata"),
                },
            }

        mcp_server._cache_team_api_app_id("T_SOURCES_TEST", "A123")
        payload = {
            "content": [{"type": "text", "text": "First [1]\nSecond [2] [3]\nAgain [1]\nFinal [4]"}],
            "structuredContent": {
                "answer_markdown": "First [1]\n\nSecond [2] [3]\n\nAgain [1]\n\nFinal [4]",
                "sources": [
                    {
                        "source_id": "source_1",
                        "url": "https://example.com/story",
                        "title": "Story",
                        "number": 1,
                    },
                    {
                        "source_id": "source_2",
                        "url": "https://example.com/second",
                        "title": "Second story",
                        "number": 2,
                    },
                    {
                        "source_id": "source_3",
                        "url": "https://example.com/third",
                        "title": "Third story",
                        "number": 3,
                    },
                    {
                        "source_id": "source_4",
                        "url": "https://example.com/fourth",
                        "title": "Fourth story",
                        "number": 4,
                    },
                ]
            },
        }

        with patch.object(mcp_server, "_call_slack_api", side_effect=call_slack_api):
            mcp_server._post_slack_message_with_work_objects(
                token="token",
                channel="C123",
                text="First [1]\nSecond [2] [3]\nAgain [1]\nFinal [4]",
                blocks=[],
                thread_ts="123.456",
                payload=payload,
                team_id="T_SOURCES_TEST",
            )

        post_calls = [payload for method, payload in calls if method == "chat.postMessage"]
        reply_calls = [payload for method, payload in calls if method == "conversations.replies"]
        update_calls = [payload for method, payload in calls if method == "chat.update"]
        self.assertEqual(len(post_calls), 3)
        self.assertEqual(len(reply_calls), 3)
        self.assertEqual(len(update_calls), 3)

        self.assertEqual(post_calls[0]["text"], "First [1]")
        self.assertEqual(
            [entity["external_ref"]["id"] for entity in post_calls[0]["metadata"]["entities"]],
            ["source_1"],
        )
        self.assertFalse(
            any(
                element.get("type") == "attachment_mention"
                for element in post_calls[0]["blocks"][0]["elements"][0]["elements"]
            )
        )
        self.assertEqual(post_calls[1]["text"], "Second [2] [3]")
        self.assertEqual(
            [entity["external_ref"]["id"] for entity in post_calls[1]["metadata"]["entities"]],
            ["source_2", "source_3"],
        )
        self.assertEqual(post_calls[2]["text"], "Again [1]\nFinal [4]")
        self.assertEqual(
            [entity["external_ref"]["id"] for entity in post_calls[2]["metadata"]["entities"]],
            ["source_4"],
        )

        third_message_elements = update_calls[2]["blocks"][0]["elements"][0]["elements"]
        repeated_source_mention = next(
            element
            for element in third_message_elements
            if element.get("type") == "attachment_mention" and element.get("entity_id") == "source_1"
        )
        self.assertEqual(repeated_source_mention["app_id"], "A123")
        self.assertEqual(repeated_source_mention["text"], "[1]")
        self.assertNotIn("channel", repeated_source_mention)
        self.assertEqual(repeated_source_mention["channel_id"], "C123")
        self.assertEqual(repeated_source_mention["ts"], "1")

        first_message_elements = update_calls[0]["blocks"][0]["elements"][0]["elements"]
        first_source_mention = next(
            element
            for element in first_message_elements
            if element.get("type") == "attachment_mention" and element.get("entity_id") == "source_1"
        )
        self.assertEqual(first_source_mention["text"], "[1]")
        self.assertEqual(first_source_mention["channel_id"], "C123")
        self.assertEqual(first_source_mention["ts"], "1")

    def test_source_continuation_blocks_do_not_repeat_header(self) -> None:
        blocks = mcp_server._work_object_registration_blocks(
            [
                {
                    "external_ref": {"id": "src_1"},
                    "url": "https://example.com/story",
                    "entity_payload": {
                        "attributes": {
                            "title": {"text": "Story"},
                            "display_id": "1",
                        }
                    },
                }
            ],
            work_object_app_id="A123",
            include_header=False,
        )

        self.assertIsNotNone(blocks)
        source_elements = blocks[0]["elements"][0]["elements"]
        self.assertFalse(
            any(
                element.get("text") == "Sources:"
                for element in source_elements
                if isinstance(element, dict)
            )
        )
        self.assertEqual(
            [element.get("type") for element in source_elements],
            ["text", "attachment_mention"],
        )


if __name__ == "__main__":
    unittest.main()
