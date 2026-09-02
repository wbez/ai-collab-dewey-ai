import asyncio
import hmac
import json
import logging
import sys
import time
import types
from hashlib import sha256
from pathlib import Path

import pytest
from starlette.applications import Starlette


APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import mcp_server
from mcp_server import MCPAuthMiddleware, _extract_bearer_token, _split_env_list


@pytest.fixture(autouse=True)
def clear_slack_thread_history():
    mcp_server._SLACK_THREAD_HISTORY.clear()


async def fake_app(scope, receive, send):
    await send({"type": "http.response.start", "status": 204, "headers": []})
    await send({"type": "http.response.body", "body": b""})


async def call_middleware(middleware, headers=None, body=b"{}"):
    messages = []
    consumed = False
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [
            (name.lower().encode("latin1"), value.encode("latin1"))
            for name, value in (headers or {}).items()
        ],
    }

    async def receive():
        nonlocal consumed
        if consumed:
            return {"type": "http.request", "body": b"", "more_body": False}
        consumed = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        messages.append(message)

    await middleware(scope, receive, send)
    return messages


def status_from_messages(messages):
    return next(message["status"] for message in messages if message["type"] == "http.response.start")


def signed_slack_headers(secret, body):
    timestamp = str(int(time.time()))
    base_string = b"v0:" + timestamp.encode("utf-8") + b":" + body
    signature = "v0=" + hmac.new(secret.encode("utf-8"), base_string, sha256).hexdigest()
    return {
        "x-slack-request-timestamp": timestamp,
        "x-slack-signature": signature,
    }


class FakeService:
    def ask_archive(self, question, conversation_context=None, **kwargs):
        return {
            "content": [{"type": "text", "text": f"answer: {question}"}],
            "_meta": {
                "slack": {
                    "blocks": [
                        {
                            "type": "section",
                            "text": {"type": "mrkdwn", "text": f"answer: {question}"},
                        }
                    ]
                }
            },
            "structuredContent": {"answer": f"answer: {question}"},
        }


class FakeStreamingService(FakeService):
    def stream_archive(self, question, conversation_context=None, **kwargs):
        yield {
            "type": "task",
            "id": "metadata",
            "title": "Generating search metadata",
            "status": "in_progress",
        }
        yield {"type": "text_delta", "text": "answer: "}
        yield {"type": "text_delta", "text": question}
        yield {
            "type": "final",
            "payload": {
                "content": [{"type": "text", "text": f"answer: {question}"}],
                "_meta": {
                    "slack": {
                        "blocks": [
                            {
                                "type": "header",
                                "text": {"type": "plain_text", "text": "Wavelength archive answer"},
                            },
                            {
                                "type": "section",
                                "text": {"type": "mrkdwn", "text": f"answer: {question}"},
                            },
                            {"type": "divider"},
                            {
                                "type": "section",
                                "text": {"type": "mrkdwn", "text": "*Sources*\n[1] Story"},
                            },
                        ]
                    }
                },
                "structuredContent": {"answer": f"answer: {question}"},
            },
        }


class SourceStreamingService(FakeStreamingService):
    def stream_archive(self, question, conversation_context=None, **kwargs):
        yield {
            "type": "final",
            "payload": {
                "content": [{"type": "text", "text": f"answer: {question}"}],
                "_meta": {"slack": {"blocks": []}},
                "structuredContent": {
                    "answer": f"answer: {question}",
                    "sources": [
                        {
                            "number": 1,
                            "title": "First source",
                            "content_type": "article",
                            "publish_date": "2026-01-01",
                            "excerpt": "First excerpt",
                        },
                        {
                            "number": 2,
                            "title": "Second source",
                            "content_type": "transcript",
                            "publish_date": "2026-01-02",
                            "excerpt": "Second excerpt",
                        },
                    ],
                },
            },
        }


class CapturingStreamingService(FakeStreamingService):
    def __init__(self):
        self.calls = []

    def stream_archive(self, question, conversation_context=None, **kwargs):
        self.calls.append(
            {
                "question": question,
                "conversation_context": list(conversation_context or []),
            }
        )
        yield from super().stream_archive(
            question,
            conversation_context=conversation_context,
            **kwargs,
        )


class FakeRequest:
    def __init__(self, body=b"", headers=None):
        self._body = body
        self.headers = headers or {}

    async def body(self):
        return self._body


def route_endpoint(app, path):
    return next(route.endpoint for route in app.routes if route.path == path)


def test_split_env_list_trims_empty_values():
    assert _split_env_list("T1, T2, ,") == {"T1", "T2"}


def test_extract_bearer_token_supports_authorization_header():
    assert _extract_bearer_token({"authorization": "Bearer secret"}) == "secret"


def test_call_slack_api_uses_get_params_for_history_methods(monkeypatch):
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"ok": True, "messages": []}

    def fake_get(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return FakeResponse()

    def fake_post(url, **kwargs):
        raise AssertionError("conversations.replies should not be sent as POST JSON")

    monkeypatch.setenv("SLACK_API_URL", "https://slack.test/api")
    monkeypatch.setattr(mcp_server.requests, "get", fake_get)
    monkeypatch.setattr(mcp_server.requests, "post", fake_post)

    result = mcp_server._call_slack_api(
        token="xoxb-test",
        method="conversations.replies",
        payload={
            "channel": "D1",
            "ts": "123.456",
            "limit": 200,
            "cursor": None,
        },
    )

    assert result == {"ok": True, "messages": []}
    assert calls == [
        {
            "url": "https://slack.test/api/conversations.replies",
            "headers": {"Authorization": "Bearer xoxb-test"},
            "params": {"channel": "D1", "ts": "123.456", "limit": 200},
            "timeout": 30,
        }
    ]
    calls.clear()

    result = mcp_server._call_slack_api(
        token="xoxb-test",
        method="conversations.history",
        payload={
            "channel": "D1",
            "latest": "123.456",
            "limit": 200,
            "cursor": None,
        },
    )

    assert result == {"ok": True, "messages": []}
    assert calls == [
        {
            "url": "https://slack.test/api/conversations.history",
            "headers": {"Authorization": "Bearer xoxb-test"},
            "params": {"channel": "D1", "latest": "123.456", "limit": 200},
            "timeout": 30,
        }
    ]


def test_archive_coverage_by_year_counts_articles_and_transcripts(tmp_path):
    (tmp_path / "2025.json").write_text(
        '[\n{"id": "a1", "headline": "One"},\n{"id": "a2", "headline": "Two"}\n]',
        encoding="utf-8",
    )
    (tmp_path / "2026.json").write_text(
        '[\n{"id": "a3", "headline": "Three"}\n]',
        encoding="utf-8",
    )
    (tmp_path / "sidecar-one.json").write_text(
        '{"recording_date": "2025-03-31T00:00:00Z", "transcript_name": "sidecar-one.vtt"}',
        encoding="utf-8",
    )
    (tmp_path / "sidecar-two.json").write_text(
        '{"transcript_name": "sidecar-two.vtt"}',
        encoding="utf-8",
    )
    (tmp_path / "sidecar-two.vtt").write_text(
        "WEBVTT\n\nNOTE date: 2026-01-02\n\n00:00:00.000 --> 00:00:01.000\nText",
        encoding="utf-8",
    )
    (tmp_path / "future-sidecar.json").write_text(
        '{"recording_date": "2051-01-01T00:00:00Z"}',
        encoding="utf-8",
    )

    coverage = mcp_server._archive_coverage_by_year(tmp_path)

    assert coverage == {
        "2025": {"articles": 2, "transcripts": 1, "total": 3},
        "2026": {"articles": 1, "transcripts": 1, "total": 2},
    }
    view = mcp_server._wavelength_home_view(tmp_path)
    assert view["type"] == "home"
    assert any(
        block["type"] == "header"
        and block["text"]["text"] == "Wavelength archive assistant"
        for block in view["blocks"]
    )


def test_auth_middleware_accepts_valid_bearer_token():
    middleware = MCPAuthMiddleware(
        fake_app,
        signing_secret=None,
        auth_token="secret",
    )

    messages = asyncio.run(
        call_middleware(middleware, {"authorization": "Bearer secret"})
    )

    assert status_from_messages(messages) == 204


def test_auth_middleware_rejects_bad_bearer_token():
    middleware = MCPAuthMiddleware(
        fake_app,
        signing_secret=None,
        auth_token="secret",
    )

    messages = asyncio.run(
        call_middleware(middleware, {"authorization": "Bearer wrong"})
    )

    assert status_from_messages(messages) == 401


def test_auth_middleware_rejects_slack_signature_for_mcp():
    body = b'{"jsonrpc":"2.0"}'
    middleware = MCPAuthMiddleware(
        fake_app,
        signing_secret="slack-secret",
        auth_token=None,
    )

    messages = asyncio.run(
        call_middleware(middleware, signed_slack_headers("slack-secret", body), body)
    )

    assert status_from_messages(messages) == 401


def test_auth_middleware_rejects_unallowed_slack_team():
    middleware = MCPAuthMiddleware(
        fake_app,
        signing_secret=None,
        auth_token="secret",
        allowed_team_ids={"T_ALLOWED"},
    )

    messages = asyncio.run(
        call_middleware(
            middleware,
            {
                "authorization": "Bearer secret",
                "x-slack-team-id": "T_BLOCKED",
            },
        )
    )

    assert status_from_messages(messages) == 403


def test_auth_middleware_rejects_when_no_auth_is_configured():
    middleware = MCPAuthMiddleware(
        fake_app,
        signing_secret=None,
        auth_token=None,
    )

    messages = asyncio.run(call_middleware(middleware))

    assert status_from_messages(messages) == 500


def test_create_mcp_server_registers_tools_and_forwards_to_service(monkeypatch):
    class FakeMCPServer:
        def __init__(self, name):
            self.name = name
            self.tools = {}

        def tool(self, name, **kwargs):
            def register(func):
                self.tools[name] = {"func": func, "kwargs": kwargs}
                return func

            return register

    fake_mcp_module = types.ModuleType("mcp.server.mcpserver")
    fake_mcp_module.MCPServer = FakeMCPServer
    monkeypatch.setitem(sys.modules, "mcp.server.mcpserver", fake_mcp_module)

    class CapturingMCPService:
        def __init__(self):
            self.calls = []

        def search_archive_data(self, **kwargs):
            self.calls.append(("search_archive_data", kwargs))
            return {"query": kwargs["query"], "results": []}

        def get_full_article_data(self, *args):
            self.calls.append(("get_full_article_data", args))
            return {"content": "article"}

        def get_full_transcript_data(self, *args):
            self.calls.append(("get_full_transcript_data", args))
            return {"cues": []}

        def get_full_script_data(self, *args):
            self.calls.append(("get_full_script_data", args))
            return {"content": "script"}

    service = CapturingMCPService()

    server = mcp_server.create_mcp_server(service)

    assert server.name == "Wavelength Archive"
    assert set(server.tools) == {
        "search_archive",
        "keyword_search_archive",
        "sample_archive",
        "get_full_article",
        "get_full_transcript",
        "get_full_script",
    }
    assert server.tools["search_archive"]["func"](
        query="parks",
        start_date="2026-01-01",
        end_date="2026-01-31",
        content_types=["article"],
        authors=["Reporter"],
        speakers=None,
        guests=None,
        program=None,
        sort="newest",
        search_top=10,
        limit=3,
    ) == {"query": "parks", "results": []}
    assert server.tools["get_full_article"]["func"]("article-1", "10", 500) == {"content": "article"}
    assert server.tools["get_full_transcript"]["func"](
        "transcript-1", "20", 600, 30.0, 90.0
    ) == {"cues": []}
    assert server.tools["get_full_script"]["func"]("script-1", None, 700) == {"content": "script"}
    assert service.calls == [
        (
            "search_archive_data",
            {
                "query": "parks",
                "start_date": "2026-01-01",
                "end_date": "2026-01-31",
                "content_types": ["article"],
                "authors": ["Reporter"],
                "speakers": None,
                "guests": None,
                "program": None,
                "limit": 3,
                "sort": "newest",
                "search_top": 10,
            },
        ),
        ("get_full_article_data", ("article-1", "10", 500)),
        ("get_full_transcript_data", ("transcript-1", "20", 600, 30.0, 90.0)),
        ("get_full_script_data", ("script-1", None, 700)),
    ]


def test_slack_events_url_verification(monkeypatch):
    monkeypatch.setenv("WAVELENGTH_SKIP_SLACK_REQUEST_AUTH", "true")
    monkeypatch.delenv("SLACK_SIGNING_SECRET", raising=False)
    app = Starlette()
    mcp_server._add_slack_routes(app, FakeService())

    response = asyncio.run(
        route_endpoint(app, "/slack/events")(
            FakeRequest(
                body=b'{"type":"url_verification","challenge":"challenge-token"}'
            )
        )
    )

    assert response.status_code == 200
    assert response.body == b"challenge-token"


def test_slack_app_home_opened_publishes_home_tab(monkeypatch):
    published = []

    def fake_publish_home_tab(**kwargs):
        published.append(kwargs)

    monkeypatch.setenv("WAVELENGTH_SKIP_SLACK_REQUEST_AUTH", "true")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setattr(mcp_server, "_publish_wavelength_home_tab", fake_publish_home_tab)
    app = Starlette()
    mcp_server._add_slack_routes(app, FakeService())

    response = asyncio.run(
        route_endpoint(app, "/slack/events")(
            FakeRequest(
                body=(
                    b'{"type":"event_callback","event":{"type":"app_home_opened",'
                    b'"user":"U1","tab":"home"}}'
                )
            )
        )
    )

    assert response.status_code == 200
    assert response.background is not None
    response.background.func(**response.background.kwargs)
    assert published == [{"token": "xoxb-test", "user_id": "U1", "team_id": "unknown"}]


def test_home_tabs_publish_on_startup_for_human_users(monkeypatch):
    calls = []

    def fake_call_slack_api(**kwargs):
        calls.append(kwargs)
        if kwargs["method"] == "users.list":
            return {
                "ok": True,
                "members": [
                    {"id": "U1", "deleted": False, "is_bot": False},
                    {"id": "U2", "deleted": True, "is_bot": False},
                    {"id": "B1", "deleted": False, "is_bot": True},
                    {"id": "U3", "deleted": False, "is_bot": False},
                ],
            }
        return {"ok": True}

    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.delenv("WAVELENGTH_HOME_USER_IDS", raising=False)
    monkeypatch.setattr(mcp_server, "_call_slack_api", fake_call_slack_api)
    monkeypatch.setattr(mcp_server, "_default_wavelength_home_view", lambda: {"type": "home", "blocks": []})

    mcp_server._publish_wavelength_home_tabs_on_startup()

    assert [call["method"] for call in calls] == ["users.list", "views.publish", "views.publish"]
    assert [call["payload"]["user_id"] for call in calls[1:]] == ["U1", "U3"]


def test_slack_app_mention_posts_threaded_answer(monkeypatch):
    posted = []

    def fake_post_message(**kwargs):
        posted.append(kwargs)

    monkeypatch.setenv("WAVELENGTH_SKIP_SLACK_REQUEST_AUTH", "true")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setattr(mcp_server, "_post_slack_message", fake_post_message)
    app = Starlette()
    mcp_server._add_slack_routes(app, FakeService())

    response = asyncio.run(
        route_endpoint(app, "/slack/events")(
            FakeRequest(
                body=(
                    b'{"type":"event_callback","event":{"type":"app_mention",'
                    b'"user":"U1","channel":"C1","text":"<@BOT> Find CTA funding stories",'
                    b'"ts":"123.456"}}'
                )
            )
        )
    )

    assert response.status_code == 200
    assert response.background is not None

    mcp_server._handle_event_question(
        service=FakeService(),
        bot_token="xoxb-test",
        question="Find CTA funding stories",
        channel="C1",
        user_id="U1",
        team_id=None,
        thread_ts="123.456",
    )

    assert posted[0]["token"] == "xoxb-test"
    assert posted[0]["channel"] == "C1"
    assert posted[0]["thread_ts"] == "123.456"
    assert posted[0]["text"] == "answer: Find CTA funding stories"
    assert posted[0]["blocks"][0]["type"] == "section"


def test_slack_dm_posts_thinking_and_final_answer_in_thread(monkeypatch):
    calls = []

    def fake_call_slack_api(**kwargs):
        calls.append(kwargs)
        if kwargs["method"] == "chat.startStream":
            return {"ok": True, "ts": "999.000"}
        return {"ok": True}

    monkeypatch.setenv("WAVELENGTH_SKIP_SLACK_REQUEST_AUTH", "true")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setattr(mcp_server, "_call_slack_api", fake_call_slack_api)
    app = Starlette()
    mcp_server._add_slack_routes(app, FakeStreamingService())

    response = asyncio.run(
        route_endpoint(app, "/slack/events")(
            FakeRequest(
                body=(
                    b'{"type":"event_callback","event":{"type":"message",'
                    b'"channel_type":"im","user":"U1","channel":"D1",'
                    b'"text":"Find tax stories","ts":"123.456"}}'
                )
            )
        )
    )

    assert response.status_code == 200
    assert response.background is not None
    assert response.background.kwargs["thread_ts"] == "123.456"
    assert response.background.kwargs["persist_thread_ts"] == mcp_server.SLACK_DM_WINDOW_CONTEXT_TS
    assert response.background.kwargs["use_dm_window_context"] is True

    mcp_server._handle_event_question(
        service=FakeStreamingService(),
        bot_token="xoxb-test",
        question="Find tax stories",
        channel="D1",
        user_id="U1",
        team_id=None,
        thread_ts="123.456",
        event_ts="123.456",
        persist_thread_ts=mcp_server.SLACK_DM_WINDOW_CONTEXT_TS,
        use_dm_window_context=True,
    )

    history_call = next(call for call in calls if call["method"] == "conversations.history")
    assert history_call["payload"]["channel"] == "D1"
    assert history_call["payload"]["latest"] == "123.456"
    stream_calls = [call for call in calls if call["method"].startswith("chat.")]
    assert [call["method"] for call in stream_calls] == [
        "chat.startStream", "chat.appendStream", "chat.stopStream", "chat.postMessage",
    ]
    answer_start_payload = stream_calls[0]["payload"]
    assert answer_start_payload["channel"] == "D1"
    assert answer_start_payload["thread_ts"] == "123.456"
    assert answer_start_payload["task_display_mode"] == "plan"
    assert answer_start_payload["chunks"] is None
    assert stream_calls[-1]["payload"]["text"] == "answer: Find tax stories"


def test_slack_dm_posts_final_answer_when_top_level_stream_is_rejected(monkeypatch):
    calls = []
    posted = []

    def fake_call_slack_api(**kwargs):
        calls.append(kwargs)
        if kwargs["method"] == "chat.startStream" and kwargs["payload"].get("thread_ts") is None:
            raise RuntimeError("Slack chat.startStream failed: {'ok': False, 'error': 'invalid_thread_ts'}")
        if kwargs["method"] == "chat.startStream":
            return {"ok": True, "ts": "999.000"}
        return {"ok": True}

    def fake_post_message(**kwargs):
        posted.append(kwargs)

    monkeypatch.setattr(mcp_server, "_call_slack_api", fake_call_slack_api)
    monkeypatch.setattr(mcp_server, "_post_slack_message", fake_post_message)

    mcp_server._handle_event_question(
        service=FakeStreamingService(),
        bot_token="xoxb-test",
        question="Find tax stories",
        channel="D1",
        user_id="U1",
        team_id=None,
        thread_ts=None,
        event_ts="123.456",
        persist_thread_ts=mcp_server.SLACK_DM_WINDOW_CONTEXT_TS,
        use_dm_window_context=True,
    )

    assert any(call["method"] == "conversations.history" for call in calls)
    assert [call["method"] for call in calls if call["method"] == "chat.startStream"]
    assert posted[-1] == {
            "token": "xoxb-test",
            "channel": "D1",
            "text": "answer: Find tax stories",
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": "Wavelength archive answer"},
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "answer: Find tax stories"},
                },
                {"type": "divider"},
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "*Sources*\n[1] Story"},
                },
            ],
            "thread_ts": None,
            "metadata": None,
    }


def test_slack_event_streams_answer_when_service_supports_streaming(monkeypatch):
    calls = []

    def fake_call_slack_api(**kwargs):
        calls.append(kwargs)
        if kwargs["method"] == "chat.startStream":
            return {"ok": True, "ts": "999.000"}
        return {"ok": True}

    monkeypatch.setattr(mcp_server, "_call_slack_api", fake_call_slack_api)

    mcp_server._handle_event_question(
        service=FakeStreamingService(),
        bot_token="xoxb-test",
        question="Find education coverage",
        channel="D1",
        user_id="U1",
        team_id="T1",
        thread_ts="123.456",
    )

    stream_calls = [call for call in calls if call["method"].startswith("chat.")]
    methods = [call["method"] for call in stream_calls]
    assert methods == ["chat.startStream", "chat.appendStream", "chat.stopStream", "chat.postMessage"]
    answer_start_payload = stream_calls[0]["payload"]
    assert answer_start_payload["channel"] == "D1"
    assert answer_start_payload["thread_ts"] == "123.456"
    assert answer_start_payload["recipient_user_id"] == "U1"
    assert answer_start_payload["recipient_team_id"] == "T1"
    assert answer_start_payload["chunks"] is None
    assert stream_calls[-1]["payload"]["text"] == "answer: Find education coverage"


def test_slack_thread_history_is_passed_to_stream_archive(monkeypatch):
    calls = []

    def fake_call_slack_api(**kwargs):
        calls.append(kwargs)
        if kwargs["method"] == "conversations.replies":
            assert kwargs["payload"]["channel"] == "C1"
            assert kwargs["payload"]["ts"] == "111.000"
            assert kwargs["payload"]["latest"] == "333.000"
            assert kwargs["payload"]["inclusive"] is True
            return {
                "ok": True,
                "messages": [
                    {
                        "type": "message",
                        "user": "U2",
                        "text": "What do we have on rent?",
                        "ts": "111.000",
                    },
                    {
                        "type": "message",
                        "bot_id": "B1",
                        "text": "Earlier Wavelength answer.",
                        "ts": "222.000",
                    },
                    {
                        "type": "message",
                        "user": "U1",
                        "text": "<@BOT> What about Chicago?",
                        "ts": "333.000",
                    },
                ],
            }
        if kwargs["method"] == "chat.startStream":
            return {"ok": True, "ts": "999.000"}
        return {"ok": True}

    service = CapturingStreamingService()
    monkeypatch.setattr(mcp_server, "_call_slack_api", fake_call_slack_api)

    mcp_server._handle_event_question(
        service=service,
        bot_token="xoxb-test",
        question="What about Chicago?",
        channel="C1",
        user_id="U1",
        team_id="T1",
        thread_ts="111.000",
        event_ts="333.000",
    )

    assert service.calls[0]["question"] == "What about Chicago?"
    assert service.calls[0]["conversation_context"] == [
        {"role": "user", "content": "What do we have on rent?"},
        {"role": "assistant", "content": "Earlier Wavelength answer."},
        {"role": "user", "content": "Slack user: <@U1>"},
    ]


def test_slack_message_text_includes_fallback_and_block_sources():
    text = mcp_server._message_text(
        {
            "text": "Answer text",
            "blocks": [
                {"type": "section", "text": {"type": "mrkdwn", "text": "Answer text"}},
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "*Sources*\n[1] First\n[2] Second"},
                },
            ],
        }
    )

    assert text == "Answer text\n\n*Sources*\n[1] First\n[2] Second"


def test_persisted_slack_thread_context_includes_source_references(monkeypatch):
    calls = []

    def fake_call_slack_api(**kwargs):
        calls.append(kwargs)
        if kwargs["method"] == "conversations.replies":
            raise RuntimeError("Slack conversations.replies failed")
        if kwargs["method"] == "chat.startStream":
            return {"ok": True, "ts": "999.000"}
        return {"ok": True}

    service = SourceStreamingService()
    monkeypatch.setattr(mcp_server, "_call_slack_api", fake_call_slack_api)

    mcp_server._handle_event_question(
        service=service,
        bot_token="xoxb-test",
        question="Find relevant coverage",
        channel="D1",
        user_id="U1",
        team_id="T1",
        thread_ts="111.000",
        event_ts="111.000",
    )

    context = mcp_server._persisted_slack_thread_context(
        channel="D1",
        thread_ts="111.000",
    )

    assert context[1]["role"] == "assistant"
    assert "Sources\n[1] First source" in context[1]["content"]
    assert "[2] Second source\ntranscript - 2026-01-02" in context[1]["content"]
    assert "Excerpt: Second excerpt" in context[1]["content"]


def test_work_object_metadata_trim_preserves_article_unfurl_fields():
    sources = [
        {
            "number": index + 1,
            "source_id": f"story-{index}",
            "title": "Long archive story title " * 12,
            "url": f"https://example.com/story/{index}",
            "publish_date": "2026-01-02",
            "content_type": "article",
            "authors": ["Reporter One", "Reporter Two"],
        }
        for index in range(7)
    ]

    batches = mcp_server._tool_payload_work_object_metadata_batches(
        {"structuredContent": {"sources": sources}}
    )

    assert len(batches) == 2
    assert sum(len(batch["entities"]) for batch in batches) == 7
    assert all(mcp_server._metadata_size(batch) <= mcp_server.SLACK_WORK_OBJECT_METADATA_MAX_BYTES for batch in batches)
    payload = batches[0]["entities"][0]["entity_payload"]
    assert payload["attributes"]["display_type"] == "Article"
    fields = {field["key"]: field for field in payload["custom_fields"]}
    assert fields["date"]["value"] == "2026-01-02"
    assert fields["author"]["value"] == "Reporter One, Reporter Two"


def test_post_slack_message_with_work_objects_posts_remaining_batches(monkeypatch):
    posted = []
    sources = [
        {
            "number": index + 1,
            "source_id": f"story-{index}",
            "title": "Long archive story title " * 12,
            "url": f"https://example.com/story/{index}",
            "publish_date": "2026-01-02",
            "content_type": "article",
            "authors": ["Reporter One", "Reporter Two"],
        }
        for index in range(7)
    ]

    monkeypatch.setattr(mcp_server, "_post_slack_message", lambda **kwargs: posted.append(kwargs))
    mcp_server._cache_team_api_app_id("T1", "A123")

    mcp_server._post_slack_message_with_work_objects(
        token="xoxb-test",
        channel="C1",
        text="answer",
        blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": "answer"}}],
        thread_ts="111.000",
        payload={"structuredContent": {"sources": sources}},
        team_id="T1",
    )

    assert len(posted) == 2
    assert posted[0]["text"] == "answer"
    assert posted[1]["text"] == "Wavelength source unfurls."
    assert posted[1]["blocks"][0]["type"] == "rich_text"
    assert sum(len(call["metadata"]["entities"]) for call in posted) == 7


def test_work_object_source_id_from_event_prefers_external_ref():
    assert mcp_server._work_object_source_id_from_event(
        {
            "entity": {"url": "https://example.com/story"},
            "external_ref": {"id": "story-1"},
        }
    ) == "story-1"
    assert mcp_server._work_object_source_id_from_event(
        {"entity": {"external_ref": {"id": "story-2"}}}
    ) == "story-2"


def test_slack_thread_history_falls_back_to_persisted_turns(monkeypatch, caplog):
    calls = []

    def fake_call_slack_api(**kwargs):
        calls.append(kwargs)
        if kwargs["method"] == "conversations.replies":
            raise RuntimeError("Slack conversations.replies failed")
        if kwargs["method"] == "chat.startStream":
            return {"ok": True, "ts": "999.000"}
        return {"ok": True}

    service = CapturingStreamingService()
    monkeypatch.setattr(mcp_server, "_call_slack_api", fake_call_slack_api)

    with caplog.at_level(logging.WARNING, logger="mcp_server"):
        mcp_server._handle_event_question(
            service=service,
            bot_token="xoxb-test",
            question="hi what do we have on bears coaches in the last five years",
            channel="C1",
            user_id="U1",
            team_id="T1",
            thread_ts="111.000",
            event_ts="111.000",
        )
        mcp_server._handle_event_question(
            service=service,
            bot_token="xoxb-test",
            question="hirings and firings for the first couple of years in that interval",
            channel="C1",
            user_id="U1",
            team_id="T1",
            thread_ts="111.000",
            event_ts="222.000",
        )

    assert service.calls[1]["question"] == (
        "hirings and firings for the first couple of years in that interval"
    )
    assert service.calls[1]["conversation_context"] == [
        {
            "role": "user",
            "content": "hi what do we have on bears coaches in the last five years",
        },
        {
            "role": "assistant",
            "content": "answer: hi what do we have on bears coaches in the last five years",
        },
        {"role": "user", "content": "Slack user: <@U1>"},
    ]
    assert "Slack conversations.replies failed; using persisted fallback" in caplog.text
    assert "channel=C1 thread_ts=111.000 event_ts=222.000" in caplog.text
    assert "persisted_turns=2" in caplog.text


def test_slack_thread_context_skips_lookup_without_required_identifiers(monkeypatch, caplog):
    calls = []

    def fake_call_slack_api(**kwargs):
        calls.append(kwargs)
        return {"ok": True, "messages": []}

    monkeypatch.setattr(mcp_server, "_call_slack_api", fake_call_slack_api)

    with caplog.at_level(logging.WARNING, logger="mcp_server"):
        context = mcp_server._slack_thread_context_or_empty(
            token="xoxb-test",
            channel=None,
            thread_ts=None,
            current_event_ts="222.000",
            current_user_id="U1",
            current_question="follow up",
        )

    assert context == []
    assert calls == []
    assert "Skipping Slack conversations.replies; missing required thread identifiers" in caplog.text
    assert "channel=None thread_ts=None event_ts=222.000" in caplog.text


def test_slack_event_uses_channel_id_and_event_ts_fallbacks(monkeypatch):
    monkeypatch.setenv("WAVELENGTH_SKIP_SLACK_REQUEST_AUTH", "true")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    app = Starlette()
    mcp_server._add_slack_routes(app, FakeStreamingService())

    response = asyncio.run(
        route_endpoint(app, "/slack/events")(
            FakeRequest(
                body=(
                    b'{"type":"event_callback","team_id":"T1","event":{"type":"app_mention",'
                    b'"user":"U1","channel_id":"C1","text":"<@BOT> Find CTA funding stories",'
                    b'"event_ts":"123.456"}}'
                )
            )
        )
    )

    assert response.status_code == 200
    assert response.background is not None
    assert response.background.kwargs["channel"] == "C1"
    assert response.background.kwargs["thread_ts"] == "123.456"
    assert response.background.kwargs["event_ts"] == "123.456"


def test_slack_event_uses_assistant_thread_identifiers(monkeypatch):
    monkeypatch.setenv("WAVELENGTH_SKIP_SLACK_REQUEST_AUTH", "true")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    app = Starlette()
    mcp_server._add_slack_routes(app, FakeStreamingService())

    response = asyncio.run(
        route_endpoint(app, "/slack/events")(
            FakeRequest(
                body=(
                    b'{"type":"event_callback","team_id":"T1","event":{"type":"message",'
                    b'"channel_type":"im","user":"U1","text":"Tell me about those",'
                    b'"event_ts":"123.789","assistant_thread":{'
                    b'"channel_id":"D1","thread_ts":"123.456"}}}'
                )
            )
        )
    )

    assert response.status_code == 200
    assert response.background is not None
    assert response.background.kwargs["channel"] == "D1"
    assert response.background.kwargs["thread_ts"] == "123.456"
    assert response.background.kwargs["persist_thread_ts"] == "123.456"
    assert response.background.kwargs["use_dm_window_context"] is False
    assert response.background.kwargs["event_ts"] == "123.789"


def test_slack_thread_history_paginates(monkeypatch):
    cursors = []

    def fake_call_slack_api(**kwargs):
        cursors.append(kwargs["payload"].get("cursor"))
        if kwargs["payload"].get("cursor") is None:
            return {
                "ok": True,
                "messages": [{"type": "message", "user": "U1", "text": "first", "ts": "1.000"}],
                "response_metadata": {"next_cursor": "next-page"},
            }
        return {
            "ok": True,
            "messages": [{"type": "message", "user": "U2", "text": "second", "ts": "2.000"}],
            "response_metadata": {"next_cursor": ""},
        }

    monkeypatch.setattr(mcp_server, "_call_slack_api", fake_call_slack_api)

    messages = mcp_server._slack_thread_messages(
        token="xoxb-test",
        channel="C1",
        thread_ts="1.000",
        latest_ts="2.000",
    )

    assert cursors == [None, "next-page"]
    assert [message["text"] for message in messages] == ["first", "second"]


def test_wavelength_slash_command_acks_and_posts_delayed_response(monkeypatch):
    responses = []

    def fake_post_slash_response(**kwargs):
        responses.append(kwargs)

    monkeypatch.setenv("WAVELENGTH_SKIP_SLACK_REQUEST_AUTH", "true")
    monkeypatch.setattr(mcp_server, "_post_slash_response", fake_post_slash_response)
    app = Starlette()
    mcp_server._add_slack_routes(app, FakeService())

    response = asyncio.run(
        route_endpoint(app, "/slack/commands/wavelength")(
            FakeRequest(
                body=(
                    b"text=Find+education+coverage&"
                    b"response_url=https%3A%2F%2Fhooks.slack.com%2Fcommands%2F1&"
                    b"user_id=U1"
                )
            )
        )
    )

    assert response.status_code == 200
    assert response.background is not None
    assert json.loads(response.body)["text"] == "Wavelength is searching the archive."

    mcp_server._handle_slash_question(
        service=FakeService(),
        question="Find education coverage",
        response_url="https://hooks.slack.com/commands/1",
        bot_token=None,
        channel=None,
        user_id="U1",
        team_id=None,
        response_type="ephemeral",
    )

    assert responses[0]["response_url"] == "https://hooks.slack.com/commands/1"
    assert responses[0]["text"] == "answer: Find education coverage"


def test_wavelength_slash_command_streams_when_bot_token_and_channel_are_available(monkeypatch):
    calls = []
    responses = []

    def fake_call_slack_api(**kwargs):
        calls.append(kwargs)
        if kwargs["method"] == "chat.startStream":
            return {"ok": True, "ts": "999.000"}
        return {"ok": True}

    def fake_post_slash_response(**kwargs):
        responses.append(kwargs)

    monkeypatch.setenv("WAVELENGTH_SKIP_SLACK_REQUEST_AUTH", "true")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setattr(mcp_server, "_call_slack_api", fake_call_slack_api)
    monkeypatch.setattr(mcp_server, "_post_slash_response", fake_post_slash_response)
    app = Starlette()
    mcp_server._add_slack_routes(app, FakeStreamingService())

    response = asyncio.run(
        route_endpoint(app, "/slack/commands/wavelength")(
            FakeRequest(
                body=(
                    b"text=Find+education+coverage&"
                    b"response_url=https%3A%2F%2Fhooks.slack.com%2Fcommands%2F1&"
                    b"channel_id=C1&user_id=U1&team_id=T1"
                )
            )
        )
    )

    assert response.status_code == 200
    assert response.background is not None
    assert response.background.kwargs["bot_token"] == "xoxb-test"
    assert response.background.kwargs["channel"] == "C1"

    mcp_server._handle_slash_question(
        service=FakeStreamingService(),
        question="Find education coverage",
        response_url="https://hooks.slack.com/commands/1",
        bot_token="xoxb-test",
        channel="C1",
        user_id="U1",
        team_id="T1",
        response_type="ephemeral",
    )

    methods = [call["method"] for call in calls]
    assert methods == ["chat.startStream", "chat.appendStream", "chat.stopStream", "chat.postMessage"]
    answer_start_payload = calls[0]["payload"]
    assert answer_start_payload["channel"] == "C1"
    assert answer_start_payload["recipient_user_id"] == "U1"
    assert answer_start_payload["recipient_team_id"] == "T1"
    assert answer_start_payload["thread_ts"] is None
    assert answer_start_payload["chunks"] is None
    assert calls[-1]["payload"]["text"] == "answer: Find education coverage"
    assert responses == []


def test_slack_stream_stop_retries_without_blocks_on_streaming_mode_mismatch(monkeypatch):
    calls = []

    def fake_call_slack_api(**kwargs):
        calls.append(kwargs)
        if kwargs["method"] == "chat.startStream":
            return {"ok": True, "ts": "999.000"}
        if kwargs["method"] == "chat.stopStream" and kwargs["payload"].get("blocks"):
            raise RuntimeError(
                "Slack chat.stopStream failed: {'ok': False, 'error': 'streaming_mode_mismatch'}"
            )
        return {"ok": True}

    monkeypatch.setattr(mcp_server, "_call_slack_api", fake_call_slack_api)

    mcp_server._handle_slash_question(
        service=FakeStreamingService(),
        question="Find education coverage",
        response_url="https://hooks.slack.com/commands/1",
        bot_token="xoxb-test",
        channel="C1",
        user_id="U1",
        team_id="T1",
        response_type="ephemeral",
    )

    stop_calls = [call for call in calls if call["method"] == "chat.stopStream"]
    assert len(stop_calls) == 1
    assert stop_calls[0]["payload"]["blocks"] is None
