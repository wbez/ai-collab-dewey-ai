# Wavelength MCP Server

Wavelength is the public-facing Slack archive agent. It runs as an MCP server and wraps the existing Dewey archive engine without renaming internal Dewey modules.

It also exposes a direct Slack bot surface:

- Users can DM Wavelength.
- Users can mention `@Wavelength` in a channel or thread.
- Users can run `/wavelength <question>`.

## What Was Added

- `app/mcp_server.py`: Streamable HTTP MCP server for Slackbot MCP Client.
- `app/mcp_server.py`: Slack Events API and slash command routes for the direct Wavelength bot.
- `app/wavelength_service.py`: Public Wavelength service wrapper around `Dewey`.
- `app/slack_format.py`: Slack `mrkdwn` citation conversion and Block Kit response builders.
- `slack/manifest.wavelength.json`: Slack app manifest for Wavelength.
- `slack/wavelength.agent.sample.json`: Sample Slack agent/app config with placeholders.

## MCP Tools

### `ask_archive`

Answers a newsroom archive question using retrieved articles, transcripts, and scripts.

Inputs:

- `question`: Required natural-language question.
- `conversation_context`: Optional prior turns as `{ "role": "...", "content": "..." }`.
- `content_types`: Optional list containing `article`, `transcript`, `script`, or a combination.
- `start_date`: Optional `YYYY-MM-DD`.
- `end_date`: Optional `YYYY-MM-DD`.

Output:

- Text fallback.
- Structured content with `answer`, `sources`, and `needs_clarification`.
- Slack Block Kit blocks under `_meta.slack.blocks`.

### `search_archive`

Returns source results without asking the model to synthesize a full answer.

Inputs:

- `question`
- `date_range`
- `authors`
- `speakers`
- `guests`
- `content_types`
- `program`
- `limit`

Output:

- Text fallback.
- Structured content with `question` and `sources`.
- Slack Block Kit blocks under `_meta.slack.blocks`.

### `get_source`

Fetches one source by stable archive identifier.

Inputs:

- `source_id`: A `chunk_id` or `occurrence_id`.

Output:

- Text fallback.
- Structured content with `source`.
- Slack Block Kit blocks under `_meta.slack.blocks`.

## Required Environment

The MCP server uses the existing Azure/OpenAI/Search settings:

```bash
AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
AZURE_OPENAI_API_KEY="..."
EMBEDDING_DEPLOYMENT_NAME="text-embedding-3-large"
EMBEDDING_MODEL_NAME="text-embedding-3-large"
CHATGPT_DEPLOYMENT_NAME="gpt-5"
CHATGPT_MODEL_NAME="gpt-5"
AZURE_SEARCH_ENDPOINT="https://your-search-service.search.windows.net"
AZURE_SEARCH_API_KEY="..."
AZURE_SEARCH_INDEX_NAME="your-search-index-name"
```

Wavelength-specific settings:

```bash
SLACK_BOT_TOKEN="xoxb-..."
SLACK_SIGNING_SECRET="..."
WAVELENGTH_MCP_AUTH_TOKEN="replace-with-a-long-random-token"
WAVELENGTH_PUBLIC_BASE_URL="https://wavelength.example.com"
WAVELENGTH_EMBED_SIGNING_SECRET="replace-with-a-long-random-token"
WAVELENGTH_WORK_OBJECT_EMBEDS="true"
WAVELENGTH_ALLOWED_SLACK_TEAM_IDS="T01234567,T76543210"
WAVELENGTH_ALLOWED_SLACK_USER_IDS="U01234567,U76543210"
WAVELENGTH_SLASH_RESPONSE_TYPE="ephemeral"
WAVELENGTH_SKIP_SLACK_REQUEST_AUTH="false"
WAVELENGTH_HTTP_HOST="127.0.0.1"
WAVELENGTH_HTTP_PORT="8000"
WAVELENGTH_MCP_HOST="127.0.0.1"
WAVELENGTH_SKIP_MCP_AUTH="false"
```

The Gradio web app uses separate settings and can run on a different port:

```bash
GRADIO_SERVER_NAME="0.0.0.0"
GRADIO_SERVER_PORT="7860"
```

Authentication behavior:

- `WAVELENGTH_MCP_AUTH_TOKEN` enables shared bearer-token auth. Send it as `Authorization: Bearer <token>` or `X-Wavelength-MCP-Token: <token>`.
- `SLACK_SIGNING_SECRET` enables Slack request signature verification for signed Slack requests.
- `WAVELENGTH_ALLOWED_SLACK_TEAM_IDS` optionally restricts authenticated requests to specific Slack workspaces when Slack identity headers are present.
- `WAVELENGTH_ALLOWED_SLACK_USER_IDS` optionally restricts authenticated requests to specific Slack users when Slack identity headers are present.
- `WAVELENGTH_WORK_OBJECT_EMBEDS` enables iframe-based Work Object previews when `WAVELENGTH_PUBLIC_BASE_URL` is HTTPS and `WAVELENGTH_EMBED_SIGNING_SECRET` is configured.
- `WAVELENGTH_EMBED_SIGNING_SECRET` signs short-lived Work Object embed URLs sent in `entity.presentDetails`.
- `WAVELENGTH_SKIP_MCP_AUTH="true"` bypasses all MCP auth. Use it only for local smoke tests.
- `WAVELENGTH_SKIP_SLACK_REQUEST_AUTH="true"` bypasses Slack Events API and slash command request signing. Use it only for local route tests.
- `WAVELENGTH_SLASH_RESPONSE_TYPE` controls final slash command visibility. Use `ephemeral` or `in_channel`.

Slack bot behavior:

- `POST /slack/events` handles Events API URL verification, `app_mention`, `message.im`, and `entity_details_requested`.
- `GET /slack/work-objects/embed/{source_id}` serves signed Work Object Embed iframe pages with Slack frame-ancestor CSP.
- `POST /slack/commands/wavelength` handles `/wavelength`.
- `GET /slack/health` is a lightweight health check.
- Mention replies are posted in the thread by default.
- DM replies post task chunks as one message, then stream the answer in a second message in the user's DM thread.
- Thread replies fetch prior Slack thread messages and pass them to Wavelength as conversation context before answering.
- Slash commands ack immediately, then post task chunks as one message and stream the answer in a second message when `SLACK_BOT_TOKEN` and `channel_id` are available. If streaming cannot start, they fall back to the command `response_url`.

## Install Dependencies

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If the virtual environment already exists:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Start Locally

From the repo root:

```bash
source .venv/bin/activate
PYTHONPATH=app \
WAVELENGTH_SKIP_MCP_AUTH=true \
WAVELENGTH_HTTP_HOST=127.0.0.1 \
WAVELENGTH_HTTP_PORT=8000 \
python app/mcp_server.py
```

The MCP endpoint is:

```text
http://127.0.0.1:8000/mcp
```

The Slack endpoints are:

```text
http://127.0.0.1:8000/slack/events
http://127.0.0.1:8000/slack/commands/wavelength
http://127.0.0.1:8000/slack/health
```

The Gradio web app remains separate. Start it with `python app/main.py`; by default it listens on `http://127.0.0.1:7860` unless `GRADIO_SERVER_PORT` is changed.

For local Slackbot MCP testing, Slack needs an HTTPS URL that can reach this server. Use a temporary tunnel only for development, then put the production endpoint behind stable TLS.

To test local bearer auth instead of bypassing auth:

```bash
source .venv/bin/activate
PYTHONPATH=app \
WAVELENGTH_MCP_AUTH_TOKEN=dev-token \
WAVELENGTH_HTTP_HOST=127.0.0.1 \
WAVELENGTH_HTTP_PORT=8000 \
python app/mcp_server.py
```

## Start In Production

Run the ASGI app with `uvicorn`:

```bash
source .venv/bin/activate
PYTHONPATH=app \
WAVELENGTH_HTTP_HOST=0.0.0.0 \
WAVELENGTH_HTTP_PORT=8000 \
uvicorn mcp_server:app --host 0.0.0.0 --port 8000
```

Production requirements:

- Set `SLACK_SIGNING_SECRET` for Slack-signed requests and/or `WAVELENGTH_MCP_AUTH_TOKEN` for bearer-token clients.
- Set `SLACK_BOT_TOKEN` so Wavelength can post replies for DMs and mentions.
- Keep `WAVELENGTH_SKIP_MCP_AUTH` unset or `false`.
- Keep `WAVELENGTH_SKIP_SLACK_REQUEST_AUTH` unset or `false`.
- Set `WAVELENGTH_ALLOWED_SLACK_TEAM_IDS` for the workspace IDs allowed to use Wavelength.
- Put the service behind HTTPS.
- Configure Slack with the public HTTPS MCP URL, for example `https://wavelength.example.org/mcp`.
- Configure Slack Events API with `https://wavelength.example.org/slack/events`.
- Configure `/wavelength` with `https://wavelength.example.org/slack/commands/wavelength`.
- Do not enable Socket Mode for this app.

## Slack Setup

Use `slack/wavelength.agent.sample.json` as the starting point.

Before installing the Slack app:

- Replace `https://wavelength.example.org/mcp` with the real HTTPS MCP endpoint.
- Replace `https://wavelength.example.org/slack/events` with the real HTTPS Events API endpoint.
- Replace `https://wavelength.example.org/slack/commands/wavelength` with the real HTTPS slash command endpoint.
- Confirm the `mcp:connect` bot scope is present.
- Confirm `app_mentions:read`, `assistant:write`, `chat:write`, `commands`, `channels:history`, `groups:history`, `im:history`, and `mpim:history` bot scopes are present.
- Keep `socket_mode_enabled` set to `false`.
- Keep `auth_type` as `slack_identity_auth` unless you intentionally add a separate OAuth flow.

## Verification

Run the test suite:

```bash
python3 -m pytest
```

Check the venv dependency graph:

```bash
.venv/bin/python -m pip check
```

Check that the MCP server object can be constructed:

```bash
PYTHONPATH=app .venv/bin/python - <<'PY'
from mcp_server import create_mcp_server

class FakeService:
    pass

server = create_mcp_server(FakeService())
print(type(server).__name__)
PY
```

Expected output:

```text
MCPServer
```
