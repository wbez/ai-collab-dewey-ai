import sys
from pathlib import Path

import pytest


APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


@pytest.fixture(autouse=True)
def isolate_remote_responses_mcp(monkeypatch):
    monkeypatch.delenv("WAVELENGTH_RESPONSES_MCP_ENABLED", raising=False)
