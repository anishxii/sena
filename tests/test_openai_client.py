import json
from unittest.mock import patch

import pytest

from emotiv_learn.openai_client import OpenAIChatClient


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_openai_client_complete_json_parses_object() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({"followup_type": "continue"}),
                }
            }
        ]
    }
    with patch("urllib.request.urlopen", return_value=FakeResponse(payload)):
        client = OpenAIChatClient(api_key="test-key", model="test-model")
        result = client.complete_json([{"role": "user", "content": "Return JSON."}])

    assert result == {"followup_type": "continue"}


def test_openai_client_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("emotiv_learn.openai_client.Path.exists", lambda _: False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIChatClient()
