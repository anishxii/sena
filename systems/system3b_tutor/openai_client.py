from __future__ import annotations

import json
import os
from pathlib import Path
import urllib.request


OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIChatClient:
    """Minimal OpenAI Chat Completions client using stdlib networking."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: int = 60,
    ) -> None:
        self.api_key = api_key or _load_api_key()
        self.model = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
        self.timeout_s = timeout_s
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is required. Set it in the environment or a local .env file."
            )

    def complete_text(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1200,
        temperature: float = 0.4,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
        }
        response = self._post(payload)
        return response["choices"][0]["message"]["content"].strip()

    def complete_json(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1200,
        temperature: float = 0.2,
    ) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        response = self._post(payload)
        content = response["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("Expected JSON object response from OpenAI")
        return parsed

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            OPENAI_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if "error" in data:
            raise RuntimeError(data["error"])
        return data


def _load_api_key() -> str | None:
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key

    env_file = Path(__file__).resolve().parents[2] / ".env"
    if not env_file.exists():
        return None

    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() == "OPENAI_API_KEY":
            return value.strip().strip('"').strip("'")
    return None
