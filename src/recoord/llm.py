"""A small model client for the judgment steps of an LLM agent.

Supports the Anthropic and OpenAI APIs, caches every completion on disk keyed
by the prompt, and reports token usage so cost comparisons are honest.  Without
a key, use `--agent simulated`.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .federation import Usage

SYSTEM = (
    "You repair cells of a relational table. You are given the answers of the "
    "queries you asked. Reply with JSON only: "
    '{"values": [...], "confidence": 0.0-1.0}. '
    "One value per requested cell, in order. Use null when you cannot tell."
)


@dataclass
class Reply:
    values: list[Any]
    confidence: float
    usage: Usage = field(default_factory=Usage)


class LLMClient:
    def __init__(self, model: str = "claude-opus-5", temperature: float = 0.0,
                 cache_dir: str | Path = ".cache/llm", provider: str | None = None,
                 max_tokens: int = 512):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.provider = provider or ("openai" if model.startswith(("gpt-", "o1", "o3")) else "anthropic")
        self._client: Any = None

    # -- public ------------------------------------------------------------
    def repair(self, rule: str, columns: Sequence[str], rows: Sequence[str],
               answers: tuple, instruction: str = "") -> Reply:
        prompt = json.dumps({
            "task": rule, "instruction": instruction,
            "cells": [{"row": r, "column": c} for r, c in zip(rows, columns)],
            "evidence": [[list(row) for row in answer] for answer in answers],
        }, sort_keys=True)
        cached = self._read_cache(prompt)
        if cached is not None:
            return Reply(cached.get("values", []), float(cached.get("confidence", 0.5)))
        text, usage = self._complete(prompt)
        payload = _parse(text)
        self._write_cache(prompt, payload)
        return Reply(payload.get("values", []), float(payload.get("confidence", 0.5)), usage)

    # -- providers ---------------------------------------------------------
    def _complete(self, prompt: str) -> tuple[str, Usage]:
        if self.provider == "anthropic":
            return self._anthropic(prompt)
        return self._openai(prompt)

    def _anthropic(self, prompt: str) -> tuple[str, Usage]:
        import anthropic
        self._client = self._client or anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        message = self._client.messages.create(
            model=self.model, max_tokens=self.max_tokens, temperature=self.temperature,
            system=SYSTEM, messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in message.content if block.type == "text")
        return text, Usage(1, message.usage.input_tokens, message.usage.output_tokens)

    def _openai(self, prompt: str) -> tuple[str, Usage]:
        import openai
        self._client = self._client or openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        reply = self._client.chat.completions.create(
            model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        )
        usage = reply.usage
        return reply.choices[0].message.content or "", Usage(1, usage.prompt_tokens, usage.completion_tokens)

    # -- cache -------------------------------------------------------------
    def _path(self, prompt: str) -> Path:
        digest = hashlib.sha256(f"{self.model}|{self.temperature}|{prompt}".encode()).hexdigest()
        return self.cache / f"{digest}.json"

    def _read_cache(self, prompt: str) -> dict | None:
        path = self._path(prompt)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return None

    def _write_cache(self, prompt: str, payload: dict) -> None:
        self._path(prompt).write_text(json.dumps(payload, sort_keys=True))


def _parse(text: str) -> dict:
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {"values": [], "confidence": 0.0}
