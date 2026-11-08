from __future__ import annotations

import hashlib
import importlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .federation import Usage

BACKBONES = {
    "gpt-4o-mini": ("gpt-4o-mini-2024-07-18", "openai", None),
    "gpt-4o": ("gpt-4o-2024-08-06", "openai", None),
    "claude-3.5-sonnet": ("claude-3-5-sonnet-20241022", "anthropic", None),
    "llama-3.1-70b": ("meta-llama/Meta-Llama-3.1-70B-Instruct", "openai",
                      "https://api.together.xyz/v1"),
    "qwen2.5-72b": ("Qwen/Qwen2.5-72B-Instruct", "openai",
                    "https://api.together.xyz/v1"),
}


def resolve(name: str) -> tuple[str, str, str | None]:
    if name in BACKBONES:
        return BACKBONES[name]
    provider = "openai" if name.startswith(("gpt-", "o1", "o3")) else "anthropic"
    return name, provider, None


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
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0,
                 cache_dir: str | Path | None = None, provider: str | None = None,
                 max_tokens: int = 512, base_url: str | None = None):
        resolved, default_provider, default_url = resolve(model)
        self.model = resolved
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache = Path(cache_dir) if cache_dir else None
        if self.cache is not None:
            self.cache.mkdir(parents=True, exist_ok=True)
        self.provider = provider or default_provider
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL") or default_url
        self._client: Any = None

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

    def _complete(self, prompt: str) -> tuple[str, Usage]:
        if self.provider == "anthropic":
            return self._anthropic(prompt)
        return self._openai(prompt)

    def _anthropic(self, prompt: str) -> tuple[str, Usage]:
        anthropic = _require("anthropic", self.model)
        self._client = self._client or anthropic.Anthropic(api_key=_key("ANTHROPIC_API_KEY"))
        message = self._client.messages.create(
            model=self.model, max_tokens=self.max_tokens, temperature=self.temperature,
            system=SYSTEM, messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in message.content if block.type == "text")
        return text, Usage(1, message.usage.input_tokens, message.usage.output_tokens)

    def _openai(self, prompt: str) -> tuple[str, Usage]:
        openai = _require("openai", self.model)
        self._client = self._client or openai.OpenAI(
            api_key=_key("OPENAI_API_KEY"), base_url=self.base_url)
        reply = self._client.chat.completions.create(
            model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        )
        usage = reply.usage
        return reply.choices[0].message.content or "", Usage(1, usage.prompt_tokens, usage.completion_tokens)

    def _path(self, prompt: str) -> Path | None:
        if self.cache is None:
            return None
        digest = hashlib.sha256(f"{self.model}|{self.temperature}|{prompt}".encode()).hexdigest()
        return self.cache / f"{digest}.json"

    def _read_cache(self, prompt: str) -> dict | None:
        path = self._path(prompt)
        if path is None or not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return None

    def _write_cache(self, prompt: str, payload: dict) -> None:
        path = self._path(prompt)
        if path is None:
            return
        path.write_text(json.dumps(payload, sort_keys=True))


def _require(package: str, model: str):
    try:
        return importlib.import_module(package)
    except ModuleNotFoundError as error:
        raise RuntimeError(
            f"the model {model!r} needs the {package!r} package: "
            f"pip install -e '.[llm]', or run with --agent offline"
        ) from error


def _key(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"set {name} to use a model, or run with --agent offline")
    return value


def _parse(text: str) -> dict:
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {"values": [], "confidence": 0.0}
