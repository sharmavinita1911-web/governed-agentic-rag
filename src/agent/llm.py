"""Local generator: a thin wrapper over Ollama serving Gemma (E4B).

Every call returns the text *and* a :class:`Usage` record (prompt/completion
tokens + wall-clock latency) so the evaluation harness can price the
governance overhead — the token + latency half of the trade-off curve.

Requires a running Ollama daemon with the model pulled:

    ollama pull hf.co/unsloth/gemma-4-E4B-it-GGUF:Q4_K_M
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from ..config import load_config


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMResult:
    text: str
    usage: Usage = field(default_factory=Usage)


class OllamaLLM:
    def __init__(self, cfg: dict | None = None):
        import ollama

        cfg = cfg or load_config()
        llm = cfg["llm"]
        self.model = llm["model"]
        self.temperature = llm.get("temperature", 0.0)
        self.max_tokens = llm.get("max_tokens", 512)
        # Generous timeout: CPU generation of a 4B model can take a minute+, and
        # httpx's default (a few seconds) would abort it mid-run.
        timeout = llm.get("timeout_s", 600)
        host = llm.get("host")
        self._client = ollama.Client(host=host, timeout=timeout) if host \
            else ollama.Client(timeout=timeout)

    def generate(self, prompt: str, system: Optional[str] = None) -> LLMResult:
        messages: List[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        t0 = time.perf_counter()
        resp = self._client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": self.temperature, "num_predict": self.max_tokens},
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # ollama returns eval counts on the response dict/object
        text = resp["message"]["content"] if isinstance(resp, dict) else resp.message.content
        prompt_tokens = _get(resp, "prompt_eval_count", 0)
        completion_tokens = _get(resp, "eval_count", 0)
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )
        return LLMResult(text=text.strip(), usage=usage)


def _get(resp, key, default):
    """Read a field from an ollama response that may be a dict or an object."""
    if isinstance(resp, dict):
        return resp.get(key, default) or default
    return getattr(resp, key, default) or default
