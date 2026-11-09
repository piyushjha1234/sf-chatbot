"""Thin wrapper over the official Anthropic SDK.

* Structured calls (query plans, condensing, classification, judging) use `messages.parse` with Pydantic.
* Answers stream via the beta Messages API with server-side refusal fallback (`fallbacks="default"`).
* No sampling parameters (rejected on current models); depth is controlled with `output_config.effort`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast

import anthropic
from pydantic import BaseModel

from sfrag.core.settings import LLMSettings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMUnavailable(RuntimeError):
    pass


@dataclass
class StreamResult:
    text: str = ""
    stop_reason: str | None = None
    model: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    refused: bool = False


class LLMClient:
    def __init__(self, cfg: LLMSettings, client: anthropic.Anthropic | None = None) -> None:
        self.cfg = cfg
        self._client = client

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            if not self.cfg.enabled:
                raise LLMUnavailable("LLM disabled by configuration")
            self._client = anthropic.Anthropic(timeout=self.cfg.timeout_seconds, max_retries=3)
        return self._client

    # ------------------------------------------------------------------ structured
    def parse(
        self,
        schema: type[T],
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int = 4000,
        effort: str | None = None,
    ) -> T | None:
        """Schema-validated output. Returns None on refusal / truncation / API failure (callers degrade)."""
        if not self.cfg.enabled:
            return None
        try:
            resp = self.client.messages.parse(
                model=model or self.cfg.fast_model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=schema,
                output_config=cast(Any, {"effort": effort or self.cfg.fast_effort}),
            )
        except anthropic.APIConnectionError as e:
            log.warning("llm connection error", extra={"err": str(e)})
            return None
        except anthropic.RateLimitError as e:
            log.warning("llm rate limited", extra={"err": str(e)})
            return None
        except anthropic.APIStatusError as e:
            log.warning("llm api error", extra={"status": e.status_code, "err": str(e)})
            return None
        if resp.stop_reason in ("refusal", "max_tokens"):
            log.warning("llm structured call incomplete", extra={"stop_reason": resp.stop_reason})
            return None
        return resp.parsed_output

    # ------------------------------------------------------------------ streaming answer
    def stream_answer(self, system: str, messages: list[dict[str, Any]], result: StreamResult) -> Iterator[str]:
        """Yield text deltas; fills `result` (stop reason, model, usage) when the stream ends."""
        kwargs: dict[str, Any] = dict(
            model=self.cfg.answer_model,
            max_tokens=self.cfg.answer_max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
            output_config={"effort": self.cfg.answer_effort},
        )
        if self.cfg.server_side_fallback:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        with self.client.beta.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                result.text += text
                yield text
            final = stream.get_final_message()
        result.stop_reason = final.stop_reason
        result.model = final.model
        result.refused = final.stop_reason == "refusal"
        u = final.usage
        result.usage = {
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", None),
        }
