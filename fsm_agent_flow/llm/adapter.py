"""LLM adapter protocol and shared data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..tools import ToolSpec


@dataclass
class ToolCall:
    """A tool call requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """A chat message."""

    role: str  # "system", "user", "assistant", "tool"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class LLMResponse:
    """Response from an LLM chat call."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


@runtime_checkable
class LLMAdapter(Protocol):
    """Minimal protocol for LLM providers.

    Implementations must provide:
      - chat(): send messages, get a response
      - format_tools(): convert ToolSpecs to provider-specific schema
    """

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...

    def format_tools(self, tools: list[ToolSpec]) -> list[dict[str, Any]]: ...
