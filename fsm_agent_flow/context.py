"""Structured context objects for workflow execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .workflow import BoundLLM


@dataclass
class SharedContext:
    """Key-value store shared across all states. Replaces the 10-bucket memory."""

    data: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def delete(self, key: str) -> None:
        self.data.pop(key, None)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SharedContext:
        return cls(data=dict(d))


@dataclass
class StateOutput:
    """Recorded output from a completed state."""

    state_name: str
    output: Any
    key_results_met: dict[str, bool] = field(default_factory=dict)


@dataclass
class ExecutionContext:
    """Provided to each state's execute function.

    Attributes:
        input: Output from the previous state (or initial workflow input).
        shared: Read/write shared context across all states.
        history: Previous states' outputs (read-only by convention).
        llm: LLM pre-configured with this state's tools.
        retry_count: How many times this state has been retried.
        feedback: Validator feedback from the previous failed attempt.
    """

    input: Any
    shared: SharedContext
    history: list[StateOutput]
    llm: BoundLLM | None = None
    retry_count: int = 0
    feedback: str | None = None


@dataclass
class WorkflowContext:
    """Top-level workflow context. Serializable for persistence/resumption."""

    objective: str
    shared: SharedContext = field(default_factory=SharedContext)
    history: list[StateOutput] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "shared": self.shared.to_dict(),
            "history": [
                {
                    "state_name": h.state_name,
                    "output": h.output,
                    "key_results_met": h.key_results_met,
                }
                for h in self.history
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WorkflowContext:
        return cls(
            objective=d["objective"],
            shared=SharedContext.from_dict(d.get("shared", {})),
            history=[
                StateOutput(
                    state_name=h["state_name"],
                    output=h["output"],
                    key_results_met=h.get("key_results_met", {}),
                )
                for h in d.get("history", [])
            ],
        )
