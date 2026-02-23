"""State specification and key result definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .context import ExecutionContext


@dataclass
class KeyResult:
    """A verifiable deliverable for a state (the 'test' in TDD).

    If `check` is provided, it runs as a programmatic assertion.
    If `check` is None, the LLM validator evaluates it from the description.
    """

    name: str
    description: str
    check: Callable[[Any], bool] | None = None


@dataclass
class StateSpec:
    """Declares a workflow state with its objective and acceptance criteria.

    Attributes:
        name: Unique state identifier.
        objective: What this state should accomplish (the OKR Objective).
        key_results: Verifiable deliverables (the OKR Key Results / tests).
        execute: Function called with an ExecutionContext to do the work.
                 If None, the state is a pass-through (output = input).
        tools: Callables available as tools in this state only.
        max_retries: How many times to retry if validation fails.
        is_initial: Whether this is the workflow entry state.
        is_final: Whether this is a terminal state.
    """

    name: str
    objective: str
    key_results: list[KeyResult] = field(default_factory=list)
    execute: Callable[[ExecutionContext], Any] | None = None
    tools: list[Callable] = field(default_factory=list)
    max_retries: int = 3
    is_initial: bool = False
    is_final: bool = False
