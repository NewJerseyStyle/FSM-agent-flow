"""Core workflow engine with built-in TDD validation loop."""

from __future__ import annotations

import json
import logging
from typing import Any

from .context import ExecutionContext, SharedContext, StateOutput, WorkflowContext
from .errors import ExecutionBreak, MaxRetriesExceeded, WorkflowError
from .llm.adapter import LLMAdapter, LLMResponse, Message, ToolCall
from .state import StateSpec
from .tools import ToolRegistry, ToolSpec
from .validation import LLMValidator, RuleValidator, ValidationResult, Validator

logger = logging.getLogger(__name__)


class BoundLLM:
    """LLM adapter pre-bound with a state's tools.

    Provides both raw `chat()` and a convenience `run_with_tools()` that
    handles the tool-call loop automatically.
    """

    def __init__(self, adapter: LLMAdapter, tool_specs: list[ToolSpec]):
        self.adapter = adapter
        self._tool_specs = tool_specs
        self._registry = ToolRegistry()
        for spec in tool_specs:
            self._registry._tools[spec.name] = spec
        self._formatted_tools = adapter.format_tools(tool_specs) if tool_specs else []

    def chat(self, messages: list[Message], **kwargs) -> LLMResponse:
        """Send a chat request with this state's tools available."""
        if self._formatted_tools:
            kwargs.setdefault("tools", self._formatted_tools)
        return self.adapter.chat(messages, **kwargs)

    def run_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        *,
        max_iterations: int = 10,
        temperature: float = 0.7,
    ) -> str:
        """Run a tool-use loop: call LLM, execute tools, repeat until done.

        Returns the final text content from the LLM.
        """
        messages: list[Message] = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_message),
        ]

        for _ in range(max_iterations):
            response = self.chat(messages, temperature=temperature)

            if not response.tool_calls:
                return response.content or ""

            # Record assistant message with tool calls
            messages.append(
                Message(role="assistant", content=response.content, tool_calls=response.tool_calls)
            )

            # Execute each tool call and add results
            for tc in response.tool_calls:
                try:
                    result = self._registry.execute(tc.name, tc.arguments)
                    messages.append(
                        Message(
                            role="tool",
                            content=str(result),
                            tool_call_id=tc.id,
                            name=tc.name,
                        )
                    )
                except ExecutionBreak:
                    raise
                except Exception as e:
                    messages.append(
                        Message(
                            role="tool",
                            content=f"Error: {e}",
                            tool_call_id=tc.id,
                            name=tc.name,
                        )
                    )

        # Exhausted iterations — return last content
        return response.content or ""


class Workflow:
    """Lightweight FSM engine with TDD/OKR validation loop.

    Usage:
        wf = Workflow(
            objective="Research and summarize a topic",
            states=[research_state, writing_state],
            transitions={"research": "writing", "writing": None},
            llm=OpenAIAdapter(model="gpt-4o"),
        )
        result = wf.run("quantum computing")
    """

    def __init__(
        self,
        objective: str,
        states: list[StateSpec],
        transitions: dict[str, str | None],
        llm: LLMAdapter,
        validator: Validator | None = None,
        validator_llm: LLMAdapter | None = None,
    ):
        self._objective = objective
        self._states: dict[str, StateSpec] = {s.name: s for s in states}
        self._transitions = transitions
        self._llm = llm

        # Determine validator
        if validator is not None:
            self._validator = validator
        elif validator_llm is not None:
            self._validator = LLMValidator(validator_llm)
        else:
            # Default: use rule-based (programmatic checks only)
            self._validator = RuleValidator()

        # Find initial state
        initial = [s for s in states if s.is_initial]
        if not initial:
            initial = [states[0]] if states else []
        if not initial:
            raise WorkflowError("Workflow must have at least one state")
        self._current_state_name: str = initial[0].name

        self._context = WorkflowContext(objective=objective)
        self._finished = False

    @property
    def current_state(self) -> str:
        return self._current_state_name

    @property
    def is_finished(self) -> bool:
        return self._finished

    @property
    def context(self) -> WorkflowContext:
        return self._context

    def _build_tool_specs(self, state: StateSpec) -> list[ToolSpec]:
        """Build ToolSpec list from a state's tool callables."""
        specs = []
        for func in state.tools:
            if isinstance(func, ToolSpec):
                specs.append(func)
            else:
                specs.append(ToolSpec.from_callable(func))
        return specs

    def step(self, state_input: Any = None) -> Any:
        """Execute the current state with TDD validation, then advance.

        Returns the state's validated output.
        """
        if self._finished:
            raise WorkflowError("Workflow has already finished")

        state = self._states[self._current_state_name]
        tool_specs = self._build_tool_specs(state)
        bound_llm = BoundLLM(self._llm, tool_specs)

        output = None
        feedback = None

        for attempt in range(state.max_retries + 1):
            # Build execution context
            ctx = ExecutionContext(
                input=state_input,
                shared=self._context.shared,
                history=list(self._context.history),
                llm=bound_llm,
                retry_count=attempt,
                feedback=feedback,
            )

            # Execute state
            if state.execute is not None:
                output = state.execute(ctx)
            else:
                output = state_input  # Pass-through

            # Validate (the "test run")
            if state.key_results:
                result = self._validator.validate(state, output, ctx)
                if result.passed:
                    self._record_and_advance(state, output, result)
                    return output
                else:
                    feedback = result.feedback
                    logger.info(
                        "State '%s' failed validation (attempt %d/%d): %s",
                        state.name,
                        attempt + 1,
                        state.max_retries + 1,
                        feedback,
                    )
            else:
                # No key results — auto-pass
                self._record_and_advance(state, output, ValidationResult(passed=True))
                return output

        # Exhausted retries
        raise MaxRetriesExceeded(state.name, state.max_retries, feedback)

    def _record_and_advance(
        self, state: StateSpec, output: Any, result: ValidationResult
    ) -> None:
        """Record state output and advance to next state."""
        state_output = StateOutput(
            state_name=state.name,
            output=output,
            key_results_met=result.key_results,
        )
        self._context.history.append(state_output)

        # Advance
        if state.is_final:
            self._finished = True
            return

        next_state = self._transitions.get(state.name)
        if next_state is None:
            self._finished = True
        else:
            if next_state not in self._states:
                raise WorkflowError(f"Transition target '{next_state}' not found in states")
            self._current_state_name = next_state

    def run(self, initial_input: Any = None) -> WorkflowContext:
        """Run the entire workflow from start to finish.

        Returns the final WorkflowContext with all state outputs.
        """
        current_input = initial_input
        while not self._finished:
            current_input = self.step(current_input)
        return self._context
