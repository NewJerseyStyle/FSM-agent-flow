"""TDD-style validation for state outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .context import ExecutionContext
from .llm.adapter import LLMAdapter, Message
from .state import KeyResult, StateSpec


@dataclass
class ValidationResult:
    """Result of validating a state's output against its key results."""

    passed: bool
    feedback: str | None = None
    key_results: dict[str, bool] = field(default_factory=dict)


@runtime_checkable
class Validator(Protocol):
    """Protocol for state output validators."""

    def validate(
        self, state: StateSpec, output: Any, context: ExecutionContext
    ) -> ValidationResult: ...


class RuleValidator:
    """Pure programmatic validator — runs only KeyResult.check functions."""

    def validate(
        self, state: StateSpec, output: Any, context: ExecutionContext
    ) -> ValidationResult:
        if not state.key_results:
            return ValidationResult(passed=True)

        kr_results: dict[str, bool] = {}
        failures: list[str] = []

        for kr in state.key_results:
            if kr.check is not None:
                try:
                    passed = kr.check(output)
                except Exception as e:
                    passed = False
                    failures.append(f"{kr.name}: check raised {e}")
                kr_results[kr.name] = passed
                if not passed and f"{kr.name}:" not in " ".join(failures):
                    failures.append(f"{kr.name}: {kr.description}")
            else:
                # No programmatic check — skip (assume pass)
                kr_results[kr.name] = True

        all_passed = all(kr_results.values())
        feedback = "; ".join(failures) if failures else None
        return ValidationResult(passed=all_passed, feedback=feedback, key_results=kr_results)


class LLMValidator:
    """Uses an LLM to evaluate key results that lack programmatic checks."""

    def __init__(self, llm: LLMAdapter):
        self._llm = llm

    def validate(
        self, state: StateSpec, output: Any, context: ExecutionContext
    ) -> ValidationResult:
        if not state.key_results:
            return ValidationResult(passed=True)

        kr_results: dict[str, bool] = {}
        failures: list[str] = []
        needs_llm: list[KeyResult] = []

        # Phase 1: run programmatic checks
        for kr in state.key_results:
            if kr.check is not None:
                try:
                    passed = kr.check(output)
                except Exception as e:
                    passed = False
                    failures.append(f"{kr.name}: check raised {e}")
                kr_results[kr.name] = passed
                if not passed and f"{kr.name}:" not in " ".join(failures):
                    failures.append(f"{kr.name}: {kr.description}")
            else:
                needs_llm.append(kr)

        # Phase 2: LLM evaluation for remaining KRs
        if needs_llm:
            llm_results = self._evaluate_with_llm(state, output, needs_llm, context)
            kr_results.update(llm_results)
            for kr in needs_llm:
                if not llm_results.get(kr.name, False):
                    failures.append(f"{kr.name}: {kr.description}")

        all_passed = all(kr_results.values())
        feedback = "; ".join(failures) if failures else None
        return ValidationResult(passed=all_passed, feedback=feedback, key_results=kr_results)

    def _evaluate_with_llm(
        self,
        state: StateSpec,
        output: Any,
        key_results: list[KeyResult],
        context: ExecutionContext,
    ) -> dict[str, bool]:
        kr_descriptions = "\n".join(
            f"- {kr.name}: {kr.description}" for kr in key_results
        )
        prompt = (
            f"You are a validator. The state '{state.name}' has objective: {state.objective}\n\n"
            f"The state produced this output:\n{output}\n\n"
            f"Evaluate whether the following key results are met:\n{kr_descriptions}\n\n"
            f"Respond with ONLY valid JSON: a dict mapping each key result name to true/false.\n"
            f"Example: {{{json.dumps({kr.name: True for kr in key_results})}}}"
        )

        messages = [Message(role="user", content=prompt)]
        response = self._llm.chat(messages, temperature=0.0)

        try:
            text = (response.content or "").strip()
            # Handle markdown code blocks
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(text)
            return {kr.name: bool(result.get(kr.name, False)) for kr in key_results}
        except (json.JSONDecodeError, AttributeError):
            # If LLM response is unparseable, fail all
            return {kr.name: False for kr in key_results}
