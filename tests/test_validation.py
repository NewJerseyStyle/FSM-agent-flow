"""Tests for the validation system."""

from fsm_agent_flow.context import ExecutionContext, SharedContext
from fsm_agent_flow.state import KeyResult, StateSpec
from fsm_agent_flow.validation import RuleValidator, ValidationResult


def make_ctx(input_val=None):
    return ExecutionContext(input=input_val, shared=SharedContext(), history=[])


class TestRuleValidator:
    def test_no_key_results_passes(self):
        state = StateSpec(name="s", objective="do something")
        v = RuleValidator()
        result = v.validate(state, "output", make_ctx())
        assert result.passed

    def test_all_checks_pass(self):
        state = StateSpec(
            name="s",
            objective="produce content",
            key_results=[
                KeyResult("has_content", "Output is non-empty", check=lambda o: bool(o)),
                KeyResult("long_enough", "At least 5 chars", check=lambda o: len(str(o)) >= 5),
            ],
        )
        v = RuleValidator()
        result = v.validate(state, "hello world", make_ctx())
        assert result.passed
        assert result.key_results["has_content"] is True
        assert result.key_results["long_enough"] is True
        assert result.feedback is None

    def test_check_fails(self):
        state = StateSpec(
            name="s",
            objective="produce content",
            key_results=[
                KeyResult("long_enough", "At least 100 chars", check=lambda o: len(str(o)) >= 100),
            ],
        )
        v = RuleValidator()
        result = v.validate(state, "short", make_ctx())
        assert not result.passed
        assert result.key_results["long_enough"] is False
        assert result.feedback is not None

    def test_check_exception_fails(self):
        def bad_check(o):
            raise ValueError("boom")

        state = StateSpec(
            name="s",
            objective="x",
            key_results=[KeyResult("bad", "will raise", check=bad_check)],
        )
        v = RuleValidator()
        result = v.validate(state, "anything", make_ctx())
        assert not result.passed
        assert "boom" in result.feedback

    def test_no_check_auto_passes(self):
        state = StateSpec(
            name="s",
            objective="x",
            key_results=[KeyResult("llm_only", "Needs LLM validation")],
        )
        v = RuleValidator()
        result = v.validate(state, "anything", make_ctx())
        assert result.passed  # RuleValidator skips KRs without checks
