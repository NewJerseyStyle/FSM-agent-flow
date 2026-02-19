"""Tests for the workflow engine."""

import pytest

from fsm_agent_flow.context import ExecutionContext, SharedContext
from fsm_agent_flow.errors import MaxRetriesExceeded, WorkflowError
from fsm_agent_flow.llm.adapter import LLMAdapter, LLMResponse, Message
from fsm_agent_flow.state import KeyResult, StateSpec
from fsm_agent_flow.tools import ToolSpec
from fsm_agent_flow.validation import RuleValidator
from fsm_agent_flow.workflow import BoundLLM, Workflow


# -- Mock LLM --

class MockLLM:
    """Simple mock LLM that returns canned responses."""

    def __init__(self, responses: list[str] | None = None):
        self._responses = list(responses or ["Mock response"])
        self._call_count = 0

    def chat(self, messages, *, tools=None, temperature=0.7, max_tokens=None):
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return LLMResponse(content=self._responses[idx], finish_reason="stop")

    def format_tools(self, tools):
        return [t.to_openai_schema() for t in tools]


# -- Tests --

class TestWorkflowBasic:
    def test_single_state_passthrough(self):
        state = StateSpec(name="start", objective="pass through", is_initial=True, is_final=True)
        wf = Workflow(
            objective="test",
            states=[state],
            transitions={"start": None},
            llm=MockLLM(),
        )
        result = wf.run("hello")
        assert wf.is_finished
        assert result.history[0].output == "hello"

    def test_two_state_workflow(self):
        def upper(ctx: ExecutionContext):
            return str(ctx.input).upper()

        def exclaim(ctx: ExecutionContext):
            return str(ctx.input) + "!"

        s1 = StateSpec(name="upper", objective="uppercase", execute=upper, is_initial=True)
        s2 = StateSpec(name="exclaim", objective="add bang", execute=exclaim, is_final=True)

        wf = Workflow(
            objective="test",
            states=[s1, s2],
            transitions={"upper": "exclaim", "exclaim": None},
            llm=MockLLM(),
        )
        result = wf.run("hello")
        assert result.history[0].output == "HELLO"
        assert result.history[1].output == "HELLO!"

    def test_shared_context(self):
        def write_ctx(ctx: ExecutionContext):
            ctx.shared.set("key", "value")
            return "written"

        def read_ctx(ctx: ExecutionContext):
            return ctx.shared.get("key")

        s1 = StateSpec(name="write", objective="write", execute=write_ctx, is_initial=True)
        s2 = StateSpec(name="read", objective="read", execute=read_ctx, is_final=True)

        wf = Workflow(
            objective="test",
            states=[s1, s2],
            transitions={"write": "read", "read": None},
            llm=MockLLM(),
        )
        result = wf.run()
        assert result.history[1].output == "value"

    def test_step_after_finish_raises(self):
        state = StateSpec(name="s", objective="x", is_initial=True, is_final=True)
        wf = Workflow(objective="test", states=[state], transitions={"s": None}, llm=MockLLM())
        wf.run()
        with pytest.raises(WorkflowError, match="already finished"):
            wf.step()


class TestWorkflowValidation:
    def test_validation_passes(self):
        state = StateSpec(
            name="s",
            objective="produce output",
            key_results=[KeyResult("ok", "non-empty", check=lambda o: bool(o))],
            execute=lambda ctx: "good output",
            is_initial=True,
            is_final=True,
        )
        wf = Workflow(
            objective="test",
            states=[state],
            transitions={"s": None},
            llm=MockLLM(),
            validator=RuleValidator(),
        )
        result = wf.run()
        assert result.history[0].key_results_met["ok"] is True

    def test_validation_retries_then_passes(self):
        call_count = {"n": 0}

        def flaky(ctx: ExecutionContext):
            call_count["n"] += 1
            if call_count["n"] < 3:
                return ""  # Fails validation
            return "success"

        state = StateSpec(
            name="s",
            objective="eventually succeed",
            key_results=[KeyResult("ok", "non-empty", check=lambda o: bool(o))],
            execute=flaky,
            max_retries=3,
            is_initial=True,
            is_final=True,
        )
        wf = Workflow(
            objective="test",
            states=[state],
            transitions={"s": None},
            llm=MockLLM(),
            validator=RuleValidator(),
        )
        result = wf.run()
        assert result.history[0].output == "success"
        assert call_count["n"] == 3

    def test_max_retries_exceeded(self):
        state = StateSpec(
            name="s",
            objective="always fail",
            key_results=[KeyResult("never", "impossible", check=lambda o: False)],
            execute=lambda ctx: "bad",
            max_retries=2,
            is_initial=True,
            is_final=True,
        )
        wf = Workflow(
            objective="test",
            states=[state],
            transitions={"s": None},
            llm=MockLLM(),
            validator=RuleValidator(),
        )
        with pytest.raises(MaxRetriesExceeded):
            wf.run()


class TestToolScoping:
    def test_tools_scoped_to_state(self):
        def tool_a(x: str) -> str:
            return f"a:{x}"

        def tool_b(x: str) -> str:
            return f"b:{x}"

        seen_tools: dict[str, list[str]] = {}

        def capture_tools(state_name):
            def execute(ctx: ExecutionContext):
                tools = ctx.llm._tool_specs
                seen_tools[state_name] = [t.name for t in tools]
                return "done"
            return execute

        s1 = StateSpec(
            name="s1", objective="x",
            execute=capture_tools("s1"), tools=[tool_a],
            is_initial=True,
        )
        s2 = StateSpec(
            name="s2", objective="x",
            execute=capture_tools("s2"), tools=[tool_b],
            is_final=True,
        )

        wf = Workflow(
            objective="test",
            states=[s1, s2],
            transitions={"s1": "s2", "s2": None},
            llm=MockLLM(),
        )
        wf.run()
        assert seen_tools["s1"] == ["tool_a"]
        assert seen_tools["s2"] == ["tool_b"]


class TestBoundLLM:
    def test_run_with_tools_no_tool_calls(self):
        llm = MockLLM(["Final answer"])
        bound = BoundLLM(llm, [])
        result = bound.run_with_tools("system", "user")
        assert result == "Final answer"


class TestWorkflowContext:
    def test_serialization_roundtrip(self):
        from fsm_agent_flow.context import WorkflowContext, SharedContext, StateOutput

        ctx = WorkflowContext(
            objective="test",
            shared=SharedContext(data={"key": "value"}),
            history=[StateOutput(state_name="s1", output="out", key_results_met={"kr": True})],
        )
        d = ctx.to_dict()
        ctx2 = WorkflowContext.from_dict(d)
        assert ctx2.objective == "test"
        assert ctx2.shared.get("key") == "value"
        assert ctx2.history[0].state_name == "s1"
        assert ctx2.history[0].key_results_met["kr"] is True
