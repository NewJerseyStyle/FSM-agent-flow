"""Tests for the OODA agent."""

from fsm_agent_flow.llm.adapter import LLMResponse, Message
from fsm_agent_flow.ooda import create_ooda_agent
from fsm_agent_flow.tools import ToolSpec


class MockLLM:
    """Mock LLM that returns deterministic responses for OODA states."""

    def __init__(self):
        self._call_count = 0

    def chat(self, messages, *, tools=None, temperature=0.7, max_tokens=None):
        self._call_count += 1
        # Return different content based on conversation context
        last_user = ""
        for m in reversed(messages):
            if m.role == "user":
                last_user = m.content or ""
                break

        if "OBSERVE" in last_user:
            content = "Observations:\n- Key finding 1\n- Key finding 2"
        elif "Analyze" in last_user:
            content = "Analysis: The situation requires action X because of factors A and B."
        elif "decide" in last_user.lower():
            content = "Decision: Execute action X with parameters Y."
        elif "Execute" in last_user:
            content = "TASK COMPLETE: Action X executed successfully. Result: Z."
        else:
            content = f"Response {self._call_count}"

        return LLMResponse(content=content, finish_reason="stop")

    def format_tools(self, tools):
        return [t.to_openai_schema() for t in tools]


def dummy_tool(query: str) -> str:
    """A dummy tool."""
    return f"result for {query}"


class TestOODAAgent:
    def test_creates_4_state_workflow(self):
        agent = create_ooda_agent(MockLLM(), [dummy_tool], "test task")
        assert "observe" in agent._states
        assert "orient" in agent._states
        assert "decide" in agent._states
        assert "act" in agent._states

    def test_runs_to_completion(self):
        agent = create_ooda_agent(MockLLM(), [dummy_tool], "test task", max_cycles=1)
        result = agent.run("initial input")
        assert agent.is_finished
        assert len(result.history) == 4  # observe, orient, decide, act

    def test_act_has_tools(self):
        agent = create_ooda_agent(MockLLM(), [dummy_tool], "test task")
        act_state = agent._states["act"]
        assert len(act_state.tools) == 1
