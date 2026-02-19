"""Tests for schema.py: JSON ↔ Workflow conversion and Python code generation."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from fsm_agent_flow import KeyResult, StateSpec, Workflow
from fsm_agent_flow.llm.adapter import LLMAdapter, LLMResponse, Message
from fsm_agent_flow.schema import (
    _expression_to_check,
    validate_workflow_json,
    workflow_from_json,
    workflow_to_json,
    workflow_to_python,
)
from fsm_agent_flow.tools import ToolSpec


# ── Fixtures ────────────────────────────────────────────────────────────


class FakeLLM:
    """Minimal LLM adapter for testing."""

    def chat(self, messages, *, tools=None, temperature=0.7, max_tokens=None):
        return LLMResponse(content="fake response")

    def format_tools(self, tools):
        return [t.to_openai_schema() for t in tools]


def _make_workflow(**overrides) -> Workflow:
    """Create a simple two-state workflow for testing."""

    def search_web(query: str) -> str:
        """Search the web."""
        return f"results for {query}"

    states = [
        StateSpec(
            name="research",
            objective="Gather information",
            key_results=[
                KeyResult("has_content", "At least 200 chars", check=lambda output: len(str(output)) >= 200),
                KeyResult("quality", "High quality content"),
            ],
            tools=[search_web],
            max_retries=2,
            is_initial=True,
        ),
        StateSpec(
            name="writing",
            objective="Write a report",
            key_results=[
                KeyResult("has_sections", "Has headings", check=lambda output: "#" in str(output)),
            ],
            max_retries=3,
            is_final=True,
        ),
    ]

    kwargs = dict(
        objective="Research and write a report",
        states=states,
        transitions={"research": "writing", "writing": None},
        llm=FakeLLM(),
    )
    kwargs.update(overrides)
    return Workflow(**kwargs)


# ── workflow_to_json tests ──────────────────────────────────────────────


class TestWorkflowToJson:
    def test_basic_structure(self):
        wf = _make_workflow()
        data = workflow_to_json(wf)

        assert data["version"] == "2.0"
        assert data["objective"] == "Research and write a report"
        assert "research" in data["states"]
        assert "writing" in data["states"]
        assert data["transitions"]["research"] == "writing"
        assert data["transitions"]["writing"] is None

    def test_state_properties(self):
        wf = _make_workflow()
        data = workflow_to_json(wf)

        research = data["states"]["research"]
        assert research["objective"] == "Gather information"
        assert research["max_retries"] == 2
        assert research["is_initial"] is True
        assert research["is_final"] is False
        assert "search_web" in research["tools"]

        writing = data["states"]["writing"]
        assert writing["is_final"] is True
        assert writing["is_initial"] is False

    def test_key_results_serialized(self):
        wf = _make_workflow()
        data = workflow_to_json(wf)

        krs = data["states"]["research"]["key_results"]
        assert len(krs) == 2
        assert krs[0]["name"] == "has_content"
        assert krs[0]["description"] == "At least 200 chars"
        # check may or may not have been extracted (depends on lambda source availability)
        assert krs[1]["name"] == "quality"
        assert krs[1]["check"] is None  # No check function for this KR

    def test_graph_layout_present(self):
        wf = _make_workflow()
        data = workflow_to_json(wf)
        assert "graph_layout" in data


# ── workflow_from_json tests ────────────────────────────────────────────


class TestWorkflowFromJson:
    def _sample_json(self) -> dict:
        return {
            "version": "2.0",
            "objective": "Test workflow",
            "states": {
                "start": {
                    "objective": "Begin",
                    "key_results": [
                        {"name": "has_output", "description": "Has some output", "check": "len(str(output)) > 0"},
                    ],
                    "tools": [],
                    "max_retries": 1,
                    "is_initial": True,
                    "is_final": False,
                    "execute_module": None,
                },
                "end": {
                    "objective": "Finish",
                    "key_results": [],
                    "tools": [],
                    "max_retries": 3,
                    "is_initial": False,
                    "is_final": True,
                    "execute_module": None,
                },
            },
            "transitions": {"start": "end", "end": None},
            "graph_layout": {},
        }

    def test_basic_load(self):
        data = self._sample_json()
        wf = workflow_from_json(data, llm=FakeLLM())

        assert wf._objective == "Test workflow"
        assert "start" in wf._states
        assert "end" in wf._states
        assert wf._states["start"].is_initial is True
        assert wf._states["end"].is_final is True

    def test_key_result_check_expression(self):
        data = self._sample_json()
        wf = workflow_from_json(data, llm=FakeLLM())

        kr = wf._states["start"].key_results[0]
        assert kr.name == "has_output"
        assert kr.check is not None
        assert kr.check("hello") is True
        assert kr.check("") is False

    def test_transitions(self):
        data = self._sample_json()
        wf = workflow_from_json(data, llm=FakeLLM())

        assert wf._transitions["start"] == "end"
        assert wf._transitions["end"] is None

    def test_with_tools(self):
        def my_tool(x: str) -> str:
            """A test tool."""
            return x

        data = self._sample_json()
        data["states"]["start"]["tools"] = ["my_tool"]
        wf = workflow_from_json(data, llm=FakeLLM(), tools={"my_tool": my_tool})

        assert len(wf._states["start"].tools) == 1

    def test_with_execute_fns(self):
        def custom_execute(ctx):
            return "custom"

        data = self._sample_json()
        wf = workflow_from_json(data, llm=FakeLLM(), execute_fns={"start": custom_execute})

        assert wf._states["start"].execute is custom_execute


# ── Roundtrip test ──────────────────────────────────────────────────────


class TestRoundtrip:
    def test_json_roundtrip_preserves_structure(self):
        """workflow -> JSON -> workflow should preserve key properties."""
        wf1 = _make_workflow()
        data = workflow_to_json(wf1)
        wf2 = workflow_from_json(data, llm=FakeLLM())

        assert wf2._objective == wf1._objective
        assert set(wf2._states.keys()) == set(wf1._states.keys())
        assert wf2._transitions == wf1._transitions

        for name in wf1._states:
            s1 = wf1._states[name]
            s2 = wf2._states[name]
            assert s2.objective == s1.objective
            assert s2.max_retries == s1.max_retries
            assert s2.is_initial == s1.is_initial
            assert s2.is_final == s1.is_final
            assert len(s2.key_results) == len(s1.key_results)


# ── workflow_to_python tests ────────────────────────────────────────────


class TestWorkflowToPython:
    def _sample_json(self) -> dict:
        return {
            "version": "2.0",
            "objective": "Demo workflow",
            "states": {
                "fetch": {
                    "objective": "Fetch data from API",
                    "key_results": [
                        {"name": "has_data", "description": "Got data", "check": "len(str(output)) > 10"},
                    ],
                    "tools": ["http_get"],
                    "max_retries": 2,
                    "is_initial": True,
                    "is_final": False,
                    "execute_module": None,
                },
                "process": {
                    "objective": "Process the data",
                    "key_results": [],
                    "tools": [],
                    "max_retries": 3,
                    "is_initial": False,
                    "is_final": True,
                    "execute_module": None,
                },
            },
            "transitions": {"fetch": "process", "process": None},
        }

    def test_generates_valid_python(self):
        data = self._sample_json()
        code = workflow_to_python(data)

        # Should be valid Python syntax
        compile(code, "workflow.py", "exec")

    def test_contains_imports(self):
        data = self._sample_json()
        code = workflow_to_python(data)

        assert "from fsm_agent_flow import" in code
        assert "Workflow" in code
        assert "StateSpec" in code
        assert "KeyResult" in code

    def test_contains_tool_stubs(self):
        data = self._sample_json()
        code = workflow_to_python(data)

        assert "def http_get(" in code
        assert "NotImplementedError" in code

    def test_contains_state_definitions(self):
        data = self._sample_json()
        code = workflow_to_python(data)

        assert 'name="fetch"' in code
        assert 'name="process"' in code
        assert "is_initial=True" in code
        assert "is_final=True" in code

    def test_contains_transitions(self):
        data = self._sample_json()
        code = workflow_to_python(data)

        assert '"fetch": "process"' in code
        assert '"process": None' in code

    def test_contains_main_block(self):
        data = self._sample_json()
        code = workflow_to_python(data)

        assert 'if __name__ == "__main__":' in code
        assert "wf.run(" in code


# ── _expression_to_check tests ──────────────────────────────────────────


class TestExpressionToCheck:
    def test_len_expression(self):
        check = _expression_to_check("len(str(output)) >= 200")
        assert check("x" * 200) is True
        assert check("short") is False

    def test_contains_expression(self):
        check = _expression_to_check("'#' in str(output)")
        assert check("# heading") is True
        assert check("no heading") is False

    def test_none_returns_none(self):
        assert _expression_to_check(None) is None

    def test_invalid_expression_raises(self):
        check = _expression_to_check("__import__('os').system('echo bad')")
        # Should fail because __import__ is not in safe builtins
        with pytest.raises(Exception):
            check("test")


# ── validate_workflow_json tests ────────────────────────────────────────


class TestValidateWorkflowJson:
    def test_valid_workflow(self):
        data = {
            "states": {
                "a": {"objective": "Do A", "is_initial": True, "is_final": False},
                "b": {"objective": "Do B", "is_initial": False, "is_final": True},
            },
            "transitions": {"a": "b", "b": None},
        }
        errors = validate_workflow_json(data)
        assert errors == []

    def test_missing_states(self):
        errors = validate_workflow_json({})
        assert any("Missing 'states'" in e for e in errors)

    def test_no_initial_state(self):
        data = {
            "states": {
                "a": {"objective": "Do A", "is_initial": False, "is_final": True},
            },
            "transitions": {"a": None},
        }
        errors = validate_workflow_json(data)
        assert any("initial" in e.lower() for e in errors)

    def test_no_final_state(self):
        data = {
            "states": {
                "a": {"objective": "Do A", "is_initial": True, "is_final": False},
            },
            "transitions": {"a": None},
        }
        errors = validate_workflow_json(data)
        assert any("final" in e.lower() for e in errors)

    def test_missing_objective(self):
        data = {
            "states": {
                "a": {"objective": "", "is_initial": True, "is_final": True},
            },
            "transitions": {"a": None},
        }
        errors = validate_workflow_json(data)
        assert any("missing an objective" in e for e in errors)

    def test_bad_transition_target(self):
        data = {
            "states": {
                "a": {"objective": "Do A", "is_initial": True, "is_final": False},
            },
            "transitions": {"a": "nonexistent"},
        }
        errors = validate_workflow_json(data)
        assert any("does not exist" in e for e in errors)

    def test_invalid_check_expression(self):
        data = {
            "states": {
                "a": {
                    "objective": "Do A",
                    "is_initial": True,
                    "is_final": True,
                    "key_results": [
                        {"name": "bad", "description": "Bad check", "check": "if else ("},
                    ],
                },
            },
            "transitions": {"a": None},
        }
        errors = validate_workflow_json(data)
        assert any("invalid check expression" in e for e in errors)
