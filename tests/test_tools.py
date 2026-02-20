"""Tests for the tool system."""

import pytest

from fsm_agent_flow.errors import ExecutionBreak
from fsm_agent_flow.tools import ToolRegistry, ToolSpec


def sample_tool(query: str, count: int = 5) -> str:
    """Search for something."""
    return f"Found {count} results for {query}"


def breaking_tool(reason: str) -> str:
    """A tool that breaks execution."""
    return f"Breaking: {reason}"


class TestToolSpec:
    def test_from_callable(self):
        spec = ToolSpec.from_callable(sample_tool)
        assert spec.name == "sample_tool"
        assert spec.description == "Search for something."
        assert spec.parameters["type"] == "object"
        assert "query" in spec.parameters["properties"]
        assert "count" in spec.parameters["properties"]
        assert "query" in spec.parameters["required"]
        assert "count" not in spec.parameters["required"]

    def test_custom_name_and_description(self):
        spec = ToolSpec.from_callable(sample_tool, name="search", description="Custom desc")
        assert spec.name == "search"
        assert spec.description == "Custom desc"

    def test_to_openai_schema(self):
        spec = ToolSpec.from_callable(sample_tool)
        schema = spec.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "sample_tool"
        assert "parameters" in schema["function"]

    def test_execute(self):
        spec = ToolSpec.from_callable(sample_tool)
        result = spec.execute({"query": "test", "count": 3})
        assert result == "Found 3 results for test"

    def test_breaking_tool_raises(self):
        spec = ToolSpec.from_callable(breaking_tool, is_breaking=True)
        with pytest.raises(ExecutionBreak) as exc_info:
            spec.execute({"reason": "need human"})
        assert exc_info.value.tool_name == "breaking_tool"


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        registry.register(sample_tool)
        spec = registry.get("sample_tool")
        assert spec is not None
        assert spec.name == "sample_tool"

    def test_instance_isolation(self):
        r1 = ToolRegistry()
        r2 = ToolRegistry()
        r1.register(sample_tool)
        assert r1.get("sample_tool") is not None
        assert r2.get("sample_tool") is None

    def test_execute_by_name(self):
        registry = ToolRegistry()
        registry.register(sample_tool)
        result = registry.execute("sample_tool", {"query": "hello"})
        assert "hello" in result

    def test_execute_missing_tool(self):
        registry = ToolRegistry()
        with pytest.raises(KeyError, match="not_found"):
            registry.execute("not_found", {})

    def test_list_tools(self):
        registry = ToolRegistry()
        registry.register(sample_tool)
        registry.register(breaking_tool, is_breaking=True)
        tools = registry.list_tools()
        assert len(tools) == 2

    def test_openai_schemas(self):
        registry = ToolRegistry()
        registry.register(sample_tool)
        schemas = registry.to_openai_schemas()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
