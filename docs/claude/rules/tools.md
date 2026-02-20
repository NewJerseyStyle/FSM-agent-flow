# Tools

## Defining Tools

Tools are plain Python functions. The framework inspects their signature and docstring to build JSON Schema automatically.

```python
def search_web(query: str, max_results: int = 10) -> str:
    """Search the web for information on a topic."""
    return api_call(query, max_results)
```

This produces the OpenAI-compatible schema:

```json
{
  "type": "function",
  "function": {
    "name": "search_web",
    "description": "Search the web for information on a topic.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {"type": "string", "description": "query"},
        "max_results": {"type": "integer", "description": "max_results"}
      },
      "required": ["query"]
    }
  }
}
```

## Assigning Tools to States

Tools are listed in the `StateSpec.tools` field:

```python
StateSpec(
    name="research",
    tools=[search_web, fetch_paper, calculate],  # Only these tools in this state
    ...
)
```

Tools from one state are NOT available in another state. This is by design.

## Type Mapping

| Python type | JSON Schema type |
|-------------|-----------------|
| `str`       | `string`        |
| `int`       | `integer`       |
| `float`     | `number`        |
| `bool`      | `boolean`       |
| `list`      | `array`         |
| `dict`      | `object`        |
| (missing)   | `string`        |

## ToolSpec for Advanced Control

For more control, create a `ToolSpec` directly:

```python
from fsm_agent_flow import ToolSpec

spec = ToolSpec(
    name="custom_search",
    description="Search with custom parameters",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "filters": {
                "type": "object",
                "description": "Filter criteria",
                "properties": {
                    "date_from": {"type": "string"},
                    "language": {"type": "string"},
                },
            },
        },
        "required": ["query"],
    },
    func=my_search_function,
)

StateSpec(name="s", tools=[spec], ...)
```

## Breaking Tools

A breaking tool pauses workflow execution (useful for human-in-the-loop):

```python
spec = ToolSpec.from_callable(wait_for_human, is_breaking=True)
# Or
StateSpec(tools=[spec], ...)
```

When the LLM calls a breaking tool, `ExecutionBreak` is raised. Catch it in the caller:

```python
try:
    wf.run(input)
except ExecutionBreak as e:
    save_state(wf.context.to_dict())
    print(f"Waiting: {e.tool_name}")
```

## ToolRegistry (Direct Use)

For programmatic tool management:

```python
from fsm_agent_flow import ToolRegistry

registry = ToolRegistry()
registry.register(search_web, name="search", description="Custom desc")
spec = registry.get("search")
result = registry.execute("search", {"query": "test"})
schemas = registry.to_openai_schemas()
```

Each `ToolRegistry` is independent (no global singleton).

## How BoundLLM Uses Tools

When `Workflow.step()` executes a state:

1. It builds `ToolSpec` from each callable in `state.tools`
2. Creates a `BoundLLM` with those specs
3. `BoundLLM.chat()` auto-includes formatted tool schemas
4. `BoundLLM.run_with_tools()` handles the full loop:
   - Calls LLM with tools
   - If LLM returns tool_calls: executes them, appends results, calls LLM again
   - Repeats until LLM returns text-only (no tool calls)
   - Returns the final text content
