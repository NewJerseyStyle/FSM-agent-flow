# Building LLM Adapters

## The LLMAdapter Protocol

Any LLM adapter must implement two methods:

```python
class LLMAdapter(Protocol):
    def chat(self, messages: list[Message], *, tools=None, temperature=0.7, max_tokens=None) -> LLMResponse: ...
    def format_tools(self, tools: list[ToolSpec]) -> list[dict]: ...
```

## Data Types

```python
# Input message
Message(role="system"|"user"|"assistant"|"tool", content=str|None,
        tool_calls=list[ToolCall]|None, tool_call_id=str|None, name=str|None)

# Tool call from LLM
ToolCall(id=str, name=str, arguments=dict)

# LLM response
LLMResponse(content=str|None, tool_calls=list[ToolCall], finish_reason=str|None, usage=dict)
```

## Implementing a Custom Adapter

### Step 1: Create the adapter class

```python
from fsm_agent_flow.llm.adapter import LLMAdapter, LLMResponse, Message, ToolCall
from fsm_agent_flow.tools import ToolSpec

class MyAdapter:
    def __init__(self, model: str, **kwargs):
        self.model = model
        # Initialize your client

    def format_tools(self, tools: list[ToolSpec]) -> list[dict]:
        # Convert ToolSpec list to your provider's format
        # For OpenAI-compatible APIs, just use:
        return [t.to_openai_schema() for t in tools]

    def chat(self, messages, *, tools=None, temperature=0.7, max_tokens=None) -> LLMResponse:
        # 1. Convert Message objects to your provider's format
        # 2. Call the API
        # 3. Parse response into LLMResponse with ToolCall objects
        ...
```

### Step 2: Handle tool calls in the response

The critical part is parsing tool calls from the API response into `ToolCall` objects:

```python
tool_calls = []
if raw_response.has_tool_calls:
    for tc in raw_response.tool_calls:
        tool_calls.append(ToolCall(
            id=tc.id,           # String ID for correlation
            name=tc.name,       # Must match a registered tool name
            arguments=tc.args,  # Dict of argument name -> value
        ))
```

### Step 3: Convert messages

Convert `Message` objects to your provider's format. Key fields:
- `role`: "system", "user", "assistant", "tool"
- `content`: text content (can be None for tool-call-only assistant messages)
- `tool_calls`: list of ToolCall (only on assistant messages)
- `tool_call_id`: correlation ID (only on tool result messages)

## Provider-Specific Notes

### Anthropic API

Anthropic uses a different tool format. Your `format_tools()` should convert:

```python
def format_tools(self, tools: list[ToolSpec]) -> list[dict]:
    return [{
        "name": t.name,
        "description": t.description,
        "input_schema": t.parameters,  # JSON Schema, same as OpenAI
    } for t in tools]
```

### OpenAI-Compatible APIs (OpenRouter, Ollama, vLLM, etc.)

Use `OpenAIAdapter` directly with `base_url`:

```python
from fsm_agent_flow.llm.openai import OpenAIAdapter
llm = OpenAIAdapter(model="model-name", api_key="...", base_url="https://your-api/v1")
```

### Non-Chat APIs (Completion, Custom)

If your LLM doesn't support chat format, build the conversion in `chat()`:
- Flatten messages into a single prompt string
- Parse tool calls from the text output (regex or structured output)
- Return a standard `LLMResponse`

## Testing Your Adapter

```python
from fsm_agent_flow import Workflow, StateSpec
from your_adapter import MyAdapter

def test_adapter_works():
    llm = MyAdapter(model="test")
    state = StateSpec(name="s", objective="test", is_initial=True, is_final=True,
                      execute=lambda ctx: ctx.llm.run_with_tools("system", "hello"))
    wf = Workflow(objective="test", states=[state], transitions={"s": None}, llm=llm)
    result = wf.run()
    assert result.history[0].output
```
