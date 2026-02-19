# Defining Workflows

## Minimal Workflow

```python
from fsm_agent_flow import Workflow, StateSpec, ExecutionContext

state = StateSpec(
    name="greet",
    objective="Greet the user",
    execute=lambda ctx: f"Hello, {ctx.input}!",
    is_initial=True,
    is_final=True,
)

wf = Workflow(
    objective="Greeting workflow",
    states=[state],
    transitions={"greet": None},  # None = no next state (terminal)
    llm=my_llm,  # Required even if states don't use LLM
)
result = wf.run("world")
# result.history[0].output == "Hello, world!"
```

## Multi-State Workflow

```python
states = [
    StateSpec(name="fetch", objective="Fetch data", execute=fetch_fn, is_initial=True),
    StateSpec(name="process", objective="Process data", execute=process_fn),
    StateSpec(name="output", objective="Format output", execute=output_fn, is_final=True),
]

transitions = {
    "fetch": "process",
    "process": "output",
    "output": None,
}
```

Transitions are a dict: `{current_state: next_state}`. Set `next_state` to `None` or mark the state `is_final=True` to end the workflow.

## Conditional & Bidirectional Transitions

Transitions support three forms:

### Static (linear)

```python
transitions = {"fetch": "process", "process": None}
```

### Conditional (branching / bidirectional)

Use a `dict` as the transition value. Keys are condition labels, values are target states:

```python
transitions = {
    "check_city": {"need_weather": "get_weather", "ready": "print_result"},
    "get_weather": {"wrong_city": "get_weather", "default": "check_city"},
    "print_result": None,
}
```

The framework resolves which branch to take by examining the execute function's output:

1. If output is a `dict` with `"_transition"` key → use its value as the lookup key
2. If output is a `str` matching a key in the transition dict → use it directly
3. Fall back to `"default"` key
4. If nothing matches → raise `WorkflowError`

```python
def check_city(ctx: ExecutionContext):
    weather = ctx.shared.get("weather")
    if weather and weather["city"] == "Tokyo":
        return {"_transition": "ready", "report": weather}
    return {"_transition": "need_weather"}

def get_weather(ctx: ExecutionContext):
    data = call_weather_api(ctx.shared.get("target_city"))
    if data["city"] != ctx.shared.get("target_city"):
        return {"_transition": "wrong_city"}  # Loop back to self
    ctx.shared.set("weather", data)
    return {"_transition": "default"}  # Return to check_city
```

This enables bidirectional flows (state A calls state B, B returns to A), retry loops within the graph, and multi-way branching.

### Dynamic (callable)

Use a function that receives the output and returns the next state name (or `None` to finish):

```python
transitions = {
    "decide": lambda output: "approve" if output.get("score") > 0.8 else "reject",
    "approve": None,
    "reject": None,
}
```

Note: callable transitions are not JSON-serializable (they only work in Python code, not in the visual editor's JSON format).

## State Execute Functions

Every execute function receives an `ExecutionContext`:

```python
def my_state(ctx: ExecutionContext):
    # Available attributes:
    ctx.input         # Output from previous state (or initial_input for first state)
    ctx.shared        # SharedContext — key-value store shared across all states
    ctx.history       # list[StateOutput] — previous states' recorded outputs
    ctx.llm           # BoundLLM — LLM adapter with this state's tools pre-bound
    ctx.retry_count   # int — current retry attempt (0 on first try)
    ctx.feedback      # str|None — why the last attempt failed (from validator)

    # Return the state's output (any type)
    return result
```

## Pass-Through States

If `execute` is `None`, the state passes its input through as output:

```python
StateSpec(name="passthrough", objective="Forward data", execute=None, ...)
```

## Using LLM Inside States

### Tool-calling loop (most common)

```python
def research(ctx: ExecutionContext):
    return ctx.llm.run_with_tools(
        system_prompt="You are a researcher. Use tools to find information.",
        user_message=f"Research: {ctx.input}",
        max_iterations=10,
        temperature=0.7,
    )
```

### Single LLM call (no tools)

```python
from fsm_agent_flow import Message

def summarize(ctx: ExecutionContext):
    response = ctx.llm.chat([
        Message(role="system", content="Summarize concisely."),
        Message(role="user", content=str(ctx.input)),
    ])
    return response.content
```

## Handling Retries

When validation fails, the state re-executes with `ctx.feedback` set:

```python
def my_state(ctx: ExecutionContext):
    prompt = f"Task: {ctx.input}"
    if ctx.feedback:
        prompt += f"\n\nYour previous attempt failed: {ctx.feedback}\nPlease fix the issues."
    return ctx.llm.run_with_tools("system prompt", prompt)
```

## Sharing Data Between States

Use `ctx.shared` (not instance variables or globals):

```python
def state_a(ctx: ExecutionContext):
    ctx.shared.set("urls", ["https://example.com"])
    return "fetched"

def state_b(ctx: ExecutionContext):
    urls = ctx.shared.get("urls", [])  # Available here
    return f"Processing {len(urls)} URLs"
```

## Nesting Workflows

A state can run another workflow internally:

```python
def complex_state(ctx: ExecutionContext):
    inner_wf = Workflow(
        objective="Sub-task",
        states=[...],
        transitions={...},
        llm=ctx.llm.adapter,
    )
    inner_result = inner_wf.run(ctx.input)
    return inner_result.history[-1].output
```

## Workflow.step() vs Workflow.run()

- `wf.run(initial_input)` — runs all states from start to finish, returns `WorkflowContext`
- `wf.step(input)` — runs one state, validates, advances. Call repeatedly for manual control.

```python
wf = Workflow(...)
# Manual stepping
output1 = wf.step("initial input")
output2 = wf.step(output1)  # Input for next state
# Check state
print(wf.current_state, wf.is_finished)
```

## Error Handling

```python
from fsm_agent_flow import MaxRetriesExceeded, ExecutionBreak, WorkflowError

try:
    result = wf.run(input)
except MaxRetriesExceeded as e:
    print(f"State '{e.state_name}' failed after {e.max_retries} retries")
    print(f"Last feedback: {e.feedback}")
except ExecutionBreak as e:
    print(f"Paused by breaking tool: {e.tool_name}")
except WorkflowError as e:
    print(f"Workflow error: {e}")
```
