# fsm-agent-flow
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/NewJerseyStyle/FSM-agent-flow)
![PyPI - Version](https://img.shields.io/pypi/v/FSM-agent-flow)

A TDD/OKR-driven workflow framework for LLM-powered applications. Each state declares an **objective** and **key results** that get validated before advancing — like running tests after writing code.

## Why

Most LLM workflow frameworks either give you too little structure (raw prompt chains) or too much (rigid agent frameworks). fsm-agent-flow sits in the middle:

- **States have acceptance criteria** — key results are checked before moving on
- **Failed states retry with feedback** — the validator tells the LLM what went wrong
- **The framework doesn't care what happens inside a state** — call an LLM, run a script, bridge to CrewAI, or nest another workflow
- **No global singletons** — tools are scoped per state, contexts are explicit
- **No heavy dependencies** — zero required runtime deps, bring your own LLM client

## Install

```bash
pip install fsm-agent-flow

# With LLM adapters
pip install fsm-agent-flow[openai]
pip install fsm-agent-flow[litellm]
pip install fsm-agent-flow[all]
```

## Quick Start

```python
from fsm_agent_flow import Workflow, StateSpec, KeyResult, ExecutionContext
from fsm_agent_flow.llm.openai import OpenAIAdapter

# Tools are just functions
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

# States declare what they must accomplish
research = StateSpec(
    name="research",
    objective="Gather information on the topic",
    key_results=[
        KeyResult("has_content", "At least 200 chars", check=lambda o: len(str(o)) > 200),
        KeyResult("has_sources", "Cites sources"),  # LLM-validated (no check function)
    ],
    execute=lambda ctx: ctx.llm.run_with_tools(
        system_prompt="Research the topic using the search tool.",
        user_message=ctx.input,
    ),
    tools=[search],
    max_retries=2,
    is_initial=True,
)

writing = StateSpec(
    name="writing",
    objective="Write a structured report",
    key_results=[
        KeyResult("has_sections", "Has clear sections", check=lambda o: str(o).count("#") >= 2),
    ],
    execute=lambda ctx: ctx.llm.run_with_tools(
        system_prompt="Write a report from this research.",
        user_message=str(ctx.input),
    ),
    is_final=True,
)

# One call to run the whole workflow
llm = OpenAIAdapter(model="gpt-4o")
wf = Workflow(
    objective="Research and report",
    states=[research, writing],
    transitions={"research": "writing"},
    llm=llm,
    validator_llm=llm,
)
result = wf.run("quantum computing")
```

## Core Concepts

### States with Objectives and Key Results

Every state has an **objective** (what it does) and **key results** (how we verify it succeeded):

```python
StateSpec(
    name="analyze",
    objective="Analyze the dataset and identify trends",
    key_results=[
        # Programmatic check — runs as code
        KeyResult("has_trends", "Identified at least 3 trends",
                  check=lambda o: len(o.get("trends", [])) >= 3),
        # LLM-validated — no check function, validator LLM evaluates
        KeyResult("actionable", "Insights are actionable with recommendations"),
    ],
    execute=my_analyze_function,
    max_retries=3,
)
```

### The TDD Validation Loop

When a state executes, the framework:

1. Calls `state.execute(ctx)` to produce output
2. Runs all key result checks (programmatic first, then LLM)
3. If any fail: retries with `ctx.feedback` explaining what went wrong
4. If all pass: records the output and advances to the next state
5. If retries exhausted: raises `MaxRetriesExceeded`

### Conditional & Bidirectional Transitions

Transitions aren't limited to simple linear flows. States can branch, loop back, and route conditionally — like a real finite state machine:

```python
# Static (linear): always goes to the same next state
transitions = {"research": "writing", "writing": None}

# Conditional (branching / bidirectional): route based on output
transitions = {
    "check_city": {"need_weather": "get_weather", "ready": "print_result"},
    "get_weather": {"wrong_city": "get_weather", "default": "check_city"},
    "print_result": None,
}

# Dynamic (callable): function decides next state
transitions = {
    "decide": lambda output: "approve" if output.get("score") > 0.8 else "reject",
    "approve": None,
    "reject": None,
}
```

**How conditional routing works:** When a transition is a `dict`, the framework resolves the next state by checking the execute function's output:

1. If output is a `dict` with a `"_transition"` key, its value selects the branch
2. If output is a `str` matching a key in the transition dict, use it
3. Otherwise, fall back to the `"default"` key

```python
def check_city(ctx: ExecutionContext):
    weather = ctx.shared.get("weather")
    if weather and weather["city"] == ctx.shared.get("target_city"):
        return {"_transition": "ready", "report": weather}
    return {"_transition": "need_weather"}

def get_weather(ctx: ExecutionContext):
    city = ctx.shared.get("target_city")
    data = fetch_weather_api(city)
    if data["city"] != city:
        return {"_transition": "wrong_city"}  # Loop back to retry
    ctx.shared.set("weather", data)
    return {"_transition": "default"}  # Return to check_city
```

This enables bidirectional flows (state A calls state B, B returns to A), retry loops, and decision branching — all without leaving the FSM model.

### Tools Are Scoped Per State

No global registry. Each state declares its own tools:

```python
research_state = StateSpec(
    name="research",
    tools=[search_web, fetch_paper],  # Only available in this state
    ...
)
writing_state = StateSpec(
    name="writing",
    tools=[save_draft],  # Different tools here
    ...
)
```

Tools are plain Python functions. The framework auto-generates JSON Schema signatures (OpenAI/Anthropic compatible) from type hints:

```python
def search_web(query: str, max_results: int = 10) -> str:
    """Search the web for information."""
    ...
```

### Shared Context

States share data through `SharedContext` (explicit key-value store, not a flat blob):

```python
def step_one(ctx: ExecutionContext):
    ctx.shared.set("findings", ["a", "b", "c"])
    return "done"

def step_two(ctx: ExecutionContext):
    findings = ctx.shared.get("findings", [])
    return f"Processing {len(findings)} findings"
```

### Execute Functions

A state's `execute` function receives an `ExecutionContext` with everything it needs:

```python
def my_state(ctx: ExecutionContext):
    ctx.input       # Output from previous state
    ctx.shared      # SharedContext (read/write)
    ctx.history     # Previous states' outputs (read-only)
    ctx.llm         # BoundLLM with this state's tools
    ctx.retry_count # Current retry attempt
    ctx.feedback    # Validator feedback from last failed attempt
```

Inside execute, you can do anything:

```python
# Option A: Use the BoundLLM tool-calling loop
result = ctx.llm.run_with_tools(system_prompt="...", user_message="...")

# Option B: Call the LLM directly (no tool loop)
response = ctx.llm.chat([Message(role="user", content="...")])

# Option C: Bridge to an external agent framework
from crewai import Agent
result = Agent(...).run(ctx.input)

# Option D: Run arbitrary code
result = my_analysis_pipeline(ctx.input)

# Option E: Nest another workflow
inner_wf = Workflow(...)
result = inner_wf.run(ctx.input)
```

### Built-in OODA Agent

For "LLM + tools" without wiring your own agent loop, use the built-in OODA agent:

```python
from fsm_agent_flow import run_ooda

def investigate(ctx: ExecutionContext):
    return run_ooda(ctx, task=f"Investigate: {ctx.input}",
                    tools=[search, analyze], max_cycles=3)
```

The OODA agent is itself a nested `Workflow` with 4 states (Observe, Orient, Decide, Act), dogfooding the framework.

### Validators

Three options for validation:

```python
# 1. RuleValidator (default) — only runs programmatic checks
from fsm_agent_flow import RuleValidator
wf = Workflow(..., validator=RuleValidator())

# 2. LLMValidator — runs checks + asks LLM for KRs without check functions
from fsm_agent_flow import LLMValidator
wf = Workflow(..., validator=LLMValidator(llm))

# 3. Shorthand — pass validator_llm to auto-create LLMValidator
wf = Workflow(..., validator_llm=cheap_llm)

# 4. Custom — implement the Validator protocol
class MyValidator:
    def validate(self, state, output, context) -> ValidationResult:
        ...
```

### LLM Adapters

The framework ships with OpenAI and LiteLLM adapters:

```python
from fsm_agent_flow.llm.openai import OpenAIAdapter
from fsm_agent_flow.llm.litellm import LiteLLMAdapter

# OpenAI (or any OpenAI-compatible API)
llm = OpenAIAdapter(model="gpt-4o", api_key="sk-...")
llm = OpenAIAdapter(model="deepseek/deepseek-r1", base_url="https://openrouter.ai/api/v1")

# LiteLLM (any provider)
llm = LiteLLMAdapter(model="anthropic/claude-sonnet-4-20250514")
```

Build your own by implementing the `LLMAdapter` protocol — see `docs/claude/rules/adapters.md` or ask Claude Code.

### Persistence

`WorkflowContext` is serializable for save/resume:

```python
# Save
data = wf.context.to_dict()
json.dump(data, open("checkpoint.json", "w"))

# Resume
data = json.load(open("checkpoint.json"))
ctx = WorkflowContext.from_dict(data)
```

## Examples

See `examples/` for complete working examples:

- **`research_workflow.py`** — Research + writing with tool calling and TDD validation
- **`ooda_example.py`** — Using the built-in OODA agent inside workflow states

## Claude Code Integration

This repo includes a `CLAUDE.md` and `docs/claude/rules/` that teach Claude Code the framework's architecture. When you open this project in Claude Code, it automatically understands how to:

- Define workflows with states, transitions, and key results
- Build custom LLM adapters
- Write validation logic
- Use the OODA agent
- Debug common issues

### Using with Claude Code in your own project

If you're using `fsm-agent-flow` as a dependency in your own project, add the following to your project's `CLAUDE.md` so Claude Code understands the framework:

```markdown
# fsm-agent-flow

TDD/OKR-driven agentic workflow framework. See the reference docs:

@https://raw.githubusercontent.com/NewJerseyStyle/FSM-agent-flow/main/CLAUDE.md
@https://NewJerseyStyle.github.io/FSM-agent-flow/claude/rules/adapters.md
@https://NewJerseyStyle.github.io/FSM-agent-flow/claude/rules/workflows.md
@https://NewJerseyStyle.github.io/FSM-agent-flow/claude/rules/validation.md
@https://NewJerseyStyle.github.io/FSM-agent-flow/claude/rules/tools.md</pre>
```

This gives Claude Code full knowledge of the framework's API, patterns, and conventions when working on your codebase.

## License

MIT

