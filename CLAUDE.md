# fsm-agent-flow

TDD/OKR-driven agentic workflow framework. v0.3.0 — lightweight built-in FSM, no `python-statemachine` dependency.

## Architecture

```
fsm_agent_flow/
    __init__.py       # Public API re-exports
    errors.py         # ExecutionBreak, MaxRetriesExceeded, WorkflowError
    state.py          # StateSpec (name, objective, key_results, execute, tools)
                      # KeyResult (name, description, check)
    context.py        # SharedContext, ExecutionContext, StateOutput, WorkflowContext
    tools.py          # ToolSpec (JSON Schema from callables), ToolRegistry (instance-scoped)
    validation.py     # Validator protocol, RuleValidator, LLMValidator
    workflow.py       # BoundLLM (LLM + tools), Workflow (FSM engine + TDD loop)
    schema.py         # JSON ↔ Workflow converters, Python code generation
    ooda.py           # Built-in OODA-loop agent as nested Workflow
    llm/
        adapter.py    # LLMAdapter protocol, Message, ToolCall, LLMResponse
        openai.py     # OpenAIAdapter
        litellm.py    # LiteLLMAdapter
    editor/           # Visual graph editor (litegraph.js, stdlib HTTP server)
        __main__.py   # CLI: python -m fsm_agent_flow.editor
        server.py     # HTTP API server
        static/       # HTML/JS/CSS frontend
```

## Key Design Decisions

- **No global state**: ToolRegistry is instance-scoped, tools are declared per StateSpec
- **JSON Schema tools**: ToolSpec.to_openai_schema() produces OpenAI function-calling format
- **Single-call execution**: Workflow.run() or Workflow.step() — no dual-call pattern
- **TDD loop built in**: every state validates key results before advancing, retries with feedback on failure
- **Framework-agnostic states**: execute functions can call any LLM, agent SDK, or plain code
- **OODA agent dogfoods the framework**: it's a nested Workflow with 4 states
- **Conditional/bidirectional transitions**: transitions can be static (str), conditional (dict), or dynamic (callable) — states can branch, loop back, and route based on output

## Transition Types

```python
# Static: always go to the same next state
transitions = {"research": "writing", "writing": None}

# Conditional: dict with keys matched against output
transitions = {
    "check": {"need_data": "fetch", "ready": "respond", "default": "respond"},
    "fetch": {"retry": "fetch", "default": "check"},  # bidirectional!
    "respond": None,
}

# Dynamic: callable receives output, returns next state name or None
transitions = {
    "decide": lambda output: "a" if output.get("ok") else "b",
}
```

Dict transition resolution:
1. If output is a `dict` with `"_transition"` key → use its value as the lookup key
2. If output is a `str` matching a key → use it directly
3. Fall back to `"default"` key

## Execution Flow

```
Workflow.step(input)
  → Build ExecutionContext with BoundLLM (state's tools only)
  → Call state.execute(ctx) → output
  → Validator.validate(state, output, ctx) → ValidationResult
  → If failed and retries left: retry with ctx.feedback set
  → If passed: record StateOutput, resolve transition (static/conditional/dynamic), advance
  → If retries exhausted: raise MaxRetriesExceeded
```

## How to Implement an LLM Adapter

See @docs/claude/rules/adapters.md

## How to Define a Workflow

See @docs/claude/rules/workflows.md

## How to Write Validation

See @docs/claude/rules/validation.md

## How to Use Tools

See @docs/claude/rules/tools.md

## Running Tests

```bash
uv run python -m pytest tests/ -v
```

## Common Patterns

- `ctx.llm.run_with_tools(system_prompt, user_message)` — auto tool-call loop
- `ctx.llm.chat(messages)` — single LLM call, manual tool handling
- `ctx.shared.set(key, value)` / `ctx.shared.get(key)` — cross-state data
- `run_ooda(ctx, task, tools, max_cycles)` — nested OODA agent in a state
- `wf.context.to_dict()` / `WorkflowContext.from_dict(d)` — persistence
- `return {"_transition": "branch_name"}` — conditional routing in execute functions
- `workflow_to_json(wf)` / `workflow_from_json(data, llm=...)` — JSON interchange
- `workflow_to_python(data)` — generate executable .py from JSON
- `python -m fsm_agent_flow.editor` — launch visual graph editor
