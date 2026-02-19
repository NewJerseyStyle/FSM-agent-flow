# fsm-agent-flow

TDD/OKR-driven agentic workflow framework. v2.0 — lightweight built-in FSM, no `python-statemachine` dependency.

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
    ooda.py           # Built-in OODA-loop agent as nested Workflow
    llm/
        adapter.py    # LLMAdapter protocol, Message, ToolCall, LLMResponse
        openai.py     # OpenAIAdapter
        litellm.py    # LiteLLMAdapter
```

## Key Design Decisions

- **No global state**: ToolRegistry is instance-scoped, tools are declared per StateSpec
- **JSON Schema tools**: ToolSpec.to_openai_schema() produces OpenAI function-calling format
- **Single-call execution**: Workflow.run() or Workflow.step() — no dual-call pattern
- **TDD loop built in**: every state validates key results before advancing, retries with feedback on failure
- **Framework-agnostic states**: execute functions can call any LLM, agent SDK, or plain code
- **OODA agent dogfoods the framework**: it's a nested Workflow with 4 states

## Execution Flow

```
Workflow.step(input)
  → Build ExecutionContext with BoundLLM (state's tools only)
  → Call state.execute(ctx) → output
  → Validator.validate(state, output, ctx) → ValidationResult
  → If failed and retries left: retry with ctx.feedback set
  → If passed: record StateOutput, advance to next state via transitions dict
  → If retries exhausted: raise MaxRetriesExceeded
```

## How to Implement an LLM Adapter

See @.claude/rules/adapters.md

## How to Define a Workflow

See @.claude/rules/workflows.md

## How to Write Validation

See @.claude/rules/validation.md

## How to Use Tools

See @.claude/rules/tools.md

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
