# Validation

## How Validation Works

Every state can declare **key results** — acceptance criteria checked after execution:

```python
StateSpec(
    name="research",
    objective="Gather comprehensive information",
    key_results=[
        # Programmatic check (runs as code, fast, deterministic)
        KeyResult("length", "At least 500 chars", check=lambda o: len(str(o)) >= 500),
        # LLM-validated (no check function — asks the validator LLM)
        KeyResult("quality", "Information is accurate and well-sourced"),
    ],
    max_retries=3,
    ...
)
```

## Validation Flow

```
State executes → output produced
  → RuleValidator or LLMValidator checks key results
  → All pass? → Record output, advance to next state
  → Any fail? → Set ctx.feedback = failure reasons, retry (up to max_retries)
  → Retries exhausted? → Raise MaxRetriesExceeded
```

## Validator Types

### RuleValidator (default)

Only runs `KeyResult.check` functions. KRs without a `check` auto-pass.

```python
from fsm_agent_flow import RuleValidator
wf = Workflow(..., validator=RuleValidator())
```

Use when: all key results have programmatic checks, or you don't want LLM validation.

### LLMValidator

Runs programmatic checks first, then asks an LLM to evaluate remaining KRs.

```python
from fsm_agent_flow import LLMValidator
wf = Workflow(..., validator=LLMValidator(llm))
# Or shorthand:
wf = Workflow(..., validator_llm=llm)
```

The LLM receives the state objective, output, and KR descriptions. It returns JSON with per-KR pass/fail.

Use when: some key results need subjective evaluation (quality, completeness, accuracy).

### Custom Validator

Implement the `Validator` protocol:

```python
from fsm_agent_flow import ValidationResult

class MyValidator:
    def validate(self, state, output, context) -> ValidationResult:
        # state: StateSpec
        # output: whatever the execute function returned
        # context: ExecutionContext
        passed = my_custom_logic(output)
        return ValidationResult(
            passed=passed,
            feedback="What went wrong" if not passed else None,
            key_results={"kr_name": True, ...},
        )
```

## Writing Good Key Results

### Programmatic checks (preferred when possible)

```python
# Length/size checks
KeyResult("min_length", "At least 200 chars", check=lambda o: len(str(o)) >= 200)

# Structure checks
KeyResult("has_sections", "Has headings", check=lambda o: "#" in str(o))

# Type checks
KeyResult("is_dict", "Returns a dict", check=lambda o: isinstance(o, dict))

# Content checks
KeyResult("has_key", "Has 'results' key", check=lambda o: "results" in o)

# Count checks
KeyResult("enough_items", "At least 3 items", check=lambda o: len(o.get("items", [])) >= 3)
```

### LLM-validated checks (for subjective criteria)

```python
# Quality
KeyResult("well_written", "Clear, professional prose with no grammatical errors")

# Completeness
KeyResult("comprehensive", "Covers all major aspects of the topic")

# Accuracy
KeyResult("factual", "Claims are supported by cited sources")

# Relevance
KeyResult("on_topic", "Directly addresses the original question")
```

## Feedback Loop

When validation fails, the feedback string is passed to the next retry via `ctx.feedback`:

```python
def my_state(ctx: ExecutionContext):
    prompt = f"Task: {ctx.input}"
    if ctx.feedback:
        prompt += f"\n\nPrevious attempt failed validation:\n{ctx.feedback}\nPlease address these issues."
    return ctx.llm.run_with_tools("system prompt", prompt)
```

The feedback contains semicolon-separated failure reasons, e.g.:
`"min_length: At least 200 chars; has_sources: Cites at least 3 sources"`
