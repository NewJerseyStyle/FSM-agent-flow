# LLM-FSM Quick Reference

## Installation

```bash
pip install llm-fsm[all]  # All integrations
pip install llm-fsm[openai]  # OpenAI only
pip install llm-fsm[litellm]  # LiteLLM only
pip install llm-fsm[smolagents]  # smolagents only
```

## Minimal Example

```python
from statemachine import State
from llm_fsm import LLMStateMachine, create_llm_client, Message

class SimpleWorkflow(LLMStateMachine):
    start = State(initial=True)
    process = State()
    end = State(final=True)
    
    go = start.to(process)
    finish = process.to(end)
    
    def on_enter_process(self, state_input=None):
        messages = [Message(role="user", content=str(state_input))]
        return self.llm_client.chat(messages).content

client = create_llm_client("openai", model="gpt-4", api_key="...")
workflow = SimpleWorkflow(llm_client=client)
workflow.go()
result = workflow.execute_state("process", state_input="Hello!")
```

## Common Imports

```python
from statemachine import State
from llm_fsm import (
    LLMStateMachine,
    ExecutionBreak,
    PersistentMemory,
    create_llm_client,
    register_tool,
    ToolType,
    Message,
)
```

## State Machine Definition

```python
class MyWorkflow(LLMStateMachine):
    # Define states
    state1 = State(initial=True)
    state2 = State()
    state3 = State(final=True)
    
    # Define transitions
    trans1 = state1.to(state2)
    trans2 = state2.to(state3)
    
    # Tool use is handled internally within state methods
    # No explicit self-transitions needed
    
    # Conditional transitions
    branch1 = state1.to(state2, cond="condition_check")
    branch2 = state1.to(state3, cond="other_condition")
    
    def condition_check(self):
        return self._some_condition
    
    # State implementation
    def on_enter_state2(self, state_input=None):
        # Tool use handled by run_llm_with_tools()
        return self.run_llm_with_tools(
            system_prompt="...",
            user_message="...",
            state_name="state2",
            max_iterations=10  # Internal tool loop
        )
```

## LLM Client Creation

```python
# OpenAI
client = create_llm_client("openai", model="gpt-4", api_key="sk-...")

# LiteLLM (multiple providers)
client = create_llm_client("litellm", model="gpt-4")
client = create_llm_client("litellm", model="claude-3-opus-20240229")
client = create_llm_client("litellm", model="ollama/llama2")

# smolagents
from smolagents import HfApiModel
model = HfApiModel()
client = create_llm_client("smolagents", agent_or_model=model)

# Google ADK 🆕
from llm_fsm.adk_client import create_adk_agent_with_tools

def my_tool(param: str) -> str:
    """Tool description."""
    return f"Result: {param}"

client = create_adk_agent_with_tools(
    model="gemini-2.0-flash",
    name="my_agent",
    instruction="Agent instructions",
    tools=[my_tool]
)
```

## Using Google ADK in States 🆕

```python
from llm_fsm.adk_client import ADKStateMixin

class MyWorkflow(LLMStateMachine, ADKStateMixin):
    process = State(initial=True)
    done = State(final=True)
    
    finish = process.to(done)
    
    def on_enter_process(self, state_input=None):
        # ADK handles tool orchestration internally
        return self.run_with_adk(
            query=f"Process: {state_input}",
            state_name="process"
        )

# ADK benefits:
# - Native tool orchestration
# - Built-in pause/resume
# - HITL tool confirmation
# - Context compaction
# - Multi-agent support
```

## Working Memory

```python
# In state methods
self.memory.working_memory.set(0, "content")
self.memory.working_memory.append(1, "more")
content = self.memory.working_memory.get(2)
self.memory.working_memory.delete(3)

# Non-empty buckets
buckets = self.memory.working_memory.get_non_empty()  # {index: content}
```

## Background Context

```python
self.memory.background.vision_mission = "Overall mission"
self.memory.background.goals = "Current goals"
self.memory.background.todo_list = "Distilled TODO"
self.memory.background.custom_fields["key"] = "value"
```

## Tool Registration

```python
# Basic tool
@register_tool(description="Tool description")
def my_tool(param: str) -> str:
    return f"Result: {param}"

# State-scoped tool
@register_tool(description="Research tool", state_scope="research")
def search(query: str) -> str:
    return do_search(query)

# Breaking tool (pauses execution)
@register_tool(description="Wait for human", tool_type=ToolType.BREAKING)
def wait_for_human(prompt: str) -> str:
    print(f"Waiting: {prompt}")
    return "[WAITING]"
```

## Running LLM with Tools

```python
def on_enter_my_state(self, state_input=None):
    system_prompt = f"""Your task.
    
{self.memory.to_context()}

Use available tools to complete the task."""
    
    try:
        result = self.run_llm_with_tools(
            system_prompt=system_prompt,
            user_message="Begin",
            state_name="my_state",
            max_iterations=10
        )
        return result
    except ExecutionBreak:
        # Breaking tool used
        raise
```

## Validation & Refinement

```python
def on_enter_my_state(self, state_input=None):
    def execute(refinement_advice=None):
        prompt = f"Do task. {refinement_advice or ''}"
        return self.llm_client.chat([Message(role="user", content=prompt)]).content
    
    return self.run_with_refinement(
        state_name="my_state",
        objective="Objective with success criteria",
        execution_func=execute,
        max_retries=3,
        throw_on_failure=False  # Continue with sub-optimal
    )
```

## Breaking Execution

```python
# Execute with breaking tool
try:
    result = workflow.execute_state("state", state_input=data)
except ExecutionBreak:
    # Save state
    workflow.memory.save_to_file("paused.json")
    print("Paused")

# Resume later
memory = PersistentMemory.load_from_file("paused.json")
memory.working_memory.set(5, "Human feedback: ...")
workflow = MyWorkflow(llm_client=client, memory=memory)
result = workflow.execute_state("state", state_input=new_data)
```

## Persistence

```python
# File
workflow.memory.save_to_file("state.json")
memory = PersistentMemory.load_from_file("state.json")

# MongoDB
from pymongo import MongoClient
client = MongoClient("mongodb://localhost:27017/")
collection = client["db"]["states"]

workflow.memory.save_to_mongodb(collection, "workflow_123")
memory = PersistentMemory.load_from_mongodb(collection, "workflow_123")

# Create workflow with loaded memory
workflow = MyWorkflow(llm_client=llm_client, memory=memory)
```

## Context Generation

```python
# Full context for LLM
context = self.memory.to_context()

# Limited history
context = self.memory.to_context(history_n=3)  # Last 3 entries only

# Individual components
background = self.memory.background.to_context()
working_mem = self.memory.working_memory.to_context()
history = self.memory.history.to_context(n=5)
```

## Workflow Initialization

```python
workflow = MyWorkflow(
    llm_client=client,
    memory=None,  # Creates new if None
    tool_registry=None,  # Uses global if None
    enable_summarization=True,
    max_refinement_retries=3
)
```

## State Execution

```python
# Transition first
workflow.my_transition()

# Then execute
result = workflow.execute_state(
    state_name="my_state",
    state_input={"key": "value"},
    raw_transition=False  # Set True to skip summarization
)
```

## Custom Validation

```python
class MyWorkflow(LLMStateMachine):
    def validate_output(self, objective, output, validation_llm=None):
        # Custom logic
        if meets_criteria(output):
            return True, None
        else:
            return False, "Refinement advice"
```

## Custom Summarization

```python
from llm_fsm import create_custom_summarizer

# With custom prompt file
summarizer = create_custom_summarizer(
    llm_client=client,
    prompt_file="my_prompt.txt"
)
workflow.set_custom_summarizer(summarizer)

# Or override entirely
class MyWorkflow(LLMStateMachine):
    def on_exit_state(self):
        # Custom summarization logic
        pass
```

## History Management

```python
# Add entry
self.memory.history.add_entry(
    state_name="my_state",
    input_data=input_data,
    output_data=output_data,
    custom_fields={"key": "value"}
)

# Get recent entries
recent = self.memory.history.get_recent(n=5)

# Access all entries
all_entries = self.memory.history.entries
```

## Built-in Memory Tools

LLM can use these tools automatically:

```python
# Available to LLM:
# - set_memory(bucket_index: int, content: str)
# - append_memory(bucket_index: int, content: str)
# - get_memory(bucket_index: int)
# - clear_memory(bucket_index: int)
# - view_all_memory()

# Example system prompt
system_prompt = """Use set_memory(0, "content") to store information.
Use view_all_memory() to see what you've stored."""
```

## Built-in Demo Tools

```python
# Pre-registered demo tools:
# - wait_for_human(prompt: str) -> BREAKING
# - think(thought: str) -> logs reasoning
# - complete_task(task_name: str, summary: str) -> marks complete
# - demo_search(query: str) -> mock search
# - demo_calculate(expression: str) -> basic calculator
# - demo_save_file(filename: str, content: str) -> mock save
```

## Error Handling

```python
def on_enter_my_state(self, state_input=None):
    try:
        result = self.run_llm_with_tools(...)
        return result
    except ExecutionBreak:
        # Breaking tool executed
        self.memory.save_to_file("paused.json")
        raise
    except Exception as e:
        # Other errors
        self.memory.working_memory.set(9, f"Error: {e}")
        raise
```

## Testing Pattern

```python
import pytest
from unittest.mock import Mock
from llm_fsm import LLMResponse

@pytest.fixture
def mock_llm():
    llm = Mock()
    llm.chat.return_value = LLMResponse(
        content="Mocked response",
        tool_calls=None,
        finish_reason="stop"
    )
    return llm

def test_workflow(mock_llm):
    workflow = MyWorkflow(llm_client=mock_llm)
    workflow.transition()
    result = workflow.execute_state("state", state_input="test")
    assert result is not None
```

## Common Patterns

### Pattern 1: Research & Writing

```python
class ResearchWorkflow(LLMStateMachine):
    research = State(initial=True)
    outline = State()
    writing = State()
    review = State()
    done = State(final=True)
    
    research_to_outline = research.to(outline)
    outline_to_writing = outline.to(writing)
    writing_to_review = writing.to(review)
    approve = review.to(done)
    revise = review.to(writing)
    
    research_loop = research.to.itself(on="tool_use")
    review_loop = review.to.itself(on="tool_use")
```

### Pattern 2: Approval Workflow

```python
class ApprovalWorkflow(LLMStateMachine):
    draft = State(initial=True)
    review = State()
    approved = State(final=True)
    rejected = State(final=True)
    
    submit = draft.to(review)
    approve = review.to(approved)
    reject = review.to(rejected)
    revise = review.to(draft)
    
    review_loop = review.to.itself(on="tool_use")
```

### Pattern 3: Data Processing Pipeline

```python
class DataPipeline(LLMStateMachine):
    ingest = State(initial=True)
    validate = State()
    transform = State()
    analyze = State()
    report = State(final=True)
    
    validate_data = ingest.to(validate)
    transform_data = validate.to(transform)
    analyze_data = transform.to(analyze)
    generate_report = analyze.to(report)
    
    # Loops for tool use
    ingest_loop = ingest.to.itself(on="tool_use")
    transform_loop = transform.to.itself(on="tool_use")
    analyze_loop = analyze.to.itself(on="tool_use")
```

## Configuration Tips

### Memory Organization

```python
# Define bucket constants
BUCKET_TASK = 0
BUCKET_STATUS = 1
BUCKET_FINDINGS_1 = 2
BUCKET_FINDINGS_2 = 3
BUCKET_DRAFT = 4
BUCKET_FEEDBACK = 5
BUCKET_NOTES = 6
# Buckets 7-9 for dynamic use

def on_enter_state(self, state_input=None):
    self.memory.working_memory.set(BUCKET_TASK, f"Task: {state_input}")
    self.memory.working_memory.set(BUCKET_STATUS, "In Progress")
```

### Context Size Management

```python
# Limit history in context
context = self.memory.to_context(history_n=3)

# Clear old buckets
for i in range(7, 10):  # Clear dynamic buckets
    self.memory.working_memory.delete(i)

# Summarize periodically
if len(self.memory.history.entries) > 20:
    # Trigger summarization
    self.summarizer.update_memory(...)
```

### Tool Scoping Strategy

```python
# Global tools (available everywhere)
@register_tool(description="General tool")
def general_tool(input: str) -> str: ...

# State-specific tools
@register_tool(description="Research tool", state_scope="research")
def search_tool(query: str) -> str: ...

@register_tool(description="Writing tool", state_scope="writing")
def writing_tool(content: str) -> str: ...

# Breaking tools (human interaction)
@register_tool(description="Human feedback", tool_type=ToolType.BREAKING)
def request_feedback(question: str) -> str: ...
```

### Validation Strategy

```python
# Strict validation (throw on failure)
result = self.run_with_refinement(
    state_name="critical_state",
    objective="Must meet all requirements",
    execution_func=execute,
    max_retries=5,
    throw_on_failure=True  # Raise exception if fails
)

# Lenient validation (continue with sub-optimal)
result = self.run_with_refinement(
    state_name="best_effort_state",
    objective="Try to meet requirements",
    execution_func=execute,
    max_retries=2,
    throw_on_failure=False  # Continue with sub-optimal
)
```

## Debugging

```python
# Print current state
print(f"Current state: {workflow.current_state.id}")

# Inspect memory
print("Working memory:", workflow.memory.working_memory.get_non_empty())
print("TODO:", workflow.memory.background.todo_list)
print("History count:", len(workflow.memory.history.entries))

# Check registered tools
from llm_fsm import get_global_registry
registry = get_global_registry()
print("All tools:", list(registry.get_all().keys()))
print("State tools:", registry.get_for_state("my_state"))

# Enable logging (future feature)
# import logging
# logging.basicConfig(level=logging.DEBUG)
```

## Environment Variables

```bash
# Recommended setup
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export HUGGINGFACE_TOKEN="hf_..."

# MongoDB
export MONGODB_URI="mongodb://localhost:27017/"
export MONGODB_DB="workflows"
```

```python
# In code
import os
api_key = os.getenv("OPENAI_API_KEY")
client = create_llm_client("openai", model="gpt-4", api_key=api_key)
```

## Performance Tips

1. **Limit context size** - Use `history_n=3` for recent history only
2. **Clear unused memory** - Delete buckets when no longer needed
3. **Batch tool calls** - LLM will batch multiple tool requests
4. **Use raw_transition** - Skip summarization for internal loops
5. **Cache LLM client** - Reuse client instances across workflows
6. **Async execution** - Use async tools for I/O-bound operations

## Common Issues

### Issue: Tool not found
```python
# Check registration
from llm_fsm import get_global_registry
print(get_global_registry().get_all().keys())
```

### Issue: Context too large
```python
# Limit history
context = self.memory.to_context(history_n=2)

# Clear memory
self.memory.working_memory.delete(unused_bucket)
```

### Issue: Breaking not working
```python
# Ensure tool type is set
@register_tool(tool_type=ToolType.BREAKING)  # Must specify!
def my_tool(param: str) -> str: ...
```

### Issue: State not persisting
```python
# Always use memory, not instance variables
# ❌ self.data = "something"
# ✅ self.memory.working_memory.set(0, "something")
```

## Resources

- **Documentation**: See README.md and SETUP.md
- **Examples**: Check examples/ directory
- **Tests**: See tests/ for comprehensive examples
- **Issues**: GitHub Issues for bugs and feature requests

## Quick Workflow Template

```python
from statemachine import State
from llm_fsm import LLMStateMachine, create_llm_client, register_tool, ToolType

# Define tools
@register_tool(description="My tool", state_scope="my_state")
def my_tool(param: str) -> str:
    return f"Result: {param}"

# Define workflow
class MyWorkflow(LLMStateMachine):
    start = State(initial=True)
    my_state = State()
    end = State(final=True)
    
    begin = start.to(my_state)
    finish = my_state.to(end)
    loop = my_state.to.itself(on="tool_use")
    
    def on_enter_my_state(self, state_input=None):
        system_prompt = f"""{self.memory.to_context()}
        Task: {state_input}"""
        
        return self.run_llm_with_tools(
            system_prompt=system_prompt,
            user_message="Execute task",
            state_name="my_state",
            max_iterations=10
        )

# Run
client = create_llm_client("openai", model="gpt-4", api_key="...")
workflow = MyWorkflow(llm_client=client)
workflow.begin()
result = workflow.execute_state("my_state", state_input="Do something")
workflow.finish()
```

---

**For more details, see:**
- Full documentation: README.md
- Setup guide: SETUP.md
- Complete structure: LIBRARY_STRUCTURE.md
- Example code: examples/