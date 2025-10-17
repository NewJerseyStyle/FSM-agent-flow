# LLM-FSM Library Structure

Complete library implementation with all core components.

## Directory Structure

```
llm-fsm/
├── llm_fsm/
│   ├── __init__.py              # Package initialization and exports
│   ├── memory.py                # 10-bucket working memory system
│   ├── state_machine.py         # Core LLM state machine class
│   ├── llm_client.py            # LLM client wrappers (OpenAI, LiteLLM, smolagents)
│   ├── tools.py                 # Tool system with registration and execution
│   └── summarizer.py            # History summarization
├── examples/
│   ├── example_smolagents.py    # Complete workflow with smolagents
│   ├── example_openai.py        # OpenAI integration example
│   └── example_custom.py        # Custom LLM framework integration
├── tests/
│   └── test_workflow.py         # Comprehensive test suite
├── README.md                    # Main documentation
├── SETUP.md                     # Setup and usage guide
├── pyproject.toml               # Project configuration
└── LICENSE                      # MIT License

```

## Core Components

### 1. Memory System (`memory.py`)

**Classes:**
- `Bucket` - Single memory bucket with content and timestamp
- `WorkingMemory` - 10-bucket system with CRUD operations
- `BackgroundContext` - Persistent background (goals, vision, TODO)
- `StateHistory` - History of state transitions
- `PersistentMemory` - Complete memory system

**Key Features:**
- 10 buckets for LLM working memory
- Automatic context generation
- Save/load to JSON files
- MongoDB integration
- Built-in memory manipulation tools

**API:**
```python
memory = PersistentMemory()

# Working memory
memory.working_memory.set(0, "content")
memory.working_memory.append(1, "more content")
memory.working_memory.get(2)
memory.working_memory.delete(3)

# Background
memory.background.vision_mission = "Mission statement"
memory.background.goals = "Current goals"
memory.background.todo_list = "Distilled TODO"

# History
memory.history.add_entry("state", input, output)

# Context
context = memory.to_context(history_n=5)

# Persistence
memory.save_to_file("state.json")
memory = PersistentMemory.load_from_file("state.json")
```

### 2. State Machine (`state_machine.py`)

**Classes:**
- `LLMStateMachine` - Base class for workflows
- `ExecutionBreak` - Exception for breaking execution

**Key Features:**
- Built on `python-statemachine`
- LLM integration in states
- Tool use with batch execution
- Validation and refinement loops
- History summarization
- Breaking execution support
- Stateless design pattern

**API:**
```python
class MyWorkflow(LLMStateMachine):
    state1 = State(initial=True)
    state2 = State()
    
    transition = state1.to(state2)
    loop = state2.to.itself(on="tool_use")
    
    def on_enter_state2(self, state_input=None):
        # Execute with tools
        return self.run_llm_with_tools(
            system_prompt="...",
            user_message="...",
            state_name="state2"
        )

workflow = MyWorkflow(llm_client=client, memory=memory)
workflow.transition()
result = workflow.execute_state("state2", state_input=data)
```

### 3. LLM Clients (`llm_client.py`)

**Classes:**
- `BaseLLMClient` - Abstract base class
- `OpenAIClient` - OpenAI API wrapper
- `LiteLLMClient` - LiteLLM wrapper (multi-provider)
- `SmolAgentsClient` - smolagents wrapper
- `Message` - Message format
- `LLMResponse` - Response format

**Key Features:**
- Unified interface across providers
- Tool calling support
- Token counting
- Streaming support (future)

**API:**
```python
# Factory function
client = create_llm_client(
    provider="openai",  # or "litellm", "smolagents"
    model="gpt-4",
    api_key="..."
)

# Direct usage
response = client.chat(
    messages=[Message(role="user", content="Hello")],
    tools=[tool_schema],
    temperature=0.7
)
```

### 4. Tool System (`tools.py`)

**Classes:**
- `Tool` - Tool definition
- `ToolType` - Enum (SYNC, ASYNC, BREAKING)
- `ToolRegistry` - Tool registration and management
- `ToolExecutor` - Tool execution engine
- `ToolCall` - Tool call request
- `ToolResult` - Tool execution result

**Key Features:**
- Decorator-based registration
- State-scoped tools
- Batch execution
- Breaking tool support
- Built-in demo tools

**API:**
```python
@register_tool(
    description="Tool description",
    tool_type=ToolType.SYNC,
    state_scope="state_name"  # Optional
)
def my_tool(param: str) -> str:
    return f"Result: {param}"

# Built-in tools
@register_tool(tool_type=ToolType.BREAKING)
def wait_for_human(prompt: str) -> str:
    return "[WAITING]"

# Access registry
registry = get_global_registry()
tools = registry.get_for_state("state_name")
```

### 5. Summarization (`summarizer.py`)

**Classes:**
- `HistorySummarizer` - History summarization engine

**Key Features:**
- OODA loop completion
- TODO list updates
- Custom prompts support
- Automatic on state exit

**API:**
```python
summarizer = HistorySummarizer(llm_client)

# Or with custom prompt
summarizer = create_custom_summarizer(
    llm_client,
    prompt_file="custom_prompt.txt"
)

workflow.set_custom_summarizer(summarizer)

# Summarization happens automatically on state exit
# Or manually:
result = summarizer.summarize(memory, state_name, input, output)
```

## Key Design Patterns

### 1. State Definition Pattern

```python
class Workflow(LLMStateMachine):
    # States
    state1 = State(initial=True)
    state2 = State()
    state3 = State(final=True)
    
    # Transitions
    trans1 = state1.to(state2)
    trans2 = state2.to(state3)
    
    # Self-transitions for tool use
    loop = state2.to.itself(on="tool_use")
    
    # State implementation
    def on_enter_state2(self, state_input=None):
        # Your logic here
        pass
```

### 2. Tool Use Pattern

```python
def on_enter_state(self, state_input=None):
    system_prompt = f"""Your task description.
    
{self.memory.to_context()}

Use available tools to accomplish the task."""
    
    try:
        result = self.run_llm_with_tools(
            system_prompt=system_prompt,
            user_message="Begin task",
            state_name="state_name",
            max_iterations=10
        )
        return result
    except ExecutionBreak:
        # Breaking tool used, save and exit
        raise
```

### 3. Refinement Pattern

```python
def on_enter_state(self, state_input=None):
    def execute_with_refinement(refinement_advice=None):
        prompt = f"Do task. {refinement_advice or ''}"
        response = self.llm_client.chat([Message(role="user", content=prompt)])
        return response.content
    
    return self.run_with_refinement(
        state_name="state",
        objective="Clear objective with success criteria",
        execution_func=execute_with_refinement,
        max_retries=3,
        throw_on_failure=False
    )
```

### 4. Breaking Execution Pattern

```python
@register_tool(tool_type=ToolType.BREAKING)
def wait_for_human(prompt: str) -> str:
    return f"[WAITING]: {prompt}"

def on_enter_state(self, state_input=None):
    # Update memory before potential break
    self.memory.working_memory.set(0, "Current context")
    
    try:
        result = self.run_llm_with_tools(...)
    except ExecutionBreak:
        # Save state
        self.memory.save_to_file("paused.json")
        raise

# Resume later
memory = PersistentMemory.load_from_file("paused.json")
workflow = Workflow(llm_client=client, memory=memory)
result = workflow.execute_state("state", state_input=new_data)
```

### 5. Memory Management Pattern

```python
# Organize buckets by purpose
BUCKET_TASK = 0
BUCKET_STATUS = 1
BUCKET_FINDINGS = 2

def on_enter_state(self, state_input=None):
    # Set structured memory
    self.memory.working_memory.set(BUCKET_TASK, f"Task: {state_input}")
    self.memory.working_memory.set(BUCKET_STATUS, "In Progress")
    
    # LLM has tools to manipulate memory
    # set_memory(bucket_index, content)
    # append_memory(bucket_index, content)
    # get_memory(bucket_index)
```

## Extension Points

### Custom LLM Integration

```python
class CustomLLMClient(BaseLLMClient):
    def chat(self, messages, tools=None, **kwargs):
        # Your custom LLM logic
        return LLMResponse(content="...", tool_calls=None)
    
    def count_tokens(self, text):
        return len(text.split())

# Use in workflow
workflow = MyWorkflow(llm_client=CustomLLMClient())
```

### Custom Validation

```python
class CustomWorkflow(LLMStateMachine):
    def validate_output(self, objective, output, validation_llm=None):
        # Your custom validation logic
        if custom_check(output):
            return True, None
        else:
            return False, "Advice for refinement"
```

### Custom Tools

```python
@register_tool(description="My custom tool", state_scope="my_state")
def custom_tool(param1: str, param2: int) -> str:
    # Your tool logic
    return f"Processed: {param1} with {param2}"
```

### Custom Summarization

```python
# Create custom prompt file
custom_prompt = """Your custom summarization prompt template.

History: {history}
State: {state_name}
Input: {state_input}
Output: {state_output}

Generate summary and updated TODO."""

# Use it
summarizer = create_custom_summarizer(
    llm_client,
    prompt_file="my_prompt.txt"
)
workflow.set_custom_summarizer(summarizer)
```

## Testing Strategy

### Unit Tests
- Memory operations (CRUD, persistence)
- Tool registration and execution
- State transitions
- Validation logic

### Integration Tests
- LLM provider integrations
- Tool use workflows
- Breaking execution
- State resumption

### Example Test
```python
def test_state_execution(mock_llm, memory):
    workflow = TestWorkflow(llm_client=mock_llm, memory=memory)
    workflow.transition()
    result = workflow.execute_state("state", state_input="test")
    assert result is not None
    assert memory.working_memory.get(0)  # Memory was used
```

## Performance Considerations

1. **Context Size**: Limit history entries in context (configurable `history_n`)
2. **Memory Cleanup**: Clear unused buckets to reduce context
3. **Tool Batching**: Batch tool calls are executed together
4. **Token Counting**: Estimate context size before LLM calls
5. **Summarization**: Distills history to keep context manageable

## Security Considerations

1. **API Keys**: Never commit API keys, use environment variables
2. **Tool Execution**: Validate tool inputs before execution
3. **Breaking Tools**: Ensure memory is properly saved before breaking
4. **Persistence**: Sanitize data before saving to databases
5. **LLM Output**: Validate and sanitize LLM outputs before using in code

## Future Enhancements

- [ ] Async/await support throughout
- [ ] Streaming responses
- [ ] Nested state machines (if python-statemachine supports)
- [ ] More LLM integrations (Anthropic direct, Cohere, etc.)
- [ ] Enhanced tool system (parallel execution, dependencies)
- [ ] Visual workflow designer
- [ ] Performance monitoring and metrics
- [ ] Built-in logging and debugging
- [ ] Cloud deployment helpers (AWS Lambda, Azure Functions)
- [ ] Workflow templates library

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Code style
- Testing requirements
- Documentation standards
- Pull request process

## License

MIT License - see [LICENSE](LICENSE) file for details.
