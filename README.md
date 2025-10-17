# LLM-FSM: Developer-Friendly Workflow Framework

A Python library combining **Finite State Machines** with **LLM integration** for building intelligent, stateful workflows with context management, tool use, and refinement loops.

## Features

- 🤖 **LLM-Powered State Machines** - Integrate OpenAI, LiteLLM, smolagents, or Google ADK
- 🧠 **10-Bucket Working Memory** - LLM-accessible working memory that persists across states
- 🔄 **Refinement Loops** - Built-in validation and refinement with configurable retry limits
- 🛠️ **Tool System** - State-scoped tools with sync/async/breaking execution modes
- 💾 **Context Management** - Automatic history summarization and context composition
- 📦 **Persistent Storage** - Save/load state to files or MongoDB
- 🎯 **Start Anywhere** - Execute from any state with external memory
- 🔌 **Extensible** - Support for Google ADK, SGLang, supercog-ai/agentic, ag2, and more

## Installation

```bash
pip install llm-fsm

# With OpenAI support
pip install llm-fsm[openai]

# With LiteLLM support  
pip install llm-fsm[litellm]

# With smolagents support
pip install llm-fsm[smolagents]

# All integrations
pip install llm-fsm[all]
```

## Quick Start

```python
from statemachine import State
from llm_fsm import LLMStateMachine, create_llm_client, register_tool, ToolType

# Define custom tools
@register_tool(
    description="Search for information",
    state_scope="research"
)
def search(query: str) -> str:
    return f"Search results for: {query}"

# Define your workflow
class MyWorkflow(LLMStateMachine):
    # Define states
    idle = State(initial=True)
    research = State()
    writing = State()
    done = State(final=True)
    
    # Define transitions
    start = idle.to(research)
    write = research.to(writing)
    finish = writing.to(done)
    
    # Tool use loop (self-transition)
    research_loop = research.to.itself(on="tool_use")
    
    def on_enter_research(self, state_input=None):
        """Research state implementation."""
        context = self.memory.to_context()
        
        system_prompt = f"""You are a researcher.
        
{context}

Research topic: {state_input}
Use the search tool to gather information."""
        
        return self.run_llm_with_tools(
            system_prompt=system_prompt,
            user_message="Begin research",
            state_name="research",
            max_iterations=10
        )

# Create and run
llm_client = create_llm_client("openai", model="gpt-4", api_key="...")
workflow = MyWorkflow(llm_client=llm_client)

workflow.start()
result = workflow.execute_state("research", state_input="AI Safety")
```

## Core Concepts

### 1. States and Transitions

States are defined using `python-statemachine`:

```python
class MyWorkflow(LLMStateMachine):
    state1 = State(initial=True)
    state2 = State()
    state3 = State(final=True)
    
    transition1 = state1.to(state2)
    transition2 = state2.to(state3)
    loop = state2.to.itself(on="tool_use")  # Self-transition for tool use
```

### 2. Working Memory (10-Bucket System)

LLM has access to 10 memory buckets that persist across state executions:

```python
# LLM can use these tools:
# - set_memory(bucket_index, content)
# - append_memory(bucket_index, content)
# - get_memory(bucket_index)
# - clear_memory(bucket_index)
# - view_all_memory()

# In code:
self.memory.working_memory.set(0, "Important information")
self.memory.working_memory.append(1, "Additional context")
```

Non-empty buckets are automatically included in LLM context.

### 3. Tool System

Tools can be registered with different execution types:

```python
from llm_fsm import register_tool, ToolType

# Sync tool (default)
@register_tool(description="Calculate something", state_scope="processing")
def calculate(expr: str) -> float:
    return eval(expr)

# Breaking tool (exits FSM, e.g., wait for human)
@register_tool(
    description="Wait for human input",
    tool_type=ToolType.BREAKING
)
def wait_for_human(prompt: str) -> str:
    return f"[WAITING]: {prompt}"

# Async tool
@register_tool(
    description="Async operation",
    tool_type=ToolType.ASYNC
)
async def async_operation(data: str) -> str:
    await some_async_call()
    return result
```

**Tool Execution:**
- LLM can request multiple tools in one batch
- All tools in batch execute before continuing
- If ANY tool is "breaking" → FSM exits immediately
- Developer must update memory at breaking point for resumption

### 4. Validation and Refinement

Built-in validation with automatic refinement:

```python
def on_enter_my_state(self, state_input=None):
    def execute_with_refinement(refinement_advice=None):
        # Your execution logic
        prompt = f"Do the task. {refinement_advice or ''}"
        return self.llm_client.chat([Message(role="user", content=prompt)])
    
    # Automatic validation and refinement
    result = self.run_with_refinement(
        state_name="my_state",
        objective="Accomplish X with criteria Y",
        execution_func=execute_with_refinement,
        max_retries=3,
        throw_on_failure=False  # Continue with sub-optimal if max retries
    )
    return result
```

### 5. History Summarization

Automatic summarization on state exit (OODA loop):

```python
# Enabled by default
workflow = MyWorkflow(
    llm_client=llm_client,
    enable_summarization=True  # Default
)

# Skip for specific transition
workflow.execute_state("state_name", raw_transition=True)

# Custom summarization prompt
from llm_fsm import create_custom_summarizer

summarizer = create_custom_summarizer(
    llm_client=llm_client,
    prompt_file="my_prompt.txt"
)
workflow.set_custom_summarizer(summarizer)
```

### 6. Breaking Execution & Resumption

Handle async operations (human input, external events):

```python
from llm_fsm import ExecutionBreak

try:
    workflow.execute_state("review", state_input=data)
except ExecutionBreak:
    # Save state
    workflow.memory.save_to_file("state.json")
    print("Paused, waiting for human...")

# Later, resume:
memory = PersistentMemory.load_from_file("state.json")
workflow = MyWorkflow(llm_client=llm_client, memory=memory)
# Continue from same state with new context
workflow.execute_state("review", state_input=updated_data)
```

### 7. Persistent Memory Structure

```python
memory = PersistentMemory()

# Working memory (10 buckets)
memory.working_memory.set(0, "content")

# Background context
memory.background.vision_mission = "Overall mission"
memory.background.goals = "Current goals"
memory.background.todo_list = "Distilled TODO from history"
memory.background.custom_fields["key"] = "value"

# History
memory.history.add_entry(
    state_name="research",
    input_data={"topic": "AI"},
    output_data={"summary": "..."}
)

# Save/Load
memory.save_to_file("state.json")
memory = PersistentMemory.load_from_file("state.json")

# MongoDB
memory.save_to_mongodb(collection, "workflow_123")
memory = PersistentMemory.load_from_mongodb(collection, "workflow_123")
```

## LLM Integration

### OpenAI

```python
from llm_fsm import create_llm_client

llm_client = create_llm_client(
    "openai",
    model="gpt-4",
    api_key="sk-...",
    base_url="https://api.openai.com/v1"  # Optional
)
```

### LiteLLM (Multiple Providers)

```python
llm_client = create_llm_client(
    "litellm",
    model="gpt-4",  # or "claude-3-opus", "gemini-pro", etc.
    api_key="..."
)
```

### smolagents

```python
from smolagents import HfApiModel, LiteLLMModel

# Hugging Face model
model = HfApiModel()
llm_client = create_llm_client("smolagents", agent_or_model=model)

# Or LiteLLM through smolagents
model = LiteLLMModel(model_id="gpt-4")
llm_client = create_llm_client("smolagents", agent_or_model=model)
```

### Google Agent Development Kit (ADK)

```python
from llm_fsm.adk_client import create_adk_agent_with_tools

# Define tools as Python functions
def my_tool(param: str) -> str:
    """Tool description for ADK."""
    return f"Result: {param}"

# Create ADK agent with tools
llm_client = create_adk_agent_with_tools(
    model="gemini-2.0-flash",
    name="my_agent",
    instruction="Agent instructions...",
    tools=[my_tool]
)

# Use ADKStateMixin for direct ADK integration
class MyWorkflow(LLMStateMachine, ADKStateMixin):
    def on_enter_state(self, state_input=None):
        return self.run_with_adk(
            query="Your query",
            state_name="state"
        )
```

### Other Frameworks

Since states are just Python methods, you can integrate:

- **Google ADK** - Full integration with ADK agents and tool orchestration (see example_google_adk.py)
- **SGLang** - Use SGLang's structured generation in state methods
- **supercog-ai/agentic** - Call agentic framework from state methods
- **ag2** - Integrate AutoGen agents in states
- **Custom** - Any Python-based LLM framework

Example with custom framework:

```python
class MyWorkflow(LLMStateMachine):
    def on_enter_my_state(self, state_input=None):
        # Use any framework you want
        import sglang as sgl
        
        context = self.memory.to_context()
        result = sgl.gen("your prompt", context=context)
        return result
```

## Advanced Usage

### Conditional Transitions

```python
class MyWorkflow(LLMStateMachine):
    state1 = State(initial=True)
    state2 = State()
    state3 = State()
    
    # Conditional transition
    go_to_state2 = state1.to(state2, cond="should_go_to_state2")
    go_to_state3 = state1.to(state3, cond="should_go_to_state3")
    
    def should_go_to_state2(self):
        # Custom condition logic
        return self._some_condition
    
    def should_go_to_state3(self):
        return not self._some_condition
```

### Custom Validation

```python
class MyWorkflow(LLMStateMachine):
    def validate_output(self, objective, output, validation_llm=None):
        # Override validation logic
        if custom_validation_logic(output):
            return True, None
        else:
            return False, "Needs improvement in X"
```

### State-Scoped Tools

```python
# Register tools for specific states
@register_tool(description="Research tool", state_scope="research")
def research_tool(query: str) -> str:
    return do_research(query)

@register_tool(description="Writing tool", state_scope="writing")
def writing_tool(content: str) -> str:
    return enhance_writing(content)

# Global tools (available everywhere)
@register_tool(description="General purpose tool")
def general_tool(input: str) -> str:
    return process(input)
```

## Design Principles

### Stateless Execution

State methods should be **stateless** - all state is in external memory:

```python
# ❌ Bad: Relying on instance variables
class BadWorkflow(LLMStateMachine):
    def on_enter_state1(self, state_input=None):
        self.temp_data = "something"  # Don't do this!
        return self.process()
    
    def on_enter_state2(self, state_input=None):
        return self.temp_data  # Not reliable!

# ✅ Good: Using persistent memory
class GoodWorkflow(LLMStateMachine):
    def on_enter_state1(self, state_input=None):
        self.memory.working_memory.set(0, "something")
        return self.process()
    
    def on_enter_state2(self, state_input=None):
        return self.memory.working_memory.get(0)
```

### Breaking Tool Design

When using breaking tools, ensure memory is updated:

```python
def on_enter_review(self, state_input=None):
    # Before breaking, store all necessary context
    self.memory.working_memory.set(5, f"Reviewing document: {doc_id}")
    self.memory.working_memory.set(6, "Waiting for approval")
    
    try:
        result = self.run_llm_with_tools(
            system_prompt="Review and request feedback...",
            user_message="Review this document",
            state_name="review"
        )
    except ExecutionBreak:
        # Memory is saved, safe to exit
        raise
```

## Examples

See `examples/` directory for complete examples:
- `example_smolagents.py` - Research and writing workflow with smolagents
- `example_openai.py` - Customer support workflow with OpenAI
- `example_nested.py` - Nested state machines (if supported)

## Contributing

Contributions welcome! Areas for contribution:
- Additional LLM integrations
- More example workflows
- Enhanced tool system
- Performance optimizations

## License

MIT License

## Acknowledgments

- Built on [python-statemachine](https://github.com/fgmacedo/python-statemachine)
- Inspired by OODA loops and agent frameworks
- Integrates with [smolagents](https://github.com/huggingface/smolagents), [LiteLLM](https://github.com/BerriAI/litellm), and OpenAI
