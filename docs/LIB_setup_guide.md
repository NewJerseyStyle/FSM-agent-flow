# LLM-FSM Setup and Usage Guide

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-fsm.git
cd llm-fsm

# Install in development mode
pip install -e .

# Or install with specific integrations
pip install -e ".[openai]"
pip install -e ".[litellm]"
pip install -e ".[smolagents]"
pip install -e ".[all]"
```

### From PyPI (when published)

```bash
pip install llm-fsm
```

## Quick Start Tutorial

### 1. Basic Workflow

Create a simple two-state workflow:

```python
from statemachine import State
from llm_fsm import LLMStateMachine, create_llm_client, Message

# Create LLM client
llm_client = create_llm_client(
    "openai",
    model="gpt-4",
    api_key="your-api-key"
)

# Define workflow
class SimpleWorkflow(LLMStateMachine):
    start = State(initial=True)
    process = State()
    end = State(final=True)
    
    begin = start.to(process)
    finish = process.to(end)
    
    def on_enter_process(self, state_input=None):
        context = self.memory.to_context()
        messages = [Message(
            role="user",
            content=f"{context}\nProcess: {state_input}"
        )]
        response = self.llm_client.chat(messages)
        return response.content

# Run workflow
workflow = SimpleWorkflow(llm_client=llm_client)
workflow.begin()
result = workflow.execute_state("process", state_input="Hello!")
print(result)
workflow.finish()
```

### 2. Adding Tools

Add tools to your workflow:

```python
from llm_fsm import register_tool

@register_tool(description="Calculate math expression")
def calculate(expression: str) -> str:
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

@register_tool(description="Search information", state_scope="research")
def search(query: str) -> str:
    # Implement actual search
    return f"Search results for: {query}"

class ToolWorkflow(LLMStateMachine):
    idle = State(initial=True)
    research = State()
    done = State(final=True)
    
    start = idle.to(research)
    finish = research.to(done)
    
    # Self-transition for tool use
    research_loop = research.to.itself(on="tool_use")
    
    def on_enter_research(self, state_input=None):
        system_prompt = f"""You are a researcher.
        
{self.memory.to_context()}

Use the search tool to find information about: {state_input}
Use the calculate tool if you need to compute anything."""
        
        return self.run_llm_with_tools(
            system_prompt=system_prompt,
            user_message="Begin research",
            state_name="research",
            max_iterations=5
        )
```

### 3. Working with Memory

Use the 10-bucket working memory system:

```python
class MemoryWorkflow(LLMStateMachine):
    start = State(initial=True)
    process = State()
    end = State(final=True)
    
    go = start.to(process)
    finish = process.to(end)
    
    def on_enter_process(self, state_input=None):
        # Store information in memory buckets
        self.memory.working_memory.set(0, f"Task: {state_input}")
        self.memory.working_memory.set(1, "Status: In Progress")
        
        # Set background context
        self.memory.background.goals = "Complete the processing task"
        self.memory.background.todo_list = "1. Analyze input\n2. Process data\n3. Generate output"
        
        # LLM can access this context and has tools to manipulate memory
        system_prompt = f"""You are processing data.

{self.memory.to_context()}

Use set_memory, append_memory, and other memory tools to track your progress.
When done, your findings should be in memory buckets 2-4."""
        
        return self.run_llm_with_tools(
            system_prompt=system_prompt,
            user_message="Start processing",
            state_name="process",
            max_iterations=10
        )

# Run and inspect memory
workflow = MemoryWorkflow(llm_client=llm_client)
workflow.go()
result = workflow.execute_state("process", state_input="Analyze customer feedback")

# View memory state
print("Working Memory:", workflow.memory.working_memory.get_non_empty())
print("TODO List:", workflow.memory.background.todo_list)
```

### 4. Refinement Loops

Add validation and automatic refinement:

```python
class RefinementWorkflow(LLMStateMachine):
    idle = State(initial=True)
    writing = State()
    done = State(final=True)
    
    start = idle.to(writing)
    finish = writing.to(done)
    
    def on_enter_writing(self, state_input=None):
        topic = state_input.get("topic")
        
        def write_with_refinement(refinement_advice=None):
            context = self.memory.to_context()
            
            prompt = f"""Write a blog post about: {topic}

{context}

Requirements:
- 3-5 paragraphs
- Clear introduction and conclusion
- Engaging writing style

{f"REFINEMENT NEEDED: {refinement_advice}" if refinement_advice else ""}"""
            
            messages = [Message(role="user", content=prompt)]
            response = self.llm_client.chat(messages, max_tokens=1000)
            return response.content
        
        # Run with automatic validation and refinement
        result = self.run_with_refinement(
            state_name="writing",
            objective="Write a well-structured, engaging blog post with clear intro and conclusion",
            execution_func=write_with_refinement,
            state_input=state_input,
            max_retries=3,
            throw_on_failure=False
        )
        
        return {"post": result}
```

### 5. Breaking Execution (Human-in-the-Loop)

Handle async operations that need human input:

```python
from llm_fsm import register_tool, ToolType, ExecutionBreak

@register_tool(
    description="Request human approval",
    tool_type=ToolType.BREAKING,
    state_scope="review"
)
def request_approval(content: str, question: str) -> str:
    print(f"\n{'='*60}")
    print("APPROVAL REQUIRED")
    print(f"Content: {content[:200]}...")
    print(f"Question: {question}")
    print(f"{'='*60}\n")
    return "[WAITING FOR APPROVAL]"

class ApprovalWorkflow(LLMStateMachine):
    draft = State(initial=True)
    review = State()
    publish = State(final=True)
    
    submit = draft.to(review)
    approve = review.to(publish)
    revise = review.to(draft)
    
    review_loop = review.to.itself(on="tool_use")
    
    def on_enter_review(self, state_input=None):
        content = state_input.get("content")
        
        # Store current state in memory
        self.memory.working_memory.set(0, f"Reviewing content: {content[:100]}...")
        
        system_prompt = """Review the content and use request_approval tool to get human feedback."""
        
        try:
            result = self.run_llm_with_tools(
                system_prompt=system_prompt,
                user_message=f"Review: {content}",
                state_name="review"
            )
            return {"approved": True, "feedback": result}
        except ExecutionBreak:
            # Save state before exiting
            self.memory.save_to_file("workflow_paused.json")
            raise

# First run - will pause
workflow = ApprovalWorkflow(llm_client=llm_client)
workflow.submit()

try:
    result = workflow.execute_state("review", state_input={"content": "My article..."})
except ExecutionBreak:
    print("Workflow paused - waiting for human approval")

# Later - resume with human feedback
from llm_fsm import PersistentMemory

memory = PersistentMemory.load_from_file("workflow_paused.json")
workflow = ApprovalWorkflow(llm_client=llm_client, memory=memory)

# Add human feedback to memory
memory.working_memory.set(1, "Human feedback: Approved with minor edits")

# Continue from review state
result = workflow.execute_state("review", state_input={"content": "My article..."})
if result["approved"]:
    workflow.approve()
```

### 6. Custom Validation

Override validation logic:

```python
class CustomValidationWorkflow(LLMStateMachine):
    start = State(initial=True)
    process = State()
    end = State(final=True)
    
    go = start.to(process)
    finish = process.to(end)
    
    def validate_output(self, objective, output, validation_llm=None):
        """Custom validation logic."""
        # Example: Check if output contains specific keywords
        required_keywords = ["summary", "conclusion", "recommendation"]
        
        output_lower = str(output).lower()
        missing = [kw for kw in required_keywords if kw not in output_lower]
        
        if missing:
            advice = f"Missing required sections: {', '.join(missing)}"
            return False, advice
        
        # All keywords present - use LLM for deeper validation
        return super().validate_output(objective, output, validation_llm)
```

### 7. Conditional Transitions

Use guards for conditional state transitions:

```python
class ConditionalWorkflow(LLMStateMachine):
    start = State(initial=True)
    simple_path = State()
    complex_path = State()
    end = State(final=True)
    
    # Conditional transitions
    go_simple = start.to(simple_path, cond="is_simple_task")
    go_complex = start.to(complex_path, cond="is_complex_task")
    
    finish_simple = simple_path.to(end)
    finish_complex = complex_path.to(end)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._task_complexity = None
    
    def is_simple_task(self):
        return self._task_complexity == "simple"
    
    def is_complex_task(self):
        return self._task_complexity == "complex"
    
    def determine_path(self, task_description):
        # Use LLM to determine complexity
        messages = [Message(
            role="user",
            content=f"Is this task simple or complex? Answer with one word.\nTask: {task_description}"
        )]
        response = self.llm_client.chat(messages, temperature=0)
        self._task_complexity = response.content.strip().lower()
        
        # Trigger appropriate transition
        if self._task_complexity == "simple":
            self.go_simple()
        else:
            self.go_complex()
```

## Integration Examples

### With smolagents

```python
from smolagents import HfApiModel, LiteLLMModel
from llm_fsm import create_llm_client

# Option 1: HuggingFace API
hf_model = HfApiModel()
llm_client = create_llm_client("smolagents", agent_or_model=hf_model)

# Option 2: LiteLLM through smolagents
litellm_model = LiteLLMModel(model_id="gpt-4")
llm_client = create_llm_client("smolagents", agent_or_model=litellm_model)

# Use in workflow
workflow = MyWorkflow(llm_client=llm_client)
```

### With LiteLLM (Multiple Providers)

```python
# OpenAI
llm_client = create_llm_client("litellm", model="gpt-4", api_key="sk-...")

# Anthropic
llm_client = create_llm_client("litellm", model="claude-3-opus-20240229", api_key="sk-ant-...")

# Google
llm_client = create_llm_client("litellm", model="gemini-pro", api_key="...")

# Local models via Ollama
llm_client = create_llm_client("litellm", model="ollama/llama2")

# Azure OpenAI
llm_client = create_llm_client(
    "litellm",
    model="azure/gpt-4",
    api_key="...",
    api_base="https://your-endpoint.openai.azure.com",
    api_version="2023-05-15"
)
```

### With Custom LLM Frameworks

Since states are just Python methods, you can use any framework:

#### SGLang

```python
import sglang as sgl

class SGLangWorkflow(LLMStateMachine):
    process = State(initial=True)
    done = State(final=True)
    
    finish = process.to(done)
    
    def on_enter_process(self, state_input=None):
        context = self.memory.to_context()
        
        # Use SGLang directly
        @sgl.function
        def process_task(s, task):
            s += f"Context: {context}\n"
            s += f"Task: {task}\n"
            s += "Response:" + sgl.gen("response", max_tokens=500)
        
        result = process_task.run(task=state_input)
        return result["response"]
```

#### AG2 (AutoGen)

```python
from autogen import AssistantAgent, UserProxyAgent

class AG2Workflow(LLMStateMachine):
    research = State(initial=True)
    done = State(final=True)
    
    finish = research.to(done)
    
    def on_enter_research(self, state_input=None):
        context = self.memory.to_context()
        
        # Create AG2 agents
        assistant = AssistantAgent("assistant")
        user_proxy = UserProxyAgent("user_proxy")
        
        # Use AutoGen conversation
        user_proxy.initiate_chat(
            assistant,
            message=f"{context}\n\nTask: {state_input}"
        )
        
        # Extract result from conversation
        result = assistant.last_message()["content"]
        return result
```

## Persistence

### Save to File

```python
# During execution
workflow.memory.save_to_file("my_workflow.json")

# Resume later
from llm_fsm import PersistentMemory

memory = PersistentMemory.load_from_file("my_workflow.json")
workflow = MyWorkflow(llm_client=llm_client, memory=memory)
```

### Save to MongoDB

```python
from pymongo import MongoClient

# Setup MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["workflows"]
collection = db["states"]

# Save
workflow.memory.save_to_mongodb(collection, document_id="workflow_123")

# Load
memory = PersistentMemory.load_from_mongodb(collection, document_id="workflow_123")
workflow = MyWorkflow(llm_client=llm_client, memory=memory)
```

## Testing

### Unit Testing States

```python
import pytest
from llm_fsm import create_llm_client, PersistentMemory

@pytest.fixture
def llm_client():
    return create_llm_client("openai", model="gpt-4", api_key="test-key")

@pytest.fixture
def memory():
    return PersistentMemory()

def test_research_state(llm_client, memory):
    workflow = MyWorkflow(llm_client=llm_client, memory=memory)
    workflow.start_research()
    
    result = workflow.execute_state("research", state_input={"topic": "Test"})
    
    assert result is not None
    assert "research_summary" in result
    assert memory.working_memory.get(0)  # Check memory was used
```

### Mocking LLM Responses

```python
from unittest.mock import Mock, MagicMock
from llm_fsm import LLMResponse

def test_with_mock_llm():
    # Create mock LLM client
    mock_llm = Mock()
    mock_llm.chat.return_value = LLMResponse(
        content="Mocked response",
        tool_calls=None,
        finish_reason="stop"
    )
    
    workflow = MyWorkflow(llm_client=mock_llm)
    result = workflow.execute_state("process", state_input="test")
    
    assert result == "Mocked response"
    mock_llm.chat.assert_called_once()
```

## Best Practices

### 1. Design States as Stateless

```python
# ❌ Bad - relies on instance variables
def on_enter_bad_state(self, state_input=None):
    self.temp_data = "something"
    return self.process()

# ✅ Good - uses persistent memory
def on_enter_good_state(self, state_input=None):
    self.memory.working_memory.set(0, "something")
    return self.process()
```

### 2. Always Update Memory Before Breaking

```python
def on_enter_state_with_breaking_tool(self, state_input=None):
    # Store all context before potential break
    self.memory.working_memory.set(0, f"Current task: {state_input}")
    self.memory.working_memory.set(1, "Status: Waiting for approval")
    
    try:
        result = self.run_llm_with_tools(...)
    except ExecutionBreak:
        # Memory is already updated, safe to exit
        raise
```

### 3. Use Meaningful Bucket Organization

```python
# Organize memory buckets by purpose
BUCKET_TASK = 0
BUCKET_STATUS = 1
BUCKET_FINDINGS_1 = 2
BUCKET_FINDINGS_2 = 3
BUCKET_DRAFT = 4

def on_enter_state(self, state_input=None):
    self.memory.working_memory.set(BUCKET_TASK, f"Task: {state_input}")
    self.memory.working_memory.set(BUCKET_STATUS, "In Progress")
```

### 4. Validate Critical Outputs

```python
def on_enter_critical_state(self, state_input=None):
    return self.run_with_refinement(
        state_name="critical_state",
        objective="Produce accurate, validated output",
        execution_func=self.execute_critical_task,
        max_retries=5,
        throw_on_failure=True  # Don't accept sub-optimal
    )
```

### 5. Use Summarization Wisely

```python
# Enable for most transitions
workflow = MyWorkflow(enable_summarization=True)

# Disable for internal loops
workflow.execute_state("process", raw_transition=True)

# Re-enable for final transition
workflow.execute_state("finalize", raw_transition=False)
```

## Troubleshooting

### Issue: Tool not found

```python
# Check tool registration
from llm_fsm import get_global_registry

registry = get_global_registry()
print("Registered tools:", registry.get_all().keys())
print("Tools for state 'research':", registry.get_for_state("research"))
```

### Issue: Memory not persisting

```python
# Verify memory is being updated
print("Working memory:", workflow.memory.working_memory.get_non_empty())
print("History:", workflow.memory.history.entries)

# Check file/database writes
workflow.memory.save_to_file("debug.json")
```

### Issue: Execution not breaking

```python
# Ensure tool is registered as BREAKING
@register_tool(tool_type=ToolType.BREAKING)  # Must specify!
def my_breaking_tool(input: str) -> str:
    return "..."
```

### Issue: Context too large

```python
# Limit history in context
context = workflow.memory.to_context(history_n=3)  # Only last 3 entries

# Clear old memory buckets
for i in range(10):
    if not_needed(i):
        workflow.memory.working_memory.delete(i)
```

## Next Steps

- Check out the [examples/](../examples/) directory for complete workflows
- Read the [API documentation](API.md) for detailed reference
- Join discussions on [GitHub Issues](https://github.com/yourusername/llm-fsm/issues)

## Support

For questions and support:
- GitHub Issues: https://github.com/yourusername/llm-fsm/issues
- Discussions: https://github.com/yourusername/llm-fsm/discussions
