# Google ADK Integration Guide

## Overview

Google Agent Development Kit (ADK) is a powerful framework for building AI agents with built-in tool management, pause/resume capabilities, and multi-agent orchestration. LLM-FSM integrates seamlessly with ADK, combining ADK's agent capabilities with FSM's workflow management.

## Why Use ADK with LLM-FSM?

### ADK Advantages

1. **Native Tool Orchestration** - ADK manages tools internally, reducing complexity
2. **Pause/Resume Built-in** - Native support for pausing agent execution
3. **HITL Tool Confirmation** - Human-in-the-loop tool confirmation flow
4. **Context Compaction** - Automatic context management for long conversations
5. **Multi-Agent Systems** - Easy coordination between multiple agents
6. **Model Agnostic** - Works with Gemini, OpenAI, Claude, etc.
7. **Production Ready** - Built by Google for production deployments

### Combined Benefits (ADK + LLM-FSM)

- **Workflow Structure** - FSM provides clear state-based workflow
- **Agent Intelligence** - ADK provides sophisticated agent behavior
- **Memory Management** - LLM-FSM's 10-bucket system complements ADK
- **History Tracking** - FSM tracks state transitions, ADK handles conversations
- **Breaking Execution** - Both support pause/resume (complementary)
- **Validation** - FSM's refinement loops validate ADK agent outputs

## Installation

```bash
# Install LLM-FSM with ADK support
pip install llm-fsm[adk]

# Or install separately
pip install google-adk
```

## Basic Usage

### 1. Create ADK Agent with Tools

```python
from llm_fsm.adk_client import create_adk_agent_with_tools

# Define tools as Python functions
def search_database(query: str) -> str:
    """Search the database for information."""
    # Your search logic
    return f"Results for: {query}"

def send_notification(message: str, recipient: str) -> str:
    """Send a notification to a recipient."""
    # Your notification logic
    return f"Sent '{message}' to {recipient}"

# Create ADK-powered client
llm_client = create_adk_agent_with_tools(
    model="gemini-2.0-flash",
    name="my_agent",
    instruction="You are a helpful agent with database and notification capabilities.",
    tools=[search_database, send_notification]
)
```

### 2. Use in LLM-FSM Workflow

```python
from statemachine import State
from llm_fsm import LLMStateMachine
from llm_fsm.adk_client import ADKStateMixin

class MyWorkflow(LLMStateMachine, ADKStateMixin):
    idle = State(initial=True)
    processing = State()
    done = State(final=True)
    
    start = idle.to(processing)
    finish = processing.to(done)
    
    def on_enter_processing(self, state_input=None):
        # Use ADK agent directly
        result = self.run_with_adk(
            query=f"Process this request: {state_input}",
            state_name="processing"
        )
        return {"result": result}

# Create and run workflow
workflow = MyWorkflow(llm_client=llm_client)
workflow.start()
result = workflow.execute_state("processing", state_input="customer request")
```

## Advanced Patterns

### Pattern 1: ADK with FSM Memory

Combine ADK's agent capabilities with FSM's structured memory:

```python
class DataAnalysisWorkflow(LLMStateMachine, ADKStateMixin):
    collect = State(initial=True)
    analyze = State()
    report = State(final=True)
    
    analyze_data = collect.to(analyze)
    generate_report = analyze.to(report)
    
    def on_enter_collect(self, state_input=None):
        # Store context in FSM memory
        self.memory.working_memory.set(0, f"Dataset: {state_input['dataset']}")
        self.memory.working_memory.set(1, f"Objective: {state_input['objective']}")
        
        # Use ADK to collect data
        result = self.run_with_adk(
            query=f"Collect data for: {state_input['dataset']}",
            state_name="collect"
        )
        
        # Store results in memory for next state
        self.memory.working_memory.set(2, f"Collected data: {result}")
        return {"data": result}
    
    def on_enter_analyze(self, state_input=None):
        # ADK agent has access to memory context
        context = self.memory.to_context()
        
        result = self.run_with_adk(
            query=f"Analyze the data based on context:\n{context}",
            state_name="analyze"
        )
        
        self.memory.working_memory.set(3, f"Analysis: {result}")
        return {"analysis": result}
```

### Pattern 2: Multi-Agent with FSM States

Use different ADK agents for different states:

```python
# Create specialized agents
research_client = create_adk_agent_with_tools(
    model="gemini-2.0-flash",
    name="researcher",
    instruction="You are a research specialist.",
    tools=[search_tool, analyze_tool]
)

writer_client = create_adk_agent_with_tools(
    model="gemini-1.5-pro",
    name="writer",
    instruction="You are a professional writer.",
    tools=[format_tool, edit_tool]
)

class MultiAgentWorkflow(LLMStateMachine):
    research = State(initial=True)
    writing = State()
    done = State(final=True)
    
    write = research.to(writing)
    finish = writing.to(done)
    
    def on_enter_research(self, state_input=None):
        # Use research agent
        self.llm_client = research_client
        messages = [Message(role="user", content=f"Research: {state_input}")]
        response = self.llm_client.chat(messages)
        return {"research": response.content}
    
    def on_enter_writing(self, state_input=None):
        # Switch to writer agent
        self.llm_client = writer_client
        messages = [Message(role="user", content=f"Write based on: {state_input['research']}")]
        response = self.llm_client.chat(messages)
        return {"document": response.content}
```

### Pattern 3: ADK with FSM Validation

Use FSM's refinement loops to validate ADK outputs:

```python
class ValidatedWorkflow(LLMStateMachine, ADKStateMixin):
    process = State(initial=True)
    done = State(final=True)
    
    finish = process.to(done)
    
    def on_enter_process(self, state_input=None):
        def execute_with_adk(refinement_advice=None):
            query = f"""Process this request: {state_input}
            
{f'REFINEMENT NEEDED: {refinement_advice}' if refinement_advice else ''}"""
            
            return self.run_with_adk(
                query=query,
                state_name="process"
            )
        
        # Use FSM's refinement loop with ADK execution
        result = self.run_with_refinement(
            state_name="process",
            objective="Produce a complete, accurate analysis",
            execution_func=execute_with_adk,
            max_retries=3,
            throw_on_failure=False
        )
        
        return {"result": result}
```

## ADK-Specific Features

### 1. Tool Confirmation (HITL)

ADK supports human-in-the-loop tool confirmation:

```python
from google.adk.agents import Agent

# Configure ADK agent with tool confirmation
agent = Agent(
    model="gemini-2.0-flash",
    name="cautious_agent",
    instruction="You are a careful agent that asks before using tools.",
    tools=[sensitive_tool],
    tool_confirmation=True  # Enable HITL confirmation
)

llm_client = ADKClient(agent)
```

### 2. Context Compaction

ADK automatically manages context size:

```python
agent = Agent(
    model="gemini-2.0-flash",
    name="efficient_agent",
    instruction="Your instructions",
    tools=[...],
    context_compaction=True,  # Enable automatic compaction
    max_context_tokens=8000
)
```

### 3. Multi-Turn Conversations

ADK maintains conversation state across turns:

```python
# Create client with session ID
llm_client = ADKClient(agent, session_id="customer_123")

# First interaction
response1 = llm_client.chat([Message(role="user", content="Hello")])

# ADK remembers context in same session
response2 = llm_client.chat([Message(role="user", content="What did I just say?")])
```

### 4. Async Execution

ADK supports async operations:

```python
class AsyncWorkflow(LLMStateMachine):
    process = State(initial=True)
    done = State(final=True)
    
    finish = process.to(done)
    
    async def on_enter_process(self, state_input=None):
        if not isinstance(self.llm_client, ADKClient):
            raise TypeError("Requires ADKClient")
        
        # Use async_chat for async execution
        messages = [Message(role="user", content=f"Process: {state_input}")]
        response = await self.llm_client.async_chat(messages)
        return {"result": response.content}
```

## Comparison: ADK vs Other Integrations

| Feature | ADK | smolagents | OpenAI Direct | LiteLLM |
|---------|-----|-----------|---------------|---------|
| Tool Management | ✅ Built-in | ✅ Built-in | ⚠️ Manual | ⚠️ Manual |
| Pause/Resume | ✅ Native | ❌ | ❌ | ❌ |
| HITL Confirmation | ✅ Yes | ❌ | ❌ | ❌ |
| Context Compaction | ✅ Automatic | ❌ | ❌ | ❌ |
| Multi-Agent | ✅ Native | ⚠️ Manual | ⚠️ Manual | ⚠️ Manual |
| Model Support | ✅ Many | ✅ Many | ❌ OpenAI only | ✅ Many |
| Production Ready | ✅ Yes | ⚠️ Beta | ✅ Yes | ✅ Yes |

## Best Practices

### 1. When to Use ADK

Use ADK when you need:
- Complex tool orchestration
- Human-in-the-loop interactions
- Multi-agent coordination
- Production-grade agent systems
- Automatic context management

### 2. When to Use Direct LLM Integration

Use direct OpenAI/LiteLLM when:
- Simple workflows without complex tools
- Full control over prompting
- No need for agent frameworks
- Simpler debugging

### 3. Combining ADK with FSM

Best practices for ADK + FSM:
- Use **FSM for workflow structure** (high-level states)
- Use **ADK for agent behavior** (within each state)
- Use **FSM memory** for persistent context
- Use **ADK sessions** for conversation state
- Use **FSM validation** to verify ADK outputs

### 4. Memory Strategy

- **FSM Working Memory**: Store structured information (findings, status, key data)
- **ADK Conversation**: Handle natural language context and tool use
- **FSM History**: Track state transitions and workflow progress
- **ADK Sessions**: Maintain agent conversation context

## Troubleshooting

### Issue: ADK not installed

```bash
pip install google-adk
```

### Issue: Authentication errors

Configure Google Cloud credentials or Gemini API key:

```bash
export GOOGLE_API_KEY="your-gemini-api-key"
# or
gcloud auth application-default login
```

### Issue: Tools not working

Ensure tools are defined as Python functions with proper docstrings:

```python
def my_tool(param: str) -> str:
    """
    Tool description here.
    
    Args:
        param: Parameter description
        
    Returns:
        Result description
    """
    return result
```

### Issue: Session state not persisting

Use consistent session IDs:

```python
# Create with specific session ID
llm_client = ADKClient(agent, session_id="unique_session_id")

# Or extract from workflow
session_id = workflow.memory.background.custom_fields.get("session_id")
llm_client = ADKClient(agent, session_id=session_id)
```

## Example: Complete Customer Service Workflow

See `examples/example_google_adk.py` for a complete example featuring:
- Multi-state customer service workflow
- ADK tool orchestration
- Breaking execution for human escalation
- FSM memory management
- Pause and resume capabilities

## Resources

- **Google ADK Documentation**: https://github.com/google/adk
- **Gemini API**: https://ai.google.dev/
- **LLM-FSM Documentation**: See README.md
- **Example Code**: examples/example_google_adk.py

## Summary

Google ADK integration provides:
✅ Professional-grade agent capabilities  
✅ Native tool orchestration  
✅ Built-in pause/resume  
✅ Human-in-the-loop support  
✅ Context management  
✅ Production readiness  

Combined with LLM-FSM's workflow structure, memory management, and validation, you get a powerful system for building sophisticated AI workflows.
