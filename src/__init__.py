"""
LLM-FSM: A developer-friendly workflow framework with LLM integration and state management.

This library combines python-statemachine with LLM capabilities to create
intelligent, stateful workflows with context management, tool use, and refinement loops.
"""

__version__ = "0.1.0"

# Core components
from .state_machine import LLMStateMachine, ExecutionBreak
from .memory import (
    PersistentMemory,
    WorkingMemory,
    BackgroundContext,
    StateHistory,
    Bucket,
    create_memory_tools
)
from .llm_client import (
    BaseLLMClient,
    OpenAIClient,
    LiteLLMClient,
    SmolAgentsClient,
    create_llm_client,
    Message,
    LLMResponse
)

# Optional ADK import (only if installed)
try:
    from .adk_client import (
        ADKClient,
        ADKStateMixin,
        create_adk_agent_with_tools
    )
    _adk_available = True
except ImportError:
    _adk_available = False
    ADKClient = None
    ADKStateMixin = None
    create_adk_agent_with_tools = None

from .tools import (
    Tool,
    ToolType,
    ToolRegistry,
    ToolExecutor,
    ToolCall,
    ToolResult,
    register_tool,
    get_global_registry
)
from .summarizer import (
    HistorySummarizer,
    create_custom_summarizer
)

__all__ = [
    # State Machine
    "LLMStateMachine",
    "ExecutionBreak",
    
    # Memory
    "PersistentMemory",
    "WorkingMemory",
    "BackgroundContext",
    "StateHistory",
    "Bucket",
    "create_memory_tools",
    
    # LLM Clients
    "BaseLLMClient",
    "OpenAIClient",
    "LiteLLMClient",
    "SmolAgentsClient",
    "create_llm_client",
    "Message",
    "LLMResponse",
    
    # Tools
    "Tool",
    "ToolType",
    "ToolRegistry",
    "ToolExecutor",
    "ToolCall",
    "ToolResult",
    "register_tool",
    "get_global_registry",
    
    # Summarization
    "HistorySummarizer",
    "create_custom_summarizer",
]
