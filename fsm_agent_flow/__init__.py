"""fsm-agent-flow: TDD/OKR-driven agentic workflow framework.

A lightweight FSM engine where each state has verifiable deliverables
(key results) that get validated before advancing — like running tests
after writing code.
"""

__version__ = "2.0.0"

from .context import ExecutionContext, SharedContext, StateOutput, WorkflowContext
from .errors import ExecutionBreak, MaxRetriesExceeded, WorkflowError
from .llm.adapter import LLMAdapter, LLMResponse, Message
from .llm.adapter import ToolCall as LLMToolCall
from .ooda import create_ooda_agent, run_ooda
from .state import KeyResult, StateSpec
from .tools import ToolRegistry, ToolSpec
from .validation import LLMValidator, RuleValidator, ValidationResult, Validator
from .workflow import BoundLLM, Workflow

__all__ = [
    # Workflow engine
    "Workflow",
    "BoundLLM",
    # State definition
    "StateSpec",
    "KeyResult",
    # Context
    "ExecutionContext",
    "SharedContext",
    "StateOutput",
    "WorkflowContext",
    # Tools
    "ToolSpec",
    "ToolRegistry",
    # LLM
    "LLMAdapter",
    "Message",
    "LLMResponse",
    "LLMToolCall",
    # Validation
    "Validator",
    "ValidationResult",
    "LLMValidator",
    "RuleValidator",
    # OODA agent
    "create_ooda_agent",
    "run_ooda",
    # Errors
    "ExecutionBreak",
    "MaxRetriesExceeded",
    "WorkflowError",
]
