"""
Tool system for LLM State Machine.
Provides tool registration, execution, and batch handling.
"""

from typing import Callable, Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import inspect


class ToolType(Enum):
    """Type of tool execution."""
    SYNC = "sync"  # Regular synchronous tool
    ASYNC = "async"  # Asynchronous tool
    BREAKING = "breaking"  # Tool that breaks FSM execution (e.g., wait for human)


@dataclass
class Tool:
    """Tool definition."""
    name: str
    func: Callable
    description: str
    tool_type: ToolType = ToolType.SYNC
    state_scope: Optional[str] = None  # Which state this tool is scoped to
    
    def __call__(self, *args, **kwargs) -> Any:
        """Execute the tool."""
        return self.func(*args, **kwargs)
    
    def get_signature(self) -> Dict[str, Any]:
        """Get tool signature for LLM."""
        sig = inspect.signature(self.func)
        params = {}
        for name, param in sig.parameters.items():
            param_info = {
                "type": param.annotation.__name__ if param.annotation != inspect.Parameter.empty else "Any",
                "required": param.default == inspect.Parameter.empty
            }
            if param.default != inspect.Parameter.empty:
                param_info["default"] = param.default
            params[name] = param_info
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": params,
            "type": self.tool_type.value
        }


class ToolRegistry:
    """Registry for managing tools."""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._state_tools: Dict[str, List[str]] = {}  # state -> tool names
    
    def register(
        self, 
        name: str, 
        func: Callable, 
        description: str,
        tool_type: ToolType = ToolType.SYNC,
        state_scope: Optional[str] = None
    ) -> Tool:
        """Register a tool."""
        tool = Tool(
            name=name,
            func=func,
            description=description,
            tool_type=tool_type,
            state_scope=state_scope
        )
        self._tools[name] = tool
        
        if state_scope:
            if state_scope not in self._state_tools:
                self._state_tools[state_scope] = []
            self._state_tools[state_scope].append(name)
        
        return tool
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_for_state(self, state_name: str) -> List[Tool]:
        """Get all tools available for a specific state."""
        tool_names = self._state_tools.get(state_name, [])
        # Also include global tools (no state scope)
        global_tools = [name for name, tool in self._tools.items() if tool.state_scope is None]
        all_names = list(set(tool_names + global_tools))
        return [self._tools[name] for name in all_names if name in self._tools]
    
    def get_all(self) -> Dict[str, Tool]:
        """Get all registered tools."""
        return self._tools.copy()
    
    def get_signatures_for_state(self, state_name: str) -> List[Dict[str, Any]]:
        """Get tool signatures for LLM in a specific state."""
        tools = self.get_for_state(state_name)
        return [tool.get_signature() for tool in tools]


# Decorator for registering tools
_global_registry = ToolRegistry()


def register_tool(
    name: Optional[str] = None,
    description: str = "",
    tool_type: ToolType = ToolType.SYNC,
    state_scope: Optional[str] = None
):
    """Decorator to register a tool.
    
    Args:
        name: Tool name (defaults to function name)
        description: Tool description
        tool_type: Type of tool (SYNC, ASYNC, BREAKING)
        state_scope: State this tool is scoped to (None for global)
    
    Example:
        @register_tool(description="Search the web", state_scope="research")
        def search(query: str) -> str:
            return do_search(query)
    """
    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or f"Tool: {tool_name}"
        
        _global_registry.register(
            name=tool_name,
            func=func,
            description=tool_desc,
            tool_type=tool_type,
            state_scope=state_scope
        )
        return func
    
    return decorator


def get_global_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return _global_registry


@dataclass
class ToolCall:
    """A single tool call request from LLM."""
    tool_name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None


@dataclass
class ToolResult:
    """Result of a tool execution."""
    tool_name: str
    result: Any
    error: Optional[str] = None
    is_breaking: bool = False
    call_id: Optional[str] = None


class ToolExecutor:
    """Executes tools and handles batching."""
    
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
    
    def execute_single(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        tool = self.registry.get(tool_call.tool_name)
        
        if not tool:
            return ToolResult(
                tool_name=tool_call.tool_name,
                result=None,
                error=f"Tool '{tool_call.tool_name}' not found",
                call_id=tool_call.call_id
            )
        
        try:
            result = tool(**tool_call.arguments)
            return ToolResult(
                tool_name=tool_call.tool_name,
                result=result,
                is_breaking=(tool.tool_type == ToolType.BREAKING),
                call_id=tool_call.call_id
            )
        except Exception as e:
            return ToolResult(
                tool_name=tool_call.tool_name,
                result=None,
                error=str(e),
                call_id=tool_call.call_id
            )
    
    def execute_batch(self, tool_calls: List[ToolCall]) -> tuple[List[ToolResult], bool]:
        """Execute a batch of tool calls.
        
        Returns:
            (results, should_break): results list and whether execution should break
        """
        results = []
        should_break = False
        
        for call in tool_calls:
            result = self.execute_single(call)
            results.append(result)
            
            # If any tool is breaking, mark for break
            if result.is_breaking:
                should_break = True
        
        return results, should_break


# Essential built-in tools
@register_tool(
    description="Wait for human input. Breaks execution until human responds.",
    tool_type=ToolType.BREAKING
)
def wait_for_human(prompt: str) -> str:
    """Wait for human input with a prompt."""
    # This is a placeholder - actual implementation depends on the runtime
    return f"[WAITING FOR HUMAN]: {prompt}"


@register_tool(
    description="Think and reason about the current situation",
    tool_type=ToolType.SYNC
)
def think(thought: str) -> str:
    """Record a thought or reasoning step."""
    return f"[THOUGHT RECORDED]: {thought}"


@register_tool(
    description="Mark a task as complete",
    tool_type=ToolType.SYNC
)
def complete_task(task_name: str, summary: str = "") -> str:
    """Mark a task as complete with optional summary."""
    return f"[TASK COMPLETED]: {task_name}" + (f" - {summary}" if summary else "")


# Demo tools for examples
@register_tool(
    description="Search for information (demo tool)",
    tool_type=ToolType.SYNC,
    state_scope="research"
)
def demo_search(query: str) -> str:
    """Demo search tool."""
    return f"Search results for: {query}\n- Result 1: Information about {query}\n- Result 2: More details on {query}"


@register_tool(
    description="Calculate a mathematical expression (demo tool)",
    tool_type=ToolType.SYNC
)
def demo_calculate(expression: str) -> str:
    """Demo calculator tool."""
    try:
        # Safe eval for demo purposes only
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating: {e}"


@register_tool(
    description="Save content to a file (demo tool)",
    tool_type=ToolType.SYNC,
    state_scope="writing"
)
def demo_save_file(filename: str, content: str) -> str:
    """Demo file saving tool."""
    return f"[DEMO] File '{filename}' saved with {len(content)} characters"
