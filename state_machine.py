"""
LLM-powered Finite State Machine using python-statemachine.
"""

from typing import Optional, Dict, Any, List, Callable
from statemachine import StateMachine, State
from statemachine.exceptions import TransitionNotAllowed
import json

from .memory import PersistentMemory, create_memory_tools
from .llm_client import BaseLLMClient, Message, LLMResponse
from .tools import (
    ToolRegistry, ToolExecutor, ToolCall, ToolResult, 
    ToolType, get_global_registry
)
from .summarizer import HistorySummarizer


class ExecutionBreak(Exception):
    """Exception raised when execution should break (e.g., waiting for human)."""
    pass


class LLMStateMachine(StateMachine):
    """
    Base class for LLM-powered state machines.
    
    Developers should inherit from this class and define their states and transitions.
    
    Example:
        class MyWorkflow(LLMStateMachine):
            # Define states
            idle = State(initial=True)
            processing = State()
            review = State()
            done = State(final=True)
            
            # Define transitions
            start = idle.to(processing)
            finish = processing.to(review)
            complete = review.to(done)
            
            # Tool use loops
            process_loop = processing.to.itself(on="tool_use")
            
            def on_enter_processing(self, state_input: Any = None):
                # State logic here
                pass
    """
    
    def __init__(
        self,
        llm_client: BaseLLMClient,
        memory: Optional[PersistentMemory] = None,
        tool_registry: Optional[ToolRegistry] = None,
        enable_summarization: bool = True,
        max_refinement_retries: int = 3,
        model: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize LLM State Machine.
        
        Args:
            llm_client: LLM client for executing states
            memory: Persistent memory (creates new if None)
            tool_registry: Tool registry (uses global if None)
            enable_summarization: Whether to summarize history on state exit
            max_refinement_retries: Maximum refinement attempts per state
            model: Optional model object for python-statemachine
            **kwargs: Additional arguments for StateMachine
        """
        # Initialize parent StateMachine
        if model is None:
            model = {}  # Empty dict as default model
        super().__init__(model=model, **kwargs)
        
        self.llm_client = llm_client
        self.memory = memory or PersistentMemory()
        self.tool_registry = tool_registry or get_global_registry()
        self.tool_executor = ToolExecutor(self.tool_registry)
        self.enable_summarization = enable_summarization
        self.max_refinement_retries = max_refinement_retries
        
        # Summarizer (lazy init)
        self._summarizer = None
        
        # Execution state (for tracking within state execution)
        self._current_state_input = None
        self._current_state_output = None
        self._conversation_history: List[Message] = []
        self._refinement_count = 0
        self._should_break = False
        
        # Add memory tools to registry
        self._register_memory_tools()
    
    def _register_memory_tools(self):
        """Register built-in memory manipulation tools."""
        memory_tools = create_memory_tools()
        for name, func in memory_tools.items():
            # Wrap to inject memory
            def wrapped_tool(func=func, **kwargs):
                return func(memory=self.memory, **kwargs)
            
            self.tool_registry.register(
                name=name,
                func=wrapped_tool,
                description=func.__doc__ or f"Memory tool: {name}",
                tool_type=ToolType.SYNC
            )
    
    @property
    def summarizer(self) -> HistorySummarizer:
        """Lazy-load summarizer."""
        if self._summarizer is None:
            self._summarizer = HistorySummarizer(self.llm_client)
        return self._summarizer
    
    def set_custom_summarizer(self, summarizer: HistorySummarizer):
        """Set a custom summarizer."""
        self._summarizer = summarizer
    
    def execute_state(
        self,
        state_name: str,
        state_input: Any = None,
        raw_transition: bool = False
    ) -> Any:
        """
        Execute a specific state with given input.
        
        Args:
            state_name: Name of the state to execute
            state_input: Input data for the state
            raw_transition: If True, skip summarization
        
        Returns:
            State output
        
        Raises:
            ExecutionBreak: If a breaking tool is used
        """
        # Store input
        self._current_state_input = state_input
        self._current_state_output = None
        self._conversation_history = []
        self._refinement_count = 0
        self._should_break = False
        
        # Get current state
        current_state_obj = getattr(self, state_name, None)
        if current_state_obj is None:
            raise ValueError(f"State '{state_name}' not found")
        
        # Call the state's on_enter method
        enter_method = getattr(self, f"on_enter_{state_name}", None)
        if enter_method:
            try:
                output = enter_method(state_input=state_input)
                self._current_state_output = output
            except ExecutionBreak:
                # Breaking tool was used, exit without summarization
                raise
        else:
            # Default execution if no on_enter method
            output = self._default_state_execution(state_name, state_input)
            self._current_state_output = output
        
        # Add to history
        self.memory.history.add_entry(
            state_name=state_name,
            input_data=state_input,
            output_data=self._current_state_output
        )
        
        # Summarize unless raw transition
        if self.enable_summarization and not raw_transition:
            self.summarizer.update_memory(
                memory=self.memory,
                state_name=state_name,
                state_input=state_input,
                state_output=self._current_state_output
            )
        
        return self._current_state_output
    
    def _default_state_execution(self, state_name: str, state_input: Any) -> str:
        """Default state execution using LLM."""
        # Build context
        context = self.memory.to_context()
        
        system_prompt = f"""You are executing the '{state_name}' state in a workflow.

{context}

Current Input: {state_input}

Execute the tasks required for this state and provide your output."""
        
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=f"Execute state: {state_name}")
        ]
        
        response = self.llm_client.chat(messages)
        return response.content
    
    def run_llm_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        state_name: str,
        max_iterations: int = 10
    ) -> str:
        """
        Run LLM with tool use capability within a state.
        
        Args:
            system_prompt: System prompt for the LLM
            user_message: Initial user message
            state_name: Current state name (for tool scoping)
            max_iterations: Maximum tool use iterations
        
        Returns:
            Final LLM response content
        
        Raises:
            ExecutionBreak: If a breaking tool is encountered
        """
        # Initialize conversation
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_message)
        ]
        
        # Get available tools for this state
        tool_signatures = self.tool_registry.get_signatures_for_state(state_name)
        
        for iteration in range(max_iterations):
            # Call LLM
            response = self.llm_client.chat(
                messages=messages,
                tools=tool_signatures if tool_signatures else None
            )
            
            # Check if tools were called
            if response.tool_calls:
                # Convert to ToolCall objects
                tool_calls = []
                for tc in response.tool_calls:
                    args = json.loads(tc["function"]["arguments"])
                    tool_calls.append(ToolCall(
                        tool_name=tc["function"]["name"],
                        arguments=args,
                        call_id=tc.get("id")
                    ))
                
                # Execute tools
                results, should_break = self.tool_executor.execute_batch(tool_calls)
                
                # Add assistant message with tool calls
                messages.append(Message(
                    role="assistant",
                    content=response.content or "",
                    tool_calls=response.tool_calls
                ))
                
                # Add tool results
                for result in results:
                    result_content = str(result.result) if result.result is not None else f"Error: {result.error}"
                    messages.append(Message(
                        role="tool",
                        content=result_content,
                        tool_call_id=result.call_id,
                        name=result.tool_name
                    ))
                
                # If breaking tool, raise exception
                if should_break:
                    self._should_break = True
                    raise ExecutionBreak("Breaking tool executed, exiting state")
                
                # Continue loop for next iteration
                continue
            else:
                # No tools, return final response
                messages.append(Message(
                    role="assistant",
                    content=response.content
                ))
                return response.content
        
        # Max iterations reached
        return response.content
    
    def validate_output(
        self,
        objective: str,
        output: Any,
        validation_llm: Optional[BaseLLMClient] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Validate state output against objective using LLM.
        
        Args:
            objective: What the state should accomplish
            output: The actual output from the state
            validation_llm: Optional separate LLM for validation (uses default if None)
        
        Returns:
            (is_valid, refinement_advice): Tuple of validation result and optional advice
        """
        llm = validation_llm or self.llm_client
        
        context = self.memory.to_context(history_n=3)
        
        validation_prompt = f"""You are validating the output of a state execution.

Context:
{context}

State Objective: {objective}

Actual Output: {output}

Does the output successfully meet the objective? 

Respond with:
VALID: yes/no
ADVICE: <refinement advice if not valid, otherwise leave empty>

Be strict but fair in your evaluation."""
        
        messages = [Message(role="user", content=validation_prompt)]
        response = llm.chat(messages, temperature=0.3)
        
        # Parse response
        is_valid, advice = self._parse_validation(response.content)
        return is_valid, advice
    
    def _parse_validation(self, response_text: str) -> tuple[bool, Optional[str]]:
        """Parse validation response."""
        is_valid = False
        advice = None
        
        lines = response_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("VALID:"):
                valid_str = line.replace("VALID:", "").strip().lower()
                is_valid = valid_str in ["yes", "true", "1"]
            elif line.startswith("ADVICE:"):
                advice = line.replace("ADVICE:", "").strip()
                if not advice:
                    advice = None
        
        return is_valid, advice
    
    def run_with_refinement(
        self,
        state_name: str,
        objective: str,
        execution_func: Callable,
        state_input: Any = None,
        max_retries: Optional[int] = None,
        throw_on_failure: bool = False
    ) -> Any:
        """
        Run state execution with refinement loop.
        
        Args:
            state_name: Name of the state
            objective: What should be accomplished
            execution_func: Function to execute (takes refinement_advice as arg)
            state_input: Input to the state
            max_retries: Max refinement attempts (uses default if None)
            throw_on_failure: If True, raise exception on max retries
        
        Returns:
            Final output (may be sub-optimal if max retries reached)
        
        Raises:
            ValueError: If throw_on_failure=True and validation fails after max retries
        """
        max_retries = max_retries or self.max_refinement_retries
        refinement_advice = None
        
        for attempt in range(max_retries + 1):
            # Execute
            output = execution_func(refinement_advice=refinement_advice)
            
            # Validate
            is_valid, advice = self.validate_output(objective, output)
            
            if is_valid:
                return output
            
            # Not valid, prepare for refinement
            refinement_advice = advice
            
            if attempt == max_retries:
                # Max retries reached
                if throw_on_failure:
                    raise ValueError(f"State '{state_name}' failed validation after {max_retries} refinement attempts")
                else:
                    # Return sub-optimal output
                    return output
        
        return output
