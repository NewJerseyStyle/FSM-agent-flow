"""
Google Agent Development Kit (ADK) integration for LLM-FSM.

This module provides integration with Google's ADK for state execution,
allowing use of ADK agents within the LLM-FSM workflow framework.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .llm_client import BaseLLMClient, Message, LLMResponse


class ADKClient(BaseLLMClient):
    """
    Client wrapper for Google Agent Development Kit (ADK).
    
    Allows using ADK agents within LLM-FSM state machine workflows.
    ADK handles tool orchestration, memory, and agent execution internally.
    
    Example:
        from google.adk.agents import Agent
        
        adk_agent = Agent(
            model="gemini-2.0-flash",
            name="my_agent",
            instruction="Your agent instructions",
            tools=[tool1, tool2]
        )
        
        llm_client = ADKClient(adk_agent)
        workflow = MyWorkflow(llm_client=llm_client)
    """
    
    def __init__(self, agent, session_id: Optional[str] = None):
        """
        Initialize ADK client.
        
        Args:
            agent: ADK Agent instance (from google.adk.agents import Agent)
            session_id: Optional session ID for stateful conversations
        """
        try:
            from google.adk.agents import Agent
        except ImportError:
            raise ImportError(
                "Google ADK package required. Install with: pip install google-adk"
            )
        
        self.agent = agent
        self.session_id = session_id or "default_session"
        self._conversation_history = []
    
    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Send chat completion via ADK agent.
        
        Note: ADK agents manage tools internally, so the tools parameter
        is ignored. Configure tools when creating the ADK agent.
        
        Args:
            messages: Conversation messages
            tools: Ignored (ADK manages tools internally)
            temperature: LLM temperature
            max_tokens: Maximum tokens
            **kwargs: Additional ADK-specific parameters
        
        Returns:
            LLMResponse with agent's response
        """
        # Convert messages to ADK format (use last user message as query)
        user_message = None
        system_instruction = None
        
        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            elif msg.role == "user":
                user_message = msg.content
        
        if not user_message:
            user_message = "Continue"
        
        # Update agent instruction if system message provided
        if system_instruction and hasattr(self.agent, 'instruction'):
            self.agent.instruction = system_instruction
        
        try:
            # Run agent synchronously (ADK supports both sync and async)
            # Using run() method for simple synchronous execution
            response = self.agent.run(
                query=user_message,
                session_id=self.session_id,
                **kwargs
            )
            
            # Extract response content
            # ADK response format may vary, handle common cases
            if isinstance(response, str):
                content = response
            elif isinstance(response, dict):
                content = response.get('content', str(response))
            elif hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            return LLMResponse(
                content=content,
                tool_calls=None,  # ADK handles tools internally
                finish_reason="stop"
            )
            
        except Exception as e:
            return LLMResponse(
                content=f"ADK Error: {str(e)}",
                tool_calls=None,
                finish_reason="error"
            )
    
    async def async_chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Async version using ADK's async_run.
        
        Args:
            messages: Conversation messages
            tools: Ignored (ADK manages tools)
            **kwargs: Additional parameters
        
        Returns:
            LLMResponse
        """
        user_message = None
        system_instruction = None
        
        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            elif msg.role == "user":
                user_message = msg.content
        
        if not user_message:
            user_message = "Continue"
        
        if system_instruction and hasattr(self.agent, 'instruction'):
            self.agent.instruction = system_instruction
        
        try:
            # Use async_run for async execution
            response = await self.agent.async_run(
                query=user_message,
                session_id=self.session_id,
                **kwargs
            )
            
            if isinstance(response, str):
                content = response
            elif isinstance(response, dict):
                content = response.get('content', str(response))
            elif hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            return LLMResponse(
                content=content,
                tool_calls=None,
                finish_reason="stop"
            )
            
        except Exception as e:
            return LLMResponse(
                content=f"ADK Async Error: {str(e)}",
                tool_calls=None,
                finish_reason="error"
            )
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count.
        
        Note: ADK may have built-in token counting for Gemini models.
        This is a fallback approximation.
        """
        # Rough estimate: ~0.75 tokens per word for English
        return int(len(text.split()) * 0.75)
    
    def reset_session(self):
        """Reset the conversation session."""
        self._conversation_history = []
        # ADK manages sessions internally, might need to create new session_id
        import uuid
        self.session_id = str(uuid.uuid4())


def create_adk_agent_with_tools(
    model: str = "gemini-2.0-flash",
    name: str = "agent",
    instruction: str = "",
    tools: Optional[List[Any]] = None,
    **kwargs
) -> 'ADKClient':
    """
    Helper function to create an ADK agent with tools and wrap it in ADKClient.
    
    Args:
        model: Model name (e.g., "gemini-2.0-flash", "gemini-1.5-pro")
        name: Agent name
        instruction: Agent instructions/system prompt
        tools: List of Python functions to use as tools
        **kwargs: Additional ADK Agent parameters
    
    Returns:
        ADKClient instance
    
    Example:
        def get_weather(city: str) -> str:
            '''Get weather for a city.'''
            return f"Weather in {city}: Sunny, 72°F"
        
        def get_time(city: str) -> str:
            '''Get current time in a city.'''
            return f"Current time in {city}: 10:30 AM"
        
        client = create_adk_agent_with_tools(
            model="gemini-2.0-flash",
            name="weather_agent",
            instruction="You help users with weather and time information.",
            tools=[get_weather, get_time]
        )
        
        workflow = MyWorkflow(llm_client=client)
    """
    try:
        from google.adk.agents import Agent
    except ImportError:
        raise ImportError(
            "Google ADK package required. Install with: pip install google-adk"
        )
    
    agent = Agent(
        model=model,
        name=name,
        instruction=instruction,
        tools=tools or [],
        **kwargs
    )
    
    return ADKClient(agent)


# Integration with LLM-FSM state machine
class ADKStateMixin:
    """
    Mixin for LLM-FSM states that use ADK agents directly.
    
    This allows bypassing the standard tool system and using ADK's
    built-in tool orchestration and agent management.
    
    Example:
        class MyWorkflow(LLMStateMachine, ADKStateMixin):
            research = State(initial=True)
            
            def on_enter_research(self, state_input=None):
                # Use ADK agent directly
                return self.run_with_adk(
                    query=f"Research: {state_input}",
                    state_name="research"
                )
    """
    
    def run_with_adk(
        self,
        query: str,
        state_name: str,
        session_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Run ADK agent within a state.
        
        Args:
            query: Query/prompt for the agent
            state_name: Current state name
            session_id: Optional session ID
            **kwargs: Additional ADK parameters
        
        Returns:
            Agent response content
        """
        if not isinstance(self.llm_client, ADKClient):
            raise TypeError("llm_client must be ADKClient to use run_with_adk()")
        
        # Build context from memory
        context = self.memory.to_context()
        
        # Combine context with query
        full_query = f"""{context}

Current State: {state_name}
Query: {query}"""
        
        # Use the session_id if provided, otherwise use default
        if session_id:
            self.llm_client.session_id = session_id
        
        messages = [Message(role="user", content=full_query)]
        response = self.llm_client.chat(messages, **kwargs)
        
        return response.content


# Update the create_llm_client factory to support ADK
def update_create_llm_client():
    """
    This function shows how to update the existing create_llm_client
    factory in llm_client.py to support ADK.
    
    Add this case to the create_llm_client function:
    
    elif provider == "adk":
        agent = kwargs.get("agent")
        if not agent:
            # Create agent with provided config
            return create_adk_agent_with_tools(
                model=model,
                name=kwargs.get("name", "agent"),
                instruction=kwargs.get("instruction", ""),
                tools=kwargs.get("tools", []),
                **{k: v for k, v in kwargs.items() 
                   if k not in ["agent", "name", "instruction", "tools"]}
            )
        return ADKClient(agent, session_id=kwargs.get("session_id"))
    """
    pass
