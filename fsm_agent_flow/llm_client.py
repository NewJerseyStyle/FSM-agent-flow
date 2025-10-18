"""
LLM client wrapper supporting OpenAI and LiteLLM.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json


@dataclass
class Message:
    """A message in the conversation."""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: str = "stop"
    usage: Optional[Dict[str, int]] = None


class BaseLLMClient(ABC):
    """Base class for LLM clients."""
    
    @abstractmethod
    def chat(
        self, 
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Send chat completion request."""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""
    
    def __init__(self, api_key: str, model: str = "gpt-4", base_url: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package required. Install with: pip install openai")
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
    
    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Send chat completion request to OpenAI."""
        # Convert our Message objects to OpenAI format
        openai_messages = []
        for msg in messages:
            msg_dict = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                msg_dict["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id
            if msg.name:
                msg_dict["name"] = msg.name
            openai_messages.append(msg_dict)
        
        # Prepare request
        request_kwargs = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": temperature,
        }
        
        if max_tokens:
            request_kwargs["max_tokens"] = max_tokens
        
        if tools:
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = "auto"
        
        request_kwargs.update(kwargs)
        
        # Make request
        response = self.client.chat.completions.create(**request_kwargs)
        
        # Parse response
        choice = response.choices[0]
        message = choice.message
        
        tool_calls = None
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in message.tool_calls
            ]
        
        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        )
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(self.model)
            return len(encoding.encode(text))
        except:
            # Fallback: rough estimate
            return len(text.split()) * 1.3


class LiteLLMClient(BaseLLMClient):
    """LiteLLM client supporting multiple providers."""
    
    def __init__(self, model: str = "gemini/gemini-pro", api_key: Optional[str] = None, **config):
        try:
            import litellm
        except ImportError:
            raise ImportError("LiteLLM package required. Install with: pip install litellm")
        
        self.model = model
        self.config = config
        if api_key:
            self.config['api_key'] = api_key
        
        # Import for use
        self.litellm = litellm
    
    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Send chat completion request via LiteLLM."""
        # Convert messages
        litellm_messages = []
        for msg in messages:
            msg_dict = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                msg_dict["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id
            if msg.name:
                msg_dict["name"] = msg.name
            litellm_messages.append(msg_dict)
        
        # Prepare request
        request_kwargs = {
            "model": self.model,
            "messages": litellm_messages,
            "temperature": temperature,
            **self.config,
            **kwargs
        }
        
        if max_tokens:
            request_kwargs["max_tokens"] = max_tokens
        
        if tools:
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = "auto"
        
        # Make request
        response = self.litellm.completion(**request_kwargs)
        
        # Parse response
        choice = response.choices[0]
        message = choice.message
        
        tool_calls = None
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in message.tool_calls
            ]
        
        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        )
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count."""
        try:
            return self.litellm.token_counter(model=self.model, text=text)
        except:
            # Fallback
            return len(text.split()) * 1.3


class SmolAgentsClient(BaseLLMClient):
    """Client wrapper for smolagents."""
    
    def __init__(self, agent_or_model):
        """
        Initialize with either a smolagents model or agent.
        
        Args:
            agent_or_model: Either a smolagents HfApiModel/LiteLLMModel or Agent
        """
        try:
            from smolagents import Agent, HfApiModel
        except ImportError:
            raise ImportError("smolagents package required. Install with: pip install smolagents")
        
        # Check if it's an Agent or a Model
        if isinstance(agent_or_model, Agent):
            self.agent = agent_or_model
            self.model = agent_or_model.model
        else:
            # It's a model, create an agent
            self.model = agent_or_model
            self.agent = Agent(model=self.model, tools=[])
    
    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Send chat completion via smolagents."""
        # Convert messages to prompt
        prompt_parts = []
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
        
        prompt = "\n\n".join(prompt_parts)
        
        # For smolagents, we use the model directly for completion
        # Tool handling is done separately in our framework
        try:
            response_text = self.model(prompt, temperature=temperature, max_tokens=max_tokens)
            
            return LLMResponse(
                content=response_text,
                tool_calls=None,
                finish_reason="stop"
            )
        except Exception as e:
            return LLMResponse(
                content=f"Error: {str(e)}",
                tool_calls=None,
                finish_reason="error"
            )
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count."""
        # Rough estimate
        return len(text.split()) * 1.3


def create_llm_client(
    provider: str = "openai",
    model: str = "gpt-4",
    api_key: Optional[str] = None,
    **kwargs
) -> BaseLLMClient:
    """Factory function to create LLM client.
    
    Args:
        provider: "openai", "litellm", or "smolagents"
        model: Model name/identifier
        api_key: API key (if needed)
        **kwargs: Additional configuration
    
    Returns:
        BaseLLMClient instance
    
    Example:
        client = create_llm_client("openai", "gpt-4", api_key="...")
        client = create_llm_client("litellm", "gpt-4")
        
        # For smolagents, pass the agent/model directly
        from smolagents import HfApiModel
        model = HfApiModel()
        client = create_llm_client("smolagents", agent_or_model=model)
    """
    if provider == "openai":
        return OpenAIClient(api_key=api_key, model=model, **kwargs)
    elif provider == "litellm":
        return LiteLLMClient(model=model, api_key=api_key, **kwargs)
    elif provider == "smolagents":
        agent_or_model = kwargs.get("agent_or_model")
        if not agent_or_model:
            raise ValueError("smolagents provider requires 'agent_or_model' kwarg")
        return SmolAgentsClient(agent_or_model)
    else:
        return OpenAIClient(api_key=api_key, model=model, **kwargs)
