"""
History summarization for state transitions.
"""

from typing import Optional, Dict, Any
from .llm_client import BaseLLMClient, Message, LLMResponse
from .memory import PersistentMemory


DEFAULT_SUMMARIZATION_PROMPT = """You are a helpful assistant that summarizes conversation history and updates TODO lists.

Given the following information:
- Current TODO list
- Recent state execution history
- New output from the latest state

Your task is to:
1. Summarize what was accomplished
2. Update the TODO list by marking completed items and adding new tasks if needed
3. Keep the summary concise and actionable

Current TODO List:
{todo_list}

Recent History:
{history}

Latest State Output:
State: {state_name}
Input: {state_input}
Output: {state_output}

Please provide:
1. A brief summary of what was accomplished
2. An updated TODO list

Format your response as:
SUMMARY: <your summary>
TODO: <updated todo list>
"""


class HistorySummarizer:
    """Summarizes history and updates context after state transitions."""
    
    def __init__(self, llm_client: BaseLLMClient, custom_prompt: Optional[str] = None):
        """
        Initialize summarizer.
        
        Args:
            llm_client: LLM client for generating summaries
            custom_prompt: Custom summarization prompt (overrides default)
        """
        self.llm_client = llm_client
        self.prompt_template = custom_prompt or DEFAULT_SUMMARIZATION_PROMPT
    
    def summarize(
        self,
        memory: PersistentMemory,
        state_name: str,
        state_input: Any,
        state_output: Any,
        temperature: float = 0.3
    ) -> Dict[str, str]:
        """
        Summarize the state execution and update TODO list.
        
        Args:
            memory: Persistent memory to summarize from
            state_name: Name of the state that just executed
            state_input: Input to the state
            state_output: Output from the state
            temperature: LLM temperature for summarization
        
        Returns:
            Dict with 'summary' and 'todo' keys
        """
        # Get current context
        todo_list = memory.background.todo_list or "No current TODO items"
        history = memory.history.to_context(n=5)
        
        # Format prompt
        prompt = self.prompt_template.format(
            todo_list=todo_list,
            history=history,
            state_name=state_name,
            state_input=str(state_input),
            state_output=str(state_output)
        )
        
        # Get summary from LLM
        messages = [Message(role="user", content=prompt)]
        response = self.llm_client.chat(messages, temperature=temperature)
        
        # Parse response
        summary, todo = self._parse_response(response.content)
        
        return {
            "summary": summary,
            "todo": todo
        }
    
    def _parse_response(self, response_text: str) -> tuple[str, str]:
        """Parse the LLM response to extract summary and TODO."""
        summary = ""
        todo = ""
        
        lines = response_text.strip().split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith("SUMMARY:"):
                current_section = "summary"
                summary = line.replace("SUMMARY:", "").strip()
            elif line.startswith("TODO:"):
                current_section = "todo"
                todo = line.replace("TODO:", "").strip()
            elif current_section == "summary":
                summary += " " + line
            elif current_section == "todo":
                todo += "\n" + line
        
        return summary.strip(), todo.strip()
    
    def update_memory(
        self,
        memory: PersistentMemory,
        state_name: str,
        state_input: Any,
        state_output: Any,
        temperature: float = 0.3
    ) -> None:
        """
        Summarize and update memory in place.
        
        Args:
            memory: Memory to update
            state_name: Name of the state
            state_input: State input
            state_output: State output
            temperature: LLM temperature
        """
        result = self.summarize(
            memory=memory,
            state_name=state_name,
            state_input=state_input,
            state_output=state_output,
            temperature=temperature
        )
        
        # Update TODO list in background context
        memory.background.todo_list = result["todo"]
        
        # Optionally store summary in custom fields
        if "last_summary" not in memory.background.custom_fields:
            memory.background.custom_fields["last_summary"] = []
        
        memory.background.custom_fields["last_summary"].append({
            "state": state_name,
            "summary": result["summary"]
        })
        
        # Keep only last 10 summaries
        if len(memory.background.custom_fields["last_summary"]) > 10:
            memory.background.custom_fields["last_summary"] = \
                memory.background.custom_fields["last_summary"][-10:]


def create_custom_summarizer(
    llm_client: BaseLLMClient,
    prompt_file: Optional[str] = None
) -> HistorySummarizer:
    """
    Create a summarizer with custom prompt from file.
    
    Args:
        llm_client: LLM client
        prompt_file: Path to file containing custom prompt template
    
    Returns:
        HistorySummarizer instance
    """
    custom_prompt = None
    if prompt_file:
        with open(prompt_file, 'r') as f:
            custom_prompt = f.read()
    
    return HistorySummarizer(llm_client, custom_prompt)
