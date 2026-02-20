"""
Example: Research and Writing Workflow using smolagents

This example demonstrates:
- Defining states and transitions
- Using smolagents for LLM integration
- Tool use with breaking tools (wait for human)
- Refinement loops with validation
- Memory persistence and context management
"""

from statemachine import State
from llm_fsm import (
    LLMStateMachine,
    ExecutionBreak,
    PersistentMemory,
    create_llm_client,
    register_tool,
    ToolType
)


# Define custom tools for this workflow
@register_tool(
    description="Search for information on a topic",
    tool_type=ToolType.SYNC,
    state_scope="research"
)
def web_search(query: str) -> str:
    """Search the web (mock implementation)."""
    # In real implementation, use actual search API
    return f"""Search results for '{query}':
1. Key finding about {query}
2. Important data point on {query}
3. Recent development in {query}"""


@register_tool(
    description="Request feedback from human reviewer",
    tool_type=ToolType.BREAKING,
    state_scope="review"
)
def request_human_feedback(content: str, question: str) -> str:
    """Request human feedback - breaks execution."""
    print(f"\n{'='*60}")
    print("HUMAN FEEDBACK REQUIRED")
    print(f"{'='*60}")
    print(f"\nContent:\n{content}\n")
    print(f"Question: {question}\n")
    print("Execution will pause here. Resume with feedback later.")
    print(f"{'='*60}\n")
    return "[WAITING FOR HUMAN FEEDBACK]"


class ResearchWritingWorkflow(LLMStateMachine):
    """
    A workflow that researches a topic and writes a report.
    
    States:
    - idle: Starting state
    - research: Gather information using tools
    - outline: Create document outline
    - writing: Write the document
    - review: Review with human feedback
    - done: Final state
    
    Note: Tool use is handled internally within each state method
    via run_llm_with_tools(). No need for explicit self-transitions.
    """
    
    # Define states
    idle = State(initial=True)
    research = State()
    outline = State()
    writing = State()
    review = State()
    done = State(final=True)
    
    # Define transitions
    start_research = idle.to(research)
    create_outline = research.to(outline)
    start_writing = outline.to(writing)
    review_draft = writing.to(review)
    revise = review.to(writing)  # Loop back for revisions
    complete = review.to(done)
    
    def on_enter_research(self, state_input: Any = None):
        """Research state: Gather information on the topic."""
        topic = state_input.get("topic") if isinstance(state_input, dict) else str(state_input)
        
        # Store topic in memory
        self.memory.working_memory.set(0, f"Research Topic: {topic}")
        self.memory.background.goals = f"Research and write a comprehensive report on: {topic}"
        
        # Build context
        context = self.memory.to_context()
        
        system_prompt = f"""You are a research assistant gathering information.

{context}

Your task: Research the topic thoroughly using the web_search tool.
Make at least 3 searches to cover different aspects of the topic.
Store key findings in your working memory (buckets 1-3).

When done, summarize your findings."""
        
        try:
            output = self.run_llm_with_tools(
                system_prompt=system_prompt,
                user_message=f"Research topic: {topic}",
                state_name="research",
                max_iterations=10
            )
            return {"research_summary": output, "topic": topic}
        except ExecutionBreak:
            # Breaking tool used, re-raise to exit
            raise
    
    def on_enter_outline(self, state_input: Any = None):
        """Outline state: Create document structure."""
        
        def create_outline_with_refinement(refinement_advice: Optional[str] = None) -> str:
            context = self.memory.to_context()
            
            system_prompt = f"""You are creating a document outline.

{context}

Create a clear, logical outline for the report. Include:
- Introduction
- Main sections (3-5)
- Conclusion

{f"REFINEMENT NEEDED: {refinement_advice}" if refinement_advice else ""}"""
            
            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content="Create the outline now.")
            ]
            response = self.llm_client.chat(messages)
            return response.content
        
        # Run with refinement
        outline = self.run_with_refinement(
            state_name="outline",
            objective="Create a clear, comprehensive outline with 3-5 main sections",
            execution_func=create_outline_with_refinement,
            state_input=state_input,
            max_retries=2,
            throw_on_failure=False
        )
        
        # Store outline in memory
        self.memory.working_memory.set(4, f"Outline:\n{outline}")
        
        return {"outline": outline}
    
    def on_enter_writing(self, state_input: Any = None):
        """Writing state: Write the document."""
        context = self.memory.to_context()
        
        # Check if this is a revision
        revision_feedback = state_input.get("feedback") if isinstance(state_input, dict) else None
        
        system_prompt = f"""You are writing a comprehensive report.

{context}

Write a well-structured document following the outline in your working memory.
Each section should be detailed and informative.

{f"REVISION FEEDBACK: {revision_feedback}" if revision_feedback else "Write the first draft."}"""
        
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content="Write the document now.")
        ]
        
        response = self.llm_client.chat(messages, max_tokens=2000)
        document = response.content
        
        # Store document in memory
        self.memory.working_memory.set(5, f"Draft Document (excerpt):\n{document[:500]}...")
        
        return {"document": document, "is_revision": bool(revision_feedback)}
    
    def on_enter_review(self, state_input: Any = None):
        """Review state: Get human feedback."""
        document = state_input.get("document") if isinstance(state_input, dict) else str(state_input)
        
        context = self.memory.to_context()
        
        system_prompt = f"""You are reviewing a document for quality.

{context}

Review the document and use the request_human_feedback tool to get reviewer input.
Ask specific questions about quality, completeness, and improvements needed."""
        
        try:
            output = self.run_llm_with_tools(
                system_prompt=system_prompt,
                user_message=f"Review this document:\n\n{document[:1000]}...",
                state_name="review",
                max_iterations=5
            )
            return {"review_complete": True, "feedback": output}
        except ExecutionBreak:
            # Human feedback requested, execution breaks here
            # Store state for resumption
            self.memory.working_memory.set(6, f"Waiting for review feedback on document")
            raise


# Example usage functions
def run_workflow_example():
    """Run the complete workflow example."""
    
    # For this example, we'll simulate with a simple LLM client
    # In real usage, initialize with smolagents:
    #
    # from smolagents import HfApiModel, LiteLLMModel
    # model = HfApiModel()  # or LiteLLMModel(model_id="gpt-4")
    # llm_client = create_llm_client("smolagents", agent_or_model=model)
    
    # For demo, use OpenAI (replace with your setup)
    llm_client = create_llm_client(
        "openai",
        model="gpt-4",
        api_key="your-api-key-here"
    )
    
    # Create memory
    memory = PersistentMemory()
    memory.background.vision_mission = "Create high-quality research reports on AI topics"
    
    # Initialize workflow
    workflow = ResearchWritingWorkflow(
        llm_client=llm_client,
        memory=memory,
        enable_summarization=True,
        max_refinement_retries=3
    )
    
    # Start workflow
    print("Starting Research & Writing Workflow")
    print("=" * 60)
    
    try:
        # Transition to research state
        workflow.start_research()
        output = workflow.execute_state("research", state_input={"topic": "Transformer Architecture in AI"})
        print(f"\n✓ Research complete: {output['research_summary'][:200]}...")
        
        # Create outline
        workflow.create_outline()
        output = workflow.execute_state("outline", state_input=output)
        print(f"\n✓ Outline created: {output['outline'][:200]}...")
        
        # Write document
        workflow.start_writing()
        output = workflow.execute_state("writing", state_input=output)
        print(f"\n✓ Document written ({len(output['document'])} chars)")
        
        # Review (will break here)
        workflow.review_draft()
        output = workflow.execute_state("review", state_input=output)
        
    except ExecutionBreak:
        print("\n⚠ Execution paused - waiting for human feedback")
        print("Save memory and resume later...")
        
        # Save memory state
        memory.save_to_file("workflow_state.json")
        print("✓ State saved to workflow_state.json")
        return workflow, memory
    
    return workflow, memory


def resume_workflow_example(feedback: str):
    """Resume workflow after human feedback."""
    
    # Load saved state
    memory = PersistentMemory.load_from_file("workflow_state.json")
    
    # Recreate LLM client and workflow
    llm_client = create_llm_client(
        "openai",
        model="gpt-4",
        api_key="your-api-key-here"
    )
    
    workflow = ResearchWritingWorkflow(
        llm_client=llm_client,
        memory=memory
    )
    
    print("Resuming workflow with feedback...")
    print("=" * 60)
    
    # Add feedback to memory
    memory.working_memory.set(7, f"Human Feedback: {feedback}")
    
    # Check feedback and decide next step
    if "revision" in feedback.lower() or "improve" in feedback.lower():
        # Need revision - go back to writing
        workflow.revise()
        output = workflow.execute_state("writing", state_input={"feedback": feedback})
        print(f"\n✓ Revision complete")
        
        # Review again (might break again)
        workflow.review_draft()
        try:
            output = workflow.execute_state("review", state_input=output)
        except ExecutionBreak:
            print("\n⚠ Execution paused again - waiting for more feedback")
            memory.save_to_file("workflow_state.json")
            return workflow, memory
    else:
        # Approved - complete workflow
        workflow.complete()
        print("\n✓ Workflow complete!")
    
    return workflow, memory


def simple_state_example():
    """Simple example showing basic state execution."""
    
    from llm_fsm import Message
    
    llm_client = create_llm_client(
        "openai",
        model="gpt-4",
        api_key="your-api-key-here"
    )
    
    class SimpleWorkflow(LLMStateMachine):
        start = State(initial=True)
        process = State()
        end = State(final=True)
        
        go = start.to(process)
        finish = process.to(end)
        
        def on_enter_process(self, state_input: Any = None):
            # Simple state execution
            context = self.memory.to_context()
            prompt = f"{context}\n\nProcess this input: {state_input}"
            
            messages = [Message(role="user", content=prompt)]
            response = self.llm_client.chat(messages)
            return response.content
    
    # Create and run
    workflow = SimpleWorkflow(llm_client=llm_client)
    workflow.go()
    result = workflow.execute_state("process", state_input="Hello, world!")
    print(f"Result: {result}")
    
    workflow.finish()
    print(f"Final state: {workflow.current_state.id}")


if __name__ == "__main__":
    print("LLM-FSM Example with smolagents")
    print("=" * 60)
    print("\nThis example demonstrates:")
    print("- State machine with LLM integration")
    print("- Tool use and breaking tools")
    print("- Memory management")
    print("- Refinement loops")
    print("- State persistence")
    print("\nNote: Replace API keys and configure smolagents for actual use")
    print("=" * 60)
    
    # Uncomment to run
    # workflow, memory = run_workflow_example()
    
    # Later, resume with feedback:
    # workflow, memory = resume_workflow_example("Looks good, approved!")