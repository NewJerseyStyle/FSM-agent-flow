"""Example: Research workflow using the TDD/OKR pattern.

Demonstrates:
- Defining states with objectives and key results
- Programmatic validation checks
- SharedContext for passing data between states
- Using BoundLLM.run_with_tools() for tool-calling loops

To run (requires an OpenAI API key):
    export OPENAI_API_KEY=sk-...
    python examples/research_workflow.py
"""

from fsm_agent_flow import (
    ExecutionContext,
    KeyResult,
    StateSpec,
    Workflow,
)
from fsm_agent_flow.llm.openai import OpenAIAdapter


# -- Tools (scoped per state) --

def search_web(query: str) -> str:
    """Search the web for information."""
    # Stub — replace with real search API
    return f"Results for '{query}':\n- Source A: key finding about {query}\n- Source B: additional data"


def save_draft(title: str, content: str) -> str:
    """Save a draft document."""
    return f"Draft '{title}' saved ({len(content)} chars)"


# -- State execute functions --

def do_research(ctx: ExecutionContext) -> str:
    """Research phase: gather information using tools."""
    topic = ctx.input or "general topic"
    prompt = f"Research the topic: {topic}"
    if ctx.feedback:
        prompt += f"\n\nPrevious attempt feedback: {ctx.feedback}"

    return ctx.llm.run_with_tools(
        system_prompt=(
            "You are a research assistant. Use the search_web tool to find information. "
            "Cite at least 3 distinct sources. Structure your findings clearly."
        ),
        user_message=prompt,
    )


def do_writing(ctx: ExecutionContext) -> str:
    """Writing phase: synthesize research into a report."""
    research = ctx.input or ""
    return ctx.llm.run_with_tools(
        system_prompt=(
            "You are a technical writer. Synthesize the research into a clear, "
            "well-structured report with sections and a conclusion."
        ),
        user_message=f"Write a report based on this research:\n\n{research}",
    )


# -- Key Result checks --

def has_sufficient_content(output) -> bool:
    return len(str(output)) > 200


def has_sections(output) -> bool:
    text = str(output)
    return text.count("#") >= 2 or text.count("\n\n") >= 3


# -- Workflow definition --

def build_research_workflow():
    research = StateSpec(
        name="research",
        objective="Gather comprehensive information on the topic",
        key_results=[
            KeyResult("sufficient_content", "At least 200 characters of research", check=has_sufficient_content),
            KeyResult("sources_cited", "At least 3 distinct sources are cited"),  # LLM-validated
        ],
        execute=do_research,
        tools=[search_web],
        max_retries=2,
        is_initial=True,
    )

    writing = StateSpec(
        name="writing",
        objective="Produce a well-structured report from research findings",
        key_results=[
            KeyResult("has_structure", "Report has clear sections", check=has_sections),
            KeyResult("sufficient_length", "Report is at least 200 chars", check=has_sufficient_content),
        ],
        execute=do_writing,
        tools=[save_draft],
        is_final=True,
    )

    llm = OpenAIAdapter(model="gpt-4o")

    return Workflow(
        objective="Research and write a report",
        states=[research, writing],
        transitions={"research": "writing", "writing": None},
        llm=llm,
        validator_llm=llm,  # Use same LLM for validation (can use cheaper model)
    )


if __name__ == "__main__":
    wf = build_research_workflow()
    result = wf.run("quantum computing advances in 2025")
    print("\n=== Workflow Complete ===")
    for h in result.history:
        print(f"\nState: {h.state_name}")
        print(f"Key Results: {h.key_results_met}")
        print(f"Output preview: {str(h.output)[:200]}...")
