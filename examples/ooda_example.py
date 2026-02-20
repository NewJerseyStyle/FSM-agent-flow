"""Example: Using the built-in OODA agent inside a workflow state.

Demonstrates:
- Using run_ooda() as a nested agent inside a parent state
- The OODA loop (Observe -> Orient -> Decide -> Act) running as a sub-workflow
- Comparing direct LLM usage vs. OODA agent usage

To run (requires an OpenAI API key):
    export OPENAI_API_KEY=sk-...
    python examples/ooda_example.py
"""

from fsm_agent_flow import (
    ExecutionContext,
    KeyResult,
    StateSpec,
    Workflow,
    run_ooda,
)
from fsm_agent_flow.llm.openai import OpenAIAdapter


# -- Tools available to the OODA agent --

def lookup_data(query: str) -> str:
    """Look up data from a knowledge base."""
    return f"Data for '{query}': [simulated knowledge base result]"


def analyze_data(data: str) -> str:
    """Analyze data and return insights."""
    return f"Analysis of '{data}': [simulated analysis result]"


# -- State that uses the OODA agent --

def investigate_with_ooda(ctx: ExecutionContext) -> str:
    """Use the OODA agent to investigate a question."""
    return run_ooda(
        ctx,
        task=f"Investigate: {ctx.input}",
        tools=[lookup_data, analyze_data],
        max_cycles=3,
    )


def summarize_findings(ctx: ExecutionContext) -> str:
    """Summarize the investigation findings."""
    return ctx.llm.run_with_tools(
        system_prompt="Summarize the findings concisely.",
        user_message=f"Summarize:\n{ctx.input}",
    )


# -- Workflow --

def build_investigation_workflow():
    llm = OpenAIAdapter(model="gpt-4o")
    # # Can also connect with other OpenAI API compatible services
    # llm = OpenAIAdapter(model="deepseek/deepseek-r1-0528:free",
    #                     api_key="sk-or-v1-e666b3...",
    #                     base_url="https://openrouter.ai/api/v1")

    investigate = StateSpec(
        name="investigate",
        objective="Thoroughly investigate the question using OODA methodology",
        key_results=[
            KeyResult("has_findings", "Produced investigation findings", check=lambda o: bool(o)),
        ],
        execute=investigate_with_ooda,
        is_initial=True,
    )

    summarize = StateSpec(
        name="summarize",
        objective="Produce a concise summary of findings",
        key_results=[
            KeyResult("has_summary", "Produced a summary", check=lambda o: len(str(o)) > 50),
        ],
        execute=summarize_findings,
        is_final=True,
    )

    return Workflow(
        objective="Investigate and summarize",
        states=[investigate, summarize],
        transitions={"investigate": "summarize", "summarize": None},
        llm=llm,
    )


if __name__ == "__main__":
    wf = build_investigation_workflow()
    result = wf.run("What are the latest developments in fusion energy?")
    print("\n=== Investigation Complete ===")
    for h in result.history:
        print(f"\nState: {h.state_name}")
        print(f"Output: {str(h.output)[:300]}...")
