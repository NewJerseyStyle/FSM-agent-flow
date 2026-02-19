"""Built-in OODA-loop agent implemented as a nested Workflow.

Dogfoods the framework: the OODA agent is itself a 4-state Workflow with
its own key results and validation, demonstrating recursive composition.
"""

from __future__ import annotations

from typing import Any, Callable

from .context import ExecutionContext
from .llm.adapter import LLMAdapter
from .state import KeyResult, StateSpec
from .tools import ToolSpec
from .validation import RuleValidator
from .workflow import Workflow


def create_ooda_agent(
    llm: LLMAdapter,
    tools: list[Callable | ToolSpec],
    task: str,
    *,
    max_cycles: int = 5,
) -> Workflow:
    """Create an OODA-loop agent as a nested Workflow.

    The agent cycles through Observe -> Orient -> Decide -> Act, with each
    state having its own key results. The cycle repeats up to `max_cycles`
    times or until the Act state determines the task is complete.

    Args:
        llm: LLM adapter to use.
        tools: Tools available during the Act phase.
        task: The task description / objective.
        max_cycles: Maximum number of OODA cycles.
    """
    cycle_count = {"n": 0, "max": max_cycles, "done": False}

    # -- State execute functions --

    def observe(ctx: ExecutionContext) -> str:
        prompt = (
            f"Task: {task}\n\n"
            f"You are in the OBSERVE phase. Gather and organize all available information.\n"
        )
        if ctx.feedback:
            prompt += f"\nPrevious attempt feedback: {ctx.feedback}\n"
        if ctx.history:
            last = ctx.history[-1]
            prompt += f"\nPrevious cycle output: {last.output}\n"
        if ctx.input:
            prompt += f"\nInput: {ctx.input}\n"

        prompt += "\nList your observations as structured bullet points."
        return ctx.llm.run_with_tools(
            system_prompt="You are an analytical observer. List key observations.",
            user_message=prompt,
        )

    def orient(ctx: ExecutionContext) -> str:
        observations = ctx.input or ""
        return ctx.llm.run_with_tools(
            system_prompt="You are a strategic analyst. Analyze observations and identify key factors.",
            user_message=(
                f"Task: {task}\n\nObservations:\n{observations}\n\n"
                "Analyze these observations. Identify patterns, key factors, and constraints. "
                "What is the current situation and what are the options?"
            ),
        )

    def decide(ctx: ExecutionContext) -> str:
        analysis = ctx.input or ""
        return ctx.llm.run_with_tools(
            system_prompt="You are a decision maker. Select a concrete action with rationale.",
            user_message=(
                f"Task: {task}\n\nAnalysis:\n{analysis}\n\n"
                "Based on this analysis, decide on a specific action to take. "
                "State the action clearly and explain your rationale."
            ),
        )

    def act(ctx: ExecutionContext) -> str:
        decision = ctx.input or ""
        cycle_count["n"] += 1

        result = ctx.llm.run_with_tools(
            system_prompt=(
                "You are an executor. Carry out the decided action using available tools. "
                "If the task is complete, start your response with 'TASK COMPLETE:' "
                "followed by the final answer."
            ),
            user_message=(
                f"Task: {task}\n\nDecision:\n{decision}\n\n"
                f"Execute this action. Cycle {cycle_count['n']}/{cycle_count['max']}."
            ),
        )

        if "TASK COMPLETE:" in result or cycle_count["n"] >= cycle_count["max"]:
            cycle_count["done"] = True

        return result

    # -- Key Results --

    def has_content(output: Any) -> bool:
        return bool(output and str(output).strip())

    # -- State Specs --

    observe_state = StateSpec(
        name="observe",
        objective="Gather and organize available information",
        key_results=[KeyResult("has_observations", "Produced observations", check=has_content)],
        execute=observe,
        is_initial=True,
    )
    orient_state = StateSpec(
        name="orient",
        objective="Analyze observations and identify key factors",
        key_results=[KeyResult("has_analysis", "Produced analysis", check=has_content)],
        execute=orient,
    )
    decide_state = StateSpec(
        name="decide",
        objective="Select a concrete action with rationale",
        key_results=[KeyResult("has_decision", "Selected an action", check=has_content)],
        execute=decide,
    )
    act_state = StateSpec(
        name="act",
        objective="Execute the decided action and produce a result",
        key_results=[KeyResult("has_result", "Produced a result", check=has_content)],
        execute=act,
        tools=list(tools),
        is_final=True,  # Each cycle is one run; caller loops if needed
    )

    return Workflow(
        objective=f"OODA cycle for: {task}",
        states=[observe_state, orient_state, decide_state, act_state],
        transitions={"observe": "orient", "orient": "decide", "decide": "act", "act": None},
        llm=llm,
        validator=RuleValidator(),
    )


def run_ooda(
    ctx: ExecutionContext,
    task: str,
    *,
    tools: list[Callable | ToolSpec] | None = None,
    max_cycles: int = 5,
) -> str:
    """Convenience: run an OODA agent inside a state's execute function.

    Args:
        ctx: The parent state's ExecutionContext.
        task: The task to accomplish.
        tools: Tools to provide (defaults to parent state's tools via ctx.llm).
        max_cycles: Maximum OODA cycles.

    Returns:
        The final output string from the agent.
    """
    effective_tools = tools or []
    # If we have a BoundLLM, use its adapter; otherwise fall back
    adapter = ctx.llm.adapter if ctx.llm else None
    if adapter is None:
        raise ValueError("ExecutionContext.llm must be set to use run_ooda")

    last_output = None
    for _ in range(max_cycles):
        agent = create_ooda_agent(adapter, effective_tools, task, max_cycles=max_cycles)
        wf_ctx = agent.run(initial_input=last_output or ctx.input)
        last_output = wf_ctx.history[-1].output if wf_ctx.history else ""
        if "TASK COMPLETE:" in str(last_output):
            break

    return str(last_output) if last_output else ""
