"""JSON interchange format for workflows + Python code generation."""

from __future__ import annotations

import inspect
import re
import textwrap
from typing import Any, Callable

from .context import ExecutionContext
from .errors import WorkflowError
from .state import KeyResult, StateSpec
from .tools import ToolSpec
from .workflow import Workflow

# Restricted builtins for safe eval of check expressions
_SAFE_BUILTINS = {
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "isinstance": isinstance,
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "sorted": sorted,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "True": True,
    "False": False,
    "None": None,
}


def _check_to_expression(check: Callable | None) -> str | None:
    """Try to extract a lambda expression string from a check callable."""
    if check is None:
        return None
    try:
        source = inspect.getsource(check).strip()
        # Match lambda patterns: lambda output: <expr>
        match = re.search(r"lambda\s+\w+\s*:\s*(.+?)(?:\)|,\s*$)", source)
        if match:
            return match.group(1).strip().rstrip(",).").strip()
        # Match check=lambda output: <expr> at end of line
        match = re.search(r"lambda\s+(\w+)\s*:\s*(.+)", source)
        if match:
            param_name = match.group(1)
            expr = match.group(2).strip().rstrip(",).")
            # Normalize to use 'output' as the parameter name
            if param_name != "output":
                expr = expr.replace(param_name, "output")
            return expr
    except (OSError, TypeError):
        pass
    return None


def _expression_to_check(expr: str | None) -> Callable[[Any], bool] | None:
    """Convert a string expression to a check callable with restricted eval."""
    if expr is None:
        return None

    def check(output: Any) -> bool:
        namespace = dict(_SAFE_BUILTINS)
        namespace["output"] = output
        return bool(eval(expr, {"__builtins__": {}}, namespace))  # noqa: S307

    # Store the expression for roundtrip
    check._expression = expr  # type: ignore[attr-defined]
    return check


def workflow_to_json(workflow: Workflow) -> dict:
    """Serialize a live Workflow object to the portable JSON format."""
    states_dict: dict[str, dict] = {}

    for name, state in workflow._states.items():
        kr_list = []
        for kr in state.key_results:
            kr_data: dict[str, Any] = {
                "name": kr.name,
                "description": kr.description,
                "check": None,
            }
            # Try to recover expression from stored attribute
            if kr.check is not None:
                expr = getattr(kr.check, "_expression", None)
                if expr is None:
                    expr = _check_to_expression(kr.check)
                kr_data["check"] = expr
            kr_list.append(kr_data)

        tool_names = []
        for t in state.tools:
            if isinstance(t, ToolSpec):
                tool_names.append(t.name)
            elif callable(t):
                tool_names.append(t.__name__)

        execute_module = None
        if state.execute is not None:
            mod = getattr(state.execute, "__module__", None)
            qual = getattr(state.execute, "__qualname__", None)
            if mod and qual and not mod.startswith("fsm_agent_flow") and "<" not in qual:
                execute_module = f"{mod}.{qual}"

        states_dict[name] = {
            "objective": state.objective,
            "key_results": kr_list,
            "tools": tool_names,
            "max_retries": state.max_retries,
            "is_initial": state.is_initial,
            "is_final": state.is_final,
            "execute_module": execute_module,
        }

    return {
        "version": "2.0",
        "objective": workflow._objective,
        "states": states_dict,
        "transitions": dict(workflow._transitions),
        "graph_layout": {},
    }


def workflow_from_json(
    data: dict,
    *,
    llm: Any,
    tools: dict[str, Callable] | None = None,
    execute_fns: dict[str, Callable] | None = None,
) -> Workflow:
    """Deserialize the JSON format into a runnable Workflow.

    Args:
        data: The workflow JSON dict.
        llm: An LLMAdapter instance.
        tools: Mapping of tool name -> callable. Tools referenced in states
               are looked up here.
        execute_fns: Mapping of state name -> execute callable. If a state
                     is not in this dict and has no execute_module, it gets
                     a default LLM run_with_tools execute.
    """
    tools = tools or {}
    execute_fns = execute_fns or {}

    states: list[StateSpec] = []
    for state_name, state_data in data["states"].items():
        # Build key results
        key_results = []
        for kr_data in state_data.get("key_results", []):
            check_fn = _expression_to_check(kr_data.get("check"))
            key_results.append(
                KeyResult(
                    name=kr_data["name"],
                    description=kr_data["description"],
                    check=check_fn,
                )
            )

        # Resolve tools
        state_tools: list[Callable] = []
        for tool_name in state_data.get("tools", []):
            if tool_name in tools:
                state_tools.append(tools[tool_name])

        # Resolve execute function
        execute_fn = None
        if state_name in execute_fns:
            execute_fn = execute_fns[state_name]
        elif state_data.get("execute_module"):
            # Try to import the dotted path
            module_path = state_data["execute_module"]
            parts = module_path.rsplit(".", 1)
            if len(parts) == 2:
                import importlib
                try:
                    mod = importlib.import_module(parts[0])
                    execute_fn = getattr(mod, parts[1], None)
                except ImportError:
                    pass
        else:
            # Default: LLM run_with_tools
            objective = state_data["objective"]

            def _make_default_execute(obj: str) -> Callable:
                def execute(ctx: ExecutionContext) -> str:
                    prompt = str(ctx.input) if ctx.input else obj
                    if ctx.feedback:
                        prompt += f"\n\nPrevious attempt failed: {ctx.feedback}\nPlease fix."
                    return ctx.llm.run_with_tools(
                        system_prompt=f"You are working on: {obj}",
                        user_message=prompt,
                    )
                return execute

            execute_fn = _make_default_execute(objective)

        states.append(
            StateSpec(
                name=state_name,
                objective=state_data["objective"],
                key_results=key_results,
                execute=execute_fn,
                tools=state_tools,
                max_retries=state_data.get("max_retries", 3),
                is_initial=state_data.get("is_initial", False),
                is_final=state_data.get("is_final", False),
            )
        )

    return Workflow(
        objective=data.get("objective", ""),
        states=states,
        transitions=data.get("transitions", {}),
        llm=llm,
    )


def workflow_to_python(data: dict) -> str:
    """Generate an executable .py file from the JSON format."""
    objective = data.get("objective", "Untitled workflow")
    states_data = data.get("states", {})
    transitions = data.get("transitions", {})

    lines: list[str] = []
    lines.append(f'"""Auto-generated workflow: {objective}"""')
    lines.append("")
    lines.append("from fsm_agent_flow import Workflow, StateSpec, KeyResult, ExecutionContext")
    lines.append("")

    # Collect all tool names
    all_tools: set[str] = set()
    for state_data in states_data.values():
        for tool_name in state_data.get("tools", []):
            all_tools.add(tool_name)

    # Generate tool stubs
    if all_tools:
        lines.append("")
        lines.append("# TODO: Implement your tools")
        for tool_name in sorted(all_tools):
            lines.append(f"def {tool_name}(**kwargs) -> str:")
            lines.append(f'    raise NotImplementedError("Implement {tool_name}")')
            lines.append("")

    # Generate execute functions
    lines.append("")
    lines.append("# State execute functions")
    for state_name, state_data in states_data.items():
        fn_name = f"execute_{_safe_identifier(state_name)}"
        obj = state_data.get("objective", state_name)
        lines.append(f"def {fn_name}(ctx: ExecutionContext):")
        lines.append(f"    prompt = str(ctx.input) if ctx.input else \"{_escape_string(obj)}\"")
        lines.append("    if ctx.feedback:")
        lines.append(
            '        prompt += f"\\n\\nPrevious attempt failed: {ctx.feedback}\\nPlease fix."'
        )
        lines.append("    return ctx.llm.run_with_tools(")
        lines.append(f'        system_prompt="You are working on: {_escape_string(obj)}",')
        lines.append("        user_message=prompt,")
        lines.append("    )")
        lines.append("")

    # Generate state definitions
    lines.append("")
    lines.append("# States")
    state_var_names: dict[str, str] = {}
    for state_name, state_data in states_data.items():
        var_name = _safe_identifier(state_name)
        state_var_names[state_name] = var_name
        fn_name = f"execute_{var_name}"

        lines.append(f"{var_name} = StateSpec(")
        lines.append(f'    name="{_escape_string(state_name)}",')
        lines.append(f'    objective="{_escape_string(state_data.get("objective", ""))}",')

        # Key results
        krs = state_data.get("key_results", [])
        if krs:
            lines.append("    key_results=[")
            for kr in krs:
                check = kr.get("check")
                if check:
                    lines.append(
                        f'        KeyResult("{_escape_string(kr["name"])}", '
                        f'"{_escape_string(kr["description"])}", '
                        f"check=lambda output: {check}),"
                    )
                else:
                    lines.append(
                        f'        KeyResult("{_escape_string(kr["name"])}", '
                        f'"{_escape_string(kr["description"])}"),'
                    )
            lines.append("    ],")

        lines.append(f"    execute={fn_name},")

        # Tools
        tool_names = state_data.get("tools", [])
        if tool_names:
            tools_str = ", ".join(tool_names)
            lines.append(f"    tools=[{tools_str}],")

        lines.append(f'    max_retries={state_data.get("max_retries", 3)},')
        if state_data.get("is_initial"):
            lines.append("    is_initial=True,")
        if state_data.get("is_final"):
            lines.append("    is_final=True,")
        lines.append(")")
        lines.append("")

    # Generate main block
    lines.append("")
    lines.append('if __name__ == "__main__":')
    lines.append("    from fsm_agent_flow.llm.openai import OpenAIAdapter")
    lines.append("")
    lines.append('    llm = OpenAIAdapter(model="gpt-4o")')

    state_list = ", ".join(state_var_names[s] for s in states_data)
    lines.append("    wf = Workflow(")
    lines.append(f'        objective="{_escape_string(objective)}",')
    lines.append(f"        states=[{state_list}],")

    # Transitions
    trans_items = []
    for src, dst in transitions.items():
        dst_str = f'"{dst}"' if dst else "None"
        trans_items.append(f'"{src}": {dst_str}')
    trans_str = "{" + ", ".join(trans_items) + "}"
    lines.append(f"        transitions={trans_str},")
    lines.append("        llm=llm,")
    lines.append("    )")
    lines.append('    result = wf.run("your input here")')
    lines.append("    print(result.history[-1].output)")
    lines.append("")

    return "\n".join(lines)


def validate_workflow_json(data: dict) -> list[str]:
    """Validate a workflow JSON dict, returning a list of error strings."""
    errors: list[str] = []

    if "states" not in data:
        errors.append("Missing 'states' field")
        return errors

    states = data["states"]
    if not states:
        errors.append("Workflow must have at least one state")
        return errors

    transitions = data.get("transitions", {})

    initial_count = sum(1 for s in states.values() if s.get("is_initial"))
    final_count = sum(1 for s in states.values() if s.get("is_final"))

    if initial_count == 0:
        errors.append("No initial state defined (set is_initial=true on one state)")
    if initial_count > 1:
        errors.append("Multiple initial states defined")
    if final_count == 0:
        errors.append("No final state defined (set is_final=true on at least one state)")

    for name, state_data in states.items():
        if not state_data.get("objective"):
            errors.append(f"State '{name}' is missing an objective")

        # Check transitions exist for non-final states
        if not state_data.get("is_final"):
            target = transitions.get(name)
            if target is None:
                errors.append(
                    f"Non-final state '{name}' has no transition "
                    f"(add a transition or mark as final)"
                )
            elif target not in states:
                errors.append(
                    f"State '{name}' transitions to '{target}' which does not exist"
                )

        # Validate key result check expressions
        for kr in state_data.get("key_results", []):
            if kr.get("check"):
                try:
                    compile(kr["check"], "<check>", "eval")
                except SyntaxError as e:
                    errors.append(
                        f"State '{name}', KR '{kr['name']}': "
                        f"invalid check expression: {e}"
                    )

    return errors


def _safe_identifier(name: str) -> str:
    """Convert a state name to a valid Python identifier."""
    result = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if result and result[0].isdigit():
        result = "_" + result
    return result or "_state"


def _escape_string(s: str) -> str:
    """Escape a string for use in generated Python code."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
