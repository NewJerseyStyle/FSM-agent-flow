"""Instance-scoped tool registry with JSON Schema signatures."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints

from .errors import ExecutionBreak

# Python type -> JSON Schema type mapping
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json_schema(annotation) -> dict[str, str]:
    """Convert a Python type annotation to a JSON Schema type descriptor."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "string"}
    return {"type": _TYPE_MAP.get(annotation, "string")}


def _build_parameters_schema(func: Callable) -> dict[str, Any]:
    """Inspect a callable's signature and build a JSON Schema 'parameters' object."""
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        prop = dict(_python_type_to_json_schema(hints.get(name, param.annotation)))
        prop["description"] = name
        properties[name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _extract_description(func: Callable) -> str:
    """Extract the first line of a callable's docstring."""
    doc = inspect.getdoc(func)
    if doc:
        return doc.split("\n")[0].strip()
    return func.__name__


@dataclass
class ToolSpec:
    """A tool definition with JSON Schema parameters (OpenAI/Anthropic compatible)."""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable
    is_breaking: bool = False

    @classmethod
    def from_callable(
        cls,
        func: Callable,
        *,
        name: str | None = None,
        description: str | None = None,
        is_breaking: bool = False,
    ) -> ToolSpec:
        """Create a ToolSpec by inspecting a callable's signature."""
        return cls(
            name=name or func.__name__,
            description=description or _extract_description(func),
            parameters=_build_parameters_schema(func),
            func=func,
            is_breaking=is_breaking,
        )

    def to_openai_schema(self) -> dict[str, Any]:
        """Return the OpenAI function-calling tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, arguments: dict[str, Any]) -> Any:
        """Execute the tool with the given arguments."""
        result = self.func(**arguments)
        if self.is_breaking:
            raise ExecutionBreak(self.name, result)
        return result


class ToolRegistry:
    """Instance-scoped tool registry. No global singleton."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(
        self,
        func: Callable,
        *,
        name: str | None = None,
        description: str | None = None,
        is_breaking: bool = False,
    ) -> ToolSpec:
        """Register a callable as a tool."""
        spec = ToolSpec.from_callable(
            func, name=name, description=description, is_breaking=is_breaking
        )
        self._tools[spec.name] = spec
        return spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool by name. Raises KeyError if not found."""
        spec = self._tools.get(name)
        if spec is None:
            raise KeyError(f"Tool not found: {name}")
        return spec.execute(arguments)

    def to_openai_schemas(self) -> list[dict[str, Any]]:
        """Return all tools as OpenAI function-calling schemas."""
        return [t.to_openai_schema() for t in self._tools.values()]
