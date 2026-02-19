"""Custom exceptions for fsm-agent-flow."""


class ExecutionBreak(Exception):
    """Raised when a breaking tool is executed, pausing the workflow."""

    def __init__(self, tool_name: str, result=None):
        self.tool_name = tool_name
        self.result = result
        super().__init__(f"Execution paused by breaking tool: {tool_name}")


class MaxRetriesExceeded(Exception):
    """Raised when a state exhausts its retry budget without passing validation."""

    def __init__(self, state_name: str, max_retries: int, feedback: str | None = None):
        self.state_name = state_name
        self.max_retries = max_retries
        self.feedback = feedback
        super().__init__(
            f"State '{state_name}' failed validation after {max_retries} retries. "
            f"Last feedback: {feedback}"
        )


class WorkflowError(Exception):
    """General workflow error."""
    pass
