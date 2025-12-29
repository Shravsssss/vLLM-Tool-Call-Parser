"""Data models for tool calls and parsing results."""

from typing import Any

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """Represents a single tool/function call extracted from LLM output.

    Attributes:
        id: Optional unique identifier for this tool call.
        name: Name of the function to call.
        arguments: Arguments to pass to the function.
    """

    id: str | None = Field(default=None)
    name: str = Field(...)
    arguments: dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "call_001",
                    "name": "get_weather",
                    "arguments": {"location": "NYC", "unit": "fahrenheit"}
                }
            ]
        }
    }


class ParseResult(BaseModel):
    """Result of parsing LLM output for tool calls.

    Attributes:
        tool_calls: List of extracted tool calls.
        raw_input: Original input text that was parsed.
        parse_time_ms: Time taken to parse in milliseconds.
        success: Whether parsing completed without errors.
        error: Error message if parsing failed.
    """

    tool_calls: list[ToolCall] = Field(default_factory=list)
    raw_input: str = Field(...)
    parse_time_ms: float = Field(default=0.0)
    success: bool = Field(default=True)
    error: str | None = Field(default=None)

    @property
    def num_calls(self) -> int:
        """Return the number of tool calls extracted."""
        return len(self.tool_calls)

    @property
    def has_calls(self) -> bool:
        """Return whether any tool calls were found."""
        return len(self.tool_calls) > 0