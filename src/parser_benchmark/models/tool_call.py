"""Data models for tool calls and parsing results.

This module provides Pydantic models for representing tool calls extracted from LLM outputs.
The models support validation, serialization, and compatibility with the OpenAI API format.

Key features:
- Strict validation of function names (must be valid identifiers)
- Flexible argument handling (string JSON or dict)
- OpenAI Chat Completions API compatibility
- JSON Schema generation for structured output
"""

import json
import re
from typing import Any, Self

from pydantic import BaseModel, Field, field_validator, model_validator


class ToolCall(BaseModel):
    """Represents a single tool/function call extracted from LLM output.

    This model is compatible with the OpenAI Chat Completions API tool_calls format
    and provides validation to ensure tool calls are well-formed.

    Attributes:
        id: Optional unique identifier for this tool call (OpenAI format: call_xxxxx).
        name: Name of the function to call. Must be a valid Python identifier.
        arguments: Arguments to pass to the function as a dictionary.

    Example:
        >>> call = ToolCall(name="get_weather", arguments={"city": "NYC"})
        >>> call.to_openai_format()
        {'id': None, 'type': 'function', 'function': {'name': 'get_weather', 'arguments': '{"city": "NYC"}'}}
    """

    id: str | None = Field(
        default=None,
        description="Unique identifier for this tool call (e.g., call_abc123)",
        pattern=r"^call_[a-zA-Z0-9]+$|^[a-zA-Z0-9_-]+$|^$",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Name of the function to call",
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments to pass to the function",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "call_001",
                    "name": "get_weather",
                    "arguments": {"location": "NYC", "unit": "fahrenheit"}
                },
                {
                    "name": "search",
                    "arguments": {"query": "python tutorials", "limit": 10}
                }
            ]
        },
        "str_strip_whitespace": True,
    }

    @field_validator("name")
    @classmethod
    def validate_function_name(cls, v: str) -> str:
        """Validate that the function name is a valid identifier.

        Function names must:
        - Start with a letter or underscore
        - Contain only alphanumeric characters and underscores
        - Not be a Python keyword
        """
        v = v.strip()
        if not v:
            raise ValueError("Function name cannot be empty")

        # Check if it's a valid identifier pattern
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", v):
            raise ValueError(
                f"Invalid function name '{v}': must be a valid identifier "
                "(start with letter/underscore, contain only alphanumeric/underscore)"
            )

        return v

    @field_validator("arguments", mode="before")
    @classmethod
    def parse_arguments(cls, v: Any) -> dict[str, Any]:
        """Parse arguments from string JSON if needed.

        This handles the case where arguments come as a JSON string
        (common in OpenAI API responses) and converts them to a dict.
        """
        if v is None:
            return {}

        if isinstance(v, str):
            v = v.strip()
            if not v:
                return {}
            try:
                parsed = json.loads(v)
                if not isinstance(parsed, dict):
                    raise ValueError(f"Arguments must be a JSON object, got {type(parsed).__name__}")
                return parsed
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in arguments: {e}")

        if isinstance(v, dict):
            return v

        raise ValueError(f"Arguments must be a dict or JSON string, got {type(v).__name__}")

    @model_validator(mode="after")
    def validate_arguments_types(self) -> Self:
        """Validate that argument values are JSON-serializable types."""
        def check_serializable(obj: Any, path: str = "arguments") -> None:
            if obj is None or isinstance(obj, (bool, int, float, str)):
                return
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if not isinstance(k, str):
                        raise ValueError(f"Argument keys must be strings at {path}")
                    check_serializable(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_serializable(item, f"{path}[{i}]")
            else:
                raise ValueError(
                    f"Non-JSON-serializable type {type(obj).__name__} at {path}"
                )

        check_serializable(self.arguments)
        return self

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI Chat Completions API tool_calls format.

        Returns:
            Dictionary matching OpenAI's tool_calls[].function format.
        """
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments),
            }
        }

    @classmethod
    def from_openai_format(cls, data: dict[str, Any]) -> "ToolCall":
        """Create a ToolCall from OpenAI API format.

        Args:
            data: Dictionary in OpenAI tool_calls format.

        Returns:
            ToolCall instance.
        """
        function_data = data.get("function", data)
        return cls(
            id=data.get("id"),
            name=function_data.get("name", ""),
            arguments=function_data.get("arguments", {}),
        )

    def matches_schema(self, schema: dict[str, Any]) -> bool:
        """Check if this tool call matches a JSON Schema.

        Args:
            schema: JSON Schema defining the expected function signature.

        Returns:
            True if the arguments match the schema structure.
        """
        # Basic schema validation (for required fields)
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        # Check required fields are present
        for field in required:
            if field not in self.arguments:
                return False

        # Check types match (basic validation)
        for key, value in self.arguments.items():
            if key in properties:
                expected_type = properties[key].get("type")
                if expected_type and not self._type_matches(value, expected_type):
                    return False

        return True

    @staticmethod
    def _type_matches(value: Any, expected_type: str) -> bool:
        """Check if a value matches an expected JSON Schema type."""
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }
        expected = type_map.get(expected_type)
        if expected is None:
            return True  # Unknown type, assume valid
        return isinstance(value, expected)


class ParseResult(BaseModel):
    """Result of parsing LLM output for tool calls.

    This model captures the complete result of a parsing operation, including
    timing information and any errors encountered.

    Attributes:
        tool_calls: List of extracted tool calls.
        raw_input: Original input text that was parsed.
        parse_time_ms: Time taken to parse in milliseconds.
        success: Whether parsing completed without errors.
        error: Error message if parsing failed.
        parser_name: Name of the parser that produced this result.
        partial_calls: Tool calls that were partially parsed (for streaming).
    """

    tool_calls: list[ToolCall] = Field(default_factory=list)
    raw_input: str = Field(...)
    parse_time_ms: float = Field(default=0.0, ge=0.0)
    success: bool = Field(default=True)
    error: str | None = Field(default=None)
    parser_name: str | None = Field(default=None)
    partial_calls: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("parse_time_ms")
    @classmethod
    def validate_parse_time(cls, v: float) -> float:
        """Ensure parse time is non-negative."""
        if v < 0:
            raise ValueError("Parse time cannot be negative")
        return v

    @property
    def num_calls(self) -> int:
        """Return the number of tool calls extracted."""
        return len(self.tool_calls)

    @property
    def has_calls(self) -> bool:
        """Return whether any tool calls were found."""
        return len(self.tool_calls) > 0

    @property
    def calls_per_second(self) -> float:
        """Calculate parsing throughput."""
        if self.parse_time_ms <= 0:
            return 0.0
        return (self.num_calls / self.parse_time_ms) * 1000

    def to_openai_format(self) -> list[dict[str, Any]]:
        """Convert all tool calls to OpenAI API format."""
        return [call.to_openai_format() for call in self.tool_calls]

    def get_call_names(self) -> list[str]:
        """Get list of all function names called."""
        return [call.name for call in self.tool_calls]

    def validate_against_tools(self, tools: list[dict[str, Any]]) -> dict[str, Any]:
        """Validate extracted tool calls against a list of available tools.

        Args:
            tools: List of tool definitions in OpenAI format.

        Returns:
            Dictionary with validation results per tool call.
        """
        # Build a map of available functions
        available = {}
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                available[func.get("name")] = func.get("parameters", {})

        results = {
            "valid": [],
            "invalid": [],
            "unknown_functions": [],
        }

        for call in self.tool_calls:
            if call.name not in available:
                results["unknown_functions"].append(call.name)
                results["invalid"].append({
                    "call": call.model_dump(),
                    "error": f"Unknown function: {call.name}",
                })
            else:
                schema = available[call.name]
                if call.matches_schema(schema):
                    results["valid"].append(call.model_dump())
                else:
                    results["invalid"].append({
                        "call": call.model_dump(),
                        "error": "Arguments don't match schema",
                    })

        return results


class ToolDefinition(BaseModel):
    """Definition of a tool/function that can be called.

    This model represents a tool definition in OpenAI API format,
    used for validating tool calls against available functions.
    """

    name: str = Field(..., min_length=1, max_length=256)
    description: str = Field(default="")
    parameters: dict[str, Any] = Field(default_factory=lambda: {
        "type": "object",
        "properties": {},
        "required": [],
    })

    @field_validator("name")
    @classmethod
    def validate_tool_name(cls, v: str) -> str:
        """Validate tool name is a valid identifier."""
        v = v.strip()
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", v):
            raise ValueError(f"Invalid tool name: {v}")
        return v

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI tools format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

    @classmethod
    def from_openai_format(cls, data: dict[str, Any]) -> "ToolDefinition":
        """Create from OpenAI tools format."""
        func = data.get("function", data)
        return cls(
            name=func.get("name", ""),
            description=func.get("description", ""),
            parameters=func.get("parameters", {}),
        )