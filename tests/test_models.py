"""Tests for data models."""

import pytest
from parser_benchmark.models import ToolCall, ParseResult


class TestToolCall:
    """Tests for ToolCall model."""

    def test_create_minimal(self):
        """Test creating a ToolCall with minimum required fields."""
        call = ToolCall(name="test_func", arguments={})
        assert call.name == "test_func"
        assert call.arguments == {}
        assert call.id is None

    def test_create_with_all_fields(self):
        """Test creating a ToolCall with all fields."""
        call = ToolCall(
            id="call_123",
            name="get_weather",
            arguments={"location": "NYC", "unit": "celsius"}
        )
        assert call.id == "call_123"
        assert call.name == "get_weather"
        assert call.arguments["location"] == "NYC"

    def test_serialization(self):
        """Test JSON serialization."""
        call = ToolCall(name="test", arguments={"a": 1})
        json_str = call.model_dump_json()
        assert '"name": "test"' in json_str or '"name":"test"' in json_str

    def test_missing_name_raises(self):
        """Test that missing name raises validation error."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            ToolCall(arguments={})


class TestParseResult:
    """Tests for ParseResult model."""

    def test_create_success(self):
        """Test creating a successful parse result."""
        call = ToolCall(name="func", arguments={})
        result = ParseResult(
            tool_calls=[call],
            raw_input="test",
            parse_time_ms=0.5,
            success=True
        )
        assert result.success
        assert result.num_calls == 1
        assert result.has_calls

    def test_create_failure(self):
        """Test creating a failed parse result."""
        result = ParseResult(
            tool_calls=[],
            raw_input="invalid",
            parse_time_ms=0.1,
            success=False,
            error="Parse error"
        )
        assert not result.success
        assert result.error == "Parse error"
        assert result.num_calls == 0
        assert not result.has_calls

    def test_empty_result(self):
        """Test parse result with no tool calls found."""
        result = ParseResult(
            tool_calls=[],
            raw_input="no calls here",
            parse_time_ms=0.1,
            success=True
        )
        assert result.success
        assert not result.has_calls