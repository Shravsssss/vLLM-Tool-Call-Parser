"""Tests for StateMachineParser."""

import pytest

from parser_benchmark.models import ParseResult
from parser_benchmark.parsers import StateMachineParser


@pytest.fixture
def parser():
    """Create parser instance."""
    return StateMachineParser()


class TestStateMachineBasics:
    """Basic functionality tests."""

    def test_parser_name(self, parser):
        """Test parser name."""
        assert parser.name == "state-machine-parser"

    def test_simple_tool_call(self, parser):
        """Test simple tool call parsing."""
        text = '{"name": "test", "arguments": {}}'
        result = parser.parse(text)

        assert result.success
        assert result.num_calls == 1
        assert result.tool_calls[0].name == "test"

    def test_embedded_tool_call(self, parser):
        """Test tool call embedded in text."""
        text = 'Some text {"name": "func", "arguments": {"x": 1}} more text'
        result = parser.parse(text)

        assert result.success
        assert result.num_calls == 1

    def test_multiple_tool_calls(self, parser):
        """Test multiple tool calls in text."""
        text = '{"name": "a", "arguments": {}} text {"name": "b", "arguments": {}}'
        result = parser.parse(text)

        assert result.success
        assert result.num_calls == 2

    def test_array_format(self, parser):
        """Test array of tool calls."""
        text = '[{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]'
        result = parser.parse(text)

        assert result.success
        assert result.num_calls == 2

    def test_xml_format(self, parser):
        """Test XML-wrapped tool calls."""
        text = '<tool_call>{"name": "test", "arguments": {}}</tool_call>'
        result = parser.parse(text)

        assert result.success
        assert result.num_calls == 1


class TestErrorRecovery:
    """Tests for error recovery."""

    def test_recovers_from_malformed(self, parser):
        """Test recovery from malformed JSON."""
        text = '{"name": invalid} {"name": "valid", "arguments": {}}'
        result = parser.parse(text)

        # Should find the valid one
        assert result.num_calls >= 1

    def test_continues_after_error(self, parser):
        """Test parsing continues after error."""
        text = '''
        {"broken":
        {"name": "good", "arguments": {}}
        '''
        result = parser.parse(text)

        # Should still find the good one
        assert isinstance(result, ParseResult)

    def test_empty_input(self, parser):
        """Test empty input handling."""
        result = parser.parse("")

        assert result.success
        assert result.num_calls == 0


class TestComplexCases:
    """Tests for complex parsing scenarios."""

    def test_nested_json(self, parser):
        """Test deeply nested JSON."""
        text = '{"name": "deep", "arguments": {"a": {"b": {"c": {"d": 1}}}}}'
        result = parser.parse(text)

        assert result.success
        assert result.tool_calls[0].arguments["a"]["b"]["c"]["d"] == 1

    def test_string_with_braces(self, parser):
        """Test strings containing braces."""
        text = '{"name": "test", "arguments": {"code": "if (x) { return y; }"}}'
        result = parser.parse(text)

        assert result.success
        assert "{" in result.tool_calls[0].arguments["code"]

    def test_escaped_quotes(self, parser):
        """Test escaped quotes in strings."""
        text = '{"name": "test", "arguments": {"msg": "He said \\"hi\\""}}'
        result = parser.parse(text)

        assert result.success

    def test_unicode_content(self, parser):
        """Test unicode in content."""
        text = '{"name": "greet", "arguments": {"text": "こんにちは 👋"}}'
        result = parser.parse(text)

        assert result.success
        assert "👋" in result.tool_calls[0].arguments["text"]

    def test_mixed_formats(self, parser):
        """Test mixed JSON and XML formats."""
        text = '''
        {"name": "json1", "arguments": {}}
        <tool_call>{"name": "xml1", "arguments": {}}</tool_call>
        {"name": "json2", "arguments": {}}
        '''
        result = parser.parse(text)

        assert result.num_calls == 3