"""Tests for IncrementalParser."""

import pytest

from parser_benchmark.models import ParseResult
from parser_benchmark.parsers import IncrementalParser, StreamingParser


@pytest.fixture
def parser():
    """Create parser instance."""
    return IncrementalParser()


class TestIncrementalParserBasics:
    """Basic functionality tests."""

    def test_parser_name(self, parser):
        """Test parser has correct name."""
        assert parser.name == "incremental-parser"

    def test_simple_tool_call(self, parser):
        """Test parsing simple tool call."""
        text = '{"name": "test", "arguments": {"x": 1}}'
        result = parser.parse(text)

        assert result.success
        assert result.num_calls == 1
        assert result.tool_calls[0].name == "test"

    def test_nested_objects(self, parser):
        """Test parsing nested JSON objects."""
        text = '{"name": "deep", "arguments": {"a": {"b": {"c": 1}}}}'
        result = parser.parse(text)

        assert result.success
        assert result.tool_calls[0].arguments["a"]["b"]["c"] == 1

    def test_embedded_in_text(self, parser):
        """Test extracting from surrounding text."""
        text = 'Some text {"name": "func", "arguments": {}} more text'
        result = parser.parse(text)

        assert result.success
        assert result.num_calls == 1

    def test_multiple_objects(self, parser):
        """Test multiple separate tool calls."""
        text = '{"name": "a", "arguments": {}} {"name": "b", "arguments": {}}'
        result = parser.parse(text)

        assert result.success
        assert result.num_calls == 2

    def test_array_of_calls(self, parser):
        """Test parsing array of tool calls."""
        text = '[{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]'
        result = parser.parse(text)

        assert result.success
        assert result.num_calls == 2

    def test_xml_wrapped(self, parser):
        """Test XML-wrapped tool calls."""
        text = '<tool_call>{"name": "test", "arguments": {}}</tool_call>'
        result = parser.parse(text)

        assert result.success
        assert result.num_calls == 1


class TestStreamingParser:
    """Tests for streaming functionality."""

    def test_streaming_basic(self):
        """Test basic streaming parsing."""
        parser = StreamingParser()

        # Feed partial chunks
        calls = parser.feed('{"name": "te')
        assert len(calls) == 0

        calls = parser.feed('st", "arguments": {}}')
        assert len(calls) == 1
        assert calls[0].name == "test"

    def test_streaming_multiple_chunks(self):
        """Test streaming with many small chunks."""
        parser = StreamingParser()
        text = '{"name": "test", "arguments": {"x": 1}}'

        # Feed character by character
        for char in text[:-1]:
            calls = parser.feed(char)
            assert len(calls) == 0

        # Final character completes the call
        calls = parser.feed(text[-1])
        assert len(calls) == 1

    def test_streaming_reset(self):
        """Test parser reset."""
        parser = StreamingParser()
        parser.feed('{"name": "test", "arguments": {}}')

        result = parser.get_result()
        assert result.num_calls == 1

        parser.reset()
        result = parser.get_result()
        assert result.num_calls == 0


class TestEdgeCases:
    """Edge case tests for incremental parser."""

    def test_escaped_quotes(self, parser):
        """Test handling escaped quotes in strings."""
        text = '{"name": "test", "arguments": {"msg": "say \\"hello\\""}}'
        result = parser.parse(text)

        assert result.success
        assert "hello" in result.tool_calls[0].arguments["msg"]

    def test_unicode(self, parser):
        """Test unicode handling."""
        text = '{"name": "greet", "arguments": {"text": "Hello 👋"}}'
        result = parser.parse(text)

        assert result.success
        assert "👋" in result.tool_calls[0].arguments["text"]

    def test_deeply_nested(self, parser):
        """Test deeply nested structures."""
        text = '{"name": "deep", "arguments": {"l1": {"l2": {"l3": {"l4": "value"}}}}}'
        result = parser.parse(text)

        assert result.success

    def test_empty_input(self, parser):
        """Test empty input."""
        result = parser.parse("")

        assert result.success
        assert result.num_calls == 0

    def test_incomplete_json(self, parser):
        """Test incomplete JSON doesn't crash."""
        text = '{"name": "test", "arguments": {'
        result = parser.parse(text)

        assert result.success
        assert result.num_calls == 0