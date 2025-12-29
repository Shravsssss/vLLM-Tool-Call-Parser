"""Tests for streaming parsing functionality."""

import pytest

from parser_benchmark.parsers import StreamingParser


class TestStreamingParsing:
    """Tests for streaming parser."""

    def test_basic_streaming(self):
        """Test basic streaming functionality."""
        parser = StreamingParser()

        # Simulate token-by-token arrival
        tokens = ['{"', 'name', '": "', 'test', '", "', 'arguments', '": {}}']

        all_calls = []
        for token in tokens:
            calls = parser.feed(token)
            all_calls.extend(calls)

        assert len(all_calls) == 1
        assert all_calls[0].name == "test"

    def test_multiple_calls_streaming(self):
        """Test streaming with multiple tool calls."""
        parser = StreamingParser()

        text = '{"name": "a", "arguments": {}} {"name": "b", "arguments": {}}'

        # Feed in chunks
        chunk_size = 10
        all_calls = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            calls = parser.feed(chunk)
            all_calls.extend(calls)

        assert len(all_calls) == 2

    def test_partial_then_complete(self):
        """Test partial JSON followed by completion."""
        parser = StreamingParser()

        # Partial - should return nothing
        calls = parser.feed('{"name": "incomplete')
        assert len(calls) == 0

        # Complete it
        calls = parser.feed('", "arguments": {}}')
        assert len(calls) == 1

    def test_streaming_with_text(self):
        """Test streaming with surrounding text."""
        parser = StreamingParser()

        parser.feed("I will call a function: ")
        calls = parser.feed('{"name": "func", "arguments": {}}')

        assert len(calls) == 1

    def test_reset_clears_state(self):
        """Test that reset clears accumulated state."""
        parser = StreamingParser()

        parser.feed('{"name": "test')
        parser.reset()
        parser.feed('{"name": "new", "arguments": {}}')

        result = parser.get_result()
        assert result.num_calls == 1
        assert result.tool_calls[0].name == "new"