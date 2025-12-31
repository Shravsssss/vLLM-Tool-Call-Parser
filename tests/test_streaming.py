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


class TestAdvancedStreamingEdgeCases:
    """Advanced streaming tests for edge cases."""

    def test_xml_attr_streaming(self):
        """Test XML attribute format in streaming mode."""
        parser = StreamingParser()

        # Feed XML attribute format in chunks
        parser.feed('<get_weather ')
        calls = parser.feed('city="Tokyo" ')
        assert len(calls) == 0  # Not complete yet

        calls = parser.feed('unit="fahrenheit"/>')
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert calls[0].arguments["city"] == "Tokyo"

    def test_split_at_attribute_boundary(self):
        """Test split exactly at attribute name/value boundary."""
        parser = StreamingParser()

        # Split in middle of attribute value
        parser.feed('<search query="Python')
        calls = parser.feed(' tutorials"/>')

        assert len(calls) == 1
        assert calls[0].name == "search"
        assert calls[0].arguments["query"] == "Python tutorials"

    def test_very_slow_token_stream(self):
        """Test character-by-character streaming."""
        parser = StreamingParser()

        json_str = '{"name": "slow_test", "arguments": {"key": "value"}}'

        all_calls = []
        for char in json_str:
            calls = parser.feed(char)
            all_calls.extend(calls)

        assert len(all_calls) == 1
        assert all_calls[0].name == "slow_test"
        assert all_calls[0].arguments["key"] == "value"

    def test_streaming_with_noise(self):
        """Test streaming with noise characters between valid calls."""
        parser = StreamingParser()

        parser.feed("Some text before... ")
        parser.feed('{"name": "first", "arguments": {}}')
        parser.feed("\n\n---separator---\n\n")
        calls = parser.feed('{"name": "second", "arguments": {}}')

        result = parser.get_result()
        assert result.num_calls == 2

    def test_multiple_calls_slow_stream(self):
        """Test multiple tool calls with slow streaming."""
        parser = StreamingParser()

        # First call in chunks
        parser.feed('{"name": ')
        parser.feed('"call_one", ')
        parser.feed('"arguments": {}}')

        # Separator
        parser.feed(' then ')

        # Second call in chunks
        parser.feed('{"name": ')
        parser.feed('"call_two", ')
        parser.feed('"arguments": {}}')

        result = parser.get_result()
        assert result.num_calls == 2
        assert result.tool_calls[0].name == "call_one"
        assert result.tool_calls[1].name == "call_two"

    def test_stream_interrupt_and_resume(self):
        """Test incomplete stream followed by reset and new stream."""
        parser = StreamingParser()

        # Start but don't finish
        parser.feed('{"name": "interrupted"')

        # Check we have nothing complete
        result = parser.get_result()
        assert result.num_calls == 0

        # Reset and start fresh
        parser.reset()
        parser.feed('{"name": "fresh_start", "arguments": {"x": 1}}')

        result = parser.get_result()
        assert result.num_calls == 1
        assert result.tool_calls[0].name == "fresh_start"

    def test_xml_wrapper_streaming(self):
        """Test XML wrapper format in streaming mode.

        Note: The inner JSON is detected when complete, before the XML wrapper closes.
        """
        parser = StreamingParser()

        # Feed <tool_call> wrapper format in chunks
        parser.feed('<tool_call>')
        parser.feed('{"name": "wrapped_func"')
        # When JSON object closes, it's detected even inside incomplete XML wrapper
        calls = parser.feed(', "arguments": {"param": "value"}}')
        parser.feed('</tool_call>')

        # Tool call found when JSON completed (third feed)
        assert len(calls) == 1
        assert calls[0].name == "wrapped_func"
        assert calls[0].arguments["param"] == "value"

        # Final result also has exactly 1 tool call
        result = parser.get_result()
        assert result.num_calls == 1