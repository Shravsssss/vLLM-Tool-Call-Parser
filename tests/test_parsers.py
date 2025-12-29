"""Tests for RegexParser."""

import pytest

from parser_benchmark.models import ParseResult
from parser_benchmark.parsers import RegexParser


@pytest.fixture
def parser():
    """Create a parser instance for tests."""
    return RegexParser()


class TestRegexParserBasics:
    """Basic functionality tests."""

    def test_parser_name(self, parser):
        """Test parser has correct name."""
        assert parser.name == "regex-parser"

    def test_returns_parse_result(self, parser):
        """Test parse returns ParseResult."""
        result = parser.parse("{}")
        assert isinstance(result, ParseResult)

    def test_includes_timing(self, parser):
        """Test parse includes timing info."""
        result = parser.parse('{"name": "test", "arguments": {}}')
        assert result.parse_time_ms > 0


class TestSimpleToolCalls:
    """Tests for simple tool call formats."""

    def test_simple_tool_call(self, parser):
        """Test parsing a simple tool call."""
        text = '{"name": "get_weather", "arguments": {"city": "NYC"}}'
        result = parser.parse(text)

        assert result.success
        assert result.num_calls == 1
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].arguments["city"] == "NYC"

    def test_tool_call_with_string_arg(self, parser):
        """Test tool call with string argument."""
        text = '{"name": "search", "arguments": {"query": "python tutorial"}}'
        result = parser.parse(text)

        assert result.success
        assert result.tool_calls[0].arguments["query"] == "python tutorial"

    def test_tool_call_with_number_arg(self, parser):
        """Test tool call with numeric argument."""
        text = '{"name": "calculate", "arguments": {"x": 42, "y": 3.14}}'
        result = parser.parse(text)

        assert result.success
        assert result.tool_calls[0].arguments["x"] == 42
        assert result.tool_calls[0].arguments["y"] == 3.14

    def test_tool_call_with_boolean_arg(self, parser):
        """Test tool call with boolean argument."""
        text = '{"name": "toggle", "arguments": {"enabled": true}}'
        result = parser.parse(text)

        assert result.success
        assert result.tool_calls[0].arguments["enabled"] is True

    def test_tool_call_with_null_arg(self, parser):
        """Test tool call with null argument."""
        text = '{"name": "clear", "arguments": {"value": null}}'
        result = parser.parse(text)

        assert result.success
        assert result.tool_calls[0].arguments["value"] is None


class TestComplexFormats:
    """Tests for complex tool call formats."""

    def test_array_of_tool_calls(self, parser):
        """Test parsing multiple tool calls in array."""
        text = '''[
            {"name": "func1", "arguments": {"a": 1}},
            {"name": "func2", "arguments": {"b": 2}}
        ]'''
        result = parser.parse(text)

        assert result.success
        assert result.num_calls == 2
        assert result.tool_calls[0].name == "func1"
        assert result.tool_calls[1].name == "func2"

    def test_xml_wrapped_tool_call(self, parser):
        """Test parsing XML-wrapped tool call."""
        text = '<tool_call>{"name": "api_call", "arguments": {"url": "http://test.com"}}</tool_call>'
        result = parser.parse(text)

        assert result.success
        assert result.num_calls == 1
        assert result.tool_calls[0].name == "api_call"

    def test_embedded_in_text(self, parser):
        """Test extracting tool call embedded in text."""
        text = '''I will help you with that.
        {"name": "search", "arguments": {"query": "test"}}
        Let me know if you need more.'''
        result = parser.parse(text)

        assert result.success
        assert result.num_calls == 1
        assert result.tool_calls[0].name == "search"

    def test_nested_arguments(self, parser):
        """Test tool call with nested object in arguments."""
        text = '{"name": "create", "arguments": {"data": {"nested": "value"}}}'
        result = parser.parse(text)

        assert result.success
        assert result.tool_calls[0].arguments["data"]["nested"] == "value"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_input(self, parser):
        """Test parsing empty string."""
        result = parser.parse("")

        assert result.success
        assert result.num_calls == 0

    def test_no_tool_calls(self, parser):
        """Test parsing text with no tool calls."""
        result = parser.parse("Just some regular text with no JSON.")

        assert result.success
        assert result.num_calls == 0

    def test_empty_arguments(self, parser):
        """Test tool call with empty arguments."""
        text = '{"name": "ping", "arguments": {}}'
        result = parser.parse(text)

        assert result.success
        assert result.tool_calls[0].arguments == {}

    def test_whitespace_handling(self, parser):
        """Test parsing with extra whitespace."""
        text = '  { "name" : "test" , "arguments" : { } }  '
        result = parser.parse(text)

        assert result.success
        assert result.num_calls == 1

    def test_unicode_in_arguments(self, parser):
        """Test tool call with unicode characters."""
        text = '{"name": "greet", "arguments": {"message": "Hello, world!"}}'
        result = parser.parse(text)

        assert result.success
        assert "Hello" in result.tool_calls[0].arguments["message"]


class TestWithId:
    """Tests for tool calls with ID field."""

    def test_tool_call_with_id(self, parser):
        """Test parsing tool call that includes id field."""
        text = '{"id": "call_abc123", "name": "test", "arguments": {}}'
        result = parser.parse(text)

        assert result.success
        assert result.tool_calls[0].id == "call_abc123"

    def test_tool_call_without_id(self, parser):
        """Test that id is None when not provided."""
        text = '{"name": "test", "arguments": {}}'
        result = parser.parse(text)

        assert result.success
        assert result.tool_calls[0].id is None


class TestArrayArguments:
    """Tests for tool calls with array arguments."""

    def test_simple_array_argument(self, parser):
        """Test tool call with simple array argument."""
        text = '{"name": "process_items", "arguments": {"items": [1, 2, 3]}}'
        result = parser.parse(text)

        assert result.success
        assert result.num_calls == 1
        assert result.tool_calls[0].arguments["items"] == [1, 2, 3]

    def test_array_of_strings(self, parser):
        """Test tool call with array of strings."""
        text = '{"name": "send_emails", "arguments": {"recipients": ["a@test.com", "b@test.com"]}}'
        result = parser.parse(text)

        assert result.success
        assert len(result.tool_calls[0].arguments["recipients"]) == 2

    def test_array_of_objects(self, parser):
        """Test tool call with array of objects."""
        text = '{"name": "create_users", "arguments": {"users": [{"name": "Alice"}, {"name": "Bob"}]}}'
        result = parser.parse(text)

        assert result.success
        assert result.tool_calls[0].arguments["users"][0]["name"] == "Alice"

    def test_empty_array(self, parser):
        """Test tool call with empty array argument."""
        text = '{"name": "clear_list", "arguments": {"items": []}}'
        result = parser.parse(text)

        assert result.success
        assert result.tool_calls[0].arguments["items"] == []

    def test_mixed_array(self, parser):
        """Test tool call with mixed type array."""
        text = '{"name": "mixed", "arguments": {"data": [1, "two", true, null]}}'
        result = parser.parse(text)

        assert result.success
        assert result.tool_calls[0].arguments["data"] == [1, "two", True, None]


class TestMultipleXMLWrapped:
    """Tests for multiple XML-wrapped tool calls."""

    def test_two_xml_wrapped_calls(self, parser):
        """Test parsing two XML-wrapped tool calls."""
        text = '''<tool_call>{"name": "func1", "arguments": {"a": 1}}</tool_call>
        <tool_call>{"name": "func2", "arguments": {"b": 2}}</tool_call>'''
        result = parser.parse(text)

        assert result.success
        assert result.num_calls == 2
        assert result.tool_calls[0].name == "func1"
        assert result.tool_calls[1].name == "func2"

    def test_three_xml_wrapped_calls(self, parser):
        """Test parsing three XML-wrapped tool calls."""
        text = '''<tool_call>{"name": "step1", "arguments": {}}</tool_call>
        <tool_call>{"name": "step2", "arguments": {}}</tool_call>
        <tool_call>{"name": "step3", "arguments": {}}</tool_call>'''
        result = parser.parse(text)

        assert result.success
        assert result.num_calls == 3

    def test_xml_wrapped_with_surrounding_text(self, parser):
        """Test XML-wrapped calls with surrounding text."""
        text = '''I will execute these functions:
        <tool_call>{"name": "first", "arguments": {"x": 1}}</tool_call>
        And then:
        <tool_call>{"name": "second", "arguments": {"y": 2}}</tool_call>
        Done!'''
        result = parser.parse(text)

        assert result.success
        assert result.num_calls == 2

    def test_xml_wrapped_on_same_line(self, parser):
        """Test multiple XML-wrapped calls on same line."""
        text = '<tool_call>{"name": "a", "arguments": {}}</tool_call><tool_call>{"name": "b", "arguments": {}}</tool_call>'
        result = parser.parse(text)

        assert result.success
        assert result.num_calls == 2


class TestLongFunctionNames:
    """Tests for very long function names."""

    def test_long_function_name(self, parser):
        """Test tool call with a very long function name."""
        long_name = "get_user_profile_with_extended_metadata_and_preferences_v2"
        text = f'{{"name": "{long_name}", "arguments": {{}}}}'
        result = parser.parse(text)

        assert result.success
        assert result.tool_calls[0].name == long_name

    def test_very_long_function_name(self, parser):
        """Test tool call with an extremely long function name (100+ chars)."""
        long_name = "a" * 150
        text = f'{{"name": "{long_name}", "arguments": {{}}}}'
        result = parser.parse(text)

        assert result.success
        assert result.tool_calls[0].name == long_name
        assert len(result.tool_calls[0].name) == 150

    def test_function_name_with_underscores(self, parser):
        """Test function name with many underscores."""
        name = "get__double__underscore__function__name"
        text = f'{{"name": "{name}", "arguments": {{}}}}'
        result = parser.parse(text)

        assert result.success
        assert result.tool_calls[0].name == name

    def test_function_name_with_numbers(self, parser):
        """Test function name with numbers."""
        name = "api_v2_endpoint_123_handler"
        text = f'{{"name": "{name}", "arguments": {{}}}}'
        result = parser.parse(text)

        assert result.success
        assert result.tool_calls[0].name == name


class TestSpecialCharacters:
    """Tests for special characters in arguments."""

    def test_unicode_emoji(self, parser):
        """Test arguments containing emoji."""
        text = '{"name": "send_message", "arguments": {"text": "Hello! 👋🎉"}}'
        result = parser.parse(text)

        assert result.success
        assert "👋" in result.tool_calls[0].arguments["text"]

    def test_newlines_in_string(self, parser):
        """Test arguments with escaped newlines."""
        text = '{"name": "format", "arguments": {"text": "line1\\nline2\\nline3"}}'
        result = parser.parse(text)

        assert result.success
        assert "\\n" in result.tool_calls[0].arguments["text"] or "\n" in result.tool_calls[0].arguments["text"]

    def test_quotes_in_string(self, parser):
        """Test arguments with escaped quotes."""
        text = '{"name": "quote", "arguments": {"text": "He said \\"hello\\""}}'
        result = parser.parse(text)

        assert result.success
        assert "hello" in result.tool_calls[0].arguments["text"]

    def test_backslashes(self, parser):
        """Test arguments with backslashes (file paths)."""
        text = '{"name": "read_file", "arguments": {"path": "C:\\\\Users\\\\test\\\\file.txt"}}'
        result = parser.parse(text)

        assert result.success
        assert "Users" in result.tool_calls[0].arguments["path"]

    def test_special_json_chars(self, parser):
        """Test arguments with special JSON characters."""
        text = '{"name": "process", "arguments": {"data": "tab:\\there, return:\\r"}}'
        result = parser.parse(text)

        assert result.success

    def test_unicode_languages(self, parser):
        """Test arguments with various unicode languages."""
        text = '{"name": "translate", "arguments": {"texts": {"chinese": "你好", "japanese": "こんにちは", "arabic": "مرحبا"}}}'
        result = parser.parse(text)

        assert result.success
        assert result.tool_calls[0].arguments["texts"]["chinese"] == "你好"

    def test_html_in_arguments(self, parser):
        """Test arguments containing HTML."""
        text = '{"name": "render", "arguments": {"html": "<div class=\\"test\\">Hello</div>"}}'
        result = parser.parse(text)

        assert result.success
        assert "div" in result.tool_calls[0].arguments["html"]

    def test_url_in_arguments(self, parser):
        """Test arguments containing URLs with special chars."""
        text = '{"name": "fetch", "arguments": {"url": "https://api.example.com/search?q=test&limit=10"}}'
        result = parser.parse(text)

        assert result.success
        assert "example.com" in result.tool_calls[0].arguments["url"]


class TestMalformedJSON:
    """Tests for malformed JSON handling."""

    def test_missing_closing_brace_graceful(self, parser):
        """Test that missing closing brace doesn't crash."""
        text = '{"name": "test", "arguments": {"x": 1}'
        result = parser.parse(text)

        assert isinstance(result, ParseResult)

    def test_trailing_comma_handling(self, parser):
        """Test handling of trailing comma in arguments."""
        text = '{"name": "test", "arguments": {"x": 1,}}'
        result = parser.parse(text)

        assert isinstance(result, ParseResult)

    def test_single_quotes_handling(self, parser):
        """Test handling of single quotes instead of double quotes."""
        text = "{'name': 'test', 'arguments': {'x': 1}}"
        result = parser.parse(text)

        assert isinstance(result, ParseResult)

    def test_unquoted_keys(self, parser):
        """Test handling of unquoted keys."""
        text = '{name: "test", arguments: {}}'
        result = parser.parse(text)

        assert isinstance(result, ParseResult)

    def test_mixed_valid_invalid(self, parser):
        """Test text with both valid and invalid JSON."""
        text = '''Invalid: {"name": broken
        Valid: {"name": "good", "arguments": {}}'''
        result = parser.parse(text)

        assert isinstance(result, ParseResult)


class TestParseMultiple:
    """Tests for parse_multiple helper function."""

    def test_parse_multiple_basic(self, parser):
        """Test parsing multiple texts."""
        texts = [
            '{"name": "func1", "arguments": {}}',
            '{"name": "func2", "arguments": {"x": 1}}',
            '{"name": "func3", "arguments": {"y": 2}}',
        ]
        results = parser.parse_multiple(texts)

        assert len(results) == 3
        assert all(isinstance(r, ParseResult) for r in results)
        assert results[0].tool_calls[0].name == "func1"
        assert results[1].tool_calls[0].name == "func2"
        assert results[2].tool_calls[0].name == "func3"

    def test_parse_multiple_empty_list(self, parser):
        """Test parse_multiple with empty list."""
        results = parser.parse_multiple([])

        assert results == []

    def test_parse_multiple_with_failures(self, parser):
        """Test parse_multiple with some invalid inputs."""
        texts = [
            '{"name": "valid", "arguments": {}}',
            'not json at all',
            '{"name": "also_valid", "arguments": {}}',
        ]
        results = parser.parse_multiple(texts)

        assert len(results) == 3
        assert results[0].num_calls == 1
        assert results[1].num_calls == 0
        assert results[2].num_calls == 1

    def test_parse_multiple_preserves_order(self, parser):
        """Test that parse_multiple preserves input order."""
        texts = [f'{{"name": "func{i}", "arguments": {{}}}}' for i in range(10)]
        results = parser.parse_multiple(texts)

        for i, result in enumerate(results):
            assert result.tool_calls[0].name == f"func{i}"
