"""Comprehensive edge case tests for all parsers.

Categories:
1. Malformed JSON (10 cases)
2. Nested Structures (8 cases)
3. Unicode & Special Characters (10 cases)
4. Streaming Boundaries (6 cases)
5. Injection Attempts (6 cases)
6. Large Inputs (5 cases)
7. Format Variations (10 cases)
"""

import pytest

from parser_benchmark.parsers import RegexParser, IncrementalParser, StateMachineParser
from parser_benchmark.models import ParseResult


# All parsers to test
@pytest.fixture(params=[RegexParser, IncrementalParser, StateMachineParser])
def parser(request):
    """Parameterized fixture for all parsers."""
    return request.param()


# =============================================================================
# Category 1: Malformed JSON (10 cases)
# =============================================================================

class TestMalformedJSON:
    """Tests for malformed JSON handling."""

    def test_missing_closing_brace(self, parser):
        """Test missing closing brace."""
        text = '{"name": "test", "arguments": {"x": 1}'
        result = parser.parse(text)
        assert isinstance(result, ParseResult)

    def test_missing_opening_brace(self, parser):
        """Test missing opening brace."""
        text = '"name": "test", "arguments": {}}'
        result = parser.parse(text)
        assert isinstance(result, ParseResult)

    def test_trailing_comma_object(self, parser):
        """Test trailing comma in object."""
        text = '{"name": "test", "arguments": {"x": 1,}}'
        result = parser.parse(text)
        assert isinstance(result, ParseResult)

    def test_trailing_comma_array(self, parser):
        """Test trailing comma in array."""
        text = '{"name": "test", "arguments": {"items": [1, 2, 3,]}}'
        result = parser.parse(text)
        assert isinstance(result, ParseResult)

    def test_single_quotes(self, parser):
        """Test single quotes instead of double quotes."""
        text = "{'name': 'test', 'arguments': {}}"
        result = parser.parse(text)
        assert isinstance(result, ParseResult)

    def test_unquoted_keys(self, parser):
        """Test unquoted object keys."""
        text = '{name: "test", arguments: {}}'
        result = parser.parse(text)
        assert isinstance(result, ParseResult)

    def test_unquoted_string_value(self, parser):
        """Test unquoted string value."""
        text = '{"name": test, "arguments": {}}'
        result = parser.parse(text)
        assert isinstance(result, ParseResult)

    def test_extra_commas(self, parser):
        """Test multiple consecutive commas."""
        text = '{"name": "test",, "arguments": {}}'
        result = parser.parse(text)
        assert isinstance(result, ParseResult)

    def test_missing_colon(self, parser):
        """Test missing colon after key."""
        text = '{"name" "test", "arguments": {}}'
        result = parser.parse(text)
        assert isinstance(result, ParseResult)

    def test_truncated_json(self, parser):
        """Test truncated/incomplete JSON."""
        text = '{"name": "test", "argum'
        result = parser.parse(text)
        assert isinstance(result, ParseResult)


# =============================================================================
# Category 2: Nested Structures (8 cases)
# =============================================================================

class TestNestedStructures:
    """Tests for deeply nested JSON structures."""

    def test_depth_5(self, parser):
        """Test 5 levels of nesting."""
        text = '{"name": "test", "arguments": {"l1": {"l2": {"l3": {"l4": {"l5": 1}}}}}}'
        result = parser.parse(text)
        assert result.success

    def test_depth_10(self, parser):
        """Test 10 levels of nesting."""
        nested = '{"deep": ' * 10 + '1' + '}' * 10
        text = f'{{"name": "test", "arguments": {nested}}}'
        result = parser.parse(text)
        assert isinstance(result, ParseResult)

    def test_array_in_object(self, parser):
        """Test arrays nested in objects."""
        text = '{"name": "test", "arguments": {"items": [{"a": 1}, {"b": 2}]}}'
        result = parser.parse(text)
        assert result.success

    def test_object_in_array(self, parser):
        """Test objects nested in arrays."""
        text = '[{"name": "a", "arguments": {"data": [1, 2]}}, {"name": "b", "arguments": {}}]'
        result = parser.parse(text)
        assert result.success

    def test_mixed_nesting(self, parser):
        """Test mixed array/object nesting."""
        text = '{"name": "test", "arguments": {"a": [{"b": [{"c": 1}]}]}}'
        result = parser.parse(text)
        assert result.success

    def test_empty_nested_objects(self, parser):
        """Test empty nested objects."""
        text = '{"name": "test", "arguments": {"a": {"b": {"c": {}}}}}'
        result = parser.parse(text)
        assert result.success

    def test_empty_nested_arrays(self, parser):
        """Test empty nested arrays."""
        text = '{"name": "test", "arguments": {"a": [[], [[]]]}}'
        result = parser.parse(text)
        assert result.success

    def test_json_string_in_arguments(self, parser):
        """Test JSON string as argument value."""
        text = '{"name": "test", "arguments": {"data": "{\\"nested\\": true}"}}'
        result = parser.parse(text)
        assert result.success


# =============================================================================
# Category 3: Unicode & Special Characters (10 cases)
# =============================================================================

class TestUnicodeSpecialChars:
    """Tests for unicode and special characters."""

    def test_emoji_simple(self, parser):
        """Test simple emoji."""
        text = '{"name": "test", "arguments": {"emoji": "👍"}}'
        result = parser.parse(text)
        assert result.success

    def test_emoji_complex(self, parser):
        """Test complex emoji sequences."""
        text = '{"name": "test", "arguments": {"emoji": "👨‍👩‍👧‍👦🏳️‍🌈"}}'
        result = parser.parse(text)
        assert result.success

    def test_chinese(self, parser):
        """Test Chinese characters."""
        text = '{"name": "test", "arguments": {"text": "你好世界"}}'
        result = parser.parse(text)
        assert result.success

    def test_arabic(self, parser):
        """Test Arabic characters (RTL)."""
        text = '{"name": "test", "arguments": {"text": "مرحبا بالعالم"}}'
        result = parser.parse(text)
        assert result.success

    def test_japanese(self, parser):
        """Test Japanese characters."""
        text = '{"name": "test", "arguments": {"text": "こんにちは世界"}}'
        result = parser.parse(text)
        assert result.success

    def test_korean(self, parser):
        """Test Korean characters."""
        text = '{"name": "test", "arguments": {"text": "안녕하세요"}}'
        result = parser.parse(text)
        assert result.success

    def test_escape_sequences(self, parser):
        """Test JSON escape sequences."""
        text = '{"name": "test", "arguments": {"text": "line1\\nline2\\ttab\\r\\n"}}'
        result = parser.parse(text)
        assert result.success

    def test_unicode_escapes(self, parser):
        """Test unicode escape sequences."""
        text = '{"name": "test", "arguments": {"text": "\\u0048\\u0065\\u006c\\u006c\\u006f"}}'
        result = parser.parse(text)
        assert result.success

    def test_null_character(self, parser):
        """Test null character in string."""
        text = '{"name": "test", "arguments": {"text": "before\\u0000after"}}'
        result = parser.parse(text)
        assert isinstance(result, ParseResult)

    def test_special_whitespace(self, parser):
        """Test special whitespace characters."""
        text = '{"name": "test", "arguments": {"text": "a\\u00A0b\\u2003c"}}'
        result = parser.parse(text)
        assert result.success


# =============================================================================
# Category 4: Streaming Boundaries (6 cases)
# =============================================================================

class TestStreamingBoundaries:
    """Tests for streaming/chunked parsing."""

    def test_split_at_brace(self, parser):
        """Test split exactly at opening brace."""
        text = '{"name": "test", "arguments": {}}'
        # Simulate getting text in pieces
        result = parser.parse(text)
        assert result.success

    def test_split_in_key(self, parser):
        """Test content that would split in middle of key."""
        text = '{"name": "test_function_name", "arguments": {}}'
        result = parser.parse(text)
        assert result.success
        assert result.tool_calls[0].name == "test_function_name"

    def test_split_in_string_value(self, parser):
        """Test content that would split in string value."""
        text = '{"name": "test", "arguments": {"message": "hello world how are you"}}'
        result = parser.parse(text)
        assert result.success

    def test_split_at_comma(self, parser):
        """Test split at comma between fields."""
        text = '{"name": "test", "arguments": {"a": 1, "b": 2}}'
        result = parser.parse(text)
        assert result.success

    def test_split_in_number(self, parser):
        """Test split in middle of number."""
        text = '{"name": "test", "arguments": {"value": 123456789}}'
        result = parser.parse(text)
        assert result.success

    def test_split_between_objects(self, parser):
        """Test split between multiple objects."""
        text = '{"name": "a", "arguments": {}} {"name": "b", "arguments": {}}'
        result = parser.parse(text)
        assert result.num_calls == 2


# =============================================================================
# Category 5: Injection Attempts (6 cases)
# =============================================================================

class TestInjectionAttempts:
    """Tests for potential injection/confusion attacks."""

    def test_json_in_string(self, parser):
        """Test JSON object inside a string value."""
        text = '{"name": "test", "arguments": {"data": "{\\"fake\\": \\"call\\"}"}}'
        result = parser.parse(text)
        assert result.success
        assert result.num_calls == 1

    def test_tool_call_in_string(self, parser):
        """Test tool call syntax inside string."""
        text = '{"name": "real", "arguments": {"text": "{\\"name\\": \\"fake\\", \\"arguments\\": {}}"}}'
        result = parser.parse(text)
        assert result.success
        assert result.tool_calls[0].name == "real"

    def test_xml_in_string(self, parser):
        """Test XML markers inside string."""
        text = '{"name": "test", "arguments": {"html": "<tool_call>fake</tool_call>"}}'
        result = parser.parse(text)
        assert result.success

    def test_nested_quotes(self, parser):
        """Test deeply nested escaped quotes."""
        text = '{"name": "test", "arguments": {"text": "a\\"b\\"c\\"d"}}'
        result = parser.parse(text)
        assert result.success

    def test_backslash_sequence(self, parser):
        """Test multiple backslashes."""
        text = '{"name": "test", "arguments": {"path": "C:\\\\Users\\\\test"}}'
        result = parser.parse(text)
        assert result.success

    def test_comment_like_content(self, parser):
        """Test content that looks like comments."""
        text = '{"name": "test", "arguments": {"code": "// comment\\n/* block */"}}'
        result = parser.parse(text)
        assert result.success


# =============================================================================
# Category 6: Large Inputs (5 cases)
# =============================================================================

class TestLargeInputs:
    """Tests for large inputs and performance edge cases."""

    def test_long_function_name(self, parser):
        """Test very long function name."""
        long_name = "a" * 500
        text = f'{{"name": "{long_name}", "arguments": {{}}}}'
        result = parser.parse(text)
        assert result.success

    def test_many_arguments(self, parser):
        """Test object with many arguments."""
        args = ", ".join(f'"arg{i}": {i}' for i in range(50))
        text = f'{{"name": "test", "arguments": {{{args}}}}}'
        result = parser.parse(text)
        assert result.success

    def test_large_array(self, parser):
        """Test large array argument."""
        items = list(range(500))
        text = f'{{"name": "test", "arguments": {{"items": {items}}}}}'
        result = parser.parse(text)
        assert result.success

    def test_long_string_value(self, parser):
        """Test very long string value."""
        long_string = "x" * 10000
        text = f'{{"name": "test", "arguments": {{"text": "{long_string}"}}}}'
        result = parser.parse(text)
        assert result.success

    def test_many_tool_calls(self, parser):
        """Test many consecutive tool calls."""
        calls = '{"name": "test", "arguments": {}} ' * 20
        result = parser.parse(calls)
        assert result.num_calls == 20


# =============================================================================
# Category 7: Format Variations (10 cases)
# =============================================================================

class TestFormatVariations:
    """Tests for different tool call formats."""

    def test_compact_format(self, parser):
        """Test compact JSON (no spaces)."""
        text = '{"name":"test","arguments":{"x":1}}'
        result = parser.parse(text)
        assert result.success

    def test_pretty_format(self, parser):
        """Test pretty-printed JSON."""
        text = '''{
            "name": "test",
            "arguments": {
                "x": 1
            }
        }'''
        result = parser.parse(text)
        assert result.success

    def test_with_id_field(self, parser):
        """Test with id field present."""
        text = '{"id": "call_123", "name": "test", "arguments": {}}'
        result = parser.parse(text)
        assert result.success
        assert result.tool_calls[0].id == "call_123"

    def test_with_type_field(self, parser):
        """Test with type field (OpenAI format)."""
        text = '{"type": "function", "name": "test", "arguments": {}}'
        result = parser.parse(text)
        assert result.success

    def test_arguments_as_string(self, parser):
        """Test arguments as JSON string (OpenAI format)."""
        text = '{"name": "test", "arguments": "{\\"x\\": 1}"}'
        result = parser.parse(text)
        assert result.success

    def test_xml_single_call(self, parser):
        """Test XML-wrapped single call."""
        text = '<tool_call>{"name": "test", "arguments": {}}</tool_call>'
        result = parser.parse(text)
        assert result.success

    def test_xml_multiple_calls(self, parser):
        """Test XML-wrapped multiple calls."""
        text = '''<tool_call>{"name": "a", "arguments": {}}</tool_call>
                  <tool_call>{"name": "b", "arguments": {}}</tool_call>'''
        result = parser.parse(text)
        assert result.num_calls == 2

    def test_mixed_with_text(self, parser):
        """Test tool call mixed with natural language."""
        text = '''Let me help you with that.
        {"name": "search", "arguments": {"query": "python"}}
        I found some results.'''
        result = parser.parse(text)
        assert result.success

    def test_array_format(self, parser):
        """Test array of tool calls."""
        text = '[{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]'
        result = parser.parse(text)
        assert result.num_calls == 2

    def test_extra_fields_ignored(self, parser):
        """Test that extra fields are handled gracefully."""
        text = '{"name": "test", "arguments": {}, "extra": "field", "another": 123}'
        result = parser.parse(text)
        assert result.success


# =============================================================================
# Bonus: Regression Tests
# =============================================================================

class TestRegressions:
    """Tests for previously found bugs."""

    def test_empty_string_argument(self, parser):
        """Test empty string as argument value."""
        text = '{"name": "test", "arguments": {"value": ""}}'
        result = parser.parse(text)
        assert result.success
        assert result.tool_calls[0].arguments["value"] == ""

    def test_zero_value(self, parser):
        """Test zero as argument value."""
        text = '{"name": "test", "arguments": {"value": 0}}'
        result = parser.parse(text)
        assert result.success
        assert result.tool_calls[0].arguments["value"] == 0

    def test_false_value(self, parser):
        """Test false as argument value."""
        text = '{"name": "test", "arguments": {"value": false}}'
        result = parser.parse(text)
        assert result.success
        assert result.tool_calls[0].arguments["value"] is False

    def test_null_value(self, parser):
        """Test null as argument value."""
        text = '{"name": "test", "arguments": {"value": null}}'
        result = parser.parse(text)
        assert result.success
        assert result.tool_calls[0].arguments["value"] is None

    def test_negative_number(self, parser):
        """Test negative number."""
        text = '{"name": "test", "arguments": {"value": -42}}'
        result = parser.parse(text)
        assert result.success
        assert result.tool_calls[0].arguments["value"] == -42

    def test_scientific_notation(self, parser):
        """Test scientific notation number."""
        text = '{"name": "test", "arguments": {"value": 1.23e10}}'
        result = parser.parse(text)
        assert result.success
