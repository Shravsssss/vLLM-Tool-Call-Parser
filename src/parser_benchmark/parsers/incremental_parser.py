"""Incremental JSON parser for streaming tool call extraction."""

import json
import time
from enum import Enum, auto
from typing import Any

from parser_benchmark.models import ToolCall, ParseResult
from parser_benchmark.parsers.base import BaseParser


class JsonState(Enum):
    """Parser states for JSON parsing."""
    IDLE = auto()           # Waiting for content
    IN_OBJECT = auto()      # Inside { }
    IN_ARRAY = auto()       # Inside [ ]
    IN_STRING = auto()      # Inside " "
    IN_KEY = auto()         # Reading object key
    AFTER_KEY = auto()      # After key, expecting :
    IN_VALUE = auto()       # Reading value
    IN_NUMBER = auto()      # Reading number
    IN_LITERAL = auto()     # Reading true/false/null
    ESCAPE = auto()         # After backslash in string
    COMPLETE = auto()       # Finished parsing object


class IncrementalParser(BaseParser):
    """Parser that processes input character by character.

    Supports streaming input and provides precise error locations.
    Handles nested structures and maintains parsing state across calls.

    Attributes:
        name: Parser identifier.
    """

    @property
    def name(self) -> str:
        """Return the parser identifier."""
        return "incremental-parser"

    def parse(self, text: str) -> ParseResult:
        """Parse text incrementally and extract tool calls.

        Args:
            text: Text to parse (can be partial for streaming).

        Returns:
            ParseResult with extracted tool calls.
        """
        start_time = time.perf_counter()

        try:
            tool_calls = self._parse_incremental(text)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return ParseResult(
                tool_calls=tool_calls,
                raw_input=text,
                parse_time_ms=elapsed_ms,
                success=True
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return ParseResult(
                tool_calls=[],
                raw_input=text,
                parse_time_ms=elapsed_ms,
                success=False,
                error=str(e)
            )

    def parse_multiple(self, texts: list[str]) -> list[ParseResult]:
        """Parse multiple texts in batch."""
        return [self.parse(text) for text in texts]

    def _parse_incremental(self, text: str) -> list[ToolCall]:
        """Parse text character by character."""
        tool_calls: list[ToolCall] = []

        i = 0
        while i < len(text):
            # Find start of JSON object
            if text[i] == '{':
                obj_str, end_pos = self._extract_json_object(text, i)
                if obj_str:
                    tool_call = self._try_parse_tool_call(obj_str)
                    if tool_call:
                        tool_calls.append(tool_call)
                    i = end_pos
                else:
                    i += 1
            # Find start of JSON array
            elif text[i] == '[':
                arr_str, end_pos = self._extract_json_array(text, i)
                if arr_str:
                    calls = self._try_parse_array(arr_str)
                    tool_calls.extend(calls)
                    i = end_pos
                else:
                    i += 1
            # Check for XML-style markers (wrapper format)
            elif text[i:i+11] == '<tool_call>':
                content, end_pos = self._extract_xml_content(text, i)
                if content:
                    tool_call = self._try_parse_tool_call(content)
                    if tool_call:
                        tool_calls.append(tool_call)
                    i = end_pos
                else:
                    i += 1
            # Check for XML attribute format (Qwen-style)
            # e.g., <get_weather city="Tokyo" unit="fahrenheit"/>
            elif text[i] == '<' and i + 1 < len(text) and (text[i+1].isalpha() or text[i+1] == '_'):
                tool_call, end_pos = self._extract_xml_attr_call(text, i)
                if tool_call:
                    tool_calls.append(tool_call)
                    i = end_pos
                else:
                    i += 1
            else:
                i += 1

        return tool_calls

    def _extract_json_object(self, text: str, start: int) -> tuple[str | None, int]:
        """Extract a complete JSON object starting at position."""
        if start >= len(text) or text[start] != '{':
            return None, start + 1

        depth = 0
        in_string = False
        escape_next = False
        i = start

        while i < len(text):
            char = text[i]

            if escape_next:
                escape_next = False
                i += 1
                continue

            if char == '\\' and in_string:
                escape_next = True
                i += 1
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
            elif not in_string:
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        return text[start:i+1], i + 1

            i += 1

        return None, start + 1

    def _extract_json_array(self, text: str, start: int) -> tuple[str | None, int]:
        """Extract a complete JSON array starting at position."""
        if start >= len(text) or text[start] != '[':
            return None, start + 1

        depth = 0
        in_string = False
        escape_next = False
        i = start

        while i < len(text):
            char = text[i]

            if escape_next:
                escape_next = False
                i += 1
                continue

            if char == '\\' and in_string:
                escape_next = True
                i += 1
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
            elif not in_string:
                if char == '[':
                    depth += 1
                elif char == ']':
                    depth -= 1
                    if depth == 0:
                        return text[start:i+1], i + 1

            i += 1

        return None, start + 1

    def _extract_xml_content(self, text: str, start: int) -> tuple[str | None, int]:
        """Extract content between <tool_call> tags."""
        marker_start = '<tool_call>'
        marker_end = '</tool_call>'

        if not text[start:].startswith(marker_start):
            return None, start + 1

        content_start = start + len(marker_start)
        end_pos = text.find(marker_end, content_start)

        if end_pos == -1:
            return None, start + 1

        content = text[content_start:end_pos].strip()
        return content, end_pos + len(marker_end)

    def _try_parse_tool_call(self, json_str: str) -> ToolCall | None:
        """Try to parse a JSON string as a tool call."""
        try:
            data = json.loads(json_str)
            if isinstance(data, dict) and 'name' in data:
                arguments = data.get('arguments', {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                return ToolCall(
                    id=data.get('id'),
                    name=data['name'],
                    arguments=arguments if isinstance(arguments, dict) else {}
                )
        except json.JSONDecodeError:
            pass
        return None

    def _try_parse_array(self, json_str: str) -> list[ToolCall]:
        """Try to parse a JSON array of tool calls."""
        tool_calls = []
        try:
            data = json.loads(json_str)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'name' in item:
                        arguments = item.get('arguments', {})
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except json.JSONDecodeError:
                                arguments = {}
                        tool_calls.append(ToolCall(
                            id=item.get('id'),
                            name=item['name'],
                            arguments=arguments if isinstance(arguments, dict) else {}
                        ))
        except json.JSONDecodeError:
            pass
        return tool_calls

    def _extract_xml_attr_call(self, text: str, start: int) -> tuple[ToolCall | None, int]:
        """Extract XML attribute-style tool call starting at position.

        Matches patterns like:
        - <func_name attr1="val1" attr2="val2"/>
        - <func_name attr1="val1">

        Args:
            text: Full text being parsed.
            start: Starting position (should be '<').

        Returns:
            Tuple of (ToolCall or None, end position).
        """
        if start >= len(text) or text[start] != '<':
            return None, start + 1

        i = start + 1

        # Extract tag name (must be valid identifier)
        tag_start = i
        while i < len(text) and (text[i].isalnum() or text[i] == '_'):
            i += 1

        if i == tag_start:  # No valid tag name
            return None, start + 1

        tag_name = text[tag_start:i]

        # Must start with letter or underscore
        if not (tag_name[0].isalpha() or tag_name[0] == '_'):
            return None, start + 1

        # Skip known wrapper tags
        if tag_name in ('tool_call', 'function', 'functions'):
            return None, start + 1

        # Parse attributes until we hit > or />
        arguments: dict[str, Any] = {}
        while i < len(text):
            # Skip whitespace
            while i < len(text) and text[i].isspace():
                i += 1

            if i >= len(text):
                return None, start + 1

            # Check for self-closing or opening tag end
            if text[i] == '/':
                if i + 1 < len(text) and text[i + 1] == '>':
                    # Self-closing tag - we found a complete tag
                    if arguments:  # Only return if we have attributes
                        return ToolCall(name=tag_name, arguments=arguments), i + 2
                    return None, start + 1
                return None, start + 1

            if text[i] == '>':
                # Open tag end
                if arguments:
                    return ToolCall(name=tag_name, arguments=arguments), i + 1
                return None, start + 1

            # Try to parse attribute
            attr_name, attr_value, new_pos = self._parse_single_attribute(text, i)
            if attr_name is not None:
                arguments[attr_name] = attr_value
                i = new_pos
            else:
                # Not a valid attribute, bail out
                return None, start + 1

        return None, start + 1

    def _parse_single_attribute(self, text: str, start: int) -> tuple[str | None, Any, int]:
        """Parse a single XML attribute at position.

        Args:
            text: Full text being parsed.
            start: Starting position.

        Returns:
            Tuple of (attr_name, attr_value, end_position) or (None, None, start) if invalid.
        """
        i = start

        # Extract attribute name
        name_start = i
        while i < len(text) and (text[i].isalnum() or text[i] == '_'):
            i += 1

        if i == name_start:
            return None, None, start

        attr_name = text[name_start:i]

        # Skip whitespace and expect =
        while i < len(text) and text[i].isspace():
            i += 1

        if i >= len(text) or text[i] != '=':
            return None, None, start
        i += 1

        # Skip whitespace and expect "
        while i < len(text) and text[i].isspace():
            i += 1

        if i >= len(text) or text[i] != '"':
            return None, None, start
        i += 1

        # Extract value until closing "
        value_start = i
        while i < len(text) and text[i] != '"':
            i += 1

        if i >= len(text):
            return None, None, start

        attr_value = text[value_start:i]
        i += 1  # Skip closing "

        # Convert value to appropriate type
        converted_value = self._convert_attr_value(attr_value)

        return attr_name, converted_value, i

    def _convert_attr_value(self, value: str) -> Any:
        """Convert string attribute value to appropriate Python type.

        Args:
            value: String value from XML attribute.

        Returns:
            Converted value (bool, None, int, float, or str).
        """
        if value.lower() == 'true':
            return True
        if value.lower() == 'false':
            return False
        if value.lower() in ('null', 'none'):
            return None
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value


class StreamingParser:
    """Streaming wrapper for incremental parsing.

    Maintains state across multiple parse calls, allowing
    partial input to be processed as it arrives.

    Example:
        parser = StreamingParser()
        parser.feed('{"name": "te')
        parser.feed('st", "arguments": {}}')
        result = parser.get_result()
    """

    def __init__(self):
        """Initialize the streaming parser."""
        self._buffer = ""
        self._tool_calls: list[ToolCall] = []
        self._parser = IncrementalParser()

    def feed(self, chunk: str) -> list[ToolCall]:
        """Feed a chunk of text and return any complete tool calls.

        Args:
            chunk: Partial text to add to buffer.

        Returns:
            List of newly completed tool calls.
        """
        self._buffer += chunk
        result = self._parser.parse(self._buffer)

        new_calls = result.tool_calls[len(self._tool_calls):]
        self._tool_calls = result.tool_calls

        return new_calls

    def get_result(self) -> ParseResult:
        """Get the current parse result."""
        return self._parser.parse(self._buffer)

    def reset(self):
        """Reset the parser state."""
        self._buffer = ""
        self._tool_calls = []