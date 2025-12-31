"""State machine-based parser for tool call extraction."""

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from parser_benchmark.models import ToolCall, ParseResult
from parser_benchmark.parsers.base import BaseParser


class ParserState(Enum):
    """High-level parser states."""
    SCANNING = auto()        # Looking for tool call start
    IN_JSON_OBJECT = auto()  # Inside a JSON object
    IN_JSON_ARRAY = auto()   # Inside a JSON array
    IN_XML_TAG = auto()      # Inside XML-style tags
    IN_XML_ATTR_TAG = auto() # Inside XML attribute-style tag (Qwen format)
    ERROR = auto()           # Error state
    COMPLETE = auto()        # Successfully parsed


@dataclass
class ParserContext:
    """Maintains parser state and accumulated data."""
    state: ParserState = ParserState.SCANNING
    position: int = 0
    depth: int = 0
    buffer: str = ""
    in_string: bool = False
    escape_next: bool = False
    tool_calls: list[ToolCall] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class StateMachineParser(BaseParser):
    """Parser using explicit state machine for robustness.

    Supports multiple formats with clear state transitions:
        - JSON objects and arrays
        - XML-wrapped tool calls
        - Hermes/ChatML format
        - Custom markers

    Features:
        - Error recovery (continues after errors)
        - Position tracking for debugging
        - Extensible state system

    Attributes:
        name: Parser identifier.
    """

    # Format markers
    XML_START = '<tool_call>'
    XML_END = '</tool_call>'
    HERMES_START = '<tool_call>'
    HERMES_END = '</tool_call>'

    @property
    def name(self) -> str:
        """Return the parser identifier."""
        return "state-machine-parser"

    def parse(self, text: str) -> ParseResult:
        """Parse text using state machine.

        Args:
            text: Text to parse.

        Returns:
            ParseResult with extracted tool calls.
        """
        start_time = time.perf_counter()

        ctx = ParserContext()

        try:
            self._run_state_machine(text, ctx)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return ParseResult(
                tool_calls=ctx.tool_calls,
                raw_input=text,
                parse_time_ms=elapsed_ms,
                success=len(ctx.errors) == 0,
                error="; ".join(ctx.errors) if ctx.errors else None
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return ParseResult(
                tool_calls=ctx.tool_calls,
                raw_input=text,
                parse_time_ms=elapsed_ms,
                success=False,
                error=str(e)
            )

    def parse_multiple(self, texts: list[str]) -> list[ParseResult]:
        """Parse multiple texts in batch."""
        return [self.parse(text) for text in texts]

    def _run_state_machine(self, text: str, ctx: ParserContext):
        """Execute the state machine on input text."""
        while ctx.position < len(text):
            if ctx.state == ParserState.SCANNING:
                self._state_scanning(text, ctx)
            elif ctx.state == ParserState.IN_JSON_OBJECT:
                self._state_in_json_object(text, ctx)
            elif ctx.state == ParserState.IN_JSON_ARRAY:
                self._state_in_json_array(text, ctx)
            elif ctx.state == ParserState.IN_XML_TAG:
                self._state_in_xml_tag(text, ctx)
            elif ctx.state == ParserState.IN_XML_ATTR_TAG:
                self._state_in_xml_attr_tag(text, ctx)
            elif ctx.state == ParserState.ERROR:
                self._state_error(text, ctx)
            elif ctx.state == ParserState.COMPLETE:
                ctx.state = ParserState.SCANNING
            else:
                ctx.position += 1

    def _state_scanning(self, text: str, ctx: ParserContext):
        """Look for the start of a tool call."""
        pos = ctx.position

        # Check for XML-style wrapper marker first
        if text[pos:pos + len(self.XML_START)] == self.XML_START:
            ctx.state = ParserState.IN_XML_TAG
            ctx.position = pos + len(self.XML_START)
            ctx.buffer = ""
            return

        # Check for XML attribute format (Qwen-style)
        # e.g., <get_weather city="Tokyo" unit="fahrenheit"/>
        if text[pos] == '<' and self._is_xml_attr_start(text, pos):
            ctx.state = ParserState.IN_XML_ATTR_TAG
            ctx.position = pos + 1  # Skip <
            ctx.buffer = ""
            return

        # Check for JSON object
        if text[pos] == '{':
            ctx.state = ParserState.IN_JSON_OBJECT
            ctx.buffer = "{"
            ctx.depth = 1
            ctx.position = pos + 1
            return

        # Check for JSON array
        if text[pos] == '[':
            ctx.state = ParserState.IN_JSON_ARRAY
            ctx.buffer = "["
            ctx.depth = 1
            ctx.position = pos + 1
            return

        ctx.position += 1

    def _state_in_json_object(self, text: str, ctx: ParserContext):
        """Parse inside a JSON object."""
        pos = ctx.position

        if pos >= len(text):
            ctx.state = ParserState.SCANNING
            return

        char = text[pos]

        # Handle escape sequences
        if ctx.escape_next:
            ctx.buffer += char
            ctx.escape_next = False
            ctx.position += 1
            return

        if char == '\\' and ctx.in_string:
            ctx.escape_next = True
            ctx.buffer += char
            ctx.position += 1
            return

        # Handle string boundaries
        if char == '"':
            ctx.in_string = not ctx.in_string
            ctx.buffer += char
            ctx.position += 1
            return

        # Track depth only outside strings
        if not ctx.in_string:
            if char == '{':
                ctx.depth += 1
            elif char == '}':
                ctx.depth -= 1
                if ctx.depth == 0:
                    ctx.buffer += char
                    self._emit_json_object(ctx)
                    ctx.position += 1
                    return

        ctx.buffer += char
        ctx.position += 1

    def _state_in_json_array(self, text: str, ctx: ParserContext):
        """Parse inside a JSON array."""
        pos = ctx.position

        if pos >= len(text):
            ctx.state = ParserState.SCANNING
            return

        char = text[pos]

        # Handle escape sequences
        if ctx.escape_next:
            ctx.buffer += char
            ctx.escape_next = False
            ctx.position += 1
            return

        if char == '\\' and ctx.in_string:
            ctx.escape_next = True
            ctx.buffer += char
            ctx.position += 1
            return

        # Handle string boundaries
        if char == '"':
            ctx.in_string = not ctx.in_string
            ctx.buffer += char
            ctx.position += 1
            return

        # Track depth only outside strings
        if not ctx.in_string:
            if char in '{[':
                ctx.depth += 1
            elif char in '}]':
                ctx.depth -= 1
                if ctx.depth == 0:
                    ctx.buffer += char
                    self._emit_json_array(ctx)
                    ctx.position += 1
                    return

        ctx.buffer += char
        ctx.position += 1

    def _state_in_xml_tag(self, text: str, ctx: ParserContext):
        """Parse inside XML-style tags."""
        pos = ctx.position
        end_marker = self.XML_END

        end_pos = text.find(end_marker, pos)
        if end_pos == -1:
            # No end marker found, consume rest
            ctx.buffer += text[pos:]
            ctx.position = len(text)
            return

        # Extract content between tags
        ctx.buffer = text[pos:end_pos].strip()
        ctx.position = end_pos + len(end_marker)

        # Try to parse the content as JSON
        self._emit_xml_content(ctx)

    def _state_error(self, text: str, ctx: ParserContext):
        """Handle error state with recovery."""
        # Skip to next potential start
        ctx.buffer = ""
        ctx.depth = 0
        ctx.in_string = False
        ctx.escape_next = False
        ctx.state = ParserState.SCANNING
        ctx.position += 1

    def _emit_json_object(self, ctx: ParserContext):
        """Process completed JSON object."""
        try:
            data = json.loads(ctx.buffer)
            if isinstance(data, dict) and 'name' in data:
                tool_call = self._dict_to_tool_call(data)
                ctx.tool_calls.append(tool_call)
        except json.JSONDecodeError as e:
            ctx.errors.append(f"JSON error at {ctx.position}: {e}")

        ctx.buffer = ""
        ctx.state = ParserState.COMPLETE

    def _emit_json_array(self, ctx: ParserContext):
        """Process completed JSON array."""
        try:
            data = json.loads(ctx.buffer)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'name' in item:
                        tool_call = self._dict_to_tool_call(item)
                        ctx.tool_calls.append(tool_call)
        except json.JSONDecodeError as e:
            ctx.errors.append(f"JSON array error at {ctx.position}: {e}")

        ctx.buffer = ""
        ctx.state = ParserState.COMPLETE

    def _emit_xml_content(self, ctx: ParserContext):
        """Process content from XML tags."""
        content = ctx.buffer.strip()
        if not content:
            ctx.state = ParserState.SCANNING
            return

        try:
            data = json.loads(content)
            if isinstance(data, dict) and 'name' in data:
                tool_call = self._dict_to_tool_call(data)
                ctx.tool_calls.append(tool_call)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'name' in item:
                        tool_call = self._dict_to_tool_call(item)
                        ctx.tool_calls.append(tool_call)
        except json.JSONDecodeError as e:
            ctx.errors.append(f"XML content error: {e}")

        ctx.buffer = ""
        ctx.state = ParserState.COMPLETE

    def _dict_to_tool_call(self, data: dict) -> ToolCall:
        """Convert dictionary to ToolCall."""
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

    def _is_xml_attr_start(self, text: str, pos: int) -> bool:
        """Check if position starts an XML attribute-style tag.

        Args:
            text: Full text being parsed.
            pos: Position to check (should be '<').

        Returns:
            True if this looks like an XML attribute tag start.
        """
        if pos >= len(text) or text[pos] != '<':
            return False

        # Skip <
        i = pos + 1

        # Must start with letter or underscore
        if i >= len(text) or not (text[i].isalpha() or text[i] == '_'):
            return False

        # Continue with alphanumeric or underscore
        while i < len(text) and (text[i].isalnum() or text[i] == '_'):
            i += 1

        # Get tag name
        tag = text[pos+1:i]

        # Exclude known wrapper tags
        if tag in ('tool_call', 'function', 'functions'):
            return False

        # Must have whitespace before attributes, or = sign nearby (indicating attributes)
        if i < len(text):
            # Check if there's an attribute pattern ahead (allow spaces around =)
            remaining = text[i:i+50]  # Look ahead
            # Look for = followed eventually by " (with possible spaces)
            if '=' in remaining and '"' in remaining and ('>' in remaining or '/>' in remaining):
                return True

        return False

    def _state_in_xml_attr_tag(self, text: str, ctx: ParserContext):
        """Parse XML attribute-style tool call.

        Handles patterns like:
        - <func_name attr1="val1" attr2="val2"/>
        - <func_name attr1="val1">
        """
        pos = ctx.position

        # Extract tag name
        tag_start = pos
        while pos < len(text) and (text[pos].isalnum() or text[pos] == '_'):
            pos += 1

        if pos == tag_start:
            ctx.state = ParserState.ERROR
            return

        tag_name = text[tag_start:pos]

        # Skip known wrapper tags
        if tag_name in ('tool_call', 'function', 'functions'):
            ctx.state = ParserState.SCANNING
            return

        arguments: dict[str, Any] = {}

        # Parse attributes until we hit > or />
        while pos < len(text):
            # Skip whitespace
            while pos < len(text) and text[pos].isspace():
                pos += 1

            if pos >= len(text):
                ctx.state = ParserState.ERROR
                ctx.position = pos
                return

            # Check for end of tag
            if text[pos] == '/':
                if pos + 1 < len(text) and text[pos + 1] == '>':
                    pos += 2
                    break
                ctx.state = ParserState.ERROR
                ctx.position = pos
                return

            if text[pos] == '>':
                pos += 1
                break

            # Parse attribute name
            attr_start = pos
            while pos < len(text) and (text[pos].isalnum() or text[pos] == '_'):
                pos += 1

            if pos == attr_start:
                ctx.state = ParserState.ERROR
                ctx.position = pos
                return

            attr_name = text[attr_start:pos]

            # Expect =
            while pos < len(text) and text[pos].isspace():
                pos += 1

            if pos >= len(text) or text[pos] != '=':
                ctx.state = ParserState.ERROR
                ctx.position = pos
                return
            pos += 1

            # Expect "
            while pos < len(text) and text[pos].isspace():
                pos += 1

            if pos >= len(text) or text[pos] != '"':
                ctx.state = ParserState.ERROR
                ctx.position = pos
                return
            pos += 1

            # Extract value
            value_start = pos
            while pos < len(text) and text[pos] != '"':
                pos += 1

            if pos >= len(text):
                ctx.state = ParserState.ERROR
                ctx.position = pos
                return

            attr_value = text[value_start:pos]
            pos += 1  # Skip closing "

            arguments[attr_name] = self._convert_attr_value(attr_value)

        # Emit tool call if we have arguments
        if arguments:
            ctx.tool_calls.append(ToolCall(name=tag_name, arguments=arguments))

        ctx.position = pos
        ctx.state = ParserState.COMPLETE

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