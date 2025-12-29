"""Regex-based parser for extracting tool calls from text."""

import json
import re
import time
from typing import Any

from parser_benchmark.models import ToolCall, ParseResult
from parser_benchmark.parsers.base import BaseParser


class RegexParser(BaseParser):
    """Fast regex-based parser for extracting tool calls from LLM output.

    Supports multiple formats:
        - Simple JSON objects
        - JSON arrays of tool calls
        - XML-wrapped tool calls (<tool_call>...</tool_call>)
        - Tool calls embedded in natural language text
        - Common malformed JSON (trailing commas, single quotes)

    Attributes:
        name: Parser identifier used in benchmarks and logging.
    """

    XML_PATTERN = re.compile(
        r'<tool_call>\s*(.*?)\s*</tool_call>',
        re.DOTALL
    )

    JSON_OBJECT_PATTERN = re.compile(
        r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}',
        re.DOTALL
    )

    FLEXIBLE_JSON_PATTERN = re.compile(
        r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{(?:[^{}]|\{[^{}]*\})*\})\s*\}',
        re.DOTALL
    )

    SINGLE_QUOTE_PATTERN = re.compile(
        r"\{\s*'name'\s*:\s*'([^']+)'\s*,\s*'arguments'\s*:\s*(\{(?:[^{}]|\{[^{}]*\})*\})\s*\}",
        re.DOTALL
    )

    @property
    def name(self) -> str:
        """Return the parser identifier."""
        return "regex-parser"

    def parse(self, text: str) -> ParseResult:
        """Parse text and extract tool calls.

        Args:
            text: Raw text potentially containing tool calls.

        Returns:
            ParseResult containing extracted tool calls and metadata.
        """
        start_time = time.perf_counter()

        try:
            tool_calls = self._extract_tool_calls(text)
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
        """Parse multiple texts in batch.

        Args:
            texts: List of text strings to parse.

        Returns:
            List of ParseResult objects, one per input text.
        """
        return [self.parse(text) for text in texts]

    def _extract_tool_calls(self, text: str) -> list[ToolCall]:
        """Extract tool calls using multiple strategies."""
        tool_calls: list[ToolCall] = []

        xml_matches = self.XML_PATTERN.findall(text)
        if xml_matches:
            for match in xml_matches:
                calls = self._parse_json_content(match)
                tool_calls.extend(calls)
            return tool_calls

        stripped = text.strip()

        if stripped.startswith('['):
            parsed = self._try_parse_with_fixes(stripped)
            if parsed is not None and isinstance(parsed, list):
                for item in parsed:
                    if self._is_tool_call(item):
                        tool_calls.append(self._dict_to_tool_call(item))
                if tool_calls:
                    return tool_calls

        if stripped.startswith('{'):
            parsed = self._try_parse_with_fixes(stripped)
            if parsed is not None and self._is_tool_call(parsed):
                return [self._dict_to_tool_call(parsed)]

        single_quote_matches = self.SINGLE_QUOTE_PATTERN.findall(text)
        for name, args_str in single_quote_matches:
            fixed_args = self._convert_single_to_double_quotes(args_str)
            try:
                arguments = json.loads(fixed_args)
                tool_calls.append(ToolCall(name=name, arguments=arguments))
            except json.JSONDecodeError:
                continue

        if tool_calls:
            return tool_calls

        matches = self.FLEXIBLE_JSON_PATTERN.findall(text)
        for name, args_str in matches:
            arguments = None
            try:
                arguments = json.loads(args_str)
            except json.JSONDecodeError:
                fixed_args = self._fix_malformed_json(args_str)
                try:
                    arguments = json.loads(fixed_args)
                except json.JSONDecodeError:
                    continue

            if arguments is not None:
                tool_calls.append(ToolCall(name=name, arguments=arguments))

        return tool_calls

    def _parse_json_content(self, content: str) -> list[ToolCall]:
        """Parse JSON content that may be an object or array."""
        content = content.strip()
        parsed = self._try_parse_with_fixes(content)

        if parsed is not None:
            if isinstance(parsed, list):
                return [
                    self._dict_to_tool_call(item)
                    for item in parsed
                    if self._is_tool_call(item)
                ]
            elif isinstance(parsed, dict) and self._is_tool_call(parsed):
                return [self._dict_to_tool_call(parsed)]

        return []

    def _is_tool_call(self, data: Any) -> bool:
        """Check if data matches tool call structure."""
        return (
            isinstance(data, dict) and
            "name" in data and
            isinstance(data.get("name"), str)
        )

    def _dict_to_tool_call(self, data: dict) -> ToolCall:
        """Convert a dictionary to a ToolCall instance."""
        arguments = data.get("arguments", {})

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

        return ToolCall(
            id=data.get("id"),
            name=data["name"],
            arguments=arguments if isinstance(arguments, dict) else {}
        )

    def _fix_malformed_json(self, text: str) -> str:
        """Attempt to fix common JSON syntax errors."""
        fixed = self._convert_single_to_double_quotes(text)
        fixed = re.sub(r',\s*}', '}', fixed)
        fixed = re.sub(r',\s*]', ']', fixed)

        open_braces = fixed.count('{')
        close_braces = fixed.count('}')
        if open_braces > close_braces:
            fixed += '}' * (open_braces - close_braces)

        open_brackets = fixed.count('[')
        close_brackets = fixed.count(']')
        if open_brackets > close_brackets:
            fixed += ']' * (open_brackets - close_brackets)

        return fixed

    def _convert_single_to_double_quotes(self, text: str) -> str:
        """Convert single quotes to double quotes for JSON compatibility."""
        result = []
        i = 0
        in_string = False
        string_char = None

        while i < len(text):
            char = text[i]

            if char in ('"', "'") and (i == 0 or text[i-1] != '\\'):
                if not in_string:
                    in_string = True
                    string_char = char
                    result.append('"' if char == "'" else char)
                elif char == string_char:
                    in_string = False
                    result.append('"' if char == "'" else char)
                    string_char = None
                else:
                    result.append(char)
            else:
                result.append(char)
            i += 1

        return ''.join(result)

    def _try_parse_with_fixes(self, text: str) -> dict | list | None:
        """Try to parse JSON, applying fixes if initial parse fails."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        fixed = self._fix_malformed_json(text)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        return None
