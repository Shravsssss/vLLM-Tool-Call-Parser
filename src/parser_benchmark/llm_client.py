"""LLM client for tool calling comparison.

Works with vLLM, OpenAI, Together AI, Groq, Fireworks, and other
OpenAI-compatible APIs.
"""

import json
import time
from typing import Any
from dataclasses import dataclass, field

from openai import OpenAI

from parser_benchmark.models import ToolCall, ParseResult


@dataclass
class LLMConfig:
    """Configuration for LLM server connection.

    Works with vLLM, OpenAI, Together AI, Groq, Fireworks, etc.
    """
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model: str | None = None

    @classmethod
    def vllm_local(cls) -> "LLMConfig":
        """Config for local vLLM server."""
        return cls(base_url="http://localhost:8000/v1", api_key="EMPTY")

    @classmethod
    def openai(cls, api_key: str, model: str = "gpt-4o-mini") -> "LLMConfig":
        """Config for OpenAI API."""
        return cls(
            base_url="https://api.openai.com/v1",
            api_key=api_key,
            model=model
        )

    @classmethod
    def together(cls, api_key: str, model: str = "meta-llama/Llama-3.2-3B-Instruct-Turbo") -> "LLMConfig":
        """Config for Together AI (free tier available)."""
        return cls(
            base_url="https://api.together.xyz/v1",
            api_key=api_key,
            model=model
        )

    @classmethod
    def groq(cls, api_key: str, model: str = "llama-3.1-8b-instant") -> "LLMConfig":
        """Config for Groq (free tier, very fast)."""
        return cls(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
            model=model
        )

    @classmethod
    def fireworks(cls, api_key: str, model: str = "accounts/fireworks/models/llama-v3p1-8b-instruct") -> "LLMConfig":
        """Config for Fireworks AI (free tier)."""
        return cls(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=api_key,
            model=model
        )


@dataclass
class ToolDefinition:
    """Definition of a tool for the LLM."""
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_openai_format(self) -> dict:
        """Convert to OpenAI tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters.get("properties", {}),
                    "required": self.parameters.get("required", []),
                },
            },
        }


class LLMClient:
    """Client for interacting with LLM servers with tool calling.

    Works with vLLM, OpenAI, Together AI, Groq, Fireworks, etc.
    """

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig.vllm_local()
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )
        self._model = self.config.model

    @property
    def name(self) -> str:
        return "llm-server"

    @property
    def model(self) -> str:
        """Get the model name, auto-detecting if needed."""
        if self._model is None:
            models = self.client.models.list()
            self._model = models.data[0].id
        return self._model

    def chat_raw(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        """Send a chat request and get raw text output (no tool parsing).

        Use this to get raw LLM output for parser testing.

        Args:
            messages: Conversation history.
            max_tokens: Maximum tokens to generate.

        Returns:
            Dict with 'content' (raw text) and 'elapsed_ms'.
        """
        start_time = time.perf_counter()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return {
            "content": response.choices[0].message.content,
            "elapsed_ms": elapsed_ms,
        }

    def chat_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolDefinition],
        tool_choice: str = "auto",
        stream: bool = False,
    ) -> dict[str, Any]:
        """Send a chat request with tools.

        Args:
            messages: Conversation history.
            tools: List of tool definitions.
            tool_choice: "auto", "required", "none", or specific function.
            stream: Whether to stream the response.

        Returns:
            Full response including tool calls.
        """
        openai_tools = [t.to_openai_format() for t in tools]

        start_time = time.perf_counter()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=openai_tools,
            tool_choice=tool_choice,
            stream=stream,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if stream:
            chunks = list(response)
            return self._process_stream(chunks, elapsed_ms)
        else:
            return self._process_response(response, elapsed_ms)

    def _process_response(self, response, elapsed_ms: float) -> dict[str, Any]:
        """Process non-streaming response."""
        message = response.choices[0].message

        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": args,
                })

        return {
            "content": message.content,
            "tool_calls": tool_calls,
            "raw_response": response,
            "elapsed_ms": elapsed_ms,
        }

    def _process_stream(self, chunks: list, elapsed_ms: float) -> dict[str, Any]:
        """Process streaming response chunks."""
        tool_calls = {}
        content_parts = []

        for chunk in chunks:
            delta = chunk.choices[0].delta

            if delta.content:
                content_parts.append(delta.content)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls:
                        tool_calls[idx] = {
                            "id": "",
                            "name": "",
                            "arguments": "",
                        }

                    if tc.id:
                        tool_calls[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls[idx]["arguments"] += tc.function.arguments

        processed_calls = []
        for tc in tool_calls.values():
            args = tc["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            processed_calls.append({
                "id": tc["id"],
                "name": tc["name"],
                "arguments": args,
            })

        return {
            "content": "".join(content_parts),
            "tool_calls": processed_calls,
            "raw_chunks": chunks,
            "elapsed_ms": elapsed_ms,
        }

    def parse(self, prompt: str, tools: list[ToolDefinition] = None) -> ParseResult:
        """Parse a prompt and extract tool calls."""
        if tools is None:
            tools = [DEFAULT_WEATHER_TOOL]

        messages = [{"role": "user", "content": prompt}]

        start_time = time.perf_counter()

        try:
            result = self.chat_with_tools(messages, tools, tool_choice="auto")
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    name=tc["name"],
                    arguments=tc["arguments"],
                )
                for tc in result["tool_calls"]
            ]

            return ParseResult(
                tool_calls=tool_calls,
                raw_input=prompt,
                parse_time_ms=elapsed_ms,
                success=True,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return ParseResult(
                tool_calls=[],
                raw_input=prompt,
                parse_time_ms=elapsed_ms,
                success=False,
                error=str(e),
            )


# Default tools for testing
DEFAULT_WEATHER_TOOL = ToolDefinition(
    name="get_current_weather",
    description="Get the current weather in a given location",
    parameters={
        "properties": {
            "city": {
                "type": "string",
                "description": "The city name, e.g. 'San Francisco'",
            },
            "state": {
                "type": "string",
                "description": "Two-letter state code, e.g. 'CA'",
            },
            "unit": {
                "type": "string",
                "description": "Temperature unit",
                "enum": ["celsius", "fahrenheit"],
            },
        },
        "required": ["city", "state", "unit"],
    },
)

DEFAULT_SEARCH_TOOL = ToolDefinition(
    name="search",
    description="Search for information on the web",
    parameters={
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query",
            },
        },
        "required": ["query"],
    },
)

DEFAULT_CALCULATOR_TOOL = ToolDefinition(
    name="calculate",
    description="Perform a mathematical calculation",
    parameters={
        "properties": {
            "expression": {
                "type": "string",
                "description": "Math expression to evaluate, e.g. '2 + 2'",
            },
        },
        "required": ["expression"],
    },
)


def get_default_tools() -> list[ToolDefinition]:
    """Get default tools for testing."""
    return [DEFAULT_WEATHER_TOOL, DEFAULT_SEARCH_TOOL, DEFAULT_CALCULATOR_TOOL]
