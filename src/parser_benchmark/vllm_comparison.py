"""vLLM Comparison Module.

Compares custom parsers (Regex, Incremental, StateMachine) against vLLM's
native tool call parsing. Measures accuracy, latency, streaming advantage,
and error recovery capabilities.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generator

from openai import OpenAI

from parser_benchmark.parsers import RegexParser, IncrementalParser, StateMachineParser
from parser_benchmark.parsers.incremental_parser import StreamingParser
from parser_benchmark.models import ToolCall, ParseResult


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class VLLMConfig:
    """Configuration for vLLM server."""
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model: str | None = None
    tool_call_parser: str = "hermes"  # hermes, llama3_json, mistral, etc.

    @classmethod
    def local(cls, port: int = 8000, parser: str = "hermes") -> "VLLMConfig":
        return cls(base_url=f"http://localhost:{port}/v1", tool_call_parser=parser)

    @classmethod
    def from_url(cls, url: str, parser: str = "hermes") -> "VLLMConfig":
        base_url = url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        return cls(base_url=base_url, tool_call_parser=parser)


@dataclass
class StreamingMetrics:
    """Metrics for streaming comparison."""
    vllm_total_time_ms: float
    incremental_first_call_ms: float
    incremental_total_time_ms: float
    tokens_before_first_call: int
    total_tokens: int
    advantage_ms: float
    advantage_percent: float


@dataclass
class ErrorRecoveryCase:
    """Test case for error recovery."""
    name: str
    category: str
    description: str
    input_text: str
    expected_calls: int  # Number of calls that should be recovered


@dataclass
class ErrorRecoveryResult:
    """Result of error recovery test."""
    case_name: str
    category: str
    input_text: str
    vllm_recovered: bool
    vllm_calls_found: int
    parser_results: dict[str, dict]  # parser_name -> {recovered, calls_found}
    winner: str  # Which parser did best


@dataclass
class ParserComparisonResult:
    """Result comparing a parser against vLLM."""
    parser_name: str
    accuracy: float  # Match rate with expected
    latency_ms: float
    calls_found: int
    success: bool


@dataclass
class SingleComparisonResult:
    """Result of comparing all parsers on a single prompt."""
    prompt: str
    raw_output: str
    vllm_calls: list[dict]
    vllm_latency_ms: float
    vllm_success: bool
    parser_results: dict[str, ParserComparisonResult]


@dataclass
class VLLMComparisonReport:
    """Complete comparison report."""
    timestamp: str
    vllm_model: str
    vllm_parser: str

    # Accuracy comparison
    accuracy_scores: dict[str, float]
    vllm_accuracy: float

    # Latency comparison
    avg_latency_ms: dict[str, float]
    vllm_avg_latency_ms: float

    # Streaming advantage
    streaming_advantage: StreamingMetrics | None

    # Error recovery
    error_recovery_results: list[ErrorRecoveryResult]
    error_recovery_summary: dict[str, int]  # parser_name -> wins

    # Raw results
    detailed_results: list[SingleComparisonResult]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "metadata": {
                "timestamp": self.timestamp,
                "vllm_model": self.vllm_model,
                "vllm_parser": self.vllm_parser,
            },
            "accuracy": {
                "vllm_native": self.vllm_accuracy,
                **self.accuracy_scores,
            },
            "latency_ms": {
                "vllm_native": self.vllm_avg_latency_ms,
                **self.avg_latency_ms,
            },
            "streaming_advantage": {
                "vllm_total_time_ms": self.streaming_advantage.vllm_total_time_ms,
                "incremental_first_call_ms": self.streaming_advantage.incremental_first_call_ms,
                "advantage_percent": self.streaming_advantage.advantage_percent,
            } if self.streaming_advantage else None,
            "error_recovery": {
                "summary": self.error_recovery_summary,
                "cases": [
                    {
                        "name": r.case_name,
                        "category": r.category,
                        "vllm_recovered": r.vllm_recovered,
                        "parser_results": r.parser_results,
                        "winner": r.winner,
                    }
                    for r in self.error_recovery_results
                ],
            },
            "detailed_results": [
                {
                    "prompt": r.prompt,
                    "raw_output": r.raw_output,
                    "vllm_calls": r.vllm_calls,
                    "vllm_latency_ms": r.vllm_latency_ms,
                    "parser_results": {
                        name: {
                            "accuracy": pr.accuracy,
                            "latency_ms": pr.latency_ms,
                            "calls_found": pr.calls_found,
                        }
                        for name, pr in r.parser_results.items()
                    },
                }
                for r in self.detailed_results
            ],
        }

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "VLLMComparisonReport":
        """Deserialize from JSON."""
        data = json.loads(json_str)
        # Simplified reconstruction
        return cls(
            timestamp=data["metadata"]["timestamp"],
            vllm_model=data["metadata"]["vllm_model"],
            vllm_parser=data["metadata"]["vllm_parser"],
            accuracy_scores={k: v for k, v in data["accuracy"].items() if k != "vllm_native"},
            vllm_accuracy=data["accuracy"]["vllm_native"],
            avg_latency_ms={k: v for k, v in data["latency_ms"].items() if k != "vllm_native"},
            vllm_avg_latency_ms=data["latency_ms"]["vllm_native"],
            streaming_advantage=StreamingMetrics(
                vllm_total_time_ms=data["streaming_advantage"]["vllm_total_time_ms"],
                incremental_first_call_ms=data["streaming_advantage"]["incremental_first_call_ms"],
                incremental_total_time_ms=data["streaming_advantage"].get("incremental_total_time_ms", 0),
                tokens_before_first_call=data["streaming_advantage"].get("tokens_before_first_call", 0),
                total_tokens=data["streaming_advantage"].get("total_tokens", 0),
                advantage_ms=data["streaming_advantage"].get("advantage_ms", 0),
                advantage_percent=data["streaming_advantage"]["advantage_percent"],
            ) if data.get("streaming_advantage") else None,
            error_recovery_results=[
                ErrorRecoveryResult(
                    case_name=c["name"],
                    category=c["category"],
                    input_text="",
                    vllm_recovered=c["vllm_recovered"],
                    vllm_calls_found=0,
                    parser_results=c["parser_results"],
                    winner=c["winner"],
                )
                for c in data.get("error_recovery", {}).get("cases", [])
            ],
            error_recovery_summary=data.get("error_recovery", {}).get("summary", {}),
            detailed_results=[],
        )


# ============================================================================
# ERROR RECOVERY TEST CASES
# ============================================================================

ERROR_RECOVERY_CASES = [
    # Truncated Output
    ErrorRecoveryCase(
        name="truncated_mid_json",
        category="truncated",
        description="JSON truncated in middle",
        input_text='{"name": "get_weather", "arguments": {"city": "NYC',
        expected_calls=0,
    ),
    ErrorRecoveryCase(
        name="truncated_after_first",
        category="truncated",
        description="Second call truncated, first should parse",
        input_text='{"name": "first", "arguments": {}} {"name": "second", "argu',
        expected_calls=1,
    ),
    ErrorRecoveryCase(
        name="truncated_array",
        category="truncated",
        description="Array truncated after first element",
        input_text='[{"name": "first", "arguments": {}}, {"name": "sec',
        expected_calls=1,
    ),

    # Malformed JSON
    ErrorRecoveryCase(
        name="missing_closing_brace",
        category="malformed",
        description="Missing closing brace",
        input_text='{"name": "test", "arguments": {"x": 1}',
        expected_calls=0,
    ),
    ErrorRecoveryCase(
        name="extra_comma",
        category="malformed",
        description="Trailing comma in object",
        input_text='{"name": "test", "arguments": {"x": 1,}}',
        expected_calls=0,
    ),
    ErrorRecoveryCase(
        name="single_quotes",
        category="malformed",
        description="Single quotes instead of double",
        input_text="{'name': 'test', 'arguments': {'city': 'NYC'}}",
        expected_calls=1,  # Regex parser can handle this
    ),
    ErrorRecoveryCase(
        name="unquoted_key",
        category="malformed",
        description="Unquoted key in JSON",
        input_text='{name: "test", "arguments": {}}',
        expected_calls=0,
    ),

    # Mixed Valid/Invalid
    ErrorRecoveryCase(
        name="valid_then_garbage",
        category="mixed",
        description="Valid call followed by garbage",
        input_text='{"name": "valid", "arguments": {}} this is garbage {not json}',
        expected_calls=1,
    ),
    ErrorRecoveryCase(
        name="garbage_then_valid",
        category="mixed",
        description="Garbage followed by valid call",
        input_text='random text {broken} more text {"name": "valid", "arguments": {"x": 1}}',
        expected_calls=1,
    ),
    ErrorRecoveryCase(
        name="valid_invalid_valid",
        category="mixed",
        description="Valid calls sandwiching invalid",
        input_text='{"name": "a", "arguments": {}} {bad json} {"name": "b", "arguments": {}}',
        expected_calls=2,
    ),

    # Unicode Edge Cases
    ErrorRecoveryCase(
        name="emoji_in_value",
        category="unicode",
        description="Emoji in argument value",
        input_text='{"name": "react", "arguments": {"emoji": "🔥"}}',
        expected_calls=1,
    ),
    ErrorRecoveryCase(
        name="chinese_characters",
        category="unicode",
        description="Chinese characters in arguments",
        input_text='{"name": "translate", "arguments": {"text": "你好世界"}}',
        expected_calls=1,
    ),
    ErrorRecoveryCase(
        name="rtl_text",
        category="unicode",
        description="Right-to-left Arabic text",
        input_text='{"name": "translate", "arguments": {"text": "مرحبا"}}',
        expected_calls=1,
    ),

    # Format Confusion
    ErrorRecoveryCase(
        name="xml_malformed_inner",
        category="format",
        description="XML wrapper with malformed JSON inside",
        input_text='<tool_call>{"name": "test", broken json}</tool_call>',
        expected_calls=0,
    ),
    ErrorRecoveryCase(
        name="nested_json_string",
        category="format",
        description="JSON containing escaped JSON in string",
        input_text='{"name": "parse", "arguments": {"data": "{\\"inner\\": \\"value\\"}"}}',
        expected_calls=1,
    ),
    ErrorRecoveryCase(
        name="xml_then_json",
        category="format",
        description="XML format followed by JSON format",
        input_text='<tool_call>{"name": "first", "arguments": {}}</tool_call> {"name": "second", "arguments": {}}',
        expected_calls=2,
    ),

    # Streaming Edge Cases
    ErrorRecoveryCase(
        name="incomplete_escape",
        category="streaming",
        description="Incomplete escape sequence",
        input_text='{"name": "test", "arguments": {"path": "C:\\',
        expected_calls=0,
    ),
    ErrorRecoveryCase(
        name="partial_unicode",
        category="streaming",
        description="Partial unicode escape",
        input_text='{"name": "test", "arguments": {"char": "\\u00',
        expected_calls=0,
    ),

    # Deeply Nested
    ErrorRecoveryCase(
        name="deep_nesting",
        category="nested",
        description="Deeply nested valid JSON",
        input_text='{"name": "test", "arguments": {"a": {"b": {"c": {"d": {"e": 1}}}}}}',
        expected_calls=1,
    ),
    ErrorRecoveryCase(
        name="nested_arrays",
        category="nested",
        description="Nested arrays in arguments",
        input_text='{"name": "test", "arguments": {"matrix": [[1, 2], [3, 4]]}}',
        expected_calls=1,
    ),
]


# ============================================================================
# DEFAULT TOOLS FOR TESTING
# ============================================================================

DEFAULT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform math calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"},
                },
                "required": ["expression"],
            },
        },
    },
]

DEFAULT_TEST_PROMPTS = [
    "What's the weather like in San Francisco?",
    "Search for the latest Python tutorials",
    "Calculate 15 * 7 + 23",
    "What's the weather in Tokyo in celsius?",
    "Find information about machine learning",
]


# ============================================================================
# VLLM COMPARISON RUNNER
# ============================================================================

class VLLMComparisonRunner:
    """Run comparisons between custom parsers and vLLM native parsing."""

    def __init__(self, config: VLLMConfig):
        self.config = config
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )
        self._model = config.model

        # Initialize parsers
        self.parsers = {
            "regex": RegexParser(),
            "incremental": IncrementalParser(),
            "state_machine": StateMachineParser(),
        }

    @property
    def model(self) -> str:
        """Get model name, auto-detecting if needed."""
        if self._model is None:
            try:
                models = self.client.models.list()
                self._model = models.data[0].id
            except Exception:
                self._model = "unknown"
        return self._model

    def check_connection(self) -> tuple[bool, str]:
        """Check if vLLM server is reachable."""
        try:
            models = self.client.models.list()
            return True, f"Connected to vLLM with model: {models.data[0].id}"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"

    def get_raw_output(
        self,
        prompt: str,
        tools: list[dict] = None,
        system_prompt: str = None,
    ) -> tuple[str, float]:
        """Get raw text output from vLLM (no native tool parsing).

        Returns (raw_text, elapsed_ms).
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()

        # Request without tool_choice to get raw output
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=512,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        content = response.choices[0].message.content or ""

        return content, elapsed_ms

    def get_native_tool_calls(
        self,
        prompt: str,
        tools: list[dict] = None,
    ) -> tuple[list[dict], float, bool]:
        """Get tool calls using vLLM's native parsing.

        Returns (tool_calls, elapsed_ms, success).
        """
        if tools is None:
            tools = DEFAULT_TOOLS

        messages = [{"role": "user", "content": prompt}]

        start = time.perf_counter()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )

            elapsed_ms = (time.perf_counter() - start) * 1000

            tool_calls = []
            if response.choices[0].message.tool_calls:
                for tc in response.choices[0].message.tool_calls:
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

            return tool_calls, elapsed_ms, True

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return [], elapsed_ms, False

    def compare_single(
        self,
        prompt: str,
        tools: list[dict] = None,
        system_prompt: str = None,
    ) -> SingleComparisonResult:
        """Compare all parsers on a single prompt."""

        # Get raw output for custom parsers to parse
        raw_output, raw_elapsed = self.get_raw_output(prompt, tools, system_prompt)

        # Get vLLM native tool calls
        vllm_calls, vllm_elapsed, vllm_success = self.get_native_tool_calls(prompt, tools)

        # Test each custom parser
        parser_results = {}
        for name, parser in self.parsers.items():
            result = parser.parse(raw_output)

            # Calculate accuracy (match with vLLM)
            if vllm_calls and result.tool_calls:
                matches = sum(
                    1 for vc in vllm_calls
                    if any(tc.name == vc["name"] for tc in result.tool_calls)
                )
                accuracy = matches / len(vllm_calls) * 100
            elif not vllm_calls and not result.tool_calls:
                accuracy = 100.0
            else:
                accuracy = 0.0

            parser_results[name] = ParserComparisonResult(
                parser_name=name,
                accuracy=accuracy,
                latency_ms=result.parse_time_ms,
                calls_found=len(result.tool_calls),
                success=result.success,
            )

        return SingleComparisonResult(
            prompt=prompt,
            raw_output=raw_output,
            vllm_calls=vllm_calls,
            vllm_latency_ms=vllm_elapsed,
            vllm_success=vllm_success,
            parser_results=parser_results,
        )

    def compare_streaming(self, prompt: str, tools: list[dict] = None) -> StreamingMetrics:
        """Compare streaming performance: incremental parser vs vLLM buffered.

        This demonstrates the streaming advantage of the incremental parser.
        """
        if tools is None:
            tools = DEFAULT_TOOLS

        messages = [{"role": "user", "content": prompt}]

        # Stream response and measure time-to-first-call for incremental parser
        streaming_parser = StreamingParser()

        first_call_time = None
        tokens = 0
        tokens_before_first = 0

        start = time.perf_counter()

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=256,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    tokens += 1

                    # Feed to streaming parser
                    new_calls = streaming_parser.feed(content)

                    if new_calls and first_call_time is None:
                        first_call_time = (time.perf_counter() - start) * 1000
                        tokens_before_first = tokens

            total_time = (time.perf_counter() - start) * 1000

        except Exception:
            total_time = (time.perf_counter() - start) * 1000
            first_call_time = total_time
            tokens_before_first = tokens

        # vLLM native must wait for complete response
        vllm_time = total_time  # Same as total streaming time

        if first_call_time is None:
            first_call_time = total_time
            tokens_before_first = tokens

        advantage_ms = vllm_time - first_call_time
        advantage_percent = (advantage_ms / vllm_time * 100) if vllm_time > 0 else 0

        return StreamingMetrics(
            vllm_total_time_ms=vllm_time,
            incremental_first_call_ms=first_call_time,
            incremental_total_time_ms=total_time,
            tokens_before_first_call=tokens_before_first,
            total_tokens=tokens,
            advantage_ms=advantage_ms,
            advantage_percent=advantage_percent,
        )

    def test_error_recovery(self, cases: list[ErrorRecoveryCase] = None) -> list[ErrorRecoveryResult]:
        """Test error recovery on malformed inputs."""
        if cases is None:
            cases = ERROR_RECOVERY_CASES

        results = []

        for case in cases:
            parser_results = {}
            best_recovery = 0
            winner = "none"

            for name, parser in self.parsers.items():
                result = parser.parse(case.input_text)
                calls_found = len(result.tool_calls)
                recovered = calls_found >= case.expected_calls and case.expected_calls > 0

                parser_results[name] = {
                    "recovered": recovered,
                    "calls_found": calls_found,
                    "success": result.success,
                }

                if calls_found > best_recovery:
                    best_recovery = calls_found
                    winner = name

            # For this test, vLLM wouldn't be able to parse raw malformed text
            # (it's designed for model output, not arbitrary text)
            vllm_recovered = False

            results.append(ErrorRecoveryResult(
                case_name=case.name,
                category=case.category,
                input_text=case.input_text,
                vllm_recovered=vllm_recovered,
                vllm_calls_found=0,
                parser_results=parser_results,
                winner=winner if best_recovery > 0 else "none",
            ))

        return results

    def run_full_comparison(
        self,
        prompts: list[str] = None,
        tools: list[dict] = None,
        test_streaming: bool = True,
        test_error_recovery: bool = True,
    ) -> VLLMComparisonReport:
        """Run complete comparison and generate report."""
        if prompts is None:
            prompts = DEFAULT_TEST_PROMPTS
        if tools is None:
            tools = DEFAULT_TOOLS

        # Run comparisons for each prompt
        detailed_results = []
        for prompt in prompts:
            result = self.compare_single(prompt, tools)
            detailed_results.append(result)

        # Calculate aggregate metrics
        accuracy_scores = {name: 0.0 for name in self.parsers}
        latency_sums = {name: 0.0 for name in self.parsers}
        vllm_accuracy = 0.0
        vllm_latency_sum = 0.0

        for result in detailed_results:
            if result.vllm_success:
                vllm_accuracy += 100.0
            vllm_latency_sum += result.vllm_latency_ms

            for name, pr in result.parser_results.items():
                accuracy_scores[name] += pr.accuracy
                latency_sums[name] += pr.latency_ms

        n = len(detailed_results)
        accuracy_scores = {k: v / n for k, v in accuracy_scores.items()}
        avg_latency = {k: v / n for k, v in latency_sums.items()}
        vllm_accuracy = vllm_accuracy / n
        vllm_avg_latency = vllm_latency_sum / n

        # Streaming comparison
        streaming = None
        if test_streaming and prompts:
            streaming = self.compare_streaming(prompts[0], tools)

        # Error recovery
        error_results = []
        error_summary = {name: 0 for name in self.parsers}
        if test_error_recovery:
            error_results = self.test_error_recovery()
            for er in error_results:
                if er.winner != "none":
                    error_summary[er.winner] = error_summary.get(er.winner, 0) + 1

        return VLLMComparisonReport(
            timestamp=datetime.now().isoformat(),
            vllm_model=self.model,
            vllm_parser=self.config.tool_call_parser,
            accuracy_scores=accuracy_scores,
            vllm_accuracy=vllm_accuracy,
            avg_latency_ms=avg_latency,
            vllm_avg_latency_ms=vllm_avg_latency,
            streaming_advantage=streaming,
            error_recovery_results=error_results,
            error_recovery_summary=error_summary,
            detailed_results=detailed_results,
        )

    def save_results(self, report: VLLMComparisonReport, filepath: str):
        """Save report to JSON file."""
        with open(filepath, "w") as f:
            f.write(report.to_json())


# ============================================================================
# STANDALONE FUNCTIONS FOR SIMPLE USE
# ============================================================================

def run_quick_comparison(vllm_url: str, parser: str = "hermes") -> VLLMComparisonReport:
    """Quick comparison with default settings."""
    config = VLLMConfig.from_url(vllm_url, parser)
    runner = VLLMComparisonRunner(config)
    return runner.run_full_comparison()


def test_error_recovery_standalone(test_cases: list[ErrorRecoveryCase] = None) -> list[ErrorRecoveryResult]:
    """Test error recovery without vLLM connection."""
    parsers = {
        "regex": RegexParser(),
        "incremental": IncrementalParser(),
        "state_machine": StateMachineParser(),
    }

    if test_cases is None:
        test_cases = ERROR_RECOVERY_CASES

    results = []
    for case in test_cases:
        parser_results = {}
        best_recovery = 0
        winner = "none"

        for name, parser in parsers.items():
            result = parser.parse(case.input_text)
            calls_found = len(result.tool_calls)
            recovered = calls_found >= case.expected_calls and case.expected_calls > 0

            parser_results[name] = {
                "recovered": recovered,
                "calls_found": calls_found,
            }

            if calls_found > best_recovery:
                best_recovery = calls_found
                winner = name

        results.append(ErrorRecoveryResult(
            case_name=case.name,
            category=case.category,
            input_text=case.input_text,
            vllm_recovered=False,
            vllm_calls_found=0,
            parser_results=parser_results,
            winner=winner if best_recovery > 0 else "none",
        ))

    return results
