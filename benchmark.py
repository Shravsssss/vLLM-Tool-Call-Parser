#!/usr/bin/env python
"""Run benchmarks comparing all parsers."""

import sys
sys.path.insert(0, 'src')

from parser_benchmark.parsers import RegexParser, IncrementalParser, StateMachineParser
from parser_benchmark.benchmark import BenchmarkRunner, BenchmarkConfig


# Test data
TEST_CASES = [
    '{"name": "simple", "arguments": {}}',
    '{"name": "with_args", "arguments": {"x": 1, "y": "test"}}',
    '[{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]',
    '<tool_call>{"name": "xml", "arguments": {}}</tool_call>',
    'Text {"name": "embedded", "arguments": {"key": "value"}} more text',
    '{"name": "nested", "arguments": {"data": {"inner": {"deep": 1}}}}',
    '{"name": "unicode", "arguments": {"text": "Hello 👋 世界"}}',
    '{"name": "special", "arguments": {"code": "if (x) { return y; }"}}',
] * 10  # Repeat for more iterations


def main():
    config = BenchmarkConfig(iterations=50, warmup_iterations=5)
    runner = BenchmarkRunner(config)

    parsers = [
        RegexParser(),
        IncrementalParser(),
        StateMachineParser(),
    ]

    print("Running benchmarks...")
    results = runner.compare(parsers, TEST_CASES)
    print(runner.format_results(results))


if __name__ == "__main__":
    main()