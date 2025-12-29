#!/usr/bin/env python
"""
Parser Benchmark Suite

Measures parsing performance across various input types and provides
detailed latency statistics.

Usage:
    python benchmark.py
    python benchmark.py --iterations 1000
    python benchmark.py --category --verbose
"""

import argparse
import itertools
import statistics
import sys
import time
from typing import NamedTuple

sys.path.insert(0, 'src')

from parser_benchmark.parsers import RegexParser
from parser_benchmark.models import ParseResult


class BenchmarkResult(NamedTuple):
    """Results from a benchmark run."""
    name: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    median_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    parses_per_second: float
    success_rate: float


SIMPLE_TOOL_CALLS = [
    '{"name": "get_weather", "arguments": {"city": "NYC"}}',
    '{"name": "search", "arguments": {"query": "python tutorial"}}',
    '{"name": "calculate", "arguments": {"x": 42, "y": 3.14}}',
    '{"name": "toggle", "arguments": {"enabled": true}}',
    '{"name": "clear", "arguments": {"value": null}}',
    '{"name": "ping", "arguments": {}}',
    '{"name": "greet", "arguments": {"message": "Hello, World!"}}',
    '{"name": "fetch_data", "arguments": {"url": "https://api.example.com"}}',
    '{"name": "send_email", "arguments": {"to": "test@example.com", "subject": "Test"}}',
    '{"name": "create_file", "arguments": {"path": "/tmp/test.txt", "content": "Hello"}}',
]

COMPLEX_TOOL_CALLS = [
    '{"name": "create_users", "arguments": {"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}}',
    '{"name": "process", "arguments": {"data": {"nested": {"deep": {"value": 123}}}}}',
    '{"name": "batch", "arguments": {"items": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}}',
    '{"name": "config", "arguments": {"settings": {"theme": "dark", "language": "en", "notifications": true}}}',
]

XML_WRAPPED_CALLS = [
    '<tool_call>{"name": "api_call", "arguments": {"endpoint": "/users"}}</tool_call>',
    '<tool_call>{"name": "func1", "arguments": {"a": 1}}</tool_call><tool_call>{"name": "func2", "arguments": {"b": 2}}</tool_call>',
    '''<tool_call>{"name": "step1", "arguments": {}}</tool_call>
    <tool_call>{"name": "step2", "arguments": {}}</tool_call>
    <tool_call>{"name": "step3", "arguments": {}}</tool_call>''',
]

EMBEDDED_IN_TEXT = [
    '''I will help you with that.
    {"name": "search", "arguments": {"query": "test"}}
    Let me know if you need more.''',
    'Here is the result: {"name": "calc", "arguments": {"x": 5}} Done!',
    'Processing... {"name": "fetch", "arguments": {"url": "http://example.com"}} Complete.',
]

ARRAY_OF_CALLS = [
    '[{"name": "func1", "arguments": {"a": 1}}, {"name": "func2", "arguments": {"b": 2}}]',
    '[{"name": "step1", "arguments": {}}, {"name": "step2", "arguments": {}}, {"name": "step3", "arguments": {}}]',
]

UNICODE_CALLS = [
    '{"name": "translate", "arguments": {"text": "Hello, World!", "lang": "es"}}',
    '{"name": "process", "arguments": {"data": "Line1\\nLine2\\nLine3"}}',
    '{"name": "search", "arguments": {"query": "test & verify"}}',
]

MALFORMED_CALLS = [
    '{"name": "test", "arguments": {"x": 1,}}',
    "{'name': 'test', 'arguments': {'x': 1}}",
    '{"name": "test", "arguments": {"x": 1}',
]

LARGE_CALLS = [
    '{"name": "bulk_create", "arguments": {"items": ' + str(list(range(100))) + '}}',
    '{"name": "long_text", "arguments": {"content": "' + 'x' * 1000 + '"}}',
]


def generate_test_data(count: int = 100) -> list[str]:
    """Generate a mixed set of test data."""
    all_data = (
        SIMPLE_TOOL_CALLS * 5 +
        COMPLEX_TOOL_CALLS * 2 +
        XML_WRAPPED_CALLS * 2 +
        EMBEDDED_IN_TEXT * 2 +
        ARRAY_OF_CALLS +
        UNICODE_CALLS +
        MALFORMED_CALLS +
        LARGE_CALLS
    )
    return list(itertools.islice(itertools.cycle(all_data), count))


def run_benchmark(
    parser: RegexParser,
    test_data: list[str],
    iterations: int = 100,
    name: str = "benchmark"
) -> BenchmarkResult:
    """Run a benchmark on the parser."""
    times: list[float] = []
    successes = 0
    total_parses = 0

    for _ in range(iterations):
        for text in test_data:
            start = time.perf_counter()
            result = parser.parse(text)
            elapsed_ms = (time.perf_counter() - start) * 1000

            times.append(elapsed_ms)
            total_parses += 1
            if result.success:
                successes += 1

    times_sorted = sorted(times)
    total_time = sum(times)

    p95_idx = int(len(times_sorted) * 0.95)
    p99_idx = int(len(times_sorted) * 0.99)

    return BenchmarkResult(
        name=name,
        iterations=iterations,
        total_time_ms=total_time,
        avg_time_ms=statistics.mean(times),
        min_time_ms=min(times),
        max_time_ms=max(times),
        median_time_ms=statistics.median(times),
        p95_time_ms=times_sorted[p95_idx] if p95_idx < len(times_sorted) else times_sorted[-1],
        p99_time_ms=times_sorted[p99_idx] if p99_idx < len(times_sorted) else times_sorted[-1],
        parses_per_second=(total_parses / total_time) * 1000 if total_time > 0 else 0,
        success_rate=(successes / total_parses) * 100 if total_parses > 0 else 0,
    )


def print_result(result: BenchmarkResult, verbose: bool = False) -> None:
    """Print benchmark results."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {result.name}")
    print(f"{'='*60}")
    print(f"  Iterations:        {result.iterations}")
    print(f"  Total time:        {result.total_time_ms:.2f} ms")
    print(f"  Parses/second:     {result.parses_per_second:,.0f}")
    print(f"  Success rate:      {result.success_rate:.1f}%")
    print()
    print("  Latency Statistics:")
    print(f"    Average:         {result.avg_time_ms:.4f} ms")
    print(f"    Median (p50):    {result.median_time_ms:.4f} ms")
    print(f"    Min:             {result.min_time_ms:.4f} ms")
    print(f"    Max:             {result.max_time_ms:.4f} ms")
    print(f"    p95:             {result.p95_time_ms:.4f} ms")
    print(f"    p99:             {result.p99_time_ms:.4f} ms")


def run_category_benchmarks(
    parser: RegexParser,
    iterations: int,
    verbose: bool
) -> list[BenchmarkResult]:
    """Run benchmarks for each category of test data."""
    results = []

    categories = [
        ("Simple Tool Calls", SIMPLE_TOOL_CALLS),
        ("Complex Tool Calls", COMPLEX_TOOL_CALLS),
        ("XML Wrapped", XML_WRAPPED_CALLS),
        ("Embedded in Text", EMBEDDED_IN_TEXT),
        ("Array of Calls", ARRAY_OF_CALLS),
        ("Unicode/Special Chars", UNICODE_CALLS),
        ("Malformed JSON", MALFORMED_CALLS),
        ("Large Payloads", LARGE_CALLS),
    ]

    for name, data in categories:
        if verbose:
            print(f"  Running: {name}...")
        result = run_benchmark(parser, data, iterations, name)
        results.append(result)

    return results


def print_summary(results: list[BenchmarkResult]) -> None:
    """Print a summary table of all benchmark results."""
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Category':<25} {'Avg (ms)':<12} {'p95 (ms)':<12} {'Parses/s':<15} {'Success':<10}")
    print(f"{'-'*80}")

    for r in results:
        print(f"{r.name:<25} {r.avg_time_ms:<12.4f} {r.p95_time_ms:<12.4f} {r.parses_per_second:<15,.0f} {r.success_rate:<10.1f}%")

    avg_throughput = statistics.mean(r.parses_per_second for r in results)
    avg_latency = statistics.mean(r.avg_time_ms for r in results)

    print(f"{'-'*80}")
    print(f"{'AVERAGE':<25} {avg_latency:<12.4f} {'-':<12} {avg_throughput:<15,.0f}")


def main():
    """Main entry point."""
    arg_parser = argparse.ArgumentParser(
        description="Benchmark the RegexParser",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    arg_parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=100,
        help="Number of iterations per test (default: 100)"
    )
    arg_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )
    arg_parser.add_argument(
        "--category", "-c",
        action="store_true",
        help="Run per-category benchmarks instead of mixed"
    )

    args = arg_parser.parse_args()

    print("=" * 60)
    print("Parser Benchmark Suite")
    print("=" * 60)
    print(f"Iterations: {args.iterations}")

    parser = RegexParser()
    print(f"Parser: {parser.name}")

    if args.category:
        print("\nRunning category benchmarks...")
        results = run_category_benchmarks(parser, args.iterations, args.verbose)

        if args.verbose:
            for result in results:
                print_result(result, args.verbose)

        print_summary(results)
    else:
        print("\nGenerating test data...")
        test_data = generate_test_data(100)
        print(f"Test samples: {len(test_data)}")

        print("\nRunning benchmark...")
        result = run_benchmark(parser, test_data, args.iterations, "Mixed Workload")
        print_result(result, args.verbose)

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
