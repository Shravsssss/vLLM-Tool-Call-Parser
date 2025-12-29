"""Benchmark runner for parser evaluation."""

import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from parser_benchmark.models import ParseResult, ToolCall
from parser_benchmark.parsers.base import BaseParser


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    iterations: int = 100
    warmup_iterations: int = 10
    measure_memory: bool = False


@dataclass
class TimingResult:
    """Timing statistics from a benchmark."""
    total_ms: float
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    p95_ms: float
    p99_ms: float
    std_dev_ms: float
    iterations: int
    parses_per_second: float


@dataclass
class BenchmarkResult:
    """Complete result from a benchmark run."""
    parser_name: str
    timing: TimingResult
    success_rate: float
    total_calls_found: int
    errors: list[str] = field(default_factory=list)


class BenchmarkRunner:
    """Runs benchmarks across multiple parsers."""

    def __init__(self, config: BenchmarkConfig | None = None):
        """Initialize the benchmark runner.

        Args:
            config: Benchmark configuration.
        """
        self.config = config or BenchmarkConfig()

    def run(
        self,
        parser: BaseParser,
        test_cases: list[str],
    ) -> BenchmarkResult:
        """Run benchmark on a parser.

        Args:
            parser: Parser to benchmark.
            test_cases: List of test input strings.

        Returns:
            BenchmarkResult with timing and accuracy data.
        """
        # Warmup
        for _ in range(self.config.warmup_iterations):
            for text in test_cases[:min(5, len(test_cases))]:
                parser.parse(text)

        # Actual benchmark
        times: list[float] = []
        successes = 0
        total_calls = 0
        errors: list[str] = []

        for _ in range(self.config.iterations):
            for text in test_cases:
                start = time.perf_counter()
                result = parser.parse(text)
                elapsed_ms = (time.perf_counter() - start) * 1000

                times.append(elapsed_ms)
                if result.success:
                    successes += 1
                total_calls += result.num_calls
                if result.error:
                    errors.append(result.error)

        # Calculate statistics
        times_sorted = sorted(times)
        total_parses = len(times)
        total_time = sum(times)

        p95_idx = int(len(times_sorted) * 0.95)
        p99_idx = int(len(times_sorted) * 0.99)

        timing = TimingResult(
            total_ms=total_time,
            mean_ms=statistics.mean(times),
            median_ms=statistics.median(times),
            min_ms=min(times),
            max_ms=max(times),
            p95_ms=times_sorted[min(p95_idx, len(times_sorted) - 1)],
            p99_ms=times_sorted[min(p99_idx, len(times_sorted) - 1)],
            std_dev_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
            iterations=self.config.iterations,
            parses_per_second=(total_parses / total_time) * 1000 if total_time > 0 else 0
        )

        return BenchmarkResult(
            parser_name=parser.name,
            timing=timing,
            success_rate=(successes / total_parses) * 100 if total_parses > 0 else 0,
            total_calls_found=total_calls,
            errors=list(set(errors))[:10]  # Dedupe and limit
        )

    def compare(
        self,
        parsers: list[BaseParser],
        test_cases: list[str],
    ) -> dict[str, BenchmarkResult]:
        """Run benchmarks on multiple parsers for comparison.

        Args:
            parsers: List of parsers to compare.
            test_cases: Test inputs.

        Returns:
            Dictionary mapping parser names to results.
        """
        results = {}
        for parser in parsers:
            results[parser.name] = self.run(parser, test_cases)
        return results

    def format_results(self, results: dict[str, BenchmarkResult]) -> str:
        """Format benchmark results as a table.

        Args:
            results: Dictionary of benchmark results.

        Returns:
            Formatted string table.
        """
        lines = []
        lines.append("=" * 80)
        lines.append("BENCHMARK RESULTS")
        lines.append("=" * 80)
        lines.append(
            f"{'Parser':<25} {'Avg (ms)':<12} {'p95 (ms)':<12} "
            f"{'Parses/s':<15} {'Success':<10}"
        )
        lines.append("-" * 80)

        for name, result in results.items():
            lines.append(
                f"{name:<25} {result.timing.mean_ms:<12.4f} "
                f"{result.timing.p95_ms:<12.4f} "
                f"{result.timing.parses_per_second:<15,.0f} "
                f"{result.success_rate:<10.1f}%"
            )

        lines.append("=" * 80)
        return "\n".join(lines)