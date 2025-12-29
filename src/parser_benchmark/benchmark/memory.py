"""Memory profiling for parsers."""

import tracemalloc
from dataclasses import dataclass
from typing import Callable


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    peak_mb: float
    current_mb: float
    allocations: int


class MemoryProfiler:
    """Profile memory usage of parser operations."""

    @staticmethod
    def profile(func: Callable, *args, **kwargs) -> tuple[any, MemoryStats]:
        """Profile memory usage of a function call.

        Args:
            func: Function to profile.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Tuple of (function result, MemoryStats).
        """
        tracemalloc.start()

        result = func(*args, **kwargs)

        current, peak = tracemalloc.get_traced_memory()
        snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()

        stats = MemoryStats(
            peak_mb=peak / 1024 / 1024,
            current_mb=current / 1024 / 1024,
            allocations=len(snapshot.statistics('lineno'))
        )

        return result, stats

    @staticmethod
    def compare_parsers(parsers: list, test_input: str) -> dict[str, MemoryStats]:
        """Compare memory usage across parsers.

        Args:
            parsers: List of parser instances.
            test_input: Input to parse.

        Returns:
            Dictionary mapping parser names to MemoryStats.
        """
        results = {}
        for parser in parsers:
            _, stats = MemoryProfiler.profile(parser.parse, test_input)
            results[parser.name] = stats
        return results