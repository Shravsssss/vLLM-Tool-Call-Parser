"""Accuracy metrics for parser evaluation."""

from dataclasses import dataclass

from parser_benchmark.models import ToolCall


@dataclass
class AccuracyMetrics:
    """Accuracy metrics comparing parsed vs expected results."""
    precision: float
    recall: float
    f1_score: float
    exact_match: bool
    true_positives: int
    false_positives: int
    false_negatives: int


def calculate_accuracy(
    parsed: list[ToolCall],
    expected: list[ToolCall]
) -> AccuracyMetrics:
    """Calculate accuracy metrics.

    Args:
        parsed: Tool calls extracted by parser.
        expected: Ground truth tool calls.

    Returns:
        AccuracyMetrics with precision, recall, F1.
    """
    # Convert to comparable format (name + sorted args)
    def to_key(tc: ToolCall) -> str:
        return f"{tc.name}:{sorted(tc.arguments.items())}"

    parsed_keys = set(to_key(tc) for tc in parsed)
    expected_keys = set(to_key(tc) for tc in expected)

    true_positives = len(parsed_keys & expected_keys)
    false_positives = len(parsed_keys - expected_keys)
    false_negatives = len(expected_keys - parsed_keys)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return AccuracyMetrics(
        precision=precision,
        recall=recall,
        f1_score=f1,
        exact_match=parsed_keys == expected_keys,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives
    )
