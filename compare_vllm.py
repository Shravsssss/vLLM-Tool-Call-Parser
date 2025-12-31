#!/usr/bin/env python
"""Compare custom parsers against vLLM native tool parsing.

This script runs comprehensive comparisons and generates results JSON
that can be uploaded to the HuggingFace Space dashboard.

Usage:
    # With vLLM server running locally
    python compare_vllm.py --url http://localhost:8000

    # With specific parser type
    python compare_vllm.py --url http://localhost:8000 --parser llama3_json

    # Test error recovery only (no vLLM needed)
    python compare_vllm.py --error-recovery-only

    # Save results to specific file
    python compare_vllm.py --url http://localhost:8000 --output my_results.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from parser_benchmark.vllm_comparison import (
    VLLMConfig,
    VLLMComparisonRunner,
    test_error_recovery_standalone,
    ERROR_RECOVERY_CASES,
    DEFAULT_TEST_PROMPTS,
)


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_comparison_summary(report):
    """Print a human-readable summary of the comparison."""
    print_header("COMPARISON SUMMARY")

    print(f"\nvLLM Model: {report.vllm_model}")
    print(f"vLLM Parser: {report.vllm_parser}")
    print(f"Timestamp: {report.timestamp}")

    print_header("ACCURACY COMPARISON")
    print(f"\n{'Parser':<25} {'Accuracy':>10}")
    print("-" * 40)
    print(f"{'vLLM Native':<25} {report.vllm_accuracy:>9.1f}%")
    for name, acc in report.accuracy_scores.items():
        print(f"{name:<25} {acc:>9.1f}%")

    print_header("LATENCY COMPARISON (parsing only)")
    print(f"\n{'Parser':<25} {'Avg Latency':>12}")
    print("-" * 40)
    print(f"{'vLLM Native':<25} {report.vllm_avg_latency_ms:>10.2f} ms")
    for name, lat in report.avg_latency_ms.items():
        speedup = report.vllm_avg_latency_ms / lat if lat > 0 else 0
        print(f"{name:<25} {lat:>10.2f} ms ({speedup:.1f}x faster)")

    if report.streaming_advantage:
        print_header("STREAMING ADVANTAGE")
        sa = report.streaming_advantage
        print(f"\nvLLM waits for complete response: {sa.vllm_total_time_ms:.1f} ms")
        print(f"Incremental parser first detection: {sa.incremental_first_call_ms:.1f} ms")
        print(f"Tokens before first call: {sa.tokens_before_first_call}/{sa.total_tokens}")
        print(f"\nAdvantage: {sa.advantage_ms:.1f} ms ({sa.advantage_percent:.1f}% earlier detection)")

    if report.error_recovery_results:
        print_header("ERROR RECOVERY RESULTS")
        print(f"\nTotal test cases: {len(report.error_recovery_results)}")
        print(f"\nWins by parser:")
        for parser, wins in report.error_recovery_summary.items():
            print(f"  {parser}: {wins} cases")

        print(f"\n{'Case':<30} {'Category':<12} {'Winner':<15}")
        print("-" * 60)
        for er in report.error_recovery_results[:10]:  # Show first 10
            print(f"{er.case_name:<30} {er.category:<12} {er.winner:<15}")
        if len(report.error_recovery_results) > 10:
            print(f"... and {len(report.error_recovery_results) - 10} more cases")


def print_error_recovery_results(results):
    """Print error recovery test results."""
    print_header("ERROR RECOVERY TEST RESULTS")

    # Group by category
    categories = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = []
        categories[r.category].append(r)

    for category, cases in categories.items():
        print(f"\n{category.upper()}")
        print("-" * 50)
        for case in cases:
            print(f"\n  {case.case_name}:")
            for parser, result in case.parser_results.items():
                status = "OK" if result["calls_found"] > 0 else "FAIL"
                print(f"    {parser}: {status} ({result['calls_found']} calls)")
            print(f"    Winner: {case.winner}")

    # Summary
    print_header("SUMMARY")
    wins = {}
    for r in results:
        if r.winner != "none":
            wins[r.winner] = wins.get(r.winner, 0) + 1

    print("\nRecovery wins by parser:")
    for parser, count in sorted(wins.items(), key=lambda x: -x[1]):
        print(f"  {parser}: {count} cases")


def main():
    parser = argparse.ArgumentParser(
        description="Compare custom parsers vs vLLM native parsing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_vllm.py --url http://localhost:8000
  python compare_vllm.py --url http://localhost:8000 --parser llama3_json
  python compare_vllm.py --error-recovery-only
  python compare_vllm.py --url http://localhost:8000 --output results.json
        """
    )

    parser.add_argument(
        "--url",
        help="vLLM server URL (e.g., http://localhost:8000)",
    )
    parser.add_argument(
        "--parser",
        default="hermes",
        choices=["hermes", "llama3_json", "mistral", "granite", "jamba", "internlm"],
        help="vLLM tool-call-parser type (default: hermes)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file path (default: vllm_comparison_TIMESTAMP.json)",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        help="Custom test prompts (space-separated)",
    )
    parser.add_argument(
        "--error-recovery-only",
        action="store_true",
        help="Only run error recovery tests (no vLLM connection needed)",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Skip streaming comparison test",
    )
    parser.add_argument(
        "--no-error-recovery",
        action="store_true",
        help="Skip error recovery tests",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Error recovery only mode
    if args.error_recovery_only:
        print_header("ERROR RECOVERY TEST (No vLLM)")
        results = test_error_recovery_standalone()
        print_error_recovery_results(results)

        # Save results
        if args.output:
            output_path = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"error_recovery_results_{timestamp}.json"

        with open(output_path, "w") as f:
            json.dump({
                "type": "error_recovery_only",
                "timestamp": datetime.now().isoformat(),
                "results": [
                    {
                        "name": r.case_name,
                        "category": r.category,
                        "input": r.input_text,
                        "parser_results": r.parser_results,
                        "winner": r.winner,
                    }
                    for r in results
                ]
            }, f, indent=2)

        print(f"\nResults saved to: {output_path}")
        return

    # Full comparison requires vLLM URL
    if not args.url:
        parser.error("--url is required (or use --error-recovery-only)")

    # Setup config
    config = VLLMConfig.from_url(args.url, args.parser)
    runner = VLLMComparisonRunner(config)

    # Check connection
    print_header("CONNECTING TO VLLM")
    connected, message = runner.check_connection()
    print(f"\n{message}")

    if not connected:
        print("\nError: Could not connect to vLLM server")
        print("Make sure vLLM is running with:")
        print(f"  python -m vllm.entrypoints.openai.api_server \\")
        print(f"    --model YOUR_MODEL \\")
        print(f"    --tool-call-parser {args.parser} \\")
        print(f"    --enable-auto-tool-choice")
        sys.exit(1)

    # Get prompts
    prompts = args.prompts if args.prompts else DEFAULT_TEST_PROMPTS

    # Run comparison
    print_header("RUNNING COMPARISON")
    print(f"\nTesting with {len(prompts)} prompts...")

    report = runner.run_full_comparison(
        prompts=prompts,
        test_streaming=not args.no_streaming,
        test_error_recovery=not args.no_error_recovery,
    )

    # Print summary
    print_comparison_summary(report)

    # Save results
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"vllm_comparison_{timestamp}.json"

    runner.save_results(report, output_path)
    print(f"\n{'=' * 60}")
    print(f"  Results saved to: {output_path}")
    print(f"  Upload this file to the HuggingFace dashboard!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
