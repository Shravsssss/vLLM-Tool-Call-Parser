#!/usr/bin/env python
"""Compare how different parsers handle raw LLM tool call output.

Tests parsers against REAL LLM output (not pre-parsed API responses).
Uses Groq API to get raw text containing tool calls.

Usage:
    set GROQ_API_KEY=your-key-here
    python compare_llm.py
"""

import sys
import json
import os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, 'src')

# Load .env file if it exists
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

from parser_benchmark.parsers import RegexParser, IncrementalParser, StateMachineParser
from parser_benchmark.llm_client import LLMClient, LLMConfig


# Models to compare on Groq
GROQ_MODELS = [
    # Llama 4 (newest!)
    ("meta-llama/llama-4-scout-17b-16e-instruct", "Llama 4 Scout", "llama4_json"),
    # Llama 3.x
    ("llama-3.1-8b-instant", "Llama 3.1 8B", "llama3_json"),
    ("llama-3.3-70b-versatile", "Llama 3.3 70B", "llama3_json"),
    # Qwen (uses hermes format)
    ("qwen/qwen3-32b", "Qwen 3 32B", "hermes"),
    # GPT-OSS
    ("openai/gpt-oss-20b", "GPT-OSS 20B", "llama3_json"),
]


# System prompts for different tool call formats (matching vLLM parsers)
TOOL_CALL_FORMATS = {
    "llama3_json": """You are a helpful assistant with access to tools.
When you need to use a tool, output it in this exact JSON format:
{"name": "tool_name", "arguments": {"param1": "value1"}}

Available tools:
- get_weather: Get weather for a location. Parameters: city (string), state (string), unit (celsius/fahrenheit)
- search: Search the web. Parameters: query (string)
- calculate: Do math. Parameters: expression (string)

IMPORTANT: Output ONLY the JSON tool call, nothing else. No explanation before or after.""",

    "llama4_json": """You are a helpful assistant with access to tools.
When you need to use a tool, output it in this exact JSON format:
{"name": "tool_name", "arguments": {"param1": "value1"}}

Available tools:
- get_weather: Get weather for a location. Parameters: city (string), state (string), unit (celsius/fahrenheit)
- search: Search the web. Parameters: query (string)
- calculate: Do math. Parameters: expression (string)

IMPORTANT: Output ONLY the JSON tool call, nothing else. No explanation before or after.""",

    "hermes": """You are a helpful assistant with access to tools.
When you need to use a tool, wrap it in XML tags like this:
<tool_call>{"name": "tool_name", "arguments": {"param1": "value1"}}</tool_call>

Available tools:
- get_weather: Get weather for a location. Parameters: city (string), state (string), unit (celsius/fahrenheit)
- search: Search the web. Parameters: query (string)
- calculate: Do math. Parameters: expression (string)

IMPORTANT: Output ONLY the tool call in the format shown, nothing else.""",
}


# Test prompts
TEST_PROMPTS = [
    "What's the weather in San Francisco, CA?",
    "Search for Python programming news",
    "Calculate 25 * 4 + 10",
    "What's the temperature in New York City in fahrenheit?",
    "Look up machine learning",
]


def get_raw_llm_output(client: LLMClient, prompt: str, format_type: str) -> dict:
    """Get raw text output from LLM (not parsed tool calls)."""
    system_prompt = TOOL_CALL_FORMATS.get(format_type, TOOL_CALL_FORMATS["llama3_json"])

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    try:
        response = client.chat_raw(messages, max_tokens=256)
        return {
            "raw_output": response["content"],
            "elapsed_ms": response["elapsed_ms"],
            "success": True,
        }
    except Exception as e:
        return {
            "raw_output": "",
            "elapsed_ms": 0,
            "success": False,
            "error": str(e),
        }


def test_parsers_on_raw_output(raw_output: str) -> dict:
    """Test all parsers on raw LLM output.

    Returns dict with each parser's results.
    """
    parsers = [
        ("regex", RegexParser()),
        ("incremental", IncrementalParser()),
        ("state_machine", StateMachineParser()),
    ]

    results = {}
    for name, parser in parsers:
        parsed = parser.parse(raw_output)
        results[name] = {
            "success": parsed.success,
            "num_calls": parsed.num_calls,
            "tool_calls": [
                {"name": tc.name, "arguments": tc.arguments}
                for tc in parsed.tool_calls
            ],
            "parse_time_ms": parsed.parse_time_ms,
        }

    return results


def run_model_comparison(api_key: str, models: list = None) -> dict:
    """Run comparison across multiple models with raw output parsing."""
    if models is None:
        models = GROQ_MODELS

    all_results = {}

    for model_id, model_name, format_type in models:
        print(f"\n{'='*70}")
        print(f"Model: {model_name} ({model_id})")
        print(f"Expected format: {format_type}")
        print("=" * 70)

        try:
            config = LLMConfig.groq(api_key, model=model_id)
            client = LLMClient(config)

            model_results = {
                "model_name": model_name,
                "format_type": format_type,
                "prompts": [],
                "parser_scores": {
                    "regex": {"extracted": 0, "total": 0},
                    "incremental": {"extracted": 0, "total": 0},
                    "state_machine": {"extracted": 0, "total": 0},
                },
            }

            for i, prompt in enumerate(TEST_PROMPTS, 1):
                print(f"\n  [{i}/{len(TEST_PROMPTS)}] {prompt[:45]}...")

                # Get raw LLM output
                llm_result = get_raw_llm_output(client, prompt, format_type)

                if not llm_result["success"]:
                    print(f"      LLM Error: {llm_result.get('error', 'Unknown')[:40]}")
                    continue

                raw_output = llm_result["raw_output"]
                print(f"      Raw output: {raw_output[:60]}...")

                # Test each parser
                parser_results = test_parsers_on_raw_output(raw_output)

                prompt_result = {
                    "prompt": prompt,
                    "raw_output": raw_output,
                    "elapsed_ms": llm_result["elapsed_ms"],
                    "parser_results": parser_results,
                }
                model_results["prompts"].append(prompt_result)

                # Print parser comparison
                print(f"      Parser results:")
                for parser_name, result in parser_results.items():
                    status = "[OK]" if result["num_calls"] > 0 else "[--]"
                    model_results["parser_scores"][parser_name]["total"] += 1
                    if result["num_calls"] > 0:
                        model_results["parser_scores"][parser_name]["extracted"] += 1
                        calls = [tc["name"] for tc in result["tool_calls"]]
                        print(f"        {parser_name}: {status} {calls}")
                    else:
                        print(f"        {parser_name}: {status} (no calls found)")

            # Calculate success rates
            print(f"\n  Summary for {model_name}:")
            for parser_name, scores in model_results["parser_scores"].items():
                if scores["total"] > 0:
                    rate = scores["extracted"] / scores["total"] * 100
                    print(f"    {parser_name}: {scores['extracted']}/{scores['total']} ({rate:.0f}%)")

            all_results[model_id] = model_results

        except Exception as e:
            print(f"  Model failed: {e}")
            all_results[model_id] = {
                "model_name": model_name,
                "format_type": format_type,
                "error": str(e),
            }

    return all_results


def print_final_summary(results: dict):
    """Print final comparison table."""
    print("\n" + "=" * 90)
    print("PARSER COMPARISON SUMMARY - Raw LLM Output Parsing")
    print("=" * 90)

    print(f"\n{'Model':<25} {'Format':<12} {'Regex':<12} {'Incremental':<12} {'StateMachine':<12}")
    print("-" * 90)

    for model_id, data in results.items():
        if "error" in data:
            print(f"{data['model_name']:<25} {'ERROR':<12} {'-':<12} {'-':<12} {'-':<12}")
            continue

        name = data["model_name"][:24]
        fmt = data["format_type"][:11]

        scores = data.get("parser_scores", {})

        def format_score(parser_name):
            s = scores.get(parser_name, {})
            if s.get("total", 0) > 0:
                return f"{s['extracted']}/{s['total']} ({s['extracted']/s['total']*100:.0f}%)"
            return "-"

        regex = format_score("regex")
        incr = format_score("incremental")
        state = format_score("state_machine")

        print(f"{name:<25} {fmt:<12} {regex:<12} {incr:<12} {state:<12}")

    print("=" * 90)


def save_results(results: dict, filename: str = None):
    """Save results to JSON for dashboard."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_results_{timestamp}.json"

    # Clean results for JSON
    clean_results = {}
    for model_id, data in results.items():
        clean_results[model_id] = {
            "model_name": data.get("model_name"),
            "format_type": data.get("format_type"),
            "parser_scores": data.get("parser_scores", {}),
            "error": data.get("error"),
        }

    with open(filename, "w") as f:
        json.dump(clean_results, f, indent=2)

    print(f"\nResults saved to: {filename}")
    return filename


def main():
    print("=" * 70)
    print("Raw LLM Output Parser Comparison")
    print("=" * 70)
    print("\nThis test compares how well each parser extracts tool calls")
    print("from REAL raw LLM output (not pre-parsed API responses).")

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        print("\n[!] GROQ_API_KEY not set!")
        print("\nOption 1: Create a .env file:")
        print('  Copy .env.example to .env and add your key')
        print("\nOption 2: Set environment variable:")
        print('  set GROQ_API_KEY=your-key-here')
        print("\nGet a free key at: https://console.groq.com")
        return

    print(f"\n[*] Testing {len(GROQ_MODELS)} models:")
    for model_id, name, fmt in GROQ_MODELS:
        print(f"    - {name} ({fmt} format)")

    # Run comparison
    results = run_model_comparison(api_key)

    # Print summary
    print_final_summary(results)

    # Save results
    save_results(results)

    print("\nThis shows which parser works best for each model's output format!")


if __name__ == "__main__":
    main()
