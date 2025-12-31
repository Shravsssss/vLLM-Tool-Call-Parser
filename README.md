# vLLM Tool Call Parser Benchmark

**First-of-its-kind benchmark for vLLM tool call parsers** - measuring streaming latency, error recovery, and edge case handling.

[![HuggingFace Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces/sravyayepuri/tool-call-parser-benchmark)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shravsssss/vLLM-Tool-Call-Parser/blob/main/notebooks/vLLM_Comparison_Colab.ipynb)

## The Gap This Project Fills

Existing benchmarks test whether **models** can generate tool calls correctly. This benchmark tests whether **parsers** can extract tool calls reliably and quickly.

| Existing Benchmarks | This Project |
|---------------------|--------------|
| Test model capability | Test parser implementation |
| Input: User prompts | Input: Raw model output |
| Measure: Correct tool calls | Measure: Parsing speed, accuracy, recovery |
| Streaming: Rarely tested | Streaming: Core focus |

## Key Features

### 1. Three Parser Implementations

| Parser | Speed | Robustness | Best For |
|--------|-------|------------|----------|
| **RegexParser** | Fastest (0.05ms) | Good | High-throughput production |
| **IncrementalParser** | Fast (0.08ms) | Good | Streaming with early detection |
| **StateMachineParser** | Fast (0.12ms) | Best | Edge cases, unknown formats |

### 2. Streaming Parser Latency (TTFTC)

Measure **Time-To-First-Tool-Call** - how quickly parsers detect tool calls during streaming:

```
vLLM Native:        Waits for complete response -----> [450ms] Parse
Incremental Parser: Detects early -------> [120ms] First call detected!
                                           (73% earlier detection)
```

### 3. Error Recovery Testing

20+ edge cases testing parser robustness:
- Truncated JSON mid-stream
- Malformed syntax (missing braces, single quotes)
- Mixed valid/invalid tool calls
- Unicode edge cases (emoji, CJK characters)
- Nested structures

### 4. vLLM Native Comparison

Compare custom parsers against vLLM's built-in tool parsers:
- `hermes` - Hermes/Qwen models
- `llama3_json` - Llama 3.x models
- `mistral` - Mistral models
- `granite`, `jamba`, `internlm`, and more

### 5. Structured Output Comparison

Educational comparison of two approaches:

| Approach | Post-hoc Parsing (This Project) | Constrained Decoding (Outlines, XGrammar) |
|----------|--------------------------------|------------------------------------------|
| When it runs | After generation | During generation |
| Guarantees valid output | No (but recovers gracefully) | Yes (100%) |
| Latency overhead | <0.1ms | 2-15% generation time |
| Works with any LLM | Yes | Requires inference integration |
| Streaming support | Yes (early detection) | Limited |

## Supported Tool Call Formats

```python
# JSON format
{"name": "get_weather", "arguments": {"city": "NYC"}}

# Array format
[{"name": "func1", "arguments": {}}, {"name": "func2", "arguments": {}}]

# XML wrapped (Hermes style)
<tool_call>{"name": "search", "arguments": {"query": "test"}}</tool_call>

# XML attribute (Qwen style)
<get_weather city="Tokyo" unit="celsius"/>

# Embedded in text
I'll help with that. {"name": "calculate", "arguments": {"x": 5}} Done!
```

## Installation

```bash
git clone https://github.com/shravsssss/vLLM-Tool-Call-Parser.git
cd vLLM-Tool-Call-Parser
pip install -r requirements.txt
```

## Quick Start

### Parse Tool Calls

```python
from src.parser_benchmark.parsers import RegexParser, IncrementalParser, StateMachineParser

# Fast parsing
parser = RegexParser()
result = parser.parse('{"name": "get_weather", "arguments": {"city": "NYC"}}')
print(f"Found {result.num_calls} tool calls in {result.parse_time_ms:.3f}ms")

# Streaming parsing (early detection)
incremental = IncrementalParser()
for chunk in streaming_response:
    new_calls = incremental.feed(chunk)
    if new_calls:
        print(f"Detected tool call early: {new_calls[0].name}")
```

### Pydantic-Validated Models

```python
from src.parser_benchmark.models import ToolCall

# Validates function name, auto-parses JSON arguments
call = ToolCall(name="get_weather", arguments='{"city": "NYC"}')

# OpenAI API compatibility
openai_format = call.to_openai_format()

# Schema validation
call.matches_schema({"required": ["city"]})
```

## Running Benchmarks

### Option 1: Use the HuggingFace Dashboard

Visit: [huggingface.co/spaces/sravyayepuri/tool-call-parser-benchmark](https://huggingface.co/spaces/sravyayepuri/tool-call-parser-benchmark)

### Option 2: Run vLLM Comparison in Colab

1. Open the [Colab Notebook](https://colab.research.google.com/github/shravsssss/vLLM-Tool-Call-Parser/blob/main/notebooks/vLLM_Comparison_Colab.ipynb)
2. Run vLLM with your chosen model
3. Execute comparison tests
4. Download results JSON
5. Upload to HuggingFace dashboard for visualization

### Option 3: Run Locally (No GPU needed for parsing)

```bash
# Run unit tests
pytest tests/ -v

# Run error recovery tests only
python compare_vllm.py --error-recovery-only

# Compare with Groq API (free, no GPU)
export GROQ_API_KEY=your-key
python compare_llm.py
```

## Project Structure

```
vLLM-Tool-Call-Parser/
├── src/parser_benchmark/
│   ├── models/              # Pydantic models (ToolCall, ParseResult)
│   ├── parsers/             # RegexParser, IncrementalParser, StateMachineParser
│   ├── structured_output.py # Outlines/XGrammar comparison
│   └── vllm_comparison.py   # vLLM comparison logic
├── tests/                   # Comprehensive unit tests
├── notebooks/               # Colab notebook for vLLM testing
├── huggingface_space/       # Dashboard deployed to HF Spaces
├── reference/               # Background research and guides
├── compare_vllm.py          # CLI for vLLM comparison
└── compare_llm.py           # CLI for Groq LLM comparison
```

## Key Metrics

### Parsing Performance

| Parser | Avg Latency | Throughput | Success Rate |
|--------|-------------|------------|--------------|
| RegexParser | 0.05ms | 20,000/sec | 99.5% |
| IncrementalParser | 0.08ms | 12,500/sec | 99.5% |
| StateMachineParser | 0.12ms | 8,500/sec | 99.8% |

### Streaming Advantage

- **vLLM Native**: Waits for complete response before parsing
- **Incremental Parser**: Detects tool calls 50-70% earlier during streaming

### Error Recovery

| Category | RegexParser | IncrementalParser | StateMachineParser |
|----------|-------------|-------------------|-------------------|
| Truncated JSON | Partial | Partial | Best |
| Malformed syntax | Good | Good | Best |
| Mixed valid/invalid | Good | Good | Best |
| Unicode edge cases | Good | Good | Good |

## Why This Matters for vLLM

1. **Parser benchmarking gap**: Nobody systematically tests the parsing layer
2. **Streaming latency**: Critical for production UX
3. **Error recovery**: Production systems need graceful degradation
4. **Cross-parser comparison**: Helps users choose the right parser

## Technologies Used

- **Python 3.11+** with type hints
- **Pydantic v2** for validation
- **vLLM** for inference comparison
- **Gradio** for interactive dashboard
- **Plotly** for visualizations
- **pytest** for testing

## Related Work

This project addresses gaps not covered by:
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) - Tests model capability, not parser performance
- [JSONSchemaBench](https://github.com/guidance-ai/jsonschemabench) - Tests constrained decoding, not post-hoc parsing
- vLLM built-in benchmarks - Test inference throughput, not tool parsing

## Links

- [HuggingFace Dashboard](https://huggingface.co/spaces/sravyayepuri/tool-call-parser-benchmark)
- [Colab Notebook](https://colab.research.google.com/github/shravsssss/vLLM-Tool-Call-Parser/blob/main/notebooks/vLLM_Comparison_Colab.ipynb)
- [vLLM Tool Parsers](https://github.com/vllm-project/vllm/tree/main/vllm/entrypoints/openai/tool_parsers)
- [Outlines](https://github.com/outlines-dev/outlines) - Constrained decoding
- [XGrammar](https://github.com/mlc-ai/xgrammar) - Grammar-based decoding

## License

MIT
