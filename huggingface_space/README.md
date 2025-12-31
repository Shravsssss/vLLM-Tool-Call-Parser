---
title: vLLM Tool Call Parser Benchmark
emoji: 🔧
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
short_description: Compare custom parsers vs vLLM native tool parsing
---

# vLLM Tool Call Parser Benchmark

Interactive dashboard comparing custom tool call parsers against vLLM's native parsing. See accuracy, latency, streaming advantages, and error recovery capabilities.

## Features

- **3 Parser Implementations**: Regex, Incremental (streaming), State Machine
- **vLLM Comparison**: Compare your parsers vs vLLM native tool parsing
- **Streaming Advantage**: See how incremental parsing detects calls earlier
- **Error Recovery Demo**: Test parser robustness on malformed inputs
- **Live Benchmarks**: Run performance comparisons in real-time

## New: vLLM Comparison Tab

Upload results from running `compare_vllm.py` in Colab to see:
- Accuracy comparison charts
- Latency comparison (your parsers are faster!)
- Streaming advantage visualization
- Error recovery test results

## Supported Tool Call Formats

```json
// Simple JSON
{"name": "get_weather", "arguments": {"city": "NYC"}}

// Array format
[{"name": "func1", "arguments": {}}, {"name": "func2", "arguments": {}}]

// XML wrapped (Hermes style)
<tool_call>{"name": "search", "arguments": {"q": "test"}}</tool_call>

// XML attribute (Qwen style)
<get_weather city="Tokyo" unit="celsius"/>
```

## Links

- [GitHub Repository](https://github.com/shravsssss/vLLM-Tool-Call-Parser)
- [Google Colab Notebook](https://colab.research.google.com/drive/1kQtpX4R1PvOm2yuhjnZLQuYmkin0Qi0D?usp=sharing)
