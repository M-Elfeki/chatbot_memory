# Chatbot Memory with Web Search

A Python-based chatbot system that integrates GPT-OSS reasoning model with web search capabilities. This project demonstrates how to build an AI assistant that can search the web and provide informed responses using a reasoning-capable language model.

## Features

- **GPT-OSS Integration**: Client for GPT-OSS reasoning model with support for:
  - Configurable reasoning levels (low, medium, high)
  - Temperature and top-p sampling controls
  - Streaming and non-streaming modes
  - Tool calling support

- **Web Search Tool**: Robust web search functionality using DuckDuckGo:
  - No API key required
  - Rate limiting and caching support
  - Error handling and retry logic
  - HTML parsing fallback mechanisms

- **Server Management**: Script to run GPT-OSS server with optimized GPU settings

## Project Structure

```
chatbot_memory/
├── client.py              # GPT-OSS client with tool support
├── server.py              # Server launcher for GPT-OSS
├── web_search_tool.py     # Web search implementation
├── minimal_web_search.py  # Example demonstrating web search usage
└── README.md             # This file
```

## Requirements

- Python 3.8+
- vLLM (for running GPT-OSS server)
- requests
- beautifulsoup4
- duckduckgo-search (optional, falls back to HTML scraping)

## Installation

1. Install required dependencies:
```bash
pip install requests beautifulsoup4 duckduckgo-search
```

2. Ensure you have vLLM installed and GPT-OSS model available for the server.

## Usage

### Running the Server

Start the GPT-OSS server:
```bash
python server.py
```

The server will run on `http://localhost:8000` with GPU acceleration.

### Using Web Search Example

Run the minimal web search example:
```bash
python minimal_web_search.py "What's the weather in San Francisco today?"
```

With custom parameters:
```bash
python minimal_web_search.py "Search for Python tutorials" --temp 0.8 --top-p 0.9 --reasoning-level medium
```

Disable streaming:
```bash
python minimal_web_search.py "your query" --no-stream
```

### Programmatic Usage

```python
from client import generate_with_tools_stream
from web_search_tool import WEB_SEARCH_TOOL, web_search_executor

result = generate_with_tools_stream(
    prompt="What are the latest developments in AI?",
    tools=[WEB_SEARCH_TOOL],
    tool_executor=web_search_executor,
    tool_choice="auto",
    temperature=1.0,
    top_p=0.95,
    reasoning_level="medium",
    on_reasoning=lambda delta: print(delta, end=""),
    on_content=lambda delta: print(delta, end=""),
    on_tool_call=lambda name, args: print(f"\n[TOOL] {name}({args})\n")
)
```

## Configuration

### Reasoning Levels
- `low`: Minimal reasoning tokens, faster responses
- `medium`: Balanced reasoning and output
- `high`: Maximum reasoning tokens, more thorough analysis

### Server Settings
The server is configured with:
- Model: `openai/gpt-oss-20b`
- Port: `8000`
- Max context length: `16384` tokens
- GPU memory utilization: `95%`
- Quantization: `mxfp4`

## License

This project is provided as-is for educational and research purposes.

