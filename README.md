# Chatbot Memory with Web Search

A Python-based chatbot system that integrates GPT-OSS reasoning model with semantic memory and web search capabilities. This project demonstrates how to build an AI assistant that can remember information and search the web using a reasoning-capable language model.

## Features

- **GPT-OSS Integration**: Client for GPT-OSS reasoning model with support for:
  - Configurable reasoning levels (low, medium, high)
  - Temperature and top-p sampling controls
  - Streaming and non-streaming modes
  - Tool calling support

- **Semantic Memory**: Persistent memory storage using Qwen/Qwen3-Embedding-8B:
  - Embeddings generated via HTTP API (zero query-time latency)
  - FAISS-based vector storage for efficient similarity search
  - Pre-loaded models on server startup

- **Web Search Tool**: Robust web search functionality using DuckDuckGo:
  - No API key required
  - Rate limiting and caching support
  - Error handling and retry logic
  - HTML parsing fallback mechanisms

- **Unified Server**: Single server that runs both GPT-OSS and embedding API:
  - GPT-OSS on CUDA:6 (port 8000)
  - Embedding API on CUDA:7 (port 8001)
  - Models pre-loaded at startup for minimal latency

## Project Structure

```
chatbot_memory/
├── client.py              # GPT-OSS client with tool support
├── server.py              # Unified server launcher (GPT-OSS + Embedding API)
├── embedding_server.py     # Embedding API server (Qwen/Qwen3-Embedding-8B)
├── tools/                  # Tool implementations
│   ├── __init__.py
│   ├── memory_tool.py     # Semantic memory implementation
│   ├── web_search_tool.py # Web search implementation
│   └── unified_executor.py # Unified tool executor
├── examples/               # Example scripts
│   ├── minimal_example.py  # Minimal example with all tools (RECOMMENDED)
│   ├── memory_example.py   # Memory-only example
│   └── web_search_example.py # Web search-only example
└── README.md             # This file
```

## Requirements

- Python 3.8+
- Conda environment: `genai_dev`
- vLLM (for running GPT-OSS server)
- Flask (for embedding API server)
- sentence-transformers (for embedding models)
- requests, faiss-cpu, beautifulsoup4
- duckduckgo-search (optional, falls back to HTML scraping)

## Installation

1. Activate the conda environment:
```bash
conda activate genai_dev
```

2. Install required dependencies:
```bash
pip install flask sentence-transformers requests faiss-cpu beautifulsoup4 duckduckgo-search
```

3. Ensure you have vLLM installed and GPT-OSS model available for the server.

## Usage

### Running the Server

Start the unified server (GPT-OSS + Embedding API):
```bash
conda activate genai_dev
python server.py
```

The server will start:
- GPT-OSS-20B on `http://localhost:8000` (CUDA:6)
- Embedding API on `http://localhost:8001` (CUDA:7) with Qwen/Qwen3-Embedding-8B

### Using the Minimal Example (Recommended)

The minimal example enables both memory and web search tools automatically:

```bash
# Basic usage (streaming by default)
python examples/minimal_example.py "Remember that I like strawberries"

# With custom parameters
python examples/minimal_example.py "What do I like?" --temp 0.8 --top-p 0.9 --reasoning-level medium

# Web search example
python examples/minimal_example.py "What's the weather in San Francisco?" --reasoning-level medium

# Disable streaming
python examples/minimal_example.py "your prompt" --no-stream

# Custom memory storage path
python examples/minimal_example.py "Remember X" --storage-path custom_memory
```

### Programmatic Usage

```python
from client import generate_with_tools_stream
from tools import ALL_TOOLS, unified_tool_executor

result = generate_with_tools_stream(
    prompt="Remember that I like Python and search for Python tutorials",
    tools=ALL_TOOLS,  # Includes memory_store, memory_retrieve, web_search
    tool_executor=lambda name, args: unified_tool_executor(name, args, storage_path="memory_store"),
    tool_choice="auto",
    temperature=1.0,
    top_p=0.95,
    reasoning_level="medium",
    on_content=lambda delta: print(delta, end=""),
    on_tool_call=lambda name, args: print(f"\n[TOOL] {name}\n")
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
- Port: `8000` (GPT-OSS), `8001` (Embedding API)
- Max context length: `16384` tokens
- GPU memory utilization: `95%`
- Quantization: `mxfp4` (GPT-OSS), `bfloat16` (Qwen embeddings)
- Embedding model: `Qwen/Qwen3-Embedding-8B`

### Environment Variables
- `EMBEDDING_API_URL`: Override embedding API URL (default: `http://localhost:8001`)

## Architecture

The system is optimized for minimal latency:

1. **Server Startup**: Both GPT-OSS and embedding models are loaded at startup
2. **Query Time**: No model loading - embeddings generated via HTTP API
3. **Tool Execution**: Unified executor routes to appropriate tool handler
4. **Memory Storage**: FAISS index stored locally, embeddings generated on server

## License

This project is provided as-is for educational and research purposes.
