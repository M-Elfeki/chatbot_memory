#!/usr/bin/env python3
"""
Run GPT-OSS server with embedding API server.
GPT-OSS runs on CUDA:6, embedding models run on CUDA:7 via API server.
"""
import os
import subprocess
import sys
import threading
import time

# Add tools directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import after path setup
from embedding_server import run_embedding_server  # noqa: E402


def start_embedding_server_thread():
    """Start embedding API server on CUDA:7 in a separate thread."""
    print("\n" + "="*60)
    print("Starting Embedding API Server on CUDA:7...")
    print("="*60)

    try:
        # Start embedding server (loads Qwen model with bfloat16)
        run_embedding_server(
            host="0.0.0.0",
            port=8001,
            qwen_model_name="Qwen/Qwen3-Embedding-8B",
            qwen_device="cuda:7",
            use_bfloat16=True,
            debug=False
        )
    except Exception as e:
        print(f"⚠ Warning: Failed to start embedding server: {e}")
        print("="*60 + "\n")


def run_gptoss_server():
    """Launch GPT-OSS reasoning model on CUDA:6."""
    # Set CUDA_VISIBLE_DEVICES to only show GPU 6 for GPT-OSS
    # This ensures GPT-OSS only sees and uses GPU 6
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "openai/gpt-oss-20b",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--dtype", "auto",
        "--quantization", "mxfp4",
        "--gpu-memory-utilization", "0.95",
        "--max-model-len", "16384",  # Extended context length
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--reasoning-parser", "openai_gptoss",  # GPT-OSS parser (vLLM 0.11.0+)
        # Tool calling configuration (requires vLLM 0.10.2+):
        "--enable-auto-tool-choice",
        "--tool-call-parser", "openai",  # OpenAI parser for GPT-OSS
        # Performance optimizations
        "--max-num-seqs", "8",  # Conservative for 16K without fp8 KV cache
        "--max-num-batched-tokens", "8192",  # Limit batch size
        "--speculative-config",
        '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":5}',
    ]

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 GPT-OSS server stopped")
        sys.exit(0)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Starting Unified Server:")
    print("  - GPT-OSS-20B (CUDA:6) → http://localhost:8000")
    print("  - Embedding API Server (CUDA:7) → http://localhost:8001")
    print("    * Model: Qwen/Qwen3-Embedding-8B")
    print("    * Precision: bfloat16")
    print("  - Reasoning: ENABLED (thinking tokens separated)")
    print("  - Tool calling: ENABLED (openai parser)")
    print("  - Streaming: ENABLED (SSE format)")
    print("="*60 + "\n")

    # Start embedding API server in a separate thread
    # This loads Qwen model at startup and provides HTTP API
    embedding_thread = threading.Thread(
        target=start_embedding_server_thread, daemon=True
    )
    embedding_thread.start()

    # Give the embedding server a moment to start loading models
    time.sleep(3)

    try:
        run_gptoss_server()
    except KeyboardInterrupt:
        print("\n🛑 Stopping servers...")
        sys.exit(0)
