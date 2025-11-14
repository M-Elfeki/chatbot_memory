#!/usr/bin/env python3
"""
Run GPT-OSS server on CUDA:0.
"""
import os
import subprocess
import sys


def run_gptoss_server():
    """Launch GPT-OSS reasoning model on GPU 0."""
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
    print("Starting GPT-OSS server:")
    print("  - GPT-OSS-20B (CUDA:2) → http://localhost:8000")
    print("  - Reasoning: ENABLED (thinking tokens separated)")
    print("  - Tool calling: ENABLED (openai parser)")
    print("  - Streaming: ENABLED (SSE format)")
    print("="*60 + "\n")

    try:
        run_gptoss_server()
    except KeyboardInterrupt:
        print("\n🛑 Stopping server...")
        sys.exit(0)
