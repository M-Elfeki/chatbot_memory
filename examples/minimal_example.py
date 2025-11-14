#!/usr/bin/env python3
"""
Minimal example with GPT-OSS, memory, and web search tools.
Streams by default with optional parameters.

Usage:
    python examples/minimal_example.py "your prompt here" [--temp TEMP] [--top-p TOP_P] [--reasoning-level LEVEL] [--no-stream] [--storage-path PATH]

Example:
    python examples/minimal_example.py "Remember that I like strawberries"
    python examples/minimal_example.py "What do I like?" --temp 0.8 --top-p 0.9
    python examples/minimal_example.py "What's the weather in San Francisco?" --reasoning-level medium
"""

import sys
import os
import argparse

# Add parent directory to path to import client
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client import generate_with_tools_stream
from tools import ALL_TOOLS, unified_tool_executor


def main():
    parser = argparse.ArgumentParser(
        description="Minimal example with GPT-OSS, memory, and web search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="The prompt/question to ask"
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=1.0,
        help="Temperature for sampling (default: 1.0)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus sampling) parameter (default: 0.95)"
    )
    parser.add_argument(
        "--reasoning-level",
        type=str,
        choices=["low", "medium", "high"],
        default="low",
        help="Reasoning level: low, medium, or high (default: low)"
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming mode (streaming is default)"
    )
    parser.add_argument(
        "--storage-path",
        type=str,
        default="memory_store",
        help="Path to memory storage directory (default: memory_store)"
    )
    
    args = parser.parse_args()
    
    # Track streaming output
    output_tokens = []
    tool_calls_made = []
    
    def on_content(delta):
        """Callback for output/content tokens."""
        output_tokens.append(delta)
        print(delta, end="", flush=True)
    
    def on_tool_call(tool_name, arguments):
        """Callback when a tool is called."""
        tool_calls_made.append({
            "name": tool_name,
            "arguments": arguments
        })
        print(f"\n[TOOL] {tool_name}\n", flush=True)
    
    # Create tool executor with storage path
    def tool_executor(tool_name, arguments):
        return unified_tool_executor(tool_name, arguments, storage_path=args.storage_path)
    
    # Generate response with tools (streaming by default)
    print(f"Prompt: {args.prompt}")
    print(f"Temperature: {args.temp}, Top-p: {args.top_p}, Reasoning: {args.reasoning_level}")
    print(f"Streaming: {not args.no_stream}")
    print("-" * 80)
    print()
    
    if args.no_stream:
        # Non-streaming mode (import here to avoid if not needed)
        from client import generate_with_tools
        result = generate_with_tools(
            prompt=args.prompt,
            tools=ALL_TOOLS,
            tool_executor=tool_executor,
            tool_choice="auto",
            temperature=args.temp,
            top_p=args.top_p,
            reasoning_level=args.reasoning_level,
        )
        print(result.get("content", ""))
        tool_calls_made = result.get("tool_calls", [])
    else:
        # Streaming mode (default)
        result = generate_with_tools_stream(
            prompt=args.prompt,
            tools=ALL_TOOLS,
            tool_executor=tool_executor,
            tool_choice="auto",
            temperature=args.temp,
            top_p=args.top_p,
            reasoning_level=args.reasoning_level,
            on_content=on_content,
            on_tool_call=on_tool_call,
        )
    
    # Display tool calls summary
    if tool_calls_made:
        print(f"\n\n[Used {len(tool_calls_made)} tool(s)]")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        parser = argparse.ArgumentParser()
        parser.print_help()
        sys.exit(1)
    main()

