#!/usr/bin/env python3
"""
Minimal example for web search with GPT-OSS.

Usage:
    python minimal_web_search.py "your prompt here" [--no-stream] [--temp TEMP] [--top-p TOP_P] [--reasoning-level LEVEL]

Example:
    python minimal_web_search.py "What's the weather in San Francisco today?"
    python minimal_web_search.py "Search for Python tutorials" --temp 0.8 --top-p 0.9 --reasoning-level medium
"""

import sys
import os
import argparse
import json

# Add parent directory to path to import client and web_search_tool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client import generate_with_tools_stream, generate_with_tools
from web_search_tool import WEB_SEARCH_TOOL, web_search_executor


def main():
    parser = argparse.ArgumentParser(
        description="Minimal web search example with GPT-OSS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="The prompt/question to ask"
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming mode"
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
    
    args = parser.parse_args()
    
    # Track streaming output
    thinking_tokens = []
    output_tokens = []
    tool_calls_made = []
    
    def on_reasoning(delta):
        """Callback for reasoning/thinking tokens."""
        thinking_tokens.append(delta)
        print(delta, end="", flush=True)
    
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
        print(f"\n[TOOL CALL] {tool_name}({arguments})\n", flush=True)
    
    
    # Generate response with tools
    print("=" * 80)
    print(f"Prompt: {args.prompt}")
    print(f"Temperature: {args.temp}, Top-p: {args.top_p}, Reasoning: {args.reasoning_level}")
    print(f"Streaming: {not args.no_stream}")
    print("=" * 80)
    print("\n[RESPONSE]\n")
    
    if args.no_stream:
        # Non-streaming mode
        result = generate_with_tools(
            prompt=args.prompt,
            tools=[WEB_SEARCH_TOOL],
            tool_executor=web_search_executor,
            tool_choice="auto",
            temperature=args.temp,
            top_p=args.top_p,
            reasoning_level=args.reasoning_level,
        )
        # Extract content from result
        output_content = result.get("content", "")
        thinking_content = result.get("reasoning_content", "")
        all_tool_calls = result.get("tool_calls", [])
        print(output_content)
    else:
        # Streaming mode
        result = generate_with_tools_stream(
            prompt=args.prompt,
            tools=[WEB_SEARCH_TOOL],
            tool_executor=web_search_executor,
            tool_choice="auto",
            temperature=args.temp,
            top_p=args.top_p,
            reasoning_level=args.reasoning_level,
            on_reasoning=on_reasoning,
            on_content=on_content,
            on_tool_call=on_tool_call,
        )
        # Extract content from accumulated buffers
        output_content = "".join(output_tokens) if output_tokens else result.get("content", "")
        thinking_content = "".join(thinking_tokens) if thinking_tokens else result.get("reasoning_content", "")
        all_tool_calls = result.get("tool_calls", [])
    
    # Display results
    print("\n\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    # Thinking tokens
    if thinking_content:
        print("\n[THINKING TOKENS]")
        print("-" * 80)
        print(thinking_content.strip())
    else:
        print("\n[THINKING TOKENS] (none)")
    
    # Output tokens
    if output_content:
        print("\n[OUTPUT TOKENS]")
        print("-" * 80)
        print(output_content.strip())
    else:
        print("\n[OUTPUT TOKENS] (none)")
    
    # Tool calls - prefer tool_calls_made from callbacks, fallback to result
    display_calls = tool_calls_made if tool_calls_made else all_tool_calls
    if display_calls:
        print("\n[TOOL CALLS MADE]")
        print("-" * 80)
        for i, call in enumerate(display_calls, 1):
            if isinstance(call, dict):
                if "name" in call:
                    args_str = json.dumps(call.get('arguments', {}), indent=2)
                    print(f"{i}. {call['name']}({args_str})")
                else:
                    print(f"{i}. {json.dumps(call, indent=2)}")
            else:
                print(f"{i}. {call}")
    else:
        print("\n[TOOL CALLS MADE] (none)")
    
    # Metadata
    print("\n[METADATA]")
    print("-" * 80)
    print(f"Turns: {result.get('turns', 'N/A')}")
    print(f"Finish reason: {result.get('finish_reason', 'N/A')}")
    if result.get('usage'):
        usage = result['usage']
        print(f"Tokens: {usage.get('total_tokens', 'N/A')} "
              f"(prompt: {usage.get('prompt_tokens', 'N/A')}, "
              f"completion: {usage.get('completion_tokens', 'N/A')})")
    
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        parser = argparse.ArgumentParser()
        parser.print_help()
        sys.exit(1)
    main()

