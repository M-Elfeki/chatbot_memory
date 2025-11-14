#!/usr/bin/env python3
"""
Unified tool executor that handles both memory and web search tools.
"""
from typing import Dict, Any
from .memory_tool import memory_executor
from .web_search_tool import web_search_executor


def unified_tool_executor(
    tool_name: str,
    arguments: Dict[str, Any],
    storage_path: str = "memory_store"
) -> str:
    """
    Unified tool executor that routes to the appropriate tool handler.
    
    Args:
        tool_name: Name of the tool ('memory_store', 'memory_retrieve', or 'web_search')
        arguments: Tool-specific arguments
        storage_path: Storage path for memory operations (default: "memory_store")
    
    Returns:
        JSON string with result
    """
    # Add storage_path to memory tool arguments if not present
    if tool_name in ["memory_store", "memory_retrieve"]:
        if "storage_path" not in arguments:
            arguments["storage_path"] = storage_path
        return memory_executor(tool_name, arguments)
    elif tool_name == "web_search":
        return web_search_executor(tool_name, arguments)
    else:
        import json
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

