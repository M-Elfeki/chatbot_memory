#!/usr/bin/env python3
"""
Client for GPT-OSS reasoning model.
Supports top_p, temperature, reasoning levels, tools, and streaming.
"""
import requests
import json
import re
from typing import Optional, List, Dict, Any, Iterator, Callable


def _is_tool_call_json(content: str) -> bool:
    """
    Check if content looks like tool call JSON that shouldn't be output as text.
    """
    if not content or not content.strip():
        return False
    
    content_stripped = content.strip()
    # Check for JSON array/object with "name" and "parameters" keys
    if ('"name"' in content_stripped and 
        '"parameters"' in content_stripped and
        ('web_search' in content_stripped or 'tool' in content_stripped.lower())):
        try:
            # Try to parse as JSON
            parsed = json.loads(content_stripped)
            # If it's a list/array with tool call structure
            if isinstance(parsed, list) and len(parsed) > 0:
                first_item = parsed[0]
                if isinstance(first_item, dict) and "name" in first_item:
                    return True
            # If it's a dict with tool call structure
            if isinstance(parsed, dict) and "name" in parsed:
                return True
        except (json.JSONDecodeError, ValueError):
            pass
    
    return False


def generate(
    prompt: str,
    reasoning_level: str = "medium",  # 'low', 'medium', 'high'
    temperature: float = 0.5,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,  # 'auto', 'none', or specific tool name
    stream: bool = False,
) -> Dict[str, Any]:
    """
    Generate a response using GPT-OSS reasoning model.
    
    Args:
        prompt: The input prompt
        reasoning_level: Reasoning level ('low', 'medium', 'high')
        temperature: Sampling temperature (default: 0.5)
        top_p: Nucleus sampling parameter (default: 1.0)
        max_tokens: Maximum tokens to generate (default: 1024)
        tools: Optional list of tool definitions for function calling
        tool_choice: How to handle tool calls ('auto', 'none', or tool name)
        stream: Whether to stream the response (default: False)
    
    Returns:
        Dictionary containing the response and metadata
        If stream=True, returns an iterator of dictionaries with delta updates
    """
    url = "http://localhost:8000/v1/chat/completions"

    # Validate reasoning level
    if reasoning_level not in ["low", "medium", "high"]:
        raise ValueError(
            "reasoning_level must be 'low', 'medium', or 'high', "
            f"got '{reasoning_level}'"
        )

    # Note: GPT-OSS may handle system messages differently with tools
    # Keep system message simple when using tools to avoid conflicts
    if tools is not None:
        system_msg = (
            "You are a helpful assistant. "
            f"Reasoning: {reasoning_level}"
        )
    else:
        system_msg = (
            "You are an advanced reasoning model developed by OpenAI. "
            "Respond in concise and direct manner.\n"
            "Knowledge cutoff: 2024-06\n\n"
            f"Reasoning: {reasoning_level}"
        )

    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    # Add tools if provided
    if tools is not None:
        payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        else:
            payload["tool_choice"] = "auto"

    try:
        if stream:
            return _stream_response(url, payload)
        else:
            response = requests.post(url, json=payload, timeout=300)
            if not response.ok:
                # Try to get error details from response
                error_detail = f"Status: {response.status_code}, "
                try:
                    error_json = response.json()
                    error_detail += f"JSON: {error_json}"
                except Exception:
                    error_text = response.text[:1000]
                    error_detail += f"Text: {error_text if error_text else '(empty)'}"
                    error_detail += f", Headers: {dict(response.headers)}"
                raise RuntimeError(
                    f"Server error {response.status_code}: {error_detail}"
                )
            response.raise_for_status()
            result = response.json()

            message = result["choices"][0]["message"]
            
            # Extract content and tool calls if present
            # content can be None if model only generated reasoning tokens
            content = message.get("content") or ""
            # Thinking tokens
            reasoning_content = message.get("reasoning_content") or ""
            tool_calls = message.get("tool_calls", None)
            
            return {
                "content": content.strip() if content else "",
                # Thinking tokens
                "reasoning_content": reasoning_content.strip() if reasoning_content else "",
                "tool_calls": tool_calls,
                "finish_reason": result["choices"][0]["finish_reason"],
                "usage": result.get("usage", {}),
            }

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to get response from GPT-OSS: {e}")


def generate_with_tools_stream(
    prompt: str,
    tools: List[Dict[str, Any]],
    tool_executor: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
    reasoning_level: str = "medium",
    temperature: float = 0.5,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    max_turns: int = 10,
    tool_choice: Optional[str] = None,
    max_retries: int = 3,
    on_reasoning: Optional[Callable[[str], None]] = None,
    on_content: Optional[Callable[[str], None]] = None,
    on_tool_call: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Generate a response with tool calling support and streaming.
    
    Streams responses in real-time and executes tools when called.
    
    Args:
        prompt: The input prompt
        tools: List of tool definitions for function calling
        tool_executor: Function to execute tools. Signature: (tool_name, arguments) -> result
        reasoning_level: Reasoning level ('low', 'medium', 'high')
        temperature: Sampling temperature (default: 0.5)
        top_p: Nucleus sampling parameter (default: 1.0)
        max_tokens: Maximum tokens per turn (default: 1024)
        max_turns: Maximum number of conversation turns (default: 10)
        tool_choice: How to handle tool calls ('auto', 'required', 'none', or tool name)
        max_retries: Maximum number of retries for invalid content (default: 3)
        on_reasoning: Callback for streaming reasoning tokens
        on_content: Callback for streaming content tokens
        on_tool_call: Callback when tool is called: (tool_name, arguments)
    
    Returns:
        Dictionary containing the final response, all tool calls made, and metadata
    """
    url = "http://localhost:8000/v1/chat/completions"
    
    # Validate reasoning level
    if reasoning_level not in ["low", "medium", "high"]:
        raise ValueError(
            "reasoning_level must be 'low', 'medium', or 'high', "
            f"got '{reasoning_level}'"
        )
    
    # System message
    system_msg = (
        "You are a helpful assistant. "
        "CRITICAL: When you receive tool results, write summaries in your own words. "
        "Do NOT output raw tool results, formatted lists with URLs, or tool result formats. "
        f"Reasoning: {reasoning_level}"
    )
    
    # Conversation history
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt}
    ]
    
    # Track tool calls across turns
    all_tool_calls = []
    all_tool_results = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    guidance_added = False  # Track if we've added guidance message for reasoning-detected tool calls
    
    for turn in range(max_turns):
        # Prepare payload
        payload = {
            "model": "openai/gpt-oss-20b",
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice if tool_choice is not None else "auto",
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": True,  # Enable streaming
        }
        
        try:
            # Make streaming request
            response = requests.post(url, json=payload, stream=True, timeout=300)
            response.raise_for_status()
            
            # Accumulate streaming response
            accumulated_content = ""
            accumulated_reasoning = ""
            tool_calls = []
            tool_calls_seen = set()  # Track which tool calls we've already notified
            finish_reason = None
            
            for line in response.iter_lines():
                if not line:
                    continue
                
                # Parse SSE format
                if line.startswith(b"data: "):
                    line = line[6:]
                
                if line == b"[DONE]":
                    break
                
                try:
                    chunk = json.loads(line)
                    if "choices" not in chunk or len(chunk["choices"]) == 0:
                        continue
                    
                    choice = chunk["choices"][0]
                    delta = choice.get("delta", {})
                    
                    # Extract reasoning delta
                    reasoning_delta = delta.get("reasoning_content", "")
                    if reasoning_delta:
                        accumulated_reasoning += reasoning_delta
                        if on_reasoning:
                            on_reasoning(reasoning_delta)
                        
                        # Check if reasoning contains tool call JSON (model may output it in reasoning)
                        # Look for JSON patterns like {"query": "...", "max_results": ...}
                        # or {"name": "web_search", "arguments": {...}}
                        if not tool_calls:  # Only check if we haven't found tool calls yet
                            reasoning_so_far = accumulated_reasoning
                            # Try to extract tool call JSON from reasoning
                            # Look for common patterns: {"query": or {"name": or web_search({
                            if ('"query"' in reasoning_so_far or '"name"' in reasoning_so_far or 
                                'web_search' in reasoning_so_far.lower()):
                                # Try to find and parse JSON object in reasoning
                                try:
                                    # Look for JSON object boundaries
                                    start_idx = reasoning_so_far.rfind('{')
                                    if start_idx >= 0:
                                        # Try to find matching closing brace
                                        brace_count = 0
                                        end_idx = start_idx
                                        for i in range(start_idx, len(reasoning_so_far)):
                                            if reasoning_so_far[i] == '{':
                                                brace_count += 1
                                            elif reasoning_so_far[i] == '}':
                                                brace_count -= 1
                                                if brace_count == 0:
                                                    end_idx = i + 1
                                                    break
                                        
                                        if brace_count == 0 and end_idx > start_idx:
                                            json_str = reasoning_so_far[start_idx:end_idx]
                                            parsed_json = json.loads(json_str)
                                            
                                            # Check if it looks like tool arguments
                                            if isinstance(parsed_json, dict):
                                                # Validate tool name against available tools
                                                valid_tool_names = {tool.get("function", {}).get("name") for tool in tools} if tools else set()
                                                
                                                # If it has "query", it's likely web_search arguments
                                                if "query" in parsed_json:
                                                    # Validate web_search is available
                                                    if "web_search" in valid_tool_names or not valid_tool_names:
                                                        # Create a tool call structure
                                                        tool_calls.append({
                                                            "id": f"call_{len(tool_calls)}",
                                                            "type": "function",
                                                            "function": {
                                                                "name": "web_search",
                                                                "arguments": json_str
                                                            }
                                                        })
                                                        # Notify callback
                                                        if on_tool_call:
                                                            on_tool_call("web_search", parsed_json)
                                                        tool_calls_seen.add(0)
                                                # If it has "name" and "arguments", it's a tool call structure
                                                elif "name" in parsed_json and "arguments" in parsed_json:
                                                    tool_name = parsed_json["name"]
                                                    # Only accept if it's a valid tool name
                                                    if tool_name in valid_tool_names or not valid_tool_names:
                                                        tool_calls.append({
                                                            "id": f"call_{len(tool_calls)}",
                                                            "type": "function",
                                                            "function": {
                                                                "name": tool_name,
                                                                "arguments": json.dumps(parsed_json.get("arguments", {}))
                                                            }
                                                        })
                                                        if on_tool_call:
                                                            on_tool_call(tool_name, parsed_json.get("arguments", {}))
                                                        tool_calls_seen.add(0)
                                except (json.JSONDecodeError, ValueError, KeyError):
                                    # Not valid JSON yet or not a tool call, continue
                                    pass
                    
                    # Extract content delta
                    content_delta = delta.get("content", "")
                    if content_delta:
                        accumulated_content += content_delta
                        if on_content:
                            on_content(content_delta)
                    
                    # Check for tool calls in delta (they appear incrementally in streaming)
                    # Tool calls come as deltas with index, id, and function.name/arguments
                    if "tool_calls" in delta:
                        delta_tool_calls = delta["tool_calls"]
                        for tc_delta in delta_tool_calls:
                            tc_index = tc_delta.get("index", 0)
                            # Ensure tool_calls list is large enough
                            while len(tool_calls) <= tc_index:
                                tool_calls.append({})
                            
                            # Merge delta into existing tool call
                            if not tool_calls[tc_index]:
                                # Initialize new tool call
                                tool_calls[tc_index] = {
                                    "id": tc_delta.get("id"),
                                    "type": tc_delta.get("type", "function"),
                                    "function": {}
                                }
                            
                            # Update function name/arguments incrementally
                            if "function" in tc_delta:
                                func_delta = tc_delta["function"]
                                if "name" in func_delta:
                                    tool_calls[tc_index]["function"]["name"] = func_delta["name"]
                                if "arguments" in func_delta:
                                    existing_args = tool_calls[tc_index]["function"].get("arguments", "")
                                    tool_calls[tc_index]["function"]["arguments"] = existing_args + func_delta["arguments"]
                            
                            # Try to notify if tool call looks complete
                            if tc_index not in tool_calls_seen:
                                tc = tool_calls[tc_index]
                                func = tc.get("function", {})
                                tool_name = func.get("name")
                                arguments_str = func.get("arguments", "")
                                
                                # Only notify if we have name and arguments that parse as JSON
                                if tool_name and arguments_str:
                                    try:
                                        arguments = json.loads(arguments_str)
                                        # Valid JSON - tool call is complete
                                        if on_tool_call:
                                            on_tool_call(tool_name, arguments)
                                        tool_calls_seen.add(tc_index)
                                    except json.JSONDecodeError:
                                        # Arguments still incomplete, wait for more deltas
                                        pass
                    
                    # Check finish reason
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    
                    # Update usage
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                        total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                        total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                        total_usage["total_tokens"] += usage.get("total_tokens", 0)
                
                except json.JSONDecodeError:
                    continue
            
            # Get final tool calls from accumulated message
            # Need to make a non-streaming request to get complete tool_calls
            if finish_reason == "tool_calls":
                # Get complete message with tool calls
                non_stream_payload = payload.copy()
                non_stream_payload["stream"] = False
                non_stream_response = requests.post(url, json=non_stream_payload, timeout=300)
                non_stream_response.raise_for_status()
                non_stream_result = non_stream_response.json()
                message = non_stream_result["choices"][0]["message"]
                final_tool_calls = message.get("tool_calls", [])
                
                # Notify about any tool calls we haven't seen yet
                for i, tc in enumerate(final_tool_calls):
                    if i not in tool_calls_seen:
                        func = tc.get("function", {})
                        tool_name = func.get("name")
                        arguments_str = func.get("arguments", "{}")
                        try:
                            if isinstance(arguments_str, str):
                                arguments = json.loads(arguments_str)
                            else:
                                arguments = arguments_str
                            if tool_name and on_tool_call:
                                on_tool_call(tool_name, arguments)
                            tool_calls_seen.add(i)
                        except json.JSONDecodeError:
                            pass
                
                tool_calls = final_tool_calls
            
            # Check reasoning content for tool call patterns if no tool calls found yet
            # This handles cases where model outputs tool calls in reasoning but finish_reason is "stop"
            # Do this BEFORE adding assistant message so tool calls are properly associated
            if not tool_calls and finish_reason == "stop" and accumulated_reasoning:
                # Try to extract tool calls from reasoning content
                reasoning_text = accumulated_reasoning.lower()
                # Look for memory_store or memory_retrieve patterns
                if "memory_store" in reasoning_text or "memory_retrieve" in reasoning_text:
                    # Helper function to extract JSON object after a keyword
                    def extract_json_after_keyword(text, keyword):
                        """Extract JSON object that appears after a keyword."""
                        # Find all occurrences of the keyword
                        keyword_lower = keyword.lower()
                        text_lower = text.lower()
                        idx = text_lower.find(keyword_lower)
                        if idx == -1:
                            return None
                        
                        # Look for JSON object starting after the keyword
                        # Skip the keyword and any whitespace/punctuation
                        search_start = idx + len(keyword)
                        # Skip whitespace, dots, parentheses, colons
                        while search_start < len(text) and text[search_start] in ' \t\n.:()':
                            search_start += 1
                        
                        # Find the first opening brace
                        brace_start = text.find('{', search_start)
                        if brace_start == -1:
                            return None
                        
                        # Extract JSON by matching braces
                        brace_count = 0
                        brace_end = brace_start
                        for i in range(brace_start, len(text)):
                            if text[i] == '{':
                                brace_count += 1
                            elif text[i] == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    brace_end = i + 1
                                    break
                        
                        if brace_count == 0:
                            json_str = text[brace_start:brace_end]
                            try:
                                # Validate it's valid JSON
                                json.loads(json_str)
                                return json_str
                            except (json.JSONDecodeError, ValueError):
                                return None
                        return None
                    
                    # Try to extract memory_store call
                    if "memory_store" in reasoning_text:
                        json_str = extract_json_after_keyword(accumulated_reasoning, "memory_store")
                        if json_str:
                            try:
                                args = json.loads(json_str)
                                if "text" in args:  # Validate it's a memory_store call
                                    tool_calls.append({
                                        "id": f"call_reasoning_{len(tool_calls)}",
                                        "type": "function",
                                        "function": {
                                            "name": "memory_store",
                                            "arguments": json_str
                                        }
                                    })
                                    # Don't call on_tool_call here - it will be called during execution
                            except (json.JSONDecodeError, ValueError):
                                pass
                    
                    # Try to extract memory_retrieve call
                    if "memory_retrieve" in reasoning_text and not tool_calls:
                        json_str = extract_json_after_keyword(accumulated_reasoning, "memory_retrieve")
                        if json_str:
                            try:
                                args = json.loads(json_str)
                                if "query" in args:  # Validate it's a memory_retrieve call
                                    tool_calls.append({
                                        "id": f"call_reasoning_{len(tool_calls)}",
                                        "type": "function",
                                        "function": {
                                            "name": "memory_retrieve",
                                            "arguments": json_str
                                        }
                                    })
                                    # Don't call on_tool_call here - it will be called during execution
                            except (json.JSONDecodeError, ValueError):
                                pass
                    
                    # If we found tool calls in reasoning, treat as tool_calls finish reason
                    if tool_calls:
                        finish_reason = "tool_calls"
            
            # If we detected tool calls in reasoning but finish_reason is not "tool_calls",
            # treat it as if we need to execute tools (model output tool call in reasoning)
            if tool_calls and finish_reason != "tool_calls":
                # Model output tool call in reasoning tokens, treat as tool_calls finish reason
                finish_reason = "tool_calls"
            
            # Add assistant message to history (after detecting tool calls from reasoning)
            messages.append({
                "role": "assistant",
                "content": accumulated_content,
                "tool_calls": tool_calls if tool_calls else None
            })
            
            if not tool_calls or finish_reason != "tool_calls":
                # Final response received
                content = accumulated_content.strip()
                reasoning_content = accumulated_reasoning.strip()
                
                # Check if content is invalid (raw tool results, etc.) even if it exists
                is_invalid_content = False
                if content:
                    # Check for raw tool results format
                    if (content.startswith("Found ") and "search results for" in content) or \
                       content.count("\n   URL:") >= 2 or \
                       (content.count("URL:") >= 2 and content.count("\n   ") >= 2):
                        is_invalid_content = True
                
                # If we have no content or invalid content, and we have tool results, retry
                if (not content or is_invalid_content) and all_tool_results and turn < max_turns:
                    # Check if we have successful tool results
                    has_successful_results = False
                    for tr in all_tool_results:
                        result_str = tr.get('result', '')
                        if result_str and not result_str.startswith('{"error"'):
                            if len(result_str.strip()) > 10:
                                has_successful_results = True
                                break
                    
                    if has_successful_results:
                        # Remove invalid/empty assistant message
                        if messages and messages[-1].get("role") == "assistant":
                            messages.pop()
                        
                        # Count retries to avoid loops
                        retry_count = sum(1 for msg in messages[-3:] if 
                                         msg.get("role") == "user" and 
                                         ("write a summary" in msg.get("content", "").lower() or
                                          "do not output" in msg.get("content", "").lower()))
                        
                        if retry_count < max_retries:
                            # Add retry message
                            messages.append({
                                "role": "user",
                                "content": "Write a summary of the tool results in your own words. Do NOT output raw tool results or formatted lists."
                            })
                            
                            # Update system message
                            original_system_msg = messages[0]["content"]
                            if "CRITICAL:" not in original_system_msg:
                                messages[0]["content"] = (
                                    original_system_msg + 
                                    "\n\nCRITICAL: You must write a summary in your own words. "
                                    "Do NOT output tool results or formatted lists."
                                )
                            
                            continue
                
                return {
                    "content": content,
                    "reasoning_content": reasoning_content,
                    "tool_calls": all_tool_calls,
                    "tool_results": all_tool_results,
                    "finish_reason": finish_reason,
                    "usage": total_usage,
                    "turns": turn + 1,
                }
            
            # Execute tools and prepare tool response messages
            tool_messages = []
            for tool_call in tool_calls:
                tool_call_id = tool_call.get("id")
                function = tool_call.get("function", {})
                tool_name = function.get("name")
                arguments_str = function.get("arguments", "{}")
                
                # Parse arguments
                try:
                    if isinstance(arguments_str, str):
                        arguments = json.loads(arguments_str)
                    else:
                        arguments = arguments_str
                except json.JSONDecodeError:
                    arguments = {}
                
                # Notify tool call
                if on_tool_call:
                    on_tool_call(tool_name, arguments)
                
                # Track tool call
                all_tool_calls.append({
                    "id": tool_call_id,
                    "name": tool_name,
                    "arguments": arguments
                })
                
                # Execute tool
                if tool_executor:
                    try:
                        tool_result = tool_executor(tool_name, arguments)
                        if not isinstance(tool_result, str):
                            tool_result = json.dumps(tool_result)
                    except Exception as e:
                        tool_result = json.dumps({"error": str(e)})
                else:
                    tool_result = json.dumps({
                        "error": f"No tool executor provided for {tool_name}"
                    })
                
                # Track tool result
                all_tool_results.append({
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": tool_result
                })
                
                # Add tool response message
                tool_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": tool_result
                })
            
            # Add tool response messages to conversation
            messages.extend(tool_messages)
            
            # After tool execution, add guidance to ensure content generation
            # This applies to both normal tool calls and tool calls detected in reasoning tokens
            # Only add guidance once per conversation to avoid confusion
            if tool_calls and not guidance_added:
                # Check if this is the last turn or if we've made multiple tool calls
                # Add guidance to prompt for summary generation
                messages.append({
                    "role": "user",
                    "content": "IMPORTANT: Write a summary of the tool results in your own words. Do NOT output raw tool results, formatted lists, URLs, or numbered lists. Write a natural language summary with complete sentences."
                })
                guidance_added = True
                # Continue to get response to guidance, even if we're at max_turns
                # This ensures we get a summary instead of raw tool results
                continue
        
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to get response from GPT-OSS on turn {turn + 1}: {e}")
    
    # Max turns reached - check if last content is invalid
    last_content = ""
    last_reasoning = ""
    if messages:
        last_msg = messages[-1]
        if last_msg.get("role") == "assistant":
            last_content = last_msg.get("content", "").strip()
            last_reasoning = last_msg.get("reasoning_content", "").strip()
    
    is_invalid_final = False
    if last_content:
        # Check for raw tool results format
        if (last_content.startswith("Found ") and "search results for" in last_content) or \
           last_content.count("\n   URL:") >= 2 or \
           (last_content.count("URL:") >= 2 and last_content.count("\n   ") >= 2):
            is_invalid_final = True
    
    # If content is invalid, return empty content with error
    if is_invalid_final:
        last_content = ""
    
    return {
        "content": last_content,
        "reasoning_content": last_reasoning,
        "tool_calls": all_tool_calls,
        "tool_results": all_tool_results,
        "finish_reason": "max_turns",
        "usage": total_usage,
        "turns": max_turns,
        "error": "Maximum number of turns reached" + (" - invalid content detected" if is_invalid_final else "")
    }


def generate_with_tools(
    prompt: str,
    tools: List[Dict[str, Any]],
    tool_executor: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
    reasoning_level: str = "medium",
    temperature: float = 0.5,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    max_turns: int = 10,
    tool_choice: Optional[str] = None,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Generate a response with tool calling support (multi-turn conversation).
    
    After GPT-OSS emits a tool call, this function:
    1. Executes the tool
    2. Sends back the result as a tool response message
    3. Continues the conversation until a final response is received
    
    Args:
        prompt: The input prompt
        tools: List of tool definitions for function calling
        tool_executor: Optional function to execute tools. Signature: (tool_name, arguments) -> result
                      If None, uses a default executor that expects tools to be callable Python functions
        reasoning_level: Reasoning level ('low', 'medium', 'high')
        temperature: Sampling temperature (default: 0.5)
        top_p: Nucleus sampling parameter (default: 1.0)
        max_tokens: Maximum tokens per turn (default: 1024)
        max_turns: Maximum number of conversation turns (default: 10)
        max_retries: Maximum number of retries for invalid content (default: 3)
    
    Returns:
        Dictionary containing the final response, all tool calls made, and metadata
    """
    url = "http://localhost:8000/v1/chat/completions"
    
    # Validate reasoning level
    if reasoning_level not in ["low", "medium", "high"]:
        raise ValueError(
            "reasoning_level must be 'low', 'medium', or 'high', "
            f"got '{reasoning_level}'"
        )
    
    # System message - generic for all tools, not web-search specific
    system_msg = (
        "You are a helpful assistant. When you use tools:\n"
        "1. Provide all required parameters for each tool\n"
        "2. Do not use invalid parameters that aren't part of the tool definition\n"
        "3. After receiving tool results, synthesize the information and provide a clear response\n"
        "4. CRITICAL: Do NOT output raw tool results, formatted lists with URLs, or tool result formats\n"
        "5. CRITICAL: Write summaries in your own words - never copy/paste tool output\n"
        "6. Do NOT echo instructions or prompts - write actual content\n"
        "7. Write in complete sentences and paragraphs when appropriate\n"
        f"Reasoning: {reasoning_level}"
    )
    
    # Conversation history
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt}
    ]
    
    # Track tool calls across turns
    all_tool_calls = []
    all_tool_results = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    guidance_added = False  # Track if we've added guidance message after tool execution
    
    for turn in range(max_turns):
        # Prepare payload
        payload = {
            "model": "openai/gpt-oss-20b",
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice if tool_choice is not None else "auto",
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }
        
        try:
            # Make request
            response = requests.post(url, json=payload, timeout=300)
            if not response.ok:
                # Try to get error details from response
                error_detail = f"Status: {response.status_code}, "
                try:
                    error_json = response.json()
                    error_detail += f"JSON: {error_json}"
                except Exception:
                    error_text = response.text[:1000]
                    error_detail += f"Text: {error_text if error_text else '(empty)'}"
                    error_detail += f", Headers: {dict(response.headers)}"
                raise RuntimeError(
                    f"Server error {response.status_code}: {error_detail}"
                )
            response.raise_for_status()
            result = response.json()
            
            # Update usage
            usage = result.get("usage", {})
            total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
            total_usage["total_tokens"] += usage.get("total_tokens", 0)
            
            message = result["choices"][0]["message"]
            finish_reason = result["choices"][0]["finish_reason"]
            
            # Add assistant message to history
            messages.append({
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": message.get("tool_calls")
            })
            
            # Check for tool calls
            tool_calls = message.get("tool_calls", [])
            
            if not tool_calls or finish_reason != "tool_calls":
                # Final response received
                content = (message.get("content") or "").strip()
                reasoning_content = (message.get("reasoning_content") or "").strip()
                
                # Check if content is invalid (raw tool results, etc.) even if it exists
                is_invalid_content = False
                if content:
                    # Check for raw tool results format
                    if (content.startswith("Found ") and "search results for" in content) or \
                       content.count("\n   URL:") >= 2 or \
                       (content.count("URL:") >= 2 and content.count("\n   ") >= 2):
                        is_invalid_content = True
                
                # If we have reasoning but no content, OR invalid content, and we have tool results, retry
                if (not content or is_invalid_content) and all_tool_results:
                    # Check if we have successful tool results
                    # Generic check: any result that's not an error
                    has_successful_results = False
                    for tr in all_tool_results:
                        result_str = tr.get('result', '')
                        if result_str and not result_str.startswith('{"error"'):
                            # Check if result looks successful (not empty, not error format)
                            if len(result_str.strip()) > 10:
                                has_successful_results = True
                                break
                    
                    # Retry if we have successful results
                    # Also check finish_reason - if it's "tool_calls", model wants more tools, don't retry yet
                    # Allow retry even at max_turns if we have invalid content (but not if we're already past max)
                    if has_successful_results and turn <= max_turns and finish_reason != "tool_calls":
                        # Count existing retries - check for any retry-like user messages
                        retry_count = sum(1 for msg in messages[-5:] if 
                                         msg.get("role") == "user" and 
                                         any(phrase in msg.get("content", "").lower() for phrase in [
                                             "write a summary", "summarize", "now that you have",
                                             "what are the main points", "what information can you extract",
                                             "please analyze", "please write a summary", "write the summary",
                                             "focus on the main points"
                                         ]))
                        
                        # Use imperative statements, not questions, to avoid echo
                        # Vary messages based on retry count
                        if retry_count == 0:
                            follow_up_msg = (
                                "Please write a summary of the key information from the search results. "
                                "Focus on the main points and organize them clearly."
                            )
                        elif retry_count == 1:
                            follow_up_msg = "Write a summary of the search results above."
                        elif retry_count == 2:
                            follow_up_msg = "Write the summary."
                        else:
                            # For additional retries, use very direct imperative
                            follow_up_msg = "Summarize the results in your own words."
                        
                        # Remove any invalid assistant message first
                        if messages and messages[-1].get("role") == "assistant":
                            messages.pop()  # Remove invalid assistant message
                        
                        messages.append({
                            "role": "user",
                            "content": follow_up_msg
                        })
                        
                        # Update system message to emphasize writing summaries
                        original_system_msg = messages[0]["content"]
                        if "CRITICAL:" not in original_system_msg:
                            messages[0]["content"] = (
                                original_system_msg + 
                                "\n\nCRITICAL: You must write a summary in your own words. "
                                "Do NOT output tool results or echo instructions. "
                                "Write actual summary content with complete sentences."
                            )
                        
                        continue
                
                # Filter out invalid content
                # First check if content matches raw tool results format (already checked above)
                is_invalid = is_invalid_content
                
                # Check if content is tool call JSON
                if _is_tool_call_json(content):
                    is_invalid = True
                
                # Check if content is a follow-up prompt we added (exact match or contains key phrases)
                follow_up_phrases = [
                    "based on the search results provided above",
                    "based on the search results provided",
                    "you have already thought about it",
                    "now please write the actual summary",
                    "please provide a comprehensive summary",
                    "do not just think about it - actually write",
                    "summarize the search results above",
                    "write a summary now using the information",
                    "be clear, detailed, and complete",
                    "write a summary of the search results",
                    "summarize the results",
                    "synthesize the information into a well-written summary",
                    "create a summary from the search results",
                    "provide a summary now",
                    "generate the summary content",
                    "synthesize the search results into a summary",
                    "now that you have the search results",
                    "what are the main points and key information",
                    "what information can you extract from the search results",
                    "please analyze and summarize the information provided",
                    "please write a summary of the key information",
                    "write the summary now",
                    "write the summary",
                    "focus on the main points and organize them clearly"
                ]
                # Check if content starts with or contains these phrases
                content_lower = content.lower().strip()
                
                # Check for exact old retry message pattern
                old_retry_pattern = "based on the search results provided above, please write a comprehensive summary"
                if old_retry_pattern in content_lower:
                    is_invalid = True
                
                # Check if content starts with any of these phrases (most common case)
                starts_with_prompt = any(content_lower.startswith(phrase) for phrase in follow_up_phrases)
                # Check if content contains multiple prompt phrases (echoing the whole prompt)
                prompt_phrase_count = sum(1 for phrase in follow_up_phrases if phrase in content_lower)
                
                # Also check for exact short prompt matches
                exact_short_prompts = [
                    "write a summary of the search results.",
                    "write a summary of the search results",
                    "summarize the results.",
                    "summarize the results",
                    "write the summary.",
                    "write the summary",
                    "write the summary now.",
                    "write the summary now"
                ]
                is_exact_short_prompt = content_lower.strip() in exact_short_prompts or \
                                        content_lower.strip().rstrip('.') in [p.rstrip('.') for p in exact_short_prompts]
                
                # Check if content IS one of our retry prompts (exact or near-exact match)
                # This catches cases where LLM echoes the retry message verbatim
                is_retry_echo = False
                if len(content) < 200:  # Retry prompts are usually short
                    # Check if content matches any retry prompt closely
                    for phrase in follow_up_phrases:
                        if phrase in content_lower and len(content_lower) < len(phrase) * 1.5:
                            # Content is very similar to a retry prompt
                            is_retry_echo = True
                            break
                
                if starts_with_prompt or prompt_phrase_count >= 2 or is_exact_short_prompt or is_retry_echo:
                    # Content is echoing a retry prompt
                    is_invalid = True
                
                # Check if content starts with "Found X search results" (tool result format)
                # Also check if it contains the tool result format anywhere
                # This indicates the LLM is outputting tool results instead of summarizing
                if (content.startswith("Found ") and "search results for" in content) or \
                   ("Found " in content and "search results for" in content):
                    # Content is the tool result format being output - need a summary instead
                    is_invalid = True
                
                # Check if content looks like formatted tool results (numbered list with URLs)
                # Pattern: "1. Title\n   URL: ...\n   snippet..."
                if content.count("\n   URL:") >= 2 or (content.count("URL:") >= 2 and content.count("\n   ") >= 2):
                    # Looks like formatted search results, not a summary
                    is_invalid = True
                
                # Check if content is a JSON error message
                if content.startswith('{"error"') or content.startswith('{"success":false'):
                    try:
                        error_obj = json.loads(content)
                        if "error" in error_obj:
                            is_invalid = True
                    except (json.JSONDecodeError, ValueError):
                        pass
                
                # Check if content is just an error message string
                if content.startswith('{"error":') or '"error"' in content[:50]:
                    is_invalid = True
                
                # If invalid content and we have successful tool results, retry
                # Check if we have at least one successful tool result
                # Generic check: any result that's not an error
                has_successful_results = False
                for tr in all_tool_results:
                    result_str = tr.get('result', '')
                    if result_str and not result_str.startswith('{"error"'):
                        # Check if result looks successful (not empty, not error format)
                        if len(result_str.strip()) > 10:
                            has_successful_results = True
                            break
                
                # Only retry if we have successful results AND the finish_reason indicates
                # the model is done (not "tool_calls" which means more tools are coming)
                # Allow retry even at max_turns if we have invalid content
                should_retry = (is_invalid and has_successful_results and 
                               turn <= max_turns and 
                               finish_reason != "tool_calls")
                
                if should_retry:
                    # Check if we've already retried (to avoid infinite loops)
                    # Count follow-up messages we've added
                    retry_count = sum(1 for msg in messages[-5:] if 
                                     msg.get("role") == "user" and 
                                     ("Based on the search results" in msg.get("content", "") or
                                      "comprehensive summary" in msg.get("content", "").lower() or
                                      "write the actual summary" in msg.get("content", "").lower() or
                                      "write a summary" in msg.get("content", "").lower() or
                                      "summarize the results" in msg.get("content", "").lower() or
                                      "now that you have the search results" in msg.get("content", "").lower() or
                                      "what are the main points" in msg.get("content", "").lower()))
                    
                    if retry_count < max_retries:
                        # Detect what type of invalid content we got
                        is_tool_results = (content.startswith("Found ") and "search results for" in content) or \
                                         content.count("\n   URL:") >= 2
                        is_prompt_echo = any(phrase in content_lower for phrase in follow_up_phrases)
                        
                        if is_tool_results:
                            # LLM output tool results - use imperative statements, not questions
                            if retry_count == 0:
                                follow_up_msg = (
                                    "Please write a summary of the key information from the search results. "
                                    "Focus on the main points and organize them clearly."
                                )
                            elif retry_count == 1:
                                follow_up_msg = "Write the summary now."
                            else:
                                # For additional retries, use very direct imperative
                                follow_up_msg = "Summarize the results in your own words."
                        elif is_prompt_echo:
                            # LLM echoed the prompt - use very short, direct imperative
                            follow_up_msg = "Write the summary."
                        else:
                            # Other invalid content - use simple imperative
                            follow_up_msg = "Write a summary of the search results."
                        
                        # Remove the invalid assistant message and any recent retry attempts
                        # This prevents the LLM from seeing its own invalid output
                        # Remove up to 2 assistant messages if they're invalid
                        removed_count = 0
                        while messages and removed_count < 2:
                            last_msg = messages[-1]
                            if last_msg.get("role") == "assistant":
                                messages.pop()
                                removed_count += 1
                            elif last_msg.get("role") == "user" and removed_count > 0:
                                # Also remove the user message that triggered the invalid response
                                # if it was a retry prompt
                                user_content = last_msg.get("content", "").lower()
                                if any(phrase in user_content for phrase in [
                                    "write a summary", "summarize", "what information",
                                    "now that you have", "please provide"
                                ]):
                                    messages.pop()
                                    removed_count += 1
                                else:
                                    break
                            else:
                                break
                        
                        # Add directive user message
                        messages.append({
                            "role": "user",
                            "content": follow_up_msg
                        })
                        
                        # Also update system message temporarily to emphasize writing summaries
                        original_system_msg = messages[0]["content"]
                        if "CRITICAL:" not in original_system_msg:
                            messages[0]["content"] = (
                                original_system_msg + 
                                "\n\nCRITICAL: You must write a summary in your own words. "
                                "Do NOT output tool results, formatted lists, or echo instructions. "
                                "Write actual summary content with complete sentences."
                            )
                        
                        # Continue to next turn to get proper response
                        continue
                    else:
                        # Too many retries, return empty content
                        content = ""
                elif is_invalid:
                    # No successful tool results or max turns - return empty content
                    content = ""
                
                return {
                    "content": content,
                    "reasoning_content": (message.get("reasoning_content") or "").strip(),
                    "tool_calls": all_tool_calls,
                    "tool_results": all_tool_results,
                    "finish_reason": finish_reason,
                    "usage": total_usage,
                    "turns": turn + 1,
                }
            
            # Execute tools and prepare tool response messages
            tool_messages = []
            for tool_call in tool_calls:
                tool_call_id = tool_call.get("id")
                function = tool_call.get("function", {})
                tool_name = function.get("name")
                arguments_str = function.get("arguments", "{}")
                
                # Parse arguments
                try:
                    if isinstance(arguments_str, str):
                        arguments = json.loads(arguments_str)
                    else:
                        arguments = arguments_str
                except json.JSONDecodeError:
                    arguments = {}
                
                # Track tool call
                all_tool_calls.append({
                    "id": tool_call_id,
                    "name": tool_name,
                    "arguments": arguments
                })
                
                # Execute tool
                if tool_executor:
                    try:
                        tool_result = tool_executor(tool_name, arguments)
                        if not isinstance(tool_result, str):
                            tool_result = json.dumps(tool_result)
                    except Exception as e:
                        tool_result = json.dumps({"error": str(e)})
                else:
                    # Default: return error message
                    tool_result = json.dumps({
                        "error": f"No tool executor provided for {tool_name}"
                    })
                
                # Track tool result
                all_tool_results.append({
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": tool_result
                })
                
                # Add tool response message
                tool_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": tool_result
                })
            
            # Add tool response messages to conversation
            messages.extend(tool_messages)
            
            # After tool execution, add guidance to ensure content generation
            # This applies to both normal tool calls and tool calls detected in reasoning tokens
            # Only add guidance once per conversation to avoid confusion
            if tool_calls and not guidance_added:
                # Add guidance to prompt for summary generation
                messages.append({
                    "role": "user",
                    "content": "IMPORTANT: Write a summary of the tool results in your own words. Do NOT output raw tool results, formatted lists, URLs, or numbered lists. Write a natural language summary with complete sentences."
                })
                guidance_added = True
                # Continue to get response to guidance, even if we're at max_turns
                # This ensures we get a summary instead of raw tool results
                continue
        
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to get response from GPT-OSS on turn {turn + 1}: {e}")
    
    # Max turns reached - check if last content is invalid
    last_content = messages[-1].get("content", "").strip() if messages else ""
    is_invalid_final = False
    if last_content:
        # Check for raw tool results format
        if (last_content.startswith("Found ") and "search results for" in last_content) or \
           last_content.count("\n   URL:") >= 2 or \
           (last_content.count("URL:") >= 2 and last_content.count("\n   ") >= 2):
            is_invalid_final = True
    
    # If content is invalid, return empty content with error
    if is_invalid_final:
        last_content = ""
    
    return {
        "content": last_content,
        "reasoning_content": "",
        "tool_calls": all_tool_calls,
        "tool_results": all_tool_results,
        "finish_reason": "max_turns",
        "usage": total_usage,
        "turns": max_turns,
        "error": "Maximum number of turns reached" + (" - invalid content detected" if is_invalid_final else "")
    }


def _stream_response(
    url: str, payload: Dict[str, Any]
) -> Iterator[Dict[str, Any]]:
    """
    Stream responses from GPT-OSS and yield delta updates.
    Separates thinking tokens (reasoning_content) from final output tokens.
    """
    response = requests.post(url, json=payload, stream=True, timeout=300)
    response.raise_for_status()
    
    # Accumulators for content and reasoning
    accumulated_content = ""
    accumulated_reasoning = ""
    
    for line in response.iter_lines():
        if not line:
            continue
        
        # Parse SSE format: "data: {...}"
        if line.startswith(b"data: "):
            line = line[6:]  # Remove "data: " prefix
        
        if line == b"[DONE]":
            break
        
        try:
            chunk = json.loads(line)
            
            if "choices" not in chunk or len(chunk["choices"]) == 0:
                continue
            
            choice = chunk["choices"][0]
            delta = choice.get("delta", {})
            
            # Extract content delta (final output tokens)
            content_delta = delta.get("content", "")
            if content_delta:
                accumulated_content += content_delta
            
            # Extract reasoning_content delta (thinking tokens)
            # Note: reasoning_content may be in delta or message object
            reasoning_delta = delta.get("reasoning_content", "")
            if reasoning_delta:
                accumulated_reasoning += reasoning_delta
            
            # Check reasoning_content in message object (for some vLLM versions)
            message = delta.get("message", {})
            if isinstance(message, dict):
                msg_reasoning = message.get("reasoning_content", "")
                if msg_reasoning:
                    accumulated_reasoning += msg_reasoning
            
            # Yield delta update
            yield {
                "delta_content": content_delta,
                "delta_reasoning": reasoning_delta,
                "accumulated_content": accumulated_content,
                "accumulated_reasoning": accumulated_reasoning,
                "finish_reason": choice.get("finish_reason"),
                "usage": chunk.get("usage"),
            }
            
        except json.JSONDecodeError:
            continue


def generate_stream(
    prompt: str,
    reasoning_level: str = "medium",
    temperature: float = 0.5,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
    on_content: Optional[Callable[[str], None]] = None,
    on_reasoning: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Generate a streamed response with callbacks for content and reasoning.
    
    Args:
        prompt: The input prompt
        reasoning_level: Reasoning level ('low', 'medium', 'high')
        temperature: Sampling temperature (default: 0.5)
        top_p: Nucleus sampling parameter (default: 1.0)
        max_tokens: Maximum tokens to generate (default: 1024)
        tools: Optional list of tool definitions for function calling
        tool_choice: How to handle tool calls ('auto', 'none', or tool name)
        on_content: Optional callback called with each content delta
        on_reasoning: Optional callback called with each reasoning delta
    
    Returns:
        Dictionary containing the final accumulated response and metadata
    """
    stream = generate(
        prompt=prompt,
        reasoning_level=reasoning_level,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        tools=tools,
        tool_choice=tool_choice,
        stream=True,
    )
    
    final_content = ""
    final_reasoning = ""
    finish_reason = None
    usage = {}
    
    for chunk in stream:
        # Call callbacks if provided
        if chunk["delta_content"] and on_content:
            on_content(chunk["delta_content"])
        if chunk["delta_reasoning"] and on_reasoning:
            on_reasoning(chunk["delta_reasoning"])
        
        # Accumulate final values
        final_content = chunk["accumulated_content"]
        final_reasoning = chunk["accumulated_reasoning"]
        if chunk.get("finish_reason"):
            finish_reason = chunk["finish_reason"]
        if chunk.get("usage"):
            usage = chunk["usage"]
    
    return {
        "content": final_content.strip(),
        "reasoning_content": final_reasoning.strip(),
        "finish_reason": finish_reason,
        "usage": usage,
    }


def get_content(response: Dict[str, Any]) -> str:
    """Extract just the content string from a response."""
    return response["content"]


# Example tool definitions
EXAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get the current weather in a given location"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit for temperature"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "A mathematical expression to evaluate, "
                            "e.g. '2 + 2' or 'sqrt(16)'"
                        ),
                    }
                },
                "required": ["expression"]
            }
        }
    }
]


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("Testing GPT-OSS client")
    print("="*60)

    # Test 1: Basic generation with different reasoning levels
    print("\n" + "-"*60)
    print("Test 1: Basic generation (reasoning_level='high')")
    print("-"*60)
    test_prompt = "What is 25 * 37? Show your work."
    try:
        result = generate(
            prompt=test_prompt,
            reasoning_level="high",
            temperature=0.5,
            top_p=0.9,
            max_tokens=1024
        )
        print(f"Prompt: {test_prompt}")
        print(f"Response: {result['content']}")
        if result.get('reasoning_content'):
            print(f"Reasoning: {result['reasoning_content']}")
        print(f"Finish reason: {result['finish_reason']}")
        print(f"Tokens used: {result['usage']}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 2: Streaming with callbacks
    print("\n" + "-"*60)
    print("Test 2: Streaming with reasoning separation")
    print("-"*60)
    test_prompt = "What is 15 * 23? Show your work step by step."
    try:
        print("Streaming response:")
        print("  [Thinking tokens]: ", end="", flush=True)
        
        reasoning_buffer = []
        content_buffer = []
        
        def on_reasoning(delta):
            reasoning_buffer.append(delta)
            print(delta, end="", flush=True)
        
        def on_content(delta):
            if not content_buffer:
                print("\n  [Final output]: ", end="", flush=True)
            content_buffer.append(delta)
            print(delta, end="", flush=True)
        
        result = generate_stream(
            prompt=test_prompt,
            reasoning_level="high",
            temperature=0.5,
            top_p=0.9,
            max_tokens=300,
            on_reasoning=on_reasoning,
            on_content=on_content,
        )
        print("\n\nFinal result:")
        print(f"  Reasoning: {result['reasoning_content']}")
        print(f"  Content: {result['content']}")
        print(f"  Finish reason: {result['finish_reason']}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 3: Manual streaming iteration
    print("\n" + "-"*60)
    print("Test 3: Manual streaming iteration")
    print("-"*60)
    test_prompt = "Explain quantum computing briefly."
    try:
        print("Streaming chunks:")
        stream = generate(
            prompt=test_prompt,
            reasoning_level="medium",
            temperature=0.7,
            top_p=0.9,
            max_tokens=200,
            stream=True
        )
        
        for i, chunk in enumerate(stream):
            if chunk["delta_reasoning"]:
                print(
                    f"  [Chunk {i}] Reasoning: {chunk['delta_reasoning']}"
                )
            if chunk["delta_content"]:
                print(
                    f"  [Chunk {i}] Content: {chunk['delta_content']}"
                )
        
        print(
            f"\nFinal accumulated reasoning: "
            f"{chunk['accumulated_reasoning']}"
        )
        print(
            f"Final accumulated content: {chunk['accumulated_content']}"
        )
    except Exception as e:
        print(f"Error: {e}")

    # Test 4: With tools
    print("\n" + "-"*60)
    print("Test 4: With function calling tools")
    print("-"*60)
    test_prompt = "What's the weather in San Francisco and calculate 15 * 23?"
    try:
        result = generate(
            prompt=test_prompt,
            reasoning_level="medium",
            temperature=0.7,
            top_p=0.95,
            max_tokens=1024,
            tools=EXAMPLE_TOOLS
        )
        print(f"Prompt: {test_prompt}")
        print(f"Response: {result['content']}")
        if result.get('reasoning_content'):
            print(f"Reasoning: {result['reasoning_content']}")
        if result['tool_calls']:
            print(f"Tool calls: {result['tool_calls']}")
        print(f"Finish reason: {result['finish_reason']}")
    except Exception as e:
        print(f"Error: {e}")

