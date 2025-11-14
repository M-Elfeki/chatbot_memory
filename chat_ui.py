#!/usr/bin/env python3
"""
Multi-turn chat client with streaming, web browsing, and memory tools.
Features:
- Always streams responses
- Automatically uses web_search and memory tools
- Saves multi-turn chat history
- Elegant Gradio UI with reasoning chain display
- User controls for reasoning level, temperature, top-p
- Visual indicators for tool usage
"""
import os
import sys
import json
import gradio as gr
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import threading
import queue

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from client import generate_with_tools_stream
from tools.memory_tool import MEMORY_STORE_TOOL, MEMORY_RETRIEVE_TOOL, memory_executor
from tools.web_search_tool import WEB_SEARCH_TOOL, web_search_executor
from tools.unified_executor import unified_tool_executor


class ChatHistory:
    """Manages chat history persistence."""
    
    def __init__(self, storage_dir: str = "chat_history"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.history_file = os.path.join(storage_dir, "chat_history.json")
        self.history: List[Dict[str, Any]] = []
        self.load_history()
    
    def load_history(self):
        """Load chat history from disk."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load chat history: {e}")
                self.history = []
        else:
            self.history = []
    
    def save_history(self):
        """Save chat history to disk."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to save chat history: {e}")
    
    def add_message(self, role: str, content: str, reasoning: str = "", 
                    tool_calls: List[Dict] = None, metadata: Dict = None):
        """Add a message to history."""
        message = {
            "role": role,
            "content": content,
            "reasoning": reasoning,
            "tool_calls": tool_calls or [],
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        self.history.append(message)
        self.save_history()
    
    def clear_history(self):
        """Clear chat history."""
        self.history = []
        self.save_history()
    
    def get_history_for_llm(self) -> List[Dict[str, str]]:
        """Get history in format suitable for LLM API."""
        messages = []
        for msg in self.history:
            if msg["role"] == "user":
                messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                assistant_msg = {"role": "assistant", "content": msg["content"]}
                if msg.get("tool_calls"):
                    assistant_msg["tool_calls"] = msg["tool_calls"]
                messages.append(assistant_msg)
            elif msg["role"] == "tool":
                messages.append({
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "content": msg["content"]
                })
        return messages


class StreamingChatClient:
    """Streaming chat client with tool support."""
    
    def __init__(self, storage_path: str = "memory_store"):
        self.storage_path = storage_path
        self.chat_history = ChatHistory()
        self.tools = [WEB_SEARCH_TOOL, MEMORY_STORE_TOOL, MEMORY_RETRIEVE_TOOL]
        
    def tool_executor(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Unified tool executor."""
        return unified_tool_executor(tool_name, arguments, self.storage_path)
    
    def chat(
        self,
        message: str,
        history: List[Tuple[str, str]],
        reasoning_level: str,
        temperature: float,
        top_p: float,
        progress=gr.Progress()
    ):
        """
        Process a chat message with streaming.
        Yields updates for Gradio streaming.
        """
        if not message.strip():
            yield history, "", "", ""
            return
        
        # Add user message to history
        self.chat_history.add_message("user", message)
        
        # Build prompt with conversation history for multi-turn
        # Convert Gradio history format to a comprehensive prompt
        # This ensures the model has full context of the conversation
        conversation_context = ""
        if history:
            conversation_context = "Previous conversation:\n\n"
            for user_msg, assistant_msg in history:
                conversation_context += f"User: {user_msg}\n"
                if assistant_msg:
                    conversation_context += f"Assistant: {assistant_msg}\n\n"
                else:
                    conversation_context += "Assistant: [No response yet]\n\n"
        
        conversation_context += f"Current question:\nUser: {message}\nAssistant:"
        
        # Accumulators for streaming
        reasoning_buffer = []
        content_buffer = []
        tool_calls_made = []
        tool_status_updates = []
        
        # Create a queue for streaming updates
        update_queue = queue.Queue()
        stream_complete = threading.Event()
        
        def on_reasoning(delta: str):
            """Callback for reasoning tokens."""
            reasoning_buffer.append(delta)
            update_queue.put(("reasoning", "".join(reasoning_buffer)))
        
        def on_content(delta: str):
            """Callback for content tokens."""
            content_buffer.append(delta)
            update_queue.put(("content", "".join(content_buffer)))
        
        def on_tool_call(tool_name: str, arguments: Dict[str, Any]):
            """Callback when tool is called."""
            tool_calls_made.append({"name": tool_name, "arguments": arguments})
            tool_status = ""
            if tool_name == "web_search":
                query = arguments.get('query', '')[:50]
                tool_status = f"🌐 Searching web: {query}..."
            elif tool_name == "memory_store":
                text = arguments.get('text', '')[:50]
                tool_status = f"💾 Storing memory: {text}..."
            elif tool_name == "memory_retrieve":
                query = arguments.get('query', '')[:50]
                tool_status = f"🔍 Retrieving memory: {query}..."
            else:
                tool_status = f"🔧 Using {tool_name}"
            
            tool_status_updates.append(tool_status)
            update_queue.put(("tool", "\n".join(tool_status_updates)))
        
        # Start streaming in a thread
        result_container = {"result": None, "error": None}
        
        def stream_worker():
            try:
                # Use conversation context as prompt for multi-turn
                result = generate_with_tools_stream(
                    prompt=conversation_context,
                    tools=self.tools,
                    tool_executor=self.tool_executor,
                    tool_choice="auto",
                    reasoning_level=reasoning_level,
                    temperature=temperature,
                    top_p=top_p,
                    on_reasoning=on_reasoning,
                    on_content=on_content,
                    on_tool_call=on_tool_call,
                )
                result_container["result"] = result
            except Exception as e:
                result_container["error"] = str(e)
            finally:
                stream_complete.set()
        
        stream_thread = threading.Thread(target=stream_worker, daemon=True)
        stream_thread.start()
        
        # Yield updates while streaming
        current_output = ""
        current_reasoning = ""
        current_tool_status = ""
        
        # Poll for updates
        while not stream_complete.is_set() or not update_queue.empty():
            try:
                update_type, update_value = update_queue.get(timeout=0.05)
                if update_type == "reasoning":
                    current_reasoning = update_value
                elif update_type == "content":
                    current_output = update_value
                elif update_type == "tool":
                    current_tool_status = update_value
                
                # Update history with current output
                updated_history = history + [(message, current_output)]
                yield updated_history, current_reasoning, current_tool_status
            except queue.Empty:
                # Yield current state even if no new update
                if current_output or current_reasoning or current_tool_status:
                    updated_history = history + [(message, current_output)]
                    yield updated_history, current_reasoning, current_tool_status
                continue
        
        # Wait for thread to complete
        stream_thread.join(timeout=60)
        
        # Get final result
        if result_container["error"]:
            error_msg = f"Error: {result_container['error']}"
            updated_history = history + [(message, error_msg)]
            yield updated_history, "", ""
            return
        
        result = result_container["result"]
        if not result:
            error_msg = "Error: No response received"
            updated_history = history + [(message, error_msg)]
            yield updated_history, "", ""
            return
        
        # Get final content
        final_content = "".join(content_buffer) if content_buffer else result.get("content", "")
        final_reasoning = "".join(reasoning_buffer) if reasoning_buffer else result.get("reasoning_content", "")
        
        # Add assistant message to history
        self.chat_history.add_message(
            "assistant",
            final_content,
            reasoning=final_reasoning,
            tool_calls=result.get("tool_calls", []),
            metadata={
                "temperature": temperature,
                "top_p": top_p,
                "reasoning_level": reasoning_level,
                "usage": result.get("usage", {})
            }
        )
        
        # Add tool results to history
        tool_results = result.get("tool_results", [])
        for i, tool_result in enumerate(tool_results):
            tool_call = result.get("tool_calls", [])[i] if i < len(result.get("tool_calls", [])) else {}
            self.chat_history.add_message(
                "tool",
                tool_result.get("result", ""),
                metadata={"tool_call_id": tool_call.get("id", f"call_{i}")}
            )
        
        # Final update
        updated_history = history + [(message, final_content)]
        final_tool_status = "\n".join(tool_status_updates) if tool_status_updates else ""
        yield updated_history, final_reasoning, final_tool_status


def create_ui():
    """Create and launch the Gradio UI."""
    client = StreamingChatClient()
    
    # Custom CSS for better styling
    custom_css = """
    .reasoning-box {
        background-color: #f5f5f5;
        border-left: 3px solid #4CAF50;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
    }
    .tool-status {
        background-color: #e3f2fd;
        border-left: 3px solid #2196F3;
        padding: 8px;
        margin: 5px 0;
        border-radius: 5px;
        font-size: 0.85em;
    }
    """
    
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🤖 GPT-OSS Multi-Turn Chat
        
        Chat with GPT-OSS reasoning model with automatic web browsing and memory tools.
        - **Always streams** responses in real-time
        - **Automatically uses** web search and memory tools as needed
        - **Saves** multi-turn conversation history
        - **Shows** reasoning chain separately from output
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    show_label=True,
                    container=True,
                    bubble_full_width=False
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Your message",
                        placeholder="Type your message here...",
                        scale=4,
                        lines=2
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("Clear History", variant="stop")
                    save_btn = gr.Button("Save Chat", variant="secondary")
            
            with gr.Column(scale=2):
                gr.Markdown("### ⚙️ Settings")
                
                reasoning_level = gr.Radio(
                    choices=["low", "medium", "high"],
                    value="medium",
                    label="Reasoning Level",
                    info="Controls the amount of reasoning tokens"
                )
                
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                    info="Controls randomness (0.0 = deterministic, 2.0 = very random)"
                )
                
                top_p = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="Top-p (Nucleus Sampling)",
                    info="Controls diversity via nucleus sampling"
                )
                
                gr.Markdown("### 🧠 Reasoning Chain")
                reasoning_display = gr.Textbox(
                    label="",
                    lines=8,
                    max_lines=15,
                    interactive=False,
                    show_label=False,
                    placeholder="Reasoning tokens will appear here...",
                    elem_classes=["reasoning-box"]
                )
                
                gr.Markdown("### 🔧 Tool Usage")
                tool_status = gr.Textbox(
                    label="",
                    lines=5,
                    max_lines=10,
                    interactive=False,
                    show_label=False,
                    placeholder="Tool usage indicators will appear here...",
                    elem_classes=["tool-status"]
                )
        
        # State for chat history
        chat_history_state = gr.State([])
        
        # Event handlers
        def user_message(message, history, reasoning_lvl, temp, top_p_val):
            """Handle user message submission."""
            if not message.strip():
                yield history, "", "", ""
                return
            
            # Stream updates from chat
            for updated_history, reasoning, tools in client.chat(
                message, history, reasoning_lvl, temp, top_p_val
            ):
                yield updated_history, reasoning, tools
        
        def clear_chat():
            """Clear chat history."""
            client.chat_history.clear_history()
            return [], "", "", ""
        
        def save_chat():
            """Save current chat."""
            client.chat_history.save_history()
            return "Chat saved successfully!"
        
        # Wire up events
        submit_btn.click(
            user_message,
            inputs=[msg, chatbot, reasoning_level, temperature, top_p],
            outputs=[chatbot, reasoning_display, tool_status]
        ).then(
            lambda: "",  # Clear message box
            outputs=[msg]
        )
        
        msg.submit(
            user_message,
            inputs=[msg, chatbot, reasoning_level, temperature, top_p],
            outputs=[chatbot, reasoning_display, tool_status]
        ).then(
            lambda: "",  # Clear message box
            outputs=[msg]
        )
        
        clear_btn.click(
            clear_chat,
            outputs=[chatbot, msg, reasoning_display, tool_status]
        )
        
        save_btn.click(
            save_chat,
            outputs=[gr.Textbox(visible=False)]
        )
        
        # Load existing history on startup
        def load_history():
            """Load existing chat history into the UI."""
            if client.chat_history.history:
                # Convert history to Gradio format
                gradio_history = []
                current_user_msg = ""
                current_assistant_msg = ""
                
                for msg_data in client.chat_history.history:
                    if msg_data["role"] == "user":
                        if current_assistant_msg:
                            gradio_history.append((current_user_msg, current_assistant_msg))
                            current_assistant_msg = ""
                        current_user_msg = msg_data["content"]
                    elif msg_data["role"] == "assistant":
                        current_assistant_msg = msg_data["content"]
                
                if current_user_msg and current_assistant_msg:
                    gradio_history.append((current_user_msg, current_assistant_msg))
                elif current_user_msg:
                    gradio_history.append((current_user_msg, ""))
                
                return gradio_history
            return []
        
        # Load history on page load
        demo.load(load_history, outputs=[chatbot])
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

