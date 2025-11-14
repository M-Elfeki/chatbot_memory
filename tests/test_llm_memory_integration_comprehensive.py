#!/usr/bin/env python3
"""
Comprehensive integration test for LLM + memory tool.
Tests various prompts to verify the LLM can robustly use memory_store and memory_retrieve.
Lists all memories after tests complete.
"""
import sys
import os
import json
import tempfile
import shutil
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client import generate_with_tools
from tools.memory_tool import (
    MEMORY_STORE_TOOL,
    MEMORY_RETRIEVE_TOOL,
    memory_executor,
    MemoryStore
)


class TestLLMMemoryIntegration:
    """Comprehensive LLM memory integration tests."""
    
    def __init__(self, storage_path=None):
        """Initialize test with a temporary storage path."""
        if storage_path is None:
            self.test_dir = tempfile.mkdtemp()
            self.storage_path = os.path.join(self.test_dir, "test_memory")
        else:
            self.storage_path = storage_path
            self.test_dir = None
        
        # Track tool calls
        self.tool_calls_log = []
        self.memories_stored = []
        self.memories_retrieved = []
        
        # Create unified tool executor
        self.tool_executor = self._create_tool_executor()
    
    def _create_tool_executor(self):
        """Create a unified tool executor that tracks calls."""
        def executor(tool_name, arguments):
            # Track the call
            call_info = {
                "tool": tool_name,
                "arguments": arguments.copy(),
                "timestamp": time.time()
            }
            self.tool_calls_log.append(call_info)
            
            # Add storage_path to arguments
            if "storage_path" not in arguments:
                arguments["storage_path"] = self.storage_path
            
            # Execute the tool
            result_str = memory_executor(tool_name, arguments)
            
            # Parse and track results
            try:
                result = json.loads(result_str)
                if tool_name == "memory_store" and result.get("success"):
                    self.memories_stored.append({
                        "id": result.get("id"),
                        "text": result.get("text"),
                        "timestamp": result.get("timestamp")
                    })
                elif tool_name == "memory_retrieve" and result.get("success"):
                    self.memories_retrieved.append({
                        "query": arguments.get("query"),
                        "count": result.get("count", 0),
                        "results": result.get("results", [])
                    })
            except json.JSONDecodeError:
                pass
            
            return result_str
        
        return executor
    
    def test_store_prompts(self):
        """Test prompts that should trigger memory_store."""
        print("\n" + "="*80)
        print("TEST 1: Memory Store Prompts")
        print("="*80)
        
        store_prompts = [
            "Remember that I prefer Python programming language over Java.",
            "Please remember that my favorite color is blue.",
            "Store this information: I work as a software engineer.",
            "I want you to remember that I like machine learning.",
            "Save this: My favorite programming framework is PyTorch.",
        ]
        
        results = []
        for i, prompt in enumerate(store_prompts, 1):
            print(f"\n[{i}/{len(store_prompts)}] Prompt: {prompt}")
            print("-" * 80)
            
            try:
                result = generate_with_tools(
                    prompt=prompt,
                    tools=[MEMORY_STORE_TOOL, MEMORY_RETRIEVE_TOOL],
                    tool_executor=self.tool_executor,
                    tool_choice="auto",
                    temperature=0.7,
                    top_p=0.95,
                    reasoning_level="low",
                    max_turns=5
                )
                
                tool_calls = result.get("tool_calls", [])
                content = result.get("content", "")
                
                print(f"Response: {content[:200]}...")
                print(f"Tool calls made: {len(tool_calls)}")
                for call in tool_calls:
                    print(f"  - {call.get('name')}: {call.get('arguments', {})}")
                
                results.append({
                    "prompt": prompt,
                    "success": True,
                    "tool_calls": len(tool_calls),
                    "has_store_call": any(c.get("name") == "memory_store" for c in tool_calls)
                })
                
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({
                    "prompt": prompt,
                    "success": False,
                    "error": str(e)
                })
            
            time.sleep(0.5)  # Small delay between requests
        
        return results
    
    def test_retrieve_prompts(self):
        """Test prompts that should trigger memory_retrieve."""
        print("\n" + "="*80)
        print("TEST 2: Memory Retrieve Prompts")
        print("="*80)
        
        retrieve_prompts = [
            "What programming language do I prefer?",
            "What is my favorite color?",
            "What do you remember about my work?",
            "Tell me what you know about my preferences.",
            "What information do you have stored about me?",
        ]
        
        results = []
        for i, prompt in enumerate(retrieve_prompts, 1):
            print(f"\n[{i}/{len(retrieve_prompts)}] Prompt: {prompt}")
            print("-" * 80)
            
            try:
                result = generate_with_tools(
                    prompt=prompt,
                    tools=[MEMORY_STORE_TOOL, MEMORY_RETRIEVE_TOOL],
                    tool_executor=self.tool_executor,
                    tool_choice="auto",
                    temperature=0.7,
                    top_p=0.95,
                    reasoning_level="low",
                    max_turns=5
                )
                
                tool_calls = result.get("tool_calls", [])
                content = result.get("content", "")
                
                print(f"Response: {content[:200]}...")
                print(f"Tool calls made: {len(tool_calls)}")
                for call in tool_calls:
                    print(f"  - {call.get('name')}: {call.get('arguments', {})}")
                
                results.append({
                    "prompt": prompt,
                    "success": True,
                    "tool_calls": len(tool_calls),
                    "has_retrieve_call": any(c.get("name") == "memory_retrieve" for c in tool_calls)
                })
                
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({
                    "prompt": prompt,
                    "success": False,
                    "error": str(e)
                })
            
            time.sleep(0.5)  # Small delay between requests
        
        return results
    
    def test_mixed_prompts(self):
        """Test prompts that might trigger both store and retrieve."""
        print("\n" + "="*80)
        print("TEST 3: Mixed Prompts (Store + Retrieve)")
        print("="*80)
        
        mixed_prompts = [
            "Remember that I like Python, then tell me what you remember.",
            "Save that I prefer dark mode, and also check what other preferences you have stored.",
            "I want you to remember I'm a data scientist. What do you know about my profession?",
        ]
        
        results = []
        for i, prompt in enumerate(mixed_prompts, 1):
            print(f"\n[{i}/{len(mixed_prompts)}] Prompt: {prompt}")
            print("-" * 80)
            
            try:
                result = generate_with_tools(
                    prompt=prompt,
                    tools=[MEMORY_STORE_TOOL, MEMORY_RETRIEVE_TOOL],
                    tool_executor=self.tool_executor,
                    tool_choice="auto",
                    temperature=0.7,
                    top_p=0.95,
                    reasoning_level="low",
                    max_turns=10  # More turns for complex operations
                )
                
                tool_calls = result.get("tool_calls", [])
                content = result.get("content", "")
                
                print(f"Response: {content[:300]}...")
                print(f"Tool calls made: {len(tool_calls)}")
                for call in tool_calls:
                    print(f"  - {call.get('name')}: {call.get('arguments', {})}")
                
                results.append({
                    "prompt": prompt,
                    "success": True,
                    "tool_calls": len(tool_calls),
                    "store_calls": sum(1 for c in tool_calls if c.get("name") == "memory_store"),
                    "retrieve_calls": sum(1 for c in tool_calls if c.get("name") == "memory_retrieve")
                })
                
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({
                    "prompt": prompt,
                    "success": False,
                    "error": str(e)
                })
            
            time.sleep(0.5)  # Small delay between requests
        
        return results
    
    def test_conversational_flow(self):
        """Test a conversational flow with multiple interactions."""
        print("\n" + "="*80)
        print("TEST 4: Conversational Flow")
        print("="*80)
        
        conversation = [
            "Remember that I'm working on a machine learning project.",
            "What project am I working on?",
            "Also remember that I use PyTorch for this project.",
            "What tools do I use for my project?",
            "What do you remember about my work?",
        ]
        
        results = []
        for i, prompt in enumerate(conversation, 1):
            print(f"\n[Turn {i}/{len(conversation)}] User: {prompt}")
            print("-" * 80)
            
            try:
                result = generate_with_tools(
                    prompt=prompt,
                    tools=[MEMORY_STORE_TOOL, MEMORY_RETRIEVE_TOOL],
                    tool_executor=self.tool_executor,
                    tool_choice="auto",
                    temperature=0.7,
                    top_p=0.95,
                    reasoning_level="low",
                    max_turns=5
                )
                
                tool_calls = result.get("tool_calls", [])
                content = result.get("content", "")
                
                print(f"Assistant: {content[:200]}...")
                if tool_calls:
                    print(f"Tools used: {[c.get('name') for c in tool_calls]}")
                
                results.append({
                    "turn": i,
                    "prompt": prompt,
                    "success": True,
                    "tool_calls": len(tool_calls),
                    "response_length": len(content)
                })
                
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({
                    "turn": i,
                    "prompt": prompt,
                    "success": False,
                    "error": str(e)
                })
            
            time.sleep(0.5)  # Small delay between requests
        
        return results
    
    def list_all_memories(self):
        """List all memories stored in the memory store."""
        print("\n" + "="*80)
        print("ALL MEMORIES IN STORE")
        print("="*80)
        
        try:
            store = MemoryStore(storage_path=self.storage_path)
            stats = store.get_stats()
            
            print(f"\nTotal memories stored: {stats['total_items']}")
            print(f"Storage path: {stats['storage_path']}")
            print(f"Model: {stats['model_name']}")
            
            if stats['total_items'] > 0:
                print("\nMemory Contents:")
                print("-" * 80)
                for i, item in enumerate(store.metadata, 1):
                    print(f"\n[{i}] ID: {item.get('id')}")
                    print(f"    Text: {item.get('text')}")
                    print(f"    Timestamp: {item.get('timestamp')}")
                    if 'source' in item or 'category' in item:
                        metadata_items = {k: v for k, v in item.items() 
                                         if k not in ['id', 'text', 'timestamp']}
                        if metadata_items:
                            print(f"    Metadata: {metadata_items}")
            else:
                print("\nNo memories stored yet.")
            
            return store.metadata
            
        except Exception as e:
            print(f"ERROR listing memories: {e}")
            return []
    
    def print_summary(self):
        """Print summary of all tests."""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        print(f"\nTotal tool calls logged: {len(self.tool_calls_log)}")
        print(f"Memory store calls: {sum(1 for c in self.tool_calls_log if c['tool'] == 'memory_store')}")
        print(f"Memory retrieve calls: {sum(1 for c in self.tool_calls_log if c['tool'] == 'memory_retrieve')}")
        
        print(f"\nMemories stored: {len(self.memories_stored)}")
        for mem in self.memories_stored:
            print(f"  - [{mem['id']}] {mem['text'][:60]}...")
        
        print(f"\nRetrieval queries: {len(self.memories_retrieved)}")
        for ret in self.memories_retrieved:
            print(f"  - Query: '{ret['query']}' -> Found {ret['count']} results")
        
        print("\n" + "="*80)
    
    def cleanup(self):
        """Clean up temporary directory if created."""
        if self.test_dir and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            print(f"\nCleaned up test directory: {self.test_dir}")


def main():
    """Run all comprehensive tests."""
    print("="*80)
    print("COMPREHENSIVE LLM MEMORY INTEGRATION TEST")
    print("="*80)
    print("\nThis test will:")
    print("1. Test prompts that should trigger memory_store")
    print("2. Test prompts that should trigger memory_retrieve")
    print("3. Test mixed prompts (store + retrieve)")
    print("4. Test conversational flow")
    print("5. List all memories at the end")
    print("\nMake sure the LLM server is running on localhost:8000")
    print("="*80)
    
    # Create test instance
    test = TestLLMMemoryIntegration()
    
    try:
        # Run all tests
        store_results = test.test_store_prompts()
        retrieve_results = test.test_retrieve_prompts()
        mixed_results = test.test_mixed_prompts()
        conversation_results = test.test_conversational_flow()
        
        # List all memories
        all_memories = test.list_all_memories()
        
        # Print summary
        test.print_summary()
        
        # Print test statistics
        print("\n" + "="*80)
        print("TEST STATISTICS")
        print("="*80)
        print(f"Store prompts tested: {len(store_results)}")
        print(f"  Successful: {sum(1 for r in store_results if r.get('success'))}")
        print(f"  With store calls: {sum(1 for r in store_results if r.get('has_store_call'))}")
        
        print(f"\nRetrieve prompts tested: {len(retrieve_results)}")
        print(f"  Successful: {sum(1 for r in retrieve_results if r.get('success'))}")
        print(f"  With retrieve calls: {sum(1 for r in retrieve_results if r.get('has_retrieve_call'))}")
        
        print(f"\nMixed prompts tested: {len(mixed_results)}")
        print(f"  Successful: {sum(1 for r in mixed_results if r.get('success'))}")
        
        print(f"\nConversation turns: {len(conversation_results)}")
        print(f"  Successful: {sum(1 for r in conversation_results if r.get('success'))}")
        
        print(f"\nTotal memories in store: {len(all_memories)}")
        print("="*80)
        
        return True
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return False
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Optionally cleanup - comment out if you want to inspect the memory store
        # test.cleanup()
        pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

