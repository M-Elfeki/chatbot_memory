#!/usr/bin/env python3
"""
Integration tests for LLM + memory tool interaction.
Tests that the LLM can reliably use the memory tool to store and retrieve information.

Note: These tests require the GPT-OSS server to be running.
Set SKIP_LLM_TESTS=1 to skip these tests if server is not available.
"""
import unittest
import os
import shutil
import tempfile
import sys
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.memory_tool import (
    MEMORY_STORE_TOOL,
    MEMORY_RETRIEVE_TOOL,
    memory_executor,
    MemoryStore
)


# Check if LLM tests should be skipped
SKIP_LLM_TESTS = os.environ.get("SKIP_LLM_TESTS", "0") == "1"


@unittest.skipIf(SKIP_LLM_TESTS, "Skipping LLM integration tests (set SKIP_LLM_TESTS=0 to enable)")
class TestLLMMemoryIntegration(unittest.TestCase):
    """Integration tests for LLM + memory tool."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.storage_path = os.path.join(self.test_dir, "test_memory")
        
        # Try to import client to verify server availability
        try:
            from client import generate_with_tools
            self.client_available = True
        except Exception as e:
            self.client_available = False
            self.skip_reason = f"Client not available: {e}"
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_llm_store_information(self):
        """Test that LLM can store information using memory tool."""
        if not self.client_available:
            self.skipTest(self.skip_reason)
        
        from client import generate_with_tools
        
        # Custom tool executor
        def tool_executor(tool_name, arguments):
            arguments["storage_path"] = self.storage_path
            return memory_executor(tool_name, arguments)
        
        # Prompt LLM to store information
        prompt = "Please remember that I prefer Python programming language over Java."
        
        result = generate_with_tools(
            prompt=prompt,
            tools=[MEMORY_STORE_TOOL, MEMORY_RETRIEVE_TOOL],
            tool_executor=tool_executor,
            tool_choice="auto",
            temperature=0.7,
            top_p=0.95,
            reasoning_level="low",
            max_turns=5
        )
        
        # Verify tool was called
        tool_calls = result.get("tool_calls", [])
        self.assertGreater(len(tool_calls), 0, "LLM should have called memory_store tool")
        
        # Verify memory_store was called
        store_calls = [call for call in tool_calls if call.get("name") == "memory_store"]
        self.assertGreater(len(store_calls), 0, "LLM should have called memory_store")
        
        # Verify information was stored
        store = MemoryStore(storage_path=self.storage_path)
        self.assertGreater(len(store.metadata), 0, "Information should be stored")
        
        # Verify stored content is relevant
        stored_texts = [item["text"] for item in store.metadata]
        relevant_texts = [text for text in stored_texts if "python" in text.lower() or "java" in text.lower()]
        self.assertGreater(len(relevant_texts), 0, "Stored text should be relevant")
    
    def test_llm_retrieve_information(self):
        """Test that LLM can retrieve information using memory tool."""
        if not self.client_available:
            self.skipTest(self.skip_reason)
        
        from client import generate_with_tools
        
        # Pre-populate memory
        store = MemoryStore(storage_path=self.storage_path)
        store.store("The user prefers Python programming language.")
        store.store("The user likes machine learning.")
        
        # Custom tool executor
        def tool_executor(tool_name, arguments):
            arguments["storage_path"] = self.storage_path
            return memory_executor(tool_name, arguments)
        
        # Prompt LLM to retrieve information
        prompt = "What programming language do I prefer?"
        
        result = generate_with_tools(
            prompt=prompt,
            tools=[MEMORY_STORE_TOOL, MEMORY_RETRIEVE_TOOL],
            tool_executor=tool_executor,
            tool_choice="auto",
            temperature=0.7,
            top_p=0.95,
            reasoning_level="low",
            max_turns=5
        )
        
        # Verify tool was called
        tool_calls = result.get("tool_calls", [])
        self.assertGreater(len(tool_calls), 0, "LLM should have called memory_retrieve tool")
        
        # Verify memory_retrieve was called
        retrieve_calls = [call for call in tool_calls if call.get("name") == "memory_retrieve"]
        self.assertGreater(len(retrieve_calls), 0, "LLM should have called memory_retrieve")
        
        # Verify response mentions Python
        content = result.get("content", "").lower()
        self.assertIn("python", content, "Response should mention Python")
    
    def test_llm_store_and_retrieve_workflow(self):
        """Test complete workflow: LLM stores then retrieves information."""
        if not self.client_available:
            self.skipTest(self.skip_reason)
        
        from client import generate_with_tools
        
        # Custom tool executor
        def tool_executor(tool_name, arguments):
            arguments["storage_path"] = self.storage_path
            return memory_executor(tool_name, arguments)
        
        # Step 1: Store information
        prompt1 = "Remember that my favorite color is blue."
        
        result1 = generate_with_tools(
            prompt=prompt1,
            tools=[MEMORY_STORE_TOOL, MEMORY_RETRIEVE_TOOL],
            tool_executor=tool_executor,
            tool_choice="auto",
            temperature=0.7,
            top_p=0.95,
            reasoning_level="low",
            max_turns=5
        )
        
        # Verify storage
        tool_calls1 = result1.get("tool_calls", [])
        store_calls1 = [call for call in tool_calls1 if call.get("name") == "memory_store"]
        self.assertGreater(len(store_calls1), 0, "Should store information")
        
        # Step 2: Retrieve information
        prompt2 = "What is my favorite color?"
        
        result2 = generate_with_tools(
            prompt=prompt2,
            tools=[MEMORY_STORE_TOOL, MEMORY_RETRIEVE_TOOL],
            tool_executor=tool_executor,
            tool_choice="auto",
            temperature=0.7,
            top_p=0.95,
            reasoning_level="low",
            max_turns=5
        )
        
        # Verify retrieval
        tool_calls2 = result2.get("tool_calls", [])
        retrieve_calls2 = [call for call in tool_calls2 if call.get("name") == "memory_retrieve"]
        self.assertGreater(len(retrieve_calls2), 0, "Should retrieve information")
        
        # Verify response mentions blue
        content2 = result2.get("content", "").lower()
        self.assertIn("blue", content2, "Response should mention blue")
    
    def test_llm_semantic_retrieval(self):
        """Test that LLM can retrieve semantically similar information."""
        if not self.client_available:
            self.skipTest(self.skip_reason)
        
        from client import generate_with_tools
        
        # Pre-populate memory with semantically similar items
        store = MemoryStore(storage_path=self.storage_path)
        store.store("The weather is lovely today.")
        store.store("It's so sunny outside!")
        store.store("He drove to the stadium.")
        
        # Custom tool executor
        def tool_executor(tool_name, arguments):
            arguments["storage_path"] = self.storage_path
            return memory_executor(tool_name, arguments)
        
        # Query with semantically similar but different wording
        prompt = "What's the weather like?"
        
        result = generate_with_tools(
            prompt=prompt,
            tools=[MEMORY_STORE_TOOL, MEMORY_RETRIEVE_TOOL],
            tool_executor=tool_executor,
            tool_choice="auto",
            temperature=0.7,
            top_p=0.95,
            reasoning_level="low",
            max_turns=5
        )
        
        # Verify retrieval was called
        tool_calls = result.get("tool_calls", [])
        retrieve_calls = [call for call in tool_calls if call.get("name") == "memory_retrieve"]
        self.assertGreater(len(retrieve_calls), 0, "Should retrieve information")
        
        # Verify response mentions weather-related content
        content = result.get("content", "").lower()
        weather_keywords = ["weather", "sunny", "lovely"]
        has_weather_content = any(keyword in content for keyword in weather_keywords)
        self.assertTrue(has_weather_content, "Response should mention weather-related content")


class TestMemoryToolReliability(unittest.TestCase):
    """Test reliability of memory tool operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.storage_path = os.path.join(self.test_dir, "test_memory")
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_reliable_store_and_retrieve(self):
        """Test that store and retrieve operations are reliable."""
        store = MemoryStore(storage_path=self.storage_path)
        
        # Store multiple items
        items = [
            "Python is a programming language.",
            "Java is a programming language.",
            "The weather is nice."
        ]
        
        stored_ids = []
        for item in items:
            result = store.store(item)
            self.assertTrue(result["success"])
            stored_ids.append(result["id"])
        
        # Verify all items stored
        self.assertEqual(len(store.metadata), len(items))
        
        # Retrieve each item
        for i, item in enumerate(items):
            result = store.retrieve(item, top_k=1)
            self.assertTrue(result["success"])
            self.assertGreaterEqual(len(result["results"]), 1)
            # Top result should be the same item
            self.assertEqual(result["results"][0]["text"], item)
    
    def test_consistent_similarity_scores(self):
        """Test that similarity scores are consistent."""
        store = MemoryStore(storage_path=self.storage_path)
        
        # Store items
        store.store("Python programming")
        store.store("Java programming")
        
        # Multiple retrievals should give consistent results
        results1 = store.retrieve("programming language", top_k=2)
        results2 = store.retrieve("programming language", top_k=2)
        
        # Results should be consistent
        self.assertEqual(len(results1["results"]), len(results2["results"]))
        
        # Similarity scores should be the same
        for r1, r2 in zip(results1["results"], results2["results"]):
            self.assertAlmostEqual(r1["similarity"], r2["similarity"], places=5)


if __name__ == "__main__":
    unittest.main()

