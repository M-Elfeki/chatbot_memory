#!/usr/bin/env python3
"""
Integration tests for memory tool vector store operations.
Tests end-to-end workflows and robustness.
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
    MemoryStore,
    memory_store,
    memory_retrieve,
    memory_executor
)


class TestVectorStoreIntegration(unittest.TestCase):
    """Integration tests for vector store operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.storage_path = os.path.join(self.test_dir, "test_memory")
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_store_and_retrieve_workflow(self):
        """Test complete store and retrieve workflow."""
        store = MemoryStore(storage_path=self.storage_path)
        
        # Store multiple items
        items = [
            "Python is a high-level programming language.",
            "Java is an object-oriented programming language.",
            "The weather forecast predicts rain tomorrow.",
            "Machine learning algorithms learn from data.",
            "The user prefers dark mode for the interface."
        ]
        
        stored_ids = []
        for item in items:
            result = store.store(item)
            self.assertTrue(result["success"])
            stored_ids.append(result["id"])
        
        # Verify all items stored
        self.assertEqual(len(store.metadata), len(items))
        
        # Retrieve programming-related items
        result = store.retrieve("programming languages", top_k=3)
        self.assertTrue(result["success"])
        self.assertGreaterEqual(len(result["results"]), 2)
        
        # Verify retrieved items are programming-related
        for item in result["results"]:
            text = item["text"].lower()
            self.assertTrue(
                "python" in text or "java" in text or "programming" in text
            )
    
    def test_persistence_across_sessions(self):
        """Test that data persists across multiple store instances."""
        # First session: store data
        store1 = MemoryStore(storage_path=self.storage_path)
        store1.store("Session 1: Python is great.")
        store1.store("Session 1: Machine learning is fascinating.")
        
        # Second session: load and add more data
        store2 = MemoryStore(storage_path=self.storage_path)
        self.assertEqual(len(store2.metadata), 2)
        store2.store("Session 2: JavaScript is versatile.")
        
        # Third session: verify all data
        store3 = MemoryStore(storage_path=self.storage_path)
        self.assertEqual(len(store3.metadata), 3)
        
        # Verify we can retrieve from all sessions
        result = store3.retrieve("programming", top_k=5)
        self.assertGreaterEqual(len(result["results"]), 2)
    
    def test_large_scale_storage(self):
        """Test storing and retrieving large numbers of items."""
        store = MemoryStore(storage_path=self.storage_path)
        
        # Store 50 items
        for i in range(50):
            store.store(f"Item {i}: This is test content number {i}.")
        
        self.assertEqual(len(store.metadata), 50)
        
        # Retrieve top 10
        result = store.retrieve("test content", top_k=10)
        self.assertEqual(len(result["results"]), 10)
        
        # Verify all results are relevant
        for item in result["results"]:
            self.assertIn("test content", item["text"].lower())
    
    def test_similarity_threshold_filtering(self):
        """Test that similarity threshold filtering works correctly."""
        store = MemoryStore(storage_path=self.storage_path)
        
        # Store items with varying similarity
        store.store("Python is a programming language.")
        store.store("Java is a programming language.")
        store.store("The weather is nice today.")
        store.store("It's sunny outside.")
        
        # Retrieve without threshold
        result_all = store.retrieve("programming", top_k=10)
        
        # Retrieve with threshold
        result_filtered = store.retrieve(
            "programming",
            top_k=10,
            min_similarity=0.3
        )
        
        # Filtered results should be subset of all results
        self.assertLessEqual(len(result_filtered["results"]), len(result_all["results"]))
        
        # All filtered results should meet threshold
        for item in result_filtered["results"]:
            self.assertGreaterEqual(item["similarity"], 0.3)
    
    def test_concurrent_operations(self):
        """Test that multiple operations work correctly."""
        store = MemoryStore(storage_path=self.storage_path)
        
        # Store items
        for i in range(10):
            store.store(f"Content {i}")
        
        # Multiple retrievals
        results = []
        queries = ["content", "test", "data"]
        for query in queries:
            result = store.retrieve(query, top_k=5)
            results.append(result)
            self.assertTrue(result["success"])
        
        # Verify all retrievals worked
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertGreaterEqual(len(result["results"]), 0)
    
    def test_metadata_persistence(self):
        """Test that metadata persists correctly."""
        store = MemoryStore(storage_path=self.storage_path)
        
        # Store with metadata
        metadata1 = {"source": "test", "category": "A"}
        metadata2 = {"source": "test", "category": "B"}
        
        result1 = store.store("Item 1", metadata=metadata1)
        result2 = store.store("Item 2", metadata=metadata2)
        
        # Reload store
        store2 = MemoryStore(storage_path=self.storage_path)
        
        # Verify metadata persisted
        item1 = store2.metadata[result1["id"]]
        item2 = store2.metadata[result2["id"]]
        
        self.assertEqual(item1["category"], "A")
        self.assertEqual(item2["category"], "B")
        self.assertEqual(item1["source"], "test")
        self.assertEqual(item2["source"], "test")
    
    def test_clear_and_rebuild(self):
        """Test clearing and rebuilding the store."""
        store = MemoryStore(storage_path=self.storage_path)
        
        # Store items
        store.store("Item 1")
        store.store("Item 2")
        self.assertEqual(len(store.metadata), 2)
        
        # Clear
        store.clear()
        self.assertEqual(len(store.metadata), 0)
        
        # Rebuild
        store.store("New Item 1")
        store.store("New Item 2")
        self.assertEqual(len(store.metadata), 2)
        
        # Verify old items are gone
        result = store.retrieve("Item", top_k=5)
        self.assertEqual(len(result["results"]), 0)
        
        # Verify new items are present
        result = store.retrieve("New", top_k=5)
        self.assertEqual(len(result["results"]), 2)


class TestMemoryExecutorIntegration(unittest.TestCase):
    """Integration tests for memory executor functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.storage_path = os.path.join(self.test_dir, "test_memory")
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_memory_executor_store(self):
        """Test memory executor for store operations."""
        # Store via executor
        result_str = memory_executor(
            "memory_store",
            {"text": "Test content", "storage_path": self.storage_path}
        )
        
        result = json.loads(result_str)
        self.assertTrue(result["success"])
        self.assertIn("id", result)
    
    def test_memory_executor_retrieve(self):
        """Test memory executor for retrieve operations."""
        # Store first
        memory_store("Test content", storage_path=self.storage_path)
        
        # Retrieve via executor
        result_str = memory_executor(
            "memory_retrieve",
            {"query": "test", "top_k": 5, "storage_path": self.storage_path}
        )
        
        result = json.loads(result_str)
        self.assertTrue(result["success"])
        self.assertGreaterEqual(len(result["results"]), 1)
    
    def test_memory_executor_invalid_tool(self):
        """Test memory executor with invalid tool name."""
        result_str = memory_executor("invalid_tool", {})
        result = json.loads(result_str)
        self.assertIn("error", result)
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow using executors."""
        # Store multiple items
        items = [
            "Python programming",
            "Java programming",
            "Weather forecast"
        ]
        
        for item in items:
            result_str = memory_executor(
                "memory_store",
                {"text": item, "storage_path": self.storage_path}
            )
            result = json.loads(result_str)
            self.assertTrue(result["success"])
        
        # Retrieve
        result_str = memory_executor(
            "memory_retrieve",
            {"query": "programming", "top_k": 5, "storage_path": self.storage_path}
        )
        
        result = json.loads(result_str)
        self.assertTrue(result["success"])
        self.assertGreaterEqual(len(result["results"]), 2)
        
        # Verify results are programming-related
        for item in result["results"]:
            text = item["text"].lower()
            self.assertTrue("python" in text or "java" in text)


class TestRobustness(unittest.TestCase):
    """Test robustness and error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.storage_path = os.path.join(self.test_dir, "test_memory")
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_special_characters(self):
        """Test handling of special characters."""
        store = MemoryStore(storage_path=self.storage_path)
        
        special_texts = [
            "Text with 'quotes' and \"double quotes\"",
            "Text with\nnewlines\tand\ttabs",
            "Text with unicode: 你好世界 🌍",
            "Text with symbols: @#$%^&*()",
        ]
        
        for text in special_texts:
            result = store.store(text)
            self.assertTrue(result["success"])
        
        # Retrieve
        result = store.retrieve("quotes", top_k=5)
        self.assertTrue(result["success"])
    
    def test_long_text(self):
        """Test handling of long text."""
        store = MemoryStore(storage_path=self.storage_path)
        
        long_text = " ".join(["word"] * 1000)  # 1000 words
        result = store.store(long_text)
        self.assertTrue(result["success"])
        
        # Retrieve
        result = store.retrieve("word", top_k=5)
        self.assertTrue(result["success"])
        self.assertGreaterEqual(len(result["results"]), 1)
    
    def test_empty_retrieval_after_clear(self):
        """Test retrieval after clearing store."""
        store = MemoryStore(storage_path=self.storage_path)
        
        # Store and clear
        store.store("Test")
        store.clear()
        
        # Retrieve should return empty
        result = store.retrieve("test")
        self.assertTrue(result["success"])
        self.assertEqual(len(result["results"]), 0)


if __name__ == "__main__":
    unittest.main()

