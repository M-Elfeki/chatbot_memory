#!/usr/bin/env python3
"""
Unit tests for memory tool.
Tests store, retrieve, and similarity search functionality.
"""
import unittest
import os
import shutil
import tempfile
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.memory_tool import (
    MemoryStore,
    memory_store,
    memory_retrieve,
    memory_store_executor,
    memory_retrieve_executor,
    MemoryError
)


class TestMemoryStore(unittest.TestCase):
    """Test cases for MemoryStore class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.storage_path = os.path.join(self.test_dir, "test_memory")
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_store_basic(self):
        """Test basic storage functionality."""
        store = MemoryStore(storage_path=self.storage_path)
        
        result = store.store("Python is a programming language.")
        self.assertTrue(result["success"])
        self.assertIn("id", result)
        self.assertEqual(result["text"], "Python is a programming language.")
        self.assertIn("timestamp", result)
    
    def test_store_with_metadata(self):
        """Test storage with additional metadata."""
        store = MemoryStore(storage_path=self.storage_path)
        
        metadata = {"source": "test", "category": "programming"}
        result = store.store("Python is great.", metadata=metadata)
        
        self.assertTrue(result["success"])
        # Check that metadata is stored
        stored_item = store.metadata[result["id"]]
        self.assertEqual(stored_item["source"], "test")
        self.assertEqual(stored_item["category"], "programming")
    
    def test_store_empty_text(self):
        """Test that storing empty text raises an error."""
        store = MemoryStore(storage_path=self.storage_path)
        
        with self.assertRaises(ValueError):
            store.store("")
        
        with self.assertRaises(ValueError):
            store.store("   ")
    
    def test_retrieve_basic(self):
        """Test basic retrieval functionality."""
        store = MemoryStore(storage_path=self.storage_path)
        
        # Store some items
        store.store("The weather is lovely today.")
        store.store("It's so sunny outside!")
        store.store("He drove to the stadium.")
        
        # Retrieve similar items
        result = store.retrieve("What's the weather like?", top_k=2)
        
        self.assertTrue(result["success"])
        self.assertGreaterEqual(len(result["results"]), 1)
        self.assertEqual(result["query"], "What's the weather like?")
        
        # Check result structure
        for item in result["results"]:
            self.assertIn("text", item)
            self.assertIn("similarity", item)
            self.assertIn("distance", item)
            self.assertGreaterEqual(item["similarity"], 0)
            self.assertLessEqual(item["similarity"], 1)
    
    def test_retrieve_empty_store(self):
        """Test retrieval from empty store."""
        store = MemoryStore(storage_path=self.storage_path)
        
        result = store.retrieve("test query")
        self.assertTrue(result["success"])
        self.assertEqual(len(result["results"]), 0)
        self.assertIn("message", result)
    
    def test_retrieve_empty_query(self):
        """Test that retrieving with empty query raises an error."""
        store = MemoryStore(storage_path=self.storage_path)
        store.store("Some text")
        
        with self.assertRaises(ValueError):
            store.retrieve("")
        
        with self.assertRaises(ValueError):
            store.retrieve("   ")
    
    def test_retrieve_top_k(self):
        """Test that top_k parameter works correctly."""
        store = MemoryStore(storage_path=self.storage_path)
        
        # Store multiple items
        texts = [
            "Python programming",
            "Java programming",
            "C++ programming",
            "JavaScript programming",
            "Ruby programming"
        ]
        for text in texts:
            store.store(text)
        
        # Retrieve with different top_k values
        result1 = store.retrieve("programming language", top_k=2)
        self.assertEqual(len(result1["results"]), 2)
        
        result2 = store.retrieve("programming language", top_k=5)
        self.assertEqual(len(result2["results"]), 5)
        
        result3 = store.retrieve("programming language", top_k=10)
        self.assertLessEqual(len(result3["results"]), 5)  # Can't return more than stored
    
    def test_retrieve_min_similarity(self):
        """Test minimum similarity filtering."""
        store = MemoryStore(storage_path=self.storage_path)
        
        # Store items with varying similarity
        store.store("The weather is lovely today.")
        store.store("It's so sunny outside!")
        store.store("He drove to the stadium.")  # Less similar to weather query
        
        # Retrieve without filter
        result_all = store.retrieve("What's the weather like?", top_k=5)
        
        # Retrieve with high similarity threshold
        result_filtered = store.retrieve(
            "What's the weather like?", 
            top_k=5, 
            min_similarity=0.5
        )
        
        self.assertLessEqual(len(result_filtered["results"]), len(result_all["results"]))
        
        # All filtered results should meet similarity threshold
        for item in result_filtered["results"]:
            self.assertGreaterEqual(item["similarity"], 0.5)
    
    def test_similarity_ordering(self):
        """Test that results are ordered by similarity."""
        store = MemoryStore(storage_path=self.storage_path)
        
        # Store items
        store.store("The weather is lovely today.")
        store.store("It's so sunny outside!")
        store.store("He drove to the stadium.")
        
        # Retrieve
        result = store.retrieve("What's the weather like?", top_k=3)
        
        # Check that results are ordered by similarity (descending)
        similarities = [item["similarity"] for item in result["results"]]
        self.assertEqual(similarities, sorted(similarities, reverse=True))
    
    def test_persistence(self):
        """Test that stored items persist across store instances."""
        # Create first store and add items
        store1 = MemoryStore(storage_path=self.storage_path)
        store1.store("Python is a programming language.")
        store1.store("Machine learning uses neural networks.")
        
        # Create new store instance (should load from disk)
        store2 = MemoryStore(storage_path=self.storage_path)
        
        # Verify items are loaded
        self.assertEqual(len(store2.metadata), 2)
        
        # Verify we can retrieve them
        result = store2.retrieve("programming", top_k=5)
        self.assertEqual(len(result["results"]), 1)
        self.assertIn("Python", result["results"][0]["text"])
    
    def test_clear(self):
        """Test clearing the memory store."""
        store = MemoryStore(storage_path=self.storage_path)
        
        # Store some items
        store.store("Item 1")
        store.store("Item 2")
        self.assertEqual(len(store.metadata), 2)
        
        # Clear
        result = store.clear()
        self.assertTrue(result["success"])
        self.assertEqual(len(store.metadata), 0)
        
        # Verify persistence
        store2 = MemoryStore(storage_path=self.storage_path)
        self.assertEqual(len(store2.metadata), 0)
    
    def test_get_stats(self):
        """Test getting statistics."""
        store = MemoryStore(storage_path=self.storage_path)
        
        store.store("Test item 1")
        store.store("Test item 2")
        
        stats = store.get_stats()
        self.assertTrue(stats["success"])
        self.assertEqual(stats["total_items"], 2)
        self.assertIn("embedding_dim", stats)
        self.assertIn("model_name", stats)
        self.assertEqual(stats["model_name"], "Qwen/Qwen3-Embedding-8B")


class TestMemoryFunctions(unittest.TestCase):
    """Test cases for module-level functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.storage_path = os.path.join(self.test_dir, "test_memory")
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_memory_store_function(self):
        """Test memory_store function."""
        result = memory_store("Test text", storage_path=self.storage_path)
        self.assertTrue(result["success"])
        self.assertIn("id", result)
    
    def test_memory_retrieve_function(self):
        """Test memory_retrieve function."""
        # Store first
        memory_store("Python programming", storage_path=self.storage_path)
        
        # Retrieve
        result = memory_retrieve("programming language", storage_path=self.storage_path)
        self.assertTrue(result["success"])
        self.assertGreaterEqual(len(result["results"]), 1)
    
    def test_memory_store_executor(self):
        """Test memory_store_executor."""
        # Valid call
        result_str = memory_store_executor(
            "memory_store",
            {"text": "Test text"}
        )
        self.assertIn("success", result_str)
        
        # Invalid tool name
        result_str = memory_store_executor(
            "invalid_tool",
            {"text": "Test"}
        )
        self.assertIn("error", result_str)
        
        # Missing text
        result_str = memory_store_executor(
            "memory_store",
            {}
        )
        self.assertIn("error", result_str)
    
    def test_memory_retrieve_executor(self):
        """Test memory_retrieve_executor."""
        # Store first
        memory_store("Test text", storage_path=self.storage_path)
        
        # Valid call
        result_str = memory_retrieve_executor(
            "memory_retrieve",
            {"query": "test"}
        )
        self.assertIn("success", result_str)
        
        # Invalid tool name
        result_str = memory_retrieve_executor(
            "invalid_tool",
            {"query": "test"}
        )
        self.assertIn("error", result_str)
        
        # Missing query
        result_str = memory_retrieve_executor(
            "memory_retrieve",
            {}
        )
        self.assertIn("error", result_str)


class TestSemanticSimilarity(unittest.TestCase):
    """Test cases for semantic similarity functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.storage_path = os.path.join(self.test_dir, "test_memory")
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_similar_phrases(self):
        """Test that semantically similar phrases are retrieved."""
        store = MemoryStore(storage_path=self.storage_path)
        
        # Store similar phrases
        store.store("The weather is lovely today.")
        store.store("It's so sunny outside!")
        store.store("He drove to the stadium.")
        
        # Query for weather-related content
        result = store.retrieve("What's the weather like?", top_k=3)
        
        # Should retrieve weather-related items with higher similarity
        weather_items = [
            item for item in result["results"]
            if "weather" in item["text"].lower() or "sunny" in item["text"].lower()
        ]
        self.assertGreater(len(weather_items), 0)
        
        # Weather items should have higher similarity than unrelated items
        if len(result["results"]) > 1:
            weather_similarities = [
                item["similarity"] for item in result["results"]
                if "weather" in item["text"].lower() or "sunny" in item["text"].lower()
            ]
            stadium_similarities = [
                item["similarity"] for item in result["results"]
                if "stadium" in item["text"].lower()
            ]
            if weather_similarities and stadium_similarities:
                self.assertGreater(
                    max(weather_similarities),
                    max(stadium_similarities)
                )
    
    def test_different_topics(self):
        """Test that different topics are distinguished."""
        store = MemoryStore(storage_path=self.storage_path)
        
        # Store items from different topics
        store.store("Python is a programming language.")
        store.store("The weather is nice today.")
        store.store("Machine learning uses neural networks.")
        
        # Query for programming
        result = store.retrieve("programming language", top_k=3)
        
        # Should prioritize programming-related items
        programming_items = [
            item for item in result["results"]
            if "Python" in item["text"] or "programming" in item["text"].lower()
        ]
        self.assertGreater(len(programming_items), 0)
        
        # Top result should be programming-related
        if result["results"]:
            top_result = result["results"][0]
            self.assertIn("Python", top_result["text"])


if __name__ == "__main__":
    unittest.main()

