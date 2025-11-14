#!/usr/bin/env python3
"""
Basic smoke test for memory tool.
Quick test to verify core functionality works.
"""
import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.memory_tool import memory_store, memory_retrieve, MemoryStore


def test_basic_functionality():
    """Test basic store and retrieve."""
    test_dir = tempfile.mkdtemp()
    storage_path = os.path.join(test_dir, "test_memory")
    
    try:
        # Test 1: Store
        result = memory_store("Python is a programming language.", storage_path=storage_path)
        assert result["success"], "Store should succeed"
        assert "id" in result, "Result should have id"
        
        # Test 2: Retrieve
        result = memory_retrieve("programming", storage_path=storage_path, top_k=1)
        assert result["success"], "Retrieve should succeed"
        assert len(result["results"]) > 0, "Should retrieve at least one result"
        assert "Python" in result["results"][0]["text"], "Should retrieve Python-related content"
        
        # Test 3: Similarity ordering
        memory_store("Java is a programming language.", storage_path=storage_path)
        memory_store("The weather is nice.", storage_path=storage_path)
        
        result = memory_retrieve("programming language", storage_path=storage_path, top_k=3)
        assert len(result["results"]) >= 2, "Should retrieve programming-related items"
        
        # Results should be ordered by similarity
        similarities = [r["similarity"] for r in result["results"]]
        assert similarities == sorted(similarities, reverse=True), "Results should be ordered by similarity"
        
        print("✓ All basic tests passed!")
        return True
        
    finally:
        shutil.rmtree(test_dir)


if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)

