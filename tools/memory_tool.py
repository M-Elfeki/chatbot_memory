#!/usr/bin/env python3
"""
Semantic memory tool for LLM with robust storage and retrieval.
Uses embedding API server (Qwen/Qwen3-Embedding-8B) for embeddings and FAISS for vector storage.
All embedding operations are performed via HTTP API to minimize query-time latency.
"""
import json
import os
import pickle
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
import requests

try:
    import faiss
except ImportError as e:
    raise ImportError(
        f"Required packages not installed: {e}\n"
        "Please install: pip install faiss-cpu requests"
    )

# Embedding API server configuration
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "http://localhost:8001")
EMBEDDING_API_TIMEOUT = 30  # seconds


class MemoryError(Exception):
    """Custom exception for memory operations."""
    pass


class MemoryStore:
    """
    Semantic memory store using embedding API server and FAISS.
    Stores and retrieves information based on semantic similarity.
    All embeddings are generated via HTTP API to avoid model loading latency.
    """
    
    def __init__(
        self,
        storage_path: str = "memory_store",
        embedding_api_url: str = EMBEDDING_API_URL,
        embedding_dim: Optional[int] = None
    ):
        """
        Initialize the memory store.
        
        Args:
            storage_path: Directory path for storing the FAISS index and metadata
            embedding_api_url: URL of the embedding API server (default: http://localhost:8001)
            embedding_dim: Embedding dimension (auto-detected from API if None)
        """
        self.storage_path = storage_path
        self.embedding_api_url = embedding_api_url.rstrip('/')
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_path, exist_ok=True)
        
        # Get embedding dimension from API if not provided
        if embedding_dim is None:
            try:
                self.embedding_dim = self._get_embedding_dimension()
            except Exception as e:
                # Fallback: use Qwen3-Embedding-8B default dimension
                self.embedding_dim = 1024  # Qwen3-Embedding-8B default
                print(f"Warning: Could not get embedding dimension from API: {e}")
                print(f"Using default dimension: {self.embedding_dim}")
        else:
            self.embedding_dim = embedding_dim
        
        # Initialize FAISS index
        self.index = None
        self.metadata = []  # List of dicts with 'text', 'timestamp', 'id'
        self.next_id = 0
        
        # Load existing index if available
        self._load_index()
    
    def _get_embedding_dimension(self) -> int:
        """Get embedding dimension from API server."""
        try:
            response = requests.get(
                f"{self.embedding_api_url}/models",
                timeout=5
            )
            response.raise_for_status()
            models_info = response.json()
            
            # Qwen model info
            qwen_info = models_info.get("qwen", {})
            if qwen_info.get("loaded"):
                return qwen_info["dimension"]
            else:
                raise MemoryError("Qwen model not loaded on server")
        except requests.exceptions.RequestException as e:
            raise MemoryError(f"Failed to connect to embedding API: {e}")
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings using the embedding API server.
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            numpy array of embeddings (shape: [len(texts), embedding_dim])
        """
        try:
            response = requests.post(
                f"{self.embedding_api_url}/embed",
                json={
                    "texts": texts,
                    "normalize": True,
                    "batch_size": 32
                },
                timeout=EMBEDDING_API_TIMEOUT
            )
            response.raise_for_status()
            result = response.json()
            
            embeddings = np.array(result["embeddings"], dtype=np.float32)
            return embeddings
        except requests.exceptions.RequestException as e:
            raise MemoryError(f"Failed to generate embeddings via API: {e}")
    
    def _load_index(self):
        """Load existing FAISS index and metadata from disk."""
        index_path = os.path.join(self.storage_path, "index.faiss")
        metadata_path = os.path.join(self.storage_path, "metadata.pkl")
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            try:
                # Load FAISS index
                self.index = faiss.read_index(index_path)
                
                # Load metadata
                with open(metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                
                # Set next_id based on existing metadata
                if self.metadata:
                    max_id = max(item.get('id', 0) for item in self.metadata)
                    self.next_id = max_id + 1
                else:
                    self.next_id = 0
                
                print(f"Loaded existing index with {len(self.metadata)} entries")
            except Exception as e:
                print(f"Warning: Failed to load existing index: {e}")
                print("Creating new index...")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index."""
        # Use L2 distance (Euclidean) - FAISS default
        # Inner product is also common for normalized embeddings
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.metadata = []
        self.next_id = 0
    
    def _save_index(self):
        """Save FAISS index and metadata to disk."""
        index_path = os.path.join(self.storage_path, "index.faiss")
        metadata_path = os.path.join(self.storage_path, "metadata.pkl")
        
        try:
            # Save FAISS index
            faiss.write_index(self.index, index_path)
            
            # Save metadata
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            raise MemoryError(f"Failed to save index: {e}")
    
    def store(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Store a piece of information in the memory.
        
        Args:
            text: The text content to store
            metadata: Optional additional metadata to store with the text
        
        Returns:
            Dictionary with success status and stored item info
        """
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            raise ValueError("Text cannot be empty")
        
        # Generate embedding via API
        try:
            embeddings = self._generate_embeddings([text])
            embedding = embeddings[0].reshape(1, -1)
        except Exception as e:
            raise MemoryError(f"Failed to generate embedding: {e}")
        
        # Create metadata entry
        item_id = self.next_id
        self.next_id += 1
        
        item_metadata = {
            'id': item_id,
            'text': text,
            'timestamp': datetime.now().isoformat(),
            **(metadata or {})
        }
        
        # Add to FAISS index
        self.index.add(embedding)
        
        # Store metadata
        self.metadata.append(item_metadata)
        
        # Save to disk
        try:
            self._save_index()
        except Exception as e:
            # Rollback if save fails
            self.index.remove_ids(np.array([len(self.metadata) - 1]))
            self.metadata.pop()
            self.next_id -= 1
            raise MemoryError(f"Failed to save: {e}")
        
        return {
            "success": True,
            "id": item_id,
            "text": text,
            "timestamp": item_metadata['timestamp']
        }
    
    def retrieve(self, query: str, top_k: int = 5, min_similarity: Optional[float] = None) -> Dict[str, Any]:
        """
        Retrieve information from memory based on semantic similarity.
        
        Args:
            query: The query text to search for
            top_k: Number of results to return (default: 5)
            min_similarity: Optional minimum similarity threshold (0-1)
                           If None, returns top_k results regardless of similarity
        
        Returns:
            Dictionary with success status and list of retrieved items
        """
        if not query or not isinstance(query, str) or len(query.strip()) == 0:
            raise ValueError("Query cannot be empty")
        
        if len(self.metadata) == 0:
            return {
                "success": True,
                "results": [],
                "query": query,
                "message": "Memory store is empty"
            }
        
        # Generate query embedding via API
        try:
            embeddings = self._generate_embeddings([query])
            query_embedding = embeddings[0].reshape(1, -1)
        except Exception as e:
            raise MemoryError(f"Failed to generate query embedding: {e}")
        
        # Search in FAISS index
        # top_k should not exceed number of items
        k = min(top_k, len(self.metadata))
        
        try:
            distances, indices = self.index.search(query_embedding, k)
        except Exception as e:
            raise MemoryError(f"Failed to search index: {e}")
        
        # Convert distances to similarities (for L2: similarity = 1 / (1 + distance))
        # For normalized embeddings with L2, we can also use cosine similarity
        # Since we normalized embeddings, L2 distance correlates with cosine distance
        # Cosine similarity = 1 - (L2_distance^2 / 2) for normalized vectors
        similarities = 1 - (distances[0] ** 2 / 2)
        
        # Build results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(self.metadata):
                continue
            
            similarity = similarities[i]
            
            # Apply minimum similarity filter if specified
            if min_similarity is not None and similarity < min_similarity:
                continue
            
            item = self.metadata[idx].copy()
            item['similarity'] = float(similarity)
            item['distance'] = float(distance)
            results.append(item)
        
        return {
            "success": True,
            "results": results,
            "query": query,
            "count": len(results)
        }
    
    def clear(self) -> Dict[str, Any]:
        """
        Clear all stored memories.
        
        Returns:
            Dictionary with success status
        """
        self._create_new_index()
        try:
            self._save_index()
        except Exception as e:
            raise MemoryError(f"Failed to clear memory: {e}")
        
        return {
            "success": True,
            "message": "Memory store cleared"
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory store.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "success": True,
            "total_items": len(self.metadata),
            "embedding_dim": self.embedding_dim,
            "model_name": "Qwen/Qwen3-Embedding-8B",
            "embedding_api_url": self.embedding_api_url,
            "storage_path": self.storage_path
        }


# Global memory store instances (keyed by storage_path)
_memory_stores: Dict[str, MemoryStore] = {}


def get_memory_store(
    storage_path: str = "memory_store",
    embedding_api_url: str = EMBEDDING_API_URL
) -> MemoryStore:
    """
    Get or create a memory store instance for the given storage path.
    Uses embedding API server - no local model loading.
    
    Args:
        storage_path: Directory path for storing the memory
        embedding_api_url: URL of the embedding API server
    
    Returns:
        MemoryStore instance
    """
    global _memory_stores
    
    if storage_path not in _memory_stores:
        _memory_stores[storage_path] = MemoryStore(
            storage_path=storage_path,
            embedding_api_url=embedding_api_url
        )
    return _memory_stores[storage_path]


def memory_store(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    storage_path: str = "memory_store",
    embedding_api_url: str = EMBEDDING_API_URL
) -> Dict[str, Any]:
    """
    Store information in semantic memory.
    Uses embedding API server - no local model loading.
    
    Args:
        text: The text content to store
        metadata: Optional additional metadata
        storage_path: Directory path for storing the memory
        embedding_api_url: URL of the embedding API server
    
    Returns:
        Dictionary with success status and stored item info
    """
    store = get_memory_store(storage_path, embedding_api_url=embedding_api_url)
    return store.store(text, metadata)


def memory_retrieve(
    query: str,
    top_k: int = 5,
    min_similarity: Optional[float] = None,
    storage_path: str = "memory_store",
    embedding_api_url: str = EMBEDDING_API_URL
) -> Dict[str, Any]:
    """
    Retrieve information from semantic memory based on similarity.
    Uses embedding API server - no local model loading.
    
    Args:
        query: The query text to search for
        top_k: Number of results to return (default: 5)
        min_similarity: Optional minimum similarity threshold (0-1)
        storage_path: Directory path for storing the memory
        embedding_api_url: URL of the embedding API server
    
    Returns:
        Dictionary with success status and list of retrieved items
    """
    store = get_memory_store(storage_path, embedding_api_url=embedding_api_url)
    return store.retrieve(query, top_k, min_similarity)


def memory_store_executor(tool_name: str, arguments: Dict[str, Any]) -> str:
    """
    Tool executor function for memory_store tool.
    
    Args:
        tool_name: Should be "memory_store"
        arguments: Dictionary with 'text' key and optional 'metadata'
    
    Returns:
        JSON string with result
    """
    if tool_name != "memory_store":
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    
    if not isinstance(arguments, dict):
        return json.dumps({"error": "Invalid arguments: must be a dictionary"})
    
    text = arguments.get("text", "")
    if not text or not isinstance(text, str):
        return json.dumps({"error": "Missing required parameter 'text'"})
    
    metadata = arguments.get("metadata")
    storage_path = arguments.get("storage_path", "memory_store")
    
    try:
        result = memory_store(text, metadata, storage_path)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


def memory_retrieve_executor(tool_name: str, arguments: Dict[str, Any]) -> str:
    """
    Tool executor function for memory_retrieve tool.
    
    Args:
        tool_name: Should be "memory_retrieve"
        arguments: Dictionary with 'query' key and optional 'top_k', 'min_similarity'
    
    Returns:
        JSON string with result
    """
    if tool_name != "memory_retrieve":
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    
    if not isinstance(arguments, dict):
        return json.dumps({"error": "Invalid arguments: must be a dictionary"})
    
    query = arguments.get("query", "")
    if not query or not isinstance(query, str):
        return json.dumps({"error": "Missing required parameter 'query'"})
    
    top_k = arguments.get("top_k", 5)
    if not isinstance(top_k, int) or top_k < 1:
        top_k = 5
    
    min_similarity = arguments.get("min_similarity")
    if min_similarity is not None:
        try:
            min_similarity = float(min_similarity)
            if min_similarity < 0 or min_similarity > 1:
                min_similarity = None
        except (ValueError, TypeError):
            min_similarity = None
    
    storage_path = arguments.get("storage_path", "memory_store")
    
    try:
        result = memory_retrieve(query, top_k, min_similarity, storage_path)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


def memory_executor(tool_name: str, arguments: Dict[str, Any]) -> str:
    """
    Unified tool executor that routes to the appropriate memory function.
    
    Args:
        tool_name: Either "memory_store" or "memory_retrieve"
        arguments: Tool-specific arguments
    
    Returns:
        JSON string with result
    """
    if tool_name == "memory_store":
        return memory_store_executor(tool_name, arguments)
    elif tool_name == "memory_retrieve":
        return memory_retrieve_executor(tool_name, arguments)
    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})


# Tool definitions for LLM
MEMORY_STORE_TOOL = {
    "type": "function",
    "function": {
        "name": "memory_store",
        "description": (
            "Store information in semantic memory. Use this to remember facts, "
            "conversations, preferences, or any information that might be useful later. "
            "The information is stored using semantic embeddings and can be retrieved "
            "based on similarity to future queries."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text content to store in memory"
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional additional metadata to store with the text",
                    "additionalProperties": True
                }
            },
            "required": ["text"]
        }
    }
}

MEMORY_RETRIEVE_TOOL = {
    "type": "function",
    "function": {
        "name": "memory_retrieve",
        "description": (
            "Retrieve information from semantic memory based on similarity to a query. "
            "Use this to recall previously stored information that is semantically "
            "similar to the query. Returns the most similar stored items."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query text to search for in memory"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5, max: 20)",
                    "minimum": 1,
                    "maximum": 20
                },
                "min_similarity": {
                    "type": "number",
                    "description": "Optional minimum similarity threshold (0-1). Results below this will be filtered out.",
                    "minimum": 0,
                    "maximum": 1
                }
            },
            "required": ["query"]
        }
    }
}


if __name__ == "__main__":
    # Test the memory tool
    print("Testing memory tool...")
    
    import shutil
    
    # Use a test storage path
    test_storage = "test_memory_store"
    
    # Clean up if exists
    if os.path.exists(test_storage):
        shutil.rmtree(test_storage)
    
    try:
        # Test 1: Store some information
        print("\n" + "="*60)
        print("Test 1: Storing information")
        print("="*60)
        
        test_texts = [
            "The weather is lovely today.",
            "It's so sunny outside!",
            "He drove to the stadium.",
            "Python is a programming language.",
            "Machine learning uses neural networks.",
            "The user prefers dark mode interface."
        ]
        
        for text in test_texts:
            result = memory_store(text, storage_path=test_storage)
            print(f"Stored: {text[:50]}... (ID: {result['id']})")
        
        # Test 2: Retrieve similar information
        print("\n" + "="*60)
        print("Test 2: Retrieving similar information")
        print("="*60)
        
        test_queries = [
            "What's the weather like?",
            "Tell me about programming",
            "What does the user like?"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            result = memory_retrieve(query, top_k=3, storage_path=test_storage)
            if result["success"]:
                print(f"Found {result['count']} results:")
                for i, item in enumerate(result["results"], 1):
                    print(f"  {i}. [{item['similarity']:.3f}] {item['text']}")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
        
        # Test 3: Get stats
        print("\n" + "="*60)
        print("Test 3: Memory store statistics")
        print("="*60)
        store = get_memory_store(test_storage)
        stats = store.get_stats()
        print(f"Total items: {stats['total_items']}")
        print(f"Embedding dimension: {stats['embedding_dim']}")
        print(f"Model: {stats['model_name']}")
        
    finally:
        # Clean up
        if os.path.exists(test_storage):
            shutil.rmtree(test_storage)
            print(f"\nCleaned up test storage: {test_storage}")

