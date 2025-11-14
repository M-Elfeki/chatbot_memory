#!/usr/bin/env python3
"""
Latency benchmark tests for memory tool.
Measures average latency for store and retrieve operations with optimizations.
"""
import unittest
import time
import statistics
import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.memory_tool import MemoryStore, memory_store, memory_retrieve


class TestMemoryLatency(unittest.TestCase):
    """Latency benchmark tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.storage_path = os.path.join(self.test_dir, "test_memory")
        # Force CUDA:7 if available
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.device_count() > 7:
                self.device = "cuda:7"
            elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
                self.device = f"cuda:{torch.cuda.device_count() - 1}"
            elif torch.cuda.is_available():
                self.device = "cuda:0"
            else:
                self.device = "cpu"
        except ImportError:
            self.device = "cpu"
        
        print(f"\nUsing device: {self.device}")
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_store_latency(self):
        """Measure average latency for store operations."""
        store = MemoryStore(storage_path=self.storage_path, device=self.device, use_quantization=True)
        
        # Warmup
        store.store("Warmup text")
        
        # Test texts of varying lengths
        test_texts = [
            "Short text",
            "This is a medium length text that contains more words and information.",
            "This is a longer text that contains multiple sentences. It has more content and complexity. The embedding model needs to process all of this information and generate a semantic representation.",
            "Python is a high-level programming language known for its simplicity.",
            "Machine learning algorithms learn patterns from data to make predictions.",
            "The weather forecast predicts sunny skies with temperatures in the 70s.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning uses neural networks with multiple layers for complex tasks.",
            "Cloud computing provides scalable infrastructure for applications.",
            "Data science combines statistics, programming, and domain expertise."
        ]
        
        latencies = []
        num_iterations = len(test_texts) * 3  # 3 runs per text
        
        print(f"\nMeasuring store latency ({num_iterations} operations)...")
        
        for i, text in enumerate(test_texts * 3):
            start_time = time.perf_counter()
            result = store.store(text)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency_ms)
            
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{num_iterations} operations")
        
        avg_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0
        
        print(f"\nStore Latency Results:")
        print(f"  Average: {avg_latency:.2f} ms")
        print(f"  Median: {median_latency:.2f} ms")
        print(f"  Min: {min_latency:.2f} ms")
        print(f"  Max: {max_latency:.2f} ms")
        print(f"  Std Dev: {std_latency:.2f} ms")
        print(f"  Total operations: {len(latencies)}")
        
        # Store results for reporting
        self.store_latency_avg = avg_latency
        self.store_latency_median = median_latency
    
    def test_retrieve_latency(self):
        """Measure average latency for retrieve operations."""
        store = MemoryStore(storage_path=self.storage_path, device=self.device, use_quantization=True)
        
        # Pre-populate with diverse content
        texts = [
            "Python programming language",
            "Java object-oriented programming",
            "Weather forecast sunny",
            "Machine learning neural networks",
            "Data science statistics",
            "Cloud computing infrastructure",
            "Natural language processing",
            "Deep learning algorithms",
            "Computer vision image recognition",
            "Web development frameworks",
            "Database management systems",
            "Software engineering practices",
            "Artificial intelligence applications",
            "Cybersecurity threat detection",
            "Mobile app development"
        ]
        
        # Store all texts
        print(f"\nPre-populating store with {len(texts)} items...")
        for text in texts:
            store.store(text)
        
        # Warmup
        store.retrieve("warmup query")
        
        # Test queries
        test_queries = [
            "programming languages",
            "weather information",
            "machine learning",
            "data analysis",
            "cloud services",
            "language processing",
            "neural networks",
            "software development",
            "artificial intelligence",
            "web applications"
        ]
        
        latencies = []
        num_iterations = len(test_queries) * 5  # 5 runs per query
        
        print(f"\nMeasuring retrieve latency ({num_iterations} operations)...")
        
        for i, query in enumerate(test_queries * 5):
            start_time = time.perf_counter()
            result = store.retrieve(query, top_k=5)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency_ms)
            
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{num_iterations} operations")
        
        avg_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0
        
        print(f"\nRetrieve Latency Results:")
        print(f"  Average: {avg_latency:.2f} ms")
        print(f"  Median: {median_latency:.2f} ms")
        print(f"  Min: {min_latency:.2f} ms")
        print(f"  Max: {max_latency:.2f} ms")
        print(f"  Std Dev: {std_latency:.2f} ms")
        print(f"  Total operations: {len(latencies)}")
        
        # Store results for reporting
        self.retrieve_latency_avg = avg_latency
        self.retrieve_latency_median = median_latency
    
    def test_end_to_end_latency(self):
        """Measure end-to-end latency for store + retrieve workflow."""
        store = MemoryStore(storage_path=self.storage_path, device=self.device, use_quantization=True)
        
        # Warmup
        store.store("warmup")
        store.retrieve("warmup")
        
        test_cases = [
            ("Python is great", "programming language"),
            ("Weather is nice", "weather forecast"),
            ("ML is fascinating", "machine learning"),
            ("Cloud is scalable", "cloud computing"),
            ("NLP is powerful", "natural language")
        ]
        
        latencies = []
        
        print(f"\nMeasuring end-to-end latency ({len(test_cases)} operations)...")
        
        for store_text, query_text in test_cases:
            start_time = time.perf_counter()
            
            # Store
            store.store(store_text)
            
            # Retrieve
            result = store.retrieve(query_text, top_k=3)
            
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        avg_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        
        print(f"\nEnd-to-End Latency Results:")
        print(f"  Average: {avg_latency:.2f} ms")
        print(f"  Median: {median_latency:.2f} ms")
        
        self.e2e_latency_avg = avg_latency
    
    def test_batch_latency(self):
        """Measure latency for batch operations."""
        store = MemoryStore(storage_path=self.storage_path, device=self.device, use_quantization=True)
        
        batch_texts = [
            f"Text item {i}: This is test content number {i} for batch processing."
            for i in range(20)
        ]
        
        # Warmup
        store.store("warmup")
        
        print(f"\nMeasuring batch store latency (20 items)...")
        
        start_time = time.perf_counter()
        for text in batch_texts:
            store.store(text)
        end_time = time.perf_counter()
        
        total_time_ms = (end_time - start_time) * 1000
        avg_per_item = total_time_ms / len(batch_texts)
        
        print(f"\nBatch Store Latency Results:")
        print(f"  Total time: {total_time_ms:.2f} ms")
        print(f"  Average per item: {avg_per_item:.2f} ms")
        print(f"  Items: {len(batch_texts)}")
        
        self.batch_latency_avg = avg_per_item


def run_benchmark():
    """Run all latency benchmarks and report results."""
    print("=" * 80)
    print("MEMORY TOOL LATENCY BENCHMARK")
    print("=" * 80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMemoryLatency)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Extract results if available
    if hasattr(result, 'testsRun') and result.testsRun > 0:
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        
        # Try to get results from test instances
        for test_case in suite:
            if hasattr(test_case, 'store_latency_avg'):
                print(f"\nStore Operation Average Latency: {test_case.store_latency_avg:.2f} ms")
            if hasattr(test_case, 'retrieve_latency_avg'):
                print(f"Retrieve Operation Average Latency: {test_case.retrieve_latency_avg:.2f} ms")
            if hasattr(test_case, 'e2e_latency_avg'):
                print(f"End-to-End Average Latency: {test_case.e2e_latency_avg:.2f} ms")
            if hasattr(test_case, 'batch_latency_avg'):
                print(f"Batch Store Average Latency: {test_case.batch_latency_avg:.2f} ms")
    
    return result


if __name__ == "__main__":
    result = run_benchmark()
    
    # Also run individual tests to capture results
    test = TestMemoryLatency()
    test.setUp()
    
    try:
        test.test_store_latency()
        store_avg = test.store_latency_avg
        
        test.test_retrieve_latency()
        retrieve_avg = test.retrieve_latency_avg
        
        test.test_end_to_end_latency()
        e2e_avg = test.e2e_latency_avg
        
        test.test_batch_latency()
        batch_avg = test.batch_latency_avg
        
        print("\n" + "=" * 80)
        print("FINAL LATENCY REPORT")
        print("=" * 80)
        print(f"Store Average Latency: {store_avg:.2f} ms")
        print(f"Retrieve Average Latency: {retrieve_avg:.2f} ms")
        print(f"End-to-End Average Latency: {e2e_avg:.2f} ms")
        print(f"Batch Store Average Latency: {batch_avg:.2f} ms")
        print("=" * 80)
        
    finally:
        test.tearDown()

