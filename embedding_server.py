#!/usr/bin/env python3
"""
Embedding API server that pre-loads both GPT-OSS and Qwen embedding models.
Provides HTTP endpoints for embedding generation to minimize query-time latency.
"""
import os
import sys
import threading
import time
from typing import List, Optional, Dict, Any

try:
    from flask import Flask, request, jsonify
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False
    raise ImportError(
        "Flask is required for the embedding server. "
        "Please install: pip install flask"
    )

import numpy as np

# Add tools directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from sentence_transformers import SentenceTransformer
    import torch
    HAS_TORCH = True
except ImportError as e:
    raise ImportError(
        f"Required packages not installed: {e}\n"
        "Please install: pip install sentence-transformers torch"
    )

try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False


# Global model - loaded once at startup
_qwen_model: Optional[SentenceTransformer] = None
_qwen_device: Optional[str] = None
_models_ready = False
_models_lock = threading.Lock()


def load_embedding_model(
    qwen_model_name: str = "Qwen/Qwen3-Embedding-8B",
    qwen_device: str = "cuda:7",
    use_bfloat16: bool = True
) -> Dict[str, Any]:
    """
    Load Qwen embedding model at startup.
    
    Args:
        qwen_model_name: Name of Qwen embedding model (default: "Qwen/Qwen3-Embedding-8B")
        qwen_device: Device for Qwen model (default: cuda:7)
        use_bfloat16: Whether to use bfloat16 precision (default: True)
    
    Returns:
        Dictionary with loading status and model info
    """
    global _qwen_model, _qwen_device, _models_ready
    
    with _models_lock:
        if _models_ready:
            return {
                "status": "already_loaded",
                "device": _qwen_device,
                "dimension": _qwen_model.get_sentence_embedding_dimension() if _qwen_model else None
            }
        
        # Load Qwen embedding model
        print(f"\n[Embedding Server] Loading Qwen embedding model: {qwen_model_name} on {qwen_device}")
        try:
            qwen_model = SentenceTransformer(qwen_model_name, device=qwen_device)
            
            # Optimize for inference with bfloat16
            if HAS_TORCH and qwen_device.startswith("cuda"):
                qwen_model.eval()
                if use_bfloat16:
                    try:
                        # Convert to bfloat16 for better numerical stability
                        qwen_model = qwen_model.to(torch.bfloat16)
                        print(f"[Embedding Server] Qwen model converted to bfloat16")
                    except Exception as e:
                        print(f"[Embedding Server] bfloat16 conversion failed: {e}")
                        # Fallback to half precision if bfloat16 not supported
                        try:
                            qwen_model = qwen_model.half()
                            print(f"[Embedding Server] Qwen model converted to FP16 (fallback)")
                        except Exception as e2:
                            print(f"[Embedding Server] FP16 conversion also failed: {e2}")
            
            _qwen_model = qwen_model
            _qwen_device = qwen_device
            dimension = qwen_model.get_sentence_embedding_dimension()
            _models_ready = True
            
            print(f"[Embedding Server] ✓ Qwen embedding model loaded (dim: {dimension})")
            
            return {
                "status": "loaded",
                "device": qwen_device,
                "dimension": dimension
            }
        except Exception as e:
            error_msg = str(e)
            print(f"[Embedding Server] ✗ Failed to load Qwen model: {error_msg}")
            return {
                "status": "error",
                "error": error_msg
            }


def get_model() -> Optional[SentenceTransformer]:
    """
    Get the Qwen embedding model.
    
    Returns:
        The SentenceTransformer model or None if not loaded
    """
    global _qwen_model
    
    with _models_lock:
        return _qwen_model


# Flask app for embedding API
app = Flask(__name__)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    global _models_ready, _qwen_model
    
    with _models_lock:
        return jsonify({
            "status": "ready" if _models_ready else "loading",
            "qwen_loaded": _qwen_model is not None
        })


@app.route('/embed', methods=['POST'])
def embed():
    """
    Generate embeddings for input text(s) using Qwen/Qwen3-Embedding-8B.
    
    Request body:
        {
            "texts": ["text1", "text2", ...] or "text" (single string),
            "normalize": true/false (default: true),
            "batch_size": int (default: 32)
        }
    
    Response:
        {
            "embeddings": [[...], [...]],
            "model": "Qwen/Qwen3-Embedding-8B",
            "dimension": 1024,
            "count": 3
        }
    """
    global _qwen_model, _models_ready
    
    if not _models_ready:
        return jsonify({"error": "Model not yet loaded"}), 503
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Get texts
        if "texts" in data:
            texts = data["texts"]
            if isinstance(texts, str):
                texts = [texts]
        elif "text" in data:
            texts = [data["text"]]
        else:
            return jsonify({"error": "Missing 'text' or 'texts' field"}), 400
        
        if not texts or not all(isinstance(t, str) for t in texts):
            return jsonify({"error": "Invalid texts: must be non-empty strings"}), 400
        
        # Get model
        model = get_model()
        if model is None:
            return jsonify({"error": "Qwen model not loaded"}), 503
        
        # Get parameters
        normalize = data.get("normalize", True)
        batch_size = data.get("batch_size", 32)
        
        # Generate embeddings efficiently
        with torch.no_grad() if HAS_TORCH else None:
            embeddings = model.encode(
                texts,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=batch_size
            )
        
        # Convert to list format
        if len(texts) == 1:
            embeddings = [embeddings.tolist()]
        else:
            embeddings = embeddings.tolist()
        
        return jsonify({
            "embeddings": embeddings,
            "model": "Qwen/Qwen3-Embedding-8B",
            "dimension": len(embeddings[0]) if embeddings else 0,
            "count": len(embeddings)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/models', methods=['GET'])
def models():
    """Get information about loaded Qwen model."""
    global _qwen_model, _qwen_device
    
    with _models_lock:
        if _qwen_model is not None:
            info = {
                "qwen": {
                    "loaded": True,
                    "device": _qwen_device,
                    "dimension": _qwen_model.get_sentence_embedding_dimension(),
                    "model_name": "Qwen/Qwen3-Embedding-8B"
                }
            }
        else:
            info = {"qwen": {"loaded": False}}
        
        return jsonify(info)


def run_embedding_server(
    host: str = "0.0.0.0",
    port: int = 8001,
    qwen_model_name: str = "Qwen/Qwen3-Embedding-8B",
    qwen_device: str = "cuda:7",
    use_bfloat16: bool = True,
    debug: bool = False
):
    """
    Run the embedding API server with Qwen model.
    
    Args:
        host: Host to bind to (default: 0.0.0.0)
        port: Port to bind to (default: 8001)
        qwen_model_name: Qwen embedding model name (default: "Qwen/Qwen3-Embedding-8B")
        qwen_device: Device for Qwen model (default: cuda:7)
        use_bfloat16: Whether to use bfloat16 precision (default: True)
        debug: Flask debug mode
    """
    print("\n" + "="*60)
    print("Starting Embedding API Server")
    print("="*60)
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Model: {qwen_model_name} on {qwen_device}")
    print(f"Precision: bfloat16" if use_bfloat16 else "Precision: FP32")
    print("="*60 + "\n")
    
    # Load model before starting server
    load_results = load_embedding_model(
        qwen_model_name=qwen_model_name,
        qwen_device=qwen_device,
        use_bfloat16=use_bfloat16
    )
    
    print("\n" + "="*60)
    print("Embedding Server Status")
    print("="*60)
    print(f"Status: {load_results['status']}")
    if load_results.get('error'):
        print(f"  Error: {load_results['error']}")
    elif load_results.get('dimension'):
        print(f"  Dimension: {load_results['dimension']}")
        print(f"  Device: {load_results['device']}")
    print("="*60 + "\n")
    
    # Start Flask server
    app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Embedding API Server with Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--qwen-model", type=str, default="Qwen/Qwen3-Embedding-8B", help="Qwen embedding model")
    parser.add_argument("--qwen-device", type=str, default="cuda:7", help="Device for Qwen model")
    parser.add_argument("--no-bfloat16", action="store_true", help="Disable bfloat16 precision (use FP32)")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    
    args = parser.parse_args()
    
    run_embedding_server(
        host=args.host,
        port=args.port,
        qwen_model_name=args.qwen_model,
        qwen_device=args.qwen_device,
        use_bfloat16=not args.no_bfloat16,
        debug=args.debug
    )

