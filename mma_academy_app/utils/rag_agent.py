"""
Reusable RAG (Retrieval Augmented Generation) agent logic.
Uses sentence-transformers for embeddings, FAISS for indexing, and Hugging Face Transformers for LLM.
Enhanced with adaptive multi-model selection, robust fallbacks, and caching.
"""
# utils/rag_agent.py + # RLHF components for rag_agent.py
import os
import json
import time
import torch
import numpy as np
from pathlib import Path
import streamlit as st
import traceback
import sys
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass
from functools import lru_cache
from datetime import datetime
import re
import pickle
import logging
"""
RAG (Retrieval-Augmented Generation) agent for the MMA Academy assistant.
Handles retrieval from the manual database and generation of responses.
"""

# Fix for torch._classes error
try:
    import torch

    # Try to fix potential torch._classes issue with streamlit
    # This is a workaround for RuntimeError: Tried to instantiate class '__path__._path'
    try:
        from types import ModuleType


        class PathFixModule(ModuleType):
            def __getattr__(self, attr):
                if attr == "_path":
                    return []
                raise AttributeError(f"module 'torch._classes' has no attribute '{attr}'")


        import sys

        if 'torch._classes' in sys.modules:
            sys.modules['torch._classes.__path__'] = PathFixModule('torch._classes.__path__')
    except Exception as e:
        print(f"Warning: Could not apply torch._classes fix: {e}")

except ImportError as e:
    print(f"Warning: Could not import torch: {e}")

    # Monkey patch torch._classes to prevent Streamlit watcher errors
    if hasattr(torch, '_classes'):
        original_getattr = torch._classes.__getattr__


        def safe_getattr(self, name):
            if name == '__path__':
                # Return an empty list instead of raising error
                class EmptyPath:
                    _path = []

                return EmptyPath()
            return original_getattr(self, name)


        torch._classes.__getattr__ = safe_getattr.__get__(torch._classes)
except Exception as e:
    print(f"Warning: Could not apply torch._classes fix: {e}")

# Rest of your imports
from sentence_transformers import SentenceTransformer
import faiss

# Constants
DEFAULT_TIMEOUT = 180  # 3 minutes max for generation
TOP_K_CHUNKS = 3  # Number of context chunks to retrieve

# --- Check for required packages ---
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available, memory management will be limited")

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence_transformers not available, will use HuggingFace models")

try:
    import transformers

    transformers_version = transformers.__version__
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    transformers_version = "Not installed"
    print("Warning: transformers not available, LLM functionality will be disabled")

# --- RLHF Constants ---
FEEDBACK_LOG_PATH = Path("logs/feedback_ratings.csv")
RLHF_MODEL_PATH = Path("data/rlhf_model.json")
DEFAULT_LEARNING_RATE = 0.01
REWARD_POSITIVE = 1.0
REWARD_NEGATIVE = -0.5


class RLHFManager:
    """Manages the RLHF process for the RAG system"""

    def __init__(self):
        """Initialize the RLHF manager"""
        self.model_weights = self._load_or_create_model()
        self.learning_rate = DEFAULT_LEARNING_RATE

    def _load_or_create_model(self) -> Dict[str, Any]:
        """Load existing RLHF model or create a new one"""
        if RLHF_MODEL_PATH.exists():
            try:
                with open(RLHF_MODEL_PATH, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading RLHF model: {e}")

        # Initialize default model weights
        return {
            "version": 1.0,
            "last_updated": datetime.now().isoformat(),
            "query_context_weights": {
                "query_tokens": 1.0,
                "context_relevance": 1.0,
                "source_count": 0.5,
                "query_length": 0.3
            },
            "source_weights": {},
            "token_weights": {},
            "retrieval_weights": {
                "similarity_threshold": 0.75,
                "top_k_multiplier": 1.0,
                "reranking_factor": 0.5
            },
            "feedback_count": 0,
            "positive_feedback": 0
        }

    def save_model(self):
        """Save RLHF model weights to disk"""
        self.model_weights["last_updated"] = datetime.now().isoformat()
        with open(RLHF_MODEL_PATH, 'w') as f:
            json.dump(self.model_weights, f, indent=2)

    def process_feedback(self, feedback_entry: Dict[str, Any]):
        """Process a feedback entry and update model weights"""
        # Extract feedback data
        query = feedback_entry.get("question", "")
        answer = feedback_entry.get("answer", "")
        satisfaction = feedback_entry.get("satisfaction", "")
        is_positive = satisfaction in ("positive", "yes", "helpful", "thumbs_up")

        # Update overall statistics
        self.model_weights["feedback_count"] += 1
        if is_positive:
            self.model_weights["positive_feedback"] += 1

        # Calculate reward value
        reward = REWARD_POSITIVE if is_positive else REWARD_NEGATIVE

        # Update token weights based on query and answer
        self._update_token_weights(query, answer, reward)

        # Update source weights if available
        sources = feedback_entry.get("sources", "").split("|") if feedback_entry.get("sources") else []
        if sources:
            self._update_source_weights(sources, reward)

        # Update retrieval weights (advanced, optional)
        self._update_retrieval_weights(is_positive)

        # Save model after updates
        self.save_model()

    def _update_token_weights(self, query: str, answer: str, reward: float):
        """Update weight values for tokens in query and answer"""
        # Simple implementation - extract tokens and update weights
        tokens = set([token.lower() for token in query.split() + answer.split()
                      if len(token) > 3 and token.isalpha()])

        for token in tokens:
            if token not in self.model_weights["token_weights"]:
                self.model_weights["token_weights"][token] = 0.0

            # Apply learning rate to reward
            self.model_weights["token_weights"][token] += reward * self.learning_rate

    def _update_source_weights(self, sources: List[str], reward: float):
        """Update weights for document sources based on feedback"""
        for source in sources:
            source_key = str(source).strip()
            if not source_key:
                continue

            if source_key not in self.model_weights["source_weights"]:
                self.model_weights["source_weights"][source_key] = 0.0

            # Apply learning rate to reward
            self.model_weights["source_weights"][source_key] += reward * self.learning_rate

    def _update_retrieval_weights(self, is_positive: bool):
        """Update retrieval parameters based on feedback"""
        # Adjust similarity threshold
        if is_positive:
            # If positive, slightly lower threshold to include more similar results
            self.model_weights["retrieval_weights"]["similarity_threshold"] *= 0.99
        else:
            # If negative, increase threshold to be more selective
            self.model_weights["retrieval_weights"]["similarity_threshold"] *= 1.01

        # Keep threshold in reasonable bounds
        threshold = self.model_weights["retrieval_weights"]["similarity_threshold"]
        self.model_weights["retrieval_weights"]["similarity_threshold"] = max(0.5, min(0.95, threshold))

    def get_retrieval_params(self) -> Dict[str, float]:
        """Get current retrieval parameters based on learned weights"""
        return self.model_weights["retrieval_weights"]

    def rerank_results(self, results: List[Tuple[float, int]], texts: List[str]) -> List[Tuple[float, int]]:
        """Rerank search results based on learned weights"""
        # If no results or no token weights yet, return as-is
        if not results or not self.model_weights["token_weights"]:
            return results

        # Apply source weights and token weights to reranking
        reranked_results = []
        for score, idx in results:
            # Get content at this index
            content = texts[idx] if idx < len(texts) else ""

            # Get source weight if available
            source_key = f"Page {idx + 1}"  # Default source key format
            source_weight = self.model_weights["source_weights"].get(source_key, 0.0)

            # Calculate token weight influence
            token_weight = 0.0
            content_tokens = set([token.lower() for token in content.split()
                                  if len(token) > 3 and token.isalpha()])

            for token in content_tokens:
                if token in self.model_weights["token_weights"]:
                    token_weight += self.model_weights["token_weights"][token]

            # Normalize token weight
            if content_tokens:
                token_weight /= len(content_tokens)

            # Apply both adjustments to score
            adjusted_score = score + (source_weight * 0.2) + (token_weight * 0.3)

            reranked_results.append((adjusted_score, idx))

        # Sort by adjusted score
        return sorted(reranked_results, key=lambda x: x[0], reverse=True)

    def process_feedback_file(self):
        """Process all feedback entries from CSV"""
        if not FEEDBACK_LOG_PATH.exists():
            return

        import csv

        try:
            with open(FEEDBACK_LOG_PATH, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.process_feedback(row)
        except Exception as e:
            print(f"Error processing feedback file: {e}")


# Initialize RLHF manager (to be used in the main query processing)
rlhf_manager = RLHFManager()


# Function to update query_rag_agent.py to use RLHF
def apply_rlhf_to_search(query: str, vector: np.ndarray, index, texts: List[str], top_k: int) -> Tuple[
    List[str], List[str]]:
    """
    Apply RLHF to search process

    Args:
        query: User question
        vector: Embedded query vector
        index: FAISS index
        texts: Document texts
        top_k: Number of results to retrieve

    Returns:
        Tuple of (context_chunks, page_refs)
    """
    # Get retrieval parameters from RLHF model
    retrieval_params = rlhf_manager.get_retrieval_params()

    # Adjust top_k based on learned weights
    adjusted_top_k = int(top_k * retrieval_params.get("top_k_multiplier", 1.0))

    # Get initial search results
    distances, indices = index.search(vector, adjusted_top_k)

    # Convert to list of tuples for reranking
    results = [(float(distances[0][i]), int(indices[0][i])) for i in range(len(indices[0]))]

    # Apply reranking from RLHF
    reranked_results = rlhf_manager.rerank_results(results, texts)

    # Extract final indices (limiting to original top_k after reranking)
    final_indices = [idx for _, idx in reranked_results[:top_k] if idx < len(texts)]

    # Create context from final selection
    context_chunks = [texts[i] for i in final_indices]

    # Create page references
    page_keys = st.session_state.get("rag", {}).get("page_keys", [])
    if page_keys and len(page_keys) > 0:
        page_refs = []
        for i in final_indices:
            if i < len(page_keys):
                page_refs.append(page_keys[i])
            else:
                page_refs.append(f"Unknown-{i}")
    else:
        page_refs = [f"Section {i + 1}" for i, _ in enumerate(final_indices)]

    return context_chunks, page_refs


# --- Model Configuration ---
def get_embedding_model():
    """
    Returns only the embedding model part of the deepseek models tuple.
    This function exists for backward compatibility with code that expects
    to retrieve just the embedding model.

    Returns:
        The embedding model (second element of the tuple returned by get_deepseek_models)
    """
    # Use the model path from secrets if available, otherwise empty string will cause
    # get_deepseek_models to use its fallback mechanisms
    model_path = st.secrets.get("llm", {}).get("model_path", "")
    _, embedding_model = get_deepseek_models(model_path)
    return embedding_model

class ModelType(Enum):
    TINY = "tiny"  # For simple queries, ultra-fast
    MEDIUM = "medium"  # Balanced option
    LARGE = "large"  # Full model for complex queries


# Model configurations with paths to be overridden from secrets.toml
MODEL_CONFIGS = {
    "tiny": {
        "name": "TinyLlama-1.1B",
        "path": "D:/LLM_Models/tinyllama",  # Default path, override from secrets
        "type": ModelType.TINY,
        "max_tokens": 128,
        "memory_required": 2048,  # ~2GB
        "strengths": ["Speed", "Basic information", "Follow-up questions"],
        "best_for": ["simple", "clarification", "follow-up", "hello", "hi", "greetings"]
    },
    "medium": {
        "name": "Phi-2",
        "path": "D:/LLM_Models/phi-2",  # Default path, override from secrets
        "type": ModelType.MEDIUM,
        "max_tokens": 256,
        "memory_required": 5120,  # ~5GB
        "strengths": ["Balance of speed/accuracy", "Good explanations", "Technique descriptions"],
        "best_for": ["techniques", "explain", "describe", "what is", "how to"]
    },
    "large": {
        "name": "DeepSeek-7B",
        "path": "D:/LLM_Models/deepseek-7b-manual",  # Default path, override from secrets
        "type": ModelType.LARGE,
        "max_tokens": 512,
        "memory_required": 14336,  # ~14GB
        "strengths": ["Detailed knowledge", "Complex reasoning", "Comprehensive answers"],
        "best_for": ["complex", "detailed", "compare", "analyze", "why", "philosophy", "strategy"]
    }
}

# --- Constants ---
DEFAULT_TIMEOUT = 60  # Max seconds to wait for model response
MAX_NEW_TOKENS = 512  # Default max tokens to generate
EMBEDDING_BATCH_SIZE = 8  # Number of chunks to embed at once
TOP_K_CHUNKS = 3  # Number of chunks to retrieve from vector store
FALLBACK_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fallback embedding model

# --- File paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANUAL_JSON_PATH = os.path.join(BASE_DIR, "data", "manual.json")
INDEX_PATH = os.path.join(
    BASE_DIR, "data", "vector_store", "faiss_index.bin"
)
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# --- Globals (in-memory caching) ---
_index = None  # FAISS index
_texts = None  # Chunked texts
_tokenizer = None  # Tokenizer
_embedding_model = None  # Embedding model
_llm_pipeline = None  # LLM pipeline
_sentence_transformer = None  # Sentence transformer
_loaded_models = {}  # Cache for loaded models
_system_info = None  # System information cache


# --- Logging Functions ---
def log_query_to_file(email: str, question: str, sources: List[str],
                      model: Optional[str] = None,
                      tokens_used: Optional[int] = None,
                      latency_s: Optional[float] = None) -> None:
    """
    Log query details to a JSON file for analytics and debugging.

    Args:
        email: User's email
        question: The query text
        sources: List of source references (e.g., page numbers)
        model: Name of the model used
        tokens_used: Number of tokens processed (optional)
        latency_s: Query processing time in seconds (optional)
    """
    log_path = os.path.join(LOGS_DIR, "query_logs.json")

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "email": email,
        "question": question,
        "sources": sources,
        "model": model,
        "tokens_used": tokens_used,
        "latency_s": latency_s
    }

    try:
        # Load existing logs or create new list
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append(entry)

        # Write updated logs back to file
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        print(f"[LOG ERROR] {e}")
        # Don't break user experience if logging fails


def log_error(error_type: str, details: str, context: Optional[Dict] = None) -> None:
    """
    Log system errors to a separate file for debugging.

    Args:
        error_type: Category of error
        details: Error message
        context: Additional context information
    """
    log_path = os.path.join(LOGS_DIR, "system_errors.json")

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "error_type": error_type,
        "details": details,
        "context": context or {}
    }

    try:
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append(entry)

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        print(f"[ERROR LOG FAILURE] {e}")


# --- Resource Management Functions ---
def estimate_available_memory() -> int:
    """
    Estimate available system memory in MB

    Returns:
        Available memory in MB
    """
    if PSUTIL_AVAILABLE:
        try:
            return int(psutil.virtual_memory().available / (1024 * 1024))
        except Exception as e:
            log_error("memory_check", str(e))

    # Conservative estimate if psutil fails/unavailable
    return 4096  # Assume 4GB available


def classify_query_complexity(query: str) -> float:
    """
    Rate query complexity from 0.0 (simple) to 1.0 (complex)

    Args:
        query: User's question

    Returns:
        Complexity score from 0.0 to 1.0
    """
    # Simple heuristics-based classifier
    query = query.lower()

    # Special cases
    if len(query.strip()) < 5:
        return 0.0  # Very short queries are simple

    # Count question words (more = more complex)
    question_words = ["why", "how", "compare", "explain", "analyze", "difference",
                      "philosophy", "strategy", "versus", "vs", "between"]
    question_word_count = sum(1 for word in question_words if word in query)

    # Count length (longer = more complex)
    length_factor = min(len(query.split()) / 15, 1.0)

    # Check for specialized terminology
    mma_terms = ["technique", "grappling", "striking", "submission", "training", "discipline",
                 "philosophy", "stance", "form", "strategy", "competition", "sparring"]
    term_count = sum(1 for term in mma_terms if term in query)

    # Simple weighted score
    complexity = (
            (question_word_count * 0.4) +
            (length_factor * 0.3) +
            (min(term_count / 4, 1.0) * 0.3)
    )

    return min(complexity, 1.0)  # Cap at 1.0


def select_model_for_query(query: str) -> Dict:
    """
    Select the most appropriate model based on query complexity, keywords, and available resources.

    Args:
        query: User's question

    Returns:
        Model configuration dictionary
    """
    query = query.lower()
    complexity = classify_query_complexity(query)
    available_memory = estimate_available_memory()

    # Check for administrator override
    if st.session_state.get("is_admin", False):
        model_preference = st.session_state.get("model_preference", None)
        if model_preference and model_preference in MODEL_CONFIGS:
            if available_memory >= MODEL_CONFIGS[model_preference]["memory_required"]:
                return MODEL_CONFIGS[model_preference]
            else:
                st.warning(f"⚠️ Not enough memory for {model_preference} model")

    # Check for keywords that suggest specific model strengths
    for model_key, config in MODEL_CONFIGS.items():
        for keyword in config["best_for"]:
            if keyword in query:
                # Check if we have enough memory
                if available_memory >= config["memory_required"]:
                    return config

    # If no keywords matched, base selection on complexity and available memory
    if complexity < 0.3 and available_memory >= MODEL_CONFIGS["tiny"]["memory_required"]:
        return MODEL_CONFIGS["tiny"]
    elif complexity < 0.7 and available_memory >= MODEL_CONFIGS["medium"]["memory_required"]:
        return MODEL_CONFIGS["medium"]
    elif available_memory >= MODEL_CONFIGS["large"]["memory_required"]:
        return MODEL_CONFIGS["large"]

    # Fallback to the model that fits in available memory
    if available_memory >= MODEL_CONFIGS["medium"]["memory_required"]:
        return MODEL_CONFIGS["medium"]
    else:
        return MODEL_CONFIGS["tiny"]


def unload_unused_models(keep_model: str = None):
    """
    Unload models to free memory

    Args:
        keep_model: Name of model to keep loaded
    """
    global _loaded_models

    models_to_unload = [model for model in _loaded_models.keys() if model != keep_model]

    for model_name in models_to_unload:
        _loaded_models.pop(model_name, None)

    # Force garbage collection
    import gc
    gc.collect()

    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- Embedding and Retrieval Functions ---
def get_deepseek_models(model_path: str) -> Tuple[Any, Any]:
    """
    Gets tokenizer and embedding model for DeepSeek (or fallback)

    Args:
        model_path: Path to model directory

    Returns:
        Tuple of (tokenizer, embedding_model)
    """
    global _tokenizer, _embedding_model, _sentence_transformer

    # Return cached models if available
    if _tokenizer is not None and _embedding_model is not None:
        return _tokenizer, _embedding_model

    # Check for DeepSeek embedding model
    try:
        # First try sentence transformers for better embedding quality
        if SENTENCE_TRANSFORMERS_AVAILABLE and _sentence_transformer is None:
            try:
                # Try local first, fallback to Hugging Face
                try:
                    # Try to load from local model path first
                    model_variants = [
                        os.path.join(model_path, "embeddings"),
                        model_path
                    ]

                    embedding_model = None
                    for variant in model_variants:
                        if os.path.exists(variant):
                            try:
                                # If path exists, try to load as SentenceTransformer
                                embedding_model = SentenceTransformer(variant)
                                break
                            except Exception:
                                # If that fails, continue to next variant
                                pass

                    if embedding_model is None:
                        # If no local model worked, use fallback from HF
                        embedding_model = SentenceTransformer(FALLBACK_EMBEDDING_MODEL)
                except Exception as e:
                    # Fallback to the default smaller model
                    embedding_model = SentenceTransformer(FALLBACK_EMBEDDING_MODEL)

                _sentence_transformer = embedding_model

                # For sentence transformers, we return a special tokenizer
                class SentenceTokenizer:
                    def __call__(self, text, **kwargs):
                        return {"input_ids": [text]}

                    @property
                    def vocab_size(self):
                        return 50000  # Dummy value

                _tokenizer = SentenceTokenizer()
                _embedding_model = _sentence_transformer

                return _tokenizer, _embedding_model
            except Exception as e:
                # Handle exceptions in the outer try block
                log_error("sentence_transformer_setup", str(e))

        # If sentence transformers not available, try transformers
        if TRANSFORMERS_AVAILABLE:
            # Check if we must load the models
            if _tokenizer is None or _embedding_model is None:
                from transformers import AutoTokenizer, AutoModel

                # Try loading tokenizer
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_path,
                        local_files_only=True,
                        trust_remote_code=True
                    )
                except Exception as tokenizer_error:
                    log_error("tokenizer_loading", str(tokenizer_error), {"model_path": model_path})

                    # Check if model path exists
                    if not os.path.exists(model_path):
                        error_msg = f"Model path does not exist: {model_path}"
                        log_error("model_path", error_msg)
                        return None, None

                    # If we're here, path exists but tokenizer loading failed
                    # Try fallback to all-mpnet
                    try:
                        from transformers import AutoTokenizer
                        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
                    except Exception as e:
                        log_error("fallback_tokenizer", str(e))
                        return None, None

                # Try loading model
                try:
                    model = AutoModel.from_pretrained(
                        model_path,
                        local_files_only=True,
                        trust_remote_code=True
                    )
                except Exception as model_error:
                    log_error("model_loading", str(model_error), {"model_path": model_path})

                    # Try fallback model
                    try:
                        from transformers import AutoModel
                        model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
                    except Exception as e:
                        log_error("fallback_model", str(e))
                        return None, None

                # Cache the models
                _tokenizer = tokenizer
                _embedding_model = model

            return _tokenizer, _embedding_model

        # If both options failed, return None
        return None, None

    except Exception as e:
        error_details = traceback.format_exc()
        log_error("embedding_setup", str(e), {"traceback": error_details})
        return None, None


def embed_chunks(text_chunks: List[str]) -> Optional[np.ndarray]:
    """
    Create embeddings for text chunks

    Args:
        text_chunks: List of text chunks to embed

    Returns:
        numpy array of embeddings or None if failed
    """
    # Get the tokenizer and model from path in secrets
    model_path = st.secrets.get("llm", {}).get("model_path", "D:/LLM_Models/deepseek-7b-manual")
    tokenizer, emb_model = get_deepseek_models(model_path)

    if tokenizer is None or emb_model is None:
        # Try fallback embedding model
        try:
            from sentence_transformers import SentenceTransformer
            emb_model = SentenceTransformer(FALLBACK_EMBEDDING_MODEL)
            embeddings = emb_model.encode(text_chunks, show_progress_bar=True, convert_to_numpy=True)
            return embeddings.astype("float32")
        except Exception as e:
            log_error("fallback_embedding", str(e))
            return None

    try:
        # Use batching to avoid OOM
        embeddings = []

        # Progress display
        total_chunks = len(text_chunks)
        with st.spinner(f"Creating embeddings for {total_chunks} chunks..."):
            progress_bar = st.progress(0)

            # Process in batches
            for i in range(0, total_chunks, EMBEDDING_BATCH_SIZE):
                batch = text_chunks[i:i + EMBEDDING_BATCH_SIZE]

                # Update progress
                progress = min(1.0, (i + EMBEDDING_BATCH_SIZE) / total_chunks)
                progress_bar.progress(progress)

                # Create embeddings based on model type
                if hasattr(emb_model, 'encode') and callable(getattr(emb_model, 'encode')):
                    # SentenceTransformer approach
                    batch_embeddings = emb_model.encode(batch, convert_to_numpy=True)
                    embeddings.append(batch_embeddings)
                else:
                    # Standard HuggingFace approach
                    batch_embs = []
                    for text in batch:
                        # Encode and get embedding
                        with torch.no_grad():
                            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                            outputs = emb_model(**inputs)
                            # Use CLS token or average of last hidden state
                            emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                            batch_embs.append(emb[0])

                    embeddings.append(np.array(batch_embs))

            # Complete progress
            progress_bar.progress(1.0)

        # Combine and return embeddings
        if embeddings:
            combined = np.vstack(embeddings)
            return combined.astype("float32")
        return None

    except Exception as e:
        error_details = traceback.format_exc()
        log_error("embedding_creation", str(e), {"traceback": error_details})
        return None


# --- FAISS Index Management ---
def build_or_load_faiss_index(model_path=None, force_rebuild=False) -> Tuple[Any, List[str]]:
    """
    Build or load a FAISS index for vector similarity search

    Args:
        model_path: Optional model path (not directly used, but for cache invalidation)
        force_rebuild: Force rebuild the index

    Returns:
        Tuple of (faiss_index, text_chunks)
    """
    global _index, _texts

    # Force rebuild if requested
    if force_rebuild or st.session_state.get("force_rebuild_index", False):
        if "force_rebuild_index" in st.session_state:
            st.session_state["force_rebuild_index"] = False
        _index, _texts = None, None

    # Return cached index if available
    if _index is not None and _texts is not None:
        return _index, _texts

    try:
        # Check if index file exists
        if os.path.exists(INDEX_PATH) and not force_rebuild:
            try:
                import faiss
            except ImportError:
                st.error("❌ FAISS not installed. Please install with: pip install faiss-cpu or faiss-gpu")
                return None, None

            # Load index and chunks
            try:
                _index = faiss.read_index(INDEX_PATH)
            except Exception as e:
                log_error("index_loading", str(e))
                os.rename(INDEX_PATH, f"{INDEX_PATH}.broken")
                st.warning(f"⚠️ Existing index was corrupt, renamed to {INDEX_PATH}.broken")
                _index = None

            # Load text chunks
            chunks_path = os.path.join(os.path.dirname(INDEX_PATH), "chunks.pkl")
            if os.path.exists(chunks_path):
                try:
                    with open(chunks_path, 'rb') as f:
                        _texts = pickle.load(f)
                except Exception as e:
                    log_error("chunks_loading", str(e))
                    _texts = None

            # Also load page keys if available
            keys_path = os.path.join(os.path.dirname(INDEX_PATH), "page_keys.pkl")
            if os.path.exists(keys_path):
                try:
                    with open(keys_path, 'rb') as f:
                        page_keys = pickle.load(f)
                        # Store in session state for reference
                        if "rag" not in st.session_state:
                            st.session_state["rag"] = {}
                        st.session_state["rag"]["page_keys"] = page_keys
                except Exception as e:
                    log_error("page_keys_loading", str(e))
                    # We'll generate default keys below if needed

            # If either index or texts failed to load, force rebuild
            if _index is not None and _texts is not None:
                # Store page keys if missing
                if "rag" not in st.session_state or "page_keys" not in st.session_state["rag"]:
                    if "rag" not in st.session_state:
                        st.session_state["rag"] = {}
                    # Generate default keys that match the length of texts
                    st.session_state["rag"]["page_keys"] = [f"Section {i + 1}" for i in range(len(_texts))]

                # Index loaded successfully
                return _index, _texts

        # If we're here, we need to build the index
        if not os.path.exists(MANUAL_JSON_PATH):
            st.error(f"❌ Manual data not found at {MANUAL_JSON_PATH}")
            return None, None

        # Load manual data
        with open(MANUAL_JSON_PATH, 'r', encoding='utf-8') as f:
            manual_data = json.load(f)

        # Extract text and page keys
        page_keys = list(manual_data.keys())
        texts = list(manual_data.values())

        # Create text chunks for better retrieval
        _texts = []
        chunk_to_page_key = []  # Track which page each chunk belongs to

        st.info("🔄 Creating text chunks for vector database...")
        for i, (page_key, text) in enumerate(zip(page_keys, texts)):
            # Simple chunking: split on paragraphs and limit size
            paragraphs = text.split("\n\n")
            for para in paragraphs:
                if len(para.strip()) > 10:  # Only keep non-trivial paragraphs
                    _texts.append(f"PAGE {page_key}: {para}")
                    chunk_to_page_key.append(page_key)  # Store page key

        # If no valid text chunks, return failure
        if not _texts:
            st.error("❌ No valid text chunks found in manual data")
            return None, None

        # Store page keys in session state
        if "rag" not in st.session_state:
            st.session_state["rag"] = {}
        st.session_state["rag"]["page_keys"] = chunk_to_page_key

        # Save page keys for future use
        keys_path = os.path.join(os.path.dirname(INDEX_PATH), "page_keys.pkl")
        os.makedirs(os.path.dirname(keys_path), exist_ok=True)
        with open(keys_path, 'wb') as f:
            pickle.dump(chunk_to_page_key, f)

        # Create embeddings and build FAISS index
        try:
            import faiss
        except ImportError:
            st.error("❌ FAISS not installed. Please install with: pip install faiss-cpu or faiss-gpu")
            return None, None

        # Get embeddings for text chunks
        st.info("🧠 Creating embeddings for text chunks...")
        embeddings = embed_chunks(_texts)
        if embeddings is None:
            st.error("❌ Failed to create embeddings")
            return None, None

        # Create FAISS index
        st.info("🔍 Building FAISS index...")
        dimension = embeddings.shape[1]
        _index = faiss.IndexFlatL2(dimension)
        _index.add(embeddings)

        # Save index and chunks for future use
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        faiss.write_index(_index, INDEX_PATH)

        chunks_path = os.path.join(os.path.dirname(INDEX_PATH), "chunks.pkl")
        with open(chunks_path, 'wb') as f:
            pickle.dump(_texts, f)

        st.success("✅ Vector index built successfully!")
        return _index, _texts

    except Exception as e:
        error_details = traceback.format_exc()
        log_error("index_building", str(e), {"traceback": error_details})
        st.error(f"❌ Error building index: {str(e)}")
        return None, None


# --- Model Management Functions ---
def get_model_pipeline(model_config):
    """
    Load and return the specified model pipeline

    Args:
        model_config: Model configuration dictionary

    Returns:
        HuggingFace pipeline or None if failed
    """
    global _loaded_models

    # Return cached model if available
    if model_config["name"] in _loaded_models:
        return _loaded_models[model_config["name"]]

    # Map model name to user-friendly coach name
    coach_name = "Youth Coach"  # Default
    if model_config["name"] == MODEL_CONFIGS["large"]["name"]:
        coach_name = "Master Coach Olympus"
    elif model_config["name"] == MODEL_CONFIGS["medium"]["name"]:
        coach_name = "Pro Coach"

    # Create status placeholder - only show when actually loading a new model
    status_placeholder = st.empty()
    status_placeholder.info(f"⚙ Initializing {coach_name}...")


    try:
        # Verify the path exists
        model_path = model_config["path"]
        if not os.path.exists(model_path):
            status_placeholder.error(f"❌ Coach initialization failed")

            # Try to get from secrets.toml
            model_path = st.secrets.get("llm", {}).get("model_path")
            if not model_path or not os.path.exists(model_path):
                status_placeholder.error("❌ Cannot access training knowledge base")
                return None

        # Verify transformers is available
        if not TRANSFORMERS_AVAILABLE:
            status_placeholder.error("❌ Transformers library not installed")
            return None

        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        # Try loading tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True
            )
        except Exception as e:
            error_msg = f"Error loading tokenizer: {e}"
            log_error("tokenizer_loading", error_msg, {"model_path": model_path})
            status_placeholder.error(f"❌ {error_msg}")
            return None

        # Unload other models if memory is limited
        available_memory = estimate_available_memory()
        if available_memory < model_config["memory_required"] * 1.2:  # 20% buffer
            unload_unused_models(keep_model=model_config["name"])

        # Try optimized loading
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
        except Exception as e:
            status_placeholder.warning(f"⚠️ Optimized loading failed: {e}")

            # Fallback to basic loading
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    local_files_only=True
                ).to("cpu")
            except Exception as e2:
                error_msg = f"Failed to load model: {e2}"
                log_error("model_loading", error_msg, {"original_error": str(e)})
                status_placeholder.error(f"❌ {error_msg}")
                return None

        # Ensure padding token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if hasattr(model.config, 'pad_token_id') and model.config.pad_token_id is None:
            model.config.pad_token_id = model.config.eos_token_id

        # Create pipeline
        model_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=model_config["max_tokens"],
            temperature=0.7,
            top_p=0.9
        )

        # Cache the loaded model
        _loaded_models[model_config["name"]] = model_pipeline
        status_placeholder.success(f"✅ {coach_name} is ready!")
        time.sleep(1)  # Keep success message visible briefly
        status_placeholder.empty()

        return model_pipeline

    except Exception as e:
        error_details = traceback.format_exc()
        log_error("model_loading", str(e), {"traceback": error_details})
        status_placeholder.error(f"❌ Error loading model: {str(e)}")
    return None


def get_llm_pipeline(force_reload=False) -> Any:
    """
    Get the most appropriate LLM pipeline based on available resources

    Args:
        force_reload: Force reload the model even if cached

    Returns:
        HuggingFace pipeline or None if failed
    """
    global _llm_pipeline

    # Return cached pipeline if available and not forced to reload
    if _llm_pipeline is not None and not force_reload:
        return _llm_pipeline

    # Try to get path from secrets
    model_path = st.secrets.get("llm", {}).get("model_path")
    if not model_path:
        st.error("❌ model_path missing in secrets.toml")
        return None

    # Select model based on available memory
    available_memory = estimate_available_memory()

    if available_memory >= MODEL_CONFIGS["large"]["memory_required"]:
        config = MODEL_CONFIGS["large"]
    elif available_memory >= MODEL_CONFIGS["medium"]["memory_required"]:
        config = MODEL_CONFIGS["medium"]
    else:
        config = MODEL_CONFIGS["tiny"]

    # Override with path from secrets
    config["path"] = model_path

    # Load pipeline
    _llm_pipeline = get_model_pipeline(config)
    return _llm_pipeline


# --- Query Processing Functions ---
def create_prompt(query: str, context: str) -> str:
    """
    Create a prompt for the LLM

    Args:
        query: User's question
        context: Retrieved context from manual

    Returns:
        Formatted prompt
    """
    prompt = f"""
You are Coach Olympus 🏛️, the elite AI trainer for the Indonesia MMA Youth Excellence Academy.
Respond only using the provided manual context. If the answer is not present, reply:
"The manual does not contain this information. Please ask about techniques, training methods, or academy policies that might be covered in the training manual."

📘 CONTEXT:
{context}

❓ QUESTION:
{query}

💬 ANSWER:"""

    # Store prompt in session state for debugging
    st.session_state["last_prompt"] = prompt

    return prompt


def process_with_adaptive_model(
        query: str,
        context: str,
        page_refs: List[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        suppress_status: bool = False
) -> Tuple[str, List[str], Dict[str, Any]]:
    """
    Process a query using the adaptive model selection.

    Args:
        query: User's question
        context: Retrieved context
        page_refs: Source references
        timeout: Maximum processing time in seconds
        suppress_status: Whether to suppress status messages for user-facing interfaces

    Returns:
        Tuple of (answer_text, source_references, metadata)
    """
    # Create placeholders if not suppressing status
    progress_placeholder = st.empty() if not suppress_status else None
    status_placeholder = st.empty() if not suppress_status else None
    progress_bar = progress_placeholder.progress(0) if progress_placeholder else None

    # Remaining function implementation...
    # Update all placeholders to check if they exist before using:
    # Example:
    # if status_placeholder:
    #     status_placeholder.info("Loading model...")
    # And for the model selection display, only show to admins and when not suppressed:
    # if st.session_state.get("is_admin", False) and not suppress_status:
    #     st.write(f"Selected model: {model_name}")
    # Rest of function...

    # Prepare metadata for return
    metadata = {
        "model": None,
        "tokens_used": None,
        "latency_s": None
    }

    # Measure start time
    start_time = time.time()

    try:
        # Select appropriate model based on query complexity
        model_config = select_model_for_query(query)
        metadata["model"] = model_config["name"]

        # Get user-friendly coach name based on model type
        coach_name = "Youth Coach"  # Default
        if model_config["name"] == MODEL_CONFIGS["large"]["name"]:
            coach_name = "Master Coach Olympus"
        elif model_config["name"] == MODEL_CONFIGS["medium"]["name"]:
            coach_name = "Pro Coach"

        if status_placeholder:
            # Use user-friendly coach naming instead of technical model name
            status_placeholder.empty()  # Clear previous messages first
            status_placeholder.info(f"🧠 {coach_name} is generating your answer...")

        # Create LLM prompt
        prompt = create_prompt(query, context)

        # Load model pipeline
        llm = get_model_pipeline(model_config)
        if llm is None:
            # If preferred model fails, try fallback to basic LLM
            if status_placeholder:
                status_placeholder.warning("⚠️ Switching to a different coach due to technical reasons...")
            llm = get_llm_pipeline()

        if llm is None:
                # If all models fail, return manual context with explanation
                if progress_placeholder:
                    progress_placeholder.empty()
                if status_placeholder:
                    status_placeholder.empty()

                fallback_answer = f"""
I found relevant information in the manual but couldn't generate a complete answer due to 
technical issues with the AI model.

Here's what the manual says about your question:

{context[:500]}...

This information comes from {', '.join(page_refs)}. You can read the complete content in the Manual Reader section.
"""
                # Update metadata
                metadata["latency_s"] = round(time.time() - start_time, 2)

                return fallback_answer, page_refs, metadata

        # Initialize generation variables
        result = None
        timeout_occurred = False
        quick_generation_attempted = False

        # Update progress in increments
        for i in range(10):
            # Check if we've exceeded timeout
            if time.time() - start_time > timeout:
                if status_placeholder:
                    status_placeholder.warning("⏱️ Taking too long! Providing partial answer...")
                timeout_occurred = True
                break

            # Update progress display
            if progress_bar:
                progress_percent = min(0.95, (i + 1) / 10)
                progress_bar.progress(progress_percent)

            # Try generation with shorter max_tokens on first attempt
            if not quick_generation_attempted:
                try:
                    from transformers import pipeline

                    # Use shorter tokens and higher temperature for faster response
                    quick_pipeline = pipeline(
                        "text-generation",
                        model=llm.model,
                        tokenizer=llm.tokenizer,
                        max_new_tokens=128,  # Shorter for quick response
                        temperature=0.7,
                        top_p=0.9
                    )

                    with torch.inference_mode():
                        result = quick_pipeline(prompt)
                    break
                except Exception as e:
                    # If quick generation fails, continue with standard approach
                    quick_generation_attempted = True
                    log_error("quick_generation", str(e))

            # Wait before next update
            time.sleep(timeout / 20)

        # Handle timeout or generation failure
        if result is None or timeout_occurred:
            # Try once more with even shorter context and tokens
            try:
                # Create a shorter context with just first paragraph from each source
                short_context = "\n".join([c.split("\n")[0] for c in context.split("\n---\n")])
                short_prompt = create_prompt(query, short_context[:500])

                # Try final generation with minimal settings
                with torch.inference_mode():
                    final_pipeline = pipeline(
                        "text-generation",
                        model=llm.model,
                        tokenizer=llm.tokenizer,
                        max_new_tokens=64,  # Very short
                        temperature=0.9,  # Higher temp for faster generation
                        top_p=0.95
                    )
                    result = final_pipeline(short_prompt)
            except Exception as final_e:
                log_error("final_generation", str(final_e))
                result = None

        # Process the final result
        if result is not None:
            # Extract answer from generated text
            output = result[0]["generated_text"]
            marker = "💬 ANSWER:"
            idx = output.rfind(marker)
            answer = output[idx + len(marker):].strip() if idx != -1 else output.strip()

            # Update progress display
            if progress_bar:
                progress_bar.progress(1.0)

            # Update metadata
            metadata["tokens_used"] = result[0].get("generated_token_count", None)
            metadata["latency_s"] = round(time.time() - start_time, 2)

            # Clean up progress indicators
            if progress_placeholder:
                progress_placeholder.empty()
            if status_placeholder:
                status_placeholder.empty()

            # Log the successful query
            log_query_to_file(
                email=st.session_state.get("user_email", "guest"),
                question=query,
                sources=page_refs,
                model=model_config["name"],
                tokens_used=metadata["tokens_used"],
                latency_s=metadata["latency_s"]
            )

            # Return the answer and metadata
            return answer, page_refs, metadata

        # If we got here, all generation attempts failed
        if progress_placeholder:
            progress_placeholder.empty()
        if status_placeholder:
            status_placeholder.empty()

        # Provide retrieval results instead
        retrieval_answer = f"""
I found relevant information in the manual but couldn't generate a complete answer.

Here's what the manual says about your question:

{context[:500]}...

This information comes from {', '.join(page_refs)}. You can read the complete content in the Manual Reader section.
"""
        # Update metadata
        metadata["latency_s"] = round(time.time() - start_time, 2)

        # Log the query
        log_query_to_file(
            email=st.session_state.get("user_email", "guest"),
            question=query,
            sources=page_refs,
            model="retrieval_only",
            latency_s=metadata["latency_s"]
        )

        return retrieval_answer, page_refs, metadata

    except Exception as e:
        # Handle any unexpected errors
        error_details = traceback.format_exc()
        log_error("processing", str(e), {"traceback": error_details})

        # Clean up UI elements
        if progress_placeholder:
            progress_placeholder.empty()
        if status_placeholder:
            status_placeholder.empty()

        # Provide a graceful error message with retrieval results
        error_answer = f"""
I encountered an error while generating your answer, but I found these relevant sections:

{context[:300]}...

This information comes from {', '.join(page_refs)}. You can read the complete content in the Manual Reader section.

Technical details (for admin): {str(e)}
"""
        # Update metadata
        metadata["latency_s"] = round(time.time() - start_time, 2)

        # Log the error
        log_query_to_file(
            email=st.session_state.get("user_email", "guest"),
            question=query,
            sources=page_refs,
            model="error",
            latency_s=metadata["latency_s"]
        )

        return error_answer, page_refs, metadata


def query_rag_agent(
        query: str,
        model_path: Optional[str] = None,
        top_k: int = TOP_K_CHUNKS,
        llm: Optional[Any] = None,
        timeout: int = DEFAULT_TIMEOUT
) -> Tuple[str, List[str]]:
    """
    Process a question using the RAG system with intelligent model selection and robust fallbacks.

    Args:
        query: User's question
        model_path: Path to LLM model directory (optional, will use from secrets)
        top_k: Number of chunks to retrieve
        llm: Optional pre-loaded LLM pipeline
        timeout: Maximum time to wait for generation in seconds

    Returns:
        Tuple of (answer_text, source_references)
    """
    # Check for special rebuild command
    if query == "__rebuild_index__":
        status_placeholder = st.empty()
        status_placeholder.info("🔄 Rebuilding RAG index...")

        # Force rebuild the index
        index, texts = build_or_load_faiss_index(force_rebuild=True)

        status_placeholder.empty()
        if index is not None:
            return "✅ Index rebuilt successfully!", []
        else:
            return "❌ Failed to rebuild index. Check logs for details.", []

    # Get model path from secrets if not provided
    if not model_path:
        model_path = st.secrets.get("llm", {}).get("model_path")
    if not model_path:
        return "❌ model_path missing in secrets.toml", []

    # Create UI placeholder elements
    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    # Step 1: Load resources
    status_placeholder.info("🔍 Step 1/4: Loading knowledge base...")
    model_path = os.path.abspath(model_path)

    # Load or build FAISS index
    index, texts = build_or_load_faiss_index(model_path)
    if index is None:
        progress_placeholder.empty()
        status_placeholder.empty()
        return "❌ RAG index not ready. Contact administrator.", []

    # Step 2: Encode query
    status_placeholder.info("🔍 Step 2/4: Analyzing your question...")
    tokenizer, emb_model = get_deepseek_models(model_path)

    if tokenizer is None or emb_model is None:
        progress_placeholder.empty()
        status_placeholder.empty()
        return "❌ Embedding models not available. Contact administrator.", []

    try:
        # Handle different embedding approaches
        if hasattr(emb_model, 'encode') and callable(getattr(emb_model, 'encode')):
            # Using SentenceTransformer
            vector = emb_model.encode([query], convert_to_numpy=True).astype("float32")
        else:
            # Using HuggingFace transformer
            encoded = tokenizer(query, return_tensors="pt", truncation=True)
            with torch.no_grad():
                vector = emb_model(**encoded).last_hidden_state[:, 0, :].cpu().numpy().astype("float32")
    except Exception as e:
        error_msg = f"Embedding failed: {e}"
        log_error("query_embedding", error_msg)
        progress_placeholder.empty()
        status_placeholder.empty()
        return f"❌ {error_msg}", []

    # Step 3: Retrieve relevant content
    status_placeholder.info("🔍 Step 3/4: Finding relevant content in the manual...")
    try:
        distances, indices = index.search(vector, top_k)

        # Validate indices to ensure they're in range
        valid_indices = [i for i in indices[0] if i < len(texts)]

        # Create context chunks from valid indices only
        context_chunks = [texts[i] for i in valid_indices]

        # Check if the page_keys exists and handle safely
        page_keys = st.session_state.get("rag", {}).get("page_keys", [])

        # Create page references safely
        if page_keys and len(page_keys) > 0:
            # Only use valid indices and ensure each index is in range for page_keys
            page_refs = []
            for i in valid_indices:
                if i < len(page_keys):
                    page_refs.append(page_keys[i])
                else:
                    page_refs.append(f"Unknown-{i}")
        else:
            # Create generic references if page_keys is missing or empty
            page_refs = [f"Section {i + 1}" for i, _ in enumerate(context_chunks)]

        # Store generic page_keys for future use if missing
        if not page_keys:
            if "rag" not in st.session_state:
                st.session_state["rag"] = {}
            st.session_state["rag"]["page_keys"] = [f"Section {i + 1}" for i in range(len(texts))]

        # Combine chunks into context
        if context_chunks:
            context = "\n---\n".join(context_chunks)
        else:
            # Handle case with no relevant chunks
            context = "No directly relevant content was found in the manual."
            page_refs = ["Not specified"]
    except Exception as e:
        error_msg = f"Search failed: {str(e)}"
        log_error("search_error", error_msg, {"traceback": traceback.format_exc()})
        progress_placeholder.empty()
        status_placeholder.empty()
        return f"❌ {error_msg}", []

    # Step 4: Generate answer - Updated to use coach terminology
    status_placeholder.info("🔍 Step 4/4: Coach Olympus is preparing your answer...")

    if llm is not None:
        # Legacy mode - use the provided LLM directly
        prompt = create_prompt(query, context)

        # Attempt generation with timeout
        try:
            # Initialize generation variables
            result = None
            quick_generation_attempted = False
            start_time = time.time()
            progress_bar = progress_placeholder.progress(0)

            # Update progress bar in increments
            for i in range(10):
                # Check if we've exceeded timeout
                if time.time() - start_time > timeout:
                    status_placeholder.warning("⏱️ Taking too long! Providing partial answer...")
                    break

                # Update progress display
                progress_percent = min(0.95, (i + 1) / 10)
                progress_bar.progress(progress_percent)

                # Try generation with shorter max_tokens first time through
                if not quick_generation_attempted:
                    try:
                        from transformers import pipeline
                        quick_pipeline = pipeline(
                            "text-generation",
                            model=llm.model,
                            tokenizer=llm.tokenizer,
                            max_new_tokens=128,  # Even shorter for quick response
                            temperature=0.7,
                            top_p=0.9
                        )
                        result = quick_pipeline(prompt)
                        break
                    except Exception as e:
                        # If quick generation fails, continue with timeout loop
                        quick_generation_attempted = True
                        log_error("quick_generation", str(e))

                # Wait a bit before next update
                time.sleep(timeout / 20)

            # Clear UI elements
            progress_placeholder.empty()
            status_placeholder.empty()

            # Calculate total processing time
            duration = time.time() - start_time

            # Handle timeout or generation failure
            if result is None or time.time() - start_time > timeout:
                timeout_answer = f"""
I found relevant information in the manual but couldn't complete a full answer in time.

Here are the key points from the relevant sections:

{context[:500]}...

This information comes from {', '.join(page_refs)}. You can read the complete content in the Manual Reader section.
"""
                # Log the query
                log_query_to_file(
                    email=st.session_state.get("user_email", "guest"),
                    question=query,
                    sources=page_refs,
                    model="Coach Olympus",  # Use coach name instead of technical name
                    latency_s=round(duration, 2)
                )
                return timeout_answer, page_refs

            # Process successful generation
            output = result[0]["generated_text"]
            marker = "💬 ANSWER:"
            idx = output.rfind(marker)
            answer = output[idx + len(marker):].strip() if idx != -1 else output.strip()

            # Log the successful query
            log_query_to_file(
                email=st.session_state.get("user_email", "guest"),
                question=query,
                sources=page_refs,
                model="Coach Olympus",  # Use coach name instead of technical name
                tokens_used=result[0].get("generated_token_count", None),
                latency_s=round(duration, 2)
            )

            return answer, page_refs

        except Exception as e:
            # Handle errors during generation
            error_details = traceback.format_exc()
            log_error("generation", str(e), {"traceback": error_details})

            # Clear UI elements
            progress_placeholder.empty()
            status_placeholder.empty()

            # Calculate duration and log
            duration = time.time() - start_time
            log_query_to_file(
                email=st.session_state.get("user_email", "guest"),
                question=query,
                sources=page_refs,
                model="Coach Olympus",  # Use coach name instead of technical name
                latency_s=round(duration, 2)
            )

            # Return retrieval results even if generation failed
            fallback_answer = f"""
I found relevant information in the manual but couldn't generate a complete answer.

Here's what the manual says about your question:

{context[:500]}...

This information comes from {', '.join(page_refs)}. You can read the complete content in the Manual Reader section.
"""
            return fallback_answer, page_refs
    else:
        # Use adaptive model selection approach
        answer, sources, metadata = process_with_adaptive_model(query, context, page_refs, timeout)
        progress_placeholder.empty()
        status_placeholder.empty()

        # Display model info to admin only, with coach name conversion
        if st.session_state.get("is_admin", False) and metadata.get("model"):
            model_name = metadata.get("model")
            coach_name = "Youth Coach"
            if model_name == MODEL_CONFIGS["large"]["name"]:
                coach_name = "Master Coach Olympus"
            elif model_name == MODEL_CONFIGS["medium"]["name"]:
                coach_name = "Pro Coach"

            st.caption(f"Generated using {coach_name} in {metadata.get('latency_s', 0):.2f}s")

        return answer, sources


# Path constants
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "data", "faiss_index")

# --- Utility Functions ---
def get_system_info() -> Dict[str, Any]:
    """
    Get system and environment information

    Returns:
        Dictionary with system info
    """
    global _system_info

    # Return cached info if available
    if _system_info is not None:
        return _system_info

    _system_info = {
        "python_version": sys.version,
        "libraries": {
            "transformers": transformers_version if TRANSFORMERS_AVAILABLE else "Not installed",
            "sentence_transformers": "Installed" if SENTENCE_TRANSFORMERS_AVAILABLE else "Not installed",
            "torch": torch.__version__ if 'torch' in sys.modules else "Not installed",
            "streamlit": st.__version__ if 'streamlit' in sys.modules else "Not installed",
        },
        "cuda": {
            "available": torch.cuda.is_available() if 'torch' in sys.modules else False,
            "device_count": torch.cuda.device_count() if 'torch' in sys.modules and torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(
                0) if 'torch' in sys.modules and torch.cuda.is_available() else "N/A",
        },
        "memory": {
            "available_mb": estimate_available_memory(),
        },
        "models": {
            "embedding_model": getattr(_embedding_model, "__class__.__name__", "Not loaded"),
            "loaded_llms": list(_loaded_models.keys()) if _loaded_models else [],
        }
    }

    return _system_info


def clear_model_cache() -> bool:
    """
    Clear all cached models to free memory

    Returns:
        Success flag
    """
    global _index, _texts, _tokenizer, _embedding_model
    global _llm_pipeline, _sentence_transformer, _loaded_models

    try:
        # Clear all model caches
        _tokenizer = None
        _embedding_model = None
        _llm_pipeline = None
        _sentence_transformer = None
        _loaded_models = {}

        # Keep _index and _texts for faster reuse

        # Force garbage collection
        import gc
        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return True
    except Exception as e:
        log_error("cache_clearing", str(e))
        return False


# For testing
if __name__ == "__main__":
    test_path = os.path.join(BASE_DIR, "models", "deepseek")
    test_query = "What are the main MMA disciplines taught at the academy?"

    print("Testing RAG agent...")
    print(f"Using model path: {test_path}")
    print(f"Test query: {test_query}")

    response, refs = query_rag_agent(test_query, model_path=test_path)
    print(f"Response: {response}")
    print(f"References: {refs}")
    # This part is for testing the RAG agent functionality directly.
    print("Testing RAG Agent (run from project root or ensure paths are correct)...")


    # Simulate Streamlit secrets for local testing if needed
    class MockSecrets:
        def __init__(self):
            self.secrets = {
                "llm": {"model_path": "D:/models/deepseek-7b"},  # Replace with your actual test model path
                "embedding": {"model_name": "all-MiniLM-L6-v2"}  # Default or your choice
            }

        def get(self, key, default=None):
            return self.secrets.get(key, default)


    # st.secrets = MockSecrets() # Uncomment and adjust if testing locally

    if not hasattr(st, 'secrets'):  # Simple check if running in streamlit context or not
        print("Warning: Not running in Streamlit context. st.secrets might not be available.")
        print("Ensure manual.json exists at: ", MANUAL_JSON_PATH)
        print("Ensure FAISS index will be saved/loaded from: ", FAISS_INDEX_PATH)

    idx, txts, p_keys = build_or_load_faiss_index()
    if idx and txts:
        print(f"FAISS index loaded/built with {idx.ntotal} entries.")
        test_query = "What are the core principles of the academy?"
        print(f"\nQuerying with: 	{test_query}")

        # Determine model path for testing
        model_for_test = st.secrets.get("llm", {}).get("model_path") if hasattr(st,
                                                                                'secrets') else "D:/models/deepseek-7b"
        if not model_for_test:
            print("Skipping LLM query as no model_path is configured or provided for test.")
        else:
            answer, sources = query_rag_agent(test_query, model_path=model_for_test)
            print(f"\nAnswer:\n{answer}")
            print(f"\nSources (Page Keys): {sources}")
    else:
        print("Failed to load/build FAISS index for testing.")

