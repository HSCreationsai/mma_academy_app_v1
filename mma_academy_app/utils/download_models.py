# utils/model_manager.py
import os
import json
import time
import torch
import numpy as np
import streamlit as st
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import platform
import psutil
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Model Configuration ---
class ModelType(Enum):
    TINY = "tiny"  # For simple queries, ultra-fast
    MEDIUM = "medium"  # Balanced option
    LARGE = "large"  # Full model for complex queries


@dataclass
class ModelConfig:
    name: str
    path: str
    type: ModelType
    max_tokens: int
    memory_required: int  # Approx RAM needed in MB
    strengths: List[str]  # What this model is good at
    best_for: List[str]  # Types of queries it's best suited for


# Pre-configured models
AVAILABLE_MODELS = {
    "tiny": ModelConfig(
        name="TinyLlama-1.1B",
        path="D:/LLM_Models/tinyllama",  # You'll need to download this
        type=ModelType.TINY,
        max_tokens=128,
        memory_required=2048,  # ~2GB
        strengths=["Speed", "Basic information", "Follow-up questions"],
        best_for=["simple", "clarification", "follow-up", "hello", "hi", "greetings"]
    ),
    "medium": ModelConfig(
        name="Phi-2",
        path="D:/LLM_Models/phi-2",  # You'll need to download this
        type=ModelType.MEDIUM,
        max_tokens=256,
        memory_required=5120,  # ~5GB
        strengths=["Balance of speed/accuracy", "Good explanations", "Technique descriptions"],
        best_for=["techniques", "explain", "describe", "what is", "how to"]
    ),
    "large": ModelConfig(
        name="DeepSeek-7B",
        path="D:/LLM_Models/deepseek-7b-manual",  # Already downloaded
        type=ModelType.LARGE,
        max_tokens=512,
        memory_required=14336,  # ~14GB
        strengths=["Detailed knowledge", "Complex reasoning", "Comprehensive answers"],
        best_for=["complex", "detailed", "compare", "analyze", "why", "philosophy", "strategy"]
    )
}


# --- Model Selection Logic ---
def classify_query_complexity(query: str) -> float:
    """Rate query complexity from 0.0 (simple) to 1.0 (complex)"""
    # Simple heuristics-based classifier
    query = query.lower()

    # Count question words (more = more complex)
    question_words = ["why", "how", "compare", "explain", "analyze", "difference", "philosophy", "strategy"]
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


def estimate_available_memory() -> int:
    """Estimate available system memory in MB"""
    try:
        return int(psutil.virtual_memory().available / (1024 * 1024))
    except:
        # Fallback to a conservative estimate if psutil not available
        return 4096  # Assume 4GB available


def select_model_for_query(query: str) -> ModelConfig:
    """
    Select the most appropriate model based on:
    1. Query complexity
    2. Available system resources
    3. Query keywords and patterns
    """
    query = query.lower()
    complexity = classify_query_complexity(query)
    available_memory = estimate_available_memory()

    # Check for keywords that suggest specific model strengths
    for model_key, config in AVAILABLE_MODELS.items():
        for keyword in config.best_for:
            if keyword in query:
                # Check if we have enough memory
                if available_memory >= config.memory_required:
                    return config

    # Base selection on complexity if no keywords matched
    if complexity < 0.3 and available_memory >= AVAILABLE_MODELS["tiny"].memory_required:
        return AVAILABLE_MODELS["tiny"]
    elif complexity < 0.7 and available_memory >= AVAILABLE_MODELS["medium"].memory_required:
        return AVAILABLE_MODELS["medium"]
    elif available_memory >= AVAILABLE_MODELS["large"].memory_required:
        return AVAILABLE_MODELS["large"]

    # Fallback to the model that fits in available memory
    if available_memory >= AVAILABLE_MODELS["medium"].memory_required:
        return AVAILABLE_MODELS["medium"]
    else:
        return AVAILABLE_MODELS["tiny"]


# --- Model Loading & Caching ---
_loaded_models = {}  # Cache for loaded models


def get_model_pipeline(model_config: ModelConfig):
    """Load and return the specified model pipeline"""
    global _loaded_models

    # Return cached model if available
    if model_config.name in _loaded_models:
        return _loaded_models[model_config.name]

    # Import here to allow fallback if not installed
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.path,
            local_files_only=True
        )

        # Try optimized loading
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_config.path,
                torch_dtype=torch.float16,
                device_map="auto",
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
        except Exception as e:
            # Fallback to basic loading
            model = AutoModelForCausalLM.from_pretrained(
                model_config.path,
                local_files_only=True
            ).to("cpu")

        # Ensure padding token is set
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

        # Create optimized pipeline
        model_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=model_config.max_tokens,
            temperature=0.7,
            top_p=0.9
        )

        # Cache the loaded model
        _loaded_models[model_config.name] = model_pipeline
        return model_pipeline

    except Exception as e:
        st.error(f"❌ Error loading {model_config.name}: {e}")
        return None


def unload_unused_models(keep_model: str = None):
    """Unload models to free memory"""
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


# --- Query Processing Functions ---
def process_with_model(query: str, context: str, model_config: ModelConfig, timeout: int = 60) -> Tuple[str, dict]:
    """
    Process a query using the specified model

    Args:
        query: User's question
        context: Retrieved context from documents
        model_config: Configuration for model to use
        timeout: Maximum time to wait for generation in seconds

    Returns:
        Tuple of (answer, metadata)
    """
    # Create the prompt based on the model's capabilities
    if model_config.type == ModelType.TINY:
        # Simplified prompt for small models
        prompt = f"""
You are Coach Olympus, the MMA Academy trainer.
CONTEXT: {context[:1000]}
QUESTION: {query}
ANSWER:"""
    else:
        # More detailed prompt for larger models
        prompt = f"""
You are Coach Olympus, the elite AI trainer for the Indonesia MMA Youth Excellence Academy.
Respond only using the provided manual context. If the answer is not present, reply:
"The manual does not contain this information."

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""

    # Get the model pipeline
    model_pipeline = get_model_pipeline(model_config)

    if not model_pipeline:
        return f"I couldn't load the {model_config.name} model. Please try again later.", {
            "model": model_config.name,
            "error": "Model loading failed",
            "success": False
        }

    try:
        # Start timing
        start_time = time.time()

        # Generate the response
        result = model_pipeline(prompt, max_new_tokens=model_config.max_tokens)
        output = result[0]["generated_text"]

        # Extract just the answer part
        marker = "ANSWER:"
        idx = output.rfind(marker)
        answer = output[idx + len(marker):].strip() if idx != -1 else output.strip()

        # Calculate duration
        duration = time.time() - start_time

        return answer, {
            "model": model_config.name,
            "duration": duration,
            "success": True,
            "tokens": len(result[0].get("generated_token_ids", [])),
        }
    except Exception as e:
        return f"Error generating response with {model_config.name}: {str(e)}", {
            "model": model_config.name,
            "error": str(e),
            "success": False
        }
# download_models.py
from huggingface_hub import snapshot_download

MODELS = {
    "phi-2": {
        "repo": "microsoft/phi-2",
        "path": "D:/LLM_Models/phi-2"
    },
    "tinyllama": {
        "repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "path": "D:/LLM_Models/tinyllama"
    }
}

for name, cfg in MODELS.items():
    print(f"⬇️ Downloading {name} from {cfg['repo']} ...")
    snapshot_download(
        repo_id=cfg["repo"],
        local_dir=cfg["path"],
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print(f"✅ {name} downloaded to {cfg['path']}")
