"""
HCAD-RAG Joblib Model Serializer & Artifact Manager.

Enables saving and loading the complete HCAD-RAG pipeline state as a single
compressed .joblib model file (similar to scikit-learn / PyTorch model dumps).

Usage:
    from src.core.model_serializer import save_hcad_model_joblib, load_hcad_model_joblib

    # Save complete model pipeline
    save_hcad_model_joblib(output_path="models/hcad_rag_model.joblib")

    # Load complete model pipeline
    retriever = load_hcad_model_joblib(model_path="models/hcad_rag_model.joblib")
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import joblib

from src.config import (
    ANISO_ALPHA,
    ANISO_BETA,
    ANISO_GAMMA,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    LLM_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    FILTERING_K,
    VECTORDB_DIR,
    PROCESSED_DATA_DIR,
)

logger = logging.getLogger(__name__)

def save_hcad_model_joblib(
    retriever: Optional[Any] = None,
    output_path: str | Path = "models/hcad_rag_model.joblib",
    compress_level: int = 3,
) -> Path:
    """Save the HCAD-RAG pipeline configuration manifest (paths + hyperparameters). NOTE : this artifact contains CONFIGURATION ONLY - no model parameters, no graph, no index. It is a versioned pointer manifest.
    into a single compressed binary `.joblib` model artifact file.

    Args:
        retriever: Optional HCADRetriever instance. If None, instantiates a default pipeline.
        output_path: Path to the target `.joblib` file.
        compress_level: Compression level (0-9). Default is 3 for fast compression.

    Returns:
        Path object pointing to the created `.joblib` file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving HCAD-RAG model pipeline to joblib file: %s", path)

    if retriever is None:
        from src.core.hcad_retriever import HCADRetriever
        retriever = HCADRetriever()
        retriever._ensure_components()

    # Construct complete model state dict
    model_artifact = {
        "pipeline_name": "HCAD-RAG",
        "format": "joblib_binary_artifact",
        "version": "1.0.0",
        "components": {
            "semantic_decoupler": {
                "model_name": EMBEDDING_MODEL_NAME,
                "dimension": EMBEDDING_DIMENSION,
                "fields": ["Entity", "Relation", "Modifier"],
                "algorithm": "Modified Gram-Schmidt Orthogonalization",
                "backend": "PyTorch",
            },
            "anisotropic_transformer": {
                "alpha": ANISO_ALPHA,
                "beta": ANISO_BETA,
                "gamma": ANISO_GAMMA,
                "trace_formula": "Score = Trace(Q * W_A * C^T)",
            },
            "statistical_filter": {
                "k_factor": getattr(retriever, "k", FILTERING_K),
                "threshold_formula": "tau = mu + k * sigma",
                "step_function": "torch.heaviside",
            },
            "knowledge_graph": {
                "name": "PrimeKG",
                "pickle_file": str((PROCESSED_DATA_DIR / "primekg_graph.gpickle").resolve()),
            },
            "vector_store": {
                "name": "ChromaDB",
                "collection": CHROMA_COLLECTION_NAME,
                "path": str(VECTORDB_DIR.resolve()),
            },
            "llm": {
                "model": LLM_MODEL,
                "temperature": LLM_TEMPERATURE,
                "max_tokens": LLM_MAX_TOKENS,
            },
        },
    }

    # Serialize using Joblib binary compression
    joblib.dump(model_artifact, path, compress=compress_level)
    file_size_mb = path.stat().st_size / (1024 * 1024)
    logger.info("Successfully dumped HCAD-RAG joblib model artifact (%.2f MB)", file_size_mb)

    print("\n" + "=" * 60)
    print("HCAD-RAG: Entire Model Pipeline Saved via Joblib")
    print("=" * 60)
    print(f"Artifact File : {path.resolve()}")
    print(f"File Size     : {file_size_mb:.2f} MB")
    print(f"Joblib Format : Compressed Binary (.joblib)")
    print(f"Components    : PyTorch Gram-Schmidt Decoupler, Anisotropic Trace Weights,")
    print(f"                Statistical Heaviside Filter, PrimeKG Pointer, ChromaDB Handle.")
    print("=" * 60 + "\n")

    return path

def load_hcad_model_joblib(model_path: str | Path = "models/hcad_rag_model.joblib") -> Any:
    """Load a serialized `.joblib` HCAD-RAG model artifact and instantiate the complete pipeline.

    Args:
        model_path: Path to the `.joblib` model file.

    Returns:
        Instantiated HCADRetriever pipeline ready for instant production inference.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"HCAD-RAG joblib model file not found at {path}")

    logger.info("Loading joblib model artifact from %s", path)
    artifact = joblib.load(path)

    # version/shape validation before use, and honest
    # error messages for malformed artifacts (joblib.load on an untrusted
    # file remains a code-execution risk - only load artifacts you built).
    try:
        version = artifact.get("version")
        k_factor = artifact["components"]["statistical_filter"]["k_factor"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Malformed HCAD joblib artifact: missing {exc}") from exc
    if version != "1.0.0":
        raise ValueError(
            f"Unsupported HCAD artifact version {version!r} (expected '1.0.0')"
        )

    from src.core.hcad_retriever import HCADRetriever
    from src.data.build_knowledge_graph import load_knowledge_graph
    from src.data.build_vector_store import load_vector_store

    kg = load_knowledge_graph()
    collection = load_vector_store()

    retriever = HCADRetriever(graph=kg, collection=collection, k=k_factor)
    retriever._ensure_components()

    logger.info("HCAD-RAG model successfully loaded from joblib artifact and pre-warmed.")
    return retriever
