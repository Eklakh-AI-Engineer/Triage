"""
HCAD-RAG Pipeline Serializer & Production Loader.

Provides enterprise-grade serialization and loading for the entire HCAD-RAG pipeline.
Allows saving all trained/configured components (Gram-Schmidt decoupler settings,
anisotropic weight matrices, hypergraph index metadata, vector store configuration,
and LLM sampling hyper-parameters) into a unified production bundle.

Usage:
    from src.core.pipeline_saver import HCADPipelineSaver

    # Save
    saver = HCADPipelineSaver()
    saver.save_pipeline(output_dir="models/hcad_pipeline_v1")

    # Load
    retriever = HCADPipelineSaver.load_pipeline(bundle_dir="models/hcad_pipeline_v1")
"""

import json
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any

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

class HCADPipelineSaver:
    """Enterprise Pipeline Serializer for HCAD-RAG production deployment."""

    def __init__(self, bundle_dir: Path | str = "models/hcad_pipeline_v1") -> None:
        self.bundle_dir = Path(bundle_dir)

    def save_pipeline(
        self,
        version: str = "1.0.0",
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Serialize the complete HCAD-RAG pipeline state into a production bundle.

        Args:
            version: Version tag for the pipeline bundle.
            custom_metadata: Optional dict of clinical deployment notes or environment info.

        Returns:
            Path to the saved pipeline bundle directory.
        """
        self.bundle_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Serializing HCAD-RAG pipeline to bundle: %s", self.bundle_dir)

        # 1. Pipeline Configuration Schema
        config_payload = {
            "pipeline_name": "HCAD-RAG",
            "version": version,
            "architecture": "Hypergraph-Centric Anisotropic Decoupled RAG",
            "innovations": {
                "semantic_decoupling": {
                    "method": "Modified Gram-Schmidt Orthogonalization",
                    "fields": ["Entity", "Relation", "Modifier"],
                    "embedding_model": EMBEDDING_MODEL_NAME,
                    "dimension": EMBEDDING_DIMENSION,
                    "tensor_backend": "PyTorch",
                },
                "anisotropic_transformer": {
                    "alpha_degree": ANISO_ALPHA,
                    "beta_confidence": ANISO_BETA,
                    "gamma_idf": ANISO_GAMMA,
                    "similarity_metric": "Trace(Q * W_A * C^T)",
                },
                "statistical_filter": {
                    "threshold_formula": "tau = mu + k * sigma",
                    "k_factor": FILTERING_K,
                    "step_function": "torch.heaviside",
                },
            },
            "vector_store": {
                "provider": "ChromaDB",
                "collection_name": CHROMA_COLLECTION_NAME,
                "vector_db_path": str(VECTORDB_DIR.resolve()),
            },
            "knowledge_graph": {
                "source": "PrimeKG",
                "graph_pickle_path": str((PROCESSED_DATA_DIR / "primekg_graph.gpickle").resolve()),
            },
            "llm_engine": {
                "provider": "Ollama",
                "model": LLM_MODEL,
                "temperature": LLM_TEMPERATURE,
                "max_tokens": LLM_MAX_TOKENS,
            },
            "metadata": custom_metadata or {"environment": "production", "framework": "PyTorch 2.1.2"},
        }

        # 2. Save Config JSON
        config_file = self.bundle_dir / "pipeline_config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_payload, f, indent=2)
        logger.info("Saved pipeline configuration to %s", config_file)

        # 3. Create Integrity Manifest (MD5 Hashes)
        manifest_payload = {
            "files": {
                "pipeline_config.json": self._compute_md5(config_file),
            }
        }
        kg_path = PROCESSED_DATA_DIR / "primekg_graph.gpickle"
        if kg_path.exists():
            manifest_payload["files"]["primekg_graph.gpickle"] = self._compute_md5(kg_path)

        manifest_file = self.bundle_dir / "manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_payload, f, indent=2)
        logger.info("Saved integrity manifest to %s", manifest_file)

        print("\n" + "=" * 60)
        print("HCAD-RAG: Pipeline Successfully Serialized")
        print("=" * 60)
        print(f"Bundle Directory : {self.bundle_dir.resolve()}")
        print(f"Pipeline Version : {version}")
        print(f"Components Saved : Semantic Decoupler, Anisotropic Transformer,")
        print(f"                   Statistical Heaviside Filter, PrimeKG Graph Pointer,")
        print(f"                   ChromaDB Vector Store Pointer, Ollama LLM Config.")
        print("=" * 60 + "\n")

        return self.bundle_dir

    @staticmethod
    def load_pipeline(bundle_dir: Path | str = "models/hcad_pipeline_v1"):
        """Load and instantiate a full HCADRetriever pipeline from a saved bundle.

        Args:
            bundle_dir: Path to the serialized pipeline directory.

        Returns:
            Instantiated HCADRetriever pipeline ready for instant production inference.
        """
        bundle_path = Path(bundle_dir)
        config_file = bundle_path / "pipeline_config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Pipeline config file not found at {config_file}")

        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        # the integrity manifest was written but never
        # verified. Fail closed on any checksum mismatch.
        manifest_file = bundle_path / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            for rel_name, expected_md5 in manifest.get("files", {}).items():
                target = bundle_path / rel_name
                if not target.exists():
                    raise ValueError(f"Manifest references missing file: {rel_name}")
                if HCADPipelineSaver._compute_md5(target) != expected_md5:
                    raise ValueError(f"Integrity check failed for {rel_name}")

        logger.info("Loading production HCAD-RAG pipeline v%s from %s", config.get("version"), bundle_path)

        from src.core.hcad_retriever import HCADRetriever
        from src.data.build_knowledge_graph import load_knowledge_graph
        from src.data.build_vector_store import load_vector_store

        # Load shared graph and vector store handles
        kg = load_knowledge_graph()
        collection = load_vector_store()

        k_factor = config["innovations"]["statistical_filter"]["k_factor"]
        retriever = HCADRetriever(graph=kg, collection=collection, k=k_factor)

        # apply the bundle's RECORDED anisotropic weights
        # instead of silently falling back to current-code defaults.
        aniso = config.get("innovations", {}).get("anisotropic_transformer", {})
        if aniso:
            retriever._bundle_aniso = {
                "alpha": aniso.get("alpha_degree"),
                "beta": aniso.get("beta_confidence"),
                "gamma": aniso.get("gamma_idf"),
            }

        # Pre-warm pipeline components
        retriever._ensure_components()
        if getattr(retriever, "_bundle_aniso", None) and retriever._transformer is not None:
            retriever._transformer.alpha = retriever._bundle_aniso["alpha"]
            retriever._transformer.beta = retriever._bundle_aniso["beta"]
            retriever._transformer.gamma = retriever._bundle_aniso["gamma"]
        logger.info("HCAD-RAG Pipeline loaded successfully and ready for inference.")

        return retriever

    @staticmethod
    def _compute_md5(file_path: Path) -> str:
        """Compute MD5 checksum of a file."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
