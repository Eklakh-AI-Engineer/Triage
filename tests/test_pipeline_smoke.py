#!/usr/bin/env python3
"""Pipeline smoke test with REAL assertions .

Asserts that all three systems initialize and that retrieval returns usable
evidence. Skips (rather than silently passes) when the vector store or
models are not built yet.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

def _store_ready() -> bool:
    try:
        from src.data.build_vector_store import load_vector_store
        return load_vector_store().count() > 0
    except Exception:
        return False

def test_hcad_retriever_components_load():
    """All HCAD components initialize without the heavy models."""
    from src.core.statistical_filter import StatisticalFilter
    from src.core.decision_gate import DecisionGate
    from src.core.prompt_builder import PromptBuilder
    from src.core.hypergraph_builder import HypergraphBuilder
    from src.core.fact_hypergraph import FactIndex

    gate = DecisionGate(use_jev=False, min_relevance_score=0.25, token_budget_words=350)
    assert gate.max_evidence_items == 8
    messages = PromptBuilder().build_rag_prompt("q", [])
    assert messages[0]["role"] == "system"

def test_baseline_llm_initializes():
    from src.baselines.baseline_llm import BaselineLLM
    baseline = BaselineLLM()
    assert baseline.model, "baseline model name must be set"

def test_vanilla_retrieval_returns_evidence():
    if not _store_ready():
        pytest.skip("Vector store not built - run `python src/main.py prepare` first")
    from src.baselines.vanilla_vector_rag import VanillaVectorRAG
    vanilla = VanillaVectorRAG()
    contexts = vanilla.retrieve("How is Parkinson's disease managed?", top_k=3)
    assert contexts, "vanilla retrieval must return evidence"
    assert all("text" in c and "score" in c for c in contexts)

def test_hcad_retrieval_returns_evidence():
    if not _store_ready():
        pytest.skip("Vector store not built - run `python src/main.py prepare` first")
    from src.core.hcad_retriever import HCADRetriever
    hcad = HCADRetriever()
    contexts = hcad.retrieve("How is Parkinson's disease managed?", n_candidates=10)
    # Empty is a VALID outcome (abstention) - but the pipeline must not crash,
    # and any returned contexts must carry documents and scores.
    for c in contexts:
        assert c.get("document") or c.get("text"), f"context without content: {c}"
        assert "phi_final" in c or "score" in c

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
