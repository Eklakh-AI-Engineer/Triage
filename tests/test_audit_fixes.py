#!/usr/bin/env python3
"""Regression tests for the audit remediation - real assertions only.

Previously this file printed "PASSED" without asserting behavior (audit
item 58). Every test now asserts concrete properties of the fixed code.
Run:  python -m pytest tests/test_audit_fixes.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

def test_negation_uses_word_boundaries():
    """Item 21: 'not' must NOT match 'noted'/'normal'; real cues must match."""
    from src.core.relation_extractor import RelationExtractor
    benign = "The patient was noted to have normal node function; the drug treats migraine."
    is_neg, cue = RelationExtractor.detect_negation(benign)
    assert not is_neg, f"spurious negation on benign text (cue={cue!r})"

    real = "Metformin is not recommended in severe renal failure."
    is_neg, cue = RelationExtractor.detect_negation(real)
    assert is_neg and cue == "not recommended", (is_neg, cue)

    ruled_out = "The ECG rules out STEMI."
    is_neg, cue = RelationExtractor.detect_negation(ruled_out)
    assert is_neg and cue == "rules out", (is_neg, cue)

def test_no_fabricated_relations():
    """Item 22: extract() must NOT invent ASSOCIATED_WITH."""
    from src.core.relation_extractor import RelationExtractor
    ents = {"Drug": [{"text": "metformin", "start": 0, "end": 9}],
            "Disease": [{"text": "diabetes", "start": 20, "end": 28}]}
    rels = RelationExtractor().extract("metformin something something diabetes", ents)
    assert all(r["relation"] != "ASSOCIATED_WITH" or r.get("_grounded") for r in rels) or rels == []

def test_protein_fallback_regex_is_not_word_garbage():
    """Item 20: SimpleNER must not tag 'the'/'is' as proteins."""
    from src.core.ner_extractor import SimpleNER
    ents = SimpleNER().extract_entities("the patient is not improving today")
    assert ents["Protein"] == [], f"false protein spans: {ents['Protein']}"

def test_chunking_never_splits_sentences_and_validates_params():
    """Items 37/39: sentence-safe chunks; overlap guard."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
    from prepare_chunks import chunk_text
    text = ("Metformin is contraindicated in severe renal failure. "
            "It causes lactic acidosis. The dose must be reduced.")
    chunks = chunk_text(text, chunk_size=10, overlap=2)
    assert chunks, "no chunks produced"
    for c in chunks:
        assert c.endswith((".", "!", "?")), f"sentence severed inside chunk: {c!r}"
    try:
        chunk_text(text, chunk_size=5, overlap=5)
        raise AssertionError("overlap >= chunk_size must raise")
    except ValueError:
        pass

def test_anisotropic_block_partition_is_respected():
    """Item 3: post-normalization block mass must follow alpha/beta/gamma."""
    import networkx as nx
    from src.core.anisotropic_transformer import AnisotropicTransformer
    g = nx.MultiDiGraph()
    g.add_node("m", node_type="drug")
    g.add_node("d", node_type="disease")
    g.add_edge("m", "d", relation="treats", weight=0.9)
    for i in range(20):
        g.add_node(f"n{i}")
        g.add_edge("m", f"n{i}", relation="x", weight=0.5)
    w = AnisotropicTransformer(alpha=0.4, beta=0.4, gamma=0.2).compute_weights(g, "m")
    s = 768 // 3
    masses = (w[:s].sum(), w[s:2 * s].sum(), w[2 * s:].sum())
    assert abs(sum(masses) - 1.0) < 1e-5
    # blocks may deviate from 0.4/0.4/0.2 with the node's stats, but the IDF
    # block may no longer monopolize 75-86% of the mass
    assert masses[2] < 0.5, f"IDF block still dominates: {masses}"
    assert masses[0] > 0.05 and masses[1] > 0.05, f"degree/conf crushed: {masses}"

def test_statistical_filter_numpy_and_floor():
    """Items 7/9: pure-NumPy filter; dual-gating abstains below the floor."""
    from src.core.statistical_filter import StatisticalFilter
    sf = StatisticalFilter(k=1.0)
    scores = np.array([0.05, 0.02, 0.07])
    out = sf.filter_contexts([{"document": f"d{i}"} for i in range(3)], scores,
                              min_score=0.10)
    assert out == [], "sub-floor candidates must be rejected (Gate 1)"

    tau, mu, sigma = sf.compute_threshold(np.array([0.5, 0.5, 0.5]))
    assert abs(tau - 0.5) < 1e-9 and sigma < 1e-9
    mask = sf.heaviside_filter(np.array([0.5, 0.5]))
    assert mask.tolist() == [1, 1]  # sigma->0 retains all (documented behavior)

def test_decision_gate_budget_and_floor_invariants():
    """Evidence budgets must never truncate a sentence or a clinical qualifier."""
    from src.core.decision_gate import DecisionGate
    gate = DecisionGate(use_jev=False, min_relevance_score=0.25,
                        max_evidence_items=8, token_budget_words=12)
    long_doc = "word " * 60  # no complete sentence boundary within budget
    kept = gate.verify_and_prune_candidates(
        [{"document": long_doc, "metadata": {"head_name": "A", "relation": "TREATS",
                                              "tail_name": "B"}, "phi_final": 0.9}],
        "q",
    )
    assert kept == [], "unbounded sentence must be omitted, not fragment-truncated"

    qualifier_doc = (
        "Metformin is contraindicated in severe renal failure. "
        "It must not be used during this condition."
    )
    kept = gate.verify_and_prune_candidates(
        [{"document": qualifier_doc, "metadata": {}, "phi_final": 0.9}], "q"
    )
    assert len(kept) == 1
    assert kept[0]["document"] == "Metformin is contraindicated in severe renal failure."
    assert kept[0]["document"].endswith(".")
    assert "contraindicated" in kept[0]["document"]

    below = gate.verify_and_prune_candidates(
        [{"document": "tiny doc", "metadata": {}, "phi_final": 0.2}], "q")
    assert below == [], "sub-floor single candidate must not be re-admitted"

def test_llm_client_has_no_stateful_mutation(monkeypatch=None):
    """Item 24: fallback must not permanently flip the instance provider."""
    from src.config import JUDGE_HF_MODEL
    from src.core.llm_client import UnifiedLLMClient
    client = UnifiedLLMClient(model=JUDGE_HF_MODEL, provider="ollama")
    assert client.provider == "ollama"
    before = (client.provider, client.model)
    try:
        client.chat([{"role": "user", "content": "hi"}])
    except Exception:
        pass  # ollama not running locally - state check below still applies
    assert (client.provider, client.model) == before

def test_hypergraph_includes_in_edges():
    """Item 15: triples must include facts where the entity is the tail."""
    import networkx as nx
    from src.core.hypergraph_builder import HypergraphBuilder
    g = nx.MultiDiGraph()
    g.add_node(1, name="DrugX", node_type="Drug")
    g.add_node(2, name="DiseaseY", node_type="Disease")
    g.add_edge(1, 2, relation="treats")  # entity 2 is only ever a TAIL
    hg = HypergraphBuilder(g)
    triples = hg.get_triples_for_entities([2])
    assert triples and triples[0][2] == 2, f"in-edge triple lost: {triples}"

def test_fact_index_and_entropy_gate_blueprints():
    """P3 modules: fact traversal preserves provenance; gate abstains."""
    from src.core.fact_hypergraph import ClinicalFact, FactIndex
    idx = FactIndex()
    idx.add(ClinicalFact(predicate="contraindicated",
                         roles={"drug": "metformin", "disease": "renal failure",
                                "phenotype": "lactic acidosis"},
                         polarity="asserts", confidence=0.95, source_id="primekg"))
    idx.add(ClinicalFact(predicate="treats",
                         roles={"drug": "metformin", "disease": "T2DM"},
                         confidence=0.9, source_id="primekg"))
    got = idx.retrieve({"metformin"})
    assert len(got) == 2
    got = idx.retrieve({"metformin", "renal failure"})
    assert got[0].predicate == "contraindicated", f"ranking wrong: {[f.predicate for f in got]}"

    from src.core.entropy_gate import EntropyRegulatedDualGate
    gate = EntropyRegulatedDualGate()
    assert gate.gate(np.array([0.05, 0.06]), rho=0.5).sum() == 0  # abstain
    keep = gate.gate(np.array([0.85, 0.3, 0.2, 0.11]), rho=0.5)
    assert keep[0] and not keep[2]

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))

def test_paper_parity_metrics():
    """New robustness metrics: NDCG@K, McNemar, citation precision."""
    from src.evaluation.metrics import ndcg_at_k, recall_at_k, mcnemar_test, citation_precision
    # NDCG: perfect ranking = 1.0; one relevant item at rank 3
    assert ndcg_at_k([1, 0, 1, 0], k=8) < ndcg_at_k([1, 1, 0, 0], k=8)
    assert abs(ndcg_at_k([1, 1, 0, 0], k=8) - 1.0) < 1e-9
    assert recall_at_k([1, 0, 1], k=1) == 0.5
    # McNemar: 8 discordant pairs split 7/1 => significant
    b = [True]*8 + [False]*2 + [True]*5
    a = [False]*8 + [False]*2 + [True]*5
    res = mcnemar_test(b, a)
    assert res["n01"] == 8 and res["n10"] == 0 and res["significant_at_05"]
    res2 = mcnemar_test([True, False], [False, True])
    assert not res2["significant_at_05"]  # 1/1 split is not significant
    # Citation precision
    out = citation_precision("Claim A [Doc 1]. Claim B [Doc 9].", ["e1", "e2"])
    assert out == {"n_citations": 2, "n_valid": 1, "precision": 0.5}

def test_fact_index_built_from_kg():
    """Paper-parity: n-ary facts mined from PrimeKG shared-tail contexts."""
    import networkx as nx
    from types import SimpleNamespace
    from src.core.fact_hypergraph import build_fact_index_from_kg
    g = nx.MultiDiGraph()
    g.add_node(1, name="DrugA"); g.add_node(2, name="DrugB"); g.add_node(3, name="DiseaseC")
    g.add_edge(1, 3, relation="contraindication")
    g.add_edge(2, 3, relation="contraindication")
    kg = SimpleNamespace(graph=g)
    idx = build_fact_index_from_kg(kg)
    assert len(idx.facts) == 1, "both drugs + disease = ONE n-ary fact"
    fact = idx.facts[0]
    assert fact.predicate == "contraindication"
    assert set(fact.roles.values()) == {"DrugA", "DrugB", "DiseaseC"}
    got = idx.retrieve({"diseasec", "druga"})
    assert got and got[0].predicate == "contraindication"

def test_relation_keyword_extraction():
    """X_cor correlation keywords surface for fact retrieval."""
    from src.core.relation_extractor import RelationExtractor
    kws = RelationExtractor().extract_keywords("Is metformin contraindicated in renal failure?")
    assert "contraindicated" in kws, kws

def test_dead_math_removed_from_decoupler():
    """Section A: Orphaned _orthogonalize and _soft_component removed."""
    from src.core.semantic_decoupler import SemanticDecoupler
    assert not hasattr(SemanticDecoupler, "_orthogonalize")
    assert not hasattr(SemanticDecoupler, "_soft_component")
    assert hasattr(SemanticDecoupler, "_absent_anchor")

def test_context_document_prompt_builder():
    """Section A: Unified context_document in prompt_builder."""
    from src.core.prompt_builder import context_document
    ctx1 = {"document": "Patient presents with chest pain."}
    assert context_document(ctx1) == "Patient presents with chest pain."
    ctx2 = {"metadata": {"head_name": "Aspirin", "display_relation": "treats", "tail_name": "Headache"}}
    assert context_document(ctx2) == "Aspirin treats Headache"

def test_relation_extractor_clause_boundary_and_mock_entities():
    """Section D: Clause boundary regex has [?!\n] and handles mock entities without offsets."""
    from src.core.relation_extractor import RelationExtractor
    # Verify punctuation boundaries
    assert RelationExtractor._CLAUSE_BOUNDARY_RE.search("Clause one? Clause two") is not None
    assert RelationExtractor._CLAUSE_BOUNDARY_RE.search("Clause one! Clause two") is not None
    assert RelationExtractor._CLAUSE_BOUNDARY_RE.search("Clause one\nClause two") is not None

    # Verify mock entities without offsets
    extractor = RelationExtractor()
    text = "Metformin is indicated for diabetes mellitus."
    mock_entities = {
        "Drug": [{"text": "Metformin"}],
        "Disease": [{"text": "diabetes mellitus"}],
    }
    rels = extractor.extract_rules(text, mock_entities)
    assert len(rels) > 0
    assert rels[0]["relation"] in ("TREATS", "INDICATED_FOR")

def test_llm_client_ollama_timeout_from_config():
    """Section A: UnifiedLLMClient references config timeout."""
    import src.core.llm_client as client_mod
    from src.config import LLM_OLLAMA_TIMEOUT_S
    assert client_mod.OLLAMA_TIMEOUT_S == LLM_OLLAMA_TIMEOUT_S

def test_knowledge_graph_min_rows_guard(tmp_path):
    """Section B: PrimeKG graph build raises ValueError on truncated CSV."""
    import pandas as pd
    import pytest
    from src.data.build_knowledge_graph import _build_graph_from_csv

    truncated_csv = tmp_path / "truncated_primekg.csv"
    df = pd.DataFrame({
        "relation": ["treats"],
        "display_relation": ["treats"],
        "x_index": [1],
        "y_index": [2],
        "x_name": ["DrugA"],
        "y_name": ["DiseaseB"],
        "x_type": ["drug"],
        "y_type": ["disease"],
    })
    df.to_csv(truncated_csv, index=False)
    with pytest.raises(ValueError, match="truncated"):
        _build_graph_from_csv(truncated_csv, min_rows=10)

def test_decoupler_cache_flush_and_dirty_handling(tmp_path):
    """Section E: DecouplerCache dirty tracking and atomic flush."""
    from src.core.decoupler_cache import DecouplerCache
    cache = DecouplerCache("test_v1", disk_dir=tmp_path)
    assert cache._disk_dirty == 0
    cache.flush()  # clean flush does nothing
    assert not (tmp_path / "decoupler_components_test_v1.jsonl").exists()

    cache.put_components("text A", "ent", "rel", "mod")
    assert cache._disk_dirty == 1
    cache.flush()
    assert cache._disk_dirty == 0
    disk_file = tmp_path / "decoupler_components_test_v1.jsonl"
    assert disk_file.exists()

def test_semantic_decoupler_dynamic_cache_resolution(monkeypatch):
    """Section E: SemanticDecoupler._get_cache dynamically respects HCAD_DECOUPLER_CACHE."""
    from src.core.semantic_decoupler import SemanticDecoupler
    decoupler = SemanticDecoupler(lazy=True)
    monkeypatch.setenv("HCAD_DECOUPLER_CACHE", "0")
    assert decoupler._get_cache() is None

    monkeypatch.setenv("HCAD_DECOUPLER_CACHE", "1")
    cache = decoupler._get_cache()
    assert cache is not None

def test_eval_dev_pool_baseline_pins_desc_expansion():
    """Section F: Dev pool baseline arm pins desc expansion rule."""
    from src.evaluation.dev_pool_retrieval import CONFIGS
    baseline_env = dict(CONFIGS)[ "baseline" ]
    assert baseline_env.get("HCAD_EXPANSION_RULE") == "desc"
    assert baseline_env.get("HCAD_PRESENCE_MASKING") == "0"
    assert baseline_env.get("HCAD_RELATION_VOCAB") == "0"


def test_cli_pool_and_question_pool_arguments():
    """Verify CLI accepts both --pool and --question-pool for retrieval-eval, hub-eval, base-paper."""
    import argparse
    from src.main import main

    # Inspect parser configuration via a non-executing check
    import sys
    from unittest.mock import patch

    # retrieval-eval
    with patch.object(sys, "argv", ["triage", "retrieval-eval", "--pool", "dev", "--samples", "10"]):
        with patch("src.main.cmd_retrieval_eval") as mock_cmd:
            main()
            args = mock_cmd.call_args[0][0]
            assert args.question_pool == "dev"
            assert args.samples == 10

    # hub-eval
    with patch.object(sys, "argv", ["triage", "hub-eval", "--pool", "dev", "--samples", "10"]):
        with patch("src.main.cmd_hub_eval") as mock_cmd:
            main()
            args = mock_cmd.call_args[0][0]
            assert args.question_pool == "dev"
            assert args.samples == 10

    # base-paper
    with patch.object(sys, "argv", ["triage", "base-paper", "--pool", "test", "--samples", "5"]):
        with patch("src.main.cmd_base_paper") as mock_cmd:
            main()
            args = mock_cmd.call_args[0][0]
            assert args.question_pool == "test"
            assert args.samples == 5




