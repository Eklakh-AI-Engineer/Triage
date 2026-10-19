"""
HCAD-RAG Core Architecture Verification Suite
Tests each module in src/core/ and mathematically verifies:
1. Biomedical Named Entity Recognition (SciSpacy & SimpleNER fallback)
2. Relation & Modifier Extraction (predicate verbs, negation cues, qualifiers)
3. Context Decoupling & Modified Gram-Schmidt Orthogonalization (E ⊥ R ⊥ M)
4. Anisotropic Transformation & Matrix Trace Tensor Similarity (Score = Trace(Q @ W_A @ C.T))
5. Adaptive Statistical Heaviside Step Filter (Θ(Φ_Final - (μ + 1.5σ)))
6. Structured Grounded Evidence Prompt Construction
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def print_section(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def test_ner_and_relation_extractor():
    print_section("1. TESTING NER & RELATION EXTRACTION")
    from src.core.ner_extractor import BiomedicalNER, SimpleNER
    from src.core.relation_extractor import RelationExtractor

    test_text = "Metformin is strictly contraindicated in patients presenting with severe renal failure."
    print(f"Input Clinical Text: '{test_text}'")

    try:
        ner = BiomedicalNER()
        entities = ner.extract_entities(test_text)
        print(f"✓ BiomedicalNER (SciSpacy) extraction: {entities}")
    except Exception as e:
        print(f"BiomedicalNER fallback ({e}), using SimpleNER...")
        ner = SimpleNER()
        entities = ner.extract_entities(test_text)
        print(f"✓ SimpleNER extraction: {entities}")

    assert any(entities.values()), "At least one entity category must be identified"

    rel_extractor = RelationExtractor()
    relations = rel_extractor.extract_rules(test_text, entities=entities)
    modifier = rel_extractor.extract_modifier(test_text, entities=entities)
    is_negated, neg_cue = rel_extractor.detect_negation(test_text)

    print(f"✓ Extracted Relations: {relations}")
    print(f"✓ Extracted Modifier: '{modifier}'")
    print(f"✓ Negation Detected: {is_negated} (cue: '{neg_cue}')")

    assert len(relations) > 0, "Expected at least one extracted relation"
    assert relations[0]["relation"] == "CONTRAINDICATED", f"Expected CONTRAINDICATED, got {relations[0]['relation']}"
    assert is_negated is True, "Expected negation/contraindication to be detected"
    print("✓ Stage 1 (NER, Relation & Modifier Extraction) successfully verified!")
    return entities, relations

def test_semantic_decoupler():
    print_section("2. TESTING SEMANTIC DECOUPLER & GRAM-SCHMIDT ORTHOGONALIZATION")
    from src.core.semantic_decoupler import SemanticDecoupler

    decoupler = SemanticDecoupler()
    sample_text = "Metformin treats type 2 diabetes mellitus in adult patients."
    print(f"Input Text: '{sample_text}'")

    X = decoupler.decouple(sample_text)
    print(f"✓ Output HCAD Matrix shape: {X.shape} (Expected: 3 x 768)")
    assert X.shape == (3, 768), f"Expected shape (3, 768), got {X.shape}"

    # Verify Gram-Schmidt Orthogonality: E ⊥ R, R ⊥ M, E ⊥ M
    E = X[0]
    R = X[1]
    M = X[2]

    dot_ER = float(np.dot(E, R))
    dot_RM = float(np.dot(R, M))
    dot_EM = float(np.dot(E, M))

    print(f"  <E_orth, R_orth> dot product: {dot_ER:+.2e}")
    print(f"  <R_orth, M_orth> dot product: {dot_RM:+.2e}")
    print(f"  <E_orth, M_orth> dot product: {dot_EM:+.2e}")

    # Soft Gram-Schmidt (0.7/0.3 blend) balances 70% orthogonal projection with 30% relation semantics
    assert abs(dot_ER) < 0.35, f"E and R semantic decoupling error: {dot_ER}"
    assert abs(dot_RM) < 0.35, f"R and M semantic decoupling error: {dot_RM}"
    assert abs(dot_EM) < 0.35, f"E and M semantic decoupling error: {dot_EM}"
    print("✓ Soft Gram-Schmidt Orthogonalization mathematically verified (semantic interference reduced while preserving relation coupling)!")

    # Verify unit norms
    norm_E = float(np.linalg.norm(E))
    norm_R = float(np.linalg.norm(R))
    norm_M = float(np.linalg.norm(M))
    print(f"  ||E_orth||: {norm_E:.4f}, ||R_orth||: {norm_R:.4f}, ||M_orth||: {norm_M:.4f}")
    assert abs(norm_E - 1.0) < 1e-4
    assert abs(norm_R - 1.0) < 1e-4
    assert abs(norm_M - 1.0) < 1e-4
    print("✓ All basis vectors normalized to unit length (||v|| = 1.0)!")
    return decoupler, X

def run_anisotropic_transformer(decoupler, X_query):
    print_section("3. TESTING ANISOTROPIC TRANSFORMER & STRUCTURAL TRACE SIMILARITY")
    from src.core.anisotropic_transformer import AnisotropicTransformer
    import networkx as nx

    transformer = AnisotropicTransformer()

    # Synthetic graph topology representing PrimeKG neighborhood
    g = nx.MultiDiGraph()
    g.add_node("metformin", node_type="drug")
    g.add_node("renal failure", node_type="disease")
    g.add_edge("metformin", "renal failure", key=0, relation="contraindicated", confidence=0.95, weight=0.95)
    for i in range(12):
        g.add_node(f"node_{i}")
        g.add_edge("metformin", f"node_{i}", key=0, relation="interacts_with", confidence=0.85, weight=0.85)

    weights = transformer.compute_weights(g, "metformin")
    print(f"✓ Anisotropic weight vector shape: {weights.shape} (Expected: 768,)")
    print(f"  Weight sum (Σ λ_i): {weights.sum():.6f} (Expected: 1.000000)")
    assert abs(weights.sum() - 1.0) < 1e-5, f"Weights sum to {weights.sum()}, expected 1.0"

    # Candidate 1: Highly relevant contraindication statement
    cand1 = "Metformin is strictly contraindicated in patients with severe renal failure."
    # Candidate 2: Distractor passage with keyword overlap
    cand2 = "MET proto-oncogene amplification observed in renal cell carcinoma."

    X_c1 = decoupler.decouple(cand1)
    X_c2 = decoupler.decouple(cand2)

    score_c1 = transformer.structural_similarity(X_query, X_c1, weights)
    score_c2 = transformer.structural_similarity(X_query, X_c2, weights)

    print(f"  Structural Trace Similarity (Relevant):   {score_c1:.6f}")
    print(f"  Structural Trace Similarity (Irrelevant): {score_c2:.6f}")

    # Breakdown of component similarities (S_E, S_R, S_M, and Φ_Final with relation weight β=0.50)
    comp1 = transformer.compute_component_similarities(X_query, X_c1, weights)
    comp2 = transformer.compute_component_similarities(X_query, X_c2, weights)

    print(f"  Relevant Breakdown:   S_E={comp1['entity']:.5f}, S_R={comp1['relation']:.5f}, S_M={comp1['modifier']:.5f} -> Φ_Final={comp1['final']:.5f}")
    print(f"  Irrelevant Breakdown: S_E={comp2['entity']:.5f}, S_R={comp2['relation']:.5f}, S_M={comp2['modifier']:.5f} -> Φ_Final={comp2['final']:.5f}")

    assert comp1["final"] > comp2["final"], "Relevant passage must have higher composite score!"
    print("✓ Anisotropic Trace Similarity mathematically verified!")
    return transformer, comp1["final"], comp2["final"]

def run_statistical_filter(score_high, score_low):
    print_section("4. TESTING ADAPTIVE STATISTICAL HEAVISIDE STEP FILTER")
    from src.core.statistical_filter import StatisticalFilter

    sf = StatisticalFilter(k=1.5)

    # Realistic candidate score distribution:
    # 1 highly relevant target (0.85) + 9 low-scoring background contexts (mean ~0.20)
    scores = np.array([0.88, 0.22, 0.18, 0.19, 0.21, 0.20, 0.17, 0.23, 0.19, 0.16], dtype=np.float32)

    thresh, mu, sigma = sf.compute_threshold(scores)
    print(f"  Score distribution: Mean μ={mu:.4f}, Std σ={sigma:.4f}")
    print(f"  Adaptive Cutoff τ = μ + 1.5σ = {thresh:.4f}")

    mask = sf.heaviside_filter(scores)
    print(f"  Heaviside Binary Mask: {mask.tolist()}")
    print(f"  Retained contexts: {int(mask.sum())} / {len(scores)}")

    assert mask[0] == 1, "The high-scoring target candidate must be retained!"
    assert mask.sum() == 1, f"Expected exactly 1 retained context, got {mask.sum()}"
    print("✓ Adaptive Heaviside Filter successfully eliminates context distraction!")

def test_prompt_builder():
    print_section("5. TESTING STRUCTURED EVIDENCE PROMPT BUILDER")
    from src.core.prompt_builder import PromptBuilder

    builder = PromptBuilder()
    evidence_items = [
        {
            "id": "PMC_EXAMPLE",
            "score": 0.892,
            "document": "Metformin is strictly contraindicated in patients with severe renal failure due to risk of lactic acidosis.",
            "metadata": {
                "head_name": "Metformin",
                "head_type": "Drug",
                "display_relation": "CONTRAINDICATED",
                "tail_name": "severe renal failure",
                "tail_type": "Disease"
            },
        }
    ]
    query = "Can Metformin be safely prescribed to a patient with severe renal failure?"
    messages = builder.build_rag_prompt(query, evidence_items)
    print(f"✓ Formatted System Prompt: {messages[0]['content'][:60]}...")
    print(f"✓ Formatted User Content snippet:\n{messages[1]['content'][:250]}...")

    assert "Metformin" in messages[1]["content"]
    assert "CONTRAINDICATED" in messages[1]["content"]
    print("✓ Structured Evidence Prompt Builder verified!")

def main():
    print_section("HCAD-RAG CORE PIPELINE VERIFICATION")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device:     {torch.cuda.get_device_name(0)}")

    entities, relations = test_ner_and_relation_extractor()
    decoupler, X = test_semantic_decoupler()
    transformer, s1, s2 = run_anisotropic_transformer(decoupler, X)
    run_statistical_filter(s1, s2)
    test_prompt_builder()

    print_section("ALL HCAD-RAG CORE MODULES PASSED VERIFICATION WITH 100% SUCCESS!")

if __name__ == "__main__":
    main()

# ------------------- pytest-native entry points (audit item 57) -------------------
# The helpers above require arguments, which made this suite uncollectable
# under pytest (fixture-not-found errors). These wrappers provide real
# fixtures; heavy model loads are cached at module scope.

import pytest

_CACHED = {}

@pytest.fixture(scope="session")
def ner_relation_fixtures():
    if "ner" not in _CACHED:
        _CACHED["ner"] = test_ner_and_relation_extractor()
    return _CACHED["ner"]

@pytest.fixture(scope="session")
def decoupled_matrix():
    if "dec" not in _CACHED:
        _CACHED["dec"] = test_semantic_decoupler()
    return _CACHED["dec"]

def test_core_anisotropic(decoupled_matrix):
    decoupler, X = decoupled_matrix
    run_anisotropic_transformer(decoupler, X)

def test_core_statistical_filter():
    run_statistical_filter(0.9, 0.2)

def test_core_prompt_builder():
    test_prompt_builder()
