"""Fairness tests: the vanilla RAG baseline must operate under the same
evidence word budget as HCAD-RAG (EVIDENCE_WORD_BUDGET)."""

import pytest

from src.baselines.vanilla_vector_rag import VanillaVectorRAG
from src.config import EVIDENCE_WORD_BUDGET


def _make_rag():
    """Build the baseline without touching ChromaDB / the embedding model."""
    rag = VanillaVectorRAG.__new__(VanillaVectorRAG)
    return rag


def _long_context(n_words=400, seed=0):
    """A synthetic passage of distinct sentences, n_words words long."""
    sentences = [
        f"Sentence {seed}-{i} describes a clinical finding about drug X."
        for i in range(200)
    ]
    text = " ".join(sentences)
    return " ".join(text.split()[:n_words])


def test_context_block_respects_word_budget():
    rag = _make_rag()
    contexts = [
        {"text": _long_context(400, seed=i), "score": 0.9 - 0.1 * i}
        for i in range(5)
    ]
    prompt = rag._build_rag_prompt("What treats condition X?", contexts)

    # extract the evidence region between 'Retrieved Contexts:' and the
    # instructions block
    start = prompt.index("Retrieved Contexts:") + len("Retrieved Contexts:")
    end = prompt.index("Instructions for Clinical Synthesis:")
    evidence = prompt[start:end]

    # allowed slack: one sentence can overrun the exact boundary before the
    # sentence-boundary clip, but the block must stay near the budget.
    assert len(evidence.split()) <= EVIDENCE_WORD_BUDGET + 60, (
        f"vanilla evidence block is {len(evidence.split())} words; "
        f"budget is {EVIDENCE_WORD_BUDGET}"
    )


def test_contexts_clipped_at_sentence_boundary():
    rag = _make_rag()
    contexts = [{"text": _long_context(400), "score": 0.85}]
    prompt = rag._build_rag_prompt("Query?", contexts)
    assert prompt.count("[Context 1]") == 1
    # every evidence line ends with a sentence terminator, not a mid-sentence cut
    line = [l for l in prompt.splitlines() if l.startswith("[Context 1]") is False
            and l and not l.startswith(("Clinical", "Retrieved", "Use ", "IMPORTANT",
                                        "Structure", "1.", "2.", "3.", "4.", "---"))]
    first_passage = next(l for l in prompt.splitlines()
                         if l and not l.startswith(("[Context", "Clinical", "Retrieved",
                                                    "Use ", "IMPORTANT", "Structure",
                                                    "1.", "2.", "3.", "4.", "---")))
    assert first_passage.rstrip().endswith((".", "!", "?"))


def test_empty_contexts_return_bare_query():
    rag = _make_rag()
    assert rag._build_rag_prompt("What is X?", []) == "What is X?"


def test_multiple_contexts_share_the_single_budget():
    rag = _make_rag()
    contexts = [
        {"text": _long_context(300, seed=i), "score": 0.9}
        for i in range(5)
    ]
    prompt = rag._build_rag_prompt("Query?", contexts)
    # later contexts must be truncated or dropped once the budget is spent
    n_contexts = sum(prompt.count(f"[Context {i}]") for i in range(1, 6))
    assert n_contexts <= 5
    start = prompt.index("Retrieved Contexts:")
    end = prompt.index("Instructions for Clinical Synthesis:")
    assert len(prompt[start:end].split()) <= EVIDENCE_WORD_BUDGET + 60
