"""Tests for the robustness-metrics wiring in the base-paper benchmark runner.

Covers the retrieval relevance proxy (paper-protocol gold labels), the
checkpoint-based aggregation used for resume-safe results, and the new
per-sample records (retrieval quality, citation precision, McNemar inputs).
"""

import json

from src.evaluation.run_base_paper_benchmark import (
    CORRECTNESS_THRESHOLD,
    RETRIEVAL_K,
    _content_tokens,
    _context_document,
    _ranked_relevance,
    _records_from_checkpoint,
)

REFERENCE = (
    "Metformin is contraindicated in severe renal impairment because reduced "
    "clearance causes drug accumulation and elevates the risk of lactic acidosis."
)


def _record(question, overall_hcad=80.0, overall_vanilla=60.0, source="paper_protocol",
            tier=1, retrieval=None, citations=None):
    return {
        "question": question,
        "source": source,
        "difficulty_tier": tier,
        "scores": {
            "Baseline LLM": {"overall_score": 40.0},
            "Vanilla RAG": {"overall_score": overall_vanilla},
            "HCAD-RAG": {"overall_score": overall_hcad},
        },
        "selection": {
            "HCAD_vs_Vanilla": {"overall_winner": "HCAD-RAG", "criteria_winners": {}},
            "HCAD_vs_Baseline": {"overall_winner": "HCAD-RAG", "criteria_winners": {}},
        },
        "latency": {"Vanilla RAG": 0.5, "HCAD-RAG": 0.7},
        "retrieval": retrieval if retrieval is not None else {},
        "citations": citations if citations is not None else {},
    }


class TestContextDocument:
    def test_prefers_document_field(self):
        assert _context_document({"document": "doc text"}) == "doc text"

    def test_falls_back_to_metadata_triple(self):
        context = {
            "metadata": {
                "head_name": "Metformin",
                "display_relation": "contraindication",
                "tail_name": "CKD",
            }
        }
        assert _context_document(context) == "Metformin contraindication CKD"


class TestRankedRelevance:
    def test_source_chunk_is_relevant(self):
        contexts = [{"document": REFERENCE}]
        assert _ranked_relevance(REFERENCE, contexts) == [1]

    def test_unrelated_context_is_not_relevant(self):
        contexts = [{"document": "Aspirin inhibits platelet aggregation via COX-1."}]
        assert _ranked_relevance(REFERENCE, contexts) == [0]

    def test_partial_overlap_below_threshold_is_zero(self):
        # One shared content token out of a large union: Jaccard ~ 0.06,
        # far below the 0.3 cut.
        contexts = [{"document": "clearance of the airport was delayed."}]
        assert _ranked_relevance(REFERENCE, contexts) == [0]

    def test_empty_reference_and_contexts(self):
        assert _ranked_relevance("", []) == []
        assert _ranked_relevance("", [{"document": "x"}]) == [0]
        assert _ranked_relevance(REFERENCE, []) == []

    def test_content_tokens_drop_stopwords_and_short_tokens(self):
        tokens = _content_tokens("The eGFR is < 30 mL/min for CKD patients")
        assert "egfr" in tokens and "ckd" in tokens
        assert "the" not in tokens and "is" not in tokens and "30" not in tokens


class TestRecordsFromCheckpoint:
    def test_resumed_questions_are_included(self, tmp_path):
        ckpt = tmp_path / "base_paper_checkpoint.jsonl"
        with open(ckpt, "w", encoding="utf-8") as f:
            f.write(json.dumps(_record("q1")) + "\n")   # evaluated before interruption
            f.write(json.dumps(_record("q2")) + "\n")   # evaluated in this session

        records = _records_from_checkpoint(ckpt, {"q1", "q2", "q3"})
        assert len(records["scoring"]["HCAD-RAG"]) == 2
        assert len(records["per_question"]) == 2
        # q3 was never evaluated; it contributes nothing.
        assert all("q3" not in r["question"] for r in records["per_question"])

    def test_newest_record_wins_on_duplicate_lines(self, tmp_path):
        ckpt = tmp_path / "base_paper_checkpoint.jsonl"
        with open(ckpt, "w", encoding="utf-8") as f:
            f.write(json.dumps(_record("q1", overall_hcad=10.0)) + "\n")
            f.write(json.dumps(_record("q1", overall_hcad=90.0)) + "\n")

        records = _records_from_checkpoint(ckpt, {"q1"})
        assert len(records["per_question"]) == 1
        assert records["scoring"]["HCAD-RAG"][0]["overall_score"] == 90.0

    def test_missing_checkpoint_returns_empty_records(self, tmp_path):
        records = _records_from_checkpoint(tmp_path / "absent.jsonl", {"q1"})
        assert records["scoring"]["HCAD-RAG"] == []
        assert records["per_question"] == []

    def test_corrupt_line_is_skipped(self, tmp_path):
        ckpt = tmp_path / "base_paper_checkpoint.jsonl"
        with open(ckpt, "w", encoding="utf-8") as f:
            f.write("{not json\n")
            f.write(json.dumps(_record("q2")) + "\n")
        records = _records_from_checkpoint(ckpt, {"q2"})
        assert len(records["per_question"]) == 1

    def test_retrieval_and_citation_records_collected(self, tmp_path):
        ckpt = tmp_path / "base_paper_checkpoint.jsonl"
        retrieval = {"HCAD-RAG": {"ndcg_at_k": 1.0, "recall_at_k": 0.5},
                      "Vanilla RAG": {"ndcg_at_k": 0.5, "recall_at_k": 0.25}}
        citations = {"HCAD-RAG": {"n_citations": 2, "n_valid": 2, "precision": 1.0},
                     "Vanilla RAG": {"n_citations": 0, "n_valid": 0, "precision": None}}
        with open(ckpt, "w", encoding="utf-8") as f:
            f.write(json.dumps(_record("q1", retrieval=retrieval, citations=citations)) + "\n")

        records = _records_from_checkpoint(ckpt, {"q1"})
        assert records["retrieval"]["HCAD-RAG"][0]["ndcg_at_k"] == 1.0
        assert records["citations"]["HCAD-RAG"][0]["precision"] == 1.0

    def test_by_tier_and_by_source_split(self, tmp_path):
        ckpt = tmp_path / "base_paper_checkpoint.jsonl"
        with open(ckpt, "w", encoding="utf-8") as f:
            f.write(json.dumps(_record("q1", tier=1)) + "\n")
            f.write(json.dumps(_record("q2", tier=2, source="curated")) + "\n")

        records = _records_from_checkpoint(ckpt, {"q1", "q2"})
        assert set(records["scoring_by_tier"]) == {1, 2}
        assert set(records["scoring_by_source"]) == {"paper_protocol", "curated"}
        assert set(records["selection_by_source"]) == {"paper_protocol", "curated"}


class TestWiringConstants:
    def test_constants_match_protocol(self):
        # Both systems retrieve top-5, so the metric cut must match.
        assert RETRIEVAL_K == 5
        # McNemar correctness is thresholded on the 0-100 overall score.
        assert CORRECTNESS_THRESHOLD == 70.0
