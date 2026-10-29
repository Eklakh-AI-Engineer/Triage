import unittest

from src.evaluation.ranking_experiments import reciprocal_rank_fusion, _metrics_for_ranking
from src.evaluation.clinical_answer_quality import evaluate_clinical_answers
from src.evaluation.dev_pool_retrieval import _pool_based_metrics


class TestReciprocalRankFusion(unittest.TestCase):
    def test_fusion_rewards_evidence_ranked_by_both_sources(self):
        rows = [
            {"document": "dense first", "source": "vector", "score": 0.99},
            {"document": "shared candidate", "source": "vector", "score": 0.80},
            {"document": "shared candidate", "source": "graph", "score": 0.95},
            {"document": "graph only", "source": "graph_hyperedge", "score": 0.70},
        ]
        fused = reciprocal_rank_fusion(rows)
        self.assertEqual(fused[0]["document"], "shared candidate")
        self.assertEqual(len([r for r in fused if r["document"] == "shared candidate"]), 1)
        self.assertEqual(set(fused[0]["sources"]), {"dense", "graph"})


    def test_recall_uses_shared_full_candidate_pool_denominator(self):
        pool = [
            {"document": "gold alpha evidence", "source": "vector", "score": 0.9},
            {"document": "gold beta evidence", "source": "graph", "score": 0.8},
            {"document": "irrelevant distractor", "source": "vector", "score": 0.7},
        ]
        ranked = [pool[0], pool[2], pool[1]]
        recall, ndcg, mrr = _metrics_for_ranking(
            "gold alpha evidence gold beta evidence", ranked, [1, 1, 0], k=1
        )
        self.assertAlmostEqual(recall, 0.5)
        self.assertGreater(ndcg, 0.0)
        self.assertEqual(mrr, 1.0)

    def test_fusion_is_deterministic_and_validates_k(self):
        rows = [
            {"document": "A", "source": "vector", "score": 0.8},
            {"document": "B", "source": "graph", "score": 0.8},
        ]
        self.assertEqual(reciprocal_rank_fusion(rows), reciprocal_rank_fusion(rows))
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion(rows, rrf_k=0)


class TestPoolBasedRetrievalMetrics(unittest.TestCase):
    def test_recall_denominator_uses_all_relevant_pool_candidates(self):
        pool = [
            {"document": "gold alpha evidence", "source": "vector"},
            {"document": "gold beta evidence", "source": "graph"},
            {"document": "irrelevant distractor", "source": "vector"},
        ]
        recall, ndcg = _pool_based_metrics(
            "gold alpha evidence gold beta evidence", [pool[0]], pool
        )
        self.assertAlmostEqual(recall, 0.5)
        self.assertGreater(ndcg, 0.0)

class TestClinicalAnswerQuality(unittest.TestCase):
    def test_reports_independent_labels_and_telemetry(self):
        rows = [
            {"question_id": "q1", "system": "hcad", "answer": "Answer with evidence",
             "factuality_label": "correct", "citation_support_label": "supported",
             "negation_label": "correct", "abstention_expected": False, "abstained": False,
             "latency_ms": 125.0, "input_tokens": 100, "output_tokens": 50, "cost_usd": 0.002},
            {"question_id": "q2", "system": "hcad", "answer": "I cannot answer from evidence",
             "factuality_label": "incorrect", "citation_support_label": "not_applicable",
             "negation_label": "not_applicable", "abstention_expected": True, "abstained": True,
             "latency_ms": 200.0, "input_tokens": 80, "output_tokens": 20, "cost_usd": 0.001},
        ]
        result = evaluate_clinical_answers(rows)["systems"]["hcad"]
        self.assertEqual(result["n_questions"], 2)
        self.assertEqual(result["citation_support_n"], 1)
        self.assertEqual(result["negation_cases_n"], 1)
        self.assertEqual(result["abstention_decision_accuracy"]["mean"], 1.0)
        self.assertAlmostEqual(result["cost_usd"]["total"], 0.003)

    def test_missing_independent_labels_fail_closed(self):
        rows = [{"question_id": "q1", "system": "hcad", "answer": "x",
                 "abstention_expected": False, "abstained": False}]
        with self.assertRaises(ValueError):
            evaluate_clinical_answers(rows)

    def test_duplicate_question_system_pair_is_rejected(self):
        row = {"question_id": "q1", "system": "hcad", "answer": "x",
               "factuality_label": "correct", "citation_support_label": "supported",
               "negation_label": "not_applicable", "abstention_expected": False, "abstained": False}
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            evaluate_clinical_answers([row, row])


if __name__ == "__main__":
    unittest.main()
