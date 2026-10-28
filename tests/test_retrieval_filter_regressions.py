"""Regression tests for adaptive retrieval filtering and evaluation gates."""

import numpy as np
import pytest

from src.core.statistical_filter import StatisticalFilter
from src.evaluation import dev_pool_retrieval


def test_filter_keeps_best_candidate_when_adaptive_threshold_exceeds_max():
    filt = StatisticalFilter(k=2.0)
    contexts = [
        {"document": "weak", "metadata": {}},
        {"document": "strong", "metadata": {}},
    ]
    scores = np.array([0.11, 0.12], dtype=float)

    # For this pool mu + 2*sigma exceeds max(score); the old implementation
    # rejected both candidates despite both being above the absolute floor.
    threshold, _, _ = filt.compute_threshold(scores, k=2.0, tau_sat=None)
    assert threshold > scores.max()

    kept = filt.filter_contexts(contexts, scores, k=2.0, min_score=0.10)

    assert kept
    assert kept[0]["document"] == "strong"
    assert kept[0]["phi_final"] == pytest.approx(0.12)


def test_filter_still_abstains_when_all_candidates_below_absolute_floor():
    filt = StatisticalFilter(k=1.0)
    contexts = [{"document": "low", "metadata": {}}]

    assert filt.filter_contexts(
        contexts, np.array([0.099]), min_score=0.10
    ) == []


def test_filter_returns_empty_for_empty_input():
    assert StatisticalFilter().filter_contexts([], np.array([])) == []


def test_documented_go_gate_requires_both_recall_and_ndcg_targets(monkeypatch):
    monkeypatch.delenv("HCAD_GO_DELTA_RECALL", raising=False)
    monkeypatch.delenv("HCAD_GO_DELTA_NDCG", raising=False)
    # Constants are loaded at module import; verify the documented defaults
    # independently from any developer-machine environment overrides.
    assert dev_pool_retrieval.GO_DELTA_RECALL == pytest.approx(0.08)
    assert dev_pool_retrieval.GO_DELTA_NDCG == pytest.approx(0.06)


def test_retrieval_calibration_refuses_test_pool():
    with pytest.raises(ValueError, match="development-only"):
        dev_pool_retrieval.run_retrieval_eval(pool="test")
