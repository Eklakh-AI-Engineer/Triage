"""Regression tests for adaptive retrieval filtering and evaluation gates."""

import importlib

import numpy as np
import pytest

from src.core.statistical_filter import StatisticalFilter
from src.evaluation import dev_pool_retrieval


def test_filter_keeps_best_candidate_when_adaptive_threshold_exceeds_max(monkeypatch):
    filt = StatisticalFilter(k=2.0)
    contexts = [
        {"document": "weak", "metadata": {}},
        {"document": "strong", "metadata": {}},
    ]
    scores = np.array([0.11, 0.12], dtype=float)

    # For this pool mu + 2*sigma exceeds max(score); the old implementation
    # rejected both candidates despite both being above the absolute floor.
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.delenv("HCAD_TAU_SAT", raising=False)
    threshold, _, _ = filt.compute_threshold(scores, k=2.0, tau_sat=None)
    monkeypatch.undo()
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
    module = importlib.reload(dev_pool_retrieval)
    assert module.GO_DELTA_RECALL == pytest.approx(0.08)
    assert module.GO_DELTA_NDCG == pytest.approx(0.06)
    assert module.meets_go_gate(0.08, 0.06)
    assert not module.meets_go_gate(0.20, 0.059)
    assert not module.meets_go_gate(0.079, 0.20)


def test_retrieval_calibration_refuses_test_pool():
    with pytest.raises(ValueError, match="development-only"):
        dev_pool_retrieval.run_retrieval_eval(pool="test")
