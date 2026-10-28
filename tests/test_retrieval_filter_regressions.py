"""Regression tests for adaptive retrieval filtering and evaluation gates."""

import importlib
from typing import get_type_hints

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
    monkeypatch.delenv("HCAD_TAU_SAT", raising=False)
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
    module = importlib.reload(dev_pool_retrieval)
    assert module.GO_DELTA_RECALL == pytest.approx(0.08)
    assert module.GO_DELTA_NDCG == pytest.approx(0.06)
    assert module.meets_go_gate(0.08, 0.06)
    assert not module.meets_go_gate(0.20, 0.059)
    assert not module.meets_go_gate(0.079, 0.20)


def test_retrieval_calibration_refuses_test_pool():
    with pytest.raises(ValueError, match="development-only"):
        dev_pool_retrieval.run_retrieval_eval(pool="test")



@pytest.mark.parametrize("bad_score", [np.nan, np.inf, -np.inf])
def test_filter_rejects_non_finite_scores(bad_score):
    filt = StatisticalFilter()
    with pytest.raises(ValueError, match="Non-finite scores"):
        filt.filter_contexts(
            [{"document": "invalid", "metadata": {}}],
            np.array([bad_score]),
        )


def test_filter_rejects_mismatched_context_and_score_lengths():
    with pytest.raises(ValueError, match="Mismatch"):
        StatisticalFilter().filter_contexts(
            [{"document": "one", "metadata": {}}],
            np.array([0.2, 0.3]),
        )


def test_environment_cannot_weaken_preregistered_go_gate(monkeypatch):
    monkeypatch.setenv("HCAD_GO_DELTA_RECALL", "-0.10")
    monkeypatch.setenv("HCAD_GO_DELTA_NDCG", "0.00")
    with pytest.raises(ValueError, match="weakens the pre-registered minimum"):
        importlib.reload(dev_pool_retrieval)
    monkeypatch.delenv("HCAD_GO_DELTA_RECALL", raising=False)
    monkeypatch.delenv("HCAD_GO_DELTA_NDCG", raising=False)
    importlib.reload(dev_pool_retrieval)


def test_environment_can_only_tighten_preregistered_go_gate(monkeypatch):
    monkeypatch.setenv("HCAD_GO_DELTA_RECALL", "0.10")
    monkeypatch.setenv("HCAD_GO_DELTA_NDCG", "0.08")
    module = importlib.reload(dev_pool_retrieval)
    assert module.GO_DELTA_RECALL == pytest.approx(0.10)
    assert module.GO_DELTA_NDCG == pytest.approx(0.08)
    monkeypatch.delenv("HCAD_GO_DELTA_RECALL", raising=False)
    monkeypatch.delenv("HCAD_GO_DELTA_NDCG", raising=False)
    importlib.reload(dev_pool_retrieval)



def test_filter_type_annotations_resolve_without_optional_torch_import():
    hints = get_type_hints(StatisticalFilter.compute_threshold)
    assert "scores" in hints
    assert hints["scores"] is np.ndarray



def test_filter_rejects_non_1d_score_arrays():
    with pytest.raises(ValueError, match="1-D score array"):
        StatisticalFilter().filter_contexts(
            [{"document": "one", "metadata": {}}],
            np.array([[0.2]]),
        )


def test_filter_checks_empty_context_score_consistency():
    with pytest.raises(ValueError, match="Mismatch"):
        StatisticalFilter().filter_contexts([], np.array([0.2]))


@pytest.mark.parametrize("top_k", [-1, 1.5, True])
def test_filter_rejects_invalid_top_k(top_k):
    with pytest.raises(ValueError, match="max_top_k"):
        StatisticalFilter().filter_contexts(
            [{"document": "one", "metadata": {}}],
            np.array([0.2]),
            max_top_k=top_k,
        )
