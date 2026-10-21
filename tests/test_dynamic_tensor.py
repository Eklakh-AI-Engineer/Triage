"""Tests for the learned query-adaptive anisotropic tensor (Blueprint 5.1)."""

import os

import pytest
import torch

import src.core.ner_extractor  # noqa: F401  (singleton reset isolation)
from src.core.dynamic_tensor import (
    QueryAdaptiveAnisotropicTensor,
    load_tensor,
    make_query_features,
    save_tensor,
)


def _model():
    torch.manual_seed(0)
    return QueryAdaptiveAnisotropicTensor(embed_dim=768, feat_dim=25)


def _inputs():
    torch.manual_seed(1)
    Q = torch.randn(3, 768)
    feats = make_query_features(None, 0.5, 0.3, 0.2, 1.0, 0.0,
                                [1.0] * 10, [0.0] * 10)
    assert feats.shape[-1] == 25
    return Q, feats


def test_forward_output_invariants():
    lam, a = _model()(*_inputs())
    assert lam.shape == (768,)
    assert abs(float(lam.sum()) - 1.0) < 1e-4, "lambda must sum to 1"
    assert float(lam.min()) > 0, "lambda must be positive"
    assert a.shape == (3,)
    assert abs(float(a.sum()) - 1.0) < 1e-4, "component weights must be a simplex"


def test_predictions_vary_with_query():
    model = _model()
    Q, feats = _inputs()
    lam1, _ = model(Q, feats)
    lam2, _ = model(torch.randn(3, 768), feats)
    assert not torch.allclose(lam1, lam2, atol=1e-3), \
        "the tensor must be query-conditioned, not a fixed vector"


def test_save_load_roundtrip(tmp_path):
    model = _model()
    path = tmp_path / "tensor.pt"
    save_tensor(model, path)
    loaded = load_tensor(path)
    Q, feats = _inputs()
    lam_a, a_a = model(Q, feats)
    lam_b, a_b = loaded(Q, feats)
    assert torch.allclose(lam_a, lam_b, atol=1e-6)
    assert torch.allclose(a_a, a_b, atol=1e-6)


def test_trainer_refuses_non_dev_pools():
    from src.core.tensor_trainer import train_tensor

    with pytest.raises(ValueError, match="dev"):
        train_tensor(object(), [
            {"question": "q?", "reference": "r", "pool": "test"},
        ])
    with pytest.raises(ValueError, match="dev_questions is empty"):
        train_tensor(object(), [])


def test_retriever_feature_helpers():
    from src.core.hcad_retriever import _entity_type_histogram, _relation_onehot

    hist = _entity_type_histogram({"drug_disease": [{"text": "x"}],
                                    "molecular": [{"text": "y"}, {"text": "z"}]})
    assert len(hist) == 10
    assert all(0.0 <= v <= 1.0 for v in hist)
    assert hist[0] > 0, "drug category should be present"

    onehot = _relation_onehot(["contraindicated"])
    assert len(onehot) == 10
    assert onehot[1] == 1.0
    assert sum(onehot) == 1.0

    fallback = _relation_onehot([])
    assert fallback[-1] == 1.0


def test_get_tensor_falls_back_when_checkpoint_missing(tmp_path, monkeypatch):
    from src.core import hcad_retriever as hr

    retriever = hr.HCADRetriever.__new__(hr.HCADRetriever)
    retriever._tensor = None
    retriever._tensor_override = None
    retriever._last_tensor_inputs = None
    monkeypatch.setenv("HCAD_TENSOR_MODE", "learned")
    monkeypatch.setattr("src.config.TENSOR_MODEL_PATH", tmp_path / "missing.pt")

    # load_tensor reads TENSOR_MODEL_PATH at call time via config import;
    # patch the loader's config reference through the module attribute.
    import src.core.dynamic_tensor as dt
    monkeypatch.setattr(dt, "TENSOR_MODEL_PATH", tmp_path / "missing.pt",
                        raising=False)

    assert retriever._get_tensor() is None, \
        "missing checkpoint must fall back to static weights"


def test_get_tensor_static_mode_never_loads(monkeypatch):
    from src.core import hcad_retriever as hr

    retriever = hr.HCADRetriever.__new__(hr.HCADRetriever)
    retriever._tensor = None
    monkeypatch.delenv("HCAD_TENSOR_MODE", raising=False)

    def _boom():
        raise AssertionError("static mode must not touch the tensor checkpoint")
    import src.core.dynamic_tensor as dt
    monkeypatch.setattr(dt, "load_tensor", _boom)
    assert retriever._get_tensor() is None
