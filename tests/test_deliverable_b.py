"""Deliverable-B (Phase 3) unit tests.

Covers the four coupled retrieval changes:
  1. query-batch MGS symmetry (decouple == decouple_batch, exact),
  2. channel-presence flags from the decoupler,
  3. query-conditioned channel-presence masking math,
  4. Gate-2 saturation cap (tau_eff = min(tau_sat, max(Phi_min, mu + k*sigma))),
  5. PrimeKG relation-vocabulary mapping.
"""

import numpy as np
import pytest

import src.config as cfg
from src.core.anisotropic_transformer import AnisotropicTransformer
from src.core.statistical_filter import StatisticalFilter


# ---------------------------------------------------------------------------
# Deterministic fake encoder for decoupler tests (no model download)
# ---------------------------------------------------------------------------

class _FakeModel:
    """Deterministic normalized embeddings keyed by string content."""

    @staticmethod
    def encode(texts, batch_size=32, show_progress_bar=False,
               convert_to_numpy=True, normalize_embeddings=True, **_):
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            vec = rng.standard_normal(64).astype(np.float32)
            vec /= np.linalg.norm(vec) + 1e-8
            out.append(vec)
        return np.stack(out) if len(out) > 1 else out[0]


@pytest.fixture()
def decoupler():
    from src.core.semantic_decoupler import SemanticDecoupler

    d = SemanticDecoupler(lazy=True)
    d._model = _FakeModel()
    d._dim = 64  # fake encoder dimension
    return d


def _fake_entity_decoupler(decoupler, monkeypatch, entity="Metformin",
                           relation="is contraindicated in",
                           modifier="severe"):
    """Stub the NER/relation extractors with fixed channel texts."""
    from src.core.semantic_decoupler import SemanticDecoupler

    monkeypatch.setattr(
        SemanticDecoupler, "_get_ner",
        lambda self: type("N", (), {"extract_entities": staticmethod(
            lambda text: {"Drug": [{"text": entity, "start": 0, "end": len(entity)}]})}
    ))
    monkeypatch.setattr(
        SemanticDecoupler, "_get_rex",
        lambda self: type("R", (), {"extract": staticmethod(
            lambda text, ents: [{"relation": relation, "modifier": modifier}])}
    ))


# --- 1. Query-batch MGS symmetry -------------------------------------------

def test_decouple_equals_decouple_batch(decoupler):
    text = "Metformin is strictly contraindicated in patients with severe renal failure."
    single = decoupler.decouple(text, entities={}, relations=None)
    batch = decoupler.decouple_batch([text], entities_list=[{}], relations_list=[None])
    np.testing.assert_allclose(single, batch[0], atol=1e-6)


def test_decouple_equals_decouple_batch_with_real_model():
    """The review-mandated integration check with the actual encoder."""
    pytest.importorskip("sentence_transformers")
    from src.core.semantic_decoupler import SemanticDecoupler

    d = SemanticDecoupler()
    text = "Metformin is strictly contraindicated in patients with severe renal failure."
    single = d.decouple(text, entities=None, relations=None)
    batch = d.decouple_batch([text], entities_list=[None], relations_list=[None])
    np.testing.assert_allclose(single, batch[0], atol=1e-6)


def test_batch_modifier_runs_two_full_mgs_passes(decoupler):
    """The batch path must subtract R_orth in its second pass (E,R,E,R),
    matching the scalar path - previously it stopped at E,R,E."""
    n = 4
    texts = [f"Drug {i} is contraindicated in disease {i} with severe onset." for i in range(n)]
    batch = decoupler.decouple_batch(texts)
    for i in range(n):
        single = decoupler.decouple(texts[i])
        np.testing.assert_allclose(single, batch[i], atol=1e-6)


# --- 2. Channel presence flags ----------------------------------------------

def test_presence_flags_real_channels(decoupler, monkeypatch):
    _fake_entity_decoupler(decoupler, monkeypatch)
    X, presence = decoupler.decouple(
        "Metformin is contraindicated in severe renal failure.",
        return_presence=True,
    )
    assert X.shape == (3, decoupler._dim)
    assert presence.shape == (3,)
    assert bool(presence[0]), "entity channel must be present"
    assert bool(presence[1]), "explicit relation must be present"


def test_presence_flags_neutral_fallbacks_are_absent(decoupler, monkeypatch):
    # entity-only text: relation/modifier channels fall back to their
    # neutral anchors (empty relation; residual strips to nothing)
    _fake_entity_decoupler(decoupler, monkeypatch, relation="", modifier="")
    X, presence = decoupler.decouple(
        "Metformin.",
        return_presence=True,
    )
    assert not bool(presence[1]), "neutral relation anchor must read absent"
    assert not bool(presence[2]), "neutral modifier anchor must read absent"
    assert bool(presence[0])


def test_presence_flags_batch_parallel(decoupler, monkeypatch):
    _fake_entity_decoupler(decoupler, monkeypatch)
    texts = [
        "Metformin is contraindicated in severe renal failure.",
        "Aspirin treats migraine.",
    ]
    X, presence = decoupler.decouple_batch(texts, return_presence=True)
    assert presence.shape == (2, 3)
    assert presence[:, 0].all()


# --- 3. Query-conditioned masking ------------------------------------------

def _random_hcad(n, d=32, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 3, d)).astype(np.float32)


def test_masked_similarities_equal_unmasked_when_all_present():
    Q = _random_hcad(1)[0]
    C = _random_hcad(4, seed=1)
    w = np.ones(Q.shape[1], dtype=np.float32) / Q.shape[1]
    p_q = np.array([True, True, True])
    p_c = np.ones((4, 3), dtype=bool)
    phi_masked, _ = AnisotropicTransformer.compute_masked_similarities(
        Q, C, w, p_q, p_c, alpha=0.3, beta=0.5, gamma=0.2)
    phi_plain = AnisotropicTransformer.compute_batch_component_similarities(
        Q, C, w, alpha=0.3, beta=0.5, gamma=0.2)
    np.testing.assert_allclose(phi_masked, phi_plain, atol=1e-5)


def test_query_missing_channel_is_excluded():
    Q = _random_hcad(1)[0]
    C = _random_hcad(3, seed=2)
    w = np.ones(Q.shape[1], dtype=np.float32) / Q.shape[1]
    p_q = np.array([True, False, True])  # query has no relation channel
    p_c = np.ones((3, 3), dtype=bool)
    phi, S = AnisotropicTransformer.compute_masked_similarities(
        Q, C, w, p_q, p_c, alpha=0.3, beta=0.5, gamma=0.2)
    # renormalized weights: [0.3, 0, 0.2] / 0.5
    S_plain = AnisotropicTransformer.compute_channel_similarities(Q, C, w)
    expected = 0.6 * S_plain[:, 0] + 0.4 * S_plain[:, 2]
    np.testing.assert_allclose(phi, expected, atol=1e-5)
    assert np.allclose(S[:, 1], S_plain[:, 1]), "inactive channel must not be prior-replaced"


def test_candidate_missing_active_channel_gets_prior():
    Q = _random_hcad(1)[0]
    C = _random_hcad(2, seed=3)
    w = np.ones(Q.shape[1], dtype=np.float32) / Q.shape[1]
    p_q = np.array([True, True, True])
    p_c = np.array([[True, True, True], [True, False, True]])  # 2nd candidate missing relation
    phi, S = AnisotropicTransformer.compute_masked_similarities(
        Q, C, w, p_q, p_c, alpha=0.3, beta=0.5, gamma=0.2, missing_prior=0.20)
    assert S[1, 1] == pytest.approx(0.20)
    assert S[0, 1] != pytest.approx(0.20)


def test_all_absent_query_falls_back_to_entity_channel():
    Q = _random_hcad(1)[0]
    C = _random_hcad(2, seed=4)
    w = np.ones(Q.shape[1], dtype=np.float32) / Q.shape[1]
    p_q = np.array([False, False, False])
    p_c = np.ones((2, 3), dtype=bool)
    phi, _ = AnisotropicTransformer.compute_masked_similarities(
        Q, C, w, p_q, p_c, alpha=0.3, beta=0.5, gamma=0.2)
    S = AnisotropicTransformer.compute_channel_similarities(Q, C, w)
    np.testing.assert_allclose(phi, S[:, 0], atol=1e-5)
    assert np.isfinite(phi).all()


def test_empty_batch_masked():
    phi, S = AnisotropicTransformer.compute_masked_similarities(
        _random_hcad(1)[0], np.empty((0, 3, 32)), np.ones(32) / 32,
        np.ones(3, dtype=bool), np.ones((0, 3), dtype=bool),
    )
    assert phi.shape == (0,) and S.shape == (0, 3)


# --- 4. Gate-2 saturation cap ----------------------------------------------

def test_threshold_without_cap_is_mu_plus_k_sigma():
    f = StatisticalFilter(k=1.0)
    scores = np.array([0.6, 0.7, 0.8, 0.9])
    t, mu, sigma = f.compute_threshold(scores, tau_sat=None)
    assert t == pytest.approx(mu + sigma)


def test_cap_lowers_ceiling_for_homogeneous_pool(monkeypatch):
    monkeypatch.delenv("HCAD_TAU_SAT", raising=False)
    f = StatisticalFilter(k=1.0)
    scores = np.array([0.70] * 5 + [0.69] * 5)  # homogeneous high pool
    t_plain, _, _ = f.compute_threshold(scores, tau_sat=None)
    assert t_plain == pytest.approx(0.70)  # mu=0.695, sigma=0.005
    t_capped, _, _ = f.compute_threshold(scores, tau_sat=0.689)
    assert t_capped == pytest.approx(0.689)
    # survivors under the cap include candidates the adaptive gate rejected
    keep_plain = (scores >= t_plain).sum()
    keep_capped = (scores >= t_capped).sum()
    assert keep_capped > keep_plain


def test_cap_never_drops_below_absolute_floor():
    f = StatisticalFilter(k=1.0)
    scores = np.array([0.05, 0.06, 0.07])
    t, _, _ = f.compute_threshold(scores, tau_sat=0.90)
    assert t == pytest.approx(cfg.MIN_PHI_SCORE)


def test_cap_reads_env_calibration(monkeypatch):
    f = StatisticalFilter(k=1.0)
    scores = np.array([0.70] * 10)
    monkeypatch.setenv("HCAD_TAU_SAT", "0.60")
    t, _, _ = f.compute_threshold(scores, tau_sat=None)
    assert t == pytest.approx(0.60)
    monkeypatch.delenv("HCAD_TAU_SAT")
    t, _, _ = f.compute_threshold(scores, tau_sat=None)
    assert t == pytest.approx(0.70)


# --- 5. Relation-vocabulary mapping -----------------------------------------

def test_vocab_maps_primekg_labels_to_verb_phrases():
    assert cfg.map_relation_text("drug_protein") == "targets or binds protein"
    assert cfg.map_relation_text("contraindication") == \
        "is contraindicated in and must not be used for"
    assert cfg.map_relation_text("disease_phenotype_negative") == \
        "excludes phenotype and is not characterized by"


def test_vocab_unknown_relations_keep_legacy_rendering():
    assert cfg.map_relation_text("some_new_relation") == "some new relation"
    assert cfg.map_relation_text("") == ""


def test_vocab_disabled_by_env(monkeypatch):
    monkeypatch.setenv("HCAD_RELATION_VOCAB", "0")
    assert cfg.map_relation_text("drug_protein") == "drug protein"


def test_vocab_covers_benchmark_relation_strata():
    strata = [
        "contraindication", "indication", "off-label use", "drug_effect",
        "disease_disease", "drug_protein", "disease_protein",
        "disease_phenotype_positive", "disease_phenotype_negative",
        "biomarker", "phenotype_protein", "protein_protein",
    ]
    for rel in strata:
        assert rel in cfg.RELATION_VOCAB_MAP, f"unmapped benchmark relation: {rel}"


def test_masking_toggle_env(monkeypatch):
    monkeypatch.setenv("HCAD_PRESENCE_MASKING", "0")
    assert cfg.presence_masking_enabled() is False
    monkeypatch.setenv("HCAD_PRESENCE_MASKING", "1")
    assert cfg.presence_masking_enabled() is True
    monkeypatch.delenv("HCAD_PRESENCE_MASKING")
    assert cfg.presence_masking_enabled() is True  # default on
