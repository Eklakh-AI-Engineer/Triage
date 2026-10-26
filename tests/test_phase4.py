"""Phase 4 unit tests: soft hub weighting + config-versioned decoupler cache.

Covers:
  1. soft inverse-log-degree weight math,
  2. the three neighbor-expansion rules (desc / soft / asc_cap),
  3. per-rule hyperedge importance,
  4. decoupler config-version hashing,
  5. DecouplerCache component + embedding LRU behavior and version guards,
  6. decouple_batch cache integration (NER skipped + encode skipped on hits).
"""

import json

import networkx as nx
import numpy as np
import pytest

import src.config as cfg
from src.core.decoupler_cache import DecouplerCache, decoupler_config_version


# ---------------------------------------------------------------------------
# 1. Soft hub weight
# ---------------------------------------------------------------------------

def test_hub_weight_math():
    assert cfg.hub_weight(0) == pytest.approx(1.0)
    assert cfg.hub_weight(3) == pytest.approx(1.0 / (1.0 + np.log1p(3)))
    # a 100k-degree hub keeps ~8% weight: reachable, only damped
    assert cfg.hub_weight(100_000) == pytest.approx(1.0 / (1.0 + np.log1p(100_000)))
    # monotone non-increasing
    weights = [cfg.hub_weight(d) for d in (0, 1, 3, 10, 100, 1000, 100_000)]
    assert all(a >= b for a, b in zip(weights, weights[1:]))
    assert all(w > 0 for w in weights), "soft weight must never hit zero (no hard cap)"


def test_expansion_rule_env_toggle(monkeypatch):
    monkeypatch.setenv("HCAD_EXPANSION_RULE", "desc")
    assert cfg.expansion_rule() == "desc"
    monkeypatch.setenv("HCAD_EXPANSION_RULE", "asc_cap")
    assert cfg.expansion_rule() == "asc_cap"
    monkeypatch.delenv("HCAD_EXPANSION_RULE")
    assert cfg.expansion_rule() == "soft"  # deployed default
    monkeypatch.setenv("HCAD_EXPANSION_RULE", "bogus")
    with pytest.raises(ValueError, match="HCAD_EXPANSION_RULE"):
        cfg.expansion_rule()


# ---------------------------------------------------------------------------
# 2/3. Expansion rules + importance on a hub/specific toy graph
# ---------------------------------------------------------------------------

@pytest.fixture()
def toy_graph():
    """Hub node H (degree 8) + specific nodes S1/S2 (degree 2)."""
    g = nx.MultiDiGraph()
    g.add_node("H", name="Hub", node_type="protein")
    g.add_node("S1", name="Specific1", node_type="protein")
    g.add_node("S2", name="Specific2", node_type="protein")
    g.add_node("Q", name="Query", node_type="drug")
    # hub: 8 edges
    for i in range(8):
        g.add_edge("Q", "H", relation="drug_protein")
    # specific: 2 edges each
    for _ in range(2):
        g.add_edge("Q", "S1", relation="drug_protein")
        g.add_edge("Q", "S2", relation="drug_protein")
    return g


def _builder(toy_graph):
    from src.core.hypergraph_builder import HypergraphBuilder

    return HypergraphBuilder(toy_graph)


def test_desc_rule_puts_hub_first(toy_graph, monkeypatch):
    monkeypatch.setenv("HCAD_EXPANSION_RULE", "desc")
    b = _builder(toy_graph)
    ordered = b._ordered_neighbors({"H", "S1", "S2"})
    assert ordered[0] == "H", "legacy rule truncates toward the hub"


def test_soft_rule_puts_specific_first_but_keeps_hub(toy_graph, monkeypatch):
    monkeypatch.setenv("HCAD_EXPANSION_RULE", "soft")
    b = _builder(toy_graph)
    ordered = b._ordered_neighbors({"H", "S1", "S2"})
    assert ordered[0] in ("S1", "S2"), "specific targets win the slice"
    assert "H" in ordered, "soft rule must NOT hard-drop the hub"


def test_asc_cap_rule_hard_drops_hub(toy_graph, monkeypatch):
    monkeypatch.setenv("HCAD_EXPANSION_RULE", "asc_cap")
    monkeypatch.setenv("HCAD_HUB_DEGREE_CAP", "3")
    b = _builder(toy_graph)
    ordered = b._ordered_neighbors({"H", "S1", "S2"})
    assert "H" not in ordered, "hard cap must drop the >cap hub (ablation arm)"
    assert set(ordered) == {"S1", "S2"}


def test_soft_importance_penalizes_hub_stuffed_hyperedges(toy_graph, monkeypatch):
    b = _builder(toy_graph)
    hub_edge = {"nodes": ["H", "H", "H", "H"]}
    specific_edge = {"nodes": ["S1", "S2"]}
    monkeypatch.setenv("HCAD_EXPANSION_RULE", "desc")
    legacy_hub = b.compute_hyperedge_importance(hub_edge)
    legacy_specific = b.compute_hyperedge_importance(specific_edge)
    assert legacy_hub > legacy_specific, "legacy log-degree sum favors the hub"
    monkeypatch.setenv("HCAD_EXPANSION_RULE", "soft")
    soft_hub = b.compute_hyperedge_importance(hub_edge)
    soft_specific = b.compute_hyperedge_importance(specific_edge)
    assert soft_specific > soft_hub, "soft IEF must rank specific edges above hub-stuffed ones"
    assert soft_hub > 0, "hub importance is damped, never zeroed"


# ---------------------------------------------------------------------------
# 4. Config version
# ---------------------------------------------------------------------------

def test_config_version_is_stable_and_sensitive(monkeypatch):
    v1 = decoupler_config_version()
    assert v1 == decoupler_config_version()
    monkeypatch.setattr(cfg, "ORTHO_STRENGTH", 0.5)
    v2 = decoupler_config_version()
    assert v2 != v1, "eta change must invalidate the cache namespace"
    monkeypatch.setattr(cfg, "ORTHO_STRENGTH", 0.70)
    assert decoupler_config_version() == v1
    assert decoupler_config_version(["extra"]) != v1, "extra components participate"


# ---------------------------------------------------------------------------
# 5. DecouplerCache
# ---------------------------------------------------------------------------

def test_component_cache_roundtrip_and_disk(tmp_path):
    cache = DecouplerCache(version="vA", disk_dir=tmp_path)
    assert cache.get_components("doc one") is None
    cache.put_components("doc one", "Metformin", "is indicated for", "no qualifier stated")
    assert cache.get_components("doc one") == (
        "Metformin", "is indicated for", "no qualifier stated",
    )
    cache.flush()
    # reload from disk
    cache2 = DecouplerCache(version="vA", disk_dir=tmp_path)
    assert cache2.get_components("doc one") == (
        "Metformin", "is indicated for", "no qualifier stated",
    )


def test_component_cache_ignores_foreign_version(tmp_path):
    cache = DecouplerCache(version="vA", disk_dir=tmp_path)
    cache.put_components("doc one", "a", "b", "c")
    cache.flush()
    other = DecouplerCache(version="vB", disk_dir=tmp_path)
    assert other.get_components("doc one") is None, \
        "a cache file written under another config version must be ignored"


def test_embedding_lru_eviction(tmp_path):
    cache = DecouplerCache(version="vA", dim=4, embed_maxsize=3, disk_dir=tmp_path)
    for i in range(3):
        cache.put_embedding(f"t{i}", np.full(4, i, dtype=np.float32))
    assert cache.get_embedding("t0") is not None
    cache.put_embedding("t3", np.full(4, 3.0, dtype=np.float32))  # evicts t1 (LRU)
    assert cache.get_embedding("t1") is None
    assert cache.get_embedding("t0") is not None, "recently used entry survives"
    assert cache.n_embed_hits >= 2 and cache.n_embed_misses >= 1


def test_cache_stats(tmp_path):
    cache = DecouplerCache(version="vA", disk_dir=tmp_path)
    cache.put_components("d", "e", "r", "m")
    cache.put_embedding("x", np.zeros(4, dtype=np.float32))
    stats = cache.stats()
    assert stats["version"] == "vA"
    assert stats["component_entries"] == 1
    assert stats["embed_entries"] == 1


def test_cache_disabled_via_env(monkeypatch):
    monkeypatch.setenv("HCAD_DECOUPLER_CACHE", "0")
    from src.core import decoupler_cache as mod

    mod._SHARED = None
    assert mod.get_decoupler_cache() is None
    monkeypatch.setenv("HCAD_DECOUPLER_CACHE", "1")
    mod._SHARED = None
    assert mod.get_decoupler_cache() is not None
    mod._SHARED = None


# ---------------------------------------------------------------------------
# 6. decouple_batch integration: NER + encode skipped on cache hits
# ---------------------------------------------------------------------------

class _CountingModel:
    def __init__(self):
        self.encode_calls = 0

    def encode(self, texts, batch_size=32, show_progress_bar=False,
               convert_to_numpy=True, normalize_embeddings=True, **_):
        if isinstance(texts, str):
            texts = [texts]
        self.encode_calls += 1
        out = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            vec = rng.standard_normal(64).astype(np.float32)
            vec /= np.linalg.norm(vec) + 1e-8
            out.append(vec)
        # sentence-transformers always returns (N, d) for list input
        return np.stack(out)


class _CountingNER:
    def __init__(self):
        self.calls = 0

    def extract_entities(self, text):
        self.calls += 1
        first_word = (text or "Entity").split()[0].rstrip(".,;:")
        return {"Drug": [{"text": first_word, "start": 0, "end": len(first_word)}]}


def _make_decoupler_with_cache(tmp_path):
    from src.core.semantic_decoupler import SemanticDecoupler

    d = SemanticDecoupler(lazy=True)
    d._model = _CountingModel()
    d._dim = 64
    d._cache = DecouplerCache(version="vTest", dim=64, disk_dir=tmp_path)
    d._cache_resolved = True
    d._ner = _CountingNER()
    # stub the relation extractor too (unsupervised path calls it)
    d._rex = type("R", (), {"extract": staticmethod(
        lambda text, ents: [{"relation": "is indicated for", "modifier": "severe"}])})()
    return d


def test_decouple_batch_skips_ner_and_encode_on_second_call(tmp_path):
    d = _make_decoupler_with_cache(tmp_path)
    text = "Metformin is indicated for severe renal failure."
    d.decouple_batch([text])
    encode_after_first = d._model.encode_calls
    ner_after_first = d._ner.calls
    assert ner_after_first == 1

    d.decouple_batch([text])
    assert d._ner.calls == ner_after_first, "component cache must skip repeat NER"
    assert d._model.encode_calls == encode_after_first, "embedding LRU must skip repeat encodes"
    assert d._cache.n_component_hits >= 1
    assert d._cache.n_embed_hits >= 3


def test_decouple_batch_supervised_path_not_component_cached(tmp_path):
    d = _make_decoupler_with_cache(tmp_path)
    text = "Metformin is indicated for severe renal failure."
    entities = {"Drug": [{"text": "Metformin", "start": 0, "end": 9}]}
    d.decouple_batch([text], entities_list=[entities])
    assert d._cache.get_components(text) is None, \
        "explicitly-supplied entities must not populate the component cache"


def test_decouple_batch_partial_cache_hit(tmp_path):
    d = _make_decoupler_with_cache(tmp_path)
    text_a = "Metformin is indicated for severe renal failure."
    text_b = "Aspirin is contraindicated in asthma."
    d.decouple_batch([text_a])
    calls_after_a = d._model.encode_calls
    d.decouple_batch([text_a, text_b])
    # only text_b's strings should be encoded now
    assert d._model.encode_calls == calls_after_a + 1
