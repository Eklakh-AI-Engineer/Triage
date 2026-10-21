"""Tests for the paper-protocol question cache (resume safety) and the
stratified dev/test question pools (leakage safety)."""

import json

import pytest

import src.evaluation.run_base_paper_benchmark as runner


@pytest.fixture
def cache_env(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    return tmp_path


def _cache_file(tmp_path, n=5, pool="test"):
    return tmp_path / f"paper_questions_n{n}_seed{runner.RANDOM_SEED}_{pool}.json"


def _legacy_cache_file(tmp_path, n=5):
    return tmp_path / f"paper_questions_n{n}_seed{runner.RANDOM_SEED}.json"


def test_cached_questions_are_reused_without_regeneration(cache_env, monkeypatch):
    cached = [
        {"question": "Cached question?", "reference": "chunk text",
         "ground_truth": "", "source": "paper_protocol", "difficulty_tier": 1},
    ]
    _cache_file(cache_env).write_text(json.dumps(cached), encoding="utf-8")

    # Regeneration must never run: explode if it tries.
    def _boom(*a, **k):
        raise AssertionError("regeneration should not run when cache is valid")
    monkeypatch.setattr(
        "src.data.build_vector_store.load_vector_store", _boom)

    out = runner.generate_paper_protocol_questions(5)
    assert out == cached


def test_legacy_cache_is_loaded_with_a_warning(cache_env, monkeypatch):
    cached = [
        {"question": "Legacy question?", "reference": "chunk text",
         "ground_truth": "", "source": "paper_protocol", "difficulty_tier": 1},
    ]
    _legacy_cache_file(cache_env).write_text(json.dumps(cached), encoding="utf-8")

    def _boom(*a, **k):
        raise AssertionError("regeneration should not run when cache is valid")
    monkeypatch.setattr(
        "src.data.build_vector_store.load_vector_store", _boom)

    out = runner.generate_paper_protocol_questions(5)
    assert out == cached


def test_corrupt_cache_falls_back_to_regeneration(cache_env, monkeypatch):
    _cache_file(cache_env).write_text("{not json", encoding="utf-8")

    # If the cache is bypassed, regeneration is attempted - make it explode.
    def _boom(*a, **k):
        raise RuntimeError("regeneration attempted")
    monkeypatch.setattr(
        "src.data.build_vector_store.load_vector_store", _boom)

    with pytest.raises(RuntimeError, match="regeneration attempted"):
        runner.generate_paper_protocol_questions(5)


class FakeStratifiedStore:
    """Chroma-like stub: serves PMc-style and triple-style docs across strata."""

    def __init__(self, n_docs=400):
        import hashlib
        self.docs, self.ids, self.metas = [], [], []
        relations = runner.TRIPLE_RELATION_STRATA
        for i in range(n_docs):
            if i % 4 == 0:  # PMC-style long chunk
                text = ("Clinical trial evidence for therapy of condition "
                        + "C" * 40 + ". " + "Detailed methodology and outcomes "
                        "are described at length. " * 6)
                meta = {"source": "pmc", "doc_id": f"PMC{i}", "relation": "",
                        "head_name": "", "doc_type": "clinical_text"}
                doc_id = f"pmc_PMC{i}_0"
            else:
                rel = relations[i % len(relations)]
                text = (f"Drug{i} {rel.replace('_', ' ')} Target{i} in the context of clinical evidence and therapeutic use.")
                meta = {"relation": rel, "head_name": f"Drug{i}", "source": "",
                        "doc_type": ""}
                doc_id = f"edge_{i}"
            h = int(hashlib.sha1(doc_id.encode()).hexdigest()[:8], 16)
            # keep both pools represented in the stub
            self.docs.append(text)
            self.ids.append(doc_id)
            self.metas.append(meta)
        self._docs = list(zip(self.docs, self.ids, self.metas))

    def count(self):
        return len(self.docs)

    def get(self, where=None, limit=None, offset=None, include=None):
        # Mirror chromadb >= 1.x include validation: "ids" is not a valid
        # include item for get() (ids are always returned).
        if include and "ids" in include:
            raise ValueError(
                "Expected include item to be one of documents, embeddings, "
                "metadatas, distances, uris, data, got ids in get.")
        pool_pred = None
        if where:
            if "source" in where:
                items = [t for t in self._docs if t[2].get("source") == where["source"]]
            else:
                items = [t for t in self._docs if t[2].get("relation") == where["relation"]]
        else:
            items = self._docs
        offset = offset or 0
        return {
            "documents": [t[0] for t in items[offset:offset + limit]],
            "ids": [t[1] for t in items[offset:offset + limit]],
            "metadatas": [t[2] for t in items[offset:offset + limit]],
        }


def test_generated_questions_are_written_to_cache(cache_env, monkeypatch):
    monkeypatch.setattr(
        "src.data.build_vector_store.load_vector_store",
        lambda: FakeStratifiedStore())

    class FakeGen:
        def __init__(self, temperature=0.3):
            self.calls = 0

        def chat(self, messages):
            self.calls += 1
            return "What does this chunk describe in the body medically?"

    fake_gen = FakeGen()
    runner.generate_paper_protocol_questions(6, generator=fake_gen, pool="test")

    cache = _cache_file(cache_env, n=6)
    assert cache.exists()
    loaded = json.loads(cache.read_text(encoding="utf-8"))
    assert len(loaded) == 6
    assert loaded[0]["question"] == "What does this chunk describe in the body medically?"
    # stratified provenance recorded for leakage audits
    for s in loaded:
        assert s["source_chunk_id"]
        assert s["source_kind"] in ("pmc", "primekg")
        assert s["pool"] == "test"


def test_dev_and_test_pools_are_disjoint(cache_env, monkeypatch):
    """The training (dev) and evaluation (test) pools must never share a chunk."""
    monkeypatch.setattr(
        "src.data.build_vector_store.load_vector_store",
        lambda: FakeStratifiedStore(n_docs=2000))

    class FakeGen:
        def __init__(self, temperature=0.3):
            pass

        def chat(self, messages):
            return "Question about the statement in this source chunk?"

    test_set = runner.generate_paper_protocol_questions(6, generator=FakeGen(),
                                                         pool="test")
    dev_set = runner.generate_paper_protocol_questions(6, generator=FakeGen(),
                                                        pool="dev")
    test_ids = {s["source_chunk_id"] for s in test_set}
    dev_ids = {s["source_chunk_id"] for s in dev_set}
    assert test_ids and dev_ids
    assert test_ids.isdisjoint(dev_ids), "dev/test pools must be disjoint"


def test_questions_come_from_distinct_chunks(cache_env, monkeypatch):
    """Every question must originate from a distinct source chunk."""
    monkeypatch.setattr(
        "src.data.build_vector_store.load_vector_store",
        lambda: FakeStratifiedStore(n_docs=2000))

    class FakeGen:
        def __init__(self, temperature=0.3):
            pass

        def chat(self, messages):
            return "Question about the association described here?"

    out = runner.generate_paper_protocol_questions(9, generator=FakeGen(),
                                                   pool="test")
    refs = [s["reference"] for s in out]
    assert len(refs) == len(set(refs)), "questions must have distinct references"
    ids = [s["source_chunk_id"] for s in out]
    assert len(ids) == len(set(ids))


def test_invalid_pool_rejected(cache_env):
    with pytest.raises(ValueError, match="pool"):
        runner.generate_paper_protocol_questions(5, pool="train")
