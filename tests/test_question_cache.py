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
    cached = [{"question": f"Cached question {i}?", "reference": "chunk text",
               "ground_truth": "", "source": "paper_protocol", "difficulty_tier": 1,
               "pool": "test", "source_chunk_id": f"chunk{i}"}
              for i in range(5)]
    _cache_file(cache_env).write_text(json.dumps(cached), encoding="utf-8")

    # Regeneration must never run: explode if it tries.
    def _boom(*a, **k):
        raise AssertionError("regeneration should not run when cache is valid")
    monkeypatch.setattr(
        "src.data.build_vector_store.load_vector_store", _boom)

    out = runner.generate_paper_protocol_questions(5)
    assert out == cached


def test_unscoped_legacy_cache_is_ignored_for_pool_generation(cache_env, monkeypatch):
    cached = [
        {"question": "Legacy question?", "reference": "chunk text",
         "ground_truth": "", "source": "paper_protocol", "difficulty_tier": 1},
    ]
    _legacy_cache_file(cache_env).write_text(json.dumps(cached), encoding="utf-8")

    monkeypatch.setattr(
        "src.data.build_vector_store.load_vector_store",
        lambda: FakeStratifiedStore(n_docs=2000))

    class FakeGen:
        def chat(self, messages):
            return "Generated question about this source evidence?"

    out = runner.generate_paper_protocol_questions(5, generator=FakeGen(), pool="dev")
    assert len(out) == 5
    assert all(item["pool"] == "dev" for item in out)
    assert all(item["question"] != "Legacy question?" for item in out)


def test_corrupt_cache_falls_back_to_regeneration(cache_env, monkeypatch):
    _cache_file(cache_env).write_text("{not json", encoding="utf-8")

    # If the cache is bypassed, regeneration is attempted - make it explode.
    def _boom(*a, **k):
        raise RuntimeError("regeneration attempted")
    monkeypatch.setattr(
        "src.data.build_vector_store.load_vector_store", _boom)

    with pytest.raises(RuntimeError, match="regeneration attempted"):
        runner.generate_paper_protocol_questions(5)


def test_question_cache_write_failure_is_fatal(cache_env, monkeypatch):
    import builtins

    monkeypatch.setattr(
        "src.data.build_vector_store.load_vector_store",
        lambda: FakeStratifiedStore(n_docs=2000))
    real_open = builtins.open
    cache = _cache_file(cache_env, n=5)

    def fail_cache_write(path, mode="r", *args, **kwargs):
        if str(path) == str(cache) and mode == "w":
            raise OSError("simulated full disk")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fail_cache_write)

    class FakeGen:
        def chat(self, messages):
            return "Generated question about this source evidence?"

    with pytest.raises(OSError, match="simulated full disk"):
        runner.generate_paper_protocol_questions(5, generator=FakeGen())


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


class FakeHeadSortedStore(FakeStratifiedStore):
    """PrimeKG rows grouped by head, matching kg.csv physical ordering."""

    def __init__(self, n_heads=400):
        self.docs, self.ids, self.metas = [], [], []
        for head_i in range(n_heads):
            for rel_i, relation in enumerate(runner.TRIPLE_RELATION_STRATA):
                head = f"Drug{head_i:04d}"
                doc_id = f"edge_{head_i:04d}_{rel_i}"
                self.docs.append(
                    f"{head} {relation.replace('_', ' ')} Target{head_i}_{rel_i} "
                    "in the context of clinical evidence and therapeutic use."
                )
                self.ids.append(doc_id)
                self.metas.append({"relation": relation, "head_name": head,
                                   "source": "", "doc_type": ""})
        self.docs.extend(
            [("Clinical trial evidence for therapy of condition " + "C" * 220)
             for _ in range(n_heads)]
        )
        self.ids.extend([f"pmc_{i}" for i in range(n_heads)])
        self.metas.extend([{"source": "pmc", "doc_id": f"PMC{i}",
                            "relation": "", "head_name": ""}
                           for i in range(n_heads)])
        self._docs = list(zip(self.docs, self.ids, self.metas))


def test_sorted_by_head_store_yields_diverse_pool_chunks():
    import random

    store = FakeHeadSortedStore()
    chunks = runner._fetch_stratified_chunks(store, 20, "test", random.Random(42))
    dev_chunks = runner._fetch_stratified_chunks(store, 20, "dev", random.Random(42))
    assert len(chunks) >= 40
    assert {chunk["kind"] for chunk in chunks} == {"pmc", "primekg"}
    assert len({chunk["id"] for chunk in chunks}) == len(chunks)
    assert len({chunk["doc"].split()[0] for chunk in chunks
                if chunk["kind"] == "primekg"}) >= 10
    assert all(runner._question_pool_of(chunk["id"], "test") for chunk in chunks)
    assert {chunk["id"] for chunk in chunks}.isdisjoint(
        {chunk["id"] for chunk in dev_chunks})


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
    ids = [s["source_chunk_id"] for s in out]
    assert len(ids) == len(set(ids)), "questions must have distinct source chunks"


def test_invalid_pool_rejected(cache_env):
    with pytest.raises(ValueError, match="pool"):
        runner.generate_paper_protocol_questions(5, pool="train")


def test_question_cache_path_is_pool_specific(cache_env):
    assert runner._paper_question_cache_path(50, "dev").name.endswith("_dev.json")
    assert runner._paper_question_cache_path(50, "test").name.endswith("_test.json")


def test_question_set_id_is_stable_and_changes_with_the_set():
    samples = [{"question": "q1", "source_chunk_id": "chunk1", "pool": "test"}]
    same_set = [{"question": "q1", "source_chunk_id": "chunk1", "pool": "test"}]
    other_set = [{"question": "q1", "source_chunk_id": "chunk2", "pool": "test"}]
    assert runner._question_set_id(samples) == runner._question_set_id(same_set)
    assert runner._question_set_id(samples) != runner._question_set_id(other_set)


def test_pool_split_is_article_level_not_chunk_level():
    """Chunks of the same PMC article (and triples of the same head drug)
    must land in the SAME dev/test pool.

    Chunk-id hashing let one article's paragraphs straddle both pools
    (Critical Review V3 leakage vector); the pool key must be the same
    article/entity key used for dedup.
    """
    import random

    docs, ids, metas = [], [], []
    for art in range(3):
        for chunk in range(6):
            docs.append(f"Article {art} chunk {chunk} presents with condition "
                        "and was managed per guideline with monitoring and "
                        "follow-up care assessment. " * 4)
            ids.append(f"pmc_ART{art}_{chunk}")
            metas.append({"source": "pmc", "doc_id": f"ART{art}", "relation": "",
                          "head_name": "", "doc_type": "clinical_text"})
    for drug in range(3):
        for tgt in range(4):
            docs.append(f"Drug{drug} drug protein Target{drug}_{tgt} in the "
                        "context of clinical evidence and therapeutic use.")
            ids.append(f"edge_{drug}_{tgt}")
            metas.append({"relation": "drug_protein", "head_name": f"Drug{drug}",
                          "source": "", "doc_type": ""})

    class SharedArticleStore:
        def __init__(self):
            self._docs = list(zip(docs, ids, metas))

        def count(self):
            return len(docs)

        def get(self, where=None, limit=None, offset=None, include=None):
            if include and "ids" in include:
                raise ValueError("ids is not a valid include item")
            if where:
                if "source" in where:
                    items = [t for t in self._docs
                             if t[2].get("source") == where["source"]]
                else:
                    items = [t for t in self._docs
                             if t[2].get("relation") == where["relation"]]
            else:
                items = self._docs
            off = offset or 0
            return {
                "documents": [t[0] for t in items[off:off + limit]],
                "ids": [t[1] for t in items[off:off + limit]],
                "metadatas": [t[2] for t in items[off:off + limit]],
            }

    store = SharedArticleStore()
    dev = runner._fetch_stratified_chunks(store, 12, "dev", random.Random(42))
    test = runner._fetch_stratified_chunks(store, 12, "test", random.Random(42))
    dev_ids = {c["id"] for c in dev}
    test_ids = {c["id"] for c in test}
    assert not (dev_ids & test_ids), "no chunk may appear in both pools"

    for art in ("ART0", "ART1", "ART2"):
        art_chunks = {cid for cid in dev_ids | test_ids
                      if cid.startswith(f"pmc_{art}_")}
        in_dev = any(cid in dev_ids for cid in art_chunks)
        in_test = any(cid in test_ids for cid in art_chunks)
        assert not (in_dev and in_test), \
            f"article {art} straddles dev/test pools (chunk-level hashing)"

    for drug in ("0", "1", "2"):
        drug_edges = {cid for cid in dev_ids | test_ids
                      if cid.startswith(f"edge_{drug}_")}
        in_dev = any(cid in dev_ids for cid in drug_edges)
        in_test = any(cid in test_ids for cid in drug_edges)
        assert not (in_dev and in_test), \
            f"Drug{drug} triples straddle dev/test pools (chunk-level hashing)"
