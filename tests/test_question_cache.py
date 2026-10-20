"""Tests for the paper-protocol question cache (resume safety)."""

import json

import pytest

import src.evaluation.run_base_paper_benchmark as runner


@pytest.fixture
def cache_env(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    return tmp_path


def _cache_file(tmp_path, n=5):
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


def test_corrupt_cache_falls_back_to_regeneration(cache_env, monkeypatch):
    _cache_file(cache_env).write_text("{not json", encoding="utf-8")

    # If the cache is bypassed, regeneration is attempted - make it explode.
    def _boom(*a, **k):
        raise RuntimeError("regeneration attempted")
    monkeypatch.setattr(
        "src.data.build_vector_store.load_vector_store", _boom)

    with pytest.raises(RuntimeError, match="regeneration attempted"):
        runner.generate_paper_protocol_questions(5)


def test_generated_questions_are_written_to_cache(cache_env, monkeypatch):
    fake_samples = [
        {"question": "Q1?", "reference": "ref1", "source": "paper_protocol",
         "ground_truth": "", "difficulty_tier": 1},
        {"question": "Q2?", "reference": "ref2", "source": "paper_protocol",
         "ground_truth": "", "difficulty_tier": 2},
    ]

    class FakeStore:
        def count(self):
            return 100

        def get(self, limit=None, include=None):
            return {"documents": ["chunk one", "chunk two"]}

    monkeypatch.setattr(
        "src.data.build_vector_store.load_vector_store",
        lambda: FakeStore())

    class FakeGen:
        def __init__(self, temperature=0.3):
            self.calls = 0

        def chat(self, messages):
            self.calls += 1
            return "What does this chunk describe in the body medically?"

    fake_gen = FakeGen()
    runner.generate_paper_protocol_questions(2, generator=fake_gen)

    cache = _cache_file(cache_env, n=2)
    assert cache.exists()
    loaded = json.loads(cache.read_text(encoding="utf-8"))
    assert len(loaded) == 2
    assert loaded[0]["question"] == "What does this chunk describe in the body medically?"
