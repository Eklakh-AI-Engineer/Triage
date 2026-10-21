"""Tests: BiomedicalNER is a process-wide singleton (no mid-benchmark
SciSpacy reloads, which caused 40-54 s retrieval spikes)."""

import src.core.ner_extractor as ner_mod
from src.core.ner_extractor import BiomedicalNER, get_biomedical_ner


def setup_function():
    ner_mod._NER_SINGLETON = None


def teardown_function():
    ner_mod._NER_SINGLETON = None


def test_factory_returns_same_instance():
    a = get_biomedical_ner(lazy=True)
    b = get_biomedical_ner(lazy=True)
    assert a is b
    assert ner_mod._NER_SINGLETON is a


def test_non_lazy_request_upgrades_lazy_singleton():
    a = get_biomedical_ner(lazy=True)
    loaded = []

    def _fake_load(self):
        loaded.append(True)
        self._models_loaded = True

    orig = BiomedicalNER._load_models
    BiomedicalNER._load_models = _fake_load
    try:
        b = get_biomedical_ner(lazy=False)
    finally:
        BiomedicalNER._load_models = orig
    assert b is a
    assert loaded == [True], "non-lazy request must trigger model loading once"


def test_repeated_non_lazy_requests_load_only_once():
    loaded = []

    def _fake_load(self):
        loaded.append(True)
        self._models_loaded = True

    orig = BiomedicalNER._load_models
    BiomedicalNER._load_models = _fake_load
    try:
        a = get_biomedical_ner(lazy=False)
        b = get_biomedical_ner(lazy=False)
    finally:
        BiomedicalNER._load_models = orig
    assert a is b
    assert loaded == [True], "models must load exactly once per process"
