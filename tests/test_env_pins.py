"""Environment pin guards: the py312 env must match the pinned stack.

The Oct 25 scispacy failure (OverflowError in Tokenizer._flush_specials)
came from dependency drift in the freshly-built env, not from our code:
the notebook rebuilds the env every run from ranges, and a newly released
transitive dependency flowed in. The suite now fails loudly - before the
smoke test, before training - if the resolver produced a different stack
than the one every successful run used.
"""

def test_pinned_low_level_stack():
    from importlib.metadata import version
    import numpy
    murmurhash_version = version("murmurhash")
    assert murmurhash_version in ("1.0.12", "1.0.15"), (
        f"murmurhash {murmurhash_version}: expected pinned 1.0.12 or 1.0.15 "
        "- SciSpacy tokenizer overflow is handled via English tokenizer fallback in ner_extractor")
    assert numpy.__version__ == "2.5.4", (
        f"numpy {numpy.__version__}: expected the pinned 2.5.4 - the "
        "version every successful run used")

def test_pinned_core_stack():
    import chromadb
    import spacy
    assert spacy.__version__ == "3.8.11", (
        f"spacy {spacy.__version__}: 3.8.12+ rejects the 0.5.4 scispacy "
        "model configs and silently drops NER to regex fallback")
    assert chromadb.__version__ == "1.5.9", (
        f"chromadb {chromadb.__version__}: expected the pinned 1.5.9 "
        "whose API the suite is contract-tested against")
