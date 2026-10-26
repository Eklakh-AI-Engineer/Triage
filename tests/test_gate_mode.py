"""Gate-mode provenance: the JEV gate's active mode must be discoverable.

The Oct-25 benchmark silently ran with the live TypeSafe gate because a key
was present in the environment, making it incomparable with the
local-calibrated anchor. DecisionGate.mode and HCADRetriever.gate_mode
exist so every run can record - and fail-loud check - the active mode.
"""

from src.core.decision_gate import DecisionGate
from src.core.hcad_retriever import HCADRetriever


def test_gate_mode_local_when_no_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    gate = DecisionGate()
    assert gate.mode == "local-calibrated"


def test_retriever_gate_mode_uninitialized():
    retriever = HCADRetriever.__new__(HCADRetriever)
    retriever._decision_gate = None
    assert retriever.gate_mode == "uninitialized"


def test_retriever_gate_mode_delegates_to_gate(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    retriever = HCADRetriever.__new__(HCADRetriever)
    retriever._decision_gate = DecisionGate()
    assert retriever.gate_mode == "local-calibrated"


def test_retriever_gate_mode_tolerates_gate_without_mode():
    retriever = HCADRetriever.__new__(HCADRetriever)
    retriever._decision_gate = object()
    assert retriever.gate_mode == "unknown"
