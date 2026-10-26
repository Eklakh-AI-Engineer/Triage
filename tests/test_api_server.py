"""Tests for the TRIAGE workbench HTTP API (src/api/server.py).

Uses a stub pipeline provider so no knowledge graph, vector store, or LLM
is loaded. Verifies the contract the Next.js workbench consumes.
"""

import json

import networkx as nx
import pytest
from fastapi.testclient import TestClient

from src.api import server


class StubHCAD:
    def __init__(self, contexts, answer, pool=None):
        self._contexts = contexts
        self._answer = answer
        self._pool = pool if pool is not None else contexts
        self._graph = None

    def retrieve(self, query, k=None):
        return list(self._contexts)

    def retrieve_and_generate(self, query, contexts=None):
        return self._answer

    def _ensure_components(self):
        pass

    class _ner:  # noqa: N801 - mirrors retriever attribute name
        @staticmethod
        def extract_entities(query):
            return {"Drug": [{"text": "Metformin", "start": 0, "end": 9, "label": "DRUG"}],
                    "Disease": [{"text": "Renal Failure", "start": 40, "end": 53, "label": "DISEASE"}]}

    class _relation_extractor:  # noqa: N801
        @staticmethod
        def extract_rules(query, entities):
            return [{"relation": "contraindication"}]

    def _get_candidate_contexts(self, query, entities, n_vector=20):
        return list(self._pool)

    @property
    def graph(self):
        if self._graph is None:
            g = nx.MultiDiGraph()
            g.add_node("m", name="Metformin", node_type="drug")
            g.add_node("ckd", name="Chronic Kidney Disease", node_type="disease")
            g.add_node("la", name="Lactic Acidosis", node_type="disease")
            g.add_edge("m", "ckd", relation="contraindication")
            g.add_edge("m", "la", relation="drug_effect")
            self._graph = g
        return self._graph


class StubVanilla:
    def retrieve(self, query, top_k=5):
        return [{"document": "fragment", "metadata": {}, "score": 0.4}]

    def retrieve_and_generate(self, query, contexts=None):
        return "Vanilla RAG answer with distractor fragments."


class StubBaseline:
    def generate(self, query):
        return "Baseline parametric answer without sources."


def _contexts(n=2):
    return [
        {
            "document": f"Metformin is contraindicated in CKD. Fact {i}.",
            "metadata": {
                "head_name": "Metformin",
                "tail_name": "Chronic Kidney Disease",
                "relation": "contraindication",
                "display_relation": "contraindication",
            },
            "phi_final": 0.8 - i * 0.1,
            "source": "graph",
        }
        for i in range(n)
    ]


@pytest.fixture
def client():
    stub_hcad = StubHCAD(
        _contexts(2),
        "Metformin should be avoided in CKD [Evidence 1] due to lactic acidosis [Evidence 2].",
    )
    server.provider = server.PipelineProvider()
    server.provider._hcad = stub_hcad
    server.provider._vanilla = StubVanilla()
    server.provider._baseline = StubBaseline()
    return TestClient(server.app)


def test_health(client):
    res = client.get("/api/triage/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["pipelines_loaded"]["hcad"] is True


def test_query_pass_with_citation_segments(client):
    res = client.post(
        "/api/triage/query",
        json={"query": "Metformin in renal failure?", "config": {"tauEntropy": 0.25}},
    )
    assert res.status_code == 200
    body = res.json()

    assert body["gate"]["decision"] == "PASS"
    assert body["gate"]["admittedCount"] == 2
    assert body["gate"]["jev"] == pytest.approx(0.75, abs=1e-6)
    assert body["gate"]["entropyDelta"] == pytest.approx(-0.1, abs=1e-6)
    assert body["abstentionRate"] == 0.0
    assert body["latencyMs"] >= 0.0

    # P2d: grounded-mode provenance - evidence-backed answers are grounded.
    assert body["groundedMode"] == "grounded"
    assert body["gate"]["groundedMode"] == "grounded"

    # Entities mapped to workbench types.
    names = {e["name"]: e["type"] for e in body["entities"]}
    assert names["Metformin"] == "drug"
    assert names["Renal Failure"] == "disease"

    # Raw candidates carry ADMIT verdicts for admitted docs.
    verdicts = {c["verdict"] for c in body["rawCandidates"]}
    assert verdicts == {"ADMIT"}

    # Answer parsed into text + citation segments with resolvable ids.
    kinds = [s["kind"] for s in body["arena"]["triage"]]
    assert kinds[0] == "text"
    assert "citation" in kinds
    citation_ids = [s["id"] for s in body["arena"]["triage"] if s["kind"] == "citation"]
    assert citation_ids == ["Evidence:1", "Evidence:2"]
    for cid in citation_ids:
        record = body["citations"][cid]
        assert record["triple"].startswith("Metformin")
        assert record["confidence"] > 0
    assert body["arena"]["warnings"] == []


def test_query_single_context_passes(client):
    """P2d: one admitted context is sufficient grounding for a PASS.

    The dual gate already enforces the per-context relevance floor; the old
    >= 2 rule conflated "thin evidence" with "no evidence" and inflated the
    abstention rate.
    """
    server.provider._hcad = StubHCAD(
        _contexts(1), "Metformin should be avoided in CKD [Evidence 1]."
    )
    res = client.post(
        "/api/triage/query", json={"query": "Metformin in renal failure?", "config": {}}
    )
    assert res.status_code == 200
    body = res.json()
    assert body["gate"]["decision"] == "PASS"
    assert body["gate"]["admittedCount"] == 1
    assert body["groundedMode"] == "grounded"


def test_query_abstain(client):
    server.provider._hcad = StubHCAD([], "Parametric fallback answer.")
    res = client.post(
        "/api/triage/query", json={"query": "Out-of-domain noise?", "config": {}}
    )
    assert res.status_code == 200
    body = res.json()
    assert body["gate"]["decision"] == "ABSTAIN"
    assert body["gate"]["admittedCount"] == 0
    assert body["gate"]["contextWords"] == 0
    assert body["abstentionRate"] == 100.0
    assert body["groundedMode"] == "parametric"
    assert body["gate"]["groundedMode"] == "parametric"
    assert body["arena"]["warnings"]
    assert body["arena"]["triage"][0]["kind"] == "text"
    assert "Evidence" not in body["citations"]


def test_query_out_of_range_citation_tag_stays_text(client):
    server.provider._hcad = StubHCAD(
        _contexts(2), "Answer cites a missing item [Evidence 9] here."
    )
    body = client.post("/api/triage/query", json={"query": "clinical question here?"}).json()
    segments = body["arena"]["triage"]
    assert all(s["kind"] == "text" for s in segments)
    assert any("[Evidence 9]" in s["value"] for s in segments)


def test_subgraph_two_hop(client):
    res = client.get("/api/triage/subgraph", params={"query": "Metformin and CKD"})
    assert res.status_code == 200
    body = res.json()
    assert body["centerId"] == "m"
    assert body["hops"] == 2
    ids = {n["id"] for n in body["nodes"]}
    assert {"m", "ckd", "la"} <= ids
    labels = {n["id"]: n["label"] for n in body["nodes"]}
    assert labels["m"] == "Metformin"
    center = [n for n in body["nodes"] if n["id"] == "m"][0]
    assert center["trace"] == 1.0
    assert {e["relation"] for e in body["edges"]} == {"contraindication", "drug effect"}
    assert all(e["source"] in ids and e["target"] in ids for e in body["edges"])


def test_subgraph_no_match(client):
    res = client.get("/api/triage/subgraph", params={"query": "zzz-no-such-entity"})
    assert res.status_code == 404


def test_benchmark_mapping(tmp_path, monkeypatch):
    results = {
        "num_samples": 42,
        "scoring_based_assessment": {
            "HCAD-RAG": {"overall_score": 88.0, "logical": 85.0, "comprehensiveness": 84.0},
            "Vanilla RAG": {"overall_score": 71.0, "logical": 58.0, "comprehensiveness": 72.0},
            "Baseline LLM": {"overall_score": 62.0, "logical": 49.0, "comprehensiveness": 66.0},
        },
        "selection_based_assessment": {
            "HCAD_vs_Vanilla": {"criteria_win_rates_pct": {"Accuracy": 57.1, "Coherence": 71.4}},
        },
        "citation_precision": {
            "HCAD-RAG": {"mean_precision": 0.92},
            "Vanilla RAG": {"mean_precision": 0.5},
        },
        "significance_testing": {
            "HCAD_vs_Vanilla": {"n01": 7, "n10": 1, "p_value": 0.07,
                                  "significant_at_05": False},
        },
    }
    monkeypatch.setattr(server, "RESULTS_DIR", tmp_path)
    (tmp_path / "base_paper_benchmark_results.json").write_text(json.dumps(results))

    # Ensure a provider exists for app state even if unneeded for this route.
    server.provider = server.PipelineProvider()
    client = TestClient(server.app)
    res = client.get("/api/triage/benchmark")
    assert res.status_code == 200
    body = res.json()
    radar = {d["dimension"]: d for d in body["radar"]}
    assert radar["Diagnostic Accuracy"]["triage"] == 88.0
    assert radar["Evidence Faithfulness"]["triage"] == 92.0
    assert radar["Contradiction Avoidance"]["vanilla"] == 58.0
    assert {w["criterion"] for w in body["winRate"]} == {"Accuracy", "Coherence"}
    assert body["mcnemar"]["pValue"] == 0.07
    assert body["mcnemar"]["significant"] is False
    assert body["nQueries"] == 42


def test_benchmark_missing_results(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "RESULTS_DIR", tmp_path)
    server.provider = server.PipelineProvider()
    client = TestClient(server.app)
    res = client.get("/api/triage/benchmark")
    assert res.status_code == 404


def test_triage_query_error_sanitization():
    class FailingHCAD:
        def retrieve(self, query, k=None):
            raise RuntimeError("Database connection string password123 leaked")
        def _ensure_components(self):
            pass

    server.provider = server.PipelineProvider()
    server.provider._hcad = FailingHCAD()
    client = TestClient(server.app)
    res = client.post("/api/triage/query", json={"query": "Metformin and CKD"})
    assert res.status_code == 500
    assert res.json()["detail"] == "Internal pipeline processing error"
    assert "password123" not in res.text

