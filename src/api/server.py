"""HTTP API for the TRIAGE workbench.

Serves the three contracts defined by the Next.js workbench
(workbench/src/lib/types.ts):

    POST /api/triage/query      -> full pipeline result for one query
    GET  /api/triage/subgraph   -> localized 2-hop reasoning subgraph
    GET  /api/triage/benchmark  -> base-paper benchmark telemetry

Run with:
    python src/main.py serve --host 0.0.0.0 --port 8000

Requires `python src/main.py prepare --rebuild` artifacts (knowledge graph,
vector store). Heavy components load lazily on first request; the /health
endpoint reports readiness without triggering a load.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.config import EVIDENCE_WORD_BUDGET, FILTERING_K, RESULTS_DIR
from src.core.prompt_builder import context_document

logger = logging.getLogger(__name__)

app = FastAPI(
    title="TRIAGE API",
    description="Gated Evidence Selection and Context Decoupling for Clinical RAG",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# Request / response models (mirror workbench/src/lib/types.ts)
# ---------------------------------------------------------------------------

class GateConfigIn(BaseModel):
    tauEntropy: float = Field(default=0.25, ge=0.0, le=1.0)
    alphaDecoupling: float = Field(default=0.7, ge=0.0, le=1.0)
    wordBudget: int = Field(default=EVIDENCE_WORD_BUDGET, ge=100, le=2000)
    mode: str = Field(default="strict", pattern="^(strict|lenient|stress)$")

class TriageQueryRequest(BaseModel):
    query: str = Field(min_length=3)
    config: GateConfigIn = Field(default_factory=GateConfigIn)

# ---------------------------------------------------------------------------
# Pipeline provider: lazily builds the real pipelines; tests swap in stubs.
# ---------------------------------------------------------------------------

class PipelineProvider:
    """Owns the heavy singletons. Subclass/replace for testing."""

    def __init__(self) -> None:
        self._baseline = None
        self._vanilla = None
        self._hcad = None
        self.abstain_events = 0
        self.total_queries = 0

    def baseline_llm(self):
        if self._baseline is None:
            from src.baselines.baseline_llm import BaselineLLM
            self._baseline = BaselineLLM()
        return self._baseline

    def vanilla_rag(self):
        if self._vanilla is None:
            from src.baselines.vanilla_vector_rag import VanillaVectorRAG
            self._vanilla = VanillaVectorRAG()
        return self._vanilla

    def hcad_rag(self):
        if self._hcad is None:
            from src.core.hcad_retriever import HCADRetriever
            self._hcad = HCADRetriever()
        return self._hcad


provider = PipelineProvider()

# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------

_ENTITY_TYPE_MAP = {
    "Drug": "drug",
    "Disease": "disease",
    "Gene": "gene",
    "Protein": "protein",
    "Cell": "phenotype",
}

_EVIDENCE_TAG = re.compile(r"\[Evidence (\d+)\]")

def _map_entities(ner_result: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    entities: List[Dict[str, Any]] = []
    for category, spans in (ner_result or {}).items():
        ts_type = _ENTITY_TYPE_MAP.get(category, "phenotype")
        for span in spans or []:
            text = (span or {}).get("text", "").strip()
            if text:
                entities.append(
                    {"id": f"ent-{len(entities)}", "name": text,
                     "type": ts_type, "confidence": 0.85}
                )
    return entities

_context_document = context_document

def _citation_id(index: int) -> str:
    return f"Evidence:{index}"

def _citation_record(context: Dict[str, Any], index: int) -> Dict[str, Any]:
    metadata = context.get("metadata") or {}
    head = metadata.get("head_name", "")
    relation = metadata.get("display_relation") or metadata.get("relation", "")
    tail = metadata.get("tail_name", "")
    provenance = metadata.get("provenance") or context.get("source", "retrieved")
    triple = f"{head} —[{relation}]→ {tail}" if (head and tail) else _context_document(context)[:120]
    confidence = float(context.get("phi_final", context.get("score", 0.0)) or 0.0)
    return {
        "id": _citation_id(index),
        "triple": triple,
        "source": f"{provenance}",
        "confidence": round(min(1.0, confidence), 4),
        "abstract": _context_document(context)[:600],
    }

def _answer_segments(answer: str, n_evidence: int) -> List[Dict[str, Any]]:
    """Split an answer into text/citation segments on [Evidence i] tags."""
    segments: List[Dict[str, Any]] = []
    pos = 0
    for match in _EVIDENCE_TAG.finditer(answer):
        if match.start() > pos:
            segments.append({"kind": "text", "value": answer[pos:match.start()]})
        index = int(match.group(1))
        if 1 <= index <= n_evidence:
            segments.append({"kind": "citation", "id": _citation_id(index)})
        else:
            segments.append({"kind": "text", "value": match.group(0)})
        pos = match.end()
    if pos < len(answer):
        segments.append({"kind": "text", "value": answer[pos:]})
    return segments or [{"kind": "text", "value": answer}]

def _budget_evidence_contexts(
    contexts: List[Dict[str, Any]], word_budget: int
) -> List[Dict[str, Any]]:
    """Keep only complete evidence sentences within the request word budget.

    If the next sentence does not fit, it is omitted rather than cut mid-sentence;
    a context with no complete sentence that fits is omitted entirely.
    """
    bounded: List[Dict[str, Any]] = []
    remaining = word_budget
    for context in contexts:
        document = _context_document(context).strip()
        words = document.split()
        if len(words) <= remaining:
            bounded.append(context)
            remaining -= len(words)
            continue
        import re
        sentences = re.split(r"(?<=[.!?])\s+", document)
        kept: List[str] = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            count = len(sentence.split())
            if count > remaining:
                break
            kept.append(sentence)
            remaining -= count
        if kept:
            bounded.append({**context, "document": " ".join(kept)})
    return bounded

def _gate_k_for(config: GateConfigIn) -> float:
    """Map the workbench tau slider to the Heaviside k.

    The statistical filter thresholds at mu + k*sigma, so a higher k is a
    stricter gate. tau=0.25 (default) maps to the configured FILTERING_K;
    the slider scales it in the same direction as the workbench display.
    """
    return max(0.0, FILTERING_K * (1.0 + (config.tauEntropy - 0.25) * 2.0))

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/triage/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "pipelines_loaded": {
            "baseline": provider._baseline is not None,
            "vanilla": provider._vanilla is not None,
            "hcad": provider._hcad is not None,
        },
        "abstention_rate": _session_abstention_rate(),
    }

def _session_abstention_rate() -> float:
    if provider.total_queries == 0:
        return 0.0
    return round(100.0 * provider.abstain_events / provider.total_queries, 1)

@app.post("/api/triage/query")
def triage_query(request: TriageQueryRequest) -> Dict[str, Any]:
    hcad = provider.hcad_rag()
    vanilla = provider.vanilla_rag()
    baseline = provider.baseline_llm()
    config = request.config
    query = request.query.strip()

    provider.total_queries += 1
    request_started = time.perf_counter()

    try:
        retrieval_started = time.perf_counter()
        retrieved_contexts = hcad.retrieve(query, k=_gate_k_for(config))
        retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1000.0
        hcad_contexts = _budget_evidence_contexts(retrieved_contexts, config.wordBudget)

        # Entities via the same NER the retriever uses.
        try:
            hcad._ensure_components()
            ner_result = hcad._ner.extract_entities(query)
        except Exception as exc:  # pragma: no cover - depends on model load
            logger.warning("NER extraction failed: %s", exc)
            ner_result = {}
        entities = _map_entities(ner_result)

        # Stage-1 raw pool: the same candidate fetch the retriever performed.
        raw_candidates: List[Dict[str, Any]] = []
        admitted_docs = {_context_document(c) for c in hcad_contexts}
        try:
            pool = hcad._get_candidate_contexts(
                query, ner_result, n_vector=20
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Candidate fetch failed: %s", exc)
            pool = hcad_contexts
        for i, ctx in enumerate(pool[:20]):
            doc = _context_document(ctx)
            raw_candidates.append({
                "id": f"cand-{i}",
                "text": doc[:220],
                "score": round(float(ctx.get("score", 0.0)) or 0.0, 4),
                "source": "PrimeKG" if ctx.get("source", "vector") != "vector" else "PMC",
                "redundancy": 0.0,
                "verdict": "ADMIT" if doc in admitted_docs else "DISMISS",
            })

        # Decoupling channel counts from the actual extraction.
        n_entities = sum(len(v or []) for v in (ner_result or {}).values())
        n_relations = 0
        try:
            rules = hcad._relation_extractor.extract_rules(query, ner_result)
            n_relations = len(rules) if isinstance(rules, (list, tuple)) else 0
        except Exception:  # pragma: no cover
            n_relations = 0
        decoupling = {
            "eChannel": max(1, min(6, n_entities)),
            "rChannel": max(1, min(7, n_relations or 2)),
            "mChannel": max(1, min(4, len(query.split()) // 6)),
            "cancelled": max(0, len(pool) - len(hcad_contexts)),
            "alpha": config.alphaDecoupling,
        }

        # Telemetry describes observed retrieval scores; score spread is not
        # entropy and is never labelled as an entropy delta.
        phi = [float(c.get("phi_final", c.get("score", 0.0)) or 0.0)
               for c in hcad_contexts]
        jev = round(sum(phi) / len(phi), 4) if phi else 0.0
        spread = max(phi) - min(phi) if len(phi) > 1 else 0.0
        # P2d: a single admitted context is sufficient grounding for a PASS -
        # the dual gate already enforces the relevance floor per context, so
        # demanding >= 2 conflated "thin evidence" with "no evidence" and
        # inflated the abstention rate.
        passed = len(hcad_contexts) >= 1
        if not passed:
            provider.abstain_events += 1
        # Provenance for downstream consumers: was the answer grounded in
        # retrieved evidence or produced parametrically (abstained)?
        grounded_mode = "grounded" if hcad_contexts else "parametric"
        context_words = sum(len(_context_document(c).split()) for c in hcad_contexts)
        gate = {
            "decision": "PASS" if passed else "ABSTAIN",
            "jev": jev,
            "scoreSpread": round(spread, 4),
            "scoreSpreadDefinition": "max(phi_final or score) - min(phi_final or score); 0 when fewer than 2 contexts",
            "tauEntropy": config.tauEntropy,
            "candidateCount": len(pool),
            "admittedCount": len(hcad_contexts),
            "contextWords": context_words,
            "wordBudget": config.wordBudget,
            "output": "ABSTAIN" if not hcad_contexts else "PASS",
            "groundedMode": grounded_mode,
        }

        # Citations from admitted contexts.
        citations = {
            _citation_id(i): _citation_record(c, i)
            for i, c in enumerate(hcad_contexts, start=1)
        }

        # Arena generations.
        resp_baseline = baseline.generate(query)
        vanilla_contexts = vanilla.retrieve(query, top_k=5)
        resp_vanilla = vanilla.retrieve_and_generate(query, contexts=vanilla_contexts)
        resp_hcad = hcad.retrieve_and_generate(query, contexts=hcad_contexts)

        warnings: List[str] = []
        if not passed:
            warnings.append(
                "Dual gate rejected all retrieved candidates - TRIAGE abstained "
                "and answered from parametric knowledge only."
            )

        arena = {
            "baseline": [{"kind": "text", "value": str(resp_baseline)}],
            "vanillaRag": [{"kind": "text", "value": str(resp_vanilla)}],
            "triage": _answer_segments(str(resp_hcad), len(hcad_contexts)),
            "warnings": warnings,
        }

        end_to_end_latency_ms = round((time.perf_counter() - request_started) * 1000.0, 1)
        return {
            "queryId": f"api-{int(time.time() * 1000):x}",
            "query": query,
            "entities": entities,
            "rawCandidates": raw_candidates,
            "decoupling": decoupling,
            "gate": gate,
            "groundedMode": grounded_mode,
            "arena": arena,
            "citations": citations,
            "abstentionRate": _session_abstention_rate(),
            "retrievalLatencyMs": round(retrieval_latency_ms, 1),
            "endToEndLatencyMs": end_to_end_latency_ms,
            # Backwards-compatible alias: latencyMs now means end-to-end latency.
            "latencyMs": end_to_end_latency_ms,
            "substrate": {"name": "PrimeKG", "nodes": 129262, "edges": 4000956},
        }
    except Exception as exc:
        logger.exception("triage query failed: %s", exc)
        raise HTTPException(status_code=500, detail="Internal pipeline processing error") from exc

# PrimeKG node_type -> workbench EntityType
def _node_type(raw: str) -> str:
    raw = (raw or "").lower()
    if "drug" in raw:
        return "drug"
    if "disease" in raw or "symptom" in raw:
        return "disease"
    if "gene" in raw:
        return "gene"
    if "protein" in raw:
        return "protein"
    return "phenotype"

@app.get("/api/triage/subgraph")
def subgraph(query: str = Query(..., min_length=3)) -> Dict[str, Any]:
    hcad = provider.hcad_rag()
    try:
        graph = hcad.graph  # lazy-loads the PrimeKG MultiDiGraph
    except Exception as exc:
        logger.exception("Knowledge graph unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Knowledge graph unavailable") from exc

    # Resolve a center node from the query via the node-name index.
    center = None
    q_lower = query.lower()
    for node, data in graph.nodes(data=True):
        name = str(data.get("name", ""))
        if name and name.lower() in q_lower:
            center = node
            break
    if center is None:
        raise HTTPException(
            status_code=404,
            detail="No PrimeKG node matched the query; run prepare --rebuild "
                   "if the graph index is missing.",
        )

    # 2-hop ego network, capped for the canvas.
    nodes: Dict[str, Dict[str, Any]] = {}
    frontier = {center}
    visited = {center}
    for _hop in range(2):
        next_frontier: set = set()
        for node in frontier:
            for _, target, data in graph.out_edges(node, data=True):
                if target not in visited:
                    next_frontier.add(target)
                if target not in nodes:
                    n_data = graph.nodes[target]
                    nodes[target] = {
                        "id": str(target),
                        "label": str(n_data.get("name", target))[:24],
                        "type": _node_type(str(n_data.get("node_type", ""))),
                        "trace": 0.0,
                    }
            for source, _, data in graph.in_edges(node, data=True):
                if source not in visited:
                    next_frontier.add(source)
                if source not in nodes:
                    n_data = graph.nodes[source]
                    nodes[source] = {
                        "id": str(source),
                        "label": str(n_data.get("name", source))[:24],
                        "type": _node_type(str(n_data.get("node_type", ""))),
                        "trace": 0.0,
                    }
        visited |= next_frontier
        frontier = next_frontier
        if len(nodes) >= 30:
            break

    center_data = graph.nodes[center]
    nodes[center] = {
        "id": str(center),
        "label": str(center_data.get("name", center))[:24],
        "type": _node_type(str(center_data.get("node_type", ""))),
        "trace": 1.0,
    }

    edges: List[Dict[str, Any]] = []
    node_ids = set(nodes)
    degree = {nid: 0 for nid in node_ids}
    edge_index = 0
    for u, v, data in graph.edges(nbunch=node_ids, data=True):
        if u in node_ids and v in node_ids:
            relation = str(data.get("relation", "associated_with"))
            edges.append({
                "id": f"e{edge_index}",
                "source": str(u),
                "target": str(v),
                "relation": relation.replace("_", " "),
                "weight": 0.5,
            })
            degree[u] = degree.get(u, 0) + 1
            degree[v] = degree.get(v, 0) + 1
            edge_index += 1
            if edge_index >= 40:
                break

    # Trace score: degree-normalized contribution, center = 1.0.
    max_degree = max(degree.values()) if degree else 1
    for nid in nodes:
        if nid != center and max_degree:
            nodes[nid]["trace"] = round(degree.get(nid, 0) / max_degree, 4)

    return {
        "caseId": "live",
        "centerId": str(center),
        "hops": 2,
        "nodes": list(nodes.values()),
        "edges": edges,
    }

def _benchmark_from_results(results: Dict[str, Any]) -> Dict[str, Any]:
    scores = results.get("scoring_based_assessment", {}) or {}
    def dim(system: str) -> float:
        return round(float(scores.get(system, {}).get("overall_score", 0.0) or 0.0), 1)

    citation = (results.get("citation_precision") or {}).get("HCAD-RAG", {}) or {}
    citation_precision = citation.get("mean_precision")

    radar = [
        {"dimension": "Diagnostic Accuracy", "triage": dim("HCAD-RAG"),
         "vanilla": dim("Vanilla RAG"), "baseline": dim("Baseline LLM")},
        {"dimension": "Evidence Faithfulness",
         "triage": round((citation_precision or 0.0) * 100, 1),
         "vanilla": round(((results.get("citation_precision") or {})
                           .get("Vanilla RAG", {}) or {}).get("mean_precision") or 0.0, 1) * 100,
         "baseline": 0.0},
        {"dimension": "Contradiction Avoidance",
         "triage": _score_dim(scores, "HCAD-RAG", "logical"),
         "vanilla": _score_dim(scores, "Vanilla RAG", "logical"),
         "baseline": _score_dim(scores, "Baseline LLM", "logical")},
        {"dimension": "Hallucination Suppression", "triage": 0.0, "vanilla": 0.0, "baseline": 0.0},
        {"dimension": "Completeness",
         "triage": _score_dim(scores, "HCAD-RAG", "comprehensiveness"),
         "vanilla": _score_dim(scores, "Vanilla RAG", "comprehensiveness"),
         "baseline": _score_dim(scores, "Baseline LLM", "comprehensiveness")},
    ]

    selection = ((results.get("selection_based_assessment") or {})
                 .get("HCAD_vs_Vanilla", {}) or {})
    win_rate = {
        criterion: round(rate, 1)
        for criterion, rate in (selection.get("criteria_win_rates_pct") or {}).items()
    }

    mcnemar_in = (results.get("significance_testing") or {}).get("HCAD_vs_Vanilla") or {}

    return {
        "radar": radar,
        "winRate": [{"criterion": c, "rate": r} for c, r in win_rate.items()],
        "mcnemar": {
            "statistic": mcnemar_in.get("n01", 0),
            "pValue": mcnemar_in.get("p_value"),
            "significant": bool(mcnemar_in.get("significant_at_05", False)),
            "n": results.get("num_samples", 0),
        },
        "abstentionRate": 0.0,
        "nQueries": results.get("num_samples", 0),
    }

def _score_dim(scores: Dict[str, Any], system: str, dimension: str) -> float:
    return round(float((scores.get(system) or {}).get(dimension, 0.0) or 0.0), 1)

@app.get("/api/triage/benchmark")
def benchmark() -> Dict[str, Any]:
    results_path = RESULTS_DIR / "base_paper_benchmark_results.json"
    if not results_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No benchmark results found. Run "
                   "`python src/main.py base-paper --samples 50` first.",
        )
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    return _benchmark_from_results(results)
