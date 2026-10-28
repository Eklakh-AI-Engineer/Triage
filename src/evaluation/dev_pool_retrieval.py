"""Deliverable-B development-pool retrieval harness (Phase 3).

Measures the four coupled retrieval changes on the frozen development pool
with retrieval-only metrics (no judge, no generation):

  1. symmetric two-pass orthonormal MGS decoupling (always on - it is a
     correctness fix, not a toggle),
  2. query-conditioned channel-presence masking (HCAD_PRESENCE_MASKING),
  3. PrimeKG relation-vocabulary mapping (HCAD_RELATION_VOCAB),
  4. Gate-2 saturation cap tau_eff = min(tau_sat, max(Phi_min, mu + k*sigma))
     with tau_sat calibrated here as the 25th percentile of gold
     development candidate scores (HCAD_TAU_SAT).

GO/STOP (from the epistemic review): the full configuration must improve
paired mean Recall@5 by >= +0.08 and NDCG@5 by >= +0.06 over the unmodified
baseline on the development split. Citation precision (>= 0.95) is measured
by the subsequent test-pool confirmation run, not by this harness.

All calibration happens on the development pool only; the test pool stays
untouched until the GO decision.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np

from src.config import RESULTS_DIR, TAU_SAT_PERCENTILE
from src.evaluation.metrics import ndcg_at_k, recall_at_k
from src.evaluation.run_base_paper_benchmark import (
    RETRIEVAL_K,
    _ranked_relevance,
    generate_paper_protocol_questions,
)

logger = logging.getLogger(__name__)

# Pre-registered minimum acceptance criteria. Environment overrides may
# make the gate stricter, but can never silently relax the release criteria.
_MIN_GO_DELTA_RECALL = 0.08
_MIN_GO_DELTA_NDCG = 0.06


def _gate_target(name: str, minimum: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return minimum
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < minimum:
        raise ValueError(
            f"{name}={value} weakens the pre-registered minimum {minimum}; "
            "gate targets may only be made stricter."
        )
    return value


GO_DELTA_RECALL = _gate_target("HCAD_GO_DELTA_RECALL", _MIN_GO_DELTA_RECALL)
GO_DELTA_NDCG = _gate_target("HCAD_GO_DELTA_NDCG", _MIN_GO_DELTA_NDCG)

# (name, env overrides). The baseline disables masking and vocab mapping and
# pins the desc expansion rule so the paired comparison isolates exactly
# the Deliverable-B changes against a pure Phase 3 baseline.
CONFIGS = [
    ("baseline", {"HCAD_PRESENCE_MASKING": "0", "HCAD_RELATION_VOCAB": "0", "HCAD_EXPANSION_RULE": "desc", "HCAD_TAU_SAT": None}),
    ("masking", {"HCAD_PRESENCE_MASKING": "1", "HCAD_RELATION_VOCAB": "0", "HCAD_TAU_SAT": None}),
    ("masking_vocab", {"HCAD_PRESENCE_MASKING": "1", "HCAD_RELATION_VOCAB": "1", "HCAD_TAU_SAT": None}),
]


@contextlib.contextmanager
def _env_overrides(overrides: Dict[str, Optional[str]]) -> Iterator[None]:
    """Temporarily set/unset environment variables."""
    saved: Dict[str, Optional[str]] = {}
    try:
        for key, value in overrides.items():
            saved[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def meets_go_gate(delta_recall: float, delta_ndcg: float) -> bool:
    """Return whether both pre-registered development improvements are met."""
    return delta_recall >= GO_DELTA_RECALL and delta_ndcg >= GO_DELTA_NDCG


def _hub_anchor_fraction(hcad_rag, contexts: list, degree_cap: int) -> bool:
    """True when any retrieved context is anchored on a >cap-degree node.

    Retention proxy for canonical therapeutic targets (NR3C1, PTGS2, major
    diseases): a deployment that loses these hubs is clinically blind even
    if retrieval metrics look fine on low-degree gold triples.
    """
    graph_wrapper = hcad_rag.graph
    graph_nx = getattr(graph_wrapper, "graph", graph_wrapper)
    name_to_node = getattr(graph_wrapper, "name_to_node", {}) or {}
    if graph_nx is None or not hasattr(graph_nx, "degree"):
        return False
    for ctx in contexts:
        meta = ctx.get("metadata") or {}
        for key in ("head_name", "tail_name"):
            name = meta.get(key)
            if not name:
                continue
            node_id = name_to_node.get(name, name)
            try:
                if node_id in graph_nx and graph_nx.degree(node_id) > degree_cap:
                    return True
            except Exception:
                continue
    return False


def run_hub_ablation(
    num_samples: int = 60,
    pool: str = "dev",
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Phase 4a: neighbor-expansion-rule ablation on the development pool.

    Compares the three expansion rules with all downstream decoupled
    scoring held fixed (masking/vocab at current defaults, tau_sat unset):

      desc    - legacy descending-degree truncation (hub-favoring)
      soft    - soft inverse-log-degree weighting, NO hard drop (deployed)
      asc_cap - ascending degree with a hard deg<=cap drop (ablation-only)

    Reports paired mean Recall@5/NDCG@5 plus the canonical-hub retention
    rate (fraction of questions retrieving at least one >cap-degree
    anchor) - the hard-cap arm is expected to lose canonical receptors
    and major comorbidities.
    """
    from src.config import hub_degree_cap

    logger.info(
        "Phase-4a hub ablation (n=%d, pool=%s, cap=%d)",
        num_samples, pool, hub_degree_cap(),
    )
    samples = generate_paper_protocol_questions(num_samples, pool=pool)
    if not samples:
        raise RuntimeError(f"No {pool}-pool questions available for the hub ablation.")

    from src.core.hcad_retriever import HCADRetriever

    hcad_rag = HCADRetriever(fail_on_vector_error=True)
    hcad_rag._ensure_components()
    if getattr(getattr(hcad_rag, "_ner", None), "degraded", False):
        raise RuntimeError(
            "SciSpacy NER failed to load (BiomedicalNER in degraded regex "
            "mode). Refusing to run dev-pool retrieval evaluation on degraded features."
        )

    results: Dict[str, Any] = {
        "pool": pool,
        "num_samples": len(samples),
        "k": RETRIEVAL_K,
        "degree_cap": hub_degree_cap(),
    }
    cap = hub_degree_cap()

    for rule in ("desc", "soft", "asc_cap"):
        logger.info("Evaluating expansion rule: %s", rule)
        ndcgs, recalls, hub_flags = [], [], []
        with _env_overrides({"HCAD_EXPANSION_RULE": rule, "HCAD_TAU_SAT": None}):
            for sample in samples:
                contexts = hcad_rag.retrieve(sample["question"])
                relevance = _ranked_relevance(sample.get("reference", ""), contexts)
                ndcgs.append(ndcg_at_k(relevance, k=RETRIEVAL_K))
                recalls.append(recall_at_k(relevance, k=RETRIEVAL_K))
                hub_flags.append(_hub_anchor_fraction(hcad_rag, contexts, cap))

        n = len(samples)
        results[rule] = {
            "mean_ndcg_at_k": round(float(np.mean(ndcgs)), 4) if n else 0.0,
            "mean_recall_at_k": round(float(np.mean(recalls)), 4) if n else 0.0,
            "canonical_hub_retention_rate": round(float(np.mean(hub_flags)), 4) if n else 0.0,
        }
        logger.info(
            "  %-8s Recall@5=%.4f NDCG@5=%.4f hub-retention=%.4f",
            rule, results[rule]["mean_recall_at_k"],
            results[rule]["mean_ndcg_at_k"],
            results[rule]["canonical_hub_retention_rate"],
        )

    results["deployed_rule"] = "soft"
    results["note"] = (
        "desc is the legacy hub-favoring baseline; asc_cap keeps only "
        "deg<=cap nodes and is expected to lose canonical high-degree "
        "targets (NR3C1, major diseases). soft is the deployed rule: "
        "specific neighbors win the truncation slice, hubs stay reachable."
    )

    if output_path is None:
        output_path = RESULTS_DIR / f"hub_ablation_{pool}_n{len(samples)}.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Hub ablation results saved to %s", output_path)

    return results


def _evaluate_config(
    hcad_rag, samples: list[dict], env: Dict[str, Optional[str]]
) -> Dict[str, Any]:
    """Run HCAD retrieval over the pool under one configuration."""
    ndcgs, recalls, gold_scores = [], [], []
    candidate_pool_sizes = []
    candidate_gold_hits = []
    phi_top5_gold_hits = []
    raw_score_top5_gold_hits = []
    candidate_source_counts: Dict[str, int] = {}
    n_query_missing_rel = n_query_missing_mod = 0
    n_cand_missing_rel = n_cand_missing_mod = n_candidates = 0

    with _env_overrides(env):
        for sample in samples:
            query = sample["question"]
            reference = sample.get("reference", "")
            contexts = hcad_rag.retrieve(query)
            relevance = _ranked_relevance(reference, contexts)
            ndcgs.append(ndcg_at_k(relevance, k=RETRIEVAL_K))
            recalls.append(recall_at_k(relevance, k=RETRIEVAL_K))

            # Pre-gate provenance: gold candidate score for tau_sat
            # calibration, plus channel-presence diagnostics.
            pool = getattr(hcad_rag, "_last_candidate_pool", None) or []
            phi = getattr(hcad_rag, "_last_phi_scores", None)
            p_q = getattr(hcad_rag, "_last_query_presence", None)
            p_c = getattr(hcad_rag, "_last_candidate_presence", None)

            if pool:
                candidate_pool_sizes.append(len(pool))
                for ctx in pool:
                    source = str(ctx.get("source", "unknown"))
                    candidate_source_counts[source] = candidate_source_counts.get(source, 0) + 1
                pool_rel = _ranked_relevance(reference, pool)
                gold_idx = [i for i, r in enumerate(pool_rel) if r > 0]
                candidate_gold_hits.append(bool(gold_idx))
                if phi is not None and len(phi) == len(pool):
                    if gold_idx:
                        gold_scores.append(float(np.max([phi[i] for i in gold_idx])))
                    phi_order = np.argsort(np.asarray(phi, dtype=float))[::-1][:RETRIEVAL_K]
                    phi_top5_gold_hits.append(any(i in gold_idx for i in phi_order))
                # Raw candidate score is the retrieval-stage score (e.g. vector
                # cosine for vector candidates); this diagnostic is deliberately
                # separate from the decoupled Phi ranking used downstream.
                raw_scores = np.asarray([
                    float(ctx.get("score", float("-inf"))) for ctx in pool
                ], dtype=float)
                raw_order = np.argsort(raw_scores)[::-1][:RETRIEVAL_K]
                raw_score_top5_gold_hits.append(any(i in gold_idx for i in raw_order))

            if p_q is not None:
                p_q = np.asarray(p_q).ravel()
                if len(p_q) == 3:
                    n_query_missing_rel += int(not p_q[1])
                    n_query_missing_mod += int(not p_q[2])
            if p_c is not None and len(p_c):
                p_c = np.asarray(p_c).reshape(-1, 3)
                n_candidates += len(p_c)
                n_cand_missing_rel += int((~p_c[:, 1]).sum())
                n_cand_missing_mod += int((~p_c[:, 2]).sum())

    n = len(samples)
    return {
        "num_samples": n,
        "mean_ndcg_at_k": round(float(np.mean(ndcgs)), 4) if n else 0.0,
        "mean_recall_at_k": round(float(np.mean(recalls)), 4) if n else 0.0,
        "candidate_survival_diagnostics": {
            "mean_candidate_pool_size": round(float(np.mean(candidate_pool_sizes)), 2)
            if candidate_pool_sizes else 0.0,
            "gold_present_in_candidate_pool_frac": round(float(np.mean(candidate_gold_hits)), 4)
            if candidate_gold_hits else None,
            "gold_in_phi_top_k_frac": round(float(np.mean(phi_top5_gold_hits)), 4)
            if phi_top5_gold_hits else None,
            "gold_in_raw_score_top_k_frac": round(float(np.mean(raw_score_top5_gold_hits)), 4)
            if raw_score_top5_gold_hits else None,
            "candidate_source_counts": dict(sorted(candidate_source_counts.items())),
        },
        "gold_score_percentiles": {
            "p25": round(float(np.percentile(gold_scores, TAU_SAT_PERCENTILE)), 4)
            if gold_scores else None,
            "n_gold_in_pool": len(gold_scores),
        },
        "presence_diagnostics": {
            "query_missing_relation_frac": round(n_query_missing_rel / n, 4) if n else None,
            "query_missing_modifier_frac": round(n_query_missing_mod / n, 4) if n else None,
            "candidate_missing_relation_frac": round(n_cand_missing_rel / n_candidates, 4)
            if n_candidates else None,
            "candidate_missing_modifier_frac": round(n_cand_missing_mod / n_candidates, 4)
            if n_candidates else None,
        },
    }


def run_retrieval_eval(
    num_samples: int = 60,
    pool: str = "dev",
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Paired before/after retrieval evaluation on the development pool."""

    if pool != "dev":
        raise ValueError(
            "The retrieval calibration harness is development-only. "
            "The held-out test pool must remain untouched until the frozen "
            "configuration passes the documented GO gate."
        )
    logger.info(
        "Deliverable-B dev-pool retrieval eval (n=%d, pool=%s, k=%d)",
        num_samples, pool, RETRIEVAL_K,
    )

    samples = generate_paper_protocol_questions(num_samples, pool=pool)
    if not samples:
        raise RuntimeError(f"No {pool}-pool questions available for the retrieval eval.")

    from src.core.hcad_retriever import HCADRetriever

    hcad_rag = HCADRetriever(fail_on_vector_error=True)
    hcad_rag._ensure_components()
    if getattr(getattr(hcad_rag, "_ner", None), "degraded", False):
        raise RuntimeError(
            "SciSpacy NER failed to load (BiomedicalNER in degraded regex "
            "mode). Refusing to run dev-pool retrieval evaluation on degraded features."
        )

    results: Dict[str, Any] = {"pool": pool, "num_samples": len(samples), "k": RETRIEVAL_K}

    # 1-3: baseline / masking / masking+vocab (paired, same questions).
    for name, env in CONFIGS:
        logger.info("Evaluating configuration: %s", name)
        results[name] = _evaluate_config(hcad_rag, samples, env)

    # Diagnostic recommended in the review: isolate the saturation cap
    # while preserving legacy retrieval defaults (masking/vocab OFF).
    # Calibrate this cap from the legacy baseline's own gold dev scores.
    baseline_tau = results["baseline"]["gold_score_percentiles"]["p25"]
    if baseline_tau is None:
        results["baseline_with_tau_cap"] = None
        logger.warning("No baseline gold candidate scores; baseline cap ablation unavailable.")
    else:
        results["baseline_tau_sat"] = round(float(baseline_tau), 4)
        results["baseline_with_tau_cap"] = _evaluate_config(
            hcad_rag,
            samples,
            {
                "HCAD_PRESENCE_MASKING": "0",
                "HCAD_RELATION_VOCAB": "0",
                "HCAD_EXPANSION_RULE": "desc",
                "HCAD_TAU_SAT": str(baseline_tau),
            },
        )

    # Full configuration with the Gate-2 saturation cap. tau_sat is
    # calibrated from the gold candidate scores of the masking+vocab run -
    # calibration and measurement both stay on the dev pool.
    tau_sat_value = results["masking_vocab"]["gold_score_percentiles"]["p25"]
    if tau_sat_value is None:
        logger.warning(
            "No gold candidate found in any pool - Gate-2 cap disabled for "
            "the full configuration."
        )
        results["full"] = None
    else:
        logger.info("Calibrated tau_sat (p25 of gold dev candidate scores): %.4f", tau_sat_value)
        results["tau_sat"] = round(float(tau_sat_value), 4)
        results["full"] = _evaluate_config(
            hcad_rag,
            samples,
            {
                "HCAD_PRESENCE_MASKING": "1",
                "HCAD_RELATION_VOCAB": "1",
                "HCAD_TAU_SAT": str(tau_sat_value),
            },
        )

    # GO/STOP verdict.
    if results.get("full"):
        d_recall = results["full"]["mean_recall_at_k"] - results["baseline"]["mean_recall_at_k"]
        d_ndcg = results["full"]["mean_ndcg_at_k"] - results["baseline"]["mean_ndcg_at_k"]
        results["delta_vs_baseline"] = {
            "recall_at_k": round(d_recall, 4),
            "ndcg_at_k": round(d_ndcg, 4),
        }
        is_go = meets_go_gate(d_recall, d_ndcg)
        results["go_stop"] = {
            "required_delta_recall": GO_DELTA_RECALL,
            "required_delta_ndcg": GO_DELTA_NDCG,
            "decision": "GO" if is_go else "STOP",
            "note": (
                "Citation precision (>= 0.95) is validated by the test-pool "
                "confirmation run; this harness measures retrieval only."
            ),
        }
    else:
        results["go_stop"] = {"decision": "STOP", "reason": "tau_sat calibration failed"}

    if output_path is None:
        output_path = RESULTS_DIR / f"retrieval_eval_{pool}_n{len(samples)}.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Retrieval eval results saved to %s", output_path)

    return results
