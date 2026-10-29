"""Controlled retrieval ranking experiments for HCAD-RAG.

Compares dense-only, graph-only, reciprocal-rank fusion (RRF), and the
production HCAD ranking on identical generated development questions. Metrics
are based on the benchmark's existing source-chunk relevance protocol; this
does not replace independently adjudicated clinical relevance labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.config import MIN_PHI_SCORE, RESULTS_DIR
from src.evaluation.metrics import ndcg_at_k, recall_at_k
from src.evaluation.run_base_paper_benchmark import (
    RETRIEVAL_K,
    _ranked_relevance,
    generate_paper_protocol_questions,
)

RRF_K = 60
STRATEGIES = ("dense_only", "graph_only", "rrf", "hcad")


def _text(candidate: dict[str, Any]) -> str:
    return str(candidate.get("document") or candidate.get("text") or "").strip()


def _key(candidate: dict[str, Any]) -> str:
    """Stable dedup key based on normalized evidence text."""
    return " ".join(_text(candidate).lower().split())


def reciprocal_rank_fusion(
    candidates: list[dict[str, Any]], *, rrf_k: int = RRF_K
) -> list[dict[str, Any]]:
    """Fuse source-specific ranks without comparing incompatible raw scores.

    All non-vector sources are grouped as graph/fact retrieval. Each source
    list is sorted by its own score, then contributes 1 / (rrf_k + rank).
    Duplicate text is represented once and retains source provenance.
    """
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    by_source: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        source = str(candidate.get("source") or "unknown")
        channel = "dense" if source == "vector" else "graph"
        by_source.setdefault(channel, []).append(candidate)

    fused: dict[str, dict[str, Any]] = {}
    for channel, rows in by_source.items():
        rows = sorted(rows, key=lambda row: float(row.get("score") or 0.0), reverse=True)
        for rank, candidate in enumerate(rows, start=1):
            key = _key(candidate)
            if not key:
                continue
            contribution = 1.0 / (rrf_k + rank)
            if key not in fused:
                fused[key] = {**candidate, "rrf_score": 0.0, "sources": []}
            fused[key]["rrf_score"] += contribution
            if channel not in fused[key]["sources"]:
                fused[key]["sources"].append(channel)
    return sorted(
        fused.values(),
        key=lambda row: (-float(row["rrf_score"]), _key(row)),
    )


def _rank_variants(candidates: list[dict[str, Any]], final_hcad: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    dense = sorted(
        [c for c in candidates if c.get("source") == "vector"],
        key=lambda row: float(row.get("score") or 0.0),
        reverse=True,
    )
    graph = sorted(
        [c for c in candidates if c.get("source") != "vector"],
        key=lambda row: float(row.get("score") or 0.0),
        reverse=True,
    )
    return {
        "dense_only": dense[:RETRIEVAL_K],
        "graph_only": graph[:RETRIEVAL_K],
        "rrf": reciprocal_rank_fusion(candidates)[:RETRIEVAL_K],
        "hcad": list(final_hcad)[:RETRIEVAL_K],
    }


def _reciprocal_rank(relevance: list[int], k: int = RETRIEVAL_K) -> float:
    for idx, rel in enumerate(relevance[:k], start=1):
        if rel:
            return 1.0 / idx
    return 0.0


def _bootstrap(values: list[float], *, seed: int, iterations: int = 2000) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "ci95": None}
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(iterations)
    )
    return {
        "n": n,
        "mean": float(np.mean(values)),
        "ci95": [
            means[int(0.025 * (iterations - 1))],
            means[int(0.975 * (iterations - 1))],
        ],
        "method": "paired question-level percentile bootstrap",
        "seed": seed,
        "iterations": iterations,
    }


def _commit_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return os.getenv("GITHUB_SHA")


def _check_test_gate(pool: str, output_dir: Path) -> dict[str, Any] | None:
    if pool != "test":
        return None
    gate_path = os.getenv("HCAD_DEV_GO_ARTIFACT", "").strip()
    if not gate_path:
        raise RuntimeError(
            "Held-out test evaluation is locked. Set HCAD_DEV_GO_ARTIFACT to a fresh "
            "development retrieval JSON with decision=GO after the dev gate passes."
        )
    path = Path(gate_path)
    if not path.is_file():
        raise RuntimeError(f"Development GO artifact not found: {path}")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if (artifact.get("go_stop") or {}).get("decision") != "GO":
        raise RuntimeError("Held-out test evaluation is locked because dev decision is not GO.")
    current_commit = _commit_sha()
    dev_commit = artifact.get("git_commit")
    if not dev_commit or not current_commit or dev_commit != current_commit:
        raise RuntimeError(
            "Development GO artifact must record the same git_commit as this run; "
            "rerun development evaluation on this exact commit."
        )
    return {"path": str(path), "git_commit": dev_commit, "decision": "GO"}


def run_ranking_experiments(
    num_samples: int = 60,
    pool: str = "dev",
    output_path: str | Path | None = None,
    *,
    seed: int = 20261029,
) -> dict[str, Any]:
    """Run paired ranking variants and candidate-survival diagnostics."""
    if pool not in {"dev", "test"}:
        raise ValueError("pool must be 'dev' or 'test'")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    gate = _check_test_gate(pool, Path(output_path).parent if output_path else RESULTS_DIR)
    samples = generate_paper_protocol_questions(num_samples, pool=pool)
    if not samples:
        raise RuntimeError(f"No {pool}-pool questions available.")

    from src.core.hcad_retriever import HCADRetriever

    retriever = HCADRetriever(fail_on_vector_error=True)
    retriever._ensure_components()
    if getattr(getattr(retriever, "_ner", None), "degraded", False):
        raise RuntimeError("Biomedical NER is degraded; refusing to evaluate retrieval.")

    per_question: list[dict[str, Any]] = []
    diagnostics = {
        "n_queries": 0,
        "gold_in_initial_pool": 0,
        "gold_survives_phi_floor": 0,
        "gold_in_hcad_final_top5": 0,
        "source_counts": {},
        "candidate_count": [],
        "phi_rejected_count": [],
    }
    metric_values: dict[str, dict[str, list[float]]] = {
        name: {"recall_at_5": [], "ndcg_at_5": [], "mrr_at_5": []}
        for name in STRATEGIES
    }
    started_all = time.perf_counter()

    for idx, sample in enumerate(samples):
        query = str(sample["question"])
        reference = str(sample.get("reference", ""))
        started = time.perf_counter()
        hcad_final = retriever.retrieve(query)
        latency_ms = (time.perf_counter() - started) * 1000
        candidates = list(getattr(retriever, "_last_candidate_pool", None) or [])
        phi = np.asarray(getattr(retriever, "_last_phi_scores", [])).ravel()
        if candidates and len(phi) != len(candidates):
            raise RuntimeError("Candidate/score alignment mismatch; aborting evaluation.")

        initial_relevance = _ranked_relevance(reference, candidates)
        diagnostics["n_queries"] += 1
        diagnostics["gold_in_initial_pool"] += int(any(initial_relevance))
        diagnostics["candidate_count"].append(len(candidates))
        source_counts: dict[str, int] = {}
        for candidate in candidates:
            source = str(candidate.get("source") or "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
        for source, count in source_counts.items():
            diagnostics["source_counts"][source] = diagnostics["source_counts"].get(source, 0) + count

        gold_indices = [i for i, rel in enumerate(initial_relevance) if rel]
        phi_survivors = [
            i for i in gold_indices
            if i < len(phi) and float(phi[i]) >= MIN_PHI_SCORE
        ]
        diagnostics["gold_survives_phi_floor"] += int(bool(phi_survivors))
        diagnostics["phi_rejected_count"].append(
            sum(1 for score in phi if float(score) < MIN_PHI_SCORE)
        )

        variants = _rank_variants(candidates, hcad_final)
        row: dict[str, Any] = {
            "question_id": sample.get("question_id", f"{pool}-{idx + 1}"),
            "question": query,
            "pool": pool,
            "latency_hcad_end_to_end_ms": round(latency_ms, 3),
            "candidate_count": len(candidates),
            "source_counts": source_counts,
            "gold_in_initial_pool": bool(any(initial_relevance)),
            "gold_survives_phi_floor": bool(phi_survivors),
            "strategies": {},
        }
        for strategy, contexts in variants.items():
            relevance = _ranked_relevance(reference, contexts)
            recall = recall_at_k(relevance, k=RETRIEVAL_K)
            ndcg = ndcg_at_k(relevance, k=RETRIEVAL_K)
            mrr = _reciprocal_rank(relevance)
            metric_values[strategy]["recall_at_5"].append(float(recall))
            metric_values[strategy]["ndcg_at_5"].append(float(ndcg))
            metric_values[strategy]["mrr_at_5"].append(float(mrr))
            row["strategies"][strategy] = {
                "recall_at_5": float(recall),
                "ndcg_at_5": float(ndcg),
                "mrr_at_5": float(mrr),
                "top5_sources": [str(c.get("source") or "unknown") for c in contexts],
                "top5_text": [_text(c) for c in contexts],
            }
        diagnostics["gold_in_hcad_final_top5"] += int(
            any(_ranked_relevance(reference, hcad_final[:RETRIEVAL_K]))
        )
        per_question.append(row)

    summaries: dict[str, Any] = {}
    for strategy, metrics in metric_values.items():
        summaries[strategy] = {
            metric: _bootstrap(values, seed=seed + metric_idx)
            for metric_idx, (metric, values) in enumerate(metrics.items())
        }
    deltas: dict[str, Any] = {}
    for challenger in ("graph_only", "rrf", "hcad"):
        deltas[challenger] = {}
        for metric in ("recall_at_5", "ndcg_at_5", "mrr_at_5"):
            paired = [
                a - b for a, b in zip(
                    metric_values[challenger][metric],
                    metric_values["dense_only"][metric],
                )
            ]
            deltas[challenger][metric] = _bootstrap(paired, seed=seed + 100 + len(deltas[challenger]))
    n = len(samples)
    result = {
        "schema_version": 1,
        "experiment": "paired_dense_graph_rrf_hcad_retrieval",
        "pool": pool,
        "num_samples": n,
        "k": RETRIEVAL_K,
        "git_commit": _commit_sha(),
        "seed": seed,
        "rrf_k": RRF_K,
        "heldout_gate": gate,
        "runtime_seconds": round(time.perf_counter() - started_all, 3),
        "metrics": summaries,
        "paired_delta_vs_dense": deltas,
        "candidate_survival": {
            **diagnostics,
            "initial_pool_gold_coverage": diagnostics["gold_in_initial_pool"] / n,
            "phi_floor_gold_survival": diagnostics["gold_survives_phi_floor"] / n,
            "hcad_final_top5_gold_coverage": diagnostics["gold_in_hcad_final_top5"] / n,
            "mean_candidate_count": float(np.mean(diagnostics["candidate_count"])),
            "mean_phi_rejected_count": float(np.mean(diagnostics["phi_rejected_count"])),
            "interpretation": (
                "Low initial-pool gold coverage points to candidate generation/entity linking; "
                "high initial coverage but low phi-floor/final-top5 coverage points to filtering/ranking."
            ),
        },
        "per_question": per_question,
        "provenance": {
            "relevance_protocol": "existing source-chunk Jaccard overlap threshold 0.30",
            "candidate_pool_cap": 60,
            "phi_floor": MIN_PHI_SCORE,
            "limitations": [
                "RRF is a controlled offline ranking comparison over the same candidate pool, not a production default.",
                "The relevance labels are generated from source chunks and may favor lexical overlap.",
                "Clinical factuality, negation, abstention, and citation entailment require independent labels and are not inferred by this retrieval script.",
            ],
        },
    }
    target = Path(output_path) if output_path else RESULTS_DIR / f"ranking_experiments_{pool}_n{n}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--pool", choices=("dev", "test"), default="dev")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args()
    result = run_ranking_experiments(
        args.samples, args.pool, args.output, seed=args.seed
    )
    print(json.dumps({
        "pool": result["pool"],
        "num_samples": result["num_samples"],
        "git_commit": result["git_commit"],
        "metrics": result["metrics"],
        "paired_delta_vs_dense": result["paired_delta_vs_dense"],
        "candidate_survival": result["candidate_survival"],
        "heldout_gate": result["heldout_gate"],
    }, indent=2))


if __name__ == "__main__":
    main()
