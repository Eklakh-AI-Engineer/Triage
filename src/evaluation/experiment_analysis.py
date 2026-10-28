"""Paired bootstrap analysis helpers for ablation result bundles."""
from __future__ import annotations
import json, random
from pathlib import Path
from typing import Any

def paired_bootstrap_delta(baseline: list[float], treatment: list[float], *,
                           seed: int = 20261028, iterations: int = 5000) -> dict[str, Any]:
    """Return treatment-minus-baseline paired mean delta and percentile 95% CI."""
    if len(baseline) != len(treatment):
        raise ValueError("Paired arrays must have equal length and identical question order")
    if not baseline:
        return {"n": 0, "mean_delta": None, "ci95": None}
    deltas = [float(b) - float(a) for a, b in zip(baseline, treatment)]
    rng, n = random.Random(seed), len(deltas)
    samples = sorted(sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(iterations))
    return {"n": n, "mean_delta": sum(deltas) / n,
        "ci95": [samples[int(.025 * (iterations-1))], samples[int(.975 * (iterations-1))]],
        "method": "paired percentile bootstrap over fixed question IDs",
        "seed": seed, "iterations": iterations}

def compare_ablation_bundles(baseline_path: str | Path, ablation_paths: list[str | Path], *,
                             metric: str = "end_to_end_latency_ms") -> dict[str, Any]:
    """Compare aligned bundles; metric values must be independently computed per question."""
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    base_rows = baseline.get("per_question", [])
    base_ids = [row["question_id"] for row in base_rows]
    if not base_ids:
        raise ValueError("Baseline bundle contains no per-question outputs")
    comparisons = []
    for path in ablation_paths:
        treatment = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = treatment.get("per_question", [])
        if [row["question_id"] for row in rows] != base_ids:
            raise ValueError(f"Question IDs/order differ in ablation bundle {path}")
        try:
            a = [float(row["metrics"][metric]) for row in base_rows]
            b = [float(row["metrics"][metric]) for row in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Both bundles must contain per-question metric {metric!r}") from exc
        comparisons.append({"ablation_bundle": str(path), "metric": metric,
                            "paired_delta": paired_bootstrap_delta(a, b)})
    return {"metric": metric, "baseline_bundle": str(baseline_path),
            "comparisons": comparisons,
            "interpretation": "Delta is ablation minus baseline; interpret sign according to metric direction."}
