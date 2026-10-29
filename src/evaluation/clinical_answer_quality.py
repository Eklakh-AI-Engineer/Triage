"""Independent-label evaluation for clinical answer quality.

Input is JSONL with one row per question/system and human/curator annotations.
The module intentionally refuses to infer clinical correctness, citation support,
negation handling, or abstention labels from the answer text itself.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

FACTUALITY = {"correct", "partially_correct", "incorrect"}
CITATION_SUPPORT = {"supported", "partially_supported", "unsupported", "not_applicable"}
NEGATION = {"correct", "incorrect", "not_applicable"}


def _required_label(row: dict[str, Any], field: str, allowed: set[str]) -> str:
    value = str(row.get(field, "")).strip().lower()
    if value not in allowed:
        raise ValueError(f"Missing/invalid independent label {field}={value!r}; allowed={sorted(allowed)}")
    return value


def _rate(values: list[float], *, seed: int = 20261029, iterations: int = 2000) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "ci95": None}
    rng, n = random.Random(seed), len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(iterations)
    )
    return {
        "n": n,
        "mean": sum(values) / n,
        "ci95": [means[int(0.025 * (iterations - 1))], means[int(0.975 * (iterations - 1))]],
        "ci_method": "percentile bootstrap over question-level annotations",
        "seed": seed,
        "iterations": iterations,
    }


def evaluate_clinical_answers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate independently annotated clinical outputs by system.

    Required row fields: question_id, system, answer, factuality_label,
    citation_support_label, negation_label, abstention_expected, abstained.
    Optional telemetry: latency_ms, input_tokens, output_tokens, cost_usd.
    """
    if not rows:
        raise ValueError("At least one clinical answer annotation row is required")
    seen: set[tuple[str, str]] = set()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, raw in enumerate(rows, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Row {index} must be an object")
        row = dict(raw)
        qid, system = str(row.get("question_id", "")).strip(), str(row.get("system", "")).strip()
        if not qid or not system or not isinstance(row.get("answer"), str):
            raise ValueError(f"Row {index} requires question_id, system and string answer")
        pair = (qid, system)
        if pair in seen:
            raise ValueError(f"Duplicate question_id/system pair: {pair}")
        seen.add(pair)
        row["factuality_label"] = _required_label(row, "factuality_label", FACTUALITY)
        row["citation_support_label"] = _required_label(row, "citation_support_label", CITATION_SUPPORT)
        row["negation_label"] = _required_label(row, "negation_label", NEGATION)
        if not isinstance(row.get("abstention_expected"), bool) or not isinstance(row.get("abstained"), bool):
            raise ValueError(f"Row {index} requires boolean abstention_expected and abstained")
        grouped[system].append(row)

    # Multi-system quality comparisons must use the same question IDs.
    if len(grouped) > 1:
        question_sets = {name: {str(r["question_id"]) for r in items} for name, items in grouped.items()}
        reference_name = next(iter(question_sets))
        reference_ids = question_sets[reference_name]
        mismatched = {name: sorted(reference_ids.symmetric_difference(ids))
                      for name, ids in question_sets.items() if ids != reference_ids}
        if mismatched:
            raise ValueError(f"Clinical systems must be evaluated on identical question IDs: {mismatched}")

    summaries: dict[str, Any] = {}
    for system, items in sorted(grouped.items()):
        n = len(items)
        factuality = [1.0 if r["factuality_label"] == "correct" else
                      0.5 if r["factuality_label"] == "partially_correct" else 0.0 for r in items]
        citation_items = [r for r in items if r["citation_support_label"] != "not_applicable"]
        citation_scores = [1.0 if r["citation_support_label"] == "supported" else
                           0.5 if r["citation_support_label"] == "partially_supported" else 0.0
                           for r in citation_items]
        neg_items = [r for r in items if r["negation_label"] != "not_applicable"]
        neg_scores = [float(r["negation_label"] == "correct") for r in neg_items]
        abstention_correct = [float(r["abstained"] == r["abstention_expected"]) for r in items]
        latencies = [float(r["latency_ms"]) for r in items if r.get("latency_ms") is not None]
        costs = [float(r["cost_usd"]) for r in items if r.get("cost_usd") is not None]
        input_tokens = [int(r["input_tokens"]) for r in items if r.get("input_tokens") is not None]
        output_tokens = [int(r["output_tokens"]) for r in items if r.get("output_tokens") is not None]
        if any(not math.isfinite(v) or v < 0 for v in latencies + costs):
            raise ValueError(f"Negative or non-finite latency/cost for system {system}")
        summaries[system] = {
            "n_questions": n,
            "factual_correctness": _rate(factuality),
            "fully_correct_rate": _rate([float(r["factuality_label"] == "correct") for r in items]),
            "citation_support": _rate(citation_scores),
            "citation_support_n": len(citation_items),
            "negation_handling_accuracy": _rate(neg_scores),
            "negation_cases_n": len(neg_items),
            "abstention_decision_accuracy": _rate(abstention_correct),
            "expected_abstention_rate": _rate([float(r["abstention_expected"]) for r in items]),
            "observed_abstention_rate": _rate([float(r["abstained"]) for r in items]),
            "latency_ms": _rate(latencies),
            "input_tokens": {"n": len(input_tokens), "total": sum(input_tokens) if input_tokens else None},
            "output_tokens": {"n": len(output_tokens), "total": sum(output_tokens) if output_tokens else None},
            "cost_usd": {"n": len(costs), "total": sum(costs) if costs else None,
                         "mean": sum(costs) / len(costs) if costs else None},
            "cost_note": "Cost is reported only when captured by provider telemetry; no price is inferred.",
        }
    return {
        "schema_version": 1,
        "label_source": "independent_human_or_curator_annotations",
        "systems": summaries,
        "n_rows": len(rows),
        "limitations": [
            "Labels must be assigned independently of the system answer and its self-reported confidence.",
            "Citation support labels measure semantic support, not merely citation-ID validity.",
            "Only questions with applicable negation/citation labels contribute to those denominators.",
            "Latency/cost telemetry is observational and should use the same hardware/provider conditions for comparisons.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, help="JSONL with answer outputs and independent labels")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.annotations).read_text(encoding="utf-8").splitlines() if line.strip()]
    result = evaluate_clinical_answers(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
