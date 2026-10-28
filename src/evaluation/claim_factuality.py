"""Claim-level factuality evaluation for generated clinical answers.

This is deliberately separate from the MedHallu detector benchmark. Metrics
are only computed when each claim has an independent human/curator label.
Labels are 'supported', 'contradicted', or 'insufficient_evidence'; the binary
unsupported metric treats the latter two as unsupported, per the documented
annotation protocol.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

ALLOWED_LABELS = {"supported", "contradicted", "insufficient_evidence"}


def validate_claim_annotations(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate independent claim annotations; never synthesize labels from model output."""
    validated = []
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Claim annotation row {row_number} must be an object")
        claim = str(row.get("claim", "")).strip()
        label = str(row.get("label", "")).strip().lower()
        if not claim:
            raise ValueError(f"Claim annotation row {row_number} is missing claim text")
        if label not in ALLOWED_LABELS:
            raise ValueError(
                f"Claim annotation row {row_number} has invalid independent label {label!r}"
            )
        if not str(row.get("question_id", "")).strip():
            raise ValueError(f"Claim annotation row {row_number} is missing question_id")
        if not str(row.get("annotator_id", "")).strip():
            raise ValueError(f"Claim annotation row {row_number} is missing annotator_id")
        validated.append({**row, "claim": claim, "label": label})
    if not validated:
        raise ValueError("At least one independently labelled claim is required")
    return validated


def score_claim_factuality(
    annotations: Iterable[dict[str, Any]],
    predictions: dict[str, bool],
) -> dict[str, Any]:
    """Compute confusion-matrix metrics from independent labels and claim predictions.

    predictions maps stable claim_id to True when the system flags the claim as
    unsupported. The annotation row must contain the same claim_id. Missing
    predictions are excluded and explicitly reported, never counted as correct.
    """
    rows = validate_claim_annotations(annotations)
    counts = Counter(tp=0, fp=0, tn=0, fn=0)
    evaluated = 0
    missing_predictions = []
    for row in rows:
        claim_id = str(row.get("claim_id", "")).strip()
        if not claim_id:
            raise ValueError("Every claim annotation row must have claim_id")
        if claim_id not in predictions:
            missing_predictions.append(claim_id)
            continue
        actual_unsupported = row["label"] in {"contradicted", "insufficient_evidence"}
        predicted_unsupported = bool(predictions[claim_id])
        evaluated += 1
        if predicted_unsupported and actual_unsupported:
            counts["tp"] += 1
        elif predicted_unsupported and not actual_unsupported:
            counts["fp"] += 1
        elif not predicted_unsupported and not actual_unsupported:
            counts["tn"] += 1
        else:
            counts["fn"] += 1

    tp, fp, tn, fn = (counts[k] for k in ("tp", "fp", "tn", "fn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / evaluated if evaluated else 0.0
    return {
        "task": "generated_answer_claim_factuality",
        "label_source": "independent_claim_annotation",
        "annotation_rule": (
            "supported = supported by cited/reference evidence; contradicted = conflicts "
            "with evidence; insufficient_evidence = not verifiable from available evidence. "
            "Contradicted and insufficient_evidence count as unsupported for binary metrics."
        ),
        "n_annotated": len(rows),
        "n_evaluated": evaluated,
        "missing_prediction_claim_ids": missing_predictions,
        "coverage": evaluated / len(rows),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
    }
