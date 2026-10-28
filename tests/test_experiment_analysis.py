import json
from pathlib import Path
import pytest

from src.evaluation.experiment_analysis import paired_bootstrap_delta


def test_paired_bootstrap_delta_is_deterministic_and_paired():
    a = paired_bootstrap_delta([1, 2, 3], [2, 3, 4], seed=7, iterations=200)
    b = paired_bootstrap_delta([1, 2, 3], [2, 3, 4], seed=7, iterations=200)
    assert a == b
    assert a["mean_delta"] == pytest.approx(1.0)
    assert a["ci95"] == pytest.approx([1.0, 1.0])


def test_paired_bootstrap_rejects_misaligned_questions():
    with pytest.raises(ValueError, match="equal length"):
        paired_bootstrap_delta([1, 2], [1])


def test_independent_clinical_seed_set_covers_required_categories():
    path = Path("data/evaluation/clinical_test_set.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    tags = {tag for row in rows for tag in row["tags"]}
    assert len(rows) >= 10
    assert {"negation", "adverse-effects", "contraindication", "temporal-modifier",
            "multi-hop", "unsupported-query"} <= tags
    assert all(row["independent_from_indexed_passages"] for row in rows)
    assert all(row["annotation_status"] == "requires_independent_clinician_review" for row in rows)
    assert all(row["gold_labels"] is None for row in rows)
