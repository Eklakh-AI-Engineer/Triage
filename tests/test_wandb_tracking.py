import math
import sys
import types
from pathlib import Path
import unittest
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.wandb_tracking import flatten_numeric_metrics

class TestWandbTracking(unittest.TestCase):
    def test_flatten_numeric_metrics_preserves_nested_numeric_values(self):
        metrics = flatten_numeric_metrics(
            {
                "accuracy": 0.75,
                "summary": {"f1": 0.8, "count": 12, "label": "HCAD-RAG"},
                "enabled": True,
                "invalid": math.nan,
            },
            "evaluation",
        )

        self.assertEqual(
            metrics,
            {
                "evaluation/accuracy": 0.75,
                "evaluation/summary/f1": 0.8,
                "evaluation/summary/count": 12,
            },
        )

    def test_medqa_and_medhallu_logging_keeps_metrics_without_charts(self):
        from src.wandb_tracking import (
            log_hallucination_results,
            log_medqa_benchmark,
            log_medqa_results,
        )

        class Table:
            def __init__(self, columns, data):
                self.columns = columns
                self.data = data

        class Run:
            def __init__(self):
                self.logged = []

            def log(self, values):
                self.logged.append(values)

        wandb_stub = types.SimpleNamespace(Table=Table)
        run = Run()
        result = {
            "system": "hcad-rag",
            "total_questions": 2,
            "accuracy": 0.5,
            "accuracy_ci95": {"lower": 0.2, "upper": 0.8},
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
            "per_question": [
                {
                    "index": 0,
                    "correct": True,
                    "gt": "A",
                    "predicted": "A",
                    "latency_s": 0.2,
                }
            ],
        }
        benchmark = {
            "baseline": dict(result, system="baseline", per_question=[]),
            "vanilla_rag": dict(result, system="vanilla_rag", per_question=[]),
            "hcad_rag": result,
            "delta_errors_pct": 20.0,
            "hcad_vs_baseline_accuracy_delta_ci95": {
                "lower": -0.1,
                "upper": 0.3,
            },
        }

        with patch.dict(sys.modules, {"wandb": wandb_stub}):
            log_medqa_results(run, result)
            log_medqa_benchmark(run, benchmark)
            log_hallucination_results(
                run,
                {
                    "system": "hcad-rag",
                    "overall": {"precision": 0.8, "recall": 0.7, "f1": 0.75},
                    "easy": {"f1": 0.8},
                    "medium": {"f1": 0.7},
                    "hard": {"f1": 0.6},
                },
            )

        logged_values = {
            key: value
            for entry in run.logged
            for key, value in entry.items()
        }
        self.assertFalse(any(key.startswith("charts/") for key in logged_values))
        self.assertEqual(logged_values["medqa/hcad-rag/accuracy"], 0.5)
        self.assertEqual(logged_values["medqa/hcad-rag/accuracy_ci95/lower"], 0.2)
        self.assertEqual(logged_values["medqa/hcad-rag/accuracy_ci95/upper"], 0.8)
        self.assertEqual(logged_values["medqa/benchmark/delta_errors_pct"], 20.0)
        self.assertEqual(
            logged_values["medqa/benchmark/hcad_vs_baseline_accuracy_delta_ci95/lower"],
            -0.1,
        )
        self.assertEqual(logged_values["medhallu/hcad-rag/hard/f1"], 0.6)
        table = logged_values["medqa/hcad-rag/per_question_metrics"]
        self.assertEqual(
            table.columns,
            [
                "sample_index",
                "correct",
                "gold_answer",
                "predicted_answer",
                "latency_seconds",
                "running_accuracy",
            ],
        )
        self.assertEqual(len(table.data), 1)
