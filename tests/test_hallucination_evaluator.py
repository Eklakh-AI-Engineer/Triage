import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.evaluation.hallucination_evaluator import HallucinationEvaluator

class TestHallucinationEvaluator(unittest.TestCase):
    def test_scores_faithful_and_hallucinated_answers_as_paired_examples(self):
        evaluator = HallucinationEvaluator()
        evaluator._data = [
            {
                "Question": "What does the evidence show?",
                "Knowledge": ["Evidence one.", "Evidence two."],
                "Ground Truth": "A faithful answer.",
                "Hallucinated Answer": "An unsupported answer.",
                "Difficulty Level": "easy",
            }
        ]
        ollama = types.ModuleType("ollama")
        ollama.chat = MagicMock(
            side_effect=[
                SimpleNamespace(
                    message=SimpleNamespace(content="FAITHFUL")
                ),
                SimpleNamespace(
                    message=SimpleNamespace(content="HALLUCINATED")
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            sys.modules, {"ollama": ollama}
        ), patch(
            "src.evaluation.hallucination_evaluator.TABLES_DIR", Path(temp_dir)
        ), patch(
            "src.wandb_tracking.WANDB_MODE", "disabled"
        ):
            result = evaluator.evaluate(
                SimpleNamespace(model="test-model"),
                num_samples=None,
                system_name="test",
            )

        self.assertEqual(result["num_samples"], 1)
        self.assertEqual(result["num_classifications"], 2)
        self.assertEqual(
            result["overall"],
            {
                "precision": 1.0,
                "recall": 1.0,
                "specificity": 1.0,
                "f1": 1.0,
                "accuracy": 1.0,
                "tp": 1,
                "fp": 0,
                "tn": 1,
                "fn": 0,
            },
        )
        prompt = ollama.chat.call_args_list[0].kwargs["messages"][1]["content"]
        self.assertIn("Evidence one.\nEvidence two.", prompt)
        self.assertFalse(any(key in result for key in ("context_quality", "hallucination_probability")))

    def test_generated_answer_verdicts_do_not_claim_confusion_matrix_metrics(self):
        evaluator = HallucinationEvaluator()
        evaluator._data = [
            {
                "Question": "Question",
                "Knowledge": ["Reference evidence"],
                "Ground Truth": "Known supported answer",
                "Hallucinated Answer": "Known unsupported answer",
                "Difficulty Level": "easy",
            }
        ]
        system = SimpleNamespace(
            model="test-model",
            retrieve_and_generate=MagicMock(return_value="A generated answer."),
        )
        ollama = types.ModuleType("ollama")
        ollama.chat = MagicMock(return_value=SimpleNamespace(
            message=SimpleNamespace(content="FAITHFUL")
        ))
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            sys.modules, {"ollama": ollama}
        ), patch(
            "src.evaluation.hallucination_evaluator.TABLES_DIR", Path(temp_dir)
        ), patch(
            "src.wandb_tracking.WANDB_MODE", "disabled"
        ):
            result = evaluator.evaluate(system, num_samples=None, system_name="generated")

        self.assertEqual(result["evaluation_mode"], "system_answers")
        self.assertFalse(result["metrics_available"])
        self.assertNotIn("overall", result)
        self.assertEqual(result["verdict_counts"]["all"]["flagged_faithful"], 1)

    def test_invalid_model_output_is_not_counted_as_a_prediction(self):
        evaluator = HallucinationEvaluator()
        evaluator._data = [
            {
                "Question": "Question",
                "Knowledge": ["Knowledge"],
                "Ground Truth": "Faithful answer",
                "Hallucinated Answer": "False answer",
                "Difficulty Level": "easy",
            }
        ]
        ollama = types.ModuleType("ollama")
        ollama.chat = MagicMock(
            return_value=SimpleNamespace(
                message=SimpleNamespace(content="uncertain")
            )
        )

        with patch.dict(sys.modules, {"ollama": ollama}), patch(
            "src.wandb_tracking.WANDB_MODE", "disabled"
        ):
            with self.assertRaisesRegex(ValueError, "invalid hallucination label"):
                evaluator.evaluate(SimpleNamespace(model="test-model"), num_samples=None)

if __name__ == "__main__":
    unittest.main()
