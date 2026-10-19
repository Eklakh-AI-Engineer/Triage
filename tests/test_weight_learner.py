import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.config import (
    ANISO_ALPHA,
    ANISO_BETA,
    ANISO_GAMMA,
    SCORE_WEIGHT_ENTITY,
    SCORE_WEIGHT_MODIFIER,
    SCORE_WEIGHT_RELATION,
)
from src.core.weight_learner import HCADWeightLearner, train_weights

class TestWeightLearner(unittest.TestCase):
    def test_only_trains_filter_threshold_consumed_by_current_retriever(self):
        retriever = Mock()
        retriever.answer_mcq.side_effect = [("A", ""), ("A", "")]
        validation_data = [
            {"question": "q1", "options": {}, "answer_idx": "A"},
            {"question": "q2", "options": {}, "answer_idx": "B"},
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            weights_path = Path(temp_dir) / "weights.json"
            with patch("src.wandb_tracking.WANDB_MODE", "disabled"):
                result = train_weights(
                    retriever,
                    validation_data,
                    epochs=1,
                    save_path=weights_path,
                )
            saved_weights = json.loads(weights_path.read_text(encoding="utf-8"))

        self.assertEqual(retriever.answer_mcq.call_count, 2)
        self.assertTrue(result["filter_k"] > 0)
        self.assertEqual(
            [name for name, _ in HCADWeightLearner().named_parameters()],
            ["raw_k"],
        )
        self.assertEqual(result["aniso_alpha"], ANISO_ALPHA)
        self.assertEqual(result["aniso_beta"], ANISO_BETA)
        self.assertEqual(result["aniso_gamma"], ANISO_GAMMA)
        self.assertEqual(result["score_entity"], SCORE_WEIGHT_ENTITY)
        self.assertEqual(result["score_relation"], SCORE_WEIGHT_RELATION)
        self.assertEqual(result["score_modifier"], SCORE_WEIGHT_MODIFIER)
        self.assertEqual(saved_weights, result)

    def test_retrieval_failures_are_not_silently_scored_as_wrong_answers(self):
        retriever = Mock()
        retriever.answer_mcq.side_effect = RuntimeError("model unavailable")

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("src.wandb_tracking.WANDB_MODE", "disabled"):
                with self.assertRaisesRegex(RuntimeError, "validation item 0"):
                    train_weights(
                        retriever,
                        [{"question": "q", "options": {}, "answer_idx": "A"}],
                        epochs=1,
                        save_path=Path(temp_dir) / "weights.json",
                    )

if __name__ == "__main__":
    unittest.main()
