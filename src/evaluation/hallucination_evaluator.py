"""
MedHallu Hallucination Detection Evaluator.

Evaluates hallucination detection on MedHallu benchmark (10K QA pairs
from PubMedQA with controlled hallucinations at easy/medium/hard tiers).

Metrics (Sections 4.34-4.35):
    - Precision, recall, specificity, F1, and accuracy
    - Metrics stratified by difficulty (easy / medium / hard)
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    DEFAULT_EVAL_SAMPLES,
    LLM_MODEL,
    MEDHALLU_CONFIG_LABELED,
    MEDHALLU_DATASET_ID,
    RANDOM_SEED,
    RAW_DATA_DIR,
    TABLES_DIR,
)
from src.wandb_tracking import log_hallucination_results, tracked_run

logger = logging.getLogger(__name__)

class HallucinationEvaluator:
    """
    Evaluates hallucination detection on the MedHallu benchmark.
    """

    DIFFICULTY_LEVELS = ["easy", "medium", "hard"]

    def __init__(self):
        self._data: Optional[list] = None

    # Data Loading

    def load_data(self, config: str = MEDHALLU_CONFIG_LABELED) -> list[dict]:
        """Load MedHallu dataset."""
        if self._data is not None:
            return self._data

        dataset_dir = RAW_DATA_DIR / "medhallu"
        legacy_path = dataset_dir / f"{config}.json"
        split_path = dataset_dir / config / "train.json"
        for save_path in (split_path, legacy_path):
            if save_path.exists():
                with open(save_path, encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError(f"Expected a list of MedHallu rows in {save_path}")
                self._data = data
                logger.info("Loaded %d MedHallu samples from %s", len(data), save_path)
                return self._data

        logger.info("Downloading MedHallu (%s) from HuggingFace...", config)
        try:
            from datasets import load_dataset

            ds = load_dataset(MEDHALLU_DATASET_ID, config)
            split = "train" if "train" in ds else next(iter(ds))
            self._data = [dict(row) for row in ds[split]]
            split_path.parent.mkdir(parents=True, exist_ok=True)
            with open(split_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
            logger.info("Saved %d MedHallu samples to %s", len(self._data), split_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load MedHallu ({config}): {exc}") from exc

        return self._data

    # Hallucination Detection

    @tracked_run(
        name="medhallu-evaluation",
        job_type="evaluation",
        config=lambda self, system, num_samples=DEFAULT_EVAL_SAMPLES, seed=RANDOM_SEED, system_name="system": {
            "system": system_name,
            "samples": num_samples,
            "seed": seed,
        },
        log_results=log_hallucination_results,
    )
    def evaluate(
        self,
        system,
        num_samples: Optional[int] = DEFAULT_EVAL_SAMPLES,
        seed: int = RANDOM_SEED,
        system_name: str = "system",
    ) -> dict:
        """
        Evaluate hallucination detection on MedHallu.

        The system must have a `answer_mcq` or `retrieve_and_generate` method.
        We use a separate hallucination-detection prompt.

        Args:
            system: System with `answer_mcq(question, options)` or
                    `retrieve_and_generate(query)` method.
            num_samples: Number of samples (None = all).
            seed: Random seed.
            system_name: Label for output.

        Returns:
            Dict with precision, recall, f1 (overall + per difficulty).
        """
        import random
        from src.core.prompt_builder import PromptBuilder

        data = self.load_data()
        if not data:
            logger.warning("MedHallu data empty. Skipping hallucination evaluation.")
            return {"error": "No data available"}

        if num_samples is not None and num_samples <= 0:
            raise ValueError("num_samples must be positive or None")
        if num_samples is not None and num_samples < len(data):
            data = random.Random(seed).sample(data, num_samples)

        builder = PromptBuilder()
        model_name = (
            getattr(system, "model_name", None)
            or getattr(getattr(system, "llm", None), "model", None)
            or getattr(system, "model", None)
            or LLM_MODEL
        )
        # the system under test was previously
        # NEVER invoked - the evaluator classified fixed gt/hallucinated
        # answers with a raw ollama.chat() call, so every system sharing a
        # base LLM received identical scores. Two modes now exist:
        #   - "system_answers": the system generates answers to each question
        #     (through its retrieval pipeline); each generated answer is then
        #     classified - this measures RAG-mediated hallucination.
        #   - "detector": legacy paired classification of gt/hallu answers -
        #     this measures only the classifier LLM, and is labeled as such.
        rag_capable = hasattr(system, "retrieve_and_generate") and callable(
            getattr(system, "retrieve_and_generate")
        )
        eval_mode = "system_answers" if rag_capable else "detector"
        import ollama

        results_by_difficulty: dict[str, dict] = {
            level: {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
            for level in self.DIFFICULTY_LEVELS + ["all"]
        }
        generated_answer_predictions: dict[str, list[bool]] = {
            level: [] for level in self.DIFFICULTY_LEVELS + ["all"]
        }

        def classify(question: str, answer: str, knowledge: str, sample_index: int) -> bool:
            messages = builder.build_hallucination_prompt(question, answer, knowledge)
            try:
                response = ollama.chat(
                    model=model_name,
                    messages=messages,
                    options={"temperature": 0.0, "num_predict": 8},
                )
                prediction = response.message.content.strip().upper().rstrip(".!")
            except Exception as exc:
                raise RuntimeError(
                    f"MedHallu classifier failed for sample {sample_index}: {exc}"
                ) from exc

            if prediction == "HALLUCINATED":
                return True
            if prediction == "FAITHFUL":
                return False
            raise ValueError(
                f"MedHallu classifier returned an invalid hallucination label "
                f"for sample {sample_index}: {prediction!r}"
            )

        for sample_index, item in enumerate(data):
            question = item.get("Question", item.get("question", ""))
            knowledge_value = item.get("Knowledge", item.get("knowledge", ""))
            if isinstance(knowledge_value, (list, tuple)):
                knowledge = "\n".join(str(part) for part in knowledge_value if part)
            else:
                knowledge = str(knowledge_value or "")
            gt_answer = item.get("Ground Truth", item.get("ground_truth", ""))
            hallu_answer = item.get(
                "Hallucinated Answer", item.get("hallucinated_answer", "")
            )
            difficulty = str(
                item.get("Difficulty Level", item.get("difficulty_level", "medium"))
                or "medium"
            ).lower().strip()
            if not question or not knowledge or not gt_answer or not hallu_answer:
                raise ValueError(
                    f"MedHallu sample {sample_index} is missing question, knowledge, "
                    "ground-truth answer, or hallucinated answer"
                )
            if difficulty not in self.DIFFICULTY_LEVELS:
                raise ValueError(
                    f"MedHallu sample {sample_index} has unknown difficulty "
                    f"{difficulty!r}"
                )

            if eval_mode == "system_answers":
                # Measure the RAG pipeline: generate, then classify.
                try:
                    generated = system.retrieve_and_generate(question)
                except Exception as exc:
                    raise RuntimeError(
                        f"System generation failed for MedHallu sample {sample_index}: {exc}"
                    ) from exc
                if not isinstance(generated, str) or not generated.strip():
                    raise RuntimeError(
                        f"System returned an empty answer for MedHallu sample {sample_index}"
                    )
                predicted_hallu = classify(question, generated, knowledge, sample_index)
                # This is a model verdict, NOT an independently labelled target.
                # Store it as a descriptive count only; never call it TP/TN.
                for level_key in ("all", difficulty):
                    generated_answer_predictions[level_key].append(predicted_hallu)
            else:
                candidates = (
                    (str(gt_answer), False),
                    (str(hallu_answer), True),
                )
                for answer, actual_hallu in candidates:
                    predicted_hallu = classify(
                        question, answer, knowledge, sample_index
                    )
                    for level_key in ("all", difficulty):
                        counts = results_by_difficulty[level_key]
                        if actual_hallu and predicted_hallu:
                            counts["tp"] += 1
                        elif not actual_hallu and predicted_hallu:
                            counts["fp"] += 1
                        elif not actual_hallu and not predicted_hallu:
                            counts["tn"] += 1
                        else:
                            counts["fn"] += 1

        if eval_mode == "system_answers":
            # No independent claim-level gold labels exist in this path. Returning
            # confusion-matrix metrics here would invent ground truth by treating
            # the judge's verdict as the answer's actual factuality.
            output = {
                "system": system_name,
                "evaluation_mode": "system_answers",
                "task": "generated_answer_classifier_verdicts",
                "num_samples": len(data),
                "num_classifications": len(data),
                "classifier_model": model_name,
                "metrics_available": False,
                "reason": (
                    "Precision/recall/F1 are not reported because generated answers "
                    "lack independent claim-level factuality labels. Use "
                    "src.evaluation.claim_factuality with independently annotated claims."
                ),
                "verdict_counts": {
                    level: {
                        "flagged_hallucinated": sum(predictions),
                        "flagged_faithful": len(predictions) - sum(predictions),
                        "n": len(predictions),
                    }
                    for level, predictions in generated_answer_predictions.items()
                },
            }
            out_path = TABLES_DIR / f"hallucination_{system_name.replace(' ','_').lower()}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2)
            return output

        # Detector-mode confusion matrix uses the dataset's paired known labels.
        def _f1_from_counts(counts: dict) -> dict:
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = (
                counts["tn"] / (counts["tn"] + fp)
                if (counts["tn"] + fp) > 0
                else 0.0
            )
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            total = tp + fp + counts["tn"] + fn
            accuracy = (tp + counts["tn"]) / total if total else 0.0
            return {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "specificity": round(specificity, 4),
                "f1": round(f1, 4),
                "accuracy": round(accuracy, 4),
                **counts,
            }

        n_classifications = len(data) if eval_mode == "system_answers" else 2 * len(data)
        output = {
            "system": system_name,
            "evaluation_mode": eval_mode,  # mode is now explicit
            "num_samples": len(data),
            "num_classifications": n_classifications,
            "classifier_model": model_name,
            "note": ("system_answers mode classifies SYSTEM-GENERATED answers "
                     "(RAG-mediated hallucination measurement).")
                    if eval_mode == "system_answers" else
                    ("detector mode measures only the classifier LLM on fixed "
                     "gt/hallucinated pairs - it does NOT evaluate the RAG pipeline."),
            "overall": _f1_from_counts(results_by_difficulty["all"]),
        }
        for level in self.DIFFICULTY_LEVELS:
            output[level] = _f1_from_counts(results_by_difficulty[level])

        logger.info(
            f"MedHallu '{system_name}': "
            f"Overall F1={output['overall']['f1']:.3f}, "
            f"Easy={output['easy']['f1']:.3f}, "
            f"Medium={output['medium']['f1']:.3f}, "
            f"Hard={output['hard']['f1']:.3f}"
        )

        out_path = TABLES_DIR / f"hallucination_{system_name.replace(' ','_').lower()}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        return output
