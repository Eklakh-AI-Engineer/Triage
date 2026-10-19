import math
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    ANISO_ALPHA,
    ANISO_BETA,
    ANISO_GAMMA,
    FILTERING_K,
    PROJECT_ROOT,
    SCORE_WEIGHT_ENTITY,
    SCORE_WEIGHT_MODIFIER,
    SCORE_WEIGHT_RELATION,
)
from src.wandb_tracking import (
    log_active_metrics,
    log_weight_learning_results,
    tracked_run,
    weight_learning_config,
)

class HCADWeightLearner(nn.Module):
    """Learn the retrieval threshold currently consumed by HCADRetriever."""

    def __init__(self):
        super().__init__()
        raw_k = math.log(math.expm1(FILTERING_K))
        self.raw_k = nn.Parameter(torch.tensor(raw_k))

    def get_k(self) -> float:
        return torch.nn.functional.softplus(self.raw_k).item()

    def get_all_weights(self) -> dict:
        return {
            "aniso_alpha": ANISO_ALPHA,
            "aniso_beta": ANISO_BETA,
            "aniso_gamma": ANISO_GAMMA,
            "score_entity": SCORE_WEIGHT_ENTITY,
            "score_relation": SCORE_WEIGHT_RELATION,
            "score_modifier": SCORE_WEIGHT_MODIFIER,
            "filter_k": self.get_k(),
        }

@tracked_run(
    name="hcad-weight-learning",
    job_type="training",
    config=weight_learning_config,
    log_results=log_weight_learning_results,
)
def train_weights(retriever, val_data: list[dict], epochs=10, lr=0.01, save_path=None) -> dict:
    """
    Tune the adaptive-filter threshold via REINFORCE policy gradient.

    The current retriever only accepts a per-query ``k`` override; the
    anisotropic and component weights remain fixed at their configured values.
    This routine therefore learns only ``filter_k`` and reports the other
    configured values unchanged.

    Args:
        retriever: HCADRetriever instance with answer_mcq method.
        val_data: List of MedQA question dicts with 'question', 'options', 'answer_idx' keys.
        epochs: Number of training epochs.
        lr: Learning rate for Adam optimizer.
        save_path: Path to save learned weights JSON.

    Returns:
        Dict of learned weights.
    """
    logger = logging.getLogger(__name__)
    if not val_data:
        raise ValueError("val_data must contain at least one validation question")
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be a positive finite number")

    learner = HCADWeightLearner()
    optimizer = torch.optim.Adam(learner.parameters(), lr=lr)
    # REINFORCE with a moving-average baseline to cut
    # gradient variance (raw +/-1 rewards have variance ~1 per sample).
    baseline = 0.0

    if save_path is None:
        save_path = PROJECT_ROOT / "models" / "learned_weights.json"

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_accuracy = -1.0
    best_weights = learner.get_all_weights()

    for epoch in range(epochs):
        # snapshot the weights BEFORE the epoch's updates,
        # so the saved filter_k is the one that produced the measured accuracy
        # (previously snapshotted after the final optimizer.step()).
        epoch_start_weights = learner.get_all_weights()
        correct = 0
        total = 0

        for item in val_data:
            optimizer.zero_grad()

            # Sample the one parameter the current retriever can actually use.
            k_dist = torch.distributions.Normal(learner.raw_k, 0.1)
            k_sample = k_dist.sample()
            log_prob = k_dist.log_prob(k_sample).sum()

            # Get the gold answer letter
            question = item.get("question", "")
            options = item.get("options", {})
            answer = item.get("answer_idx")
            if answer is None:
                answer = item.get("answer")
            gold_letter = _answer_letter(answer, options)
            if gold_letter is None:
                raise ValueError(
                    f"Validation item {total} has no parseable correct answer"
                )

            # Use current learned k for retrieval
            k_val = torch.nn.functional.softplus(k_sample).item()

            try:
                predicted_letter, _ = retriever.answer_mcq(
                    question, options, k=k_val
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Retrieval failed for validation item {total}: {exc}"
                ) from exc

            # Reward signal
            is_correct = str(predicted_letter).strip().upper() == gold_letter
            reward = 1.0 if is_correct else -1.0
            if is_correct:
                correct += 1
            total += 1

            # Policy gradient loss (baseline-subtracted; )
            baseline = 0.9 * baseline + 0.1 * reward
            loss = -(reward - baseline) * log_prob
            loss.backward()
            torch.nn.utils.clip_grad_norm_(learner.parameters(), max_norm=1.0)
            optimizer.step()

        accuracy = correct / max(total, 1)
        logger.info(
            f"Epoch {epoch+1}/{epochs}: accuracy={accuracy:.1%} "
            f"({correct}/{total}), weights={learner.get_all_weights()}"
        )
        log_active_metrics(
            {
                "training/accuracy": accuracy,
                "training/correct": correct,
                "training/total": total,
                **{
                    f"training/{name}": value
                    for name, value in learner.get_all_weights().items()
                },
            },
            step=epoch + 1,
        )

        # Track best weights (pre-epoch snapshot - see )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weights = epoch_start_weights

    # Save best weights
    with open(save_path, "w") as f:
        json.dump(best_weights, f, indent=4)
    logger.info(f"Saved best weights (acc={best_accuracy:.1%}) to {save_path}")

    return best_weights

def _answer_letter(answer, options: dict) -> str | None:
    """Normalize MedQA answer labels or answer text to A/B/C/D."""
    labels = ("A", "B", "C", "D")
    if isinstance(answer, int) and 0 <= answer < len(labels):
        return labels[answer]
    answer_text = str(answer or "").strip()
    if answer_text.upper() in labels:
        return answer_text.upper()
    normalized = answer_text.casefold()
    for label, option_text in options.items():
        if str(option_text).strip().casefold() == normalized:
            label = str(label).strip().upper()
            if label in labels:
                return label
    return None

def load_learned_weights(path=None) -> dict:
    if path is None:
        path = PROJECT_ROOT / "models" / "learned_weights.json"

    path = Path(path)
    if not path.exists():
        return None

    with open(path, 'r') as f:
        return json.load(f)
