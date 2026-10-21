"""REINFORCE training for the Query-Adaptive Anisotropic Tensor (Blueprint 5.1).

Fairness contract:
- trains ONLY on the dev question pool (generate_paper_protocol_questions
  with pool="dev"), which is hash-disjoint from the evaluation (test) pool;
- the reward is retrieval quality (NDCG@5 against the question's gold source
  chunk) - no LLM judge in the training loop, so the judge cannot bias what
  is learned and the signal is cheap and explainable;
- the learned weights are saved to models/dynamic_tensor.pt and loaded only
  when HCAD_TENSOR_MODE=learned; the static heuristic remains the default.

Training loop: for each dev question the hypernetwork predicts (lambda, a);
Gaussian exploration noise is added (reparameterized), the retriever ranks
candidates under the sampled weights via the documented _tensor_override
hook, the NDCG@5 reward is measured, and a baseline-subtracted REINFORCE
step updates the hypernetwork.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from src.config import TENSOR_MODEL_PATH

logger = logging.getLogger(__name__)


def train_tensor(
    retriever,
    dev_questions: list[dict],
    epochs: int = 3,
    sigma: float = 0.10,
    lr: float = 1e-3,
    k: int = 5,
    save_path: Path | None = None,
) -> dict:
    """Train the query-adaptive anisotropic tensor on dev-pool questions.

    Args:
        retriever: a constructed HCADRetriever (components are loaded on
            first retrieve; the trainer activates the learned branch by
            assigning ``retriever._tensor`` directly).
        dev_questions: paper-protocol questions with pool="dev"
            (each carries "question" and "reference").
        epochs: passes over the dev set.
        sigma: exploration noise scale.
        lr: Adam learning rate.
        k: NDCG cut for the reward.
        save_path: checkpoint destination (default models/dynamic_tensor.pt).

    Returns:
        dict with mean reward per epoch and the checkpoint path.
    """
    from src.core.dynamic_tensor import (
        QueryAdaptiveAnisotropicTensor,
        save_tensor,
    )
    from src.evaluation.metrics import ndcg_at_k
    from src.evaluation.run_base_paper_benchmark import _ranked_relevance

    if not dev_questions:
        raise ValueError("dev_questions is empty - generate the dev pool first "
                         "(generate_paper_protocol_questions(pool='dev')).")
    pools = {q.get("pool", "unknown") for q in dev_questions}
    if pools - {"dev"}:
        raise ValueError(
            f"Refusing to train on non-dev question pools: {pools}. "
            "Training on test-pool questions would leak the evaluation set."
        )

    tensor = QueryAdaptiveAnisotropicTensor(embed_dim=768, feat_dim=25)
    retriever._tensor = tensor          # activate the learned ranking branch
    retriever._tensor_override = None
    optimizer = torch.optim.Adam(tensor.parameters(), lr=lr)

    baseline = 0.0
    history = []
    for epoch in range(epochs):
        rewards = []
        for item in dev_questions:
            question = item["question"]
            reference = item["reference"]

            # 1) one ranking pass without override: populates
            #    _last_tensor_inputs with (Q, feats) for this question.
            retriever._tensor_override = None
            try:
                retriever.retrieve(question)
            except Exception as exc:
                logger.warning("Warm-up retrieval failed for a dev question "
                               "(%s); skipping it.", exc)
                continue
            inputs = getattr(retriever, "_last_tensor_inputs", None)
            if inputs is None:
                continue
            Q_t, feats = inputs

            # 2) predict mean weights; sample exploration noise.
            optimizer.zero_grad()
            mu_lam, mu_a = tensor(Q_t, feats)
            eps_lam = torch.randn_like(mu_lam)
            lam_s = mu_lam + sigma * eps_lam
            lam_s = lam_s / lam_s.sum().clamp_min(1e-6)
            a_s = torch.softmax(
                torch.log(mu_a.clamp_min(1e-6)) + sigma * torch.randn_like(mu_a),
                dim=-1)

            # 3) rank under the sampled weights and measure the reward.
            retriever._tensor_override = (lam_s.detach(), a_s.detach())
            try:
                contexts = retriever.retrieve(question)
            except Exception as exc:
                logger.warning("Sampled retrieval failed (%s); reward 0.", exc)
                contexts = []
            finally:
                retriever._tensor_override = None
            relevance = _ranked_relevance(reference, contexts)
            reward = float(ndcg_at_k(relevance, k))
            rewards.append(reward)

            # 4) baseline-subtracted REINFORCE step.
            #    grad log N(s | mu, sigma) = (s - mu) / sigma^2
            gl = ((lam_s - mu_lam) / (sigma ** 2)).sum()
            ga = ((a_s - mu_a) / (sigma ** 2)).sum()
            loss = -(reward - baseline) * (gl + ga)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tensor.parameters(), max_norm=1.0)
            optimizer.step()
            baseline = 0.9 * baseline + 0.1 * reward

        mean_reward = sum(rewards) / max(len(rewards), 1)
        history.append({"epoch": epoch + 1, "mean_ndcg_reward": mean_reward,
                        "n": len(rewards)})
        logger.info("Tensor training epoch %d/%d: mean NDCG@%d reward %.4f "
                    "(%d questions)", epoch + 1, epochs, k, mean_reward,
                    len(rewards))

    retriever._tensor = None  # trainer never leaks the active tensor
    path = save_tensor(tensor, save_path)
    logger.info("Trained tensor saved to %s", path)
    return {"epochs": history, "checkpoint": str(path)}


def load_tensor_for_ablation(retriever) -> bool:
    """Activate the learned tensor for an ablation benchmark run.

    Used by the benchmark harness / notebook when HCAD_TENSOR_MODE=learned;
    returns True when the learned branch is active.
    """
    return retriever._get_tensor() is not None
