"""Shared Weights & Biases helpers for HCAD-RAG experiments."""

from __future__ import annotations

import logging
import math
import os
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Generator

from src.config import WANDB_ENTITY, WANDB_MODE, WANDB_PROJECT

logger = logging.getLogger(__name__)

def flatten_numeric_metrics(
    values: dict[str, Any], prefix: str = ""
) -> dict[str, int | float]:
    """Flatten nested numeric results into W&B-compatible scalar metrics."""
    metrics: dict[str, int | float] = {}
    for key, value in values.items():
        metric_name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            metrics.update(flatten_numeric_metrics(value, metric_name))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isfinite(value):
                metrics[metric_name] = value
    return metrics

@contextmanager
def wandb_run(
    name: str,
    job_type: str,
    config: dict[str, Any] | None = None,
) -> Generator[Any | None, None, None]:
    """Open a W&B run when credentials are configured; otherwise run locally."""
    if WANDB_MODE.strip().lower() == "disabled":
        logger.info("W&B logging disabled by WANDB_MODE.")
        yield None
        return

    # offline mode never needs an API key - previously
    # WANDB_MODE=offline without a key was silently skipped entirely.
    mode = WANDB_MODE.strip().lower()
    if mode == "offline":
        os.environ.setdefault("WANDB_MODE", "offline")
    elif not os.getenv("WANDB_API_KEY"):
        logger.info("W&B logging skipped: set WANDB_API_KEY in the local .env file.")
        yield None
        return

    import wandb

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=name,
        job_type=job_type,
        mode=WANDB_MODE,
        config=config or {},
    )
    if run is None:
        raise RuntimeError("W&B did not create a run despite being enabled.")

    try:
        yield run
    finally:
        run.finish()

def tracked_run(
    name: str,
    job_type: str,
    config: dict[str, Any] | Callable[..., dict[str, Any]] | None = None,
    log_results: Callable[[Any, Any], None] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Track a function as a W&B run and log its returned result."""

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            run_config = config(*args, **kwargs) if callable(config) else config
            with wandb_run(name, job_type, run_config) as run:
                result = function(*args, **kwargs)
                if run is not None and log_results is not None:
                    log_results(run, result)
                return result

        return wrapped

    return decorate

def log_active_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    """Log intermediate metrics to the current W&B run, if one is active."""
    if not os.getenv("WANDB_API_KEY") or WANDB_MODE.strip().lower() == "disabled":
        return

    import wandb

    if wandb.run is not None:
        wandb.log(metrics, step=step)

def upload_checkpoint_artifact(files: list[Path], name: str = "benchmark-checkpoint") -> bool:
    """Attach resumability files to the active run as a versioned artifact.

    Kaggle wipes /kaggle/working when an interactive session dies, so an
    off-box copy of the per-question checkpoint (and question cache) is the
    only recovery path for a killed session. Safe no-op when W&B is
    disabled/offline/unauthenticated or no run is active.
    """
    if WANDB_MODE.strip().lower() in ("disabled", "offline"):
        return False
    if not os.getenv("WANDB_API_KEY"):
        return False

    import wandb

    if wandb.run is None:
        return False

    existing = [str(p) for p in files if Path(p).exists()]
    if not existing:
        return False

    artifact = wandb.Artifact(name, type="checkpoint")
    for path in existing:
        artifact.add_file(path)
    wandb.run.log_artifact(artifact)
    logger.info("Uploaded checkpoint artifact '%s' with %d file(s).", name, len(existing))
    return True

def _log_bar_chart(
    run: Any,
    chart_name: str,
    rows: list[list[Any]],
    x_column: str,
    y_column: str,
    title: str,
) -> None:
    import wandb

    columns = [x_column, y_column]
    table = wandb.Table(columns=columns, data=rows)
    run.log({f"charts/{chart_name}": wandb.plot.bar(table, x_column, y_column, title=title)})

def _log_medqa_results(run: Any, results: dict[str, Any], prefix: str) -> None:
    import wandb

    scalars = {
        key: value
        for key, value in results.items()
        if key != "per_question" and isinstance(value, dict)
    }
    scalars.update(
        {
            key: value
            for key, value in results.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    run.log(flatten_numeric_metrics(scalars, prefix))

    per_question = results.get("per_question", [])
    if not per_question:
        return

    # O(N) running accuracy instead of O(N^2) reslicing.
    running_correct = 0
    rows = []
    for index, sample in enumerate(per_question):
        running_correct += int(bool(sample["correct"]))
        rows.append([
            sample["index"],
            sample["correct"],
            sample["gt"],
            sample["predicted"],
            sample["latency_s"],
            running_correct / (index + 1),
        ])
    columns = [
        "sample_index",
        "correct",
        "gold_answer",
        "predicted_answer",
        "latency_seconds",
        "running_accuracy",
    ]
    table = wandb.Table(columns=columns, data=rows)
    run.log({f"{prefix}/per_question_metrics": table})

def log_medqa_results(run: Any, results: dict[str, Any]) -> None:
    """Log MedQA aggregate metrics and a question-text-free metrics table."""
    _log_medqa_results(
        run,
        results,
        f"medqa/{results.get('system', 'system').replace(' ', '_').lower()}",
    )

def log_medqa_benchmark(run: Any, results: dict[str, Any]) -> None:
    """Log MedQA metrics for each system and benchmark-level scalar metrics."""
    systems = [
        (key, results[key])
        for key in ("baseline", "vanilla_rag", "hcad_rag")
        if isinstance(results.get(key), dict)
    ]
    for key, system_results in systems:
        _log_medqa_results(run, system_results, f"medqa/{key}")

    run.log(
        flatten_numeric_metrics(
            {
                key: value
                for key, value in results.items()
                if key not in {"baseline", "vanilla_rag", "hcad_rag"}
            },
            "medqa/benchmark",
        )
    )

def log_hallucination_results(run: Any, results: dict[str, Any]) -> None:
    """Log hallucination scalar metrics without creating charts."""
    prefix = f"medhallu/{results.get('system', 'system').replace(' ', '_').lower()}"
    run.log(flatten_numeric_metrics(results, prefix))

def log_base_paper_results(run: Any, results: dict[str, Any]) -> None:
    """Log all aggregate scoring, win-rate, source, and latency results."""
    import wandb

    run.log(flatten_numeric_metrics(results, "base_paper"))

    scores = results.get("scoring_based_assessment", {})
    dimensions = sorted(
        {dimension for system_scores in scores.values() for dimension in system_scores}
    )
    for dimension in dimensions:
        rows = [
            [system, system_scores[dimension]]
            for system, system_scores in scores.items()
            if dimension in system_scores
        ]
        _log_bar_chart(
            run,
            f"base_paper_score_{dimension}",
            rows,
            "system",
            dimension,
            f"Base-paper {dimension.replace('_', ' ')} by system",
        )

    for comparison, comparison_results in results.get(
        "selection_based_assessment", {}
    ).items():
        criteria = comparison_results.get("criteria_win_rates_pct", {})
        if criteria:
            table = wandb.Table(
                columns=["criterion", "win_rate_pct"],
                data=[[criterion, rate] for criterion, rate in criteria.items()],
            )
            run.log(
                {
                    f"charts/base_paper/{comparison}_criteria_win_rates": wandb.plot.bar(
                        table,
                        "criterion",
                        "win_rate_pct",
                        title=f"{comparison.replace('_', ' ')} criteria win rates",
                    )
                }
            )

    latencies = results.get("retrieval_latency_seconds", {})
    if latencies:
        _log_bar_chart(
            run,
            "base_paper_retrieval_latency",
            [[system, latency] for system, latency in latencies.items()],
            "system",
            "latency_seconds",
            "Average retrieval latency by system",
        )

def log_weight_learning_results(run: Any, weights: dict[str, Any]) -> None:
    """Log the learned model weights as final run metrics."""
    run.log({f"learned_weights/{key}": value for key, value in weights.items()})

def weight_learning_config(
    retriever: Any,
    val_data: list[dict[str, Any]],
    epochs: int = 10,
    lr: float = 0.01,
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    config = {
        "epochs": epochs,
        "learning_rate": lr,
        "validation_samples": len(val_data),
    }
    retriever_model = getattr(retriever, "model_name", None)
    if retriever_model is not None:
        config["retriever_model"] = str(retriever_model)
    if save_path is not None:
        config["weights_output_path"] = str(save_path)
    return config
