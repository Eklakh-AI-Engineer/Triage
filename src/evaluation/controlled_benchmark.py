"""Paired benchmark for Baseline LLM, Vanilla RAG and TRIAGE."""
from __future__ import annotations
import argparse, hashlib, json, os, random, time
from pathlib import Path
from typing import Any
from src.config import RESULTS_DIR
from src.evaluation.integrity import validate_integrity_manifest

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _load_questions(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not r.get("question_id") or not r.get("question") for r in rows):
        raise ValueError("Question JSONL must contain question_id and question for every row")
    ids = [r["question_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("question_id values must be unique")
    return rows

def _bootstrap_ci(values: list[float], seed: int = 20261028, iterations: int = 2000) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "ci95": None}
    rng, n = random.Random(seed), len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(iterations))
    return {"n": n, "mean": sum(values) / n,
            "ci95": [means[int(.025 * (iterations - 1))], means[int(.975 * (iterations - 1))]],
            "method": "percentile bootstrap over question-level observations",
            "seed": seed, "iterations": iterations}

def run_controlled_benchmark(questions_path: str | Path, output_dir: str | Path | None = None,
                             *, seed: int = 20261028) -> dict[str, Any]:
    questions_path = Path(questions_path)
    questions = _load_questions(questions_path)
    manifest_path = Path(os.getenv("TRIAGE_INTEGRITY_MANIFEST",
        str(Path(__file__).resolve().parents[2] / "data" / "processed" / "integrity_manifest.json")))
    integrity = validate_integrity_manifest(manifest_path)
    from src.baselines.baseline_llm import BaselineLLM
    from src.baselines.vanilla_vector_rag import VanillaVectorRAG
    from src.core.hcad_retriever import HCADRetriever
    baseline, vanilla = BaselineLLM(), VanillaVectorRAG()
    triage = HCADRetriever(fail_on_vector_error=True)
    triage._ensure_components()
    gate_mode = getattr(getattr(triage, "_decision_gate", None), "mode", "unknown")
    provenance = {
        "baseline_model": getattr(baseline, "model_name", os.getenv("BASELINE_MODEL", "configured-default")),
        "vanilla_model": getattr(vanilla, "model_name", os.getenv("RAG_MODEL", "configured-default")),
        "triage_model": getattr(triage, "model_name", os.getenv("HCAD_MODEL", "configured-default")),
        "judge_model": os.getenv("JUDGE_HF_MODEL", "configured-default"),
        "temperature": os.getenv("LLM_TEMPERATURE", "configured-default"),
        "context_budget_words": os.getenv("EVIDENCE_WORD_BUDGET", "config-default"),
        "gate_mode": gate_mode, "tensor_mode": os.getenv("HCAD_TENSOR_MODE", "static"),
        "question_sha256": _sha256(questions_path), "integrity_manifest_sha256": _sha256(manifest_path),
        "question_count": len(questions), "seed": seed,
    }
    per_question = []
    for item in questions:
        question = str(item["question"])
        record = {"question_id": item["question_id"], "question": question,
                  "tags": item.get("tags", []), "annotations": item.get("annotations", {}), "systems": {}}
        for name, system in (("baseline", baseline), ("vanilla_rag", vanilla), ("triage", triage)):
            started, retrieval_ms, contexts = time.perf_counter(), None, []
            try:
                if name == "baseline":
                    answer = system.generate(question)
                else:
                    retrieval_started = time.perf_counter()
                    contexts = system.retrieve(question, top_k=5) if name == "vanilla_rag" else system.retrieve(question)
                    retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
                    answer = system.retrieve_and_generate(question, contexts=contexts)
                record["systems"][name] = {"answer": str(answer), "status": "ok",
                    "retrieval_latency_ms": retrieval_ms,
                    "end_to_end_latency_ms": (time.perf_counter() - started) * 1000,
                    "context_count": len(contexts)}
            except Exception as exc:
                record["systems"][name] = {"answer": None, "status": "error",
                    "error_type": type(exc).__name__, "retrieval_latency_ms": retrieval_ms,
                    "end_to_end_latency_ms": (time.perf_counter() - started) * 1000,
                    "context_count": None}
        per_question.append(record)
    systems_summary = {}
    for name in ("baseline", "vanilla_rag", "triage"):
        rows = [r["systems"][name] for r in per_question]
        success = [r for r in rows if r["status"] == "ok"]
        retrieval = [float(r["retrieval_latency_ms"]) for r in success if r["retrieval_latency_ms"] is not None]
        e2e = [float(r["end_to_end_latency_ms"]) for r in success]
        systems_summary[name] = {"n_questions": len(rows), "n_success": len(success),
            "n_failures": len(rows) - len(success), "failure_rate": (len(rows)-len(success))/len(rows),
            "retrieval_latency_ms": _bootstrap_ci(retrieval, seed),
            "end_to_end_latency_ms": _bootstrap_ci(e2e, seed)}
    bundle = {"schema_version": 1, "experiment": "controlled_paired_three_system_benchmark",
        "provenance": provenance, "integrity_index": integrity["index"],
        "primary_metric_note": "This runner records paired outputs, latency and failure summaries. Clinical quality/factuality requires independent labels; an LLM judge verdict is not ground truth.",
        "systems": systems_summary, "per_question": per_question}
    out = Path(output_dir) if output_dir else RESULTS_DIR / "controlled_benchmark"
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_question.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in per_question), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return bundle

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=20261028)
    args = parser.parse_args()
    result = run_controlled_benchmark(args.questions, args.output_dir, seed=args.seed)
    print(json.dumps({"output_dir": args.output_dir or str(RESULTS_DIR / "controlled_benchmark"),
        "question_count": result["provenance"]["question_count"], "systems": result["systems"]}, indent=2))

if __name__ == "__main__":
    main()
