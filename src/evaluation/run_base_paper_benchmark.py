"""
Benchmark Runner replicating Base Paper  evaluation protocol.

Compares:
1. Baseline LLM (No Retrieval)
2. Vanilla Vector-RAG (Standard Dense Top-K)
3. HCAD-RAG (Full Proposed Architecture)

Outputs:
- 5-Dimension Scoring-Based Assessment (0-100)
- 8-Criteria Selection-Based Win-Rate (%)
- Retrieval Latency Benchmark (seconds)
"""

import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EVIDENCE_WORD_BUDGET, RANDOM_SEED, RESULTS_DIR, RAW_DATA_DIR
from src.evaluation.base_paper_evaluator import BasePaperEvaluator
from src.wandb_tracking import log_base_paper_results, tracked_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_clinical_benchmark_samples(num_samples: int = 5) -> List[Dict[str, str]]:
    """Load gold-standard clinical benchmark cases matching the base paper  protocol.

    Attempts to load from dataset files first, falls back to hardcoded samples if not available.
    """
    import json
    from pathlib import Path

    if num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")

    # Try to load from processed dataset files
    try:
        # Look for MedQA or other clinical datasets
        medqa_path = Path(__file__).parent.parent.parent / "data" / "processed" / "medqa_samples.json"
        if medqa_path.exists():
            with open(medqa_path, 'r', encoding='utf-8') as f:
                samples = json.load(f)
            if isinstance(samples, list) and len(samples) >= num_samples:
                for sample in samples:
                    if isinstance(sample, dict):
                        sample.setdefault("source", "medqa_processed")
                logger.info(f"Loaded {len(samples)} clinical samples from {medqa_path}")
                return samples[:num_samples]
    except Exception as e:
        logger.debug(f"Could not load clinical samples from file: {e}")

    # Fallback to hardcoded gold-standard clinical cases
    clinical_cases = [
        {
            "question": "How should long-term treatment and motor symptom recovery of patients with Parkinson's disease and tremor be managed?",
            "reference": "Parkinson's disease (PD) motor management is rooted in restoring striatal dopaminergic signaling. First-line pharmacotherapy uses carbidopa/levodopa as the most effective symptomatic agent, supplemented by dopamine receptor agonists (pramipexole, ropinirole) and MAO-B inhibitors (rasagiline) or COMT inhibitors (entacapone) to smooth out motor fluctuations and wearing-off phenomena. For medication-refractory resting tremors and disabling motor fluctuations, surgical intervention with Deep Brain Stimulation (DBS) targeting the subthalamic nucleus (STN) or globus pallidus internus (GPi) yields significant motor improvement and reduces medication requirements. Comprehensive care also incorporates physical therapy, fall risk mitigation, and neurorehabilitation.",
            "ground_truth": "Levodopa pharmacotherapy combined with deep brain stimulation (DBS) for refractory motor tremors."
        },
        {
            "question": "What are the emergency management and risk stratification guidelines for acute ST-elevation myocardial infarction (STEMI)?",
            "reference": "Acute ST-elevation myocardial infarction (STEMI) represents complete transmural myocardial ischemia typically caused by acute coronary atherothrombotic occlusion. Emergency management mandates immediate reperfusion therapy: primary percutaneous coronary intervention (PCI) is preferred if door-to-balloon time is under 90 minutes (or under 120 minutes if transferred); intravenous fibrinolysis is indicated within 30 minutes if PCI is unavailable within 120 minutes. Concomitant pharmacotherapy includes immediate dual antiplatelet therapy (aspirin plus a potent P2Y12 inhibitor such as ticagrelor or prasugrel), unfractionated heparin or bivalirudin anticoagulation, sublingual nitroglycerin for ischemic pain, and high-intensity statin therapy. Risk stratification incorporates TIMI or GRACE risk scores and assessment for cardiogenic shock or mechanical complications.",
            "ground_truth": "Immediate primary PCI within 90 minutes or fibrinolysis within 30 minutes, combined with dual antiplatelet therapy, anticoagulation, and statins."
        },
        {
            "question": "What is the stepwise diagnostic and therapeutic algorithm for acute severe exacerbation of bronchial asthma?",
            "reference": "Acute severe asthma exacerbation is marked by acute bronchospasm, airway inflammation, and ventilation-perfusion mismatch. Emergency therapeutic protocol begins with high-concentration supplemental oxygen (titrated to SpO2 93-95%), repetitive or continuous nebulized short-acting beta-2 agonists (albuterol/salbutamol) combined with inhaled anticholinergics (ipratropium bromide), and early administration of systemic corticosteroids (oral prednisone or IV methylprednisolone) to reduce airway edema. For refractory severe bronchospasm, intravenous magnesium sulfate is administered. Clinical monitoring includes serial peak expiratory flow (PEF) measurements, arterial blood gas analysis to detect hypercapnia and impending respiratory muscle fatigue, and preparation for non-invasive positive pressure ventilation or endotracheal intubation if respiratory arrest is imminent.",
            "ground_truth": "Continuous inhaled beta-agonists plus ipratropium, early systemic corticosteroids, supplemental oxygen, and IV magnesium sulfate for refractory bronchospasm."
        },
        {
            "question": "How should cognitive and behavioral symptoms in moderate-to-severe Alzheimer's dementia be evaluated and managed?",
            "reference": "Alzheimer's disease management encompasses cognitive stabilization, neuropsychiatric symptom control, and caregiver support. Pharmacological therapy for moderate-to-severe dementia combines a cholinesterase inhibitor (donepezil, rivastigmine, or galantamine) with the NMDA receptor antagonist memantine, which provides neuroprotection against excitotoxicity. Behavioral and psychological symptoms of dementia (BPSD) including agitation, psychosis, and sleep disruption require initial non-pharmacological behavioral modifications and environmental optimization. Low-dose atypical antipsychotics are reserved strictly for severe distressing agitation or psychosis unresponsive to non-pharmacological measures, with close cardiac and metabolic monitoring due to black box mortality warnings.",
            "ground_truth": "Combination of cholinesterase inhibitors and memantine with non-pharmacological behavioral management for neuropsychiatric symptoms."
        },
        {
            "question": "What are the diagnostic criteria and management strategies for type 2 diabetes mellitus?",
            "reference": "Type 2 diabetes mellitus is characterized by insulin resistance and relative insulin deficiency. Diagnostic criteria include HbA1c ≥6.5%, fasting plasma glucose ≥126 mg/dL, 2-hour plasma glucose ≥200 mg/dL during OGTT, or random plasma glucose ≥200 mg/dL with classic symptoms. Management focuses on lifestyle modification with medical nutrition therapy, physical activity, and pharmacotherapy. First-line therapy is metformin, with addition of other agents (SGLT2 inhibitors, GLP-1 receptor agonists, DPP-4 inhibitors, sulfonylureas, or insulin) based on HbA1c targets, patient comorbidities, and cost considerations. Comprehensive management includes cardiovascular risk reduction, blood pressure control, lipid management, and regular monitoring for complications.",
            "ground_truth": "Metformin as first-line therapy with comprehensive cardiovascular risk reduction and regular complication monitoring."
        }
    ]
    for sample in clinical_cases:
        sample.setdefault("source", "curated")

    # the MedQA fallback was removed - the base paper never
    # uses MCQ benchmarks, and mixing question formats made aggregate scores
    # heterogeneous. Oversize requests are truncated to the curated pool.
    return clinical_cases[:num_samples]

    return clinical_cases

def _merge_selection_results(ab_result: Dict, ba_result: Dict, target_label: str) -> Dict:
    """Merge AB and BA pairwise evaluations to counterbalance position bias."""
    merged_criteria = {}
    ab_criteria = ab_result.get("criteria_winners", {})
    ba_criteria = ba_result.get("criteria_winners", {})

    for crit in BasePaperEvaluator.CRITERIA_LIST:
        ab_winner = ab_criteria.get(crit, "Tie")
        ba_winner = ba_criteria.get(crit, "Tie")
        # If both orders agree on the same winner, keep it. Otherwise, Tie.
        if ab_winner == ba_winner:
            merged_criteria[crit] = ab_winner
        else:
            merged_criteria[crit] = "Tie"

    ab_overall = ab_result.get("overall_winner", "Tie")
    ba_overall = ba_result.get("overall_winner", "Tie")
    if ab_overall == ba_overall:
        merged_overall = ab_overall
    else:
        merged_overall = "Tie"

    return {
        "criteria_winners": merged_criteria,
        "overall_winner": merged_overall,
        "summary": f"AB: {ab_result.get('summary', '')} | BA: {ba_result.get('summary', '')}",
    }

def _validate_generated_response(system_name: str, response: str) -> str:
    """Stop the benchmark instead of scoring an error message as an answer."""
    if not isinstance(response, str) or not response.strip():
        raise RuntimeError(f"{system_name} returned an empty or non-text response")
    cleaned = response.strip()
    if cleaned.startswith("[Error:"):
        raise RuntimeError(f"{system_name} generation failed: {cleaned}")
    return cleaned

def _format_retrieved_evidence(contexts: List[Dict[str, Any]], tag: str) -> str:
    """Expose answer-specific retrieval evidence to the judge with a word cap."""
    if not contexts:
        return "No retrieved evidence was supplied for this answer."

    lines = []
    remaining_words = EVIDENCE_WORD_BUDGET
    for index, context in enumerate(contexts, 1):
        metadata = context.get("metadata") or {}
        document = context.get("document") or context.get("text") or ""
        if not document:
            head = metadata.get("head_name", "")
            relation = metadata.get("display_relation") or metadata.get("relation", "")
            tail = metadata.get("tail_name", "")
            document = f"{head} {relation} {tail}".strip()
        words = str(document).split()
        if not words or remaining_words <= 0:
            break
        # clip at the last sentence boundary within the
        # budget - mid-sentence clipping can sever negations/qualifiers.
        budget_text = " ".join(words[:remaining_words])
        cut = max(budget_text.rfind(". "), budget_text.rfind("! "), budget_text.rfind("? "))
        selected = budget_text if cut == -1 else budget_text[: cut + 1]
        lines.append(f"[{tag} {index}] {selected.strip()}")
        used = len(selected.split())
        remaining_words -= used
        if used < len(words):
            break
    return "\n".join(lines) or "Retrieved contexts contained no readable evidence."

QUESTION_GEN_PROMPT = (
    "You are given a passage from a clinical knowledge base. Generate one "
    "examination-style, open-ended question that this passage fully answers.\n"
    "{tier}\n"
    "Return ONLY the question text.\n\nPassage:\n{chunk}"
)
TIERS = {
    1: "Ask a single focused question.",
    2: "Ask a two-stage question: first one aspect, then 'and specifically' a second, dependent aspect.",
    3: "Ask a three-stage nested question chaining three dependent aspects with connecting phrases.",
}

def generate_paper_protocol_questions(num_samples: int, generator=None) -> list[dict]:
    """Base-paper protocol (Sec. 3.5): questions are GENERATED from source
    chunks by an LLM, and the SOURCE CHUNK is the scoring reference.

    Difficulty tiers follow Sec. 1.2: 1/2/3-stage nested questions. This
    replaces hand-curated cases with the exact protocol the paper uses.
    """
    from src.core.llm_client import UnifiedLLMClient
    from src.data.build_vector_store import load_vector_store
    import random as _random

    if generator is None:
        generator = UnifiedLLMClient(temperature=0.3)
    collection = load_vector_store()
    total = collection.count()
    if total == 0:
        raise RuntimeError("Vector store is empty - run `python src/main.py prepare` first.")
    got = collection.get(limit=min(max(num_samples * 3, 30), total),
                         include=["documents"])
    docs = got.get("documents") or []
    if not docs:
        raise RuntimeError("No documents available for question generation.")
    _random.Random(RANDOM_SEED).shuffle(docs)

    samples, tier_sizes = [], {1: 0, 2: 0, 3: 0}
    per_tier = max(1, num_samples // 3)
    for tier in (1, 2, 3):
        for chunk in docs:
            if tier_sizes[tier] >= per_tier or len(samples) >= num_samples:
                break
            try:
                q = generator.chat([{"role": "user", "content": QUESTION_GEN_PROMPT.format(
                    tier=TIERS[tier], chunk=chunk[:2500])}]).strip()
            except Exception:
                continue
            if len(q) < 25:
                continue
            samples.append({"question": q, "reference": chunk,
                            "ground_truth": "", "source": "paper_protocol",
                            "difficulty_tier": tier})
            tier_sizes[tier] += 1
    logger.info("Paper-protocol questions: %s", tier_sizes)
    return samples

@tracked_run(
    name="base-paper-benchmark",
    job_type="benchmark",
    config=lambda num_samples=10, output_path=None: {"samples": num_samples},
    log_results=log_base_paper_results,
)
def run_benchmark(num_samples: int = 10, output_path: Optional[Path] = None):
    """Execute complete base paper evaluation benchmark across the 3 systems."""
    logger.info("Initializing Base Paper Evaluator...")
    evaluator = BasePaperEvaluator()

    # Import system pipelines
    from src.baselines.baseline_llm import BaselineLLM
    from src.baselines.vanilla_vector_rag import VanillaVectorRAG
    from src.core.hcad_retriever import HCADRetriever

    logger.info("Loading baseline and HCAD models...")
    baseline_llm = BaselineLLM()
    vanilla_rag = VanillaVectorRAG()
    hcad_rag = HCADRetriever()
    logger.info("Pre-warming retriever components (graph, vector store, NER models)...")
    hcad_rag._ensure_components()
    _ = hcad_rag.graph
    _ = hcad_rag.hypergraph

    # Paper protocol first (Sec. 3.5): LLM-generated questions with source
    # chunks as references; fall back to the legacy curated loader only if
    # generation is unavailable (no LLM reachable).
    try:
        samples = generate_paper_protocol_questions(num_samples)
        logger.info("Using paper-protocol questions (%d).", len(samples))
    except Exception as exc:
        logger.warning("Paper-protocol question generation failed (%s); "
                       "falling back to curated loader.", exc)
        samples = load_clinical_benchmark_samples(num_samples)
    logger.info(f"Loaded {len(samples)} clinical test cases.")

    scoring_by_tier: dict[int, dict] = {}
    scoring_records = {
        "Baseline LLM": [],
        "Vanilla RAG": [],
        "HCAD-RAG": [],
    }
    scoring_records_by_source = {}

    selection_records = {
        "HCAD_vs_Vanilla": [],
        "HCAD_vs_Baseline": [],
    }
    selection_records_by_source = {}

    latency_records = {
        "Vanilla RAG": [],
        "HCAD-RAG": [],
    }

    checkpoint_path = RESULTS_DIR / "base_paper_checkpoint.jsonl"
    completed_questions = set()
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            completed_questions = {
                json.loads(line)["question"] for line in f if line.strip()
            }
        logger.info("Resuming: %d samples already checkpointed.", len(completed_questions))

    for idx, sample in enumerate(samples):
        q = sample["question"]
        ref = sample["reference"]
        if q in completed_questions:
            logger.info("Skipping checkpointed sample %d/%d.", idx + 1, len(samples))
            continue
        logger.info(f"Evaluating Sample {idx+1}/{len(samples)}: {q[:70]}...")

        # 1. Generate responses & measure retrieval latencies
        # Baseline LLM
        resp_baseline = _validate_generated_response("Baseline LLM", baseline_llm.generate(q))
        logger.info("  Baseline LLM response (%d chars): %.120s...", len(resp_baseline), resp_baseline)

        # Vanilla RAG
        t0 = time.perf_counter()
        vanilla_contexts = vanilla_rag.retrieve(q, top_k=5)
        t_vanilla_ret = time.perf_counter() - t0
        latency_records["Vanilla RAG"].append(t_vanilla_ret)
        resp_vanilla = _validate_generated_response(
            "Vanilla RAG", vanilla_rag.retrieve_and_generate(q, contexts=vanilla_contexts)
        )
        vanilla_evidence = _format_retrieved_evidence(vanilla_contexts, "Doc")  # neutral tags
        logger.info("  Vanilla RAG response (%d chars, %d contexts): %.120s...", len(resp_vanilla), len(vanilla_contexts), resp_vanilla)

        # HCAD-RAG
        t0 = time.perf_counter()
        hcad_contexts = hcad_rag.retrieve(q)
        t_hcad_ret = time.perf_counter() - t0
        latency_records["HCAD-RAG"].append(t_hcad_ret)
        logger.info("  HCAD-RAG retrieved %d contexts in %.1fs", len(hcad_contexts), t_hcad_ret)
        resp_hcad = _validate_generated_response(
            "HCAD-RAG", hcad_rag.retrieve_and_generate(q, contexts=hcad_contexts)
        )
        hcad_evidence = _format_retrieved_evidence(hcad_contexts, "Doc")  # neutral tags
        logger.info("  HCAD-RAG response (%d chars): %.120s...", len(resp_hcad), resp_hcad)

        # 2. Scoring-Based Assessment (0-100)
        score_base = evaluator.evaluate_scoring(q, resp_baseline, ref)
        score_van = evaluator.evaluate_scoring(
            q, resp_vanilla, ref, answer_evidence=vanilla_evidence
        )
        score_hcad = evaluator.evaluate_scoring(
            q, resp_hcad, ref, answer_evidence=hcad_evidence
        )
        logger.info("  Scores - Baseline: %.1f, Vanilla: %.1f, HCAD: %.1f",
                     score_base.overall_score, score_van.overall_score, score_hcad.overall_score)

        scoring_records["Baseline LLM"].append(score_base.to_dict())
        scoring_records["Vanilla RAG"].append(score_van.to_dict())
        scoring_records["HCAD-RAG"].append(score_hcad.to_dict())
        source = sample.get("source", "curated")
        tier = int(sample.get("difficulty_tier", 1) or 1)
        tier_records = scoring_by_tier.setdefault(tier, {s_: [] for s_ in scoring_records})
        source_records = scoring_records_by_source.setdefault(
            source, {system: [] for system in scoring_records}
        )
        source_records["Baseline LLM"].append(score_base.to_dict())
        source_records["Vanilla RAG"].append(score_van.to_dict())
        source_records["HCAD-RAG"].append(score_hcad.to_dict())
        for s_, rec in (("Baseline LLM", score_base), ("Vanilla RAG", score_van), ("HCAD-RAG", score_hcad)):
            tier_records[s_].append(rec.to_dict())

        # 3. Selection-Based Assessment (with order counterbalancing)
        # Run AB order
        sel_van_ab = evaluator.evaluate_selection(
            q, resp_hcad, resp_vanilla, ref,
            label_a="HCAD-RAG", label_b="Vanilla RAG",
            evidence_a=hcad_evidence, evidence_b=vanilla_evidence,
        )
        # Run BA order (swap answers)
        sel_van_ba = evaluator.evaluate_selection(
            q, resp_vanilla, resp_hcad, ref,
            label_a="Vanilla RAG", label_b="HCAD-RAG",
            evidence_a=vanilla_evidence, evidence_b=hcad_evidence,
        )
        # Merge: use AB result if both agree, otherwise Tie
        sel_van = _merge_selection_results(sel_van_ab, sel_van_ba, "HCAD-RAG")

        sel_base_ab = evaluator.evaluate_selection(
            q, resp_hcad, resp_baseline, ref,
            label_a="HCAD-RAG", label_b="Baseline LLM",
            evidence_a=hcad_evidence,
        )
        sel_base_ba = evaluator.evaluate_selection(
            q, resp_baseline, resp_hcad, ref,
            label_a="Baseline LLM", label_b="HCAD-RAG",
            evidence_b=hcad_evidence,
        )
        sel_base = _merge_selection_results(sel_base_ab, sel_base_ba, "HCAD-RAG")

        logger.info("  Selection - HCAD vs Vanilla: %s, HCAD vs Baseline: %s",
                     sel_van.get("overall_winner"), sel_base.get("overall_winner"))
        selection_records["HCAD_vs_Vanilla"].append(sel_van)
        selection_records["HCAD_vs_Baseline"].append(sel_base)
        # per-sample checkpoint with full provenance
        with open(checkpoint_path, "a", encoding="utf-8") as ckpt:
            ckpt.write(json.dumps({
                "question": q,
                "source": sample.get("source", "curated"),
                "difficulty_tier": sample.get("difficulty_tier", 1),
                "scores": {
                    sys_name: recs[-1]
                    for sys_name, recs in scoring_records.items()
                },
                "selection": {
                    "HCAD_vs_Vanilla": sel_van,
                    "HCAD_vs_Baseline": sel_base,
                },
                "latency": {
                    "Vanilla RAG": latency_records["Vanilla RAG"][-1],
                    "HCAD-RAG": latency_records["HCAD-RAG"][-1],
                },
            }) + "\n")
        source_selection = selection_records_by_source.setdefault(
            source, {comparison: [] for comparison in selection_records}
        )
        source_selection["HCAD_vs_Vanilla"].append(sel_van)
        source_selection["HCAD_vs_Baseline"].append(sel_base)

    # Compute aggregate statistics
    def calc_mean_scores(records: List[Dict[str, Any]]) -> Dict[str, float]:
        if not records:
            return {}
        dims = ["comprehensiveness", "diversity", "empowerment", "logical", "readability", "overall_score"]
        return {dim: sum(r[dim] for r in records) / len(records) for dim in dims}

    final_scores = {sys_name: calc_mean_scores(recs) for sys_name, recs in scoring_records.items()}
    scores_by_source = {
        source: {sys_name: calc_mean_scores(recs) for sys_name, recs in records.items()}
        for source, records in scoring_records_by_source.items()
    }

    # Compute Win-Rates
    def calc_win_rate(records: List[Dict[str, Any]], target_label: str = "HCAD-RAG") -> Dict[str, Any]:
        if not records:
            return {}
        total = len(records)
        overall_wins = sum(1 for r in records if r["overall_winner"] == target_label)
        ties = sum(1 for r in records if r.get("overall_winner") == "Tie")
        losses = total - overall_wins - ties
        criteria_wins = {c: 0 for c in BasePaperEvaluator.CRITERIA_LIST}
        for r in records:
            cw = r.get("criteria_winners", {})
            for c in BasePaperEvaluator.CRITERIA_LIST:
                if cw.get(c) == target_label:
                    criteria_wins[c] += 1

        return {
            "overall_win_rate_pct": (overall_wins / total) * 100.0,
            "tie_rate_pct": (ties / total) * 100.0,
            "decisive_win_rate_pct": (
                (overall_wins / (overall_wins + losses)) * 100.0
                if overall_wins + losses else 0.0
            ),
            "criteria_win_rates_pct": {c: (cnt / total) * 100.0 for c, cnt in criteria_wins.items()},
        }

    final_win_rates = {
        "HCAD_vs_Vanilla": calc_win_rate(selection_records["HCAD_vs_Vanilla"], "HCAD-RAG"),
        "HCAD_vs_Baseline": calc_win_rate(selection_records["HCAD_vs_Baseline"], "HCAD-RAG"),
    }
    win_rates_by_source = {
        source: {
            comparison: calc_win_rate(records, "HCAD-RAG")
            for comparison, records in comparisons.items()
        }
        for source, comparisons in selection_records_by_source.items()
    }

    avg_latencies = {
        sys_name: sum(lats) / len(lats) if lats else 0.0
        for sys_name, lats in latency_records.items()
    }

    summary_results = {
        "run_manifest": {  # provenance for every result file
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "num_samples": len(samples),
            "judge": {"provider": evaluator.judge.provider, "model": evaluator.judge.model},
            "prompt_mode": "zero_shot" if not __import__("src.config", fromlist=["ENABLE_FEW_SHOT"]).ENABLE_FEW_SHOT else "few_shot",
            "random_seed": RANDOM_SEED,
        },
        "scoring_based_assessment": final_scores,
        "scoring_by_difficulty_tier": {
            str(t): {s_: (sum(r["overall_score"] for r in recs) / len(recs) if recs else 0.0)
                     for s_, recs in sys_map.items()}
            for t, sys_map in scoring_by_tier.items()
        },
        "scoring_based_assessment_by_source": scores_by_source,
        "selection_based_assessment": final_win_rates,
        "selection_based_assessment_by_source": win_rates_by_source,
        "retrieval_latency_seconds": avg_latencies,
        "num_samples": len(samples),
        "sample_source_counts": {
            source: sum(1 for sample in samples if sample.get("source", "curated") == source)
            for source in sorted({sample.get("source", "curated") for sample in samples})
        },
        "judge": {
            "provider": evaluator.judge.provider,
            "model": evaluator.judge.model,
        },
    }

    out_file = output_path or (RESULTS_DIR / "base_paper_benchmark_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)

    logger.info(f"Benchmark results successfully saved to {out_file}")
    return summary_results

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run_benchmark(num_samples=n)
