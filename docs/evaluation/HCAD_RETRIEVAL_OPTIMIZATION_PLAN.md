# HCAD Retrieval Optimization Plan

## Objective

Improve HCAD-RAG on independently measured retrieval and answer-quality outcomes versus both Baseline LLM and Vanilla RAG. Scores are never edited or synthetically increased; all improvements must come from system changes and fresh, reproducible runs.

## P0 — Evaluation integrity (implemented in this branch)

- Make GO require BOTH the documented minimum paired improvements: Recall@5 >= +0.08 and NDCG@5 >= +0.06 versus the unmodified retrieval baseline.
- Environment variables may make the gate stricter, but cannot weaken the documented minimum.
- A non-negative NDCG delta alone is not sufficient to pass.
- Keep the held-out test pool locked until the development gate passes.
- Re-run the dev pool on the exact branch/commit being evaluated and preserve the JSON artifact and configuration.

## P1 — Diagnose retrieval loss

Run paired comparisons on the same frozen development questions and report:
- gold evidence present in the initial candidate pool;
- gold evidence surviving each ranking/filtering stage;
- Recall@5, NDCG@5, MRR@5 and candidate-pool coverage;
- results by question type (single-hop, multi-hop, negation, relation/modifier-heavy, unsupported);
- graph-only, dense-only, and hybrid candidate-source contributions.

If gold evidence is absent from the candidate pool, improve candidate generation/entity linking/graph expansion. If it is present but falls out of top-k, improve ranking. Do not tune the reranker to compensate for candidate-generation failures.

## P2 — Controlled ranking experiments

On development data only, compare one change at a time:
1. Dense vector top-k baseline.
2. Current HCAD graph/hypergraph ranking.
3. Reciprocal Rank Fusion (RRF) over dense and graph candidate ranks.
4. RRF plus a calibrated cross-encoder/biomedical reranker, if available within the runtime budget.
5. Optional graph boost only when entity/relation match is explicit and the boost improves paired development metrics.

Use ranks or calibrated probabilities when combining sources; raw cosine similarity, graph centrality, and heuristic Phi scores are not naturally comparable. Deduplicate candidates by stable evidence ID and retain provenance. Keep masking and relation-vocabulary features off unless an isolated ablation demonstrates a benefit.

## P3 — Answer quality and grounding

With the same questions, models, prompts, token budget, and gate mode, measure:
- claim-level factual correctness against independent labels;
- citation entailment/precision and evidence coverage (not merely whether citation IDs exist);
- contradiction and negation errors;
- correct abstention for unsupported questions;
- comprehensiveness, diversity, latency, failure rate, and cost.

A citation that points to an existing item is not necessarily a citation that supports the claim. Keep ID validity and semantic entailment as separate metrics.

## P4 — Confirmation and reporting

- Tune only on development data; run held-out evaluation only after GO.
- Use at least 200 paired questions when feasible, with question-level bootstrap 95% confidence intervals and paired significance tests.
- Record commit SHA, dataset/index hashes and document counts, question IDs/split/seed, model and prompt versions, gate/tensor modes, dependency versions, failures, latency, and cost.
- Report all three systems (Baseline LLM, Vanilla RAG, HCAD-RAG), confidence intervals, and negative as well as positive results.
- Mark HCAD as improved only when measured results support the claim. If the gate returns STOP, investigate and repeat development experiments; do not run or selectively report the held-out test set.

## Current evidence / decision

Previous logged development runs showed masking and relation-vocabulary configurations below the baseline. Those results motivate the ablations but are not a substitute for a fresh run on the current commit and a validated corpus/index. The quarantined incomplete-vector-store benchmark artifacts must not be used as valid performance evidence.


## Implementation added on the optimization branch

### Paired retrieval ranking experiment

Run on the development split (default; no held-out data is read):

```bash
python -m src.evaluation.ranking_experiments --samples 60 --pool dev
```

The JSON output records dense-only, graph-only, RRF, and production HCAD top-5 Recall/NDCG/MRR, question-level bootstrap 95% intervals, paired deltas versus dense-only, source counts, candidate-pool gold coverage, gold survival above the Phi floor, final HCAD top-5 coverage, per-question evidence and latency. The RRF implementation uses source-local ranks and does not compare raw cosine and graph scores.

Held-out execution is fail-closed and requires `HCAD_DEV_GO_ARTIFACT` pointing to a development JSON with GO and the same recorded commit SHA. The held-out split should not be run until the exact commit's development experiment passes the documented gate.

### Clinical answer annotation scoring

`src/evaluation/clinical_answer_quality.py` aggregates independent annotations for factual correctness, citation support, negation handling and abstention decisions. It accepts optional latency, token and cost telemetry. It deliberately does not infer clinical labels from generated text; cost is reported only when supplied by telemetry.

Expected JSONL fields per question/system: `question_id`, `system`, `answer`, `factuality_label` (`correct|partially_correct|incorrect`), `citation_support_label` (`supported|partially_supported|unsupported|not_applicable`), `negation_label` (`correct|incorrect|not_applicable`), `abstention_expected` and `abstained` booleans. Optional fields are `latency_ms`, `input_tokens`, `output_tokens`, and `cost_usd`.

```bash
python -m src.evaluation.clinical_answer_quality --annotations data/evaluation/clinical_answer_annotations.jsonl --output results/clinical_answer_quality.json
```

### Current limitations

- Retrieval labels use the existing source-chunk Jaccard overlap protocol (threshold 0.30); report this limitation and validate with independently adjudicated evidence labels before making clinical relevance claims.
- The ranking script is an offline comparison, not a production strategy switch. A candidate ranking is adopted only after the paired dev results and confidence intervals support it.
- The clinical quality scorer requires human/curator annotations and provider telemetry. It does not manufacture labels, prices, or a pass result.
- CI and an actual runtime evaluation must pass before this branch can be called validated. No performance gain is claimed by adding these evaluation tools.
