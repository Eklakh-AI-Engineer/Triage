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
